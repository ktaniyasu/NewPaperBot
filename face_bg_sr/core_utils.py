from __future__ import annotations
import torch
import math
import cv2
import os
import logging
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize
import torch.cuda.amp
import numpy as np
import torch.nn as nn
from collections import defaultdict
import contextlib
import time
import yaml
from abc import ABC, abstractmethod
from typing import Optional

# デバッグ用ログ
logging.basicConfig(filename='face_sr_debug.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# プロファイル専用ログ
profile_logger = logging.getLogger("face_sr_profile")
level_name = os.getenv("FACE_SR_LOGLEVEL", "INFO").upper()
profile_logger.setLevel(getattr(logging, level_name, logging.INFO))
if not profile_logger.hasHandlers():
    fh = logging.FileHandler("face_sr_profile.log")
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    profile_logger.addHandler(fh)

from norfair import Detection, Tracker
from facexlib.detection import init_detection_model

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_dp_tempcoh(ckpt_path, device, st_latent_size=None):
    import math
    from basicsr.utils.registry import ARCH_REGISTRY
    raw = torch.load(ckpt_path, map_location=device)
    params = raw['params_ema']
    latent_size = 256
    flat = params.pop('st_position_emb', None)
    if st_latent_size is not None and st_latent_size > 0:
        if flat is None:
            flat = torch.zeros(latent_size, 512, device=device)
        flat_sz = flat.size(0)
        target_tokens = st_latent_size * latent_size
        reps = math.ceil(target_tokens / flat_sz)
        params['st_position_emb'] = flat.repeat(reps, 1)[:target_tokens]
        st_tokens = target_tokens
    else:
        st_tokens = 0
    model = ARCH_REGISTRY.get('VideoCodeFormerStage2p5')(
        dim_embd=512, n_head=8, n_layers=9,
        codebook_size=1024, latent_size=latent_size, st_latent_size=st_tokens,
        connect_list=['32','64','128','256'], fix_modules=['quantize','generator'], vqgan_path=None
    ).to(device)
    model.load_state_dict(params, strict=(st_tokens>0))
    model.eval()
    return model

class FaceBgSR():
    def __init__(self, upscale, bg_upsampler, device=None, bg_device=None, 
                 detect_stride: int = 5, 
                 iou_threshold: float = 0.7,
                 max_faces_per_frame: int = 8,
                 temporal_window_size: int = 5,
                 config_path: str = 'config.yaml',
                 *,
                 use_compile: bool = True,
                 use_amp: bool = True,
                 use_channels_last: bool = True,
                 preload_dp_model: nn.Module | None = None,
                 config: dict | None = None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # 背景 SR 用 GPU（未指定なら顔側と同じ）
        self.bg_device = self.device if bg_device is None else bg_device
        self.detect_stride = detect_stride
        self.iou_threshold = iou_threshold
        self.max_faces_per_frame = max_faces_per_frame
        self.temporal_window_size = temporal_window_size
        # --- default values ---
        self.face_conf_threshold = 0.5
        self.dp_tempcoh_ckpt = os.path.join(ROOT_DIR, 'models', 'DP-TempCoh', 'weights', 'dynamic_DPTempCoh.pth')
        # --- override from YAML-like config dict if provided ---
        if config is not None:
            rt = config.get('runtime', {})
            detect_cfg = config.get('detect', {})
            model_cfg = config.get('model', {})
            self.upscale = rt.get('scale', self.upscale)
            self.detect_stride = detect_cfg.get('stride', self.detect_stride)
            self.iou_threshold = detect_cfg.get('iou_threshold', self.iou_threshold)
            self.max_faces_per_frame = detect_cfg.get('max_faces_per_frame', self.max_faces_per_frame)
            self.temporal_window_size = rt.get('temporal_window_size', self.temporal_window_size)
            self.face_conf_threshold = detect_cfg.get('face_conf_threshold', self.face_conf_threshold)
            dp_ckpt = model_cfg.get('dp_tempcoh', {}).get('ckpt')
            if dp_ckpt:
                self.dp_tempcoh_ckpt = dp_ckpt if os.path.isabs(dp_ckpt) else os.path.join(ROOT_DIR, dp_ckpt)
        # --- 最適化フラグ ---
        self.use_compile = use_compile
        self.use_amp = use_amp
        self.use_channels_last = use_channels_last

        self._detector = None
        self._face_helpers_pool = []
        self._dp_tempcoh = preload_dp_model
        self._compiled_dp_tempcoh = None
        self._compiled_bg_upsampler = None

    def _initialize_detector(self):
        if self._detector is None:
            self._detector = init_detection_model('retinaface_resnet50', device=self.device)
            profile_logger.info("RetinaFace detector initialized.")

    def _initialize_temporal_sr_model(self):
        # 既に事前ロード済みなら再ロードしない
        if self._dp_tempcoh is None:
            # Use configured ckpt path
            ckpt_path = getattr(self, 'dp_tempcoh_ckpt', None) or os.path.join(ROOT_DIR, 'models', 'DP-TempCoh', 'weights', 'dynamic_DPTempCoh.pth')
            self._dp_tempcoh = load_dp_tempcoh(ckpt_path, self.device, st_latent_size=self.temporal_window_size)
            # モデルを channels_last メモリフォーマットに変換
            if self.use_channels_last:
                self._dp_tempcoh.to(memory_format=torch.channels_last)
        profile_logger.info("DP-TempCoh model loaded.")
        if self.use_compile and self._compiled_dp_tempcoh is None and hasattr(torch, 'compile'):
            try:
                self._compiled_dp_tempcoh = torch.compile(
                    self._dp_tempcoh,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                profile_logger.info("DP-TempCoh model compiled.")
            except (RuntimeError, NotImplementedError) as e:
                profile_logger.warning(f"torch.compile for DP-TempCoh failed: {e}")
                self._compiled_dp_tempcoh = self._dp_tempcoh
        elif self._compiled_dp_tempcoh is None: # torch.compile がない場合
            self._compiled_dp_tempcoh = self._dp_tempcoh

    def _initialize_bg_upsampler(self):
        """背景 SR 用 SwinIRUpsampler を初期化し、可能なら torch.compile で高速化。

        - Graph Break が 0 であることは Dynamo explain で確認済み。
        - `mode="max-autotune"` はカーネル探索を行い、高 FPS が期待できる。
        - コンパイル失敗時は eager モデルにフォールバックし、推論を継続する。
        """
        if self.bg_upsampler is not None and self._compiled_bg_upsampler is None:
            # --- グローバル数値精度設定 (コンパイル有無に関わらず一度だけ) ---
            torch.set_float32_matmul_precision("high")
            # 背景GPUへモデルを移動
            self.bg_upsampler.model.to(self.bg_device)
            # SwinIRUpsampler を channels_last へ変換
            if self.use_channels_last:
                self.bg_upsampler.model.to(memory_format=torch.channels_last)

            if self.use_compile and hasattr(torch, "compile"):
                try:
                    self._compiled_bg_upsampler = torch.compile(
                        self.bg_upsampler.model,
                        mode="max-autotune" if os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE", "1") != "0" else "reduce-overhead",   # 高速カーネル自動探索
                        fullgraph=True,         # 静的形状で最高速
                        dynamic=False,          # 性能優先で view meta shape エラーを許容しない
                    )
                    profile_logger.info("Background SR model compiled with torch.compile (max-autotune).")
                except (RuntimeError, NotImplementedError) as e:
                    profile_logger.warning(f"torch.compile for background SR failed: {e}; falling back to eager model.")
                    self._compiled_bg_upsampler = self.bg_upsampler.model
            else:
                self._compiled_bg_upsampler = self.bg_upsampler.model

    def _get_face_helper(self) -> FaceRestoreHelper:
        return FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True, 
            device=self.device
        )

    def _update_face_helpers_pool(self, num_required: int):
        if len(self._face_helpers_pool) < num_required:
            for _ in range(num_required - len(self._face_helpers_pool)):
                self._face_helpers_pool.append(self._get_face_helper())

    def _detect_and_track_faces(self, frames_np: list[np.ndarray]) -> tuple[defaultdict[int, list[tuple[int, np.ndarray]]], Tracker]:
        self._initialize_detector()
        tracker = Tracker(distance_function="iou", distance_threshold=self.iou_threshold) # IOUベースに変更
        id2bbox_seq = defaultdict(list)
        n_frames = len(frames_np)

        for si, idx in enumerate(range(0, n_frames, self.detect_stride)):
            current_frame = frames_np[idx]
            with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and self.use_amp), dtype=torch.bfloat16):
                bboxes_with_scores = self._detector.detect_faces(current_frame, conf_threshold=self.face_conf_threshold) # スコアも取得
            
            # スコアでフィルタリングし、上位N件に絞る (max_faces_per_frame)
            bboxes_with_scores.sort(key=lambda x: x[4], reverse=True)
            bboxes = [bbox[:4] for bbox in bboxes_with_scores[:self.max_faces_per_frame]]

            detections = [
                Detection(points=np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]), data=bbox) 
                for bbox in bboxes
            ]
            
            tracked_objects = tracker.update(detections=detections, frame_idx=idx)
            for tobj in tracked_objects:
                id2bbox_seq[tobj.id].append((idx, tobj.last_detection.data))
        
        profile_logger.info(f"Detected and tracked {len(id2bbox_seq)} faces across {n_frames} frames.")
        return id2bbox_seq, tracker

    def _align_and_crop_faces(self, frames_np: list[np.ndarray], id2bbox_seq: defaultdict[int, list[tuple[int, np.ndarray]]]) -> tuple[defaultdict[int, list[tuple[int, np.ndarray]]], dict]:
        """
        OpenCV warpAffineを用いた高速クロップ・warp処理。
        bboxを正方形warpし、指定されたターゲットサイズにリサイズしてid2crop_seqとaffine_dictを返す。
        """
        id2crop_seq = defaultdict(list)
        affine_dict = {}
        # DP-TempCohモデルが期待する入力サイズ (仮に512x512とする)
        TARGET_FACE_SIZE = (512, 512) # この値を調整する必要があるかもしれません

        for face_id, bbox_seq in id2bbox_seq.items():
            for frame_idx, bbox in bbox_seq:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox[:4] # Keep original float values for logging if needed
                x1, y1, x2, y2 = map(int, bbox)
                w, h = x2 - x1, y2 - y1
                
                if w <= 0 or h <= 0:
                    continue

                L = max(w, h)
                
                src_pts = np.array([[x1, y1], [x2, y1], [x1, y2]], dtype=np.float32)
                dst_pts_warp = np.array([[0, 0], [L, 0], [0, L]], dtype=np.float32)
                M = cv2.getAffineTransform(src_pts, dst_pts_warp)
                
                warped_crop = cv2.warpAffine(frames_np[frame_idx], M, (L, L), flags=cv2.INTER_LINEAR)

                if warped_crop.shape[0] == 0 or warped_crop.shape[1] == 0:
                    continue
                
                resized_crop = cv2.resize(warped_crop, TARGET_FACE_SIZE, interpolation=cv2.INTER_LINEAR)
                
                id2crop_seq[face_id].append((frame_idx, resized_crop))
                affine_dict[(face_id, frame_idx)] = M 
        
        #print(f"[INFO][core_utils._align_and_crop_faces] Aligned and cropped faces for {len(id2crop_seq)} tracks to target size {TARGET_FACE_SIZE}. Total input bbox tracks: {len(id2bbox_seq)}.", flush=True)
        return id2crop_seq, affine_dict


    def _preprocess_crops_for_temporal_sr(self, crops: list[np.ndarray]) -> torch.Tensor:
        """クロップ画像のリストを前処理し、バッチテンソルに変換する"""
        if not crops:
            return torch.empty(0)
        #print(f"  [DEBUG][core_utils._preprocess_crops_for_temporal_sr] Shape of first crop: {crops[0].shape}", flush=True)

        # NumPyレベルでスタックと型変換
        crops_np = np.stack(crops, axis=0).astype(np.float32) / 255.0
        
        # BGR -> RGB, 正規化
        crops_np = crops_np[..., ::-1]  # BGR to RGB
        crops_np = (crops_np - 0.5) / 0.5      # Normalize to [-1, 1]
        
        # PyTorchテンソルに変換し、次元を調整
        # (N, H, W, C) -> (N, C, H, W)
        crop_t_intermediate = torch.from_numpy(crops_np.transpose(0, 3, 1, 2)).contiguous()
        #print(f"  [DEBUG][core_utils._preprocess_crops_for_temporal_sr] Shape after transpose (crop_t_intermediate): {crop_t_intermediate.shape}", flush=True)
        
        final_crop_t = crop_t_intermediate.unsqueeze(0)
        if self.use_channels_last:
            final_crop_t = final_crop_t.contiguous(memory_format=torch.channels_last)
        final_crop_t = final_crop_t.to(self.device)
        #print(
        #    f"  [DEBUG][core_utils._preprocess_crops_for_temporal_sr] Shape after unsqueeze, channels_last and to(device) (final_crop_t): {final_crop_t.shape}",
        #    flush=True,
        #)
        return final_crop_t

    def _inference_face_batch(self, id2crop_seq: dict[int, list[tuple[int, np.ndarray]]], affine_dict: dict) -> dict[int, list[tuple[int, np.ndarray]]]:
        self._initialize_temporal_sr_model()
        id2restored_seq = defaultdict(list)
        face_sr_model = self._compiled_dp_tempcoh if self._compiled_dp_tempcoh else self._dp_tempcoh
        face_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        done_evt = torch.cuda.Event() if torch.cuda.is_available() else None

        for face_id, crop_seq_with_indices in id2crop_seq.items():
            crop_seq_with_indices.sort(key=lambda x: x[0])
            frame_indices = [item[0] for item in crop_seq_with_indices]
            crops = [item[1] for item in crop_seq_with_indices]

            if not crops:
                continue

            if len(crops) > self.temporal_window_size:
                pass # DP-TempCohの挙動に依存
            
            crop_t = self._preprocess_crops_for_temporal_sr(crops)
            if crop_t.numel() == 0: continue

            with torch.no_grad(), (torch.cuda.stream(face_stream) if face_stream else contextlib.nullcontext()), torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and self.use_amp), dtype=torch.bfloat16):
                model_output = face_sr_model(crop_t) # (1, N, C, H, W)
            
            if done_evt is not None:
                done_evt.record(face_stream)

            # モデル出力の型と内容をデバッグログに出力
            if isinstance(model_output, tuple):
                restored_faces_tensor = model_output[0]
            else:
                # タプルでない場合はそのまま使用
                restored_faces_tensor = model_output
            
            # restored_faces_tensor が None でないことを確認
            if restored_faces_tensor is None:
                continue

            # restored_faces_tensor が実際にテンソルであることを確認
            if not isinstance(restored_faces_tensor, torch.Tensor):
                continue

            # --- 5次元(B,T,C,H,W)の場合は時間次元を squeeze ---
            if restored_faces_tensor.ndim == 5 and restored_faces_tensor.shape[1] == 1:
                restored_faces_tensor = restored_faces_tensor.squeeze(1)

            restored_faces_np = tensor2img(restored_faces_tensor, rgb2bgr=True, min_max=(-1, 1))

            # tensor2img can return a single np.ndarray if the batch size of the input tensor is 1.
            # We need to ensure it's always a list for the subsequent loop.
            if isinstance(restored_faces_np, np.ndarray):
                restored_faces_np = [restored_faces_np]

            # Unpack restored faces and map them back using metadata
            for i, restored_face in enumerate(restored_faces_np):
                id2restored_seq[face_id].append((frame_indices[i], restored_face))
        
        # 背景 SR 側で wait_event するため保持
        self._face_done_event = done_evt
        profile_logger.info(f"Performed face SR for {len(id2restored_seq)} tracks.")
        return id2restored_seq

    def _inference_background_batch(self, frames_tensor: torch.Tensor, no_face_indices: list[int], frames_np: list) -> Optional[list[np.ndarray]]:
        # --- Debug: 呼び出し回数を可視化 ---
        self._bg_call_count = getattr(self, "_bg_call_count", 0) + 1
        profile_logger.debug(f"[DBG] _inference_background_batch called {self._bg_call_count} times")
        profile_logger.debug(f"[DBG] bg_sr_model type: {type(self._compiled_bg_upsampler)}")
        profile_logger.debug(f"[DBG] frames_tensor id: {id(frames_tensor)}")
        profile_logger.debug(f"[DBG] frames_tensor shape: {frames_tensor.shape if hasattr(frames_tensor, 'shape') else 'N/A'}")
        profile_logger.debug(f"[DBG] devices - face_device={self.device}, bg_device={self.bg_device}, current_cuda={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")
        self._initialize_bg_upsampler()
        if not no_face_indices or self.bg_upsampler is None or self._compiled_bg_upsampler is None:
            profile_logger.debug("[DBG] Early exit _inference_background_batch: no_face_indices empty or model not ready")
            return None

        bg_stream = torch.cuda.Stream(device=self.bg_device) if torch.cuda.is_available() else None
        bg_sr_model = self._compiled_bg_upsampler
        # --- 顔 SR 完了イベントを待機し、ストリームを重ね合わせ ---
        if bg_stream and self.bg_device == self.device and getattr(self, '_face_done_event', None) is not None:
            bg_stream.wait_event(self._face_done_event)
            profile_logger.debug(f"[DBG] bg_stream waiting on face_done_event on device {self.bg_device}")

        imgs_to_process_list = [frames_tensor[i] for i in no_face_indices]
        if not imgs_to_process_list:
            return None
        
        # --- 追加デバッグ: 入力 numpy フレームの値域を確認 (先頭 5 件) ---
        for dbg_i in range(min(5, len(no_face_indices))):
            idx = no_face_indices[dbg_i]
            frame_np = frames_np[idx]
            np_min, np_max, np_mean = frame_np.min(), frame_np.max(), frame_np.mean()
            #print(
            #    f"[DEBUG][core_utils._inference_background_batch] Pre-stack frame {idx}: np.min={np_min}, np.max={np_max}, np.mean={np_mean:.2f}",
            #    flush=True,
            #)
        
        # --- Transfer & preprocess in the same CUDA stream to avoid race conditions ---
        with (torch.cuda.stream(bg_stream) if bg_stream else contextlib.nullcontext()):
            imgs_t = torch.stack(imgs_to_process_list, dim=0).to(self.bg_device, non_blocking=True)
            # --- Debug: stacked stats ---
            stacked_min = imgs_t.min().item()
            stacked_max = imgs_t.max().item()
            stacked_mean = imgs_t.float().mean().item()
            #print(
            #    f"[DEBUG][core_utils._inference_background_batch] imgs_t stacked (uint8) stats: min={stacked_min}, max={stacked_max}, mean={stacked_mean:.3f}",
            #    flush=True,
            #)

            # Ensure NCHW
            if imgs_t.ndim == 4 and imgs_t.shape[1] != 3 and imgs_t.shape[3] == 3:
                imgs_t = imgs_t.permute(0, 3, 1, 2).contiguous()

            # BGR -> RGB (SwinIR expects RGB)
            imgs_t = imgs_t[:, [2, 1, 0], ...]
            #print("[DEBUG][core_utils._inference_background_batch] Converted BGR -> RGB for model input", flush=True)

            # Normalize to 0-1
            if imgs_t.dtype != torch.float32:
                imgs_t = imgs_t.float()
            if imgs_t.max().item() > 1.01:
                imgs_t = imgs_t / 255.0
            norm_min = imgs_t.min().item()
            norm_max = imgs_t.max().item()
            norm_mean = imgs_t.mean().item()
            #print(
            #    f"[DEBUG][core_utils._inference_background_batch] imgs_t normalized stats: min={norm_min:.3f}, max={norm_max:.3f}, mean={norm_mean:.3f}",
            #    flush=True,
            #)

            # channels_last conversion if requested
            if self.use_channels_last:
                imgs_t = imgs_t.contiguous(memory_format=torch.channels_last)

            # Inference within the same stream
            with torch.no_grad(), torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(torch.cuda.is_available() and self.use_amp)
            ):
                bg_outputs_tensor = bg_sr_model(imgs_t)

        # Ensure default stream waits for bg_stream completion before using outputs
        if bg_stream:
            torch.cuda.current_stream().wait_stream(bg_stream)
        
        # --- 追加デバッグ: stack 後テンソルの値域 ---
        stacked_min = imgs_t.min().item()
        stacked_max = imgs_t.max().item()
        stacked_mean = imgs_t.float().mean().item()  # uint8 -> float32 for mean
        #print(
        #    f"[DEBUG][core_utils._inference_background_batch] imgs_t stacked (uint8) stats: min={stacked_min}, max={stacked_max}, mean={stacked_mean:.3f}",
        #    flush=True,
        #)
        


        
        # --- 追加: 出力テンソルの値域デバッグ ---
        sample_out = bg_outputs_tensor[0]
        #print(f"[DEBUG][core_utils._inference_background_batch] bg_outputs_tensor sample stats: min={sample_out.min().item():.3f}, max={sample_out.max().item():.3f}, mean={sample_out.mean().item():.3f}", flush=True)
        
        if bg_stream:
            torch.cuda.current_stream().wait_stream(bg_stream)
        
        # tensor2img を1枚ずつ呼び出す
        bg_outputs_np = []
        for i in range(bg_outputs_tensor.shape[0]):
            arr = tensor2img(bg_outputs_tensor[i], rgb2bgr=True, min_max=(0,1))
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            bg_outputs_np.append(arr)
        # --- shape debug start ---
        fixed_bg_outputs_np = []
        for i, arr in enumerate(bg_outputs_np):
            if i < len(no_face_indices):
                h, w, _ = frames_np[no_face_indices[i]].shape
            fixed_bg_outputs_np.append(arr)
        # --- shape debug end ---
        profile_logger.info(f"Performed background SR for {len(no_face_indices)} frames.")
        return fixed_bg_outputs_np

    # --- Compose Strategy + Facade ---
    class ComposeStrategy(ABC):
        @abstractmethod
        def apply(self, sr, *args, **kwargs):
            pass

    class InitComposeStrategy(ComposeStrategy):
        def apply(self, sr, frames_np, bg_sr_results, no_face_indices):
            final = [None] * len(frames_np)
            if bg_sr_results is not None:
                for i, idx in enumerate(no_face_indices):
                    final[idx] = bg_sr_results[i]
            return final

    class PasteStrategy(ComposeStrategy):
        def apply(self, sr, final_results, frames_np, id2restored_seq, id2bbox_seq, affine_dict, weight):
            helper = sr._get_face_helper()
            for face_id, seq in id2restored_seq.items():
                bdict = {fidx: affine_dict.get((face_id, fidx)) for fidx, _ in id2bbox_seq[face_id]}
                for fidx, restored in seq:
                    helper.clean_all()
                    # base_img は既存のアップスケール背景があればそれを使用、無ければオリジナルをリサイズ
                    if final_results[fidx] is not None:
                        base_img = final_results[fidx]
                    else:
                        h, w = frames_np[fidx].shape[:2]
                        base_img = cv2.resize(frames_np[fidx], (w*sr.upscale, h*sr.upscale), interpolation=cv2.INTER_LANCZOS4)
                    helper.read_image(base_img)
                    
                    if restored is None:
                        continue
 
                    helper.cropped_faces = []
                    # dtype 保証
                    restored_u8 = restored if restored.dtype == np.uint8 else np.clip(restored,0,255).astype(np.uint8)
                    helper.restored_faces = [restored_u8]
                    
                    current_affine_matrix = bdict.get(fidx)
                    if current_affine_matrix is None:
                        continue 
                    # --- 逆アフィン行列を計算し、アップスケール倍率で平行移動成分をスケール ---
                    inv = cv2.invertAffineTransform(current_affine_matrix.copy())
                    inv[:, 2] *= sr.upscale
                    helper.inverse_affine_matrices = [inv]
                    
                    pasted = helper.paste_faces_to_input_image(upsample_img=base_img)
                    pasted = np.clip(pasted, 0, 255).astype(np.uint8)
                    final_results[fidx] = pasted
            return final_results

    class FillMissingStrategy(ComposeStrategy):
        def apply(self, sr, final_results, frames_np):
            for i, v in enumerate(final_results):
                if v is None:
                    h, w = frames_np[i].shape[:2]
                    final_results[i] = cv2.resize(frames_np[i], (w*sr.upscale, h*sr.upscale), interpolation=cv2.INTER_LANCZOS4)
            return final_results

    class ComposeFacade:
        def __init__(self, strategies: list["ComposeStrategy"]):
            self.strategies = strategies
        def compose(self, sr, frames_np, id2restored_seq, id2bbox_seq, bg_sr_results, no_face_indices, affine_dict, weight):
            final = self.strategies[0].apply(sr, frames_np, bg_sr_results, no_face_indices)
            final = self.strategies[1].apply(sr, final, frames_np, id2restored_seq, id2bbox_seq, affine_dict, weight)
            final = self.strategies[2].apply(sr, final, frames_np)
            return final

    def _compose_final_results(self,
                               frames_np: list[np.ndarray],
                               id2restored_seq: defaultdict[int, list[tuple[int, np.ndarray]]],
                               id2bbox_seq: defaultdict[int, list[tuple[int, np.ndarray]]],
                               bg_sr_results: Optional[list[np.ndarray]],
                               no_face_indices: list[int],
                               affine_dict: dict,
                               weight: float = 0.5) -> list[np.ndarray]:
        """
        リファクタ済み: 前処理・貼り付け・補完をヘルパー呼び出しで実行
        """
        # Facade+Strategyを用いた合成
        facade = self.ComposeFacade([self.InitComposeStrategy(), self.PasteStrategy(), self.FillMissingStrategy()])
        final = facade.compose(self, frames_np, id2restored_seq, id2bbox_seq, bg_sr_results, no_face_indices, affine_dict, weight)
        profile_logger.info("Composed final results.")
        return final

    @torch.no_grad()
    def batch_enhance_tracked(self, frames_np: list[np.ndarray], frames_tensor: torch.Tensor, detect_stride: int = None, iou_th: float = None, weight: float = 0.5) -> list[np.ndarray]:
        """
        顔追跡による検出回数削減＋バッチ推論で高速化するバージョン。
        iou_th: norfair の追跡距離閾値
        """
        n_frames = len(frames_np)
        if n_frames == 0:
            return []
        start_time = time.time()

        # --- パラメータ設定（インスタンス変数 or 引数） ---
        detect_stride = detect_stride if detect_stride is not None else self.detect_stride
        iou_th = iou_th if iou_th is not None else self.iou_threshold
        device = self.device

        # --- RetinaFace detector 初期化 ---
        self._initialize_detector()
        detector = self._detector

        # --- Norfair Tracker 初期化 ---
        tracker = Tracker(distance_function="iou", distance_threshold=iou_th)

        # --- ストライドごとに顔検出（バッチ） ---
        bboxes_list = []
        batch_idxs = []
        for idx in range(0, n_frames, detect_stride):
            img = frames_np[idx]
            with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and self.use_amp), dtype=torch.bfloat16):
                bboxes = detector.detect_faces(img, conf_threshold=self.face_conf_threshold)
            bboxes_list.append(bboxes)
            batch_idxs.append(idx)

        # --- Norfair Trackerで顔IDごとにbbox列を構築 ---
        id2bbox_seq = defaultdict(list)
        for rel_idx, bboxes in enumerate(bboxes_list):
            abs_idx = batch_idxs[rel_idx]
            detections = [] # Reset for current frame's detections
            # single_detection_batch is the output of self.face_detect_model.detect_faces for current_frame_np
            # Expected to be a list of arrays (e.g., [array([x1,y1,x2,y2,score]), ...]) or None

            if bboxes is not None and len(bboxes) > 0:
                for face_idx, single_face_data in enumerate(bboxes): # Iterate directly over the list of detected faces
                    # single_face_data is one detected face's data, e.g., np.array([x1,y1,x2,y2,score])
                    if not hasattr(single_face_data, '__len__') or len(single_face_data) < 4: # Check if it has at least 4 elements for bbox
                        continue
                    
                    bbox_coords_raw = single_face_data[:4]
                    # score = single_face_data[4] if len(single_face_data) > 4 else None # Optional: score for logging or filtering

                    try:
                        bbox_coords = np.array(bbox_coords_raw, dtype=float)
                        if bbox_coords.shape != (4,):
                            continue
                    except Exception as e:
                        continue
                    
                    # Norfair's Detection object. points and data are both set to the bbox.
                    detections.append(Detection(points=bbox_coords, data=bbox_coords))
            else:
                pass
            # tracker.update に渡す直前の detections の内容を詳細にログ出力
            if detections: # detectionsが空でない場合のみログ出力
                pass

            active_tracks = tracker.update(detections)
            if not active_tracks:
                pass
            else:
                for t in active_tracks:
                    if t.last_detection.data is not None and hasattr(t.last_detection.data, 'shape'): # Ensure data and shape exist
                        id2bbox_seq[t.id].append((abs_idx, t.last_detection.data))

        profile_logger.info(f"[Profile] Face detection and tracking took {time.time() - start_time:.4f}s")
        current_time = time.time()

        # 3. 顔のアラインメントとクロップ
        id2crop_seq, affine_dict = self._align_and_crop_faces(frames_np, id2bbox_seq)
        profile_logger.info(f"Face alignment and cropping took {time.time() - current_time:.4f}s")
        current_time = time.time()

        # 4. 顔SRバッチ推論
        if not id2crop_seq:
            id2restored_seq = defaultdict(list)
        else:
            id2restored_seq = self._inference_face_batch(id2crop_seq, affine_dict)
        profile_logger.info(f"Face SR inference took {time.time() - current_time:.4f}s")
        current_time = time.time()

        # 5. 背景SRバッチ推論 (全フレーム対象)
        #    SwinIR で背景を高解像化し、その後に顔SR結果を貼り付ける
        bg_frame_indices = [i for i in range(n_frames) if frames_np[i].max() > 0]
        skipped_zero_bg = [i for i in range(n_frames) if frames_np[i].max() == 0]
        if skipped_zero_bg:
            profile_logger.info(
                f"Skipped {len(skipped_zero_bg)} all-zero frames ({skipped_zero_bg[:8]}…) for BG SR to prevent NaNs. They will be Lanczos-upscaled later.")
        
        # 背景SR推論
        #   --- 二重実行防止: 同じフレーム集合に対しては再実行しない ---
        # キャッシュキーはフレーム index のタプル + frames_tensor の object id で一意化
        curr_bg_key = (tuple(bg_frame_indices), id(frames_tensor))
        if getattr(self, '_cached_bg_key', None) == curr_bg_key and getattr(self, '_cached_bg_results', None) is not None:
            profile_logger.info("[Skip] Cached BG SR reused for current batch (identical content key).")
            bg_sr_results = self._cached_bg_results
        else:
            bg_sr_results = self._inference_background_batch(frames_tensor, bg_frame_indices, frames_np)
            # キャッシュ更新
            self._cached_bg_key = curr_bg_key
            self._cached_bg_results = bg_sr_results


        profile_logger.info(f"Background SR inference took {time.time() - current_time:.4f}s")
        current_time = time.time()

        # 6. 結果の合成
        final_results = self._compose_final_results(
            frames_np,
            id2restored_seq,
            id2bbox_seq,
            bg_sr_results,
            bg_frame_indices,
            affine_dict,
            weight,
        )
        profile_logger.info(f"Result composition took {time.time() - current_time:.4f}s")
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        profile_logger.info(f"Total batch_enhance_tracked took {time.time() - start_time:.4f}s for {n_frames} frames.")
        return final_results

    # --- 元の enhance, batch_enhance メソッド (必要であればリファクタリング対象) ---
    @torch.no_grad()
    def enhance(self, img: np.ndarray, has_aligned: bool = False, only_center_face: bool = False, paste_back: bool = True, weight: float = 0.5) -> np.ndarray:
        """
        1枚画像の顔・背景超解像。最新のリファクタ流儀で整理。
        - 顔検出・アラインメント・SR・合成をバッチ系と同じヘルパーで統一
        - パラメータ・型保証・エラーハンドリングも一元化
        """
        self._initialize_detector()
        face_helper = self._get_face_helper()
        face_helper.clean_all()
        upscale = self.upscale
        device = self.device
        result_img = None

        # --- 1. アライン済み画像かどうか ---
        if has_aligned:
            face_helper.cropped_faces = [img]
            affine_matrices = [np.eye(2, 3, dtype=np.float32)]
        else:
            face_helper.read_image(img)
            bboxes_with_scores = self._detector.detect_faces(img, conf_threshold=self.face_conf_threshold)
            if not bboxes_with_scores:
                # 顔が検出されなかった場合は背景SRまたはアップスケール
                if self.bg_upsampler:
                    with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and self.use_amp), dtype=torch.bfloat16):
                        result_img = self.bg_upsampler.enhance(img, outscale=upscale)[0]
                else:
                    result_img = cv2.resize(img, (img.shape[1]*upscale, img.shape[0]*upscale), interpolation=cv2.INTER_LANCZOS4)
                return result_img
            # only_center_face: 最も大きい顔のみ選択
            if only_center_face:
                bboxes_with_scores.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
                bboxes_with_scores = [bboxes_with_scores[0]]
            # 複数顔対応
            for bbox in bboxes_with_scores:
                face_helper.get_face_landmarks_5_from_bbox(bbox[:4], only_center_face=False)
                face_helper.align_warp_face()
        # --- 2. 顔SR（DP-TempCoh等） ---
        restored_faces = []
        if face_helper.cropped_faces:
            self._initialize_temporal_sr_model()
            face_sr_model = self._compiled_dp_tempcoh if self._compiled_dp_tempcoh else self._dp_tempcoh
            for crop in face_helper.cropped_faces:
                crop_t = self._preprocess_crops_for_temporal_sr([crop]) # shape: (1, 3, H, W)
                if crop_t.numel() == 0: continue

                try:
                    with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and self.use_amp), dtype=torch.bfloat16):
                        output_tensor = face_sr_model(crop_t) # (1,1,C,H,W)
                    restored_face = tensor2img(output_tensor.squeeze(0), rgb2bgr=True, min_max=(-1,1))[0]
                    restored_faces.append(restored_face)
                except (RuntimeError, ValueError) as error:
                    logging.error(f"Error during face SR: {error}", exc_info=True)
                    restored_faces.append(crop)
            face_helper.restored_faces = restored_faces
        else:
            # 顔がクロップできなかった場合は背景SRまたはアップスケール
            if self.bg_upsampler:
                with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and self.use_amp), dtype=torch.bfloat16):
                    result_img = self.bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                result_img = cv2.resize(img, (img.shape[1]*upscale, img.shape[0]*upscale), interpolation=cv2.INTER_LANCZOS4)
            return result_img
        # --- 3. 合成 ---
        if paste_back:
            # 背景SR
            upsampled_bg = None
            if self.bg_upsampler:
                with torch.amp.autocast('cuda', enabled=(torch.cuda.is_available() and self.use_amp), dtype=torch.bfloat16):
                    upsampled_bg = self.bg_upsampler.enhance(img, outscale=upscale)[0]
            result_img = face_helper.paste_faces_to_input_image(upsample_img=upsampled_bg, weight=weight)
        else:
            # 顔SRのみ返す
            result_img = restored_faces[0] if restored_faces else img
        # --- 4. 型保証・チャンネル数保証 ---
        if not (isinstance(result_img, np.ndarray) and result_img.dtype == np.uint8 and result_img.ndim == 3):
            h, w = img.shape[:2]
            target_h, target_w = h * upscale, w * upscale
            result_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)
        if result_img.shape[2] == 1:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
        elif result_img.shape[2] == 4:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2BGR)
        return result_img

    def batch_enhance(self, frames: list[np.ndarray], only_center_face=False, weight=0.5) -> list[np.ndarray]:
        # このメソッドはフレームごとにenhanceを呼び出すだけなので、大きな変更は不要かもしれない
        # ただし、frames_tensorを渡すように変更したり、効率化の余地はある
        # 現状は、リファクタリングされたenhanceメソッドを呼び出す形にする
        results = []
        for frame in frames:
            results.append(self.enhance(frame, only_center_face=only_center_face, weight=weight))
        return results
