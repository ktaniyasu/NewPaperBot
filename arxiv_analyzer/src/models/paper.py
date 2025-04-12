from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

class Author(BaseModel):
    """論文の著者を表すモデル"""
    name: str
    affiliations: Optional[List[str]] = Field(default_factory=list)

class PaperMetadata(BaseModel):
    """論文のメタデータを表すモデル"""
    arxiv_id: str
    title: str
    authors: List[Author]
    abstract: str
    pdf_url: str
    published_date: datetime
    categories: List[str]
    last_updated: datetime

class AnalysisResult(BaseModel):
    """LLMによる論文解析結果を表すモデル"""
    summary: str = Field(description="論文の要約")
    novelty: str = Field(description="論文の新規性")
    methodology: str = Field(description="研究手法の説明")
    results: str = Field(description="主要な結果と成果")
    future_work: str = Field(description="今後の課題と展望")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)

class Paper(BaseModel):
    """論文の完全な情報を表すモデル"""
    metadata: PaperMetadata
    analysis: Optional[AnalysisResult] = None
    full_text: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    error_log: Optional[str] = None
