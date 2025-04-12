from pydantic import BaseModel, Field

class PaperAnalysisSchema(BaseModel):
    """LLMによる論文解析のスキーマ（フラット構造）"""
    summary: str = Field(description="論文の要約")
    novelty: str = Field(description="論文の新規性")
    methodology: str = Field(description="研究手法の説明")
    results: str = Field(description="主要な結果と成果")
    future_work: str = Field(description="今後の課題と展望")
