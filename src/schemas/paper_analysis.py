from typing import List
from pydantic import BaseModel, Field

class PaperAnalysisSchema(BaseModel):
    """LLMによる論文解析のスキーマ（フラット構造）"""
    summary: str = Field(description="論文の要約")
    novelty: str = Field(description="論文の新規性")
    methodology: str = Field(description="研究手法の説明")
    results: str = Field(description="主要な結果と成果")
    future_work: str = Field(description="今後の課題と展望")
    research_themes: List[str] = Field(description="論文から派生する新しい研究テーマ（3件）")
