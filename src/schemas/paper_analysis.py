from pydantic import BaseModel, Field


class PaperAnalysisSchema(BaseModel):
    """LLMによる論文解析のスキーマ（フラット構造）"""

    summary: str = Field(description="論文の要約", max_length=1500)
    novelty: str = Field(description="論文の新規性", max_length=1500)
    methodology: str = Field(description="研究手法の説明", max_length=1500)
    results: str = Field(description="主要な結果と成果", max_length=1500)
    future_work: str = Field(description="今後の課題と展望", max_length=1500)
    research_themes: list[str] = Field(
        description="論文から派生する新しい研究テーマ（3件）",
        min_length=3,
        max_length=3,
    )
