from typing import Any, Protocol


class BaseLLMProvider(Protocol):
    """LLM プロバイダの共通インターフェース"""

    def supportsFileUpload(self) -> bool:
        ...

    async def generateStructuredFromFile(
        self, prompt: str, pdfPath: str, schema: Any, **kwargs
    ) -> dict[str, Any]:
        ...

    async def generateStructuredFromText(
        self, prompt: str, text: str, schema: Any, **kwargs
    ) -> dict[str, Any]:
        ...

    async def translate(self, text: str, **kwargs) -> str:
        ...
