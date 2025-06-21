from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    """
    Database configurations:

    - Values will be picked from `.env` file.
    - The prefix for database settings in `.env` file is `QDRANT__`.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="QDRANT__", extra="ignore"
    )

    URL: str = Field(env="URL")
    API_KEY: str = Field(env="API_KEY")
    COLLECTION_NAME: str = Field(env="COLLECTION_NAME")

class GroqSettings(BaseSettings):
    """
    Groq configurations:

    - Values will be picked from `.env` file.
    - The prefix for database settings in `.env` file is `GROQ__`.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="GROQ__", extra="ignore"
    )

    API_KEY: str = Field(env="API_KEY", default=None)

class OpenAISettings(BaseSettings):
    """
    OpenAI LLM configurations:

    - Values will be picked from `.env` file.
    - The prefix for database settings in `.env` file is `OPENAI__`.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OPENAI__", extra="ignore"
    )

    API_KEY: str = Field(env="API_KEY", default=None)


class CohereSettings(BaseSettings):
    """
    Cohere configurations:

    - Values will be picked from `.env` file.
    - The prefix for database settings in `.env` file is `COHERE__`.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="COHERE__", extra="ignore"
    )

    API_KEY: str = Field(env="API_KEY", default=None)

class HFSettings(BaseSettings):
    """
    HF configurations:

    - Values will be picked from `.env` file.
    - The prefix for database settings in `.env` file is `HF__`.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="HF__", extra="ignore"
    )

    TOKEN: str = Field(env="TOKEN", default=None)

class LLMSmithSettings(BaseSettings):
    """
    LLMSmith configurations:

    - Values will be picked from `.env` file.
    - The prefix for database settings in `.env` file is `LLMSMITH__`.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LLMSMITH__", extra="ignore"
    )
    DEBUG: bool = Field(env="DEBUG", default=False)
    
class LangsmithSettings(BaseSettings):
    """
    Langsmith configurations:

    - Values will be picked from `.env` file.
    - The prefix for database settings in `.env` file is `LANGSMITH__`.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LANGSMITH__", extra="ignore"
    )
    API_KEY: str = Field(env="API_KEY", default=None)
    TRACING: str = Field(env="TRACING")


class Settings(BaseSettings):
    """
    Application configurations:

    - Values will be picked from `.env` file.
    - Nested settings (like `QdrantSettings`) are separated by `__` in `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="__", extra="ignore"
    )

    # APP_NAME: str = "LLMSMITH RAG App"
    APP_NAME: str = "ITB CARE RAG App"
    ENVIRONMENT: str = Field(env="ENVIRONMENT", default="local")
    GROQ: GroqSettings
    OPENAI: OpenAISettings
    QDRANT: QdrantSettings
    COHERE: CohereSettings
    LLMSMITH: LLMSmithSettings
    LANGSMITH: LangsmithSettings

    def is_local(self):
        return self.ENVIRONMENT == "local"