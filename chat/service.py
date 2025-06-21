from textwrap import dedent
import cohere
# from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
import groq
from langsmith import traceable
import openai
from qdrant_client import AsyncQdrantClient
from llmsmith.task.retrieval.vector.qdrant import QdrantRetriever, QdrantQueryOptions
from llmsmith.reranker.cohere import CohereReranker, CohereRerankerOptions
from llmsmith.task.textgen.openai import OpenAITextGenTask, OpenAITextGenOptions
from llmsmith.task.textgen.groq import GroqTextGenTask, GroqTextGenOptions
from llmsmith.job.job import SequentialJob

from config import settings


preprocess_prompt = (
    dedent("""Kamu adalah seorang psikiater yang ahli dalam bidang kesehatan mental. 
           Jawab pertanyaan ini dengan pengetahuan yang kamu miliki: {{root}}. 
           Jangan menyebutkan ‘Jawaban pertanyaan tersebut adalah’""")
    .strip("\n")
    .replace("\n", " ")
)

class RAGService:
    def __init__(
        self,
        # llm_client: groq.AsyncGroq,
        vectordb_client: AsyncQdrantClient,
        reranker_client: cohere.AsyncClient,
        embedder: SentenceTransformer,
        llm_client: openai.AsyncOpenAI,
        # embedder: TextEmbedding,
        **_,
    ) -> None:
        self.llm_client = llm_client
        self.vectordb_client = vectordb_client
        self.reranker_client = reranker_client
        self.embedder = embedder
        
    @traceable
    async def chat(self, user_prompt):
        # Create Cohere reranker
        reranker = CohereReranker(client=self.reranker_client, options=CohereRerankerOptions(top_n=3, model="rerank-v3.5", return_documents=True))
        # reranker = cohere.rera

        # Embedding function to be passed into the Qdrant retriever
        def embedding_func(x):
            return self.embedder.encode(x)
            # return list(self.embedder.query_embed(x))
        
        # Define the Qdrant retriever task. The embedding function and reranker are passed as parameters.
        retrieval_task = QdrantRetriever(
            name="qdrant-retriever",
            client=self.vectordb_client,
            collection_name=settings.QDRANT.COLLECTION_NAME,
            embedding_func=embedding_func,
            embedded_field_name="text",
            query_options= QdrantQueryOptions(limit=10),
            reranker=reranker,
        )

        # Define the OpenAI LLM task for rephrasing the query
        # preprocess_task = GroqTextGenTask(
        #     name="groq-preprocessor",
        #     llm=self.llm_client,
        #     llm_options=GroqTextGenOptions(model="llama3-8b-8192", temperature=0),
        # )
        preprocess_task = OpenAITextGenTask(
            name="openai-preprocessor",
            llm=self.llm_client,
            llm_options=OpenAITextGenOptions(model="gpt-4o-mini", temperature=0),
        )

        # Define the OpenAI LLM task for answering the query
        # answer_generate_task = GroqTextGenTask(
        #     name="groq-answer-generator",
        #     llm=self.llm_client,
        #     llm_options=GroqTextGenOptions(model="llama3-8b-8192", temperature=0),
        # )
        answer_generate_task = OpenAITextGenTask(
            name="openai-answer-generator",
            llm=self.llm_client,
            llm_options=OpenAITextGenOptions(model="gpt-4o-mini", temperature=0),
        )

        # define the sequence of tasks
        # {{root}} is a special placeholer in `input_template` which will be replaced with the prompt entered by the user (`user_prompt`).
        # The placeholder {{qdrant-retriever.output}} will be replaced with the output from Qdrant DB retriever task.
        # The placeholder {{openai-preprocessor.output}} will be replaced with the output from the query preprocessing task done by OpenAI LLM.
        job: SequentialJob[str, str] = (
            SequentialJob()
            .add_task(
                preprocess_task,
                input_template=preprocess_prompt,
            )
            .add_task(retrieval_task, input_template="{{openai-preprocessor.output}}")
            .add_task(
                answer_generate_task,
                input_template="""Diberikan konteks berikut: {{qdrant-retriever.output}} Jawab pertanyaan ini selengkap mungkin sesuai dengan konteks yang diberikan dengan jawaban yang sesuai untuk target user yaitu mahasiswa: {{root}}. Jangan menyebutkan ‘Berdasarkan konteks yang diberikan’"""
            )
        )

        # Now, run the job
        await job.run(user_prompt)

        return job.task_output("openai-answer-generator")