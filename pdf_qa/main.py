import os
from pathlib import Path
from typing import Tuple
from textwrap import dedent
import gradio as gr
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import re

# from langchain_huggingface import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader


def get_openai_apikey() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    else:
        raise ValueError("OPENAI_API_KEY environment not found")


def extract_text_from_pdf(pdf_path: str) -> str:
    pdf_text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text().replace("\n", "").replace("\r", "")
    return pdf_text


def process_pdf(pdf_file: gr.File) -> Tuple[str, None | FAISS]:
    try:
        pdf_text = extract_text_from_pdf(pdf_file.name)

        if not pdf_text.strip():
            return (
                "PDFからテキストを抽出できませんでした。PDFが空か、テキスト抽出に対応していない可能性があります。",
                None,
            )

        # テキストを分割
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small", chunk_size=500, chunk_overlap=50
        )
        split_texts = text_splitter.split_text(pdf_text)

        # ベクトルストアを構築
        embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        # embedding = OpenAIEmbeddings(
        #     api_key=get_openai_apikey(), model="text-embedding-3-small"
        # )
        vectorstore = FAISS.from_texts(
            split_texts,
            embedding,
        )

        return "PDFの処理が完了しました。質問を入力してください。", vectorstore
    except Exception as e:
        return (
            f"PDFの処理中にエラーが発生しました: {str(e)}\n詳細: {type(e).__name__}",
            None,
        )


def answer_question(question: str, vectorstore: None | FAISS, model_name: str) -> str:
    if vectorstore is None:
        return "PDFがアップロードされていないか、処理中にエラーが発生しました。先にPDFをアップロードしてください。"

    # モデル名の検証と適切なモデルの選択
    if model_name == "elyza/Llama-3-8B-q4-GGUF":
        model_path = Path("rag/models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf")
        llm = LlamaCpp(
            model_path=model_path.as_posix(),
            n_gpu_layers=100,
            # f16_kv=True,
            temperature=0.1,
            n_ctx=2048,
        )
    elif model_name == "elyza/Llama-2-13B-q4-GGUF":
        model_path = Path(
            "rag/models/ELYZA-japanese-Llama-2-13b-fast-instruct-q4_0.gguf"
        )
        llm = LlamaCpp(
            model_path=model_path.as_posix(),
            n_gpu_layers=100,
            # f16_kv=True,
            temperature=0.1,
            n_ctx=2048,
        )
    elif model_name == "elyza/Llama-2-13B-q8-GGUF":
        model_path = Path(
            "rag/models/ELYZA-japanese-Llama-2-13b-fast-instruct-q8_0.gguf"
        )
        llm = LlamaCpp(
            model_path=model_path.as_posix(),
            n_gpu_layers=100,
            # f16_kv=True,
            temperature=0.1,
            n_ctx=2048,
        )
    elif model_name == "cyberagent/Llama-3.1-70B-Q8":
        model_path = Path("rag/models/Llama-3.1-70B-Japanese-Instruct-2407-Q8_0.gguf")
        llm = LlamaCpp(
            model_path=model_path.as_posix(),
            n_gpu_layers=50,
            f16_kv=True,
            temperature=0.1,
            n_ctx=2048,
        )
    elif model_name == "GPT-4o mini":
        model = "gpt-4o-mini"
        llm = ChatOpenAI(api_key=get_openai_apikey(), temperature=0, model_name=model)
    elif model_name == "GPT-4o":
        model = "gpt-4o"
        llm = ChatOpenAI(api_key=get_openai_apikey(), temperature=0, model_name=model)
    else:
        return f"サポートされていないモデル: {model_name}"

    # プロンプトの設定
    prompt = ChatPromptTemplate.from_template(
        dedent(
            """
    以下の前提知識を用いて、ユーザーからの質問に答えてください。

    ===
    前提知識
    {context}

    ===
    ユーザーからの質問
    {question}
    """
        )
    )

    # リトリーバーの設定
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # チェーンの構築
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 回答の生成
    return chain.invoke(question)


def pdf_qa_app() -> gr.Blocks:
    with gr.Blocks() as app:
        gr.Markdown("# PDF QA")
        gr.Markdown("PDFをアップロードし、質問に対する回答を取得するアプリケーション")

        with gr.Row():
            with gr.Column():
                # モデル選択コンポーネント
                model_dropdown = gr.Dropdown(
                    choices=[
                        "elyza/Llama-3-8B-q4-GGUF",
                        "elyza/Llama-2-13B-q8-GGUF",
                        "elyza/Llama-2-13B-q4-GGUF",
                        "cyberagent/Llama-3.1-70B-Q8",
                        "GPT-4o mini",
                        "GPT-4o",
                    ],
                    label="モデルを選択",
                    value="elyza/Llama-2-13B-q8-GGUF",
                )

                # PDFアップロードコンポーネント
                pdf_file = gr.File(label="PDFをアップロード")
                upload_button = gr.Button("PDFを処理")
                result = gr.Textbox(label="処理結果")

            with gr.Column():  # ここから縦方向に要素配置
                # 質問応答コンポーネント
                question_input = gr.Textbox(label="質問を入力してください")
                answer_button = gr.Button("回答を取得")
                answer_output = gr.Textbox(label="回答")

        # ベクトルストアを保存するための状態変数
        vectorstore_state = gr.State()

        # PDFアップロードと処理のイベントハンドラ
        upload_button.click(
            fn=process_pdf, inputs=[pdf_file], outputs=[result, vectorstore_state]
        )

        # 質問応答のイベントハンドラ
        answer_button.click(
            fn=answer_question,
            inputs=[question_input, vectorstore_state, model_dropdown],
            outputs=[answer_output],
        )

    return app


if __name__ == "__main__":
    pdf_qa_app().launch(share=True, server_port=8080)
