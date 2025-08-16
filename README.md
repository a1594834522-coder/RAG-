# Gemini RAG 应用

本项目是一个检索增强生成 (RAG) 应用，它使用 Google Gemini API 和 Streamlit 构建一个基于本地知识库的、简单且可交互的问答系统。

## 功能特性

-   **交互式用户界面:** 使用 Streamlit 构建，提供友好的网页界面。
-   **强大的语言模型:** 利用 Google Gemini Pro 2.5 模型进行文本生成。
-   **本地知识库:** 使用 `data` 目录中的文本文件作为答案来源。
-   **可扩展性:** 可以轻松地将您自己的文档添加到知识库中。
-   **RAG 流程:** 使用 LlamaIndex 实现 RAG 流程，以实现高效、相关的信息检索。
-   **重排序 流程:** 使用 bge-reranker-v2-m3 实现 重排序 流程，直接输出相似度分数，而不是嵌入向量。


## 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/a1594834522-coder/RAG-.git
    cd RAG-GEMINI
    ```

2.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **设置环境变量:**
    在项目根目录创建一个名为 `.env` 的文件，并添加您的 Google API 密钥:

    ```
    SILICONFLOW_API_KEY="在此处填入您的硅基流动API密钥"
    GOOGLE_API_KEY="在此处填入您的API密钥"
    ```

## 如何使用

1.  **添加您的数据:**
    将您的 `.txt/.pdf/.doc/.docx/.pptx...` 文件放入 `data` 目录。应用将自动把它们用作知识库。

2.  **运行应用:**
    执行 `start.bat` 脚本，或在终端中运行以下命令:
    ```bash
    streamlit run gemini_rag_app.py
    ```

3.  **打开应用:**
    打开您的网络浏览器，并访问 Streamlit 提供的本地 URL (通常是 `http://localhost:8501`)。

## 主要依赖

本项目的主要依赖包括:

-   `streamlit`
-   `python-dotenv`
-   `google-generativeai`
-   `llama-index`
-   `llama-index-llms-gemini`
-   `llama-index-embeddings-siliconflow`
-   `llama-index-embeddings-huggingface`
-   `llama-index-postprocessor-siliconflow-rerank`
