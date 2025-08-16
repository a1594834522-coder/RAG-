# Gemini RAG Application

This project is a Retrieval-Augmented Generation (RAG) application that uses the Google Gemini API and Streamlit to create a simple and interactive question-answering system based on a local knowledge base.

## Features

-   **Interactive UI:** Built with Streamlit to provide a user-friendly web interface.
-   **Powerful LLM:** Leverages the Google Gemini Pro model for text generation.
-   **Local Knowledge Base:** Uses text files in the `data` directory as the source for answers.
-   **Extensible:** Easily add your own documents to the knowledge base.
-   **RAG Pipeline:** Implements a RAG pipeline using LlamaIndex for efficient and relevant information retrieval.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd RAG-GEMINI
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the root directory and add your Google API key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

## Usage

1.  **Add your data:**
    Place your `.txt` files into the `data` directory. The application will automatically use them as the knowledge base.

2.  **Run the application:**
    Execute the `start.bat` script or run the following command in your terminal:
    ```bash
    streamlit run gemini_rag_app.py
    ```

3.  **Open the application:**
    Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Dependencies

The main dependencies for this project are:

-   `streamlit`
-   `python-dotenv`
--   `google-generativeai`
-   `llama-index`
-   `llama-index-llms-gemini`
-   `llama-index-embeddings-siliconflow`
-   `llama-index-embeddings-huggingface`
-   `llama-index-postprocessor-siliconflow-rerank`

