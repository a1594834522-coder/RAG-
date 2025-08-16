import streamlit as st
import os
import shutil
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage, QueryBundle, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
import time
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.siliconflow import SiliconFlowEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.postprocessor.siliconflow_rerank import SiliconFlowRerank

# 设置页面标题和图标
st.set_page_config(page_title="Gemini RAG 助手", page_icon="✨")
st.title("✨ Gemini RAG 智能助手")
st.caption("由 Google Gemini 2.5 Pro 驱动")

# --- 侧边栏管理 ---
with st.sidebar:
    st.header("知识库管理")
    if st.button("重新加载知识库"):
        with st.spinner("正在清空旧索引并准备重启..."):
            # 步骤1：先清空缓存和会话状态，释放文件锁
            st.cache_resource.clear()
            if "query_engine" in st.session_state:
                del st.session_state.query_engine
            
            # 步骤2：短暂等待，确保文件锁被释放
            time.sleep(0.5)

            # 步骤3：尝试删除storage文件夹，并增加错误处理
            try:
                if os.path.exists("./storage"):
                    shutil.rmtree("./storage")
                st.success("知识库已成功清空！应用将自动重启以加载新数据。")
            except Exception as e:
                st.error(f"清空知识库时出错: {e}")
                st.warning("提示：这通常因为文件仍被占用。请尝试完全关闭当前命令行窗口，然后重新运行 start.bat。")

        # 步骤4：强制应用重新运行
        st.rerun()

# --- API密钥和模型配置 ---
# 加载.env文件中的环境变量
load_dotenv()

# 从环境变量中获取API密钥
google_api_key = os.getenv("GOOGLE_API_KEY")
siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")

if not google_api_key or not siliconflow_api_key:
    st.error("请在 .env 文件中设置 GOOGLE_API_KEY 和 SILICONFLOW_API_KEY！")
    st.stop()

# --- LlamaIndex 全局设置 ---
Settings.llm = Gemini(model_name="gemini-2.5-pro", api_key=google_api_key)

# 更改2: 配置嵌入模型为 SiliconFlow API
Settings.embed_model = SiliconFlowEmbedding(
    model_name="BAAI/bge-large-zh-v1.5",
    api_key=siliconflow_api_key
)

# --- 知识库索引加载 ---
@st.cache_resource(show_spinner="正在加载您的知识库...")
def load_data():
    """如果索引存在则从磁盘加载，否则以安全模式创建新索引。"""
    storage_dir = "./storage"
    data_dir = "./data"

    # 确保数据目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        with open(os.path.join(data_dir, "example.txt"), "w", encoding="utf-8") as f:
            f.write("这是示例文档。Gemini 1.5 Pro 是一个强大的多模态模型，拥有百万级上下文窗口。")

    # 检查索引是否已存在
    if os.path.exists(storage_dir):
        print("正在从磁盘加载现有索引...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        print("索引加载完成。")
        return index
    else:
        print("未找到现有索引，正在创建新索引...")
        reader = SimpleDirectoryReader(data_dir)
        documents = reader.load_data()
        
        # 优化1: 增大切分块大小以提升处理效率
        node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=50)
        nodes = node_parser.get_nodes_from_documents(documents)

        # 优化2: 使用并行处理构建索引，移除手动分批和sleep
        print(f"正在为 {len(nodes)} 个文本块创建嵌入，这将需要一些时间...")
        index = VectorStoreIndex(nodes, show_progress=True)
        
        print("正在保存新索引...")
        index.storage_context.persist(persist_dir=storage_dir)
        print("新索引已创建并保存。")
        return index

index = load_data()

# --- 聊天引擎 ---
if "query_engine" not in st.session_state:
    print("正在初始化高级查询引擎（包含Reranker）...")
    # 1. 创建一个检索器，并设置它从知识库中初步检索10个最相关的文档
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    # 2. 创建 SiliconFlow Reranker 实例，并设置它将初步检索的文档精选至3个
    reranker = SiliconFlowRerank(
        model="BAAI/bge-reranker-v2-m3", # 这是一个效果很好的默认重排模型
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        top_n=3,
    )

    # 3. 定义一个更高级的问答提示词模板，引导模型进行分析而非复述
    qa_template_str = (
        "我们有以下背景知识：\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "你是一位专业的分析师。请你基于以上背景知识，对用户的问题进行深入、全面且有条理的分析和解答。\n"
        "请不要简单地复述原文。你需要做到以下几点：\n"
        "1. 综合并提炼背景知识中的关键信息。\n"
        "2. 如果背景知识中包含多个相关片段，请将它们有机地联系起来。\n"
        "3. 给出结构清晰、逻辑严谨的回答。\n"
        "4. 解释你得出结论的推理过程。\n"
        "现在，请根据以上要求，回答这个问题： {query_str}\n" 
    )
    qa_template = PromptTemplate(qa_template_str)

    # 4. 使用检索器、Reranker和自定义的提示词模板构建新的查询引擎
    st.session_state.query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        streaming=True,
        text_qa_template=qa_template,
    )
    print("查询引擎初始化完成。")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好！我是您的知识库助手，请问有什么可以帮您的？"}]

# --- 聊天界面 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("向我提问..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        query_bundle = QueryBundle(prompt)
        # 1. 先从查询引擎中检索相关节点
        nodes = st.session_state.query_engine.retrieve(query_bundle)
        
        # 2. 设置一个相关度阈值
        score_threshold = 0.2 

        # 3. 检查最高分的节点的得分是否低于阈值
        if not nodes or nodes[0].score < score_threshold:
            # 如果是，则说明本地知识库没有高相关度的内容，切换到通用知识模式
            st.info("本地知识库中未找到相关信息，正在使用模型的通用知识为您回答...")
            # 直接调用LLM进行流式回答，并确保正确处理流数据
            response_stream = Settings.llm.stream_complete(prompt)
            
            # 定义一个生成器函数来提取流中的文本块
            def llm_response_generator(stream):
                for chunk in stream:
                    yield chunk.delta
            
            response = st.write_stream(llm_response_generator(response_stream))
        else:
            # 如果找到了高相关度的内容，则使用原有的RAG流程（仅用检索到的节点）来合成答案
            st.info("在本地知识库中找到相关信息，正在为您生成答案...")
            response_stream = st.session_state.query_engine.synthesize(query_bundle, nodes)
            response = st.write_stream(response_stream.response_gen)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
