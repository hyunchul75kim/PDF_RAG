# Ollama_PDF_RAG.py - 개선된 다중 PDF 처리 버전
import gradio as gr
import ollama
import os
import hashlib
import yaml
from pathlib import Path
from typing import Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# 설정 파일 로드
def load_config():
    """설정 파일을 로드합니다."""
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # 기본 설정 반환
        return {
            "models": {"llm": "llama3", "embedding": "mxbai-embed-large"},
            "text_splitter": {"chunk_size": 1000, "chunk_overlap": 200},
            "retrieval": {"top_k": 4, "score_threshold": 0.0},
            "paths": {"pdf_directory": "pdfs", "vector_stores_directory": "vector_stores"},
            "system_prompt": "You are a helpful assistant. Read the PDF content and answer the question. Translate the answer in Korean with emoji."
        }

CONFIG = load_config()
PDF_DIR = CONFIG["paths"]["pdf_directory"]
VECTOR_STORES_DIR = CONFIG["paths"]["vector_stores_directory"]

# 디렉토리 생성
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(VECTOR_STORES_DIR, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """파일의 해시값을 계산합니다."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_vector_store_path(file_path: str) -> str:
    """PDF 파일에 대한 벡터 저장소 경로를 반환합니다."""
    file_hash = get_file_hash(file_path)
    return os.path.join(VECTOR_STORES_DIR, file_hash)

def load_pdf_list() -> List[str]:
    """pdfs 폴더에서 PDF 파일 목록을 가져옵니다."""
    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.exists():
        return []
    
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    return [str(f) for f in sorted(pdf_files)]

def load_and_retrieve_pdf(file_path: str, force_reload: bool = False):
    """
    PDF 문서를 로드하고 벡터화합니다.
    캐시가 있으면 재사용하고, 없으면 새로 생성합니다.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    
    vector_store_path = get_vector_store_path(file_path)
    embeddings = OllamaEmbeddings(model=CONFIG["models"]["embedding"])
    
    # 기존 벡터 저장소가 있고 강제 재로드가 아닌 경우
    if os.path.exists(vector_store_path) and not force_reload:
        print(f"📂 기존 벡터 저장소를 로드합니다: {os.path.basename(file_path)}")
        try:
            vectorstore = Chroma(
                persist_directory=vector_store_path,
                embedding_function=embeddings
            )
            return vectorstore.as_retriever(
                search_kwargs={"k": CONFIG["retrieval"]["top_k"]}
            )
        except Exception as e:
            print(f"⚠️ 기존 저장소 로드 실패, 새로 생성합니다: {e}")
            force_reload = True
    
    # 새로 벡터화
    print(f"🔄 PDF를 벡터화합니다: {os.path.basename(file_path)}")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    if not docs:
        raise ValueError("❗ PDF에서 텍스트를 추출할 수 없습니다. 다른 파일을 시도해 보세요.")
    
    print(f"✅ PDF 문서 로드 완료 ({len(docs)} 페이지). 첫 페이지 미리보기:\n{docs[0].page_content[:300]}...\n")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["text_splitter"]["chunk_size"],
        chunk_overlap=CONFIG["text_splitter"]["chunk_overlap"]
    )
    splits = text_splitter.split_documents(docs)
    print(f"📄 문서를 {len(splits)}개의 청크로 분할했습니다.")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=vector_store_path
    )
    vectorstore.persist()
    print(f"💾 벡터 저장소를 저장했습니다: {vector_store_path}")
    
    return vectorstore.as_retriever(
        search_kwargs={"k": CONFIG["retrieval"]["top_k"]}
    )

def format_docs(docs):
    """문서들을 포맷팅합니다."""
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(selected_pdf: str, question: str, top_k: int, force_reload: bool) -> str:
    """
    RAG 체인을 실행합니다.
    
    Args:
        selected_pdf: 선택된 PDF 파일 경로
        question: 질문
        top_k: 검색할 문서 수
        force_reload: 벡터 저장소 강제 재생성 여부
    """
    if not question or not question.strip():
        return "❌ 질문을 입력해주세요."
    
    if not selected_pdf or selected_pdf == "None":
        return "❌ PDF 파일을 선택해주세요."
    
    try:
        # top_k 설정 업데이트
        if top_k > 0:
            CONFIG["retrieval"]["top_k"] = top_k
        
        retriever = load_and_retrieve_pdf(selected_pdf, force_reload)
        retrieved_docs = retriever.invoke(question)
        
        if not retrieved_docs:
            return "❌ 관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 작성해 보거나 다른 PDF를 사용해 보세요."
        
        context = format_docs(retrieved_docs)
        print(f"🔍 검색된 문맥 미리보기:\n{context[:500]}...\n")
        
        prompt = f"Question: {question}\n\nContext: {context}"
        
        print(f"🤖 LLM에 질문을 전송합니다...")
        response = ollama.chat(
            model=CONFIG["models"]["llm"],
            messages=[
                {
                    "role": "system",
                    "content": CONFIG["system_prompt"]
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return response['message']['content']
    
    except FileNotFoundError as e:
        return f"❌ 파일 오류: {str(e)}"
    except ValueError as e:
        return f"❌ 값 오류: {str(e)}"
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}\n\n오류 타입: {type(e).__name__}"

def update_pdf_list():
    """PDF 목록을 업데이트합니다."""
    pdf_files = load_pdf_list()
    if not pdf_files:
        return gr.Dropdown(choices=["PDF 파일이 없습니다"], value=None)
    return gr.Dropdown(choices=pdf_files, value=pdf_files[0] if pdf_files else None)

# Gradio 인터페이스
with gr.Blocks(title="PDF RAG - 다중 PDF 질문 응답 시스템") as iface:
    gr.Markdown("# 📚 PDF RAG - 다중 PDF 질문 응답 시스템")
    gr.Markdown("여러 PDF 파일을 관리하고 질문할 수 있습니다. 각 PDF는 개별 벡터 저장소에 저장됩니다.")
    
    with gr.Row():
        with gr.Column(scale=2):
            pdf_dropdown = gr.Dropdown(
                label="📄 PDF 파일 선택",
                choices=load_pdf_list(),
                value=load_pdf_list()[0] if load_pdf_list() else None,
                interactive=True
            )
            refresh_btn = gr.Button("🔄 PDF 목록 새로고침", variant="secondary")
        
        with gr.Column(scale=1):
            force_reload_checkbox = gr.Checkbox(
                label="🔄 벡터 저장소 강제 재생성",
                value=False,
                info="체크하면 캐시를 무시하고 새로 벡터화합니다"
            )
    
    with gr.Row():
        question_input = gr.Textbox(
            label="❓ 질문을 입력하세요",
            placeholder="예: 이 문서의 주요 내용은 무엇인가요?",
            lines=3
        )
    
    with gr.Row():
        top_k_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=CONFIG["retrieval"]["top_k"],
            step=1,
            label="🔍 검색할 문서 수 (top_k)",
            info="더 많은 문서를 검색하면 더 풍부한 답변을 얻을 수 있지만 느려질 수 있습니다"
        )
    
    submit_btn = gr.Button("🚀 질문하기", variant="primary", size="lg")
    
    output = gr.Textbox(
        label="💬 답변",
        lines=10,
        interactive=False
    )
    
    # 이벤트 핸들러
    refresh_btn.click(
        fn=update_pdf_list,
        outputs=pdf_dropdown
    )
    
    submit_btn.click(
        fn=rag_chain,
        inputs=[pdf_dropdown, question_input, top_k_slider, force_reload_checkbox],
        outputs=output
    )
    
    question_input.submit(
        fn=rag_chain,
        inputs=[pdf_dropdown, question_input, top_k_slider, force_reload_checkbox],
        outputs=output
    )
    
    gr.Markdown("---")
    gr.Markdown("### ℹ️ 사용 방법")
    gr.Markdown("""
    1. **PDF 선택**: 드롭다운에서 질문할 PDF를 선택합니다
    2. **질문 입력**: 텍스트 박스에 질문을 입력합니다
    3. **검색 설정**: top_k 슬라이더로 검색할 문서 수를 조정합니다
    4. **질문하기**: 버튼을 클릭하거나 Enter를 누릅니다
    
    **팁**: 
    - 각 PDF는 처음 로드 시 벡터화되어 캐시됩니다
    - 같은 PDF를 다시 사용하면 캐시된 벡터 저장소를 재사용합니다
    - PDF 목록 새로고침 버튼으로 새로 추가된 PDF를 불러올 수 있습니다
    """)

if __name__ == "__main__":
    print("🚀 PDF RAG 시스템을 시작합니다...")
    print(f"📁 PDF 디렉토리: {PDF_DIR}")
    print(f"💾 벡터 저장소 디렉토리: {VECTOR_STORES_DIR}")
    print(f"🤖 LLM 모델: {CONFIG['models']['llm']}")
    print(f"🔤 임베딩 모델: {CONFIG['models']['embedding']}")
    iface.launch(share=False)
