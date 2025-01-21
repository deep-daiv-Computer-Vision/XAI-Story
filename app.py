# import streamlit as st
# from front.layout import render_layout
# from front.sidebar import render_sidebar
# from front.main_page import render_main_page
# from front.right_sidebar import render_right_sidebar
# # from back.chat_storage import initialize_chat_storage, load_chat_history
# # from back.llm_service import initialize_model

# # Streamlit 페이지 설정
# st.set_page_config(page_title="Prompt Explainer", layout="wide", page_icon="🧑‍💼")

# # 대화 기록 파일 초기화
# # initialize_chat_storage()

# # 상태 변수 초기화
# # if "chat_history" not in st.session_state:
#     # st.session_state["chat_history"] = load_chat_history()
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []  # 현재 대화 기록
# if "current_page" not in st.session_state:
#     st.session_state["current_page"] = 1  # 기본 페이지 번호
# if "current_prompt" not in st.session_state:
#     st.session_state["current_prompt"] = None  # 기본 시스템 프롬프트 없음
# if "model" not in st.session_state:
#     # st.session_state["model"] = initialize_model("gpt2")  # 모델 초기화
# # if "greetings" not in st.session_state:
#     st.session_state["greetings"] = False  # 초기 상태는 False로 설정

# # 레이아웃 정의: 중앙(8), 오른쪽(3) 비율
# col2, col3 = render_layout()

# # 왼쪽 사이드바
# render_sidebar()

# # 중앙 메인 페이지
# with col2:
#     st.title("Prompt Explainer 🖍️")
#     render_main_page()

# # 오른쪽 사이드바
# with col3:
#     render_right_sidebar()
import streamlit as st
from front.layout import render_layout
from front.sidebar import render_sidebar
from front.main_page import render_main_page
# from front.right_sidebar import render_right_sidebar
from back.chat_storage import initialize_chat_storage, load_chat_history
from back.llm_service import initialize_model, compute_token_attributions
from transformers import AutoTokenizer


# Streamlit 페이지 설정
st.set_page_config(page_title="Prompt Explainer", layout="wide", page_icon="🧑‍💼")

# 대화 기록 파일 초기화
initialize_chat_storage()

# 상태 변수 초기화
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = load_chat_history()
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 현재 대화 기록
if "current_page" not in st.session_state:
    st.session_state["current_page"] = 1  # 기본 페이지 번호
if "current_prompt" not in st.session_state:
    st.session_state["current_prompt"] = None  # 기본 시스템 프롬프트 없음
if "model" not in st.session_state:
    st.session_state["model"] = initialize_model("Qwen/Qwen2.5-1.5B-Instruct")  # 모델 초기화
if "tokenizer" not in st.session_state:
    st.session_state["tokenizer"] = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")  # 토크나이저 초기화

if "greetings" not in st.session_state:
    st.session_state["greetings"] = False  # 초기 상태는 False로 설정

# 레이아웃 정의: 중앙(8), 오른쪽(3) 비율
col2, col3 = render_layout()

# 왼쪽 사이드바
render_sidebar()

# 중앙 메인 페이지
with col2:
    st.title("Prompt Explainer")
    render_main_page()

# 오른쪽 사이드바
# with col3:
#     render_right_sidebar()