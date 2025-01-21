import streamlit as st
from back.llm_service import get_huggingface_response,initialize_model, compute_token_attributions
from back.chat_storage import save_chat_history
# from back.explainability import compute_lime_values
# from front.visualization import display_lime_visualization

# 알고리즘별 시스템 프롬프트
ALGORITHM_PROMPTS = {
    "Sort Algorithm": """
            You are an algorithm implementation expert.
            The user wants to implement a sorting algorithm.
            Follow these rules:

        """
}

def render_main_page():
    """중앙 메인 페이지 구현"""
    st.header(f"Conversation {st.session_state['current_page']}")

    # 안내문구 표시 (처음 한 번만)
    if not st.session_state.greetings:
        with st.chat_message("assistant"):
            intro = """
            Welcome to **Prompt Explainer**! 🤵🏻‍♀️\n
            This tool is designed to help you leverage LLMs (Large Language Models) more effectively when **solving algorithm problems**. ⛳️\n
            By visually highlighting which **parts of the prompt the LLM focuses on**, you can craft **better prompts** and receive **higher-quality response codes**. 🎲\n
            When you input a prompt, we will visualize the emphasized sections based on **SHAP values**. This allows you to learn better **prompt-writing strategies** and **maximize the utility of LLMs** in your workflow. 🎞️\n 
            Give it a try and enhance your experience in solving algorithmic problems! 🎸
            """
            st.markdown(intro)
            st.session_state.messages.append({"role": "assistant", "content": intro})  # 대화 기록에 추가
        st.session_state.greetings = True  # 상태 업데이트
        st.rerun()
    
    # 상태 관리: 버튼이 눌리지 않았을 때
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = None
        st.session_state.system_prompt = None
    # 대화 메시지 출력
    # st.subheader("Conversation")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 버튼이 눌리지 않았을 때 알고리즘 버튼 출력
    if st.session_state.button_pressed is None:
        # st.subheader("Choose an Algorithm")
        cols = st.columns(3)  # 3열 레이아웃
        for idx, (algo, prompt) in enumerate(ALGORITHM_PROMPTS.items()):
            with cols[idx % 3]:
                if st.button(algo):
                    st.session_state.button_pressed = algo  # 선택된 버튼을 상태로 저장
                    st.session_state.system_prompt = prompt  # 해당 시스템 프롬프트 저장
                    st.rerun()  # 페이지를 리프레시하여 새로운 버튼 상태 반영

    # 버튼이 눌린 후 선택된 알고리즘의 프롬프트 표시
    else:
        st.sidebar.title("🛠️ System Prompt")
        st.sidebar.info(st.session_state.system_prompt)  # 사이드바에 시스템 프롬프트 표시

        # 선택된 알고리즘의 상태에서 "다시 선택" 버튼을 만들어 상태 초기화
        col = st.columns(1)[0]  # 중앙 정렬을 위해 1열 사용
        with col:
            if st.button(st.session_state.button_pressed, key="selected_button"):
                st.session_state.button_pressed = None  # 상태 초기화
                st.session_state.system_prompt = None

        # "다시 선택" 버튼을 눌러 선택을 초기화하는 기능
        if st.button("back"):
            st.session_state.button_pressed = None  # 상태 초기화
            st.session_state.system_prompt = None
            st.rerun()  # 리프레시하여 다시 처음 상태로 돌아가기
    # 스타일 선택 라디오 버튼 추가
    tone_options = ["formal", "casual", "empathetic", "objective"]
    selected_tone = st.radio("Choose a tone for your response:", tone_options, index=0)

    # 프롬프트 입력 창 (st.chat_input() 사용)
    user_input = st.chat_input("Enter your prompt!")
    if user_input:  # 사용자가 입력을 하면
        if user_input.strip():
            # LLM 응답 생성
            # response = get_huggingface_response(st.session_state["model"], user_input)
            # LLM 응답 생성 및 기여도 계산
            # response, token_attributions = compute_token_attributions(
            #     prompt=user_input,
            #     tone="formal",
            #     model=st.session_state["model"],
            #     tokenizer=st.session_state["tokenizer"],
            #     device="cuda"  # GPU 사용
            # )
            response, token_attributions = compute_token_attributions(
                prompt=user_input,
                tone=selected_tone,  # 선택한 스타일 전달
                model=st.session_state["model"][0],  # model 객체 추출
                tokenizer=st.session_state["model"][1],  # tokenizer 객체 추출
                device="cuda"  # GPU 사용
            )

    
            # prompt 기여도 계산
            # token_html = ""
            # for token, shap_value in zip(tokens, shap_values):
            #     intensity = min(max(shap_value, 0), 1)  # SHAP 값을 0~1 범위로 정규화
            #     color = f"rgba(255, 0, 0, {intensity})"  # 빨간색 계열로 SHAP 값 표시
            #     token_html += f'<span style="background-color: {color}; padding: 2px; margin: 1px; border-radius: 4px;">{token}</span> '

            # 사용자 입력 시각화
            # st.markdown(f"**User Input:**")
            # st.markdown(token_html, unsafe_allow_html=True)

            # 모델 응답 출력
            # st.markdown(f"**LLM Response:**")
            # st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 8px;'>{response}</div>", unsafe_allow_html=True)

            # 메시지 기록 추가
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

            # 기여도 시각화
            # 기여도 시각화
            st.markdown("### Input Token Attribution")
            for idx, (token, attr) in enumerate(zip(user_input.split(), token_attributions)):
                color = f"rgba(255, 0, 0, {abs(attr)})"  #  값에 따라 빨간색 음영 조정
                st.markdown(
                    f"<span style='background-color: {color}; padding: 2px; margin: 2px; border-radius: 4px;'>{token}</span>",
                    unsafe_allow_html=True,
                )
            # 대화 기록 저장
            chat_history = st.session_state["chat_history"]
            current_page = st.session_state["current_page"] - 1

            # 기존 페이지 업데이트
            if current_page < len(chat_history):
                chat_history[current_page] = {"messages": st.session_state["messages"]}
            else:
                chat_history.append({"messages": st.session_state["messages"]})

            save_chat_history(chat_history)

            # UI 업데이트
            st.rerun()
    