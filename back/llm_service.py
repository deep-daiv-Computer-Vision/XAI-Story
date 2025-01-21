from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
    ProductBaselines,
)
import torch
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# def initialize_model(model_name="Qwen/Qwen2.5-1.5B-Instruct", device=0):
#     """Hugging Face 모델 초기화 (GPU 사용)"""
#     return pipeline("text-generation", model=model_name, device=device)

def initialize_model(model_name="Qwen/Qwen2.5-1.5B-Instruct", device=0):
    """
    Hugging Face 모델과 토크나이저를 개별적으로 초기화.
    Args:
        model_name (str): 모델 이름.
        device (int): 사용할 디바이스 (0 = GPU, -1 = CPU).
    Returns:
        model: Hugging Face 모델.
        tokenizer: Hugging Face 토크나이저.
    """
    # 모델과 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(f"cuda:{device}" if device >= 0 else "cpu")
    
    return model, tokenizer

def get_huggingface_response(model, prompt,tone="formal", max_length=200, temperature=0.7, top_k=50, top_p=0.9):
    """
    Hugging Face 모델 호출 및 파라미터 조정
    """
    # system_prompt = "Please paraphrase the following text with the given style:\n"
    # full_prompt = f"{system_prompt}{prompt}"
    system_prompt = f"Please paraphrase the following text with a {tone} tone:\n"
    full_prompt = f"{system_prompt}{prompt}"
    response = model(
        full_prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1
    )
    generated_text = response[0]["generated_text"]
    if ":" in generated_text:
        result = generated_text.split(":")[-1].strip()
    else:
        result = generated_text.strip()
    return result


def compute_token_attributions(prompt, tone, model, tokenizer, device):
    """
    Captum을 사용하여 입력 토큰이 출력 토큰에 미친 기여도를 계산합니다.
    Args:
        prompt (str): 사용자 입력 프롬프트.
        tone (str): 선택한 스타일(톤).
        model: Hugging Face 모델.
        tokenizer: 모델 토크나이저 (AutoTokenizer 사용).
        device: 실행할 디바이스 (GPU 또는 CPU).
    Returns:
        response (str): 모델 생성 응답.
        token_attr (tensor): 각 입력 토큰의 기여도.
    """
    # 시스템 프롬프트와 전체 프롬프트 생성
    system_prompt = f"Please paraphrase the following text with a {tone} tone:\n"
    full_prompt = f"{system_prompt}{prompt}"

    # 입력 텐서 생성
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    # 모델 응답 생성
    model.eval()  # 평가 모드로 설정
    with torch.no_grad():
        output_ids = model.generate(inputs["input_ids"], max_new_tokens=32)[0]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Captum을 사용한 기여도 계산
    fa = FeatureAblation(model)
    llm_attr = LLMAttribution(fa, tokenizer)

    # Captum의 TextTokenInput 객체 생성
    text_input = TextTokenInput(full_prompt, tokenizer)
    attr_res = llm_attr.attribute(text_input, target=response)

    # 결과 반환
    return response, attr_res.token_attr