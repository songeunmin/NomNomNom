import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("kfkas/Llama-2-ko-7b-Chat")
model = AutoModelForCausalLM.from_pretrained("kfkas/Llama-2-ko-7b-Chat")

# 챗봇 함수 정의
@st.cache(allow_output_mutation=True)
def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit 애플리케이션 시작
def main():
    st.title("✒️ AI 카피라이터")

    # 사용자 입력 받기
    with st.form('form'):
        product_name = st.text_input("제품명")
        max_len = st.number_input(
            label = '최대 단어 수',
            min_value = 5,
            max_value = 20,
            step = 1,
            value = 10
        )
    with col3:
        num = st.number_input(
            label = '생성할 문구 수',
            min_value = 1,
            max_value = 10,
            step = 1,
            value = 5
        )
    product_desc = st.text_input("제품 설명")
    st.text('키워드를 최대 3개 입력해주세요')
    col1, col2, col3 = st.columns(3)
    with col1:
        keyward_1 = st.text_input(
            label = 'keyward_1',
            label_visibility = 'collapsed',
            placeholder = '키워드 1' 
        )
    with col2:
        keyward_2 = st.text_input(
            label = 'keyward_2',
            label_visibility = 'collapsed',
            placeholder = '키워드 2' 
        )
    with col3:
        keyward_3 = st.text_input(
            label = 'keyward_3',
            label_visibility = 'collapsed',
            placeholder = '키워드 3' 
        )
    
    
    submit = st.form_submit_button('Submit')
if submit:
    if not product_name:
        st.error('제품명을 추가해주세요.')
    elif not product_desc:
        st.error('제품 설명을 추가해주세요.')
    else:
        st.success('생성 가능합니다.')
    
    # if user_input:
    #     # 챗봇 응답 생성
    #     response = generate_response(user_input)
        
    #     # 시스템 역할 추가
    #     st.text_area("사용자 입력:", value=user_input, height=100)
    #     st.text_area("챗봇 응답:", value=response, height=100)

if __name__ == "__main__":
    main()
