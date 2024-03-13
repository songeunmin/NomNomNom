import openai
import streamlit as st
import pandas as pd
from keybert import KeyBERT
from transformers import BertModel
from kiwipiepy import Kiwi

# openai.api_key = st.secrets['OPENAI_API_KEY']

def make_copywriter(
    prompt,
    system_role="당신은 세상에서 가장 유용한 도우미입니다.",
    model='gpt-3.5-turbo',
    stream=False
):
    messages = [
        {'role': 'system', 'content': system_role},
        {'role': 'user', 'content': prompt}
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=stream
    )
    return response

def print_streaming_copywriter(copywriter):
    message = ''
    placeholder = st.empty()
    for i in copywriter:
        delta = i.choices[0]['delta']
        if 'content' in delta:
            message += delta['content']
            placeholder.markdown(message + '▌')
            print(delta['content'], end="")
        else:
            break
    placeholder.markdown(message)
    return message
@st.cache_data
def keyword_extraction_with_noun_extraction(df):
    # KeyBERT 객체 생성
    model = BertModel.from_pretrained('skt/kobert-base-v1')
    kw_extraction = KeyBERT(model)

    # content 열을 리스트로 변환
    # list_ = ['content','review','RawText']
    text_list = df['content'].tolist()

    # 키워드 추출 및 가중치 계산
    keywords_weight = {}
    for text in text_list:
        keywords = kw_extraction.extract_keywords(text)
        for keyword, weight in keywords:
            keywords_weight[keyword] = keywords_weight.get(keyword, 0) + weight

    # 결과 정렬 및 상위 20개 선택
    top_keywords = sorted(keywords_weight.items(), key=lambda x: x[1], reverse=True)[:20]
    # DataFrame으로 변경
    result = pd.DataFrame(top_keywords, columns=['keyword', 'weight'])

    # 키워드 추출 결과를 txt 파일로 저장
    result['keyword'].to_csv(f'keyword.txt', sep='\t', index=False, header=False)

    # txt 파일에서 키워드 추출을 위한 텍스트 읽기
    with open(f'keyword.txt', 'r', encoding='UTF-8') as file:
        texts = file.read()

    # 형태소 분석기 초기화
    kiwi = Kiwi()

    # 명사 추출 함수 정의
    def noun_extractor(text):
        results = []
        result = kiwi.analyze(text)
        for token, pos, _, _ in result[0][0]:
            # tag가 N이나 SL로 시작하는 명사만 남김
            if len(token) != 1 and pos.startswith('N') or pos.startswith('SL'):
                results.append(token)
        return results

    # 명사만 추출
    nouns = noun_extractor(texts)
    # 추출한 명사를 공백으로 연결하여 텍스트로 변환
    text = ' '.join(nouns)

    # # KeyBERT 모델 재설정
    # keyword_model = KeyBERT(model)
    # 명사만으로 추출한 키워드 재추출
    keywords = kw_extraction.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=3)
    # 키워드 추출 결과를 DataFrame으로 저장
    df_keywords = pd.DataFrame(keywords, columns=['keyword', 'weight'])

    return df_keywords


def data_loader(file_path):
    df = pd.read_csv(file_path)