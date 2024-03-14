import streamlit as st
from utils import make_copywriter,print_streaming_copywriter,keyword_extraction_with_noun_extraction,data_loader
from streamlit_option_menu import option_menu
import os
import openai
import pandas as pd


st.set_page_config(
    page_title="제품 홍보를 위한 광고 카피라이터 서비스",
    page_icon="🏙️"
)

with st.sidebar:
    st.markdown(
        "## How to use ❔ \n"
        "1. [OpenAI API key](https://platform.openai.com/account/api-keys) 를 입력하세요 🔑\n"  
        "2. 파일을 넣어 키워드를 추출하세요 📄\n"
        "3. 제품의 이름과 정보, 키워드를 넣고 문구를 생성하세요 💬\n"
    )
    # OpenAI API 키 입력 받기
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your OpenAI API key here. (sk-...)",
        help="API 키는 다음에서 얻을 수 있습니다. https://platform.openai.com/account/api-keys.",  # 사용자가 API 키를 얻을 수 있는 위치에 대한 도움말
        value=os.environ.get("OPENAI_API_KEY", None)
              or st.session_state.get("OPENAI_API_KEY", "") # 기본적으로 이전에 입력한 API 키 또는 환경 변수에서 가져온 API 키를 사용
    )
    # 입력한 OpenAI API 키를 Streamlit의 세션 상태에 저장
    st.session_state["OPENAI_API_KEY"] = api_key_input
    # 세션 상태에서 OpenAI API 키를 가져와서 openai_api_key 변수에 할당
    openai_api_key = st.session_state.get("OPENAI_API_KEY")

    # OpenAI API 키 설정
    openai.api_key = openai_api_key

    if openai_api_key:
        st.success("API키가 입력되었습니다!")

    if not openai_api_key:
        st.warning(
            "당신의 OpenAI API키를 입력하세요."
            " https://platform.openai.com/account/api-keys."
        )
        st.stop() # 키 입력하지 않으면 사용금지


with st.sidebar:
    st.sidebar.image('https://img.freepik.com/premium-photo/cute-robot-cafe-working-laptop-taking-notes-with-pen_124507-161688.jpg')
    side_menu = option_menu("AI 카피라이터", ["Intro", "Keyword_extractor", "Copywriter" ],
                         icons=['bi bi-chat-right-text', 'search', 'pencil-square'],
                         menu_icon="bi bi-robot", default_index=0,
                         styles={
        "container": {"padding": "4!important", "background-color": "dark"},
        "icon": {"color": "white", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
        "nav-link-selected": {"background-color": "#08c7b4"},
        }
    )

# 사이드바 선택 시 출력 설정
if side_menu == "Intro":
    st.title('😃 환영합니다!')
    st.markdown("<br>", unsafe_allow_html=True)  # 공백 추가
    st.subheader('_AI 카피라이터 서비스 앱입니다._')

    st.divider()

    # 사용 사례
    with st.container():
        col1,col2=st.columns(2)
        with col1:
            st.header('👋 Greeting')
            st.markdown("<br>", unsafe_allow_html=True) # 공백 추가
            st.markdown(
                """
                - _멋진 광고 문구를 만들고 싶나요?_
                - _제품의 새로운 광고 문구가 필요하신가요?_
                - _제품을 돋보이게 할 문구가 필요하신가요?_
                - _그저 그냥 놀러오셨나요?_ 
                - _당신을 환영합니다!_  
                """
                )
        with col2:
            st.image('https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjVtd2t4dzJiNDh6b3VqbXV6M2hsaDhuOHdvZmZiZmZ3ZHdodGo5NSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/hZj44bR9FVI3K/giphy.gif')

    st.divider()

    with st.container():
        st.header('⭐ App Guide')
        st.markdown("<br>", unsafe_allow_html=True) # 공백 추가
        st.markdown(
            '''
            _STEP1. 파일을 업로드하여 키워드를 추출하세요._
            
            _STEP2. 제품 이름을 입력하세요._
            
            _STEP3. 생성할 문구의 최대 단어 수를 설정하세요.( 5 ~ 20 )_
            
            _STEP4. 생성할 최대 문구 수를 설정하세요.( 1 ~ 10 )_
            
            _STEP5. 제품에 대한 설명을 입력하세요._
            
            _STEP6. 추출한 키워드를 입력하세요._
            
            _STEP7. 생성 버튼을 누르고 결과를 확인하세요!_
            '''
        )

elif side_menu == "Keyword_extractor":

    st.header(" 🔍 키워드 추출")

    st.divider()

    original_title = '<h2 style="font-family:Sans serif; color:green; font-size: 22px;">     카피라이터 페이지에 넣을 키워드를 추출해드릴게요!</h2>'
    st.markdown(original_title, unsafe_allow_html=True)


    uploaded_file = st.file_uploader("*추출할 파일을 업로드하세요!*")

    if uploaded_file is not None:
        # 로딩 바 표시
        with st.spinner("키워드를 추출하고 있습니다!"):
            # UploadedFile 객체를 DataFrame 으로 읽어 오기
            df = pd.read_csv(uploaded_file)
            # 파일을 읽어 오고 키워드 추출 함수 호출
            top_five = keyword_extraction_with_noun_extraction(df)

        # 로딩 바 해제 후 결과 표시
        st.success(f"✅ 키워드 추출이 완료 되었습니다.")
        st.write(top_five.keyword)

elif side_menu == "Copywriter":
    st.title("✒️AI 카피라이터")

    st.divider()

    st.subheader('_키워드를 입력하고 멋진 광고 문구를 만들어보세요!_')

    st.markdown("<br>", unsafe_allow_html=True)  # 공백 추가

    auto_complete = st.toggle('자동 채우기')

    example = {
        'product_name': '너구리',
        'product_desc': '깊고 개운한 국물 맛과 오동통하고 쫄깃한 면발이 특징인 농심 브랜드의 라면',
        'keywards': ['다시마', '면발', '국물']
    }

    # 템플릿 설정
    prompt_template = '''
    제품 혹은 브랜드를 SNS에 광고하기 위한 문구를 {num}개 생성해줘.
    자극적이고 창의적으로 작성해줘.
    명사 위주로 간결하고 임팩트있게 작성해줘.
    반드시 {max_len} 단어 이내로 작성해줘.
    키워드가 주어질 경우 반드시 키워드 중 하나를 포함해줘.
    이모지를 적절하게 사용해줘.

    ---
    제품명 : {product_name}
    제품설명 : {product_desc}
    키워드 : {keywards}
    ---
    '''.strip()

    # form 생성
    with st.form('form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            product_name = st.text_input(
                "제품명",
                placeholder=example['product_name'],
                value=example['product_name'] if auto_complete else ''
            )
        with col2:
            max_len = st.number_input(
                label='최대 단어 수',
                min_value=5,
                max_value=20,
                step=1,
                value=10
            )
        with col3:
            num = st.number_input(
                label='생성할 문구 수',
                min_value=1,
                max_value=10,
                step=1,
                value=3
            )
        product_desc = st.text_input(
            "제품 설명",
            placeholder=example['product_desc'],
            value=example['product_desc'] if auto_complete else ''
        )
        st.text('키워드를 최대 3개 입력해주세요')
        col1, col2, col3 = st.columns(3)
        with col1:
            keyward_1 = st.text_input(
                label='keyward_1',
                label_visibility='collapsed',
                placeholder='키워드 1',
                value=example['keywards'][0] if auto_complete else ''
            )
        with col2:
            keyward_2 = st.text_input(
                label='keyward_2',
                label_visibility='collapsed',
                placeholder='키워드 2',
                value=example['keywards'][1] if auto_complete else ''
            )
        with col3:
            keyward_3 = st.text_input(
                label='keyward_3',
                label_visibility='collapsed',
                placeholder='키워드 3',
                value=example['keywards'][2] if auto_complete else ''
            )

        submit = st.form_submit_button('생성')

    if submit:
        if not product_name:
            st.error('❌ 제품명을 추가해주세요.')
        elif not product_desc:
            st.error('❌ 제품 설명을 추가해주세요.')
        else:
            keywards = [keyward_1, keyward_2, keyward_3]
            keywards = [x for x in keywards if x]
            prompt = prompt_template.format(
                product_name=product_name,
                product_desc=product_desc,
                max_len=max_len,
                num=num,
                keywards=keywards
            )
            system_role = '당신은 카피라이터 전문가입니다.'
            with st.spinner('문구를 생성하고 있습니다!'):
                copywriter = make_copywriter(
                    prompt=prompt,
                    system_role=system_role,
                    stream=True,
                )
                st.success('✅ 생성 가능합니다.')

            st.divider()

            print_streaming_copywriter(copywriter)


