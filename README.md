# ✒️ NLP project</br>감정분석 기반 AI 카피라이팅 서비스 </br> - AI copywriting service based on emotional analysis
![70f0f09a-1b95-4b7a-ad39-0aac97c533f4](https://github.com/songeunmin/NomNomNom/assets/144300743/3661d503-bb8f-41c7-9045-8eb36b90f130)
## 👥 Team
- Team name : ⚔️ 좋은 놈, 나쁜 놈, 이상한 놈 (놈놈놈)
- Team members : 송은민, 윤성진, 송영달
- * :clock1:시작일 : 2024.02.26(월)
  * ⏰목표일 : 2024.03.15(금)
## :books: skill
- **Programming** <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
- **Framework** <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"> <img src="https://img.shields.io/badge/openai-412991?style=for-the-badge&logo=openai&logoColor=white">
- **Tools** <img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"> <img src="https://img.shields.io/badge/pycharm-000000?style=for-the-badge&logo=pycharm&logoColor=white"> <img src="https://img.shields.io/badge/googlecolab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"> <img src="https://img.shields.io/badge/tableau-E97627?style=for-the-badge&logo=tableau&logoColor=white">
- **Git** <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=jupyter&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">

## 목차(INDEX)
&emsp;&ensp;Ⅰ. 주제 선정</br>&emsp;&ensp;Ⅱ. 목표 설정</br>&emsp;&ensp;Ⅲ. 프로젝트 순서도</br>&emsp;&ensp;Ⅳ. 모델 선정</br>&emsp;&ensp;Ⅴ. 데이터 전처리</br>&emsp;&ensp;Ⅵ. 모델링</br>&emsp;&ensp;Ⅶ. 키워드 추출</br>&emsp;&ensp;Ⅷ. 서비스 구현</br>&emsp;&ensp;Ⅸ. 프로젝트 결과</br>&emsp;&ensp;

## Ⅰ. 주제선정
  **1. 인공지능의 발전과 함께 마케팅 분야도 변화**</br>
       &nbsp;&nbsp;&nbsp; 1) 마케팅 분야에 인공지능을 활용함에 따라 효율성과 정확도가 크게 향상</br>
       &nbsp;&nbsp;&nbsp; 2) 고객 리뷰 분석을 통해 ‘긍정’ 및 ‘부정’ 리뷰를 자동 분류하여 빅데이터화</br>
       &nbsp;&nbsp;&nbsp; 3) AI 챗봇의 활용 - ‘KoGPT’ 등장</br>
       
  **2. 자료출처**</br>
       &nbsp;&nbsp;&nbsp; 1) 매일경제 [https://www.sedaily.com/NewsView/29RZKXMF51.htm](https://www.mk.co.kr/news/business/10745225)</br>
       &nbsp;&nbsp;&nbsp; 2) 경향신문 [https://thepoc.co.kr/58/?q=YToxOntzOjEyOiJrZXl3b3JkX3R5cGUiO3M6MzoiYWxsIjt9&bmode=view&idx=7008773&t=board.htm](https://m.khan.co.kr/economy/economy-general/article/202210131108001#c2b)https://m.khan.co.kr/economy/economy-general/article/202210131108001#c2b</br>
       &nbsp;&nbsp;&nbsp; 3) 한국경제 https://www.hankyung.com/article/2023050776871

## Ⅱ. 목표설정
**1. 리뷰 감정분석**</br>
       &nbsp;&nbsp;&nbsp; 1) KoBERT 모델을 Fine-tuning</br>
       &nbsp;&nbsp;&nbsp; 2) 감정분석 후 긍정, 중립, 부정으로 분류</br>
       &nbsp;&nbsp;&nbsp; 3) 긍정 리뷰 데이터만 사용</br>
       
**2. 핵심 키워드 추출**</br>
       &nbsp;&nbsp;&nbsp; 1) KeyBERT와 Kiwi 형태소분석기 사용해 핵심 키워드 추출 </br>
       &nbsp;&nbsp;&nbsp; 2) 긍정 리뷰 데이터에서 핵심 키워드 3개 추출</br>
       
**3. 카피라이팅 서비스**</br>
       &nbsp;&nbsp;&nbsp; 1) OpenAI API를 사용</br>
       &nbsp;&nbsp;&nbsp; 2) 긍정 리뷰 데이터 csv파일을 업로드하면 핵심 키워드 3개를 추출해주는 서비스 구현</br>
       &nbsp;&nbsp;&nbsp; 3) 추출한 핵심 키워드 3개와 광고할 제품의 이름, 설명을 조합해 카피라이팅을 해주는 서비스 구현</br>

## Ⅲ. 프로젝트 순서도
![project_process](https://github.com/songeunmin/NomNomNom/assets/144300743/4c207200-0b1c-4a3c-9b77-3ebd3d6dda95)

## Ⅳ. 모델 선정
