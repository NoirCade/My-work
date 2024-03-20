import streamlit as st
import stylecloud
# coding=utf8
# REST API 호출에 필요한 라이브러리
import requests
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from konlpy.tag import Okt
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import os
import openai




def run_home() :
    st.subheader("info for your stock")
    stylish = st.checkbox('you wanna stylish wordcloud? then check me!')

    number_of_news = st.slider('number of news to you want know', min_value=1, max_value=10)
    name = st.text_input('Search your stock!')


    if st.button("Search!"):
        st.write(kogpt(name,number_of_news))
          # 예제 Textfile

        if stylish & os.path.isfile(f'./stylecloud/{name}_WordCloud_{number_of_news + 1}.png'):
            st.image(f'./stylecloud/{name}_WordCloud_{number_of_news + 1}.png')
        elif os.path.isfile(f'./wordcloud/{name}_{number_of_news+1}.png'):
            st.image(f'./wordcloud/{name}_{number_of_news+1}.png')
        else:
            st.write('그런거 없습니다.') #예외처리
 #예외처리
    #st.write('text')  # df, err, func, keras


def kogpt(name,number_of_news):
    df_stock = pd.read_csv('./stock_list.csv', encoding='cp949')
    df_stock = df_stock.rename(columns={'회사명': 'name', '종목코드': 'code'})

    def six_digit(x):
        return '%06d' % x

    df_stock['code'] = df_stock['code'].apply(six_digit)

    #name = input('종목명 : ')

    item_list = df_stock['name'].values.tolist()


    if name in item_list:
        pass
    else:
        newsheadlines = '그런거 없습니다. 다시 확인해보세요' #예외처리
        return newsheadlines

    code = df_stock.loc[df_stock['name'] == name, 'code'].iloc[0]


    def kogpt_api(prompt, max_tokens=1, temperature=1.0, top_p=1.0, n=1):
        r = requests.post(
            'https://api.kakaobrain.com/v1/inference/kogpt/generation',
            json={
                'prompt': prompt,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'n': n
            },
            headers={
                'Authorization': 'KakaoAK ' + REST_API_KEY,
                'Content-Type': 'application/json'
            }
        )
        # 응답 JSON 형식으로 변환
        response = json.loads(r.content)
        return response


    # Start a webdriver (e.g. Chrome)
    newslist = []
    for i in range(1,number_of_news+1):
        driver = webdriver.Chrome()
        options = webdriver.ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        juso = f"https://finance.daum.net/quotes/A{code}#news/stock"

        # Navigate to a website
        driver.get(juso)


        # Find an element using XPath
        # abc = driver.find_element_by_xpath("/html/body/div/table[1]")

        def doScrollDown(whileSeconds):
            start = datetime.datetime.now()
            end = start + datetime.timedelta(seconds=whileSeconds)
            while True:
                driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
                time.sleep(1)
                if datetime.datetime.now() > end:
                    break


        scroll = doScrollDown(0.5)
        scroll

        element = driver.find_element(By.XPATH, f"//*[@id='boxContents']/div[5]/div[1]/div[2]/div/ul/li[{i}]/span/a[1]")

        time.sleep(0.3)
        # print('기사제목= ',element.text)

        element.click()
        # element = driver.find_element(By.XPATH,"/html/body/div/table[1]/tbody/tr[1]/td[1]/a").click
        # rows = element.find_elements(By.XPATH,".//tr")
        # Interact with the element (e.g. get text, click, etc.)

        time.sleep(0.3)
        texts = driver.find_element(By.XPATH, "//*[@id='boxApp']/div[3]/div[1]")
        reallytexts = texts.find_element(By.XPATH, "//*[@id='dmcfContents']/section")
        # print(reallytexts.text)

        news = reallytexts.text
        time.sleep(0.2)  # 5초 대기
        driver.quit()  # 브라우저 종료

        if len(news) < 300:
            pass
        else:
            news = news + ' \n 한줄요약:'

            # [내 애플리케이션] > [앱 키] 에서 확인한 REST API 키 값 입력
            REST_API_KEY = ''

            # KoGPT API 호출을 위한 메서드 선언
            # 각 파라미터 기본값으로 설정
            # # KoGPT에게 전달할 명령어 구성
            # # 파라미터를 전달해 kogpt_api()메서드 호출

            prompt = '''news
            한줄요약:
            '''
            response = kogpt_api(
                prompt=news,
                max_tokens=64,
                temperature=0.7,
                top_p=0.7,
                n=1
            )
            # print(response)
            # print(list(response.values())[1][1])
            # exit()

            headlines = list(response.values())[1]
            headlines = headlines[0]
            headlines = headlines['text']
            headlines = headlines.split('\n')[0]
            headlines = headlines.split('.')[0]
            headlines = headlines.split('^')[0]
            headlines = headlines.split('▶')[0]
            headlines = headlines.split('/')[0]
            headlines = headlines.split('#')[0]

            newslist.append(headlines)

    print(newslist)

    for x in newslist:
        #print(x)
        # writedata.py
        with open(f'./news/{name}_headlines_{number_of_news+1}.txt', 'a', encoding='utf-8') as f:
            f.write(x+'\n')

    # 환경 변수 강제 설정 코드
    os.environ['JAVA_HOME'] = r'D:\\java\\bin\\server'
    with open(f'./news/{name}_headlines_{number_of_news+1}.txt', 'r', encoding='utf-8') as f:
        newsheadlines = f.readlines()
    shownewslines =''
    for abc in newsheadlines:
        shownewslines = shownewslines+f'{abc}\n'

    with open(f'./news/{name}_headlines_{number_of_news+1}.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    okt = Okt()
    nouns = okt.nouns(text)  # 명사만 추출

    words = [n for n in nouns if len(n) > 1]  # 단어의 길이가 1개인 것은 제외

    c = Counter(words)  # 위에서 얻은 words를 처리하여 단어별 빈도수 형태의 딕셔너리 데이터를 구함

    a = list(c.keys())

    for x in a:
        print(x)
        # writedata.py
        with open(f'./{name}_style.txt', 'a', encoding='utf-8') as f:
            f.write(x + '\n')


    wc = WordCloud(font_path='malgun', width=400, height=400, scale=2.0, max_font_size=250, background_color='white')

    gen = wc.generate_from_frequencies(c)
    plt.figure()
    plt.imshow(gen)

    wc.to_file(f'./wordcloud/{name}_{number_of_news + 1}.png')

    sc = stylecloud.gen_stylecloud(file_path=f"./{name}_style.txt",
                                   font_path="./Galmuri.ttc",
                                   icon_name="fas fa-flag",
                                   palette="colorbrewer.diverging.Spectral_11",
                                   background_color='white',
                                   gradient="horizontal",
                                   output_name=f"./stylecloud/{name}_WordCloud_{number_of_news + 1}.png")
    sc
    return shownewslines
