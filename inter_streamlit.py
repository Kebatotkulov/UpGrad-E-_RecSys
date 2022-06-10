from gensim.models import Word2Vec
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re 
from collections import defaultdict
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins
from googletrans import Translator
from PIL import Image
import seaborn as sns
import plotly.express as px
from meta import * # file with long texts
#spreadsheet check
# from gsheetsdb import connect
# from gspread_pandas import Spread,Client
# from google.oauth2 import service_account



st.set_page_config(layout="wide")

#Create a Google Authentication connection object
# scope = ['https://spreadsheets.google.com/feeds',
#          'https://www.googleapis.com/auth/drive']

# credentials = service_account.Credentials.from_service_account_info(
#                 st.secrets["gcp_service_account"], scopes = scope)
# client = Client(scope=scope,creds=credentials)
# spreadsheetname = "Input_holder"
# spread = Spread(spreadsheetname,client = client)


#mean vectorizer
class MeanEmbeddingVectorizer(object):
    def __init__(self, model_cbow):
        self.model_cbow = model_cbow
        self.vector_size = model_cbow.wv.vector_size

    def fit(self):  
        return self

    def transform(self, docs): 
        doc_word_vector = self.doc_average_list(docs)
        return doc_word_vector

    def doc_average(self, doc):
        mean = []
        for word in doc:
            if word in self.model_cbow.wv.index_to_key:
                mean.append(self.model_cbow.wv.get_vector(word))

        if not mean: 
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean

    def doc_average_list(self, docs):
        return np.vstack([self.doc_average(doc) for doc in docs])

#tf-idf vectorized
class TfidfEmbeddingVectorizer(object):
    def __init__(self, model_cbow):

        self.model_cbow = model_cbow
        self.word_idf_weight = None
        self.vector_size = model_cbow.wv.vector_size

    def fit(self, docs): 


        text_docs = []
        for doc in docs:
            text_docs.append(" ".join(doc))

        tfidf = TfidfVectorizer()
        tfidf.fit(text_docs)  
        # if a word was never seen it is given idf of the max of known idf value
        max_idf = max(tfidf.idf_)  
        self.word_idf_weight = defaultdict(
            lambda: max_idf,
            [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()],
        )
        return self

    def transform(self, docs): 
        doc_word_vector = self.doc_average_list(docs)
        return doc_word_vector

    def doc_average(self, doc):


        mean = []
        for word in doc:
            if word in self.model_cbow.wv.index_to_key:
                mean.append(
                    self.model_cbow.wv.get_vector(word) * self.word_idf_weight[word]
                ) 

        if not mean:  
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean
    def doc_average_list(self, docs):
      return np.vstack([self.doc_average(doc) for doc in docs])

@st.cache(allow_output_mutation=True)
def load_data(check): 
    if check: 
        data = pd.read_excel('main_data-2-2_copy (1).xlsx', index_col=0)
        embeddings = pd.read_pickle('embed.pickle')
        clean_words = pd.read_pickle('words.pickle')
        swords = pd.read_pickle('swords.pickle')
        latlong = pd.read_csv('LATandLONG.csv', index_col=0)
        progs = pd.read_pickle('nwconstr.pickle')
    return data, embeddings, clean_words, swords, latlong, progs
data, doc_vec, clean_words, swords, latlong, progs = load_data(True)
data = data[data['tuition_EUR']<90000] #looks shitty, but i don't have ehough time... haha)

# @st.cache(allow_output_mutation=True)
# def corpus_l(data):
#     return list(data)

@st.cache(allow_output_mutation=True)
def load_model(mpath): 
    return Word2Vec.load(mpath)

#load up some cleaning functions
def tokenization(text):
    tokens = re.split('\s+',text)
    return tokens

def remove_stopwords(text):
    output= [i for i in text if i not in swords]
    return output

def len_control(text):
  lemm_text = [word for word in text if len(word)>=3]
  return lemm_text

def sorter(text):
  sorted_list = sorted(text)
  return sorted_list

def make_clickable(name, link):
    # target _blank to open new window
    # extract clickable text to display for your link
    text = name
    return f'<a target="_blank" href="{link}">{text}</a>'

def program_parser2(data):
    for i in range(data.shape[0]):
        data.Introduction[i] = str(re.sub('[0-9]+',' ',re.sub(r'[^\w\s]',' ',re.sub('\\\\n', ' ' ,re.sub('&.*?;.*?;|&.*?;|._....',' ',str(data.Introduction[i]))))).lower().strip())
    data['msg_sorted_clean']= (data['Introduction']
                               .apply(lambda x: tokenization(x))
                               .apply(lambda x:remove_stopwords(x))
                               .apply(lambda x:len_control(x))
                               .apply(lambda x: sorter(x)))
    return data

def pick_n_pretty(df):
    output = df[['Link', 'program', 'university', 'country', 'city', 'language', 'tuition_EUR','Score']]
    output["Link"] = output.apply(
            lambda row: make_clickable(row["program"], row["Link"]), axis=1)
    output['tuition_EUR'] = output['tuition_EUR'].fillna(0)
    output['tuition_EUR'] = output.apply(lambda row: int(row['tuition_EUR']), axis=1)
    return output#.style.applymap(lambda x: "background-color: red" if x==0 else "background-color: white")


def get_recommendations(N, scores, data_path = 'main_data-2-2_copy (1).xlsx'):
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    data = (pd.read_excel(data_path, index_col = 0)
           .drop(columns = ['msg_sorted_clean'])
           .loc[top]
           .reset_index())
    data['Score'] = sorted(scores, reverse=True)[:N]
    return data

def get_recs(sentence, N=10, mean=False):
    '''Get top-N recommendations based on your input'''
    input = pd.DataFrame({'Introduction': [str(sentence)]})
    input = program_parser2(input)
    input_embedding = tfidf_vec_tr.transform([input['msg_sorted_clean'][0]])[0].reshape(1, -1)
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    recommendations = get_recommendations(N,scores)
    return recommendations

def mfap(recs1, df=latlong):
    latlong = recs1.merge(df, left_on='city', right_on='location', how = 'inner')      
    uni_locations = latlong[["lat", "long", "location"]]
    map = folium.Map(location=[uni_locations.lat.mean(), uni_locations.long.mean()], zoom_start=4, control_scale=True)
    for index, location_info in uni_locations.iterrows():
        folium.Marker([location_info["lat"], location_info["long"]], popup=location_info["location"]).add_to(map)
    return map

def mfap_density_50(recs50, df=latlong): #try this function on the main page
    latlong = recs50.merge(df, left_on='city', right_on='location', how = 'inner')
    uni_locations = latlong[["lat", "long"]]
    map = folium.Map(location=[uni_locations.lat.mean(), uni_locations.long.mean()], zoom_start=4, control_scale=True)
    cityArr = uni_locations.values
    map.add_child(plugins.HeatMap(cityArr, radius=25))
    return map

def sim_prog(df=progs, prog=None):
    df_one = df[df['Program1']==prog]
    return df_one.sort_values(by='cosine', ascending=False)
#yes.... the code is suboptimal, but i don't care about image.pngs point now ;)
def p2p_locs(latlong=latlong, uni_info=data, recs=[], N=5): #recs is the output of sim_progs #density map for similar universities
    recs[['Uni', 'Prog']] = recs['Program2'].str.split(': ', 1, expand=True)
    recs[['Uni1', 'Prog1']] = recs['Program1'].str.split(': ', 1, expand=True)
    ps = (recs
            .merge(uni_info, left_on=['Uni', 'Prog'], right_on=['university','program'], how = 'inner')
            .merge(latlong, left_on='city', right_on='location', how ='inner'))
    fin_rec = ps[['Program1', 'Program2', 'city','cosine']].reset_index().iloc[1:N+1,:]
    uni_locations = ps[["lat", "long"]]
    map = folium.Map(location=[uni_locations.lat.mean(), uni_locations.long.mean()], zoom_start=4, control_scale=True)
    cityArr = uni_locations.values
    map.add_child(plugins.HeatMap(cityArr, radius=25))
    return map, fin_rec, ps 

def simple_output(map=True):
    col1, col2, col3 = st.columns([10, 10, 10])
    with col2:
        gif_runner = st.image("200.gif")
    recs1 = get_recs(str(text), N=int(number), mean=False)
    recs50 = get_recs(str(text), N=50, mean=False)
    recs1 = pick_n_pretty(recs1)
    gif_runner.empty()
    df = recs1.style.background_gradient(
        cmap=cmGreen,
        subset=[
            "Score",
        ],
    )
    st.write(df.to_html(escape=False), unsafe_allow_html=True)  
    if map:
        map2 = mfap_density_50(recs50) 
        map  = mfap(recs1)
        st.write('')
        st.write('')
        with st.expander('Посмотреть интерактивные карты 🌍'):
            ce, c1, ce, c2, c3 = st.columns([0.07, 4, 0.07, 4, 0.07])
            with c1:
                st.write('Расположение запрашиваемых университетов')
                folium_static(map, width=450) 
            with c2:
                st.write('POI-распределение городов топ-50 соответствующих Вашему запросу программ')
                folium_static(map2, width=450)


with st.sidebar:
    col1, col2, col3 =st.columns([2.2, 6, 2.2])
    with col1:
        st.write("")
    with col2:
        st.image('keystone-masters-degree.jpg') 
    with col3:
        st.write('')
    page = st.radio('Страница', ['Приветствие👋',"Найти программу🌍", "Найти схожие программы🙌","Данные и статистика📈"])
    
    # st.subheader('Выбери параметры')
    # location = st.multiselect('Страна', list(set(data['country'])))
    # on_site = st.selectbox('Темп обучения', ['Очное обучение', 'Заочное обучение','Очное обучение|Заочное обучение'])
    # pace = st.selectbox('Форма обучения', ['Онлайн', 'Кампус','Кампус|Онлайн'])
    # lang = st.selectbox('Форма обучения', list(set(data['Language'].dropna())))
    # cost = st.slider('Стоимость обучения, EUR', int(data['tuition_EUR'].min()), int(data['tuition_EUR'].max()), (0, 3000), step=50)

# Page 1-Intro
if page=='Приветствие👋':
    #_max_width_()
    c30, c31, c32 = st.columns([2.5, 1, 3])

    with c30:
        st.image("keystone-masters-degree.jpg", width=400)
    

    st.markdown("""
            Добро пожаловать, дорогой пользователь!
            
            Решил не останавливаться на бакалавриате и продолжить грызть гранит науки в магистратуре? Значит, ты по верному адресу!

            Мы предлагаем тебе рекомендации магистерских программ российских и зарубежных университетов на основе твоих предпочтений!

            Сайт предназначен для тех, кто желает продолжить обучение, но испытывает трудности в поиске подходящей программы, университета или страны.
        """, unsafe_allow_html = True)

    with st.expander("ℹ️ - Идея проекта", expanded=False):

        st.markdown(
            """ 
Несмотря на возможность получения дополнительного образования в своих родных городах, наблюдается стремительное увеличение числа студентов из СНГ, желающих получить больше формальной 'корочки' о какой-либо степени. Выбор университета за пределами своего города, и даже страны, не заканчивается формальными рейтингами, стоимостью и другими измеримыми факторами. *По этой причине, наша проектная группа заинтересовалась возможностью предоставления возможности найти что-то близкое к своим предпочтениям не только информированным, но и менее уверенным в себе слушателям.* Наша система состоит из четырех страниц, которые были построены по определенной логике:

1. Пользователь сначала ознакамливается с близкими к его описанию своих способностей, интересов и желаний программами на основе очевидных парамеров, локаций и информации с сайта Keystone (на сраницы которого предоставлены соотвествующие ссылки).

2. Может расширить свой список потенцильных университетов на следующей странице, выбрав самую лучшую программу для получения списка самых похожих программ.

3. Может ознакомиться с нашей базой, а дальше изучить об университетах различных стран из базы на основе интерактивных графиков.  

*На каждой странице будет инструкция! Наша тебе даже если ты совсем не знаешь чего ты хочешь!* 
            """
        )

        st.markdown("")

    col1, col2 = st.columns([5,5])
    with col1:
        with st.expander("Данные", expanded=False):

            st.markdown( """
            Проект основан на данных с сайта [masterstudies.ru](https://masterstudies.com). Сбор данных оказался более комлексной задачей, чем ожидалось изначально. Процесс сбора состоял из трёх этапов:

            *   Сбор ссылок на разделы сайта с программами из различных категорий ('Экономические исследования','Управление бизнесом', 'Есетственные науки' и другие фундаментальные направления представленные в НИУ ВШЭ (СПб))
            *   Сбор общей информации о программах () из карточек программ (название программы, название университета, вид программы, начало обучения и длительность программы, формат и тип обучения, страна, стоимость обучения, язык обучения, и дедлайны подачи заявок)
            * Сбор текстовой информации из страниц программ на Keystone 

            Сырые данные, как правило, проходят процедуру очистки, которая получила важную роль для продолжения работы над фичами. 
            * Мы очистили данные от выбросов, привели данные в столбцах к единому формату
            * Мы создали два новых датасета: один с локациями и очищенными названиями городов, а другой с близостью программ на основе текстовых описаний программ

            В итоге был получен относительно чистый датасет с 6502 программами, на который опирается наше приложение.
    """, unsafe_allow_html = True)

            st.markdown("")
    with col2:
        with st.expander("Mетоды ", expanded=False):

            st.write(
                """

**Рекомендательная система**

Для построения рекомендательной системы нам потребовалось представление *английского* текста описаний магистерских программ в эмбеддингах. Для этого была выбрана вариация CBOW (continuous bag of words) Word2Vec, до которой документы прошли небольшую обработку - очистка от символов и цифр, лемматизация, токенизация и сортировка для более коректной работы Word2Vec. Данный метод более релевантен в силу принципа работы, основанного на предсказании центральных слов в документе в зависимости от окружающих его слов. Далее каждый документ (описания и пользовательские вводы) представляются в виде препрезентативных векторов на основе метода IDF, который назначает меньший вес более распространенным словам -- это позволяет нам достичь большей различительной силы алгоритма. 

 * Пользовательские вводы

Мы нацелены на русскоговорящих пользователей, поэтому пользовательские вводы должны быть на русском языке. По этой причине, мы внедрили переводчики текста с русского на английский язык для последующей обработки и векторизации уже английского текста и подсчета косинусных расстояний между соответствующими векторами описаний программ и текстовым вводом.

 * Попарное выявление схожих программ

Для дополнения запросов пользователей интересными программами мы посчитали нужным выделение схожих программ на основе подсчета попарных косинусных расстояний между всеми программами в базе. Пользователь получает только релевантные опции со схожестью более 0,65 (иначе было бы сложно работать с базой из >40 млн пар. 

 * Интерактивная визуализация

Пользователям может быть интересны локации схожих программ, поэтому мы построили интерактивные карты POI (или KDE-плотность распределения городов на карте) и локаций для представления более широкого выбора.

""")

            st.markdown("")

    st.markdown("")
    #st.write(spread.url)

   # st.markdown(hello, unsafe_allow_html = True)

if page=='Найти программу🌍':
    #_max_width_()
    c30, c31, c32 = st.columns([2.5, 1, 3])

    with c30:
        st.image("keystone-masters-degree.jpg", width=400)

    with st.expander("ℹ️ - Об этой странице + Инструкция", expanded=False):

        st.markdown("""
                На этой странице ты сможешь ознакомиться с близки твоим предпочтениями типами программ. Интерфейс достаточно прост в применении, поэтому расскажем только о критически важных аспектах взаимодействия. У тебя есть возможность получения общих и более специфических рекомендаций, которые зависят от твоих действий.

                * Если ты желаешь получить *все возможные ильтернативы*, то тебе стоит: ПОСТАВИТЬ ГАЛОЧКУ слева от 'Выключить фильтрацию', выбрать количество программ, ввести текст о своих интересах, способностях или достижениях - это может быть любой текст на русском языке, написанный на кириллице. 

                * Если детали все-таки важны, то тебе стоит заполнить все поля с параметрами и ввести текст, но НЕ СТАВИТЬ ГАЛОЧКУ

                **Кроме этого, важно запомнить об опциональных элементах интерфейса, которые можно будет увидеть в результате запроса -- у тебя будет возможность скрыть или раскрыть таблицы и карты.**
                
                ВАЖНО: Языки и страны можно найти обычным вводом, чтобы система предложила автозаполнение. 

                Спасибо за интерес! Удачи!  
        """)

        st.markdown("")

    st.markdown("")

    #scenario_interact = st.selectbox(
     #   "Выберите сценарий взаимодействия в зависимости от существующего опыта",
      #  ["Хочу найти университет", "Хочу расширить свой список программ"],
    

    with st.form(key="my_form"):

        ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 4, 0.07])
        with c1:
            st.subheader('Выберите параметры') 
            number = st.number_input('Сколько рекоммендаций желаешь увидеть на экране?', min_value=0, max_value=50, step=1, value=5)
            agree = st.checkbox('Выключить фильтрацию')
            location = st.multiselect('Страна', sorted(list(set(data['country'].dropna()))))
            dur = st.slider('Продолжительность обучения (мес)', int(data['duration_month'].min()), int(data['duration_month'].max()), (0, 10), step=2)
            on_site = st.selectbox('Тип обучения', ['Очное обучение', 'Заочное обучение','Очное обучение|Заочное обучение'])
            pace = st.selectbox('Формат обучения', ['Онлайн', 'Кампус','Кампус|Онлайн'])
            lang = st.multiselect('Язык обучения', sorted(list(set(data['language'].dropna()))))
            cost = st.slider('Стоимость обучения в год, EUR', int(data['tuition_EUR'].min()), int(data['tuition_EUR'].max()), (0, 8000), step=50)
        with c2:
             #to make row effects
            st.markdown('')
            st.markdown('')
            sentence = st.text_area("Введи текст для выявления своих предпочтений -- можешь ввести что угодно, но цифры и символы не учитываются нашей системой", value='Например: я знаю статистику, прошел курсы по анализу данных и интересуюсь финансовыми рынками')
            submit = st.form_submit_button(label="✨ Получить рекомендацию")
            corpus = list(clean_words)
            model = load_model('model_cbow.bin')
            model.init_sims(replace=True)
            tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
            tfidf_vec_tr.fit(corpus)
            translator = Translator()
            result = translator.translate(sentence)
            text = result.text

    if not submit:
        st.stop()

    cmGreen = sns.light_palette("green", as_cmap=True)
    if submit:
        if len(text)==0:
            st.warning('Ты не рассказал(а) о своих предпочтениях! В данном случае система выдаст первые {} строк(и) нашей базы с программами.... Это не очень интересно'.format(number))
            simple_output()
        else:
            if not agree:
                if len(location)>0 and len(lang)>0:  
                    col1, col2, col3 = st.columns([10, 10, 10])
                    with col1:
                        st.write('')
                    with col2:
                        gif_runner = st.image("200.gif")
                    with col3:
                        st.write('')
                    recs = get_recs(str(text), N=int(data.shape[0]), mean=False)
                    recs50 = get_recs(str(text), N=50, mean=False) #вот здесь надо изменить про ввод количества желаемых программ, а потом просто выдавать топ 
                    gif_runner.empty()  
                    recs1 = recs[(recs['language'].isin(list(lang))) & (recs['country'].isin(list(location))) & (recs['on_site']==on_site) & (recs['format']==pace)  & (recs['tuition_EUR']>min(cost)) & (recs['tuition_EUR']<max(cost)) & (recs['duration_month']>min(dur)) & (recs['duration_month']<=max(dur))]
                    recs1 = recs1.reset_index().iloc[:number,:]
                    if recs1.shape[0]!=0:
                        recs2 = pick_n_pretty(recs1)
                        df = recs2.style.background_gradient(
                            cmap=cmGreen,
                            subset=[
                                "Score",
                            ],
                        )
                        st.write(df.to_html(escape=False), unsafe_allow_html=True)
                        map2 = mfap_density_50(recs50)
                        map  = mfap(recs2)
                        st.write('')
                        st.write('')
                        with st.expander('Посмотреть интерактивные карты🌍'):
                            ce, c1, ce, c2, c3 = st.columns([0.07, 4, 0.07, 4, 0.07])
                            with c1:
                                st.write('Расположение запрашиваемых университетов')
                                folium_static(map, width=450) 
                            with c2:
                                st.write('POI-распределение городов топ-50 программ, соответствующих твоему запросу')
                                folium_static(map2, width=450)
                        if recs1.shape[0]<number:
                            st.warning("Упс... Программ меньше чем ожидалось, но эту проблему можно решить... Обрати внимание на опцию ниже)")
                            recs1 = recs.copy()
                            recs2 = pick_n_pretty(recs1)
                            map3 = mfap(recs2)
                            df = recs2.style.background_gradient(
                                cmap=cmGreen,
                                subset=[
                                    "Score",
                                ],
                            )
                            with st.expander('Предлагаем ознакомиться с дополнительными опциями из нашей базы 👉'):
                                st.write(df.to_html(escape=False), unsafe_allow_html=True)
                                C, D, E = st.columns([2,5,2])
                                with C:
                                    st.write('')
                                with D:
                                    st.write('Расположение университетов из представленной выше таблицы')
                                    folium_static(map3)
                                with E:
                                    st.write('')

                            
                    else:
                        st.warning('Мы не смогли подобрать программы, соответствующие твоим требованиям, но просим ознакомиться с существующими в нашей базе ')
                        simple_output()
                else: 
                    st.warning('Мы не смогли подобрать программы, соответствующие твоим требованиям, но просим ознакомиться с существующими в нашей базе ')
                    simple_output()
                    # st.write('This is an error') #Надо будет полностью дописать

            else: 
                simple_output()
if page=='Найти схожие программы🙌':
    c30, c31, c32 = st.columns([2.5, 1, 3])

    with c30:
        st.image("keystone-masters-degree.jpg", width=400)

    with st.expander("ℹ️ - Об этой старнице + Инструкция", expanded=False):

        st.write("На этой странице ты сможешь ознакомиться с программами, похожими на выбранную тобой. Ты можешь выбрать или ввести первые буквы названия программы, чтобы программами предложила автозавполнение.")

        st.markdown("")

    st.markdown("")    

    st.write('Выбери одну программу для получения схожих')
    with st.form(key="my_form"):
        university_pick = st.selectbox("Список существующих в нашей базе магистерских программ", list(set(progs['Program1'].dropna())))
        number_sim = st.number_input('Количество схожих программ', min_value=0, max_value=50, step=1, value=5)
        submit = st.form_submit_button(label="✨ Показать университеты")
        cmGreen = sns.light_palette("green", as_cmap=True)
    if submit:
        recs = sim_prog(progs, str(university_pick))
        map, recs0, ps = p2p_locs(recs=recs, N=number_sim) #ps is a dirty dataset
        df = recs0.sort_values(by='cosine', ascending=False).style.background_gradient(
            cmap=cmGreen,
            subset=[
                "cosine",
            ],
        )
        see_data = st.expander('Посмотреть схожие программы 👉')
        with see_data:
            st.write(df.to_html(escape=False), unsafe_allow_html=True)
        st.write('')
        ce, c1, ce, c2, c3 = st.columns([0.07, 4, 0.07, 4, 0.07])
        with c1:
            st.write('Распределение схожих программ')
            folium_static(map, width=500)
        with c2:
            st.write('')
            c_metric = ps.groupby('Program1')['Program2'].count()[0]
            st.metric(label='Схожих программ', value='{}'.format(c_metric))
            d_metric = ps[ps.Uni==ps.Uni1].shape[0]
            st.metric(label='В одном университете', value='{}'.format(d_metric))
            top = ps.groupby(['city'])['Program2'].agg(['count']).sort_values(by = 'count', ascending=False)
            df = pd.DataFrame({'Город':list(top.index), 'Количество': list(top['count'])})
            with st.expander('Встречаемость стран схожих программ', expanded=True):
                st.dataframe(data=df)
            st.text('')



   
if page == 'Данные и статистика📈':
    #_max_width_()
    #data = pd.read_excel('main_data-2.xlsx', index_col=0)
    c30, c31, c32 = st.columns([2.5, 1, 3])

    with c30:
        st.image("keystone-masters-degree.jpg", width=400)


        st.markdown("")
    see_data = st.expander('Посмотреть данные 👉')
    with see_data:
        data['duration_month'] = data['duration_month'].astype('str')
        st.dataframe(data=data)
    st.text('')

    with st.expander('Общая статистика'):

        col1, col2, col3, col4, col5, col6, col7 = st.columns([1,4,1,4,1,4,1])
        qprogs, qcountry, quni = data.shape[0], len(set(data['country'])), len(set(data['university'])) 
        col1.write('')
        col2.metric("Стран 🌐", "{}".format(qcountry))
        col3.write('')
        col4.metric("Университетов 🎓","1440") #"{}".format(quni))
        col5.write('')
        col6.metric("Программы 🚀", "6502")#"{}".format(qprogs))
        col7.write('')

        st.write('')

        data1 = data.groupby('country').count().reset_index().sort_values(by='Link', ascending=False).head(6)
        data2 = data.groupby('format').count().reset_index()

        ce, c1, ce, c2, c3 = st.columns([0.07, 4, 0.07, 4, 0.07])
        with c1:
            fig1 = px.pie(data1, names='country', values='Link', color_discrete_sequence=px.colors.sequential.RdBu, labels={
                        "country": "Страна",
                        "n": "Количесво программ",
                    }, title='ТОП-стран по количеству программ',  width=600, height=400)
            fig1.update_layout(paper_bgcolor="black",
                                font_color="white")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = px.pie(data2, names='format', values='Link', color_discrete_sequence=px.colors.diverging.RdYlGn, labels={
                        "format": "Формат",
                        "n": "Количесво программ",
                    }, title='Соотношщение программ по форматам обучения',  width=600, height=400)
            fig2.update_layout(paper_bgcolor="black",
                                font_color="white")
            st.plotly_chart(fig2, use_container_width=True)
        data['duration_month'] = data['duration_month'].astype('float32')
        data3 = data.groupby('country')['tuition_EUR'].agg(['mean']).reset_index().sort_values(by='mean', ascending=True).head(43)
        data4 = data.groupby('country')['duration_month'].agg(['mean']).reset_index().sort_values(by='mean', ascending=False).head(45)
        ce, c1, ce, c2, c3 = st.columns([0.07, 4, 0.07, 4, 0.07])
        with c1:
            fig3 = px.bar(data3, x = "mean", y = "country", orientation='h', labels={
                                "mean": "Стоимость обучения",
                                "country": "Страна"
                            }, title='Средняя стоимость обучения по странам')
            fig3.update_traces(marker_color='red', marker_line_color='red',
                            marker_line_width=1, opacity=1)
            fig3.update_layout(legend_font_size=1, width=800,
                height=900, paper_bgcolor="black", font_color='white')
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            fig6 = px.bar(data4, x = "mean", y = "country", orientation='h', text_auto=True, labels={
                                "mean": "Длительность (мес)",
                                "country": "Страна",
                            }, title='Средняя длительность обучения')
            fig6.update_traces(marker_color='red', marker_line_color='red',
                            marker_line_width=1, opacity=1)
            fig6.update_layout(yaxis=dict(autorange="reversed"), legend_font_size=1, width=700,
                height=900, paper_bgcolor="black", font_color='white')
            st.plotly_chart(fig6, use_container_width=True)
        
    st.write('')

    st.write('Выберите страну для получения большего объема информации об университетах')
        
    with st.form(key="my_form"):
        country = st.selectbox("Список существующих в нашей базе стран", list(set(data['country'].dropna())))
        submit = st.form_submit_button(label="✨ Подтвердить выбор")
    if submit:
        data_gb = data[data['country']==country]
        with st.expander('{} - замечательный выбор!'.format(country)):
            data_f = data_gb.groupby('format').count().reset_index()
            data_l = data_gb.groupby('language').count().reset_index()
            ce, c1, ce, c2, c3 = st.columns([0.07, 4, 0.07, 4, 0.07])
            with c1:
                figf = px.pie(data_f, names='format', values='Link', color_discrete_sequence=px.colors.sequential.RdBu, labels={
                            "format": "Формат",
                            "Link": "Количесво программ",
                        }, title='Соотношение программ по форматам обучения',  width=600, height=400)
                figf.update_layout(paper_bgcolor="black",
                                    font_color="white")
                st.plotly_chart(figf, use_container_width=True)

            with c2:
                figl = px.pie(data_l, names='language', values='Link', color_discrete_sequence=px.colors.diverging.RdYlGn, labels={
                            "language": "Язык обучения",
                            "Link": "Количесво программ",
                        }, title='Существующие языки обучения',  width=600, height=400)
                figl.update_layout(paper_bgcolor="black",
                                    font_color="white")
                st.plotly_chart(figl, use_container_width=True)
            
            st.write('')
            data_count = data_gb.groupby('university').count().reset_index().sort_values(by='Link', ascending=False).head(60)
            data_cost = data_gb.groupby('university')['tuition_EUR'].agg(['mean']).reset_index().sort_values(by='mean', ascending=False).head(60)
            c1, c2, c3 = st.columns([0.05, 6 ,0.05])
            with c2:
                fig10 = px.bar(data_count, x = "Link", y = "university", orientation='h', text_auto=True, labels={
                     "Link": "Количество программ в университете",
                     "university": "Университет",
                 }, title='Встречаемость программ в университетах')

                fig10.update_traces(marker_color='red', marker_line_color='red',
                                marker_line_width=1, opacity=1)

                fig10.update_layout(yaxis=dict(autorange="reversed"), legend_font_size=1, width=1000,
                    height=1100,  paper_bgcolor="black", font_color='white')
                st.plotly_chart(fig10, use_container_width=True)
            
            st.write('')

            c1, c2, c3 = st.columns([0.05, 6 ,0.05])
            with c2:
                fig8 = px.bar(data_cost.dropna(), x = "mean", y = "university", orientation='h', text_auto=True, labels={
                        "mean": "Стоимость обучени (EUR/год)",
                        "university": "Университет",
                    }, title='Средняя стоимость обучения в университетах')

                fig8.update_traces(marker_color='red', marker_line_color='red',
                    marker_line_width=1, opacity=1)

                fig8.update_layout(yaxis=dict(autorange="reversed"), legend_font_size=1, width=1000,
                        height=1100,  paper_bgcolor="black", font_color='white')
                st.plotly_chart(fig8, use_container_width=True)
            

                


            
