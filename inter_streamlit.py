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
        data = pd.read_excel('main_data.xlsx')
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
    output = df[['Link', 'program', 'university', 'country', 'city ', 'language', 'tuition_EUR','Score']]
    output["Link"] = output.apply(
            lambda row: make_clickable(row["program"], row["Link"]), axis=1)
    output['tuition_EUR'] = output['tuition_EUR'].fillna(0)
    output['tuition_EUR'] = output.apply(lambda row: int(row['tuition_EUR']), axis=1)
    return output#.style.applymap(lambda x: "background-color: red" if x==0 else "background-color: white")


def get_recommendations(N, scores, data_path = 'main_data.xlsx'):
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
    latlong = recs1.merge(df, left_on='city ', right_on='location', how = 'inner')      
    uni_locations = latlong[["lat", "long", "location"]]
    map = folium.Map(width=1000,height=500,location=[uni_locations.lat.mean(), uni_locations.long.mean()], zoom_start=4, control_scale=True)
    for index, location_info in uni_locations.iterrows():
        folium.Marker([location_info["lat"], location_info["long"]], popup=location_info["location"]).add_to(map)
    return map

def mfap_density_50(recs50, df=latlong): #try this function on the main page
    latlong = recs50.merge(df, left_on='city ', right_on='location', how = 'inner')
    uni_locations = latlong[["lat", "long"]]
    map = folium.Map(width=1000,height=500,location=[uni_locations.lat.mean(), uni_locations.long.mean()], zoom_start=4, control_scale=True)
    cityArr = uni_locations.values
    map.add_child(plugins.HeatMap(cityArr, radius=25))
    return map

def sim_prog(df=progs, prog=None):
    df_one = df[df['Program1']==prog]
    return df_one.sort_values(by='cosine', ascending=False)

def p2p_locs(latlong=latlong, uni_info=data, recs=[], N=5): #recs is the output of sim_progs #density map for similar universities
    recs[['Uni', 'Prog']] = recs['Program2'].str.split(': ', 1, expand=True)
    recs[['Uni1', 'Prog1']] = recs['Program1'].str.split(': ', 1, expand=True)
    ps = (recs
            .merge(uni_info, left_on=['Uni', 'Prog'], right_on=['university','program'], how = 'inner')
            .merge(latlong, left_on='city ', right_on='location', how ='inner'))
    fin_rec = ps[['Program1', 'Program2', 'city ','cosine']].reset_index().iloc[1:N+1,:]
    uni_locations = ps[["lat", "long"]]
    map = folium.Map(width=1000,height=500,location=[uni_locations.lat.mean(), uni_locations.long.mean()], zoom_start=4, control_scale=True)
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
            A, B = st.columns([5, 5])
            with A:
                st.write('Расположение запрашиваемых университетов')
                folium_static(map) 
            with B:
                st.write('POI-распределение городов топ-50 соответствующих Вашему запросу программ')
                folium_static(map2)


with st.sidebar:
    col1, col2, col3 =st.columns([2.2, 6, 2.2])
    with col1:
        st.write("")
    with col2:
        st.image('keystone-masters-degree.jpg')
    with col3:
        st.write('')
    page = st.radio('Страница', ['Приветствие👋',"Найти программу🌍", "Найти похожие программы🙌","Интересная статистика📈"])
    
    # st.subheader('Выбери параметры')
    # location = st.multiselect('Страна', list(set(data['country'])))
    # on_site = st.selectbox('Темп обучения', ['Очное обучение', 'Заочное обучение','Очное обучение|Заочное обучение'])
    # pace = st.selectbox('Форма обучения', ['Онлайн', 'Кампус','Кампус|Онлайн'])
    # lang = st.selectbox('Форма обучения', list(set(data['Language'].dropna())))
    # cost = st.slider('Стоимость обучения, EUR', int(data['tuition_EUR'].min()), int(data['tuition_EUR'].max()), (0, 3000), step=50)

# Page 1-Intro
if page=='Приветствие👋':
    img = Image.open("keystone-masters-degree.jpg")
    st.image(img)
  #  st.markdown(dash, unsafe_allow_html = True)
    st.markdown("## How it works? :thought_balloon:")
    #st.write(spread.url)
    st.write(
        "For an in depth overview of the ML methods used and how I created this app, three blog posts are below."
        )
    blog1 = "https://jackmleitch.medium.com/using-beautifulsoup-to-help-make-beautiful-soups-d2670a1d1d52"
    blog2 = "https://towardsdatascience.com/building-a-recipe-recommendation-api-using-scikit-learn-nltk-docker-flask-and-heroku-bfc6c4bdd2d4"
    blog3 = "https://towardsdatascience.com/building-a-recipe-recommendation-system-297c229dda7b"
    st.markdown(
        f"1. [Web Scraping Cooking Data With Beautiful Soup]({blog1})"
        )
    st.markdown(
            f"2. [Building a Recipe Recommendation API using Scikit-Learn, NLTK, Docker, Flask, and Heroku]({blog2})"
        )
    st.markdown(
            f"3. [Building a Recipe Recommendation System Using Word2Vec, Scikit-Learn, and Streamlit]({blog3})"
        )
    #st.write(spread.url)

   # st.markdown(hello, unsafe_allow_html = True)

if page=='Найти программу🌍':
    #_max_width_()
    c30, c31, c32 = st.columns([2.5, 1, 3])

    with c30:
        st.image("keystone-masters-degree.jpg", width=400)

    with st.expander("ℹ️ - Об этой старнице", expanded=False):

        st.write(
            """
            Надо написать инструкцию по использованию этой страницы - хуета крч     
    -   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
    -   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
            """
        )

        st.markdown("")

    st.markdown("")

    #scenario_interact = st.selectbox(
     #   "Выберите сценарий взаимодействия в зависимости от существующего опыта",
      #  ["Хочу найти университет", "Хочу расширить свой список программ"],
    

    with st.form(key="my_form"):

        ce, c1, ce, c2, c3 = st.columns([0.07, 2, 0.07, 4, 0.07])
        with c1:
            st.subheader('Выберите параметры') 
            number = st.number_input('Сколько рекоммендаций желаете увидеть на экране?', min_value=0, max_value=50, step=1, value=5)
            agree = st.checkbox('Выключить фильтрацию')
            location = st.multiselect('Страна', sorted(list(set(data['country'].dropna()))))
            on_site = st.selectbox('Темп обучения', ['Очное обучение', 'Заочное обучение','Очное обучение|Заочное обучение'])
            pace = st.selectbox('Форма обучения', ['Онлайн', 'Кампус','Кампус|Онлайн'])
            lang = st.multiselect('Язык обучения', sorted(list(set(data['language'].dropna()))))
            cost = st.slider('Стоимость обучения, EUR', int(data['tuition_EUR'].min()), int(data['tuition_EUR'].max()), (0, 8000), step=50)
        with c2:
            st.write('''
            
            
            
            ''') #to make row effects
            st.markdown('')
            st.markdown('')
            sentence = st.text_area("Введите текст для выявления своих предпочтений -- можете ввести что угодно, но цифры и символы не учитываются нашей системой", value='Например: я знаю статистику, прошел курсы по анализу данных и интересуюсь финансовыми рынками')
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
            st.warning('Вы не рассказали о своих предпочтениях! В данном случае система выдаст первые {} строк(и) нашей базы с программами.... Это не очень интересно'.format(number))
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
                    recs = get_recs(str(text), N=int(number), mean=False)
                    recs50 = get_recs(str(text), N=50, mean=False)
                    gif_runner.empty()  
                    recs1 = recs[(recs['language'].isin(list(lang))) & (recs['country'].isin(list(location))) & (recs['on_site']==on_site) & (recs['format']==pace) & (recs['tuition_EUR']>min(cost)) & (recs['tuition_EUR']<max(cost))]
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
                            A, B = st.columns([5, 5])
                            with A:
                                st.write('Расположение запрашиваемых университетов')
                                folium_static(map) 
                            with B:
                                st.write('POI-распределение городов топ-50 соответствующих Вашему запросу программ')
                                folium_static(map2)
                        if recs1.shape[0]<number:
                            st.warning("Упс... Программ меньше чем ожидалось, но эту проблему можно решить... Обратите внимание на опцию ниже)")
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
                        st.warning('Мы не смогли подобрать программы, соответствующие Вашим требованиям, но просим ознакомиться с существующими в нашей базе ')
                        simple_output()
                else: 
                    st.write('This is an error') #Надо будет полностью дописать

            else: 
                simple_output()
if page=='Найти похожие программы🙌':
    c30, c31, c32 = st.columns([2.5, 1, 3])

    with c30:
        st.image("keystone-masters-degree.jpg", width=400)

    with st.expander("ℹ️ - Об этой старнице", expanded=False):

        st.write(
            """
            Надо написать инструкцию по использованию этой страницы - хуета крч     
    -   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
    -   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
            """
        )

        st.markdown("")

    st.markdown("")    

    st.write('Выберите одну программу для дальнейшего анализа')
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
        a, b = st.columns([5,5])
        with a:
            st.write('Распределение схожих программ')
            folium_static(map)
        with b:
            st.write('')
            c_metric = ps.groupby('Program1')['Program2'].count()[0]-1
            st.metric(label='Схожих программ', value='{}'.format(c_metric))
            d_metric = ps[ps.Uni==ps.Uni1].shape[0]-1
            st.metric(label='В одном университете', value='{}'.format(d_metric))
            top = ps.groupby(['city '])['Program2'].agg(['count']).sort_values(by = 'count', ascending=False).head()
            df = pd.DataFrame({'Город':list(top.index), 'Количество': list(top['count'])})
            st.write('Встречаемость стран схожих программ')
            st.table(df)


   
if page == 'Интересная статистика📈':
    st.title('Здесь должна быть описательная статистика')