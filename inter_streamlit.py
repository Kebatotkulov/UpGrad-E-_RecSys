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
from googletrans import Translator
from PIL import Image
#spreadsheet check
# from gsheetsdb import connect
# from gspread_pandas import Spread,Client
# from google.oauth2 import service_account




# Create a Google Authentication connection object
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
        data = pd.read_excel('full_info_6500_itog.xlsx')
        embeddings = pd.read_pickle('embed.pickle')
        clean_words = pd.read_pickle('words.pickle')
        swords = pd.read_pickle('swords.pickle')
        latlong = pd.read_csv('LATandLONG.csv', index_col=0)
    return data, embeddings, clean_words, swords, latlong
data, doc_vec, clean_words, swords, latlong = load_data(True)

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


def program_parser2(data):
    for i in range(data.shape[0]):
        data.Introduction[i] = str(re.sub('[0-9]+',' ',re.sub(r'[^\w\s]',' ',re.sub('\\\\n', ' ' ,re.sub('&.*?;.*?;|&.*?;|._....',' ',str(data.Introduction[i]))))).lower().strip())
    data['msg_sorted_clean']= (data['Introduction']
                               .apply(lambda x: tokenization(x))
                               .apply(lambda x:remove_stopwords(x))
                               .apply(lambda x:len_control(x))
                               .apply(lambda x: sorter(x)))
    return data

def get_recommendations(N, scores, data_path = 'full_info_6500_itog.xlsx'):
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    data = (pd.read_excel(data_path, index_col = 0)
           .drop(columns = ['msg_sorted_clean'])
           .loc[top]
           .reset_index())
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

def mfap(recs1, df = latlong):
    latlong = recs1.merge(df, left_on='city ', right_on='location', how = 'left')      
    uni_locations = latlong[["lat", "long", "location"]]
    map = folium.Map(location=[uni_locations.lat.mean(), uni_locations.long.mean()], zoom_start=4, control_scale=True)
    for index, location_info in uni_locations.iterrows():
        folium.Marker([location_info["lat"], location_info["long"]], popup=location_info["location"]).add_to(map)
    return map

with st.sidebar:
    col1, col2, col3 =st.columns([2.2,6, 2.2])
    with col1:
        st.write("")
    with col2:
        st.image('keystone-masters-degree.jpg')
    with col3:
        st.write('')
    page = st.radio('Страница', ['Приветствие','Поиск программ','Интересная статистика'])
    
    st.subheader('Выбери параметры')
    location = st.multiselect('Страна', list(set(data['country'])))
    on_site = st.selectbox('Темп обучения', ['Очное обучение', 'Заочное обучение','Очное обучение|Заочное обучение'])
    pace = st.selectbox('Форма обучения', ['Онлайн', 'Кампус','Кампус|Онлайн'])
    lang = st.selectbox('Форма обучения', list(set(data['language'].dropna())))
    cost = st.slider('Стоимость обучения, EUR', int(data['tuition_EUR'].min()), int(data['tuition_EUR'].max()), (0, 3000), step=50)

# Page 1-Intro
if page=='Приветствие':
    img = Image.open("keystone-masters-degree.jpg")
    st.image(img)
  #  st.markdown(dash, unsafe_allow_html = True)
    st.markdown("## How it works? :thought_balloon:")
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

if page=='Поиск программ':
    st.title("Университеты в Европе")

    form = st.form(key="my_form")
    sentence = form.text_input(label="Введи текст для выявления своих предпочтений", placeholder = 'Например: я знаю статистику, прошел курсы по анализу данных и интересуюсь финансовыми рынками')
    submit = form.form_submit_button(label="Найти идеальную программу")


            #deadline
    corpus = list(clean_words)
    model = load_model('model_cbow.bin')
    model.init_sims(replace=True)
    tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
    tfidf_vec_tr.fit(corpus)
    translator = Translator()
    result = translator.translate(sentence)
    text = result.text
    # if submit_button:
    # # make prediction from the input text
    #     recs = get_recs(str(text))
    #     recs1 = recs[(recs['country'].isin(location)) & (recs['on_site'].isin(on_site))]
    #     st.dataframe(recs1)
    # else: 
#    if submit:
 #       if submit_button:
        #gif_runner = st.image("giphy.gif")
        #gif_runner.empty() 
    if submit:
        if len(location)>0:    
            recs = get_recs(str(text))
            recs1 = recs[(recs['country'].isin(location)) & (recs['on_site']==on_site) & (recs['format']==pace)]
            st.dataframe(recs1)
            map  = mfap(recs1)
            folium_static(map)
        else: 
            recs1 = get_recs(str(text))
            st.dataframe(recs1)
            map  = mfap(recs1)
            folium_static(map) 
        
    
       
        #else:
         #   recs = get_recs(str(text))
         #   st.dataframe(recs)
    #else: 
        #st.write('Результат можно получить только после нажатия кнопки')
    
   
    # Display results of the NLP task
if page == 'Интересная статистика':
    st.title('Здесь должна быть описательная статистика')




    folium_static(map) 

    