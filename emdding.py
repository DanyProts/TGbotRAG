import plotly.express as px
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import psycopg2
import csv
import plotly.io as pio
import io
from rapidfuzz import fuzz, process

class CustomDataset(Dataset):

    def __init__(self, X):
        self.text = X

    def tokenize(self, text):
        return tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=150)

    def __len__(self):
        return self.text.shape[0]

    def __getitem__(self, index):
        output = self.text[index]
        output = self.tokenize(output)
        return {k: v.reshape(-1) for k, v in output.items()}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_embedding(request, model, tokenizer, device, max_length=512):
    """
    Преобразует текст запроса в эмбеддинг с помощью модели BERT.

    :param request: Текст запроса
    :param model: Предобученная модель BERT
    :param tokenizer: Токенизатор BERT
    :param device: Устройство (CPU/GPU)
    :param max_length: Максимальная длина токенов
    :return: Эмбеддинг запроса
    """
    model.to(device)
    model.eval()

    inputs = tokenizer(
        request,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = mean_pooling(outputs, inputs['attention_mask'])

    return embedding.cpu().numpy()

def find_relevant_doc(embeddings, request, model, tokenizer, device, top_n=5):
    """
    Преобразует запрос в эмбеддинг и находит наиболее близкие по косинусному расстоянию эмбеддинги.
    Возвращает словарь, где ключи — индексы эмбеддингов, а значения — сами эмбеддинги.
    """
    request_embedding = get_embedding(request, model, tokenizer, device)
    similarities = cosine_similarity(request_embedding, embeddings)

    return similarities
def find_similar_embeddings(embeddings, request, model, tokenizer, device, top_n=5):
    """
    Преобразует запрос в эмбеддинг и находит наиболее близкие по косинусному расстоянию эмбеддинги.
    Возвращает словарь, где ключи — индексы эмбеддингов, а значения — сами эмбеддинги.
    """
    request_embedding = get_embedding(request, model, tokenizer, device)
    similarities = cosine_similarity(request_embedding, embeddings)


    closest_indices = similarities.argsort()[0][-top_n:][::-1]
    if closest_indices[0]>0.85:
        idx = closest_indices[0]
        return {idx: embeddings[idx]}
    return {idx: embeddings[idx] for idx in closest_indices}

def generate_embeddings(column, model, device):
    """
    Преобразует текстовые описания в эмбеддинги с использованием модели.
    """
    eval_ds = CustomDataset(column)
    eval_dataloader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    model.to(device)
    model.eval()
    embeddings = torch.Tensor().to(device)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc='Processing batches'):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            embeddings = torch.cat([embeddings, mean_pooling(outputs, batch['attention_mask'])])

    return embeddings.cpu().numpy()

def format_embeddings(embeddings):
    return [json.dumps(emb.tolist()) for emb in embeddings]  # Преобразуем в списки для сериализации

def find_keywords(req,model,device):
    sp = 'проходной, минимальный, балл, баллы, конкурс, проходной порог, проходной на бюджет, проходной на контракт, проходной на целевое, направление, специальность, факультет, кафедра, институт, программа, профиль, 2023, 2024, прошлый год, текущий год, за какой год, последние баллы, ЕГЭ, вступительные, дополнительные испытания, экзамен, предметы, профильный предмет, какие предметы, льготы, индивидуальные достижения, бонусные баллы, ГТО, медаль, олимпиада, доп. баллы, целевое обучение, бюджет, контракт, внебюджет, платное обучение, целевое, дистанционное, подача, когда подавать, сроки, документы, заявление, оригинал, копия'
    sp = sp.replace(', ',' ')
    global_word=np.array(sp.split())
    global_word=generate_embeddings(global_word,model,device)
    req = req.split()
    req_new=''
    req_con=''
    for i in req:
        a = find_relevant_doc(global_word,i,model, tokenizer, device)

        if any(j > 0.95 for j in a.flat):
            req_new+=i+' '
        else:
            req_con+=i+' '
    return req_new,req_con
def find_with_fuzz(request,col):
    relevant_info = -1
    mx = 0
    for i in range(len(col)):

        similarity = fuzz.ratio(request,col[i])

        if similarity > mx:
            mx = similarity
            relevant_info = i


    return relevant_info
def clear_quare(req_con):
    sp ='кто, что, какой, году, Какой, какая, какое, какие, чей, был, чья, чьё, чьи, сколько, который, где, куда, откуда, как, зачем, почему, отчего, ли, разве, неужели, а что если, как насчёт, правда ли, что за, в, на, под, над, перед, за, у, около, от, до, с, без, для, о, об, из, по, при, про, через, из-за, из-под, несмотря на, ввиду, в связи с, по поводу, по причине, по сравнению с, ради, вследствие, насчёт, согласно, наподобие, год, лет, году'
    sp = sp.replace(', ', ' ')
    global_word = np.array(sp.split())
    global_word = generate_embeddings(global_word, model, device)
    req = req_con.split()

    req_new=''

    for i in req:
        a = find_relevant_doc(global_word, i, model, tokenizer, device)

        if any(j > 0.95 for j in a.flat):
            continue
        else:
            req_new += i + ' '
    return req_new


MODEL_NAME = "DeepPavlov/rubert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10
PCA_COMPONENTS_HIGH = 15
PCA_COMPONENTS_LOW = 3
DISTANCE_THRESHOLD = 0.6
METRIC = 'cosine'
LINKAGE = 'average'
import csv
import json
import numpy as np
import ast
import pandas as pd
def example(request_user,model,device,tokenizer):
    data = pd.read_csv('emb_specialties.csv')
    request_user = clear_quare(request_user)

    req = request_user.split()

    sp = 'проходной, минимальный, балл, баллы, бюджет, платное, проходной порог, проходной на бюджет, направление, специальность, специальности, факультет, кафедра, институт, программа, профиль, прошлый год, ЕГЭ, вступительные, экзамен, предметы'
    global_word = np.array(sp.split(', '))
    global_word_emb = generate_embeddings(global_word, model, device)
    sp2 = 'проходной, минимальный, балл, баллы, бюджет, платное, порог, платка'
    global_ball = np.array(sp2.split(', '))
    global_ball_emb = generate_embeddings(global_ball, model, device)



    req_ball = []
    for i in range(len(req)):
        a = find_relevant_doc(global_ball_emb, req[i], model, tokenizer, device)


        if any(j > 0.95 for j in a.flat):
            req_ball.append(req[i])



    if len(req_ball) < 3:
        return 'Уточните пожалуйста ваш вопрос, я не могу найти на него ответ'
    if '2024' in request_user:
        req_ball.append('2024')
    elif '2023' in request_user:
        req_ball.append('2024')
    else:
        return 'Скажите за какой год вас интересует статистика'

    s = '_'.join(req_ball)
    req_new = ''
    req_con = ''
    for i in req:
        a = find_relevant_doc(global_word_emb, i, model, tokenizer, device)

        if any(j > 0.95 for j in a.flat) and i not in req_ball:
            req_new += i + ' '
        elif i not in req_ball:
            req_con += i + ' '

    req_new+=s
    name_col = []

    df = pd.read_csv('emb_columns_specialties.csv')
    df['emb_column'] = df['emb_column'].apply(ast.literal_eval)  # у колонки с эмбэддингами меняет тип данных на float
    embedding = np.array(df["emb_column"].tolist())
    for i in req_new.split():
        a = find_relevant_doc(embedding,i,model, tokenizer, device)

        if df.loc[a.argmax(), 'columns'] not in name_col:
            name_col.append(df.loc[a.argmax(), 'columns'])
    index = find_with_fuzz(req_con,data['направление'])
    result = ''
    for i in name_col:
        result+=f'{i}: {data.loc[index, i]};'
    return result






df = pd.read_csv('emb_fq.csv')

request_user = "Какой проходной балл на бюджет был у специальности Информационные системы и технологии в 2024 году?"
print(request_user)

df['emb_FAQ'] = df['emb_FAQ'].apply(ast.literal_eval) #у колонки с эмбэддингами меняет тип данных на float
embeddings = np.array(df["emb_FAQ"].tolist())
a = find_relevant_doc(embeddings,request_user,model, tokenizer, device)
functions = {
    74: example,
}

num = a.argmax()
args = [request_user,model,device,tokenizer]
result = functions[num](*args)
print(result)





