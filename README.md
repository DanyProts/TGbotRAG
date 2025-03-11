![image](https://github.com/user-attachments/assets/55322988-b2c3-41a0-80e4-20172bc5ffdc)
В этой части кода меняем только словарь, ключ: номер вопроса для обработки которого написана функция, значение: имя сделанной вами функции.

Также можете менять request_user, для теста функции


Написанные функции, должны принимать (запрос пользователя, модельку для ембэддингов, device, tokinazer), этого достаточно чтобы пользоваться всеми уже написанными вспомогательными функциями.


Перед там как сделать коммит проверьте что ваши функции, не конфликтуют с другими, получиться много строк но ничего страшного проект тоже большой


Что делают вспомогательные функции кратко описано в них и интуитивно понятно из названия, если есть вопросы задавайте


Кто  какие вопросы обрабатывает напишу в телеге

Функция для примера называется example:

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


Она действует в несколько этапов, для начала открываем таблицу, содержиться ответ на вопрос
Затем обрабатываем вопрос пользователя с помощью выделения ключевых слов и их фильтрацию. Для поиска колонок в которых содержаться ответы, в некоторых файлах нету колонки с эмбэддингами код скину.
Находим индекс строки в которой содержаться ответы,через fuzzy(индивидуально для рассмотренного мной вопроса)
Данные должны идти в ответ в следующем формате:

Запрос пользователя:Какой проходной балл на бюджет был у специальности Информационные системы и технологии в 2024 году?

Данные поступающие в модель:  направление: Информационные системы и технологии;проходной_балл_бюджет_2024: 283;

Всем удачи вопросы в лс)
