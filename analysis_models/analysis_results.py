import pandas as pd
import glob
import os
import pickle

my_path = os.path.abspath(os.path.dirname('evaluation'))
eval_dir = os.path.join(my_path, "../data/evaluation\\")

files = [f for f in glob.glob(eval_dir + "**/*.pickle", recursive=True)]

evals = []
for f in files:
    x = pickle.load(open(f, "rb"))
    for e in x:
        evals.append(e)

df = pd.DataFrame(evals)
df.columns = ['entries', 'vectorizer', 'alg', 'trait', 'score']

df['alg'] = df['alg'].replace({
    'svm': 'SVM',
    'RF': 'Random Forest',
    'forest': 'Random Forest',
    'logR': 'Logistic Regression',
    'logistic': 'Logistic Regression',
    'tree': 'Decision Tree'
})

temp = df.sort_values(by=['vectorizer', 'entries', 'score'], ascending=False)

pivot_table = temp.pivot_table(values='score', index='alg', columns='vectorizer', aggfunc='mean').reset_index()

excel_file = '../data/analysis_results.xlsx'
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    pivot_table.to_excel(writer, sheet_name='Сравнение моделей', index=False)

    workbook = writer.book
    worksheet = writer.sheets['Сравнение моделей']
    chart = workbook.add_chart({'type': 'column'})
    colors = ['#3978e8', '#85acf1']

    for col_num, vectorizer in enumerate(pivot_table.columns[1:], 1):
        chart.add_series({
            'name':       [f'Сравнение моделей', 0, col_num],
            'categories': ['Сравнение моделей', 1, 0, len(pivot_table), 0],
            'values':     ['Сравнение моделей', 1, col_num, len(pivot_table), col_num],
            'gap':        200,
            'fill':       {'color': colors[col_num - 1]}
        })

    chart.set_title({
        'name': 'Точность моделей при различных алгоритмах классификации машинного обучения',
        'name_font': {'size': 10, 'bold': True}
    })
    chart.set_x_axis({'name': 'Алгоритмы классификации'})
    chart.set_y_axis({
        'name': 'Точность классификации (%)',
        'name_font': {'size': 10},
        'major_gridlines': {'visible': True},
        'num_format': '0%'
    })
    chart.set_style(11)

    worksheet.insert_chart('G2', chart)
    pivot_table = temp.pivot_table(values='score', index=['vectorizer', 'alg'], columns=['trait'], aggfunc='mean')
    temp_pivot = pivot_table.reset_index()
    temp_pivot = temp_pivot.rename(columns={'vectorizer': 'Модель', 'alg': 'Алгоритм'})
    temp_pivot.to_excel(writer, sheet_name='Сравнение по характеристикам', index=False)
    # Разделяем таблицу на две подтаблицы для моделей BoW и GloVe
    temp_pivot_bow = temp_pivot[temp_pivot['Модель'] == 'BoW']
    temp_pivot_glove = temp_pivot[temp_pivot['Модель'] == 'GloVe']
    # Записываем данные для модели BoW
    temp_pivot_bow.to_excel(writer, sheet_name='BoW', index=False)
    # Создаем диаграмму для модели BoW
    workbook = writer.book
    worksheet_bow = writer.sheets['BoW']
    chart_bow = workbook.add_chart({'type': 'column'})

    for col_num, trait in enumerate(temp_pivot_bow.columns[2:], 2):
        chart_bow.add_series({
            'name': ['BoW', 0, col_num],
            'categories': ['BoW', 1, 1, len(temp_pivot_bow), 1],
            'values': ['BoW', 1, col_num, len(temp_pivot_bow), col_num],
            'gap': 200
        })

    chart_bow.set_title({
        'name': 'Точность определения личностных характеристик BoW',
        'name_font': {'size': 10, 'bold': True}
    })
    chart_bow.set_x_axis({'name': 'Алгоритмы классификации'})
    chart_bow.set_y_axis({
        'name': 'Точность классификации (%)',
        'name_font': {'size': 10},
        'major_gridlines': {'visible': True},
        'num_format': '0%'
    })
    chart_bow.set_style(11)
    worksheet_bow.insert_chart('G2', chart_bow)

    # Записываем данные для модели GloVe
    temp_pivot_glove.to_excel(writer, sheet_name='GloVe', index=False)
    # Создаем диаграмму для модели GloVe
    worksheet_glove = writer.sheets['GloVe']
    chart_glove = workbook.add_chart({'type': 'column'})

    for col_num, trait in enumerate(temp_pivot_glove.columns[2:], 2):
        chart_glove.add_series({
            'name': ['GloVe', 0, col_num],
            'categories': ['GloVe', 1, 1, len(temp_pivot_glove), 1],
            'values': ['GloVe', 1, col_num, len(temp_pivot_glove), col_num],
            'gap': 200
        })

    chart_glove.set_title({
        'name': 'Точность определения личностных характеристик GloVe',
        'name_font': {'size': 10, 'bold': True}
    })
    chart_glove.set_x_axis({'name': 'Алгоритмы классификации'})
    chart_glove.set_y_axis({
        'name': 'Точность классификации (%)',
        'name_font': {'size': 10},
        'major_gridlines': {'visible': True},
        'num_format': '0%'
    })
    chart_glove.set_style(11)
    worksheet_glove.insert_chart('G2', chart_glove)

temp.to_pickle('data/analysis_results.pickle')
