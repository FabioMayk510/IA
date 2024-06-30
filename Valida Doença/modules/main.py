import numpy as np 
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from IPython.display import Image
from six import StringIO
import pydotplus

# Dados aleatorios, considere importar um csv com dados reais
data = {
    'Idade': [25, 30, 35, 42, 50, 22, 38, 41, 28, 36, 
              45, 52, 47, 33, 29, 40, 21, 48, 55, 39,
              31, 43, 37, 26, 34, 29, 49, 32, 27, 50,
              23, 54, 42, 31, 36, 28, 46, 53, 24, 45,
              44, 30, 35, 22, 39, 48, 29, 47, 25, 41],
    'Fadiga': ['Sim', 'Não', 'Sim', 'Não', 'Não', 'Sim', 'Não', 'Não', 'Sim', 'Sim', 
               'Não', 'Sim', 'Sim', 'Não', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não',
               'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não',
               'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim',
               'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim'],
    'Febre': ['Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Sim', 'Não',
              'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Sim', 'Não',
              'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim',
              'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim',
              'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não'],
    'FaltaDeAr': ['Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não',
                  'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim',
                  'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim',
                  'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim',
                  'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim'],
    'Doenca': ['Sim', 'Sim', 'Sim', 'Não', 'Sim', 'Sim', 'Sim', 'Não', 'Sim', 'Não',
               'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Sim', 'Não',
               'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim',
               'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não', 'Sim',
               'Não', 'Sim', 'Não', 'Sim', 'Não', 'Sim', 'Sim', 'Não', 'Sim', 'Não']
}

# Criar DataFrame
df = pd.DataFrame(data)

# Exibir os dados
#print(df)

# Features dentre os dados
x = pd.get_dummies(df[['Idade', 'Fadiga', 'Febre', 'FaltaDeAr']], drop_first=True)

# Target dos dados
y = df['Doenca']

# Dividir os dados entre dados de treino e dados de teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# Criar e treinar o modelo
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
model = tree.fit(x_train, y_train)

# Gerar img da arvore
dot_data = StringIO()
export_graphviz(model, out_file=dot_data, filled=True, feature_names=x.columns, class_names=['Não', 'Sim'], rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Salvar a imagem
graph.write_png("arvore.png")

# Exibir a imagem
Image(filename="arvore.png")

# Testar o modelo
y_predicoes = model.predict(x_test)

# Avaliar o modelo
print("Acurácia: ", accuracy_score(y_test, y_predicoes))
print(classification_report(y_test, y_predicoes))

# Função para gerar a Matriz de Confusão
def plot_confusion_matrix(cm, classes, normalize, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão sem normalização')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo Real')
    plt.xlabel('Rótulo Previsto')

# Plotar a matriz de confusão
matrix_confusao = confusion_matrix(y_test, y_predicoes)
plt.figure()
plot_confusion_matrix(matrix_confusao, normalize=False, classes=['Não', 'Sim'], title='Matriz de Confusão')

plt.show()