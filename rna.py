from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features.copy()
y = breast_cancer_wisconsin_original.data.targets

# Substituir valores ausentes
X['Bare_nuclei'] = np.where(X['Bare_nuclei'] == 0, X['Bare_nuclei'].median(), X['Bare_nuclei'])
X['Bare_nuclei'].fillna(X['Bare_nuclei'].median(), inplace=True)
print("Median after replacement:", X['Bare_nuclei'].median())

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o StandardScaler
scaler = StandardScaler()

# Ajustar o scaler aos dados de treinamento e transformar os dados
X_train_normalized = scaler.fit_transform(X_train)

# Transformar os dados de teste usando o scaler ajustado aos dados de treinamento
X_test_normalized = scaler.transform(X_test)

# Inicializar o LabelEncoder
label_encoder = LabelEncoder()

# Transformar os rótulos de treinamento
y_train_encoded = label_encoder.fit_transform(y_train.values.ravel())

# Transformar os rótulos de teste
y_test_encoded = label_encoder.transform(y_test.values.ravel())

# Definir modelos
models = {
    "Modelo 1": Sequential([
    Dense(64, activation='relu', input_shape=(X_train_normalized.shape[1],)),
    Dense(32, activation='sigmoid'),
    Dense(1, activation='sigmoid')
    ]),
    "Modelo 2": Sequential([
    Dense(128, activation='tanh', input_shape=(X_train_normalized.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='sigmoid'),
    Dense(1, activation='sigmoid')
    ]),
    "Modelo 3": Sequential([
    Dense(256, activation='leaky_relu', input_shape=(X_train_normalized.shape[1],)),
    Dense(128, activation='sigmoid'),
    Dense(1, activation='sigmoid')
    ]),
    "Modelo 4": Sequential([
    Dense(128, activation='tanh', input_shape=(X_train_normalized.shape[1],)),
    Dense(64, activation='leaky_relu'),
    Dense(32, activation='sigmoid'),
    Dense(1, activation='sigmoid')
    ])
}

results = {}

# Treinar e avaliar modelos
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}")
    
    # Compilar o modelo com uma taxa de aprendizado ajustada
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Treinar o modelo com 50 épocas
    history = model.fit(X_train_normalized, y_train_encoded, epochs=50, validation_split=0.2, verbose=0)

    # Avaliar o modelo no conjunto de teste
    test_accuracy = model.evaluate(X_test_normalized, y_test_encoded, verbose=0)
    print(f'{model_name} - Test Accuracy: {test_accuracy[1]}')

   # Armazenar os resultados
    results[model_name] = {
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'test_accuracy': test_accuracy[1]
    }
    # Plotar gráfico de perda e precisão durante o treinamento
    plt.figure(figsize=(12, 4))

    # Plotar gráfico de perda
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda treinamento')
    plt.plot(history.history['val_loss'], label='Perda validação')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    # Plotar gráfico de precisão
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.savefig(f'plotacu_{model_name}.png')
    plt.legend()

    plt.tight_layout()
    plt.show()

    y_pred_proba = model.predict(X_test_normalized)
    # Calcular e exibir a matriz de confusão
    y_pred = (y_pred_proba > 0.5).astype(int)
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Positivo', 'Negativo'], yticklabels=['Verdadeiro', 'Falso'])
    plt.xlabel('Predição')
    plt.ylabel('Atual')
    plt.title(f'{model_name} - Matriz Confusão')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.show()

# Inicializar as curvas ROC
plt.figure()

# Treinar e avaliar modelos
for model_name, model in models.items():
    print(f"Training and evaluating {model_name}")
    
    # Compilar o modelo com uma taxa de aprendizado ajustada
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Treinar o modelo com 50 épocas
    history = model.fit(X_train_normalized, y_train_encoded, epochs=50, validation_split=0.2, verbose=0)

    # Avaliar o modelo no conjunto de teste
    test_accuracy = model.evaluate(X_test_normalized, y_test_encoded, verbose=0)
    print(f'{model_name} - Test Accuracy: {test_accuracy[1]}')

    # Fazer previsões no conjunto de teste
    y_pred_proba = model.predict(X_test_normalized)

    # Calcular a curva ROC
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plotar a curva ROC no mesmo gráfico
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')

# Configurar e exibir o gráfico ROC
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Curva ROC para Todos os Modelos')
plt.legend(loc='lower right')
plt.savefig('curva_roc_todos_modelos.png')
plt.show()

for model_name, metrics in results.items():
    print(f"\n{model_name} Results:")
    print(f"Test Accuracy: {metrics['test_accuracy']}")
    print(f"Train Loss: {metrics['train_loss']}")
    print(f"Validation Loss: {metrics['val_loss']}")
    print(f"Train Accuracy: {metrics['train_accuracy']}")
    print(f"Validation Accuracy: {metrics['val_accuracy']}")
    print("\n")
