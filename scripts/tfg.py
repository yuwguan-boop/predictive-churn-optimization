# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 18:35:47 2026

@author: Administrador
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimpy import skim
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import time
from sklearn.tree import plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
import math
from sklearn.pipeline import Pipeline

# na_values=[' '] ---> Considera el espacio en blanco como un valor nulo, para que las variables numericas que esten en blanco no los considere como variable categorica
df = pd.read_csv('C:\\Users\\Administrador\\OneDrive - UMH\\TFG\\BD_Churn.csv', na_values=[' '])

skim(df)

# .T transpone la tabla para que se vea mejor (Filas <-> Columnas)
print("--- RESUMEN NUMÉRICO ---")
print(df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

print("\n--- TIPOS DE DATOS ---")
print(df.info()) 

print("\n--- RESUMEN CATEGÓRICO ---")
print(df.describe(include=['object']).T)

# =============================================================================
# --- PASO PREVIO: Comprobar quiénes son los nulos ---
# =============================================================================
nulos_df = df[df['TotalCharges'].isnull()][['customerID', 'tenure', 'TotalCharges']]
print("--- CLIENTES CON TOTALCHARGES NULO ---")
print(nulos_df)
# Vemos que la columna tenure salen ceros -->
# los valores nulos en TotalCharges corresponden a clientes nuevos sin facturación histórica

# =============================================================================
# # --- 5.5. LIMPIEZA DE DATOS ---
# =============================================================================
# Hacemos una copia de la base de datos
df_clean = df.copy()

# 1. Corregir SeniorCitizen: De 0/1 numérico a Categórico (No/Sí)
df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].replace({0: 'No', 1: 'Si'})

# 2. Convertir variables de texto (object) a categóricas (equivalente a factor en R)
cols_categoricas = df_clean.select_dtypes(include=['object']).columns
for col in cols_categoricas:
    df_clean[col] = df_clean[col].astype('category')

# 3. Tratamiento de Nulos: Eliminamos las 11 filas
df_clean = df_clean.dropna(subset=['TotalCharges'])

# 4. Eliminamos el ID del cliente (ya no aporta valor predictivo)
df_clean = df_clean.drop(columns=['customerID'])

# 5. Reajuste del Índice (Reset Index): Eliminamos el índice antiguo con saltos
df_clean = df_clean.reset_index(drop=True)

# 6. Convertir Churn a numérico para cálculos (Yes=1, No=0)
df_clean['Churn_Num'] = (df_clean['Churn'] == 'Yes').astype(int)
# Churn_Num es numerica, y Churn es categorica

print("\n--- ESTRUCTURA FINAL DEL DATASET LIMPIO ---")
df_clean.info()

# =============================================================================
# --- DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (CHURN) ---
# =============================================================================
plt.figure(figsize=(6, 6))
counts = df['Churn'].value_counts()
colors = ['#BDC3C7', '#E74C3C'] 
plt.pie(counts, labels=['No (Fieles)', 'Yes (Fugas)'], autopct='%1.1f%%', 
        colors=colors, startangle=90, explode=(0, 0.1), shadow=True)
plt.title('Figura 5.1. Distribución de la Tasa de Abandono (Desbalanceo)', fontsize=14)
plt.show()

# =============================================================================
# --- 5.6.1. VARIABLES CATEGÓRICAS vs CHURN---
# =============================================================================
# Definimos una función para no repetir código y que los gráficos sean idénticos
def plot_proportional_churn(df, columns, titles, layout, figsize):
    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
    axes = axes.flatten()
    
    colors = ['#BDC3C7', '#E74C3C'] 

    for i, col in enumerate(columns):
        # Creamos la tabla de frecuencias normalizada (proporciones)
        cross_tab = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
        
        # Graficamos
        cross_tab.plot(kind='bar', stacked=True, ax=axes[i], color=colors, legend=False)
        
        axes[i].set_title(f'Tasa de Abandono por {titles[i]}', fontweight='bold')
        axes[i].set_ylabel('Proporción (%)')
        axes[i].set_xlabel('')
        axes[i].set_ylim(0, 110) # Espacio para el título
        
        # Añadir etiquetas de porcentaje dentro de las barras 
        for n, x in enumerate([*cross_tab.index.values]):
            for (proportion, y_loc) in zip(cross_tab.loc[x], cross_tab.loc[x].cumsum()):
                if proportion > 5: # Solo poner el texto si el trozo es grande
                    axes[i].text(x=n, y=(y_loc - proportion/2), 
                                 s=f'{np.round(proportion, 1)}%', 
                                 color="white", fontsize=9, fontweight="bold", ha="center")

    # Añadir una sola leyenda para toda la figura
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Churn', loc='upper right', bbox_to_anchor=(1, 0.95))
    
    plt.tight_layout()
    return fig

# --- GENERAR ILUSTRACIÓN A: CORE Y FACTURACIÓN ---
cols_A = ['Contract', 'InternetService', 'PaymentMethod', 'PaperlessBilling']
titles_A = ['Tipo de Contrato', 'Servicio Internet', 'Método Pago', 'Factura Electrónica']
fig_A = plot_proportional_churn(df_clean, cols_A, titles_A, (2, 2), (14, 10))
fig_A.savefig('proporcional_core.png', dpi=300)

# --- GENERAR ILUSTRACIÓN B: SERVICIOS VALOR AÑADIDO ---
cols_B = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection']
titles_B = ['Seguridad Online', 'Soporte Técnico', 'Copia Seguridad', 'Prot. Dispositivo']
fig_B = plot_proportional_churn(df_clean, cols_B, titles_B, (2, 2), (14, 10))
fig_B.savefig('proporcional_servicios.png', dpi=300)

# --- GENERAR ILUSTRACIÓN C: DEMOGRÁFICAS Y OTROS ---
cols_C = ['SeniorCitizen', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'StreamingTV', 'StreamingMovies']
titles_C = ['Tercera Edad', 'Género', 'Pareja', 'Dependientes', 'Servicio Tel.', 'Líneas Mult.', 'Streaming TV', 'Streaming Pelis']
fig_C = plot_proportional_churn(df_clean, cols_C, titles_C, (4, 2), (14, 20))
fig_C.savefig('proporcional_demograficas.png', dpi=300)
    
# =============================================================================
# --- 5.6.2. VARIABLES NUMÉRICAS vs CHURN---
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 1. Antigüedad vs Churn
sns.boxplot(data=df_clean, x='Churn', y='tenure', palette='Pastel1', ax=axes[0])
axes[0].set_title('Distribución de Antigüedad según Abandono', fontweight='bold')
axes[0].set_xlabel('¿Abandona? (Churn)')
axes[0].set_ylabel('Antigüedad (Meses)')

# 2. Cargos Mensuales vs Churn
sns.boxplot(data=df_clean, x='Churn', y='MonthlyCharges', palette='Pastel1', ax=axes[1])
axes[1].set_title('Cargos Mensuales según Abandono', fontweight='bold')
axes[1].set_xlabel('¿Abandona? (Churn)')
axes[1].set_ylabel('Cargo Mensual ($)')

plt.tight_layout()
fig.savefig('boxplots_numericas.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# --- 5.6.3. MATRIZ DE CORRELACIÓN ---
# =============================================================================
# Seleccionamos solo las variables numéricas para la correlación
columnas_num = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn_Num']
matriz_corr = df_clean[columnas_num].corr()

plt.figure(figsize=(8, 6))
#el mapa de calor (Heatmap)
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación de Variables Numéricas', fontweight='bold', pad=15)

plt.tight_layout()
plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# --- 5.7.1. - 5.7.7 AMPLIACIÓN Y BALANCEO DE MODELOS (BASE VS BALANCED) ---
# =============================================================================

# 1. PREPROCESAMIENTO PARA MACHINE LEARNING
# Eliminamos la variable categórica original 'Churn' (usaremos la numérica 'Churn_Num')
# y eliminamos 'TotalCharges' por la multicolinealidad detectada
df_ml = df_clean.drop(columns=['TotalCharges', 'Churn'])

# Convertimos todas las variables categóricas a variables ficticias (One-Hot Encoding)
# drop_first=True evita crear columnas redundantes
df_ml = pd.get_dummies(df_ml, drop_first=True)

# 2. FUNCIÓN DE VALIDACIÓN CRUZADA
def validacion_cruzada(df, nombre_target, n_splits=5, random_state=42):
    X = df.drop(columns=[nombre_target])
    y = df[nombre_target]

    skf = StratifiedKFold(  #La palabra Stratified es clave-->asegura que, si en el dataset global hay un 26% de fugas, en cada uno de los 5 trozos que corte haya exactamente un 26% de fugas.
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train_list.append(X.iloc[train_idx].copy())
        X_test_list.append(X.iloc[test_idx].copy())
        y_train_list.append(y.iloc[train_idx].copy())
        y_test_list.append(y.iloc[test_idx].copy())

    return X_train_list, X_test_list, y_train_list, y_test_list

# 3. ENTRENAMIENTO SVM CON ESCALADO DE VARIABLES
n_splits = 5
# Usamos 'Churn_Num' que es nuestra variable objetivo
X_train_prov, X_test_prov, y_train_prov, y_test_prov = validacion_cruzada(df_ml, 'Churn_Num', n_splits)

#Definimos un diccionario con todos los modelos que queremos probar
modelos = {
    'LogReg_Base': LogisticRegression(random_state=42, max_iter=1000), 
    'KNN_Base': KNeighborsClassifier(),
    'SVM_Base': SVC(random_state=42),
    'DecisionTree_Base': DecisionTreeClassifier(random_state=42),
    'RandomForest_Base': RandomForestClassifier(random_state=42),
    'LogReg_Balanced': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'SVM_Balanced': SVC(class_weight='balanced', random_state=42),
    'DecisionTree_Balanced': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'RandomForest_Balanced': RandomForestClassifier(class_weight='balanced', random_state=42)
}

# Diccionario para guardar las matrices de confusión de cada modelo
matrices_confusion = {nombre: [] for nombre in modelos.keys()}

print(f"Entrenando {len(modelos)} modelos con validación cruzada ({n_splits} folds)...")

# Bucle principal de la validación cruzada
for l in range(n_splits):
    X_train = X_train_prov[l]
    X_test = X_test_prov[l]
    y_train = y_train_prov[l]
    y_test = y_test_prov[l]

    # --- Escalado de variables ---
    # Nota: Los árboles no necesitan escalado, pero aplicarlo no les afecta negativamente 
    # y nos permite mantener un pipeline de datos unificado y justo para comparar.
    scaler = StandardScaler()
    
    ## Calcula las reglas con el entrenamiento y lo escala
    X_train_scaled = scaler.fit_transform(X_train)
    ## Transforma el set de prueba usando las reglas del entrenamiento (EVITA hacer trampa/DATA LEAKAGE)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenamos cada modelo en este fold
    for nombre, clf in modelos.items():
        clf.fit(X_train_scaled, y_train)
        y_test_pred = clf.predict(X_test_scaled)
        
        # Guardamos la matriz
        cm = confusion_matrix(y_test, y_test_pred)
        matrices_confusion[nombre].append(cm)

# 4. CÁLCULO DE MATRICES MEDIAS Y GENERACIÓN DE GRÁFICOS
for nombre, lista_matrices in matrices_confusion.items():
    # Calculamos la media de las 5 particiones para este modelo
    matriz_media = np.mean(lista_matrices, axis=0)
    
    # Visualización
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_media, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=['Se Queda (0)', 'Abandona (1)'], 
                yticklabels=['Se Queda (0)', 'Abandona (1)'],
                linewidths=1, linecolor='black')

    plt.title(f'Matriz de Confusión Media\n({nombre.replace("_", " ")} | Clasificación)', 
              fontweight='bold', pad=15)
    plt.ylabel('Valor Real (Etiqueta)', fontweight='bold')
    plt.xlabel('Clasificación del Modelo', fontweight='bold')
    plt.tight_layout()
    
    plt.show()

print("--- FIN FASE BASE VS BALANCED ---")

# =============================================================================
# --- 5.7.8. - 5.7.10. DETECCIÓN DE SOBREAJUSTE Y AJUSTE DE HIPERPARÁMETROS ---
# =============================================================================
print("\n--- INICIANDO FASE DETECCIÓN DE SOBREAJUSTE (TRAIN VS TEST) ---")

n_splits = 5
X_train_prov, X_test_prov, y_train_prov, y_test_prov = validacion_cruzada(df_ml, 'Churn_Num', n_splits)

# 1. ENTRENAMIENTO: TRAIN vs TEST Y NUEVOS PARÁMETROS
modelos_sobreajuste = {
    # --- Árboles de Decisión ---
    'DT_Defecto': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'DT_Split_1%': DecisionTreeClassifier(class_weight='balanced', min_samples_split=0.01, random_state=42),
    'DT_Split_5%': DecisionTreeClassifier(class_weight='balanced', min_samples_split=0.05, random_state=42),
    'DT_Split_10%': DecisionTreeClassifier(class_weight='balanced', min_samples_split=0.10, random_state=42),
    
    # --- Support Vector Machine (SVM) ---
    'SVC_Defecto': SVC(class_weight='balanced', random_state=42), 
    'SVC_C_0.5': SVC(class_weight='balanced', C=0.5, random_state=42),
    'SVC_C_2.0': SVC(class_weight='balanced', C=2.0, random_state=42),
    'SVC_Linear': SVC(class_weight='balanced', kernel='linear', random_state=42)
}

matrices_train = {nombre: [] for nombre in modelos_sobreajuste.keys()}
matrices_test = {nombre: [] for nombre in modelos_sobreajuste.keys()}

print(f"\nEntrenando {len(modelos_sobreajuste)} modelos evaluando Train y Test...")

for l in range(n_splits):
    X_train = X_train_prov[l]
    X_test = X_test_prov[l]
    y_train = y_train_prov[l]
    y_test = y_test_prov[l]

    # --- Escalado ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for nombre, clf in modelos_sobreajuste.items():
        clf.fit(X_train_scaled, y_train)
        
        # Clasificación en TRAIN
        y_train_clasif = clf.predict(X_train_scaled)
        cm_train = confusion_matrix(y_train, y_train_clasif)
        matrices_train[nombre].append(cm_train)
        
        # Clasificación en TEST
        y_test_clasif = clf.predict(X_test_scaled)
        cm_test = confusion_matrix(y_test, y_test_clasif)
        matrices_test[nombre].append(cm_test)

# 2. VISUALIZACIÓN
for nombre in modelos_sobreajuste.keys():
    matriz_train_media = np.mean(matrices_train[nombre], axis=0)
    matriz_test_media = np.mean(matrices_test[nombre], axis=0)
    
    # Si es el DT por defecto, mostramos la comparativa Train vs Test
    if nombre == 'DT_Defecto':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico 1: TRAIN
        sns.heatmap(matriz_train_media, annot=True, fmt=".1f", cmap="Greens",
                    xticklabels=['Se Queda (0)', 'Abandona (1)'], 
                    yticklabels=['Se Queda (0)', 'Abandona (1)'],
                    linewidths=1, linecolor='black', ax=axes[0])
        axes[0].set_title(f'TRAIN (Aprendizaje)\n{nombre.replace("_", " ")}', fontweight='bold')
        axes[0].set_ylabel('Valor Real', fontweight='bold')
        axes[0].set_xlabel('Clasificación', fontweight='bold')

        # Gráfico 2: TEST
        sns.heatmap(matriz_test_media, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=['Se Queda (0)', 'Abandona (1)'], 
                    yticklabels=['Se Queda (0)', 'Abandona (1)'],
                    linewidths=1, linecolor='black', ax=axes[1])
        axes[1].set_title(f'TEST (Examen Real)\n{nombre.replace("_", " ")}', fontweight='bold')
        axes[1].set_ylabel('') 
        axes[1].set_xlabel('Clasificación', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
    # Para todos los demás, mostramos SOLO TEST
    else:
        plt.figure(figsize=(7, 5))
        sns.heatmap(matriz_test_media, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=['Se Queda (0)', 'Abandona (1)'], 
                    yticklabels=['Se Queda (0)', 'Abandona (1)'],
                    linewidths=1, linecolor='black')
        plt.title(f'Matriz de Confusión en Test\n{nombre.replace("_", " ")}', fontweight='bold')
        plt.ylabel('Valor Real', fontweight='bold')
        plt.xlabel('Clasificación', fontweight='bold')
        plt.tight_layout()
        plt.show()


# =============================================================================
# --- 5.7.11. OPTIMIZACIÓN FINAL DE HIPERPARÁMETROS (GRID SEARCH CON PIPELINE) ---
# =============================================================================
# Usamos la libreria Pipeline para que el escalado se haga de forma 
# estricta e independiente en cada fold de la validación cruzada. Antes se ha hecho esto manualmente en lugar de usar Pipeline
X = df_ml.drop(columns=['Churn_Num'])
y = df_ml['Churn_Num']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Al usar un Pipeline, hay que poner el prefijo del nombre del modelo seguido de dos guiones bajos '__'
param_grid_lr = {
    'modelo__penalty': ['l1', 'l2'], 
    'modelo__C': [0.01, 0.1, 1.0, 10.0], 
    'modelo__solver': ['liblinear']
}

param_grid_dt = {
    'modelo__min_samples_split': [0.01, 0.02, 0.05, 0.10, 0.15], 
    'modelo__max_depth': [None, 3, 5, 10, 15], 
    'modelo__criterion': ['gini', 'entropy']
}

param_grid_rf = {
    'modelo__n_estimators': [50, 100, 200], 
    'modelo__min_samples_split': [0.02, 0.05, 0.10], 
    'modelo__max_depth': [None, 5, 10]
}

param_grid_svm = {
    'modelo__C': [0.1, 0.5, 1.0, 2.0, 5.0], 
    'modelo__kernel': ['linear', 'rbf'], 
    'modelo__gamma': ['scale', 'auto']
}

# --- 1. OPTIMIZACIÓN DE REGRESIÓN LOGÍSTICA ---
print("Optimizando Regresión Logística (Sin Data Leakage)...")
pipeline_lr = Pipeline([
    ('escalador', StandardScaler()),
    ('modelo', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])
grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=skf, scoring='recall', n_jobs=-1)
grid_lr.fit(X, y) # <--- Importante: Pasamos 'X' original, NO 'X_scaled'

# --- 2. OPTIMIZACIÓN DE ÁRBOL DE DECISIÓN ---
print("Optimizando Árbol de Decisión (Sin Data Leakage)...")
pipeline_dt = Pipeline([
    ('escalador', StandardScaler()),
    ('modelo', DecisionTreeClassifier(class_weight='balanced', random_state=42))
])
grid_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=skf, scoring='recall', n_jobs=-1)
grid_dt.fit(X, y)

# --- 3. OPTIMIZACIÓN DE RANDOM FOREST ---
print("Optimizando Random Forest (Sin Data Leakage)...")
pipeline_rf = Pipeline([
    ('escalador', StandardScaler()),
    ('modelo', RandomForestClassifier(class_weight='balanced', random_state=42))
])
grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=skf, scoring='recall', n_jobs=-1)
grid_rf.fit(X, y)

# --- 4. OPTIMIZACIÓN DE SVM ---
print("Optimizando SVM (Sin Data Leakage)...")
pipeline_svm = Pipeline([
    ('escalador', StandardScaler()),
    ('modelo', SVC(class_weight='balanced', random_state=42))
])
grid_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=skf, scoring='recall', n_jobs=-1)
grid_svm.fit(X, y)

print("\n")
print("MODELOS GANADORES Y SUS MEJORES PARÁMETROS")

print(f"\n1. REGRESIÓN LOGÍSTICA ÓPTIMA:")
print(f"Mejores parámetros: {grid_lr.best_params_}")
print(f"Mejor puntuación (Recall): {grid_lr.best_score_:.4f}")

print(f"\n2. ÁRBOL DE DECISIÓN ÓPTIMO:")
print(f"Mejores parámetros: {grid_dt.best_params_}")
print(f"Mejor puntuación (Recall): {grid_dt.best_score_:.4f}")

print(f"\n3. RANDOM FOREST ÓPTIMO:")
print(f"Mejores parámetros: {grid_rf.best_params_}")
print(f"Mejor puntuación (Recall): {grid_rf.best_score_:.4f}")

print(f"\n4. SVM ÓPTIMO:")
print(f"Mejores parámetros: {grid_svm.best_params_}")
print(f"Mejor puntuación (Recall): {grid_svm.best_score_:.4f}")

# -----------------------------------------------------------------------------
# --- VISUALIZACIÓN DE MATRICES DE CONFUSIÓN DE LOS MODELOS ÓPTIMOS ---
# ----------------------------------------------------------------------------

#diccionario con los mejores estimadores que ya encontró el GridSearchCV
mejores_modelos = {
    'Regresión Logística Óptima': grid_lr.best_estimator_,
    'Árbol de Decisión Óptimo': grid_dt.best_estimator_,
    'Random Forest Óptimo': grid_rf.best_estimator_,
    'SVM Óptimo': grid_svm.best_estimator_
}

for nombre, modelo in mejores_modelos.items():
    # Usamos cross_val_predict para obtener predicciones honestas (fuera de la muestra)
    # aplicando exactamente la misma validación cruzada estratificada (skf)
    y_pred_cv = cross_val_predict(modelo, X_scaled, y, cv=skf)
    
    # Generamos la matriz de confusión global
    cm_optima = confusion_matrix(y, y_pred_cv)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_optima, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Se Queda (0)', 'Abandona (1)'], 
                yticklabels=['Se Queda (0)', 'Abandona (1)'],
                linewidths=1, linecolor='black')

    plt.title(f'Matriz de Confusión\n({nombre})', fontweight='bold', pad=15)
    plt.ylabel('Valor Real (Etiqueta)', fontweight='bold')
    plt.xlabel('Clasificación del Modelo', fontweight='bold')
    plt.tight_layout()
    plt.show()

# =============================================================================
# --- 5.7.12. EXTRACCIÓN E INTERPRETACIÓN DE ODDS RATIOS (OR) ---
# ============================================================================= 

modelo_lr_optimo = mejores_modelos['Regresión Logística Óptima']

#Extraemos los coeficientes beta (son los pesos que el modelo asignó a cada variable)
# Buscamos el paso 'modelo' dentro del pipeline y de ahí extraemos los coeficientes beta
coeficientes_lr = modelo_lr_optimo.named_steps['modelo'].coef_[0]

# Aplicamos la exponencial (e^beta) para transformarlos en Odds Ratios (OR)
odds_ratios_lr = np.exp(coeficientes_lr)

# DataFrame ordenado para visualizar los impactos reales
tabla_or = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente (Beta)': coeficientes_lr,
    'Odds Ratio (OR)': odds_ratios_lr
})

# Clasificamos el efecto según las condiciones teóricas (OR > 1 o OR < 1)
tabla_or['Efecto sobre el Churn'] = np.where(tabla_or['Odds Ratio (OR)'] > 1, 
                                             'Riesgo (Incrementa Fuga)', 
                                             'Protector (Disminuye Fuga)')

# Ordenamos de mayor a menor riesgo para identificar los disparadores principales
tabla_or = tabla_or.sort_values(by='Odds Ratio (OR)', ascending=False)

print("\n RESULTADOS FINALES DE EXPLICABILIDAD (REGRESION LOGISTICA)")
print(tabla_or.to_string(index=False, formatters={
    'Coeficiente (Beta)': '{:,.4f}'.format, 
    'Odds Ratio (OR)': '{:,.4f}'.format
}))


# =============================================================================
# --- 5.7.13. VISUALIZACIÓN DEL ÁRBOL ÓPTIMO ---
# =============================================================================
print("\n--- DIAGRAMA DEL ÁRBOL ÓPTIMO ---")

# extraemos el modelo ganador directamente del GridSearchCV
modelo_dt_optimo = grid_dt.best_estimator_

plt.figure(figsize=(20, 10))

plot_tree(modelo_dt_optimo, 
          filled=True, 
          feature_names=X.columns, #coge los nombres reales de las columnas de la base de datos
          class_names=['Se Queda (0)', 'Abandona (1)'], 
          rounded=True,   #cajas con bordes redondeados
          fontsize=10,
          proportion=False) # Muestra los clientes reales, no en porcentajes

plt.title('Diagrama del Árbol de Decisión Óptimo (max_depth=3)', fontsize=16, fontweight='bold')
plt.show()

# =============================================================================
# --- 5.7.14. IMPORTANCIA DE VARIABLES (RANDOM FOREST) ---
# =============================================================================
print("\n--- GRÁFICO DE IMPORTANCIA DE VARIABLES (RANDOM FOREST) ---")

#extraemos el modelo ganador del Random Forest directamente del GridSearch
modelo_rf_optimo = grid_rf.best_estimator_

# obtenemos la importancia de las variables
importancias = modelo_rf_optimo.feature_importances_
features = X.columns #nombres de las columnas

# ordenamos de mayor a menor importancia
indices = np.argsort(importancias)[::-1] 

#seleccionamos solo las 10 más importantes (Top 10)
top_n = 10
indices_top = indices[:top_n]

#invertimos el orden para el gráfico (para que la barra más larga salga arriba del todo)
indices_top_grafico = indices_top[::-1]

plt.figure(figsize=(10, 6))
plt.title("Top 10 Variables más Importantes para predecir el Abandono\n(Modelo Random Forest)", fontsize=14, fontweight='bold')

plt.barh(range(top_n), importancias[indices_top_grafico], color='#2E86C1', align='center', edgecolor='black')

#ponemos los nombres de las variables en el eje Y
plt.yticks(range(top_n), [features[i] for i in indices_top_grafico], fontsize=11)

plt.xlabel("Importancia Relativa (Reducción media de la impureza de Gini)", fontsize=12, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.7) 
plt.tight_layout()
plt.show()

# =============================================================================
# --- 5.7.15. OPTIMIZACIÓN DE K-NN MEDIANTE FEATURE SELECTION (TOP 3) ---
# =============================================================================
print("\n--- K-NN CON REDUCCIÓN DE DIMENSIONALIDAD (TOP 3 VARIABLES) ---")

# las 3 variables más importantes extraídas del "codo" del Random Forest
top_3_features = ['Contract_Two year', 'tenure', 'InternetService_Fiber optic']

#reducimos la base de datos (entrenamiento y prueba) quedándonos solo con estas 3 columnas
X_train_reduced = X_train[top_3_features]
X_test_reduced = X_test[top_3_features]

# ESCALADO DE DATOS (Crítico para k-NN)
# Como 'tenure' tiene valores hasta 72 y el resto son 0/1, debemos igualar su escala para que no distorsione el cálculo de distancias matemáticas del algoritmo.
scaler = StandardScaler()
X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
X_test_reduced_scaled = scaler.transform(X_test_reduced)

#hacemos GridSearchCV sobre el espacio reducido 
# 'weights': 'distance' suele ayudar en datos desbalanceados dando más peso a vecinos cercanos.
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    #'n_neighbors': list(range(3, 31, 2)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

print("\nBuscando los mejores hiperparámetros para k-NN reducido...")

#Usamos scoring='recall' para obligar al algoritmo a centrarse en detectar la clase 1 (fuga)
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=skf, scoring='recall', n_jobs=-1)
grid_knn.fit(X_train_reduced_scaled, y_train)

print(f"Mejores parámetros encontrados: {grid_knn.best_params_}")

knn_optimizado = grid_knn.best_estimator_

#predicciones
y_pred_knn_reduced = knn_optimizado.predict(X_test_reduced_scaled)

print("\n--- INFORME DE CLASIFICACIÓN: K-NN (BASE REDUCIDA) ---")
print(classification_report(y_test, y_pred_knn_reduced))

# nueva matriz de confusión
cm_knn_reduced = confusion_matrix(y_test, y_pred_knn_reduced)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn_reduced, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Se Queda (0)', 'Abandona (1)'], 
            yticklabels=['Se Queda (0)', 'Abandona (1)'],
            linewidths=1, linecolor='black')

plt.title('Matriz de Confusión\n(k-NN | Top 3 Variables)', 
          fontweight='bold', pad=15)
plt.ylabel('Valor Real (Etiqueta)', fontweight='bold')
plt.xlabel('Clasificación del Modelo', fontweight='bold')
plt.tight_layout()
plt.show()


# # =============================================================================
# # --- 5.8. SIMULACIÓN ECONÓMICA COMPLETA Y VISUALIZACIÓN ---
# # =============================================================================
# print("\n--- GENERANDO SIMULACIÓN ECONÓMICA Y GRÁFICOS ---")

# plt.figure(figsize=(12, 8))

# modelos_a_simular = [
#     ('Regresión Logística Óptima', 'Regresión Logística'),
#     ('Árbol de Decisión Óptimo', 'Árbol de Decisión'),
#     ('Random Forest Óptimo', 'Random Forest'),
#     ('SVM Óptimo', 'SVM')
# ]

# for clave_modelo, nombre_legible in modelos_a_simular:
#     modelo_negocio = mejores_modelos[clave_modelo]
#     y_pred_negocio = cross_val_predict(modelo_negocio, X_scaled, y, cv=skf)

#     # Cálculo de grupos
#     TP_mask = (y == 1) & (y_pred_negocio == 1)
#     FP_mask = (y == 0) & (y_pred_negocio == 1)
#     cargos_TP = df_clean.loc[TP_mask, 'MonthlyCharges'].sum()
#     cargos_FP = df_clean.loc[FP_mask, 'MonthlyCharges'].sum()

#     descuentos = [d / 100 for d in range(5, 101)]
#     impactos = []
    
#     mejor_desc_modelo = 0
#     max_impacto_modelo = -float('inf')

#     for d in descuentos:
#         # Probabilidad de éxito: a más descuento, más retención-->función exponencial
#         tasa_retencion = (1 - math.exp(-6 * d)) 
        
#         # Fórmula: (Ingreso salvado descontado) - (Pérdida por falsos positivos)
#         ingreso_recuperado = (cargos_TP * tasa_retencion) * (1 - d)
#         perdida_falsos_fieles = cargos_FP * d
#         impacto_neto = ingreso_recuperado - perdida_falsos_fieles
        
#         impactos.append(impacto_neto)
        
#         if impacto_neto > max_impacto_modelo:
#             max_impacto_modelo = impacto_neto
#             mejor_desc_modelo = int(d*100)

#     print(f"{nombre_legible}: Óptimo al {mejor_desc_modelo}% (Max: ${max_impacto_modelo:,.2f})")

#     plt.plot([d*100 for d in descuentos], impactos, label=f'{nombre_legible} (Máx al {mejor_desc_modelo}%)', lw=2)

# plt.axhline(0, color='black', linestyle='--', alpha=0.5) # Línea de "no perder dinero"
# plt.title('Impacto Económico Neto según % de Descuento', fontsize=14, fontweight='bold')
# plt.xlabel('% de Descuento aplicado', fontsize=12)
# plt.ylabel('Impacto Económico Mensual ($)', fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

# =============================================================================
# --- 5.8. SIMULACIÓN ECONÓMICA COMPLETA CON OPTIMIZACIÓN DE UMBRALES ---
# =============================================================================
print("\n--- GENERANDO SIMULACIÓN ECONÓMICA AVANZADA (DESCUENTO + UMBRAL) ---")

# Rango de parámetros a evaluar
descuentos = [d / 100 for d in range(5, 101)]         # Del 5% al 100%
umbrales = [u / 100 for u in range(40, 86)]           # Probabilidades del 40% al 85%

modelos_a_simular = [
    ('Regresión Logística Óptima', 'Regresión Logística'),
    ('Árbol de Decisión Óptimo', 'Árbol de Decisión'),
    ('Random Forest Óptimo', 'Random Forest'),
    ('SVM Óptimo', 'SVM')
]

plt.figure(figsize=(12, 8))

for clave_modelo, nombre_legible in modelos_a_simular:
    modelo_negocio = mejores_modelos[clave_modelo]
    
    # 1. Extraemos probabilidades honestas usando cross_val_predict (pasando el X original)
    try:
        y_probas = cross_val_predict(modelo_negocio, X, y, cv=skf, method='predict_proba')[:, 1]
    except AttributeError:
        # Para SVM si no se activó probability=True, usamos decision_function normalizada
        y_scores = cross_val_predict(modelo_negocio, X, y, cv=skf, method='decision_function')
        y_probas = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    max_impacto_modelo = -float('inf')
    mejor_desc_modelo = 0
    mejor_umbral_modelo = 0
    
    # Diccionario para almacenar la curva del mejor umbral encontrado para graficar
    mejor_curva_impactos = []

    # 2. Búsqueda en cuadrícula de negocio (Umbral vs Descuento)
    for u in umbrales:
        # Clasificamos según el umbral dinámico actual 'u'
        y_pred_dinamico = (y_probas >= u).astype(int)
        
        # Identificamos máscaras de Verdaderos y Falsos Positivos para este umbral
        TP_mask = (y == 1) & (y_pred_dinamico == 1)
        FP_mask = (y == 0) & (y_pred_dinamico == 1)
        
        cargos_TP = df_clean.loc[TP_mask, 'MonthlyCharges'].sum()
        cargos_FP = df_clean.loc[FP_mask, 'MonthlyCharges'].sum()
        
        impactos_este_umbral = []
        
        for d in descuentos:
            # Función de elasticidad exponencial de retención
            tasa_retencion = (1 - math.exp(-6 * d)) 
            
            # Impacto económico neto
            ingreso_recuperado = (cargos_TP * tasa_retencion) * (1 - d)
            perdida_falsos_fieles = cargos_FP * d
            impacto_neto = ingreso_recuperado - perdida_falsos_fieles
            
            impactos_este_umbral.append(impacto_neto)
            
            # Guardamos los parámetros si superan el máximo histórico del modelo
            if impacto_neto > max_impacto_modelo:
                max_impacto_modelo = impacto_neto
                mejor_desc_modelo = int(d * 100)
                mejor_umbral_modelo = int(u * 100)
                mejor_curva_impactos = impactos_este_umbral # Guardamos la trayectoria de este umbral

    print(f"{nombre_legible}: Óptimo con Umbral al {mejor_umbral_modelo}% y Descuento al {mejor_desc_modelo}% (Max Beneficio: ${max_impacto_modelo:,.2f})")

    # Pintamos la curva del mejor umbral óptimo hallado para este modelo
    plt.plot([d*100 for d in descuentos], mejor_curva_impactos, 
             label=f'{nombre_legible} (Umbral: {mejor_umbral_modelo}% | Máx al {mejor_desc_modelo}%)', lw=2)

# Configuración final del gráfico corporativo
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.title('Optimización Económica Neta: Curvas de Descuento Bajo Umbrales de Riesgo Óptimos', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('% de Descuento aplicado a la población seleccionada', fontsize=12)
plt.ylabel('Impacto Económico Mensual Neto ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='lower left')
plt.tight_layout()
plt.show()

# =============================================================================
# --- VISUALIZACIÓN DE LA CURVA ROC Y CÁLCULO DEL AUC ---
# =============================================================================
plt.figure(figsize=(10, 8))

# Iteramos sobre los mismos modelos que usamos para las matrices
for nombre, modelo in mejores_modelos.items():
    # Obtenemos las puntuaciones/probabilidades honestas (fuera de la muestra)
    try:
        # Intentamos obtener las probabilidades (funciona para LR, DT, RF)
        y_probas = cross_val_predict(modelo, X_scaled, y, cv=skf, method='predict_proba')
        y_scores = y_probas[:, 1] # Nos quedamos con la probabilidad de la clase 1 (Abandona)
    except AttributeError:
        # Si el modelo no soporta predict_proba (como el SVM por defecto), usamos la función de decisión
        y_scores = cross_val_predict(modelo, X_scaled, y, cv=skf, method='decision_function')
    
    # Calculamos la Tasa de Falsos Positivos y la Tasa de Verdaderos Positivos
    fpr, tpr, _ = roc_curve(y, y_scores)
    
    # Calculamos el Área Bajo la Curva (AUC)
    roc_auc = auc(fpr, tpr)
    
    # Dibujamos la curva del modelo actual, incluyendo su AUC en la leyenda
    plt.plot(fpr, tpr, lw=2, label=f'{nombre} (AUC = {roc_auc:.4f})')

# Dibujamos la línea de referencia (modelo puramente aleatorio, AUC = 0.5)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatorio (AUC = 0.5000)')

# Ajustes de diseño y etiquetas
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)', fontweight='bold')
plt.ylabel('Tasa de Verdaderos Positivos (TPR - Recall)', fontweight='bold')
plt.title('Comparativa de Curvas ROC (Modelos Optimizados)', fontweight='bold', pad=15)
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

