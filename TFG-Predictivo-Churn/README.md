# 📊 Optimización Económica de Campañas de Retención (*Churn*) mediante Machine Learning

Este repositorio contiene el desarrollo completo de mi **Trabajo de Fin de Grado (TFG)** para el Grado en Estadística Empresarial en la Universidad Miguel Hernández (Nota Media del expediente: 9,32). El proyecto aborda la problemática de la fuga de clientes (*churn*) en el sector de las telecomunicaciones desde una perspectiva dual: el modelado predictivo avanzado y la optimización financiera de los recursos de la empresa.

---

## 🎯 1. Contexto y Problema de Negocio
La pérdida de clientes es uno de los mayores desafíos financieros en los entornos empresariales actuales. Tradicionalmente, las organizaciones mitigan este problema lanzando campañas de fidelización masivas (ofreciendo descuentos). 

Durante el desarrollo de este proyecto, identifiqué una paradoja analítica crítica: el modelo estadístico que mejor detectaba las fugas reales en primera instancia (el Árbol de Decisión) era, al mismo tiempo, el **menos rentable para el negocio**. Al intentar capturar todas las bajas posibles buscando el máximo *Recall*, el modelo se volvía demasiado sensible, generando un volumen masivo de **falsos positivos** (clientes fieles clasificados erróneamente como riesgo). Activar la campaña bajo este criterio implicaba regalar descuentos innecesarios a clientes que no pensaban marcharse, destruyendo el margen comercial de la organización.

---

## 🛠️ 2. Stack Tecnológico & Metodología
* **Lenguaje:** Python 3.x
* **Librerías Clave:** `Scikit-Learn`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* **Algoritmos Evaluados:** Regresión Logística, Árbol de Decisión, *Random Forest*, SVM y $k$-NN.
* **Técnicas de Data Science:** Ingeniería de características (*feature engineering*), resolución de desbalanceo de clases y optimización de hiperparámetros mediante `GridSearchCV`.

---

## 🚀 3. Acciones y Enfoque Técnico

Para solucionar el problema de los falsos positivos y la rigidez analítica, la estrategia se dividió en dos fases integradas:

1. **Modelado Predictivo Continuo:** Se sustituyó la rigidez del Árbol de Decisión por un modelo de **Random Forest** calibrado mediante optimización de hiperparámetros (`n_estimators=50`, `max_depth=5`, `min_samples_split=0.10`). El algoritmo optó por una arquitectura compacta y una poda estricta para mitigar el ruido estadístico y evitar el sobreajuste, alcanzando un *Recall* del **80.79%**. Este comité de árboles permitió generar estimaciones de probabilidad fluidas en lugar de clasificaciones binarias bruscas, facilitando la calibración del umbral comercial.
2. **Simulación Económica Conjunta:** Comprendí que para resolver el problema no bastaba con encontrar el porcentaje de descuento óptimo a aplicar en la campaña. Diseñé una simulación financiera basada en curvas de elasticidad exponencial para identificar simultáneamente el incentivo ideal y el **umbral de riesgo óptimo**, determinando matemáticamente que el punto de corte óptimo se situaba en el **55%**.

---

## 📈 4. Impacto y Resultados Financieros

El ajuste fino del umbral de riesgo actuó como un filtro financiero definitivo sobre la campaña, logrando los siguientes hitos basados en la simulación económica conjunta:

* **Reducción de Falsos Positivos:** El ajuste personalizado del umbral de decisión en la Regresión Logística y el Random Forest sirvió para restringir la concesión de ofertas. Al volver los criterios de selección más estrictos, se evitó otorgar descuentos innecesarios a clientes con un comportamiento fiel, optimizando la asignación presupuestaria de la campaña de retención.
* **Optimización del Retorno Financiero:** La simulación demostró la alta rentabilidad de aplicar un enfoque de negocio avanzado basado en la sintonización conjunta del incentivo y el nivel de riesgo. Al generar estimaciones de probabilidad fluidas y continuas en lugar de clasificaciones rígidas, la **Regresión Logística** y el **Random Forest** permitieron calibrar de forma óptima la campaña global. Esto facilitó que la simulación determinase la combinación exacta de descuento e incentivo idóneos para la estrategia de retención, protegiendo el margen de ingresos y rescatando de forma eficiente el capital en riesgo.
* **Paradoja del Recall (Conclusión Estratégica):** El análisis financiero destapó una desconexión crítica entre las métricas matemáticas tradicionales y la rentabilidad real del negocio. A pesar de que el Árbol de Decisión individual lideraba en la fase estadística por su alta capacidad para capturar fugas (*Recall*), su rigidez estructural generó una saturación de **falsos positivos** (clientes fieles clasificados erróneamente en riesgo). Al otorgar descuentos innecesarios a usuarios que no planeaban marcharse, este modelo demostró ser la opción menos rentable para la empresa, evidenciando que maximizar la métrica matemática pura puede destruir valor comercial si no se calibra económicamente.
* **Conclusión de Negocio:** Este proyecto demuestra que en el entorno empresarial real el éxito no lo define el algoritmo con la métrica teórica pura más alta, sino aquel que posee la flexibilidad probabilística necesaria para calibrarse en función de la rentabilidad y el retorno financiero neto del negocio.
---

## 📂 5. Estructura del Repositorio

* `/data`: Contiene el conjunto de datos histórico original de 7.043 registros utilizado en el estudio.
* `/scripts`: Script de Python con las fases secuenciales de preprocesamiento, análisis exploratorio (EDA), entrenamiento y simulación económica.
* `/documentos`: Memoria final del Trabajo de Fin de Grado en formato PDF (información confidencial o personal anonimizada).
---
_Proyecto desarrollado por **Yuwei Guan** - Estudiante de último curso en Estadística Empresarial._
_Contacto: [LinkedIn](https://www.linkedin.com/in/yuwei-guan-data)_