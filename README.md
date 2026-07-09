# 📊 Optimización Económica de Campañas de Retención (*Churn*) mediante Machine Learning

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Data Science](https://img.shields.io/badge/Data_Science-Data_Driven-🚀?style=for-the-badge&color=blue)
![Business Analytics](https://img.shields.io/badge/Business-ROI_Optimization-🟢?style=for-the-badge&color=brightgreen)

Este proyecto resuelve de extremo a extremo la problemática de la fuga de clientes (*churn*) en el sector de las telecomunicaciones. La aportación principal es demostrar la **Paradoja del Recall**: cómo el modelo con mejores métricas estadísticas teóricas (*Árbol de Decisión*) puede destruir margen comercial si no se calibra correctamente, y cómo un enfoque probabilístico continuo logra optimizar el **ROI real** de la empresa.

---

## 🎯 1. El Reto de Negocio: Evitar el "Gasto Inútil"

Las campañas tradicionales de retención lanzan descuentos masivos de forma indiscriminada. Esto genera un problema financiero crítico:
* **Falsos Positivos:** El modelo clasifica como "en riesgo" a un cliente que en realidad es fiel.
* **Impacto Financiero:** La empresa regala un descuento innecesario, canibalizando sus propios ingresos y destruyendo el margen comercial.

**Solución:** Este proyecto implementa una **Simulación Económica Bidimensional** que calibra en tiempo real tanto el incentivo comercial óptimo como el umbral de riesgo exacto para maximizar el beneficio neto.

---

## 📂 2. Datos y Perfil de Alto Riesgo (EDA)

Se auditó el dataset público **"Telco Customer Churn"** de IBM (7.043 registros). El análisis bivariado destapó las palancas que activan la fuga de usuarios:

* **Estructura Contractual:** El riesgo crítico reside en los contratos mes a mes (`Month-to-month`). Las permanencias de 1 y 2 años actúan como escudos de retención.
* **La Paradoja de la Fibra Óptica:** Los clientes con conectividad premium (Fibra) se fugan sustancialmente más que los de ADSL, sugiriendo una fuerte presión competitiva o incidencias de calidad.
* **Fidelización Temprana y Precio:** El 75% de las bajas se concentran antes de los 2.5 años de relación, disparándose el riesgo en tarifas mensuales elevadas (~$80).

> ⚠️ **Filtro Estadístico Crítico:** El mapa de correlación de Pearson detectó una **multicolinealidad crítica (0.83)** entre la antigüedad (`tenure`) y los cargos totales (`TotalCharges`). Para evitar coeficientes inestables y sobreajuste (*overfitting*), la variable `TotalCharges` fue eliminada del modelado.

---

## 🛠 3. Estrategia de Ingeniería de Datos

<details>
<summary>⚙️ Haz clic aquí para ver el Pipeline de Preprocesamiento Técnico</summary>

1. **Gestión Quirúrgica de Nulos:** Se aislaron 11 valores nulos en `TotalCharges`. Al cruzarlos, se descubrió que correspondían a usuarios con `tenure = 0` (nuevos clientes sin ciclo de facturación). Se eliminaron de forma segura por representar solo el 0.15% de la muestra.
2. **Preparación de Variables:** Transformación de la variable `SeniorCitizen` a factor categórico y vectorización de variables cualitativas mediante **One-Hot Encoding**.
3. **Validación Robusta (Anti-Leakage):** Implementación de **Validación Cruzada Estratificada (`StratifiedKFold`) con $k=5$** para mantener el desbalanceo real del dataset (26.5% de churn) en cada bloque.
4. **Escalado Localizado:** Aplicación de `StandardScaler` calculando medias y desviaciones típicas *únicamente* sobre los conjuntos de entrenamiento de cada iteración para evitar la filtración de datos (*data leakage*).
</details>

---

## 🚀 4. Modelado Predictivo y Sintonización (`GridSearchCV`)

Se entrenaron y optimizaron múltiples arquitecturas utilizando `GridSearchCV` enfocado en maximizar el **Recall** (capturar la mayor cantidad de bajas potenciales) y aplicando pesos penalizados (`class_weight='balanced'`).

* 🌲 **Árbol de Decisión (Máximo Recall):** Optimizado con una profundidad muy corta (`max_depth=3`). Evita por completo el sobreajuste, capturando el mayor volumen de positivos reales, pero su naturaleza binaria ofrece poca flexibilidad.
* 📈 **Regresión Logística:** Potenciada con regularización Lasso (`L1`). Actúa como un filtro interno reduciendo a cero el peso de las variables irrelevantes para concentrar el poder predictivo en los factores críticos.
* 🤖 **Random Forest:** Configurado como un ensamblado compacto (50 árboles) para neutralizar el ruido estático de la agregación estocástica.

En capacidad discriminante global (**Métrica AUC**), la **Regresión Logística** y el **Random Forest** lideraron la ordenación de la cartera, demostrando el mejor equilibrio entre detección y control de falsos positivos.

---

## 📊 5. Simulación Financiera: Traduciendo Métricas a Dinero

Para validar el proyecto ante la dirección general, se construyó un simulador monetario basado en dos premisas reales de la **econometría del marketing**:
1. **Elasticidad de la Retención:** La probabilidad de éxito de la oferta sigue una función de saturación exponencial no lineal según el descuento otorgado.
2. **Optimización Dinámica:** Se iteraron los umbrales de decisión (del 40% al 85%) y los niveles de descuento (del 5% al 100%) para evaluar el **Impacto Económico Mensual** mediante la siguiente ecuación de negocio:

$$\text{Impacto} = (\text{Cargos}_{TP} \cdot P_{\text{éxito}}) \cdot (1 - \text{desc}) - (\text{Cargos}_{FP} \cdot \text{desc})$$

> 💡 **¿Qué significa esta fórmula para el negocio?**
> * **El beneficio (Parte izquierda):** Suma los cargos mensuales de las bajas reales que logramos salvar $(\text{Cargos}_{TP})$, multiplicados por la probabilidad de que acepten la oferta $(P_{\text{éxito}})$ e ingresando ese dinero con el descuento ya aplicado $(1 - \text{desc})$.
> * **El coste (Parte derecha):** Resta el "dinero regalado" a los clientes que el modelo marcó como bajas por error pero que en realidad eran fieles ($\text{Cargos}_{FP}$), multiplicado por el descuento que les dimos ($\text{desc}$).

### 📈 Tabla de Rendimiento y Óptimo Económico

| Modelo Evaluado | Capacidad de Ordenación (AUC) | Umbral de Riesgo Óptimo | Descuento Óptimo a Ofrecer | Impacto Económico Mensual (Ecuación Superior) |
| :--- | :---: | :---: | :---: | :---: |
| **Regresión Logística** | 🔥 **Líder (0.839)** | **55%** | **19%** | 💰 **+$43,702.46** |
| **Random Forest** | **Excelente (0.835)** | **55%** | **20%** | 💰 **+$43,426.35** |
| **Support Vector Machine** | Robusto (0.818) | 51% | 18% | +$41,365.58 |
| **Árbol de Decisión** | Competitivo (0.791) | 48% | 17% | +$40,938.31 |

### 🔍 Conclusión Ejecutiva: La Paradoja del Recall

El proyecto demostró un valioso enfoque de negocio: **las mejores métricas estadísticas no siempre equivalen a la mayor rentabilidad**. 

* **La limitación del Árbol de Decisión:** Aunque matemáticamente era un modelo excelente con el *Recall* más alto, su estructura rígida de saltos binarios le impidió ajustar finamente su probabilidad, quedando estancado en un umbral del 48%. Al ser incapaz de filtrar con precisión, saturó la campaña con **Falsos Positivos**, regalando márgenes comerciales a clientes fieles que no pensaban marcharse.
* **El éxito de Regresión Logística y Random Forest:** Su distribución de probabilidad fluida actuó como un **"bisturí financiero"**. Permitió elevar el umbral de exigencia al **55%**. Al volverse más estrictos, descartaron a los clientes estables antes de activar la oferta, concentrando los recursos de marketing *únicamente* donde generaban un retorno neto positivo, rescatando más de **$43,000 mensuales**.

---

## 🏢 6. Plan de Acción Comercial Recomendado

1. **Blindaje de Clientes de Fibra Óptica:** Diseñar campañas proactivas de control de calidad y migrarlos de contratos mensuales a contratos anuales mediante descuentos estratégicos.
2. **Automatización de Pagos:** Incentivar la transición desde el cheque electrónico hacia la domiciliación bancaria para eliminar la fricción de la revisión manual mensual de las facturas.
3. **Estrategia Quirúrgica:** Configurar el software de marketing automatizado para activar ofertas exclusivamente a clientes con un **riesgo superior al 55%**, aplicando un descuento del **19% - 20%** para garantizar el máximo beneficio neto de la campaña.

---

## 📂 7. Estructura del Repositorio

* **📁 `data/`**: Contiene el conjunto de datos histórico original de 7.043 registros utilizado en el estudio.
* **📁 `scripts/`**:  Script de Python con las fases secuenciales de preprocesamiento, análisis exploratorio (EDA), entrenamiento y simulación económica.
* **📁 `documentos/`**: Memoria académica completa del TFG en formato PDF.

---
_Proyecto desarrollado por **Yuwei Guan** - Graduada en Estadística Empresarial._  
_Contacto y redes: [LinkedIn](https://www.linkedin.com/in/yuwei-guan-data) | [Portfolio de GitHub](https://github.com/yuwguan-boop)_
