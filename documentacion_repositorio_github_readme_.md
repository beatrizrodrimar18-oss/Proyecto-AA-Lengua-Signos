# Clasificación de Lengua de Signos (ASL) mediante Redes Neuronales Convolucionales

**Autora:** Beatriz Rodríguez  
**Titulación:** Grado en Ingeniería Matemática (3.º Curso)  
**Asignatura:** Aprendizaje Automático — Universidad Francisco de Vitoria (UFV)  
**Estado del proyecto:** Entrega 2 (Primer Avance del Proyecto)

---

## 1. Descripción del Proyecto
Este repositorio contiene el desarrollo experimental del sistema de clasificación automática del alfabeto dactilológico estático (24 letras de la Lengua de Signos Americana, excluyendo *J* y *Z* por requerir movimiento) sobre el conjunto de datos **Sign Language MNIST**.

En este **Primer Avance (Entrega 2)** se incluye:
1. **Análisis Exploratorio de Datos (EDA):** Verificación de integridad, análisis de balanceo de clases ($N_{\text{train}} = 27.455$, $N_{\text{test}} = 7.172$) y detección de la discontinuidad en el índice 9.
2. **Preprocesado riguroso:** Recodificación biyectiva de etiquetas al dominio $[0, 23]$, normalización radiométrica al intervalo $[0, 1]$, reestructuración tensorial $(28 \times 28 \times 1)$ y partición estratificada de validación interna ($80\% / 20\%$).
3. **Primeros modelos funcionales:** Comparativa empírica entre un Perceptrón Multicapa denso (**MLP Baseline**, $78.42\%$ en Test) y nuestra primera arquitectura convolucional (**CNN Base**, $91.45\%$ en Test).

---

## 2. Estructura del Repositorio
```text
├── README.md                           # Documentación e instrucciones de reproducción
├── requirements.txt                    # Dependencias con versiones fijadas
├── codigo_cnn_entrega2.py              # Script completo reproducible (EDA + MLP + CNN Base)
├── Entrega2_Primer_Avance.pdf          # Memoria técnica de la Entrega 2 en formato LaTeX
└── figures/                            # Gráficas generadas a 300 DPI
    ├── eda_distribucion_clases.png
    ├── eda_muestras_ejemplo.png
    ├── curvas_entrenamiento_cnn_base.png
    └── matriz_confusion_cnn_base.png
```

---

## 3. Instrucciones de Reproducción
Para reproducir exactamente las métricas y gráficas reportadas en la memoria de la Entrega 2 (fijadas mediante la semilla global `SEED = 42`):

### Paso 1: Clonar el repositorio e instalar dependencias
```bash
git clone https://github.com/beatriz-rodriguez/Proyecto-AA-SignLanguage-CNN.git
cd Proyecto-AA-SignLanguage-CNN
pip install -r requirements.txt
```

### Paso 2: Descargar el conjunto de datos
Descargar los archivos `sign_mnist_train.csv` y `sign_mnist_test.csv` desde la página oficial de Kaggle:
* [Sign Language MNIST en Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist)

Colocar ambos archivos CSV en la carpeta raíz del proyecto (o subirlos al panel lateral de archivos si se ejecuta en Google Colab).

### Paso 3: Ejecutar el código
```bash
python codigo_cnn_entrega2.py
```
El script ejecutará el análisis exploratorio, entrenará ambos modelos (`MLP_Baseline` y `CNN_Base_Entrega2`), guardará las 4 gráficas en alta resolución y mostrará por consola la tabla comparativa y el reporte de clasificación.