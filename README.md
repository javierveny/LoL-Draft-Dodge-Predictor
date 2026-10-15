# [LoL Draft Synergy Predictor](https://github.com/javierveny/LoL-Draft-Dodge-Predictor)

_Read this in [English](#english) | Leer en [Español](#español)_

---

<a id="english"></a>

## English Version

A Machine Learning web application designed to evaluate draft synergies and counter-picks in high-ELO Solo/Duo queue for League of Legends (Patch 16.18).

This project utilizes a hybrid PyTorch neural network that combines Entity Embeddings (to map abstract champion synergies) with a robust engineered metadata set combining empirical Match-V5 data and static Data Dragon stats.

Dataset Extraction
The dataset of 60,000 matches used to train this model was collected using a separate Match Scraper built with the Riot Games API. You can view the data extraction methodology and source code in [this repository](https://github.com/javierveny/League-of-Legends-data-extractor).

### Features

- Hybrid Neural Network: Processes both categorical data (Champion IDs) via embeddings and numerical metadata through a Multilayer Perceptron (MLP).
- Empirical Data Engineering: Built on a dataset of 60,000 matches, calculating real-world stats per champion and role (Damage Per Minute, CC duration, mitigated damage).
- Static Data Integration: Merges empirical data with Riot's Data Dragon JSON to include base stats, growth per level, and primary roles.
- Interactive UI: A fully functional, dual-language (English/Spanish) web interface built with Streamlit.
- "Unseen Pick" Detection: The system automatically identifies champion/role combinations that do not exist in the current training data (e.g., Sona Top) and alerts the user, adjusting predictions safely.

### Prerequisites

- Python 3.9 or higher
- Git

### Installation & Environment Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/javierveny/LoL-Draft-Dodge-Predictor.git
   cd LoL-Draft-Dodge-Predictor
   ```

2. Create a virtual environment:
   It is highly recommended to use a virtual environment to manage dependencies.

   ```bash
   # MacOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   Install the required libraries (including PyTorch, Pandas, and Streamlit):
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To launch the interactive web application, ensure your virtual environment is active and run:

```bash
streamlit run app.py
```

If it does not work try with this command.

```bash
python -m streamlit run app.py
```

This will open a new tab in your default web browser where you can input the Blue and Red team drafts to get real-time win probability predictions.

### Project Structure

- app.py: The Streamlit frontend application containing the UI, data caching, and inference logic.
- main_2.ipynb: The Jupyter Notebook containing the PyTorch model architecture (LeaguePredictorSmall), the training loop, and the LeagueDataset class.
- model/draft_v1.pth: The saved weights of the trained neural network.
- data/: Directory containing the processed datasets (e.g., champion_stats.csv, matches_dataset.csv).

### Note to Riot Games Developer Relations

This Proof of Concept currently operates on a dataset of 60,000 matches. Due to the trillions of possible draft permutations in League of Legends, the model occasionally encounters the "Unseen Pick" problem, where the neural network lacks empirical data for rare champion/role combinations.

A Production API Key is requested to scale the dataset to several million matches. This volume is strictly necessary to mathematically map niche picks, eliminate exposure bias, and improve the accuracy ceiling from the current ~52% to the industry standard of ~55% for draft-only predictions, creating a valuable, production-ready analytical tool for the community.

---

<a id="español"></a>

## Versión en Español

Una aplicación web basada en Machine Learning diseñada para evaluar sinergias de draft y counter-picks en la cola Solo/Duo de alto ELO para League of Legends (Parche 16.18).

Este proyecto utiliza una red neuronal híbrida en PyTorch que combina Entity Embeddings (para mapear sinergias abstractas de campeones) con un robusto conjunto de metadatos diseñados a partir de datos empíricos de Match-V5 y estadísticas estáticas de Data Dragon.

Extracción del Dataset
El dataset de 60.000 partidas utilizado para entrenar este modelo se recopiló mediante un extractor de partidas independiente construido con la API de Riot Games. Puedes consultar la metodología de extracción de datos y el código fuente en [este repositorio](https://github.com/javierveny/League-of-Legends-data-extractor).

### Características

- Red Neuronal Híbrida: Procesa tanto datos categóricos (IDs de campeones) a través de embeddings como metadatos numéricos mediante un Perceptrón Multicapa (MLP).
- Ingeniería de Datos Empíricos: Construido sobre un dataset de 60,000 partidas, calculando estadísticas reales por campeón y rol (Daño por Minuto, duración de CC, daño mitigado).
- Integración de Datos Estáticos: Fusiona datos empíricos con el JSON de Data Dragon de Riot para incluir estadísticas base, crecimiento por nivel y roles primarios.
- Interfaz Interactiva: Una interfaz web completamente funcional y bilingüe (Inglés/Español) construida con Streamlit.
- Detección de "Picks No Vistos": El sistema identifica automáticamente combinaciones de campeón/rol que no existen en los datos de entrenamiento actuales (ej. Sona Top) y alerta al usuario, ajustando las predicciones de forma segura.

### Requisitos Previos

- Python 3.9 o superior
- Git

### Instalación y Configuración del Entorno

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/javierveny/LoL-Draft-Dodge-Predictor.git
   cd LoL-Draft-Dodge-Predictor
   ```

2. Crear un entorno virtual:
   Se recomienda encarecidamente utilizar un entorno virtual para aislar las dependencias.

   ```bash
   # MacOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Instalar dependencias:
   Instala las librerías necesarias (incluyendo PyTorch, Pandas y Streamlit):
   ```bash
   pip install -r requirements.txt
   ```

### Uso

Para iniciar la aplicación web interactiva, asegúrate de que tu entorno virtual esté activo y ejecuta:

```bash
streamlit run app.py
```

Si no ha funcionado prueba con este otro comando

```bash
python -m streamlit run app.py
```

Esto abrirá una nueva pestaña en tu navegador web predeterminado donde podrás introducir los drafts del equipo Azul y Rojo para obtener predicciones de probabilidad de victoria en tiempo real.

### Estructura del Proyecto

- app.py: La aplicación frontend en Streamlit que contiene la interfaz, el sistema de caché y la lógica de inferencia.
- main_2.ipynb: El Jupyter Notebook que contiene la arquitectura del modelo en PyTorch (LeaguePredictorSmall), el bucle de entrenamiento y la clase LeagueDataset.
- model/draft_v1.pth: Los pesos guardados de la red neuronal entrenada tras el proceso de aprendizaje.
- data/: Directorio que contiene los conjuntos de datos procesados (ej. champion_stats.csv, matches_dataset.csv).

### Nota para el equipo de Developer Relations de Riot Games

Esta Prueba de Concepto opera actualmente con un dataset de 60,000 partidas. Debido a los billones de permutaciones de draft posibles en League of Legends, el modelo se enfrenta ocasionalmente al problema del "Pick No Visto", donde la red neuronal carece de datos empíricos para combinaciones raras de campeón/rol.

Se solicita una Clave de API de Producción para escalar el dataset a varios millones de partidas. Este volumen es estrictamente necesario para mapear matemáticamente picks de nicho, eliminar el sesgo de exposición y elevar el techo de precisión del ~52% actual al estándar de la industria del ~55% para predicciones basadas únicamente en el draft, creando así una herramienta analítica robusta para la comunidad.
