import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. CONFIGURACIÓN Y DICCIONARIO DE IDIOMAS
# ==========================================
st.set_page_config(page_title="LoL Draft Predictor", layout="wide")

translations = {
    "title": {"en": "🛡️ LoL Draft Synergy Predictor", "es": "🛡️ LoL Draft Synergy Predictor"},
    "desc": {
        "en": "Machine Learning and Entity Embeddings tool to evaluate draft win probability on patch 16.18.",
        "es": "Herramienta analítica basada en Machine Learning y Entity Embeddings para evaluar la probabilidad de victoria de un draft en el parche 16.18."
    },
    "blue_team": {"en": "🔵 Blue Team", "es": "🔵 Equipo Azul"},
    "red_team": {"en": "🔴 Red Team", "es": "🔴 Equipo Rojo"},
    "btn_analyze": {"en": "🔮 Analyze Draft", "es": "🔮 Analizar Draft"},
    "results_title": {"en": "### Analysis Results", "es": "### Resultados del Análisis"},
    "warning_unseen": {
        "en": "⚠️ **Warning - Insufficient Data:** No record in the training set (60,000 matches) for the following picks: **{picks}**. The model used neutral values, reducing prediction accuracy. (Riot Production API key required to scale data volume).",
        "es": "⚠️ **Atención - Datos Insuficientes:** No hay registro en el set de entrenamiento (60,000 partidas) para los siguientes picks: **{picks}**. El modelo ha utilizado valores neutrales, lo que reduce la precisión de esta predicción. (Se requiere clave de API de Riot de producción para escalar)."
    },
    "prob_blue": {"en": "Blue Team Win Probability", "es": "Probabilidad de Victoria Equipo Azul"},
    "playable": {
        "en": "Recommendation: **PLAYABLE**. The blue draft is statistically superior (better synergies or counter-picks detected).",
        "es": "Recomendación: **JUGABLE**. El draft azul es estadísticamente superior (mejores sinergias o counter-picks detectados)."
    },
    "dodge": {
        "en": "Recommendation: **DODGE**. The red draft has the strategic advantage.",
        "es": "Recomendación: **DODGE**. El draft rojo tiene la ventaja estratégica."
    },
    "skill": {
        "en": "Recommendation: **SKILL MATCHUP**. Extremely balanced composition. It will depend on mechanical skill.",
        "es": "Recomendación: **SKILL MATCHUP**. Composición extremadamente equilibrada. Dependerá de la habilidad mecánica."
    }
}

# Selector de idioma en la barra lateral
lang_choice = st.sidebar.radio("Language / Idioma", ["English", "Español"])
lang = "en" if lang_choice == "English" else "es"
t = lambda key: translations[key][lang]

# ==========================================
# 1. DEFINICIÓN DEL MODELO
# ==========================================
class LeaguePredictorSmall(nn.Module):
    def __init__(self, num_champions=1000, embed_dim=16, meta_features_dim=280):
        super(LeaguePredictorSmall, self).__init__()
        self.champion_embedding = nn.Embedding(num_embeddings=num_champions, embedding_dim=embed_dim)
        input_dim = (10 * embed_dim) + meta_features_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_champs, x_meta):
        embeds = self.champion_embedding(x_champs)
        embeds = embeds.view(embeds.size(0), -1)
        x_combined = torch.cat([embeds, x_meta], dim=1)
        return self.network(x_combined).squeeze(1)


# ==========================================
# 1.1 DEFINICIÓN DEL MODELO 2
# ==========================================

class LeaguePredictor(nn.Module):
    def __init__(self, num_champions=1000, embed_dim=16, meta_features_dim=280):
        super(LeaguePredictor, self).__init__()
        
        # 1. Capa de Embeddings
        # Usamos num_champions=1000 porque los IDs de Riot no son 1-168 secuenciales (ej. Smolder es el 901)
        self.champion_embedding = nn.Embedding(num_embeddings=num_champions, embedding_dim=embed_dim)
        
        # El tamaño de entrada a las capas densas será la suma de los embeddings aplanados (10 * 16 = 160)
        # más todos los metadatos de la partida (280)
        input_dim = (10 * embed_dim) + meta_features_dim
        
        # 2. Red Neuronal Densa (Multilayer Perceptron)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), # Estabiliza y acelera el entrenamiento
            nn.ReLU(),
            nn.Dropout(0.3),     # Apaga el 30% de las neuronas aleatoriamente para evitar sobreajuste
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Capa de salida: 1 sola neurona
            nn.Linear(128, 1),
            nn.Sigmoid()         # Comprime el resultado matemático a una probabilidad exacta entre 0.0 y 1.0
        )

    def forward(self, x_champs, x_meta):
        # x_champs shape: (batch_size, 10)
        # x_meta shape:   (batch_size, 280)
        
        # Pasamos los IDs por la matriz espacial (Embeddings)
        # Result: (batch_size, 10, 16)
        embeds = self.champion_embedding(x_champs)
        
        # Aplanamos los embeddings de los 10 campeones en un solo vector de 160 posiciones
        # Result: (batch_size, 160)
        embeds = embeds.view(embeds.size(0), -1)
        
        # Concatenamos los Embeddings con las estadísticas crudas
        # Result: (batch_size, 440)
        x_combined = torch.cat([embeds, x_meta], dim=1)
        
        # Obtenemos la predicción de victoria
        probability = self.network(x_combined)
        
        # Eliminamos la dimensión extra [batch_size, 1] -> [batch_size]
        return probability.squeeze(1)

# ==========================================
# 2. CARGA DE DATOS Y PESOS
# ==========================================
@st.cache_data
def load_data():
    with open('data/champions.json', 'r', encoding='utf-8') as f:
        dragon_data = json.load(f)
    
    champ_name_to_id = {}
    champ_names = []
    for name, info in dragon_data['data'].items():
        champ_name_to_id[info['name']] = int(info['key'])
        champ_names.append(info['name'])
    champ_names.sort()

    raw_stats_df = pd.read_csv("data/champion_stats.csv").set_index(['championId', 'teamPosition'])
    cols_to_keep = [c for c in raw_stats_df.columns if 
                    c.startswith('perc_') or c.endswith('_pm') or 
                    c.startswith('base_') or c.startswith('riot_') or 
                    c.startswith('tag_') or c.endswith('_per_level') or 
                    c == 'attackrange']
    
    stats_df = raw_stats_df[cols_to_keep].copy()
    scaler = StandardScaler()
    stats_df[stats_df.columns] = scaler.fit_transform(stats_df)
    
    return champ_names, champ_name_to_id, stats_df

@st.cache_resource
def load_model():
    model = LeaguePredictor()
    model.load_state_dict(torch.load("model/draft_v1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# ==========================================
# 3. INTERFAZ DE STREAMLIT
# ==========================================
st.title(t("title"))
st.markdown(t("desc"))

champ_names, champ_name_to_id, stats_df = load_data()
model = load_model()
num_meta_features = len(stats_df.columns)

st.subheader(t("blue_team"))
col1, col2, col3, col4, col5 = st.columns(5)
blue_top = col1.selectbox("Top", champ_names, index=champ_names.index("Aatrox") if "Aatrox" in champ_names else 0, key='bt')
blue_jng = col2.selectbox("Jungle", champ_names, index=champ_names.index("Lee Sin") if "Lee Sin" in champ_names else 0, key='bj')
blue_mid = col3.selectbox("Mid", champ_names, index=champ_names.index("Ahri") if "Ahri" in champ_names else 0, key='bm')
blue_bot = col4.selectbox("ADC", champ_names, index=champ_names.index("Ashe") if "Ashe" in champ_names else 0, key='bb')
blue_sup = col5.selectbox("Support", champ_names, index=champ_names.index("Thresh") if "Thresh" in champ_names else 0, key='bs')

st.subheader(t("red_team"))
col6, col7, col8, col9, col10 = st.columns(5)
red_top = col6.selectbox("Top", champ_names, index=champ_names.index("Darius") if "Darius" in champ_names else 0, key='rt')
red_jng = col7.selectbox("Jungle", champ_names, index=champ_names.index("Vi") if "Vi" in champ_names else 0, key='rj')
red_mid = col8.selectbox("Mid", champ_names, index=champ_names.index("Zed") if "Zed" in champ_names else 0, key='rm')
red_bot = col9.selectbox("ADC", champ_names, index=champ_names.index("Jinx") if "Jinx" in champ_names else 0, key='rb')
red_sup = col10.selectbox("Support", champ_names, index=champ_names.index("Leona") if "Leona" in champ_names else 0, key='rs')

st.markdown("---")

if st.button(t("btn_analyze"), use_container_width=True):
    selected_draft = [
        (blue_top, 'TOP'), (blue_jng, 'JUNGLE'), (blue_mid, 'MIDDLE'), (blue_bot, 'BOTTOM'), (blue_sup, 'UTILITY'),
        (red_top, 'TOP'), (red_jng, 'JUNGLE'), (red_mid, 'MIDDLE'), (red_bot, 'BOTTOM'), (red_sup, 'UTILITY')
    ]
    
    champ_ids = []
    meta_features = []
    unseen_picks = [] 
    
    for champ_name, role in selected_draft:
        c_id = champ_name_to_id[champ_name]
        champ_ids.append(c_id)
        
        try:
            stats_vector = stats_df.loc[(c_id, role)].values.tolist()
        except KeyError:
            stats_vector = [0.0] * num_meta_features
            unseen_picks.append(f"{champ_name} ({role})")
            
        meta_features.extend(stats_vector)
        
    x_champs = torch.tensor(champ_ids, dtype=torch.long).unsqueeze(0)
    x_meta = torch.tensor(meta_features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prob_blue = model(x_champs, x_meta).item() * 100
        
    st.markdown(t("results_title"))
    
    if unseen_picks:
        st.warning(t("warning_unseen").format(picks=', '.join(unseen_picks)))

    if prob_blue >= 52.0:
        st.success(f"📈 {t('prob_blue')}: {prob_blue:.2f}%")
        st.info(t("playable"))
    elif prob_blue <= 48.0:
        st.error(f"📉 {t('prob_blue')}: {prob_blue:.2f}%")
        st.warning(t("dodge"))
    else:
        st.warning(f"⚖️ {t('prob_blue')}: {prob_blue:.2f}%")
        st.info(t("skill"))
