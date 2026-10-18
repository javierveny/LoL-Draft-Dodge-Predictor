import os
import requests
import urllib3
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

# Desactivar advertencias SSL del cliente de Riot
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 0. CONFIGURACIÓN Y DICCIONARIO DE IDIOMAS
# ==========================================
st.set_page_config(page_title="LoL Live Draft Predictor", layout="wide")

translations = {
    "title": {"en": "🛡️ LoL Live Draft Synergy Predictor", "es": "🛡️ LoL Live Draft Synergy Predictor"},
    "desc": {
        "en": "Real-time LCU integration + Machine Learning to evaluate draft win probability.",
        "es": "Integración en tiempo real con el cliente de LoL + Machine Learning para evaluar la probabilidad de victoria."
    },
    "blue_team": {"en": "🔵 Blue Team", "es": "🔵 Equipo Azul"},
    "red_team": {"en": "🔴 Red Team", "es": "🔴 Equipo Rojo"},
    "btn_analyze": {"en": "🔮 Analyze Draft Now", "es": "🔮 Analizar Draft Ahora"},
    "results_title": {"en": "### Analysis Results", "es": "### Resultados del Análisis"},
    "warning_unseen": {
        "en": "⚠️ **Warning - Insufficient Data:** No record in the training set for: **{picks}**.",
        "es": "⚠️ **Atención - Datos Insuficientes:** No hay registro en el set de entrenamiento para: **{picks}**."
    },
    "prob_blue": {"en": "Blue Team Win Probability", "es": "Probabilidad de Victoria Equipo Azul"},
    "playable": {
        "en": "Recommendation: **PLAYABLE**. The blue draft is statistically superior.",
        "es": "Recomendación: **JUGABLE**. El draft azul es estadísticamente superior."
    },
    "dodge": {
        "en": "Recommendation: **DODGE**. The red draft has the strategic advantage.",
        "es": "Recomendación: **DODGE**. El draft rojo tiene la ventaja estratégica."
    },
    "skill": {
        "en": "Recommendation: **SKILL MATCHUP**. Extremely balanced composition.",
        "es": "Recomendación: **SKILL MATCHUP**. Composición extremadamente equilibrada."
    }
}

lang_choice = st.sidebar.radio("Language / Idioma", ["Español", "English"])
lang = "en" if lang_choice == "English" else "es"
t = lambda key: translations[key][lang]

# ==========================================
# 1. DEFINICIÓN DEL MODELO
# ==========================================
class LeaguePredictor(nn.Module):
    def __init__(self, num_champions=1000, embed_dim=16, meta_features_dim=280):
        super(LeaguePredictor, self).__init__()
        self.champion_embedding = nn.Embedding(num_embeddings=num_champions, embedding_dim=embed_dim)
        input_dim = (10 * embed_dim) + meta_features_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_champs, x_meta):
        embeds = self.champion_embedding(x_champs)
        embeds = embeds.view(embeds.size(0), -1)
        x_combined = torch.cat([embeds, x_meta], dim=1)
        return self.network(x_combined).squeeze(1)

# ==========================================
# 2. CARGA DE DATOS Y PESOS
# ==========================================
@st.cache_data
def load_data():
    with open('data/champions.json', 'r', encoding='utf-8') as f:
        dragon_data = json.load(f)
    
    champ_name_to_id = {}
    champ_id_to_name = {}
    champ_names = []
    for name, info in dragon_data['data'].items():
        c_id = int(info['key'])
        c_name = info['name']
        champ_name_to_id[c_name] = c_id
        champ_id_to_name[c_id] = c_name
        champ_names.append(c_name)
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
    
    return champ_names, champ_name_to_id, champ_id_to_name, stats_df

@st.cache_resource
def load_model():
    model = LeaguePredictor()
    model.load_state_dict(torch.load("model/draft_v1.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

champ_names, champ_name_to_id, champ_id_to_name, stats_df = load_data()
model = load_model()
num_meta_features = len(stats_df.columns)

# ==========================================
# 3. FUNCIONES AUXILIARES (LCU Y ROLES)
# ==========================================
def obtener_credenciales_lcu():
    rutas_posibles = [
        r"C:\Riot Games\League of Legends\lockfile",
        r"D:\Riot Games\League of Legends\lockfile",
        r"P:\Riot Games\League of Legends\lockfile"
    ]
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            try:
                with open(ruta, 'r') as f:
                    parts = f.read().split(':')
                return {"puerto": parts[2], "token": parts[3], "protocolo": parts[4]}
            except Exception:
                return None
    return None

def deducir_roles_equipo(jugadores_lcu, es_aliado=True):
    """
    Asigna cada campeón a su línea (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY).
    - En aliados: usa 'assignedPosition' del cliente de LoL si está disponible.
    - En enemigos (o modos sin rol): verifica en stats_df qué roles existen para ese campeón 
      y si no, rellena por orden de llegada.
    """
    roles_orden = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
    asignados = {r: None for r in roles_orden}
    sin_rol_fijo = []

    mapa_pos_lcu = {
        'top': 'TOP', 'jungle': 'JUNGLE', 'middle': 'MIDDLE', 'mid': 'MIDDLE',
        'bottom': 'BOTTOM', 'bot': 'BOTTOM', 'utility': 'UTILITY', 'support': 'UTILITY'
    }

    # 1. Extraer campeones seleccionados (o en preselección/hover si aún no han fijado)
    for j in jugadores_lcu:
        c_id = j.get('championId', 0)
        if c_id == 0:
            c_id = j.get('championPickIntent', 0)
        
        if c_id != 0 and c_id in champ_id_to_name:
            pos_lcu = mapa_pos_lcu.get(j.get('assignedPosition', '').lower(), None)
            if es_aliado and pos_lcu and asignados[pos_lcu] is None:
                asignados[pos_lcu] = c_id
            else:
                sin_rol_fijo.append(c_id)

    # 2. IA heurística para el equipo enemigo: buscar en tu CSV qué posiciones juega cada campeón
    pendientes = []
    for c_id in sin_rol_fijo:
        colocado = False
        for rol in roles_orden:
            if asignados[rol] is None and (c_id, rol) in stats_df.index:
                asignados[rol] = c_id
                colocado = True
                break
        if not colocado:
            pendientes.append(c_id)

    # 3. Rellenar los huecos restantes en orden estricto (Top -> Jng -> Mid -> ADC -> Sup)
    for c_id in pendientes:
        for rol in roles_orden:
            if asignados[rol] is None:
                asignados[rol] = c_id
                break

    return asignados

def ejecutar_prediccion():
    roles_keys = [
        ('bt', 'TOP'), ('bj', 'JUNGLE'), ('bm', 'MIDDLE'), ('bb', 'BOTTOM'), ('bs', 'UTILITY'),
        ('rt', 'TOP'), ('rj', 'JUNGLE'), ('rm', 'MIDDLE'), ('rb', 'BOTTOM'), ('rs', 'UTILITY')
    ]
    
    champ_ids = []
    meta_features = []
    unseen_picks = [] 
    
    for key, role in roles_keys:
        champ_name = st.session_state[key]
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

# ==========================================
# 4. ESTADO DE SESIÓN (SESSION STATE)
# ==========================================
default_champs = {
    'bt': 'Aatrox', 'bj': 'Lee Sin', 'bm': 'Ahri', 'bb': 'Ashe', 'bs': 'Thresh',
    'rt': 'Darius', 'rj': 'Vi', 'rm': 'Zed', 'rb': 'Jinx', 'rs': 'Leona'
}
for k, v in default_champs.items():
    if k not in st.session_state:
        st.session_state[k] = v if v in champ_names else champ_names[0]

if 'last_lcu_picks' not in st.session_state:
    st.session_state.last_lcu_picks = {}
if 'auto_predict_triggered' not in st.session_state:
    st.session_state.auto_predict_triggered = False

# ==========================================
# 5. INTERFAZ Y SINCRONIZACIÓN EN VIVO
# ==========================================
st.title(t("title"))
st.markdown(t("desc"))

sync_lcu = st.sidebar.checkbox("🔄 Sincronizar con cliente LoL en vivo", value=True)

# Usamos st.fragment para refrescar el estado del LCU cada 2 segundos sin bloquear la interfaz
@st.fragment(run_every="2s" if sync_lcu else None)
def panel_interactivo():
    total_locked = 0

    if sync_lcu:
        credenciales = obtener_credenciales_lcu()
        if not credenciales:
            st.warning("⏳ Buscando cliente de League of Legends abierto (`lockfile`)...")
        else:
            url_base = f"{credenciales['protocolo']}://127.0.0.1:{credenciales['puerto']}"
            auth = ('riot', credenciales['token'])
            try:
                resp = requests.get(f"{url_base}/lol-champ-select/v1/session", auth=auth, verify=False, timeout=1)
                if resp.status_code == 200:
                    datos = resp.json()
                    my_team = datos.get('myTeam', [])
                    their_team = datos.get('theirTeam', [])

                    # Detectar en qué lado del mapa estamos (cellId 0-4 suele ser Equipo Azul)
                    soy_azul = True
                    if my_team and my_team[0].get('cellId', 0) >= 5:
                        soy_azul = False

                    roles_aliados = deducir_roles_equipo(my_team, es_aliado=True)
                    roles_enemigos = deducir_roles_equipo(their_team, es_aliado=False)

                    roles_azul = roles_aliados if soy_azul else roles_enemigos
                    roles_rojo = roles_enemigos if soy_azul else roles_aliados

                    map_keys_blue = {'TOP': 'bt', 'JUNGLE': 'bj', 'MIDDLE': 'bm', 'BOTTOM': 'bb', 'UTILITY': 'bs'}
                    map_keys_red = {'TOP': 'rt', 'JUNGLE': 'rj', 'MIDDLE': 'rm', 'BOTTOM': 'rb', 'UTILITY': 'rs'}

                    # Solo actualizamos el desplegable si el jugador en el LoL ha cambiado su pick,
                    # permitiendo que tú puedas moverlos manualmente sin que se sobreescriban cada 2 seg.
                    for rol, key in map_keys_blue.items():
                        c_id = roles_azul.get(rol)
                        if c_id and st.session_state.last_lcu_picks.get(key) != c_id:
                            st.session_state[key] = champ_id_to_name[c_id]
                            st.session_state.last_lcu_picks[key] = c_id

                    for rol, key in map_keys_red.items():
                        c_id = roles_rojo.get(rol)
                        if c_id and st.session_state.last_lcu_picks.get(key) != c_id:
                            st.session_state[key] = champ_id_to_name[c_id]
                            st.session_state.last_lcu_picks[key] = c_id

                    # Contar cuántos campeones están 100% fijados (championId != 0)
                    locked_allies = [j.get('championId', 0) for j in my_team if j.get('championId', 0) != 0]
                    locked_enemies = [j.get('championId', 0) for j in their_team if j.get('championId', 0) != 0]
                    total_locked = len(locked_allies) + len(locked_enemies)

                    lado_str = "Equipo Azul 🔵" if soy_azul else "Equipo Rojo 🔴"
                    st.success(f"🟢 Champ Select Activo ({lado_str}) — Campeones fijados: **{total_locked}/10**")
                else:
                    st.info("🔗 Conectado al cliente de LoL. Esperando a entrar en Selección de Campeones...")
                    st.session_state.last_lcu_picks = {}
                    st.session_state.auto_predict_triggered = False
            except Exception:
                st.error("📡 Error de conexión con el cliente de LoL.")

    # --- SELECTORES INTERACTIVOS ---
    st.subheader(t("blue_team"))
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.selectbox("Top", champ_names, key='bt')
    col2.selectbox("Jungle", champ_names, key='bj')
    col3.selectbox("Mid", champ_names, key='bm')
    col4.selectbox("ADC", champ_names, key='bb')
    col5.selectbox("Support", champ_names, key='bs')

    st.subheader(t("red_team"))
    col6, col7, col8, col9, col10 = st.columns(5)
    col6.selectbox("Top", champ_names, key='rt')
    col7.selectbox("Jungle", champ_names, key='rj')
    col8.selectbox("Mid", champ_names, key='rm')
    col9.selectbox("ADC", champ_names, key='rb')
    col10.selectbox("Support", champ_names, key='rs')

    st.markdown("---")

    # Botón para intercambiar roles o predecir manualmente en cualquier momento
    analizar_manual = st.button(t("btn_analyze"), use_container_width=True)

    # Condición automática: cuando los 10 campeones están seleccionados
    if total_locked == 10:
        st.session_state.auto_predict_triggered = True

    if analizar_manual or st.session_state.auto_predict_triggered:
        ejecutar_prediccion()

panel_interactivo()
