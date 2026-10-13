import os
import time
import requests
import urllib3
import streamlit as st
import json

# Desactivar advertencias de certificados SSL autofirmados del cliente de Riot
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. Función para encontrar y leer el archivo lockfile de LoL
def obtener_credenciales_lcu():
    # Rutas por defecto habituales (Ajústala si instalaste el juego en otro disco)
    rutas_posibles = [
        r"C:\Riot Games\League of Legends\lockfile",
        r"D:\Riot Games\League of Legends\lockfile",
        r"P:\Riot Games\League of Legends\lockfile"
    ]
    
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            try:
                with open(ruta, 'r') as f:
                    contenido = f.read()
                # El formato es: LeagueClient:PID:Puerto:Contraseña:Protocolo
                partes = contenido.split(':')
                return {
                    "puerto": partes[2],
                    "token": partes[3],
                    "protocolo": partes[4]
                }
            except Exception:
                return None
    return None

# Configuración de la interfaz de Streamlit
st.title("🏆 LoL Champ Select Tracker")
st.subheader("Monitoreo en tiempo real desde Streamlit")

# Intentar conectar con el cliente
credenciales = obtener_credenciales_lcu()

if not credenciales:
    st.error("❌ No se encontró el cliente de LoL abierto o el archivo lockfile.")
    st.info("Abre el cliente de League of Legends e inicia una partida para empezar.")
else:
    st.success(f"🔗 Conectado al LCU en el puerto: {credenciales['puerto']}")
    
    # Configurar la URL y la autenticación básica de Riot
    url_base = f"{credenciales['protocolo']}://127.0.0.1:{credenciales['puerto']}"
    auth = ('riot', credenciales['token'])
    
    # Crear contenedores vacíos en Streamlit para actualizar los datos en tiempo real
    estado_placeholder = st.empty()
    equipo_aliado_placeholder = st.empty()
    equipo_enemigo_placeholder = st.empty()
    
    # Bucle de actualización en tiempo real
    while True:
        try:
            # Hacer la petición al endpoint de selección de campeón
            respuesta = requests.get(
                f"{url_base}/lol-champ-select/v1/session", 
                auth=auth, 
                verify=False
            )

            if respuesta.status_code == 200:
                datos = respuesta.json()
                estado_placeholder.markdown("### 🔴 Fase de Selección Activa")
                
                # esto es para ver los datos q te devuelve
                with open('dataLcu.json', 'w', encoding='utf-8') as file:
                    file.write(json.dumps(datos))

                # Extraer campeones y preselecciones del equipo aliado (myTeam)
                aliados_str = []
                aliados = []
                for jugador in datos.get('myTeam', []):
                    cell_id = jugador.get('cellId')
                    champion_id = jugador.get('championId')
                    # championPickIntent contiene el ID del campeón que está preseleccionando (hover)
                    pick_intent = jugador.get('championPickIntent', 0)
                    
                    aliados_str.append(
                        f"Jugador (ID: {cell_id}) -> Seleccionado ID: {champion_id} | Hover (Preselección) ID: {pick_intent}"
                    )

                    if champion_id != 0:
                        aliados.append(champion_id)

                    if pick_intent != 0:
                        aliados.append(pick_intent)
                
                # Extraer campeones y preselecciones del equipo enemigo (theirTeam)
                enemigos_str = []
                enemigos = []
                for jugador in datos.get('theirTeam', []):
                    cell_id = jugador.get('cellId')
                    champion_id = jugador.get('championId')
                    pick_intent = jugador.get('championPickIntent', 0)
                    
                    enemigos_str.append(
                        f"Enemigo (ID: {cell_id}) -> Seleccionado ID: {champion_id} | Hover (Preselección) ID: {pick_intent}"
                    )

                    if champion_id != 0:
                        enemigos.append(champion_id)

                    if pick_intent != 0:
                        enemigos.append(pick_intent)
                
                # Mostrar los datos actualizados en los contenedores de Streamlit
                with equipo_aliado_placeholder.container():
                    st.write("**Tu Equipo:**")
                    for a in aliados_str: 
                        st.text(a)
                    
                with equipo_enemigo_placeholder.container():
                    st.write("**Equipo Enemigo:**")
                    for e in enemigos_str: 
                        st.text(e)

                # lo q haremos  es q cuando tengamos 5 aliados y 5 enemigos tenemos q entrar, cogemos sus ids, ya sea champion_id o pick_intent 
                # y usaremos el modelo para predecir. Aun asi para no calcularlo 18 veces habria q mirar es distinto al anterior calculado, y
                # si ya estan seleccionados definitivamente poner una variable para no volver a pasar por aqui
                if len(aliados) == 5 and len(enemigos) == 5: # ahora workea
                    print('tenemos equipo')
                    
            elif respuesta.status_code == 404:
                estado_placeholder.info("⏳ Esperando a entrar en la pantalla de selección de campeones...")
                equipo_aliado_placeholder.empty()
                equipo_enemigo_placeholder.empty()
                
        except requests.exceptions.ConnectionError:
            estado_placeholder.error("📡 Se perdió la conexión con el cliente de LoL.")
            break
            
        # Pausa de 1 segundo antes de volver a preguntar para no saturar el cliente
        time.sleep(1)