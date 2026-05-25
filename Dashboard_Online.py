import streamlit as st
import fitparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import re
from datetime import datetime, timedelta
import io
import hashlib
import requests
from requests.auth import HTTPBasicAuth
from sqlalchemy import text
from supabase import create_client, Client

# --- DATENBANK & KONFIGURATION ---
st.set_page_config(page_title="Advanced Power Data Analyser", layout="wide")
def get_athlete_stats_from_intervals(api_key, user_id):
    auth = HTTPBasicAuth('API_KEY', api_key)
    res_profile = requests.get("https://intervals.icu/api/v1/athlete/me/profile", auth=auth)
    res_sports = requests.get(f"https://intervals.icu/api/v1/athlete/i{user_id}/sport-settings", auth=auth)
    
    stats = {"Name": "Unbekannt", "FTP": 0, "Weight": 75, "Max HR": "-", "Age": "-"}
    
    if res_profile.status_code == 200:
        data = res_profile.json().get('athlete', {})
        stats["Name"] = data.get('name')
        stats["Weight"] = data.get('weight', 75)
        if data.get('dob'): stats["Age"] = datetime.now().year - int(data.get('dob')[:4])
        
    if res_sports.status_code == 200:
        sports = res_sports.json()
        cycling = next((s for s in sports if "Ride" in s.get('types', [])), None)
        if cycling:
            stats["FTP"] = cycling.get('ftp', 0)
            stats["Max HR"] = cycling.get('max_hr', '-')
            
    return stats
def get_db_connection():
    # Streamlit connection (nutzt automatisch psycopg3, wenn in requirements.txt)
    return st.connection("postgresql", type="sql", url=st.secrets["DB_URL"])

@st.cache_resource
def get_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# --- LOGIN FUNKTION ---
def check_login(username, password):
    supabase = get_supabase_client()
    response = supabase.table("users").select("password_hash, role").eq("name", username).execute()
    if response.data:
        user = response.data[0]
        stored_hash = user['password_hash']
        role = user['role']
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        if input_hash == stored_hash:
            return True, role
    return False, None

# --- SESSION STATES ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'role' not in st.session_state: st.session_state['role'] = None
if 'user' not in st.session_state: st.session_state['user'] = None

# --- LOGIN-TOR ---
if not st.session_state['logged_in']:
    st.title("🔒 Login erforderlich")
    user_in = st.text_input("Benutzername")
    pass_in = st.text_input("Passwort", type="password")
    if st.button("Anmelden"):
        is_valid, role = check_login(user_in, pass_in)
        if is_valid:
            st.session_state['logged_in'] = True
            st.session_state['user'] = user_in
            st.session_state['role'] = role
            st.rerun()
        else: st.error("Benutzername oder Passwort falsch!")
    st.stop()

# --- SESSION STATES (Rest) ---
if 'manual_intervals' not in st.session_state: st.session_state['manual_intervals'] = []
if 'erfassungs_modus' not in st.session_state: st.session_state['erfassungs_modus'] = "Automatisch (Algorithmus)"
if 'overwrite_warning' not in st.session_state: st.session_state['overwrite_warning'] = False
if 'workout_to_overwrite' not in st.session_state: st.session_state['workout_to_overwrite'] = None
if 'df' not in st.session_state: st.session_state['df'] = pd.DataFrame()    

# --- DATENBANK HELFER (PostgreSQL) ---
def add_new_athlete(name, api_key, password, intervals_id): # <-- Parameter ergänzt
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_db_connection()
    
    with conn.session as s:
        s.execute(text("""
            INSERT INTO users (name, api_key, password_hash, role, intervals_id) 
            VALUES (:name, :api_key, :pwd, 'user', :iid)
        """), {"name": name.strip(), "api_key": api_key.strip(), "pwd": pwd_hash, "iid": intervals_id.strip()})
        s.commit()
    return True, f"Athlet '{name}' angelegt!"

def get_authorized_athletes(current_user_name, role):
    conn = get_db_connection()
    try:
        if role == 'admin':
            return conn.query("SELECT * FROM users")
        elif role == 'trainer':
            return conn.query("SELECT * FROM users WHERE trainer_id = :tid", 
                              params={"tid": st.session_state.get('user_id')})
        else:
            return conn.query("SELECT * FROM users WHERE name = :name", 
                              params={"name": current_user_name})
    except Exception as e:
        st.error(f"Datenbankfehler in get_authorized_athletes: {e}")
        return pd.DataFrame() # Leeres DataFrame zurückgeben bei Fehler
        
def load_all_athletes():
    conn = get_db_connection()
    # Wir nehmen alle User aus der Tabelle, um sie in der Liste anzuzeigen
    return conn.query("SELECT * FROM users")
def check_duplicate_workout(date, workout_type, structure):
    conn = get_db_connection()
    result = conn.query("SELECT id FROM workouts WHERE date = :date AND type = :type AND structure = :structure",
                        params={"date": date, "type": workout_type, "structure": structure})
    return result.iloc[0]['id'] if not result.empty else None

def save_workout_to_db(metadata, interval_list, overwrite_id=None):
    conn = get_db_connection()
    with conn.session as s:
        if overwrite_id:
            s.execute(text("DELETE FROM workouts WHERE id = :id"), {"id": overwrite_id})
        
        # Workout einfügen
        res = s.execute(text("""
            INSERT INTO workouts (filename, date, type, structure, avg_power, max_power) 
            VALUES (:filename, :date, :type, :structure, :avg_p, :max_p) RETURNING id
        """), {
            "filename": metadata['filename'], "date": metadata['date'], "type": metadata['type'], 
            "structure": metadata['structure'], "avg_p": metadata['avg_power'], "max_p": metadata['max_power']
        })
        workout_id = res.scalar()
        for row in interval_list:
            # Konvertiere jeden Wert sicher in einen Python-Typ (float oder int)
            s.execute(text("""
                INSERT INTO intervals (workout_id, interval_num, avg_power, avg_hr, max_hr, duration_sec, std_hr, avg_hr_p) 
                VALUES (:wid, :num, :ap, :ahr, :mhr, :dur, :std, :ahrp)
            """), {
                "wid": int(workout_id),
                "num": int(row['Intervall']),
                "ap": float(row['Ø Leistung']),
                "ahr": float(row['Ø Herzfrequenz']),
                "mhr": float(row['Max Herzfrequenz']),
                "dur": int(row['Dauer_sec']),
                "std": float(row['Abweichung HF+-']),  # Hier wird np.float64 zu einem sauberen float
                "ahrp": float(row['Durschnittliche HF_P (20-80)'])
            })
        s.commit()
    return True, "Daten erfolgreich in die Datenbank übernommen."

def transfer_to_manual(timestamps):
    st.session_state['manual_intervals'] = timestamps
    st.session_state['erfassungs_modus'] = "Manuell (Grafische Auswahl)"
    st.session_state['overwrite_warning'] = False

# --- API CLOUD COCKPIT ---
def fetch_calendar_events(api_key, start_dt, end_dt):
    url = "https://intervals.icu/api/v1/athlete/0/activities"
    params = {"oldest": start_dt.strftime('%Y-%m-%d'), "newest": end_dt.strftime('%Y-%m-%d')}
    try:
        response = requests.get(url, params=params, auth=HTTPBasicAuth('API_KEY', api_key), timeout=12)
        if response.status_code == 200: return response.json(), None
        return None, f"Fehler {response.status_code}"
    except Exception as e: return None, str(e)

def download_original_fit_file(api_key, activity_id):
    url = f"https://intervals.icu/api/v1/activity/{activity_id}/file"
    try:
        response = requests.get(url, auth=HTTPBasicAuth('API_KEY', api_key), timeout=20)
        if response.status_code == 200: return response.content, None
        return None, f"Fehler {response.status_code}"
    except Exception as e: return None, str(e)

# --- SURFACE LAYOUT ---
st.title("Powerdata Dashboard")
# LOGOUT
if st.session_state.get('logged_in'):
    if st.sidebar.button("🚪 Ausloggen"):
        st.session_state['logged_in'] = False
        st.session_state['user'] = None
        st.session_state['role'] = None
        st.rerun()
nav_mode = st.sidebar.radio("Navigation", ["Training einlesen", "Daten & Auswertung", "👤 Athleten verwalten"])

# --- ADMIN-CHECK ---
if nav_mode == "👤 Athleten verwalten":
    # WICHTIG: Alles ab hier muss eingerückt sein!
    if st.session_state.get('user') == "Bastian":
        st.subheader("🛠️ Admin: Athleten verwalten")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### ➕ Neuen Athleten anlegen")
            with st.form("new_athlete_form", clear_on_submit=True):
                name_in = st.text_input("Name:")
                key_in = st.text_input("API Key:", type="password")
                pwd_in = st.text_input("Start-Passwort:", type="password")
                submitted = st.form_submit_button("Speichern")
                
                if submitted:
                    if name_in and key_in and pwd_in:
                        success, message = add_new_athlete(name_in, key_in, pwd_in)
                        if success: st.success(message)
                        else: st.error(message)
                    else:
                        st.warning("Bitte alle Felder ausfüllen!")
        
        with col_right:
            st.markdown("### 🗑️ Athlet löschen")
            df_all = load_all_athletes()
            if not df_all.empty:
                del_name = st.selectbox("Wähle Athlet zum Löschen:", df_all["name"])
                if st.button("Löschen"):
                    conn = get_db_connection()
                    with conn.session as s:
                        s.execute(text("DELETE FROM users WHERE name = :name"), {"name": del_name})
                        s.commit()
                    st.success(f"Athlet {del_name} wurde gelöscht.")
                    st.rerun()
    else:
        st.error("Zugriff verweigert! Nur für den Administrator.")

elif nav_mode == "Training einlesen":
    if 'df' not in st.session_state: st.session_state['df'] = pd.DataFrame()
    
    df_all_users = load_all_athletes()
    selected_activity_id = None
    filename = ""
    api_k = None

    st.sidebar.header("👤 Aktiver Athlet")
    options_list = ["Lokal (.fit-Datei)"] + (df_all_users["name"].tolist() if not df_all_users.empty else [])
    user_select = st.sidebar.selectbox("Wer hat trainiert?", options=options_list, key="user_select_key")

    if user_select != "Lokal (.fit-Datei)" and not df_all_users.empty:
        api_k = df_all_users.loc[df_all_users["name"] == user_select, "api_key"].values[0]
        start_date = st.sidebar.date_input("Start-Datum", datetime.now() - timedelta(days=60))
        end_date = st.sidebar.date_input("End-Datum", datetime.now())
        
        with st.spinner("Synchronisiere Aktivitäten..."):
            events, err = fetch_calendar_events(api_k, start_date, end_date)
            if events:
                df_cockpit = pd.DataFrame([{"Datum": e.get("start_date_local", "0000-00-00")[:10], "Name": e.get("name") or "Fahrt", "ID": e.get("id")} for e in events if e.get("id")])
                sel = st.dataframe(df_cockpit, use_container_width=True, on_select="rerun", selection_mode="single-row")
                if sel and len(sel.get("selection", {}).get("rows", [])) > 0:
                    idx = sel["selection"]["rows"][0]
                    selected_activity_id = df_cockpit.iloc[idx]["ID"]
                    filename = df_cockpit.iloc[idx]["Name"]
                    bin_data, _ = download_original_fit_file(api_k, selected_activity_id)
                    if bin_data:
                        fitfile = fitparse.FitFile(io.BytesIO(bin_data))
                        st.session_state['df'] = pd.DataFrame([r.get_values() for r in fitfile.get_messages('record')])
                        st.session_state['df']['timestamp'] = pd.to_datetime(st.session_state['df']['timestamp'])
                        st.session_state['df'].set_index('timestamp', inplace=True)
            else: st.info("Keine Aktivitäten gefunden.")
    elif user_select == "Lokal (.fit-Datei)":
        up = st.sidebar.file_uploader("Datei wählen", type=["fit"])
        if up:
            fitfile = fitparse.FitFile(up)
            st.session_state['df'] = pd.DataFrame([r.get_values() for r in fitfile.get_messages('record')])
            st.session_state['df']['timestamp'] = pd.to_datetime(st.session_state['df']['timestamp'])
            st.session_state['df'].set_index('timestamp', inplace=True)
            filename = up.name

    df = st.session_state['df']
    
    # --- VERARBEITUNG & ALGORITHMUS ---
    if not df.empty:
        try:
            st.markdown("---")
            st.subheader("🎯 Intervall-Feinjustierung & Struktur-Editor")
            match_type = re.search(r'(LIT|MIT|HIT)', filename, re.IGNORECASE)
            detected_type = match_type.group(1).upper() if match_type else "UNKNOWN"
            match_structure = re.search(r'(\d+)[xX](\d+)', filename)
            init_intervals = int(match_structure.group(1)) if match_structure else 4
            init_duration_min = int(match_structure.group(2)) if match_structure else 15

            col1, col2, col3 = st.columns(3)
            with col1: mode_type = st.selectbox("Erfassungs-Modus", ["Automatisch (Algorithmus)", "Manuell (Grafische Auswahl)"], key='erfassungs_modus')
            with col2: expected_intervals = st.number_input("Erwartete Anzahl Intervalle", value=init_intervals, min_value=1)
            with col3: expected_duration_min = st.number_input("Erwartete Intervalllänge (Min)", value=init_duration_min, min_value=1)
            
            c4, c5 = st.columns(2)
            with c4: min_power = st.slider("Mindestleistung (Watt)", 50, 400, 185, step=5)
            with c5: sg_win = st.slider("Filterfenster", 11, 121, 45, step=2)
            
            if 'power' in df.columns:
                df['p_clean'] = df['power'].fillna(0)
                df['p_sg'] = savgol_filter(df['p_clean'].rolling(7, center=True, min_periods=1).median(), window_length=sg_win, polyorder=2)
                
                if mode_type == "Automatisch (Algorithmus)":
                    target = expected_duration_min * 60
                    df['roll'] = df['p_sg'].rolling(window=target, min_periods=int(target*0.5), center=True).mean()
                    scores = df['roll'].fillna(0).values
                    candidates = [i for i in range(len(df)) if scores[i] >= min_power and scores[i] == max(scores[max(0, i-target//2):min(len(df), i+target//2)])]
                    candidates = sorted(candidates, key=lambda idx: scores[idx], reverse=True)[:expected_intervals]
                    
                    is_int = [False] * len(df)
                    block = [0] * len(df)
                    for b_n, c_idx in enumerate(sorted(candidates), 1):
                        l, r = c_idx - target//2, c_idx + target//2
                        for j in range(max(0, l), min(len(df), r)):
                            is_int[j] = True
                            block[j] = b_n
                    df['is_interval'], df['block_id'] = is_int, block
                
                df['highlight'] = df.apply(lambda row: row['power'] if row['is_interval'] else None, axis=1)
                st.metric("Intervalle erkannt", f"{(df['is_interval'].astype(int).diff() == 1).sum()} / {expected_intervals}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['power'], name='Power', line=dict(color='gray', width=1)))
                fig.add_trace(go.Scatter(x=df.index, y=df['highlight'], name='Interval', line=dict(color='orange', width=2)))
                fig.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e: st.error(f"Fehler: {e}")
    else:
        st.info("Bitte Athlet wählen oder Datei hochladen.")

elif nav_mode == "Daten & Auswertung":
    st.subheader("📊 Daten & Auswertung")
    
    authorized_athletes = get_authorized_athletes(st.session_state['user'], st.session_state['role'])
    
    if authorized_athletes.empty:
        st.warning("Keine Athleten gefunden.")
    else:
        # Layout: 3 Spalten (Auswahl/Filter schmal, Profil schmal, Rest Platz für Workouts)
        c1, c2, c3 = st.columns([1, 1, 2])
        
        with c1:
            selected_name = st.selectbox("Athlet wählen:", options=authorized_athletes["name"], key="data_eval_athlete_selector")
            athlete_row = authorized_athletes[authorized_athletes["name"] == selected_name].iloc[0]
            filter_type = st.selectbox("Typ-Filter", ["ALLE", "LIT", "MIT", "HIT"], key="data_eval_filter_type")
            search_query = st.text_input("Suche:", key="data_eval_search_input")

        with c2:
            st.markdown("##### 👤 Profil")
            
            # --- API ABFRAGE DIREKT HIER ---
            # Wir nutzen die Logik aus deinem Testscript
            auth = HTTPBasicAuth('API_KEY', athlete_row['api_key'])
            base_url = "https://intervals.icu/api/v1/athlete/i75948"
            
            # Profil & Sport-Settings abrufen
            res_p = requests.get(f"{base_url}/profile", auth=auth)
            res_s = requests.get(f"{base_url}/sport-settings", auth=auth)
            
            p_data = res_p.json() if res_p.status_code == 200 else {}
            s_data = res_s.json() if res_s.status_code == 200 else []
            
            # Daten extrahieren
            ath_info = p_data.get('athlete', {})
            ride_info = next((s for s in s_data if "Ride" in s.get('types', [])), {})
            
            # Werte zuweisen (Gewicht kommt oft aus Ride-Settings, falls Profil leer)
            ftp = ride_info.get('ftp', 0)
            weight = ride_info.get('weight', 66.8) # Fallback, falls API das nicht liefert
            max_hr = ride_info.get('max_hr', 184)
            age = "k.A." # da DOB im JSON nicht geliefert wurde
            w_kg = round(ftp / weight, 2) if weight > 0 else 0
            
            # Tabelle aufbauen
            profile_df = pd.DataFrame([
                [f"{athlete_row['name']}", f"Stand: {datetime.now().strftime('%d.%m.%y')}"],
                [f"Alter: {age}", f"Gewicht: {weight} kg"],
                [f"FTP: {ftp} W", f"{w_kg} W/kg"],
                [f"Max HF: {max_hr} bpm", ""]
            ])
            
            # Tabelle rendern ohne Header und ohne Index
            st.table(profile_df.set_axis([' ', '  '], axis=1))

        # --- WORKOUT LOGIK ---
        conn = get_db_connection()
        uid_val = int(athlete_row['id'])
        df_workouts = conn.query("SELECT * FROM workouts WHERE user_id = :uid", params={"uid": uid_val})
        
        if df_workouts.empty: 
            st.info(f"Keine Trainingsdaten für {selected_name} gefunden.")
        else:
            if filter_type != "ALLE": 
                df_workouts = df_workouts[df_workouts['type'] == filter_type]
            if search_query:
                df_workouts = df_workouts[df_workouts['filename'].str.contains(search_query, case=False, na=False) | df_workouts['date'].str.contains(search_query, case=False, na=False)]

            selected_ids = []
            for idx, row in df_workouts.iterrows():
                if st.checkbox(f"{row['date']} | {row['type']} ({row['structure']}) | {row['filename']}", key=f"eval_check_{row['id']}"):
                    selected_ids.append(row['id'])
                if st.button("🗑️", key=f"eval_del_{row['id']}"):
                    with conn.session as s:
                        s.execute(text("DELETE FROM workouts WHERE id = :id"), {"id": row['id']})
                        s.commit()
                    st.rerun()

            if len(selected_ids) >= 1:
                st.markdown("---")
                ids_string = ",".join(map(str, selected_ids))
                df_compare = conn.query(f"SELECT i.*, w.date, w.type FROM intervals i JOIN workouts w ON i.workout_id = w.id WHERE i.workout_id IN ({ids_string})")
                
                if not df_compare.empty:
                    df_compare['Workout'] = df_compare['date'].str.slice(0, 10) + " (" + df_compare['type'] + ")"
                    c_p1, c_p2, c_p3 = st.columns(3)
                    with c_p1: 
                        fig1 = px.scatter(df_compare, x="interval_num", y="avg_power", color="Workout", title="Ø Watt")
                        fig1.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        st.plotly_chart(fig1, use_container_width=True)
                    with c_p2:
                        fig_hr = go.Figure()
                        for w in df_compare['Workout'].unique():
                            sub = df_compare[df_compare['Workout'] == w]
                            fig_hr.add_trace(go.Scatter(x=sub['interval_num'], y=sub['avg_hr'], name=f"{w} (Ø)", mode='lines+markers'))
                            fig_hr.add_trace(go.Scatter(x=sub['interval_num'], y=sub['avg_hr_p'], name=f"{w} (20-80%)", mode='markers'))
                        fig_hr.update_layout(title="Ø Herzfrequenz", template="plotly_dark")
                        st.plotly_chart(fig_hr, use_container_width=True)
                    with c_p3: 
                        fig3 = px.scatter(df_compare, x="interval_num", y="max_hr", color="Workout", title="Max HF")
                        fig3.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        st.plotly_chart(fig3, use_container_width=True)
