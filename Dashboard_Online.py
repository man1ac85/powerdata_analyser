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

# --- DATENBANK & KONFIGURATION ---
st.set_page_config(page_title="Advanced Power Data Analyser", layout="wide")
@st.cache_data(ttl=3600, show_spinner=False)
def get_athlete_stats_from_intervals(api_key, athlete_id):
    auth = HTTPBasicAuth('API_KEY', api_key)
    
    # Intervals.icu erwartet meist ein "i" vor der ID, falls es keine "0" ist
    # Fallback auf "0" (Account-Besitzer), falls ID in der DB fehlt (z.B. bei alten Einträgen)
    if pd.isna(athlete_id) or str(athlete_id).strip() in ["", "None", "nan"]:
        clean_id = "0"
    else:
        clean_id = str(athlete_id).strip()
        
    if clean_id != "0" and not clean_id.lower().startswith("i"):
        clean_id = f"i{clean_id}"
        
    base_url = f"https://intervals.icu/api/v1/athlete/{clean_id}"
    
    # Defaults festlegen
    stats = {
        "Name": "Unbekannt", 
        "FTP": "-", 
        "Weight": "-",    # Standardwert auf "-" gesetzt
        "Max HR": "-", 
        "Age": "-"
    }
    
    try:
        # 1. Profil abrufen (Intervals API nutzt direkt /athlete/{id})
        res_profile = requests.get(base_url, auth=auth, timeout=10)
        if res_profile.status_code == 200:
            data = res_profile.json()
            stats["Name"] = data.get('name', "Unbekannt")
            
            # Gewicht: Mehrere Varianten abprüfen (oft icu_weight)
            weight = data.get('icu_weight') or data.get('weight') or data.get('weight_kg')
            if not weight:
                # Fallback: Dynamische Suche über alle Profil-Felder
                for k, v in data.items():
                    if "weight" in k.lower() or "gewicht" in k.lower():
                        weight = v
                        break
                        
            if weight:
                try:
                    # Extrahieren der Zahl, selbst wenn z.B. "70 kg" als String zurückkommt
                    match_w = re.search(r'\d+(\.\d+)?', str(weight))
                    if match_w:
                        stats["Weight"] = round(float(match_w.group()), 1)
                except Exception:
                    pass
            
            # Alter: Verschiedene Keys prüfen (Intervals.icu nutzt oft icu_date_of_birth)
            dob = data.get('icu_date_of_birth') or data.get('dob') or data.get('birth_date') or data.get('birthDate')
            if not dob:
                # Fallback: Dynamische Suche über alle Profil-Felder
                for k, v in data.items():
                    if "birth" in k.lower() or "dob" in k.lower() or "geburt" in k.lower():
                        dob = v
                        break
                        
            if dob and isinstance(dob, str):
                try:
                    # Extrahiert das Datum für exakte Altersberechnung (Tage und Monate berücksichtigt)
                    match_date = re.search(r'(\d{4})-(\d{2})-(\d{2})', dob)
                    if match_date:
                        b_year, b_month, b_day = map(int, match_date.groups())
                        today = datetime.now()
                        stats["Age"] = today.year - b_year - ((today.month, today.day) < (b_month, b_day))
                    else:
                        # Fallback auf reines Jahr, falls das Format abweicht
                        match_y = re.search(r'\d{4}', dob)
                        if match_y:
                            stats["Age"] = datetime.now().year - int(match_y.group())
                except Exception:
                    pass
        
        # 2. Sport Settings abrufen
        res_sports = requests.get(f"{base_url}/sport-settings", auth=auth, timeout=10)
        if res_sports.status_code == 200:
            sports = res_sports.json()
            # Finde "Ride" (auch falls es an 2. oder 3. Stelle in der Liste steht)
            cycling = next((s for s in sports if "Ride" in s.get('types', [])), None)
            if cycling:
                stats["FTP"] = cycling.get('ftp', 0)
                stats["Max HR"] = cycling.get('max_hr', '-')
                
    except Exception as e:
        st.error(f"API Sync Fehler: {e}")
        
    return stats
def get_db_connection():
    # Streamlit connection (nutzt automatisch psycopg3, wenn in requirements.txt)
    return st.connection("postgresql", type="sql", url=st.secrets["DB_URL"])

# --- LOGIN FUNKTION ---
def check_login(username, password):
    conn = get_db_connection()
    result = conn.query("SELECT id, password_hash, role FROM users WHERE name = :name", params={"name": username}, ttl=0)
    if not result.empty:
        user = result.iloc[0]
        stored_hash = user['password_hash']
        role = user['role']
        user_id = int(user['id'])
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        if input_hash == stored_hash:
            return True, role, user_id
    return False, None, None

# --- SESSION STATES ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'role' not in st.session_state: st.session_state['role'] = None
if 'user' not in st.session_state: st.session_state['user'] = None
if 'user_id' not in st.session_state: st.session_state['user_id'] = None

# --- LOGIN-TOR ---
if not st.session_state['logged_in']:
    st.title("🔒 Login erforderlich")
    user_in = st.text_input("Benutzername")
    pass_in = st.text_input("Passwort", type="password")
    if st.button("Anmelden"):
        is_valid, role, user_id = check_login(user_in, pass_in)
        if is_valid:
            st.session_state['logged_in'] = True
            st.session_state['user'] = user_in
            st.session_state['role'] = role
            st.session_state['user_id'] = user_id
            st.rerun()
        else: st.error("Benutzername oder Passwort falsch!")
    st.stop()

# --- SESSION STATES (Rest) ---
if 'manual_intervals' not in st.session_state: st.session_state['manual_intervals'] = []
if 'erfassungs_modus' not in st.session_state: st.session_state['erfassungs_modus'] = "Automatisch (Algorithmus)"
if 'overwrite_warning' not in st.session_state: st.session_state['overwrite_warning'] = False
if 'interval_mismatch_warning' not in st.session_state: st.session_state['interval_mismatch_warning'] = False
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
    st.cache_data.clear()
    return True, f"Athlet '{name}' angelegt!"

@st.cache_data(ttl=300, show_spinner=False)
def get_authorized_athletes(current_user_name, role, user_id):
    conn = get_db_connection()
    try:
        if role == 'admin':
            return conn.query("SELECT * FROM users", ttl=0)
        elif role == 'trainer':
            return conn.query("SELECT * FROM users WHERE trainer_id = :tid OR name = :name", 
                              params={"tid": user_id, "name": current_user_name}, ttl=0)
        else:
            return conn.query("SELECT * FROM users WHERE name = :name", 
                              params={"name": current_user_name}, ttl=0)
    except Exception as e:
        st.error(f"Datenbankfehler in get_authorized_athletes: {e}")
        return pd.DataFrame() # Leeres DataFrame zurückgeben bei Fehler
        
def load_all_athletes():
    return get_authorized_athletes(st.session_state['user'], 'admin', st.session_state.get('user_id'))

def check_duplicate_workout(date, workout_type, structure, user_id=None, intervals_activity_id=None):
    conn = get_db_connection()
    
    if intervals_activity_id:
        # 1. Höchste Prio: Eindeutige Cloud-ID checken
        result = conn.query("SELECT id FROM workouts WHERE intervals_activity_id = :act_id",
                            params={"act_id": str(intervals_activity_id)}, ttl=0)
        if not result.empty:
            return result.iloc[0]['id']
            
    # 2. Fallback für rein lokale FIT-Dateien (ohne Cloud-ID)
    if user_id:
        result = conn.query("SELECT id FROM workouts WHERE date = :date AND type = :type AND structure = :structure AND user_id = :uid",
                            params={"date": date, "type": workout_type, "structure": structure, "uid": user_id}, ttl=0)
    else:
        result = conn.query("SELECT id FROM workouts WHERE date = :date AND type = :type AND structure = :structure AND user_id IS NULL",
                            params={"date": date, "type": workout_type, "structure": structure}, ttl=0)
    return result.iloc[0]['id'] if not result.empty else None

def save_workout_to_db(metadata, interval_list, overwrite_id=None):
    conn = get_db_connection()
    with conn.session as s:
        if overwrite_id:
            s.execute(text("DELETE FROM workouts WHERE id = :id"), {"id": overwrite_id})
        
        # Workout einfügen
        res = s.execute(text("""
            INSERT INTO workouts (filename, date, type, structure, avg_power, max_power, intervals_activity_id, user_id) 
            VALUES (:filename, :date, :type, :structure, :avg_p, :max_p, :act_id, :uid) RETURNING id
        """), {
            "filename": metadata['filename'], "date": metadata['date'], "type": metadata['type'], 
            "structure": metadata['structure'], "avg_p": metadata['avg_power'], "max_p": metadata['max_power'],
            "act_id": metadata.get('intervals_activity_id'), "uid": metadata.get('user_id')
        })
        workout_id = res.scalar()
        for row in interval_list:
            # Konvertiere jeden Wert sicher in einen Python-Typ (float oder int)
            s.execute(text("""
                INSERT INTO intervals (workout_id, interval_num, avg_power, avg_hr, max_hr, duration_sec, std_hr, avg_hr_p, "NP_int", intervall_eff) 
                VALUES (:wid, :num, :ap, :ahr, :mhr, :dur, :std, :ahrp, :np, :eff)
            """), {
                "wid": int(workout_id),
                "num": int(row['Intervall']),
                "ap": float(row['Ø Watt']),
                "ahr": float(row['Ø HF']),
                "mhr": float(row['Max HF']),
                "dur": int(row['Dauer_sec']),
                "std": float(row['Δ HF+-']),
                "ahrp": float(row['Ø HF_P (20-80)']),
                "np": float(row['NP']),
                "eff": float(row['Efficiency'])
            })
        s.commit()
    st.cache_data.clear()
    return True, "Daten erfolgreich in die Datenbank übernommen."

@st.cache_data(ttl=60, show_spinner=False)
def fetch_workouts_from_db(uid_val):
    conn = get_db_connection()
    return conn.query("SELECT * FROM workouts WHERE user_id = :uid", params={"uid": uid_val}, ttl=0)

@st.cache_data(ttl=60, show_spinner=False)
def fetch_compare_from_db(selected_ids):
    if not selected_ids: return pd.DataFrame()
    conn = get_db_connection()
    ids_string = ",".join(map(str, selected_ids))
    return conn.query(f"SELECT i.*, w.date, w.type FROM intervals i JOIN workouts w ON i.workout_id = w.id WHERE i.workout_id IN ({ids_string})", ttl=0)

def transfer_to_manual(timestamps):
    st.session_state['manual_intervals'] = timestamps
    st.session_state['erfassungs_modus'] = "Manuell (Grafische Auswahl)"
    st.session_state['overwrite_warning'] = False
    st.session_state['interval_mismatch_warning'] = False

# --- API CLOUD COCKPIT ---
@st.cache_data(ttl=300, show_spinner=False)
def fetch_calendar_events(api_key, start_dt, end_dt):
    url = "https://intervals.icu/api/v1/athlete/0/activities"
    params = {"oldest": start_dt.strftime('%Y-%m-%d'), "newest": end_dt.strftime('%Y-%m-%d')}
    try:
        response = requests.get(url, params=params, auth=HTTPBasicAuth('API_KEY', api_key), timeout=12)
        if response.status_code == 200: return response.json(), None
        return None, f"Fehler {response.status_code}"
    except Exception as e: return None, str(e)

@st.cache_data(ttl=3600, show_spinner=False)
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
        st.session_state['user_id'] = None
        st.rerun()

nav_options = ["Training einlesen", "Daten & Auswertung"]
if st.session_state.get('role') == 'admin':
    nav_options.append("👤 Athleten verwalten")
nav_mode = st.sidebar.radio("Navigation", nav_options)

# --- ADMIN-CHECK ---
if nav_mode == "👤 Athleten verwalten":
    # WICHTIG: Alles ab hier muss eingerückt sein!
    if st.session_state.get('role') == 'admin':
        st.subheader("🛠️ Admin: Athleten verwalten")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### ➕ Neuen Athleten anlegen")
            with st.form("new_athlete_form", clear_on_submit=True):
                name_in = st.text_input("Name:")
                key_in = st.text_input("API Key:", type="password")
                id_in = st.text_input("Intervals.icu ID (z.B. 75948):")
                pwd_in = st.text_input("Start-Passwort:", type="password")
                submitted = st.form_submit_button("Speichern")
                
                if submitted:
                    if name_in and key_in and pwd_in and id_in:
                        success, message = add_new_athlete(name_in, key_in, pwd_in, id_in)
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
                    st.cache_data.clear()
                    st.success(f"Athlet {del_name} wurde gelöscht.")
                    st.rerun()
    else:
        st.error("Zugriff verweigert! Nur für den Administrator.")

elif nav_mode == "Training einlesen":
    if 'df' not in st.session_state: st.session_state['df'] = pd.DataFrame()
    
    df_all_users = get_authorized_athletes(st.session_state['user'], st.session_state['role'], st.session_state.get('user_id'))
    selected_activity_id = None
    filename = ""
    api_k = None
    ftp_val = 0
    active_user_id = None
    default_min_power = 185

    st.sidebar.header("👤 Aktiver Athlet")
    options_list = ["Lokal (.fit-Datei)"] + (df_all_users["name"].tolist() if not df_all_users.empty else [])
    
    default_index = 0
    if st.session_state['user'] in options_list:
        default_index = options_list.index(st.session_state['user'])
        
    user_select = st.sidebar.selectbox("Wer hat trainiert?", options=options_list, index=default_index, key="user_select_key")

    if user_select != "Lokal (.fit-Datei)" and not df_all_users.empty:
        active_user_row = df_all_users.loc[df_all_users["name"] == user_select].iloc[0]
        api_k = active_user_row["api_key"]
        active_user_id = int(active_user_row["id"])
        intervals_id = active_user_row.get("intervals_id", "0")
        
        if 'ftp_cache' not in st.session_state: st.session_state['ftp_cache'] = {}
        if active_user_id not in st.session_state['ftp_cache']:
            stats = get_athlete_stats_from_intervals(api_k, intervals_id)
            st.session_state['ftp_cache'][active_user_id] = stats.get("FTP", 0)
            
        ftp_val = st.session_state['ftp_cache'][active_user_id]
        try:
            if ftp_val and str(ftp_val) != "-":
                calc_power = int(float(ftp_val) * 0.7)
                default_min_power = max(50, min(calc_power, 400)) # Zwischen 50 und 400 Watt deckeln
        except Exception:
            pass

        start_date = st.sidebar.date_input("Start-Datum", datetime.now() - timedelta(days=60), format="DD.MM.YYYY")
        end_date = st.sidebar.date_input("End-Datum", datetime.now(), format="DD.MM.YYYY")
        
        with st.spinner("Synchronisiere Aktivitäten..."):
            events, err = fetch_calendar_events(api_k, start_date, end_date)
            
        if events:
            conn = get_db_connection()
            try:
                # ttl=0 erzwingt den Live-Abgleich mit Supabase (umgeht den Cache)
                existing_workouts = conn.query("SELECT intervals_activity_id FROM workouts WHERE user_id = :uid AND intervals_activity_id IS NOT NULL", params={"uid": active_user_id}, ttl=0)
                existing_ids = existing_workouts['intervals_activity_id'].astype(str).tolist() if not existing_workouts.empty else []
            except Exception:
                existing_ids = []
                
            st.markdown("### 📋 Intervals.icu Trainingdata")
            # Dummy-Spalte am Ende fängt den restlichen Platz ab, damit der Filter an die Tabelle heranrückt
            col_table, col_filter, _ = st.columns([5, 3, 6])
            
            with col_filter:
                st.markdown("##### 🔍 Filter")
                preselected_tags = st.multiselect("Intensität:", ["HIT", "MIT", "LIT", "GA", "RSH"], default=["HIT", "MIT", "LIT", "GA", "RSH"])
                custom_search = st.text_input("Freitext-Suche:", value="")
                
            filtered_events = []
            for e in events:
                if not e.get("id"): continue
                ev_name = e.get("name") or "Fahrt"
                # Exakte Suche der Tags (Case-Sensitive), um z.B. das Wort "mit" von "MIT" zu unterscheiden
                matches_tag = any(tag in ev_name for tag in preselected_tags) if preselected_tags else True
                matches_custom = custom_search.lower() in ev_name.lower() if custom_search else True
                if matches_tag and matches_custom:
                    date_raw = e.get("start_date_local", "0000-00-00")[:10]
                    date_str = f"{date_raw[8:10]}.{date_raw[5:7]}.{date_raw[0:4]}" if len(date_raw) == 10 else date_raw
                    filtered_events.append({
                        "Datum": date_str, 
                        "Name": ev_name,
                        "Status": "In Database" if str(e.get("id")) in existing_ids else "New",
                        "ID": str(e.get("id"))
                    })
                    
            with col_table:
                if filtered_events:
                    df_cockpit = pd.DataFrame(filtered_events)
                    # ID-Spalte ausblenden, Status ist sichtbar
                    sel = st.dataframe(
                        df_cockpit, 
                        use_container_width=False, 
                        on_select="rerun", 
                        selection_mode="single-row", 
                        hide_index=True, 
                        column_config={
                            "ID": None,
                            "Datum": st.column_config.TextColumn("Datum", width=90),
                            "Name": st.column_config.TextColumn("Name", width=300),
                            "Status": st.column_config.TextColumn("Status", width=90)
                        }
                    )
                    if sel and len(sel.get("selection", {}).get("rows", [])) > 0:
                        idx = sel["selection"]["rows"][0]
                        selected_activity_id = df_cockpit.iloc[idx]["ID"]
                        filename = df_cockpit.iloc[idx]["Name"]
                        with st.spinner("Lade Workout..."):
                            bin_data, _ = download_original_fit_file(api_k, selected_activity_id)
                            if bin_data:
                                fitfile = fitparse.FitFile(io.BytesIO(bin_data))
                                st.session_state['df'] = pd.DataFrame([r.get_values() for r in fitfile.get_messages('record')])
                                st.session_state['df']['timestamp'] = pd.to_datetime(st.session_state['df']['timestamp'])
                                st.session_state['df'].set_index('timestamp', inplace=True)
                else:
                    st.info("Keine Aktivitäten passend zu den Filtern gefunden.")
        elif err:
            st.error(f"API-Fehler: {err}")
        else:
            st.info("Keine Aktivitäten gefunden.")
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
            match_type = re.search(r'(LIT|MIT|HIT|GA|RSH)', filename, re.IGNORECASE)
            detected_type = match_type.group(1).upper() if match_type else "UNKNOWN"
            is_ride_analysis = detected_type in ["GA", "RSH"]
            
            if is_ride_analysis:
                st.subheader("Advanced Ride Analyser")
            else:
                st.subheader("Advanced Intervall Analyzer")
                
            match_structure = re.search(r'(\d+)[xX](\d+)', filename)
            init_intervals = int(match_structure.group(1)) if match_structure else 4
            init_duration_min = int(match_structure.group(2)) if match_structure else 15

            is_admin = st.session_state.get('role') == 'admin'

            if not is_ride_analysis:
                if is_admin:
                    col1, col2, col3, col_int, _ = st.columns([2, 2, 2, 2, 7])
                    with col3: mode_type = st.selectbox("Erfassungs-Modus", ["Automatisch (Algorithmus)", "Manuell (Grafische Auswahl)"], key='erfassungs_modus')
                else:
                    col1, col2, col_int, _ = st.columns([2, 2, 2, 9])
                    mode_type = "Automatisch (Algorithmus)"
                
                with col1: expected_intervals = st.number_input("Erwartete Anzahl Intervalle", value=init_intervals, min_value=1)
                with col2: expected_duration_min = st.number_input("Erwartete Intervalllänge (Min)", value=init_duration_min, min_value=1)
                
                interval_placeholder = col_int.empty()
                
                if is_admin:
                    c4, c5, _ = st.columns([1, 1, 2])
                    with c4: min_power = st.slider("Mindestleistung (Watt)", 50, 400, default_min_power, step=5)
                    with c5: sg_win = st.slider("Filterfenster", 11, 121, 45, step=2)
                else:
                    c4, _ = st.columns([1, 3])
                    with c4: min_power = st.slider("Mindestleistung (Watt)", 50, 400, default_min_power, step=5)
                    sg_win = 45
            else:
                equidistant = st.checkbox("Äquidistant (Gleiche Zeitabschnitte)")
                expected_intervals = 1
                expected_duration_min = 60
                mode_type = "Automatisch (Algorithmus)"
                min_power = default_min_power
                sg_win = 45
            
            if 'power' in df.columns:
                df['p_clean'] = df['power'].fillna(0)
                df['p_sg'] = savgol_filter(df['p_clean'].rolling(7, center=True, min_periods=1).median(), window_length=sg_win, polyorder=2)
                
                # Globaler 30s-Gleitdurchschnitt (WICHTIG für korrekte Intervall-NP!)
                df['power_roll_30'] = df['p_clean'].rolling(window=30, min_periods=1).mean()
                
                if is_ride_analysis:
                    total_sec = len(df)
                    if equidistant:
                        num_chunks = int((total_sec + 1800) // 3600)
                        if num_chunks < 1: num_chunks = 1
                        chunk_len = total_sec / num_chunks
                        df['block_id'] = [int(i // chunk_len) + 1 for i in range(total_sec)]
                        df.loc[df['block_id'] > num_chunks, 'block_id'] = num_chunks
                    else:
                        chunk_len = 3600
                        df['block_id'] = [int(i // chunk_len) + 1 for i in range(total_sec)]
                        
                    df['is_interval'] = True
                    num_intervals = df['block_id'].nunique()
                    expected_intervals = num_intervals
                    # Abwechselnd markieren für optische Abgrenzung im Plot
                    df['highlight'] = df.apply(lambda row: row['power'] if row['block_id'] % 2 != 0 else None, axis=1)
                    
                elif mode_type == "Automatisch (Algorithmus)":
                    target = expected_duration_min * 60
                    df['roll'] = df['p_sg'].rolling(window=target, min_periods=int(target*0.5), center=True).mean()
                    scores = df['roll'].fillna(0).values
                    candidates = [i for i in range(len(df)) if scores[i] >= min_power and scores[i] == max(scores[max(0, i-target//2):min(len(df), i+target//2)])]
                    candidates = sorted(candidates, key=lambda idx: scores[idx], reverse=True)[:expected_intervals]
                    
                    is_interval_array = [False] * len(df)
                    block_id_array = [0] * len(df)
                    for b_num, c_idx in enumerate(sorted(candidates), 1):
                        l, r = c_idx - target//2, c_idx + target//2
                        for j in range(max(0, l), min(len(df), r)):
                            is_interval_array[j] = True
                            block_id_array[j] = b_num
                    df['is_interval'] = is_interval_array
                    df['block_id'] = block_id_array
                    num_intervals = (df['is_interval'].astype(int).diff() == 1).sum()
                    df['highlight'] = df.apply(lambda row: row['power'] if row['is_interval'] else None, axis=1)
                else:
                    df['is_interval'] = False
                    df['block_id'] = 0
                    current_block_id = 1
                    st.session_state['manual_intervals'].sort(key=lambda x: x[0])
                    for s_t, e_t in st.session_state['manual_intervals']:
                        mask = (df.index >= s_t) & (df.index <= e_t)
                        df.loc[mask, 'is_interval'] = True
                        df.loc[mask, 'block_id'] = current_block_id
                        current_block_id += 1
                    num_intervals = len(st.session_state['manual_intervals'])
                    df['highlight'] = df.apply(lambda row: row['power'] if row['is_interval'] else None, axis=1)
                
                # --- METRIKEN ---
                if is_ride_analysis:
                    workout_structure = "Ride"
                else:
                    workout_structure = f"{expected_intervals}x{expected_duration_min}"
                
                mean_p = df['power'].mean() if not df['power'].empty else 0
                overall_avg_p = int(mean_p) if pd.notna(mean_p) else 0
                
                max_p = df['power'].max() if not df['power'].empty else 0
                overall_max_p = int(max_p) if pd.notna(max_p) else 0
                
                if 'power_roll_30' in df.columns and not df.empty:
                    mean_p4 = (df['power_roll_30'] ** 4).mean()
                    overall_np = int(mean_p4 ** 0.25) if pd.notna(mean_p4) else 0
                else:
                    overall_np = 0
                
                if 'heart_rate' in df.columns and not df['heart_rate'].dropna().empty:
                    mean_hr = df['heart_rate'].mean()
                    overall_avg_hr = int(mean_hr) if pd.notna(mean_hr) else 0
                    max_hr = df['heart_rate'].max()
                    overall_max_hr = int(max_hr) if pd.notna(max_hr) else 0
                else:
                    overall_avg_hr = 0
                    overall_max_hr = 0
                    
                ftp_for_tss = float(ftp_val) if ftp_val and str(ftp_val) != "-" else 250
                overall_tss = int((len(df) * (overall_np ** 2)) / ((ftp_for_tss ** 2) * 36)) if ftp_for_tss > 0 else 0

                st.markdown("""
                <style>
                .small-metric { font-size: 1.65rem !important; font-weight: bold; line-height: 1.2; }
                .small-metric-label { font-size: 14px; color: rgb(163, 168, 184); margin-bottom: 2px; }
                </style>
                """, unsafe_allow_html=True)

                cols = st.columns(6)
                cols[0].markdown(f'<div class="small-metric-label">Ø Leistung</div><div class="small-metric">{overall_avg_p} W</div>', unsafe_allow_html=True)
                cols[1].markdown(f'<div class="small-metric-label">Normalized Power</div><div class="small-metric">{overall_np} W</div>', unsafe_allow_html=True)
                cols[2].markdown(f'<div class="small-metric-label">Max Leistung</div><div class="small-metric">{overall_max_p} W</div>', unsafe_allow_html=True)
                cols[3].markdown(f'<div class="small-metric-label">Ø HF</div><div class="small-metric">{overall_avg_hr} bpm</div>', unsafe_allow_html=True)
                cols[4].markdown(f'<div class="small-metric-label">Max HF</div><div class="small-metric">{overall_max_hr} bpm</div>', unsafe_allow_html=True)
                cols[5].markdown(f'<div class="small-metric-label">TSS Score</div><div class="small-metric">{overall_tss}</div>', unsafe_allow_html=True)
                
                if not is_ride_analysis:
                    color = "#33CC33" if num_intervals == expected_intervals else "#FF3333"
                    interval_placeholder.markdown(f'<div style="font-size: 14px; color: rgb(163, 168, 184); margin-bottom: 2px;">Intervalle erkannt</div><div style="font-size: 1.65rem; font-weight: bold; line-height: 1.2; color: {color};">{num_intervals} / {expected_intervals}</div>', unsafe_allow_html=True)
                
                # --- BERECHNUNG DER KENNWERTE UND SPEICHERN ---
                intervals_calculated = []
                if num_intervals > 0:
                    unique_blocks = df[df['is_interval']]['block_id'].unique()
                    for idx, b_id in enumerate(unique_blocks, start=1):
                        block_df = df[df['block_id'] == b_id]
                        if block_df.empty: continue
                        
                        avg_p = int(block_df['power'].mean())
                        
                        # NP exakt berechnen: Greift auf den globalen 30s-Trend zu und rechnet ihn für das Intervall aus
                        np_val = (block_df['power_roll_30'] ** 4).mean() ** 0.25 if not block_df.empty else 0
                        
                        # HR fallback
                        if 'heart_rate' in block_df.columns:
                            hr_data = block_df['heart_rate'].dropna()
                        else:
                            hr_data = pd.Series(dtype=float)
                            
                        avg_hr = int(hr_data.mean()) if not hr_data.empty else 0
                        max_hr = int(hr_data.max()) if not hr_data.empty else 0
                        std_hr = round(hr_data.std(), 1) if not hr_data.empty else 0
                        
                        # Efficiency berechnen (NP / avg HR)
                        efficiency = np_val / avg_hr if avg_hr > 0 else 0
                        
                        if not hr_data.empty:
                            lower_p = hr_data.quantile(0.20)
                            upper_p = hr_data.quantile(0.80)
                            avg_hr_p = int(hr_data[(hr_data >= lower_p) & (hr_data <= upper_p)].mean())
                        else: 
                            avg_hr_p = 0
                            
                        intervals_calculated.append({
                            "Intervall": idx, 
                            "Ø Watt": avg_p, 
                            "Ø HF": avg_hr, 
                            "Efficiency": round(efficiency, 2),
                            "NP": round(np_val, 1),
                            "Max HF": max_hr, 
                            "Δ HF+-": std_hr, 
                            "Ø HF_P (20-80)": avg_hr_p, 
                            "Dauer (Min)": round(len(block_df) / 60, 2),
                            "Dauer_sec": len(block_df) 
                        })
                
                if intervals_calculated:
                    st.subheader("Kennwerte pro Intervall")
                    col_tab1, col_tab2 = st.columns([2, 1])
                    with col_tab1:
                        df_res = pd.DataFrame(intervals_calculated)
                        display_df = df_res.drop(columns=['Dauer_sec'])
                        styled_df = display_df.style.format({
                            "Ø Watt": "{:.0f}", "Ø HF": "{:.0f}", "Efficiency": "{:.2f}", "NP": "{:.1f}",
                            "Max HF": "{:.0f}", "Δ HF+-": "{:.1f}", "Ø HF_P (20-80)": "{:.0f}", "Dauer (Min)": "{:.1f}"
                        }).set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                    
                    workout_date = df.index.min().strftime('%Y-%m-%d')
                    
                    metadata = {
                        "filename": filename, 
                        "date": workout_date, 
                        "type": detected_type, 
                        "structure": workout_structure, 
                        "avg_power": int(df['power'].mean()), 
                        "max_power": int(df['power'].max()),
                        "intervals_activity_id": selected_activity_id,
                        "user_id": active_user_id
                    }
                    
                    st.markdown("### Speichern")
                    if st.session_state.get('overwrite_warning'):
                        st.warning("Workout ist bereits in der Datenbank. Überschreiben?")
                        col_w1, col_w2 = st.columns([1, 3])
                        with col_w1:
                            if st.button("Ja", type="primary"):
                                success, msg = save_workout_to_db(metadata, intervals_calculated, overwrite_id=st.session_state['workout_to_overwrite'])
                                if success: 
                                    st.success(msg)
                                    st.session_state['overwrite_warning'] = False
                                else:
                                    st.error(msg)
                        with col_w2:
                            if st.button("Abbrechen & Zurück"): 
                                st.session_state['overwrite_warning'] = False
                                st.rerun()
                    elif st.session_state.get('interval_mismatch_warning'):
                        st.warning(f"Achtung: Es wurden {num_intervals} Intervalle erkannt, aber {expected_intervals} erwartet. Trotzdem speichern?")
                        col_w1, col_w2 = st.columns([1, 3])
                        with col_w1:
                            if st.button("Ja", type="primary", key="btn_yes_mismatch"):
                                st.session_state['interval_mismatch_warning'] = False
                                dup_id = check_duplicate_workout(workout_date, detected_type, workout_structure, active_user_id, selected_activity_id)
                                if dup_id: 
                                    st.session_state['overwrite_warning'] = True
                                    st.session_state['workout_to_overwrite'] = dup_id
                                    st.rerun()
                                else: 
                                    success, message = save_workout_to_db(metadata, intervals_calculated)
                                    if success: st.success(message)
                                    else: st.error(message)
                        with col_w2:
                            if st.button("Abbrechen & Zurück", key="btn_no_mismatch"): 
                                st.session_state['interval_mismatch_warning'] = False
                                st.rerun()
                    else:
                        col_db1, col_db2 = st.columns([1, 3])
                        with col_db1:
                            if st.button("In Datenbank übernehmen"):
                                if num_intervals != expected_intervals:
                                    st.session_state['interval_mismatch_warning'] = True
                                    st.rerun()
                                else:
                                    dup_id = check_duplicate_workout(workout_date, detected_type, workout_structure, active_user_id, selected_activity_id)
                                    if dup_id: 
                                        st.session_state['overwrite_warning'] = True
                                        st.session_state['workout_to_overwrite'] = dup_id
                                        st.rerun()
                                    else: 
                                        success, message = save_workout_to_db(metadata, intervals_calculated)
                                        if success: st.success(message)
                                        else: st.error(message)
                                    
                        if mode_type == "Automatisch (Algorithmus)" and is_admin and not is_ride_analysis:
                            with col_db2:
                                auto_blocks_timestamps = [(df[df['block_id'] == b].index.min(), df[df['block_id'] == b].index.max()) for b in df[df['is_interval']]['block_id'].unique()]
                                st.button("Intervalle manuell nachjustieren", on_click=transfer_to_manual, args=(auto_blocks_timestamps,))

                # PLOTLY GRAPH (Full subplots)
                st.subheader("Trainingsdaten & Analyse")
                num_rows = 3 if is_admin else 2
                fig_main = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04)
                fig_main.add_trace(go.Scatter(x=df.index, y=df.get('power'), mode='lines', name='Rohleistung', line=dict(color='rgba(150, 150, 150, 0.4)', width=1)), row=1, col=1)
                fig_main.add_trace(go.Scatter(x=df.index, y=df.get('p_sg'), mode='lines', name='Sav-Gol Trend', line=dict(color='rgba(51, 153, 255, 0.6)', width=1)), row=1, col=1)
                fig_main.add_trace(go.Scatter(x=df.index, y=df.get('highlight'), mode='lines', name='Intervall (Power)', line=dict(color='#FFA500', width=2.5)), row=1, col=1)
                
                if 'heart_rate' in df.columns and not df['heart_rate'].isna().all():
                    df['hr_clean'] = df['heart_rate'].ffill().bfill()
                    if is_ride_analysis:
                        df['hr_highlight'] = df.apply(lambda row: row['hr_clean'] if row['block_id'] % 2 != 0 else None, axis=1)
                    else:
                        df['hr_highlight'] = df.apply(lambda row: row['hr_clean'] if row['is_interval'] else None, axis=1)
                    fig_main.add_trace(go.Scatter(x=df.index, y=df['hr_clean'], mode='lines', name='Herzfrequenz', line=dict(color='rgba(255, 102, 102, 0.5)', width=1.5)), row=2, col=1)
                    fig_main.add_trace(go.Scatter(x=df.index, y=df['hr_highlight'], mode='lines', name='Intervall (HF)', line=dict(color='#FF3333', width=2.5)), row=2, col=1)
                    
                if is_admin:
                    df['p_deriv'] = savgol_filter(df['p_clean'].rolling(7, center=True, min_periods=1).median(), window_length=sg_win, polyorder=2, deriv=1)
                    fig_main.add_trace(go.Scatter(x=df.index, y=df.get('p_deriv'), mode='lines', name='Steigung (Ableitung)', line=dict(color='#33CC33', width=1.2)), row=3, col=1)

                plot_height = 800 if is_admin else 550
                fig_main.update_layout(template="plotly_dark", height=plot_height, hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0), legend=dict(yanchor="top", y=1))
                selected_data = st.plotly_chart(fig_main, use_container_width=True, on_select="rerun")
                
                if mode_type == "Manuell (Grafische Auswahl)":
                    st.markdown("### Manuelle Intervall-Bearbeitung")
                    start_t, end_t = None, None
                    if selected_data:
                        box = selected_data.get("selection", {}).get("box", None) or selected_data.get("box", None)
                        if box and isinstance(box, list): box = box[0]
                        if isinstance(box, dict) and "x" in box: start_t, end_t = box["x"][0], box["x"][1]
                    
                    if start_t and end_t:
                        st.write(f"Auswahl: {start_t} bis {end_t}")
                        if st.button("Bereich als Intervall hinzufügen"):
                            st.session_state['manual_intervals'].append((pd.to_datetime(start_t), pd.to_datetime(end_t)))
                            st.rerun()
                            
                    if st.session_state['manual_intervals']:
                        intervals_to_keep = []
                        for idx, (s, e) in enumerate(st.session_state['manual_intervals'], start=1):
                            col_text, col_btn = st.columns([0.8, 0.2])
                            col_text.write(f"Intervall {idx}: {s.strftime('%H:%M:%S')} bis {e.strftime('%H:%M:%S')}")
                            if col_btn.button("Löschen", key=f"del_manual_{idx}"): pass 
                            else: intervals_to_keep.append((s, e))
                        if len(intervals_to_keep) != len(st.session_state['manual_intervals']):
                            st.session_state['manual_intervals'] = intervals_to_keep
                            st.rerun()
                
        except Exception as e: st.error(f"Fehler: {e}")
    else:
        st.info("Bitte Athlet wählen oder Datei hochladen.")

elif nav_mode == "Daten & Auswertung":
    st.subheader("📊 Daten & Auswertung")
    
    authorized_athletes = get_authorized_athletes(st.session_state['user'], st.session_state['role'], st.session_state.get('user_id'))
    
    if authorized_athletes.empty:
        st.warning("Keine Athleten gefunden.")
    else:
        # Layout: 3 Spalten
        c1, c2, c3 = st.columns([1, 1, 2])
        
        with c1:
            selected_name = st.selectbox("Athlet wählen:", options=authorized_athletes["name"], key="data_eval_athlete_selector")
            athlete_row = authorized_athletes[authorized_athletes["name"] == selected_name].iloc[0]
            filter_type = st.selectbox("Typ-Filter", ["ALLE", "LIT", "MIT", "HIT"], key="data_eval_filter_type")
            search_query = st.text_input("Suche:", key="data_eval_search_input")

        with c2:
            st.markdown("##### 👤 Profil")
            
            # Dynamische Abfrage mittels Helper-Funktion
            stats = get_athlete_stats_from_intervals(athlete_row['api_key'], athlete_row['intervals_id'])
            
            # W/kg Berechnung sicher machen (für den Fall, dass Werte "-" sind)
            ftp = stats["FTP"]
            weight = stats["Weight"]
            
            try:
                w_kg = f"{round(float(ftp) / float(weight), 2)} W/kg" if float(weight) > 0 else "-"
            except (ValueError, TypeError):
                w_kg = "-"
                
            # Schöne Formatierung mit Einheiten (damit bei "-" nicht "- kg" steht)
            str_weight = f"{float(weight):.1f} kg" if str(weight) != "-" else "-"
            str_ftp = f"{ftp} W" if str(ftp) != "-" else "-"
            str_hr = f"{stats['Max HR']} bpm" if str(stats['Max HR']) != "-" else "-"
            
            profile_df = pd.DataFrame([
                [f"{selected_name}", f"Stand: {datetime.now().strftime('%d.%m.%y')}"],
                [f"Alter: {stats['Age']}", f"Gewicht: {str_weight}"],
                [f"FTP: {str_ftp}", w_kg],
                [f"Max HF: {str_hr}", ""]
            ])
            st.table(profile_df.set_axis([' ', '  '], axis=1))

        # --- WORKOUT LOGIK ---
        uid_val = int(athlete_row['id'])
        df_workouts = fetch_workouts_from_db(uid_val)
        
        if df_workouts.empty: 
            st.info(f"Keine Trainingsdaten für {selected_name} gefunden.")
        else:
            if filter_type != "ALLE": 
                df_workouts = df_workouts[df_workouts['type'] == filter_type]
            if search_query:
                df_workouts = df_workouts[df_workouts['filename'].str.contains(search_query, case=False, na=False) | df_workouts['date'].str.contains(search_query, case=False, na=False)]

            # Feste Farbzuordnung für alle angezeigten Workouts erstellen, damit Farben beim Auswählen gleich bleiben
            all_possible_workouts = (df_workouts['date'].str.slice(0, 10) + " (" + df_workouts['type'] + ")").unique()
            all_possible_workouts = sorted(all_possible_workouts)
            plotly_colors = px.colors.qualitative.Plotly
            extended_colors = plotly_colors * (len(all_possible_workouts) // len(plotly_colors) + 1)
            global_color_map = {w: extended_colors[i] for i, w in enumerate(all_possible_workouts)}

            selected_ids = []
            delete_id = None
            for idx, row in df_workouts.iterrows():
                col_check, col_del = st.columns([0.9, 0.1])
                with col_check:
                    if st.checkbox(f"{row['date']} | {row['type']} ({row['structure']}) | {row['filename']}", key=f"eval_check_{row['id']}"):
                        selected_ids.append(row['id'])
                with col_del:
                    if st.button("🗑️", key=f"eval_del_{row['id']}"):
                        delete_id = row['id']
                        
            if delete_id is not None:
                conn = get_db_connection()
                with conn.session as s:
                    s.execute(text("DELETE FROM workouts WHERE id = :id"), {"id": delete_id})
                    s.commit()
                st.cache_data.clear()
                st.rerun()

            if len(selected_ids) >= 1:
                st.markdown("---")
                df_compare = fetch_compare_from_db(selected_ids)
                
                if not df_compare.empty:
                    df_compare['Workout'] = df_compare['date'].str.slice(0, 10) + " (" + df_compare['type'] + ")"
                    
                    # Layout auf 2x2 Grid umstellen
                    c_p1, c_p2 = st.columns(2)
                    c_p3, c_p4 = st.columns(2)
                    
                    with c_p1: 
                        fig1 = px.scatter(df_compare, x="interval_num", y="avg_power", color="Workout", title="Ø Watt", color_discrete_map=global_color_map)
                        fig1.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        fig1.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig1, use_container_width=True)
                    with c_p2:
                        fig_hr = go.Figure()
                        for w in df_compare['Workout'].unique():
                            sub = df_compare[df_compare['Workout'] == w]
                            c = global_color_map.get(w, '#ffffff')
                            fig_hr.add_trace(go.Scatter(x=sub['interval_num'], y=sub['avg_hr'], name=f"{w} (Ø)", 
                                                        error_y=dict(type='data', array=sub['std_hr'], visible=True),
                                                        mode='lines+markers', line=dict(color=c), marker=dict(color=c)))
                            fig_hr.add_trace(go.Scatter(x=sub['interval_num'], y=sub['avg_hr_p'], name=f"{w} (20-80%)", 
                                                        mode='markers', marker=dict(size=8, color=c, symbol='diamond', line=dict(color='white', width=1))))
                        fig_hr.update_layout(title="Ø Herzfrequenz (+- StdDev)", template="plotly_dark")
                        fig_hr.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig_hr, use_container_width=True)
                    with c_p3: 
                        fig3 = px.scatter(df_compare, x="interval_num", y="max_hr", color="Workout", title="Max HF", color_discrete_map=global_color_map)
                        fig3.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        fig3.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig3, use_container_width=True)
                    with c_p4: 
                        fig4 = px.scatter(df_compare, x="interval_num", y="intervall_eff", color="Workout", title="Efficiency (W/bpm)", color_discrete_map=global_color_map)
                        fig4.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        fig4.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig4, use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### Intervall-Werte")
                    
                    # Dynamische Spaltenauswahl
                    optional_cols = st.multiselect(
                        "Zusätzliche Tabellenspalten anzeigen:",
                        options=["Eff.", "Max HF", "± HF", "Min."],
                        default=["Eff.", "Max HF", "± HF", "Min."]
                    )
                    
                    unique_workouts = df_compare['Workout'].unique()
                    
                    num_w = len(unique_workouts)
                    # Dynamische Spaltenbreite: Abhängig von der Anzahl der sichtbaren Metriken (Spalten)
                    # weisen wir jeder Tabelle genug Platz zu, damit sie nicht gequetscht wird und nicht scrollt.
                    table_weight = 3.0 + (len(optional_cols) * 1.2)
                    col_widths = [table_weight] * num_w + [max(0.1, 24 - (num_w * table_weight))]
                    table_cols = st.columns(col_widths)
                    
                    for idx, w in enumerate(unique_workouts):
                        with table_cols[idx]:
                            st.markdown(f"**{w}**")
                            sub_df = df_compare[df_compare['Workout'] == w].copy()
                            
                            display_df = pd.DataFrame()
                            display_df['Int.'] = sub_df['interval_num']
                            display_df['Ø W'] = sub_df['avg_power']
                            display_df['Ø HF'] = sub_df['avg_hr']
                            
                            if "Eff." in optional_cols: display_df['Eff.'] = sub_df['intervall_eff'] if 'intervall_eff' in sub_df.columns else 0
                            if "Max HF" in optional_cols: display_df['Max HF'] = sub_df['max_hr']
                            if "± HF" in optional_cols: display_df['± HF'] = sub_df['std_hr'] if 'std_hr' in sub_df.columns else 0
                            if "Min." in optional_cols: display_df['Min.'] = round(sub_df['duration_sec'] / 60, 1)
                            
                            format_dict = {"Ø W": "{:.0f}", "Ø HF": "{:.0f}"}
                            if "Eff." in optional_cols: format_dict["Eff."] = "{:.2f}"
                            if "Max HF" in optional_cols: format_dict["Max HF"] = "{:.0f}"
                            if "± HF" in optional_cols: format_dict["± HF"] = "{:.1f}"
                            if "Min." in optional_cols: format_dict["Min."] = "{:.1f}"
                            
                            styled_df = display_df.style.format(format_dict).set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                            
                            st.dataframe(styled_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("⚠️ Zu diesem Workout wurden keine Intervall-Daten gefunden (vermutlich ein alter/fehlerhafter Speicherstand). Bitte lösche das Workout über den 🗑️-Button und speichere es neu aus der Cloud ein.")
