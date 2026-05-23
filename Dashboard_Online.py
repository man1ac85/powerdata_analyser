import streamlit as st
import fitparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
import re
from datetime import datetime, timedelta
import os
import requests
from requests.auth import HTTPBasicAuth
import io
import hashlib
from sqlalchemy import create_engine, text

# Das @st.cache_resource ist der Schlüssel! 
# Es sorgt dafür, dass die Engine nur EINMAL erstellt wird.
@st.cache_resource
def get_db_engine():
    # Wir benutzen hier die Secrets und die SSL-Konfiguration
    # Wir nutzen 'connect_args' für SSL, das ist sauberer als in der URL
    return create_engine(
        st.secrets["DB_URL"],
        pool_size=5,
        max_overflow=0,
        pool_pre_ping=True, # Das verhindert, dass Verbindungen "einschlafen"
        connect_args={"sslmode": "require"} 
    )

# --- LOGIN FUNKTION & LOGIK ---
def check_login(username, password):
    engine = get_db_engine()
    query = text("SELECT password_hash, role FROM users WHERE name = :name")
    with engine.connect() as conn:
        result = conn.execute(query, {"name": username}).fetchone()
    if result:
        stored_hash = result[0]
        role = result[1]
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        if input_hash == stored_hash:
            return True, role
    return False, None

# Session States für Login initialisieren
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'role' not in st.session_state: st.session_state['role'] = None
if 'user' not in st.session_state: st.session_state['user'] = None

# --- DAS LOGIN-TOR ---
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
        else:
            st.error("Benutzername oder Passwort falsch!")
    st.stop()

# --- SESSION STATES (Rest) ---
if 'manual_intervals' not in st.session_state: st.session_state['manual_intervals'] = []
if 'erfassungs_modus' not in st.session_state: st.session_state['erfassungs_modus'] = "Automatisch (Algorithmus)"
if 'overwrite_warning' not in st.session_state: st.session_state['overwrite_warning'] = False
if 'workout_to_overwrite' not in st.session_state: st.session_state['workout_to_overwrite'] = None

def init_system_dbs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, date TEXT, type TEXT, structure TEXT, avg_power INTEGER, max_power INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intervals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, workout_id INTEGER, interval_num INTEGER, 
            avg_power INTEGER, avg_hr INTEGER, max_hr INTEGER, duration_sec INTEGER, 
            std_hr REAL, avg_hr_p INTEGER,
            FOREIGN KEY (workout_id) REFERENCES workouts (id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()

    conn = sqlite3.connect(USER_DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, api_key TEXT
        )
    """)
    conn.commit()
    conn.close()

init_system_dbs()

# --- DATENBANK HELFER ---
def add_new_athlete(name, api_key):
    try:
        conn = sqlite3.connect(USER_DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, api_key) VALUES (?, ?)", (name.strip(), api_key.strip()))
        conn.commit()
        conn.close()
        return True, f"Athleten-Profil für '{name}' erfolgreich angelegt!"
    except sqlite3.IntegrityError:
        return False, "Fehler: Ein Athlet mit diesem Namen existiert bereits."

def load_all_athletes():
    conn = sqlite3.connect(USER_DB_NAME)
    df_u = pd.read_sql_query("SELECT * FROM users", conn)
    conn.close()
    return df_u

def check_duplicate_workout(date, workout_type, structure):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM workouts WHERE date = ? AND type = ? AND structure = ?", (date, workout_type, structure))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_workout_to_db(metadata, interval_list, overwrite_id=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if overwrite_id: cursor.execute("DELETE FROM workouts WHERE id = ?", (overwrite_id,))
    
    cursor.execute("""
        INSERT INTO workouts (filename, date, type, structure, avg_power, max_power) VALUES (?, ?, ?, ?, ?, ?)
    """, (metadata['filename'], metadata['date'], metadata['type'], metadata['structure'], metadata['avg_power'], metadata['max_power']))
    
    workout_id = cursor.lastrowid
    for row in interval_list:
        cursor.execute("""
            INSERT INTO intervals (workout_id, interval_num, avg_power, avg_hr, max_hr, duration_sec, std_hr, avg_hr_p) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (workout_id, row['Intervall'], row['Ø Leistung'], row['Ø Herzfrequenz'], row['Max Herzfrequenz'], row['Dauer_sec'], row['Abweichung HF+-'], row['Durschnittliche HF_P (20-80)']))
    conn.commit()
    conn.close()
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
st.set_page_config(page_title="FIT Analyzer Pro Cloud", layout="wide")
st.title("Dashboard BETA: Cloud-Schnittstelle & Smart-Filter Cockpit")

nav_mode = st.sidebar.radio("Navigation", ["Aktuelles Training einlesen", "Historie & Vergleich", "👤 Athleten verwalten"])

if nav_mode == "👤 Athleten verwalten":
    st.subheader("👤 Athleten-Profile verwalten")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### ➕ Neues Athleten-Profil anlegen")
        name_in = st.text_input("Name des Sportlers (z.B. Max):")
        key_in = st.text_input("Intervals.icu API Key:", type="password")
        if st.button("Profil dauerhaft speichern"):
            if name_in and key_in:
                success, message = add_new_athlete(name_in, key_in)
                if success: st.success(message)
                else: st.error(message)
                st.rerun()
    with col_right:
        st.markdown("### 👥 Vorhandene Profile")
        df_all_users = load_all_athletes()
        if df_all_users.empty: st.info("Noch keine Profile hinterlegt.")
        else: st.table(df_all_users[["id", "name"]])

elif nav_mode == "Aktuelles Training einlesen":
    df_all_users = load_all_athletes()
    selected_activity_id = None
    selected_activity_name = ""
    selected_activity_date = ""
    df = pd.DataFrame()
    filename = ""
    api_k = None

    st.sidebar.header("👤 Aktiver Athlet")
    if df_all_users.empty:
        user_select = "Lokal importieren"
    else:
        options_list = ["Lokal (.fit-Datei)"] + df_all_users["name"].tolist()
        user_select = st.sidebar.selectbox("Wer hat trainiert?", options=options_list)

    if user_select != "Lokal (.fit-Datei)" and not df_all_users.empty:
        api_k = df_all_users.loc[df_all_users["name"] == user_select, "api_key"].values[0]
        
        st.sidebar.markdown("---")
        st.sidebar.header("🗓️ Cloud-Zeitraum Filter")
        today_val = datetime.now()
        start_date_input = st.sidebar.date_input("Start-Datum", today_val - timedelta(days=60))
        end_date_input = st.sidebar.date_input("End-Datum", today_val)
        
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Suchbegriff / Wortfilter")
        preselected_tags = st.sidebar.multiselect("Vorauswahl Intensität:", ["MIT", "HIT", "LIT"], default=["MIT", "HIT", "LIT"])
        custom_search = st.sidebar.text_input("Freitext-Wortfilter (z.B. SST):", value="")

        with st.spinner("Synchronisiere Aktivitäten..."):
            events, err = fetch_calendar_events(api_k, start_date_input, end_date_input)
            if events is not None:
                table_rows = []
                for ev in events:
                    if ev.get("id"):
                        ev_date = ev.get("start_date_local", "0000-00-00")[:10]
                        ev_name = ev.get("name") or ev.get("description") or "Unbenannte Fahrt"
                        ev_type = ev.get("type", "Ride")
                        ev_watts = ev.get("icu_average_watts", "-")
                        ev_hr = ev.get("icu_average_hr", "-")
                        
                        matches_tag = any(tag.lower() in ev_name.lower() for tag in preselected_tags) if preselected_tags else True
                        matches_custom = custom_search.lower() in ev_name.lower() if custom_search else True
                        
                        if matches_tag and matches_custom:
                            table_rows.append({
                                "Datum": ev_date, "Name des Trainings": ev_name, "Typ": ev_type, "Avg Watt": ev_watts, "Avg HF": ev_hr, "Internal_ID": ev.get("id")
                            })
                
                if table_rows:
                    st.subheader("📋 Cloud-Trainingsübersicht (Wähle eine Zeile)")
                    df_cockpit = pd.DataFrame(table_rows)
                    selected_row = st.dataframe(
                        df_cockpit.drop(columns=["Internal_ID"]), use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row"
                    )
                    if selected_row and len(selected_row.get("selection", {}).get("rows", [])) > 0:
                        row_idx = selected_row["selection"]["rows"][0]
                        selected_activity_id = df_cockpit.iloc[row_idx]["Internal_ID"]
                        selected_activity_name = df_cockpit.iloc[row_idx]["Name des Trainings"]
                        selected_activity_date = df_cockpit.iloc[row_idx]["Datum"]
                else: st.info("Keine Trainingseinheiten passend zu den Filtern gefunden.")
            else: st.sidebar.error(f"API-Fehler: {err}")

    uploaded_file = None
    if user_select == "Lokal (.fit-Datei)":
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader("Wähle eine lokale .fit-Datei aus", type=["fit"])

    if selected_activity_id:
        with st.spinner("Lade selektiertes FIT-File aus der Cloud..."):
            binary_content, download_err = download_original_fit_file(api_k, selected_activity_id)
            if binary_content:
                try:
                    fitfile = fitparse.FitFile(io.BytesIO(binary_content))
                    records = [record.get_values() for record in fitfile.get_messages('record')]
                    df = pd.DataFrame(records)
                    if not df.empty and 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        filename = selected_activity_name
                except Exception as ex: st.error(f"Fehler beim Parsen: {ex}")

    elif uploaded_file is not None:
        filename = uploaded_file.name
        fitfile = fitparse.FitFile(uploaded_file)
        records = [record.get_values() for record in fitfile.get_messages('record')]
        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

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

            col_edit1, col_edit2, col_edit3 = st.columns(3)
            with col_edit1: mode_type = st.selectbox("Erfassungs-Modus", ["Automatisch (Algorithmus)", "Manuell (Grafische Auswahl)"], key='erfassungs_modus')
            with col_edit2: expected_intervals = st.number_input("Erwartete Anzahl Intervalle", value=init_intervals, min_value=1, step=1)
            with col_edit3: expected_duration_min = st.number_input("Erwartete Intervalllänge (Min)", value=init_duration_min, min_value=1, step=1)
            
            col_edit4, col_edit5 = st.columns(2)
            with col_edit4: min_power_threshold = st.slider("Mindestleistung für Intervalle (Watt)", 50, 400, 185, step=5)
            with col_edit5: sg_window = st.slider("Savitzky-Golay Filterfenster (Glättung)", 11, 121, 45, step=2)
            
            median_window = 7
            workout_structure = f"{expected_intervals}x{expected_duration_min}"

            if 'power' in df.columns:
                if sg_window > len(df): sg_window = len(df) if len(df) % 2 != 0 else len(df) - 1
                df['power_clean'] = df['power'].fillna(0)
                df['power_median'] = df['power_clean'].rolling(window=median_window, min_periods=1, center=True).median()
                df['power_sg'] = savgol_filter(df['power_median'], window_length=sg_window, polyorder=2)
                df['power_deriv'] = savgol_filter(df['power_median'], window_length=sg_window, polyorder=2, deriv=1)
                
                if mode_type == "Automatisch (Algorithmus)":
                    target_length_sec = expected_duration_min * 60
                    df['rolling_window_power'] = df['power_sg'].rolling(window=target_length_sec, min_periods=int(target_length_sec*0.5), center=True).mean()
                    valid_centers = df['rolling_window_power'] >= min_power_threshold
                    
                    scores = df['rolling_window_power'].fillna(0).values
                    candidates = []
                    for i in range(len(df)):
                        if not valid_centers.iloc[i]: continue
                        start_w = max(0, i - target_length_sec // 2)
                        end_w = min(len(df), i + target_length_sec // 2)
                        if scores[i] == max(scores[start_w:end_w]) and scores[i] > 0:
                            if i not in candidates: candidates.append(i)
                    
                    candidates = sorted(candidates, key=lambda idx: scores[idx], reverse=True)[:expected_intervals]
                    candidates = sorted(candidates)
                    
                    is_interval_array = [False] * len(df)
                    block_id_array = [0] * len(df)
                    
                    for b_num, center_idx in enumerate(candidates, start=1):
                        left_bound = center_idx - (target_length_sec // 2)
                        right_bound = center_idx + (target_length_sec // 2)
                        
                        while left_bound > 0 and df['power_sg'].iloc[left_bound] >= (min_power_threshold - 10):
                            if is_interval_array[left_bound]: break
                            left_bound -= 1
                            
                        while right_bound < len(df) - 1 and df['power_sg'].iloc[right_bound] >= (min_power_threshold - 10):
                            right_bound += 1
                        
                        if left_bound >= right_bound or (right_bound - left_bound) < (target_length_sec * 0.5):
                            left_bound, right_bound = center_idx - (target_length_sec // 2), center_idx + (target_length_sec // 2)
                            
                        for j in range(left_bound, right_bound + 1):
                            if not is_interval_array[j]:
                                is_interval_array[j] = True
                                block_id_array[j] = b_num
                            
                    df['is_interval'] = is_interval_array
                    df['block_id'] = block_id_array
                    num_intervals = (df['is_interval'].astype(int).diff() == 1).sum()
                    if num_intervals == 0 and len(candidates) > 0: num_intervals = len(candidates)
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

                df['interval_highlight'] = df.apply(lambda row: row['power'] if row['is_interval'] else None, axis=1)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ø Leistung Gesamt", f"{int(df['power'].mean())} W")
                col2.metric("Max Leistung", f"{int(df['power'].max())} W")
                col3.metric("Gewählte Struktur", workout_structure)
                col4.metric("Intervalle erkannt", f"{num_intervals} / {expected_intervals}")
                
                intervals_calculated = []
                if num_intervals > 0:
                    unique_blocks = df[df['is_interval']]['block_id'].unique()
                    for idx, b_id in enumerate(unique_blocks, start=1):
                        block_df = df[df['block_id'] == b_id]
                        if block_df.empty: continue
                        
                        avg_p = int(block_df['power'].mean())
                        hr_data = block_df['heart_rate'].dropna()
                        avg_hr = int(hr_data.mean()) if not hr_data.empty else 0
                        max_hr = int(hr_data.max()) if not hr_data.empty else 0
                        std_hr = round(hr_data.std(), 1) if not hr_data.empty else 0
                        
                        if not hr_data.empty:
                            lower_p = hr_data.quantile(0.20)
                            upper_p = hr_data.quantile(0.80)
                            avg_hr_p = int(hr_data[(hr_data >= lower_p) & (hr_data <= upper_p)].mean())
                        else: avg_hr_p = 0
                            
                        intervals_calculated.append({
                            "Intervall": idx, "Ø Leistung": avg_p, "Ø Herzfrequenz": avg_hr, 
                            "Max Herzfrequenz": max_hr, "Abweichung HF+-": std_hr, 
                            "Durschnittliche HF_P (20-80)": avg_hr_p, "Dauer (Min)": round(len(block_df) / 60, 2),
                            "Dauer_sec": len(block_df) 
                        })
                
                if intervals_calculated:
                    st.subheader("Kennwerte pro Intervall")
                    df_res = pd.DataFrame(intervals_calculated)
                    display_df = df_res.drop(columns=['Dauer_sec'])
                    styled_df = display_df.style.format({
                        "Ø Leistung": "{:.1f}", "Ø Herzfrequenz": "{:.1f}", "Max Herzfrequenz": "{:.1f}",
                        "Abweichung HF+-": "{:.1f}", "Durschnittliche HF_P (20-80)": "{:.1f}", "Dauer (Min)": "{:.1f}"
                    }).set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                    st.dataframe(styled_df, use_container_width=False, hide_index=True)
                    
                    workout_date = df.index.min().strftime('%Y-%m-%d')
                    metadata = {"filename": filename, "date": workout_date, "type": detected_type, "structure": workout_structure, "avg_power": int(df['power'].mean()), "max_power": int(df['power'].max())}
                    
                    st.markdown("### Speichern")
                    if st.session_state['overwrite_warning']:
                        st.warning(f"Achtung: Workout vom {workout_date} existiert bereits.")
                        col_w1, col_w2 = st.columns([1, 3])
                        with col_w1:
                            if st.button("Trotzdem überschreiben", type="primary"):
                                success, msg = save_workout_to_db(metadata, intervals_calculated, overwrite_id=st.session_state['workout_to_overwrite'])
                                if success: st.success(msg); st.session_state['overwrite_warning'] = False
                        with col_w2:
                            if st.button("Abbrechen"): st.session_state['overwrite_warning'] = False; st.rerun()
                    else:
                        if st.button("In Datenbank übernehmen"):
                            dup_id = check_duplicate_workout(workout_date, detected_type, workout_structure)
                            if dup_id: st.session_state['overwrite_warning'] = True; st.session_state['workout_to_overwrite'] = dup_id; st.rerun()
                            else: success, message = save_workout_to_db(metadata, intervals_calculated); st.success(message)
                        
                        if mode_type == "Automatisch (Algorithmus)":
                            auto_blocks_timestamps = [(df[df['block_id'] == b].index.min(), df[df['block_id'] == b].index.max()) for b in unique_blocks]
                            st.button("Intervalle manuell nachjustieren", on_click=transfer_to_manual, args=(auto_blocks_timestamps,))

                st.subheader("Trainingsdaten & Analyse")
                fig_main = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.45, 0.3, 0.25])
                fig_main.add_trace(go.Scatter(x=df.index, y=df['power'], mode='lines', name='Rohleistung', line=dict(color='rgba(150, 150, 150, 0.4)', width=1)), row=1, col=1)
                fig_main.add_trace(go.Scatter(x=df.index, y=df['power_sg'], mode='lines', name='Sav-Gol Trend', line=dict(color='rgba(51, 153, 255, 0.6)', width=1)), row=1, col=1)
                fig_main.add_trace(go.Scatter(x=df.index, y=df['interval_highlight'], mode='lines', name='Intervall (Power)', line=dict(color='#FFA500', width=2.5)), row=1, col=1)
                
                if 'heart_rate' in df.columns and not df['heart_rate'].isna().all():
                    df['hr_clean'] = df['heart_rate'].ffill().bfill()
                    df['hr_highlight'] = df.apply(lambda row: row['hr_clean'] if row['is_interval'] else None, axis=1)
                    fig_main.add_trace(go.Scatter(x=df.index, y=df['hr_clean'], mode='lines', name='Herzfrequenz', line=dict(color='rgba(255, 102, 102, 0.5)', width=1.5)), row=2, col=1)
                    fig_main.add_trace(go.Scatter(x=df.index, y=df['hr_highlight'], mode='lines', name='Intervall (HF)', line=dict(color='#FF3333', width=2.5)), row=2, col=1)
                
                fig_main.add_trace(go.Scatter(x=df.index, y=df['power_deriv'], mode='lines', name='Steigung (Ableitung)', line=dict(color='#33CC33', width=1.2)), row=3, col=1)
                fig_main.update_layout(template="plotly_dark", height=800, hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0), legend=dict(yanchor="top", y=1))
                selected_data = st.plotly_chart(fig_main, use_container_width=True, on_select="rerun")
                
                if mode_type == "Manuell (Grafische Auswahl)":
                    st.markdown("### Manuelle Intervall-Bearbeitung")
                    if selected_data and selected_data.get("selection", {}).get("box"):
                        box = selected_data["selection"]["box"][0]
                        if "x" in box:
                            start_t, end_t = pd.to_datetime(box["x"][0]), pd.to_datetime(box["x"][1])
                            if st.button("Bereich als Intervall hinzufügen"):
                                st.session_state['manual_intervals'].append((start_t, end_t)); st.rerun()
                    
                    if st.session_state['manual_intervals']:
                        for idx, (s, e) in enumerate(st.session_state['manual_intervals']):
                            if st.button(f"Löschen Intervall {idx+1}", key=f"del_{idx}"):
                                st.session_state['manual_intervals'].pop(idx); st.rerun()
        except Exception as e: st.error(f"Fehler bei der Analyse: {e}")

elif nav_mode == "Historie & Vergleich":
    st.subheader("Datenbank-Historie & Vergleich")
    
    # --- FILTERN ---
    conn = sqlite3.connect(DB_NAME)
    df_workouts = pd.read_sql_query("SELECT * FROM workouts", conn)
    conn.close()
    
    if df_workouts.empty: 
        st.info("Datenbank leer.")
    else:
        # Filter-Optionen
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_type = st.selectbox("Typ-Filter", ["ALLE", "LIT", "MIT", "HIT"])
        with col_f2:
            search_query = st.text_input("Suche nach Dateiname/Datum:")

        # Filtern des DataFrames
        df_display = df_workouts.copy()
        if filter_type != "ALLE":
            df_display = df_display[df_display['type'] == filter_type]
        if search_query:
            df_display = df_display[df_display['filename'].str.contains(search_query, case=False, na=False) | 
                                    df_display['date'].str.contains(search_query, case=False, na=False)]

        # --- ANZEIGE & LÖSCHEN ---
        st.write("---")
        selected_ids = []
        for idx, row in df_display.iterrows():
            col_check, col_del = st.columns([0.85, 0.15])
            with col_check:
                if st.checkbox(f"{row['date']} | {row['type']} ({row['structure']}) | {row['filename']}", key=f"wb_{row['id']}"):
                    selected_ids.append(row['id'])
            with col_del:
                if st.button("🗑️", key=f"del_{row['id']}"):
                    conn = sqlite3.connect(DB_NAME)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM workouts WHERE id = ?", (row['id'],))
                    conn.commit()
                    conn.close()
                    st.rerun()

        # --- VERGLEICH ---
        if len(selected_ids) >= 2:
            st.markdown("---")
            conn = sqlite3.connect(DB_NAME)
            ids_string = ",".join(map(str, selected_ids))
            df_compare = pd.read_sql_query(f"SELECT i.*, w.date, w.type FROM intervals i JOIN workouts w ON i.workout_id = w.id WHERE i.workout_id IN ({ids_string})", conn)
            conn.close()
            df_compare['Workout'] = df_compare['date'].str.slice(0, 10) + " (" + df_compare['type'] + ")"
            
            c1, c2, c3 = st.columns(3)
            
            # 1. Ø Watt
            with c1: st.plotly_chart(px.scatter(df_compare, x="interval_num", y="avg_power", color="Workout", title="Ø Watt").update_traces(mode='lines+markers').update_layout(template="plotly_dark"))
            
            # 2. Ø HF mit Standardabweichung und 20-80 Perzentil
            with c2:
                fig_hr = go.Figure()
                for w in df_compare['Workout'].unique():
                    sub = df_compare[df_compare['Workout'] == w]
                    fig_hr.add_trace(go.Scatter(x=sub['interval_num'], y=sub['avg_hr'], name=f"{w} (Ø)", 
                                                error_y=dict(type='data', array=sub['std_hr'], visible=True),
                                                mode='lines+markers'))
                    fig_hr.add_trace(go.Scatter(x=sub['interval_num'], y=sub['avg_hr_p'], name=f"{w} (20-80%)",
                                                mode='markers', marker=dict(size=8, color=None, line=dict(color='orange', width=2))))
                fig_hr.update_layout(title="Ø Herzfrequenz (+- StdDev)", template="plotly_dark")
                st.plotly_chart(fig_hr)

            # 3. Max HF (Revert auf einfache px.scatter Darstellung)
            with c3: 
                st.plotly_chart(px.scatter(df_compare, x="interval_num", y="max_hr", color="Workout", title="Max HF").update_traces(mode='lines+markers').update_layout(template="plotly_dark"))
