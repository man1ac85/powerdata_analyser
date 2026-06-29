import streamlit as st
import fitparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks
import numpy as np
import re
from datetime import datetime, timedelta
import io
import hashlib
import requests
from requests.auth import HTTPBasicAuth
from sqlalchemy import text
import warnings
import time
from contextlib import contextmanager

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# --- DATENBANK & KONFIGURATION ---
st.set_page_config(page_title="Advanced Power Data Analyser", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 5rem !important;
}
div[data-testid="stSelectbox"] label, 
div[data-testid="stDateInput"] label, 
div[data-testid="stTextInput"] label, 
div[data-testid="stMultiSelect"] label {
    font-size: 0.85rem !important;
}
div[data-testid="stDataFrame"] {
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def get_supabase_client():
    from supabase import create_client
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    
def check_and_add_eff_loss_column():
    conn = get_db_connection()
    try:
        with conn.session as s:
            s.execute(text('ALTER TABLE intervals ADD COLUMN IF NOT EXISTS eff_loss FLOAT;'))
            s.commit()
    except Exception: pass
check_and_add_eff_loss_column()

@contextmanager
def perf_track(label):
    """Context manager for performance tracking. Stores timings in st.session_state['_perf']."""
    if not globals().get('PROFILING', False):
        yield
        return
    if "_perf" not in st.session_state:
        st.session_state["_perf"] = []
    t0 = time.perf_counter()
    try:
        yield
    finally:
        st.session_state["_perf"].append((label, time.perf_counter() - t0))

def render_analysis_ui(df, filename, active_user_id, selected_activity_id, default_min_power, ftp_val, is_admin, key_suffix="", is_bulk=False):
    try:
        st.markdown("---")
        current_id_key = f'current_id_{key_suffix}'
        is_new_workout = False
        
        explicit_type = None
        if "40/20" in filename:
            explicit_type = "HIT 40/20"
        else:
            match_type = re.search(r'(LIT|MIT|HIT|GA|RSH)', filename, re.IGNORECASE)
            if match_type:
                explicit_type = match_type.group(1).upper()
                
        unique_identifier = selected_activity_id if selected_activity_id else filename
        
        if st.session_state.get(current_id_key) != unique_identifier:
            st.session_state[current_id_key] = unique_identifier
            st.session_state[f'current_filename_{key_suffix}'] = filename
            st.session_state['erfassungs_modus'] = "Automatisch (Algorithmus)"
            is_new_workout = True
            
            # Reset UI-Regler bei neuem Workout
            if f"ui_int_{key_suffix}" in st.session_state:
                del st.session_state[f"ui_int_{key_suffix}"]
            if f"ui_dur_{key_suffix}" in st.session_state:
                del st.session_state[f"ui_dur_{key_suffix}"]

            if explicit_type:
                st.session_state[f"type_{key_suffix}"] = explicit_type

        match_4020 = re.search(r'(\d+)[xX](\d+)[xX]40/20', filename, re.IGNORECASE)
        match_structure = re.search(r'(\d+)\s*[xX]\s*(\d+)', filename)

        if match_4020:
            name_blocks = int(match_4020.group(1))
            name_micro = int(match_4020.group(2))
            name_intervals = name_blocks
            expected_intervals = name_blocks
            expected_duration_min = 1
        else:
            name_blocks = 0
            name_micro = 0
            name_intervals = int(match_structure.group(1)) if match_structure else None
            name_duration_min = int(match_structure.group(2)) if match_structure else None
            expected_intervals = name_intervals if name_intervals else 0
            expected_duration_min = name_duration_min if name_duration_min else 0

        target_intervals = st.session_state.get(f"ui_int_{key_suffix}")
        target_duration = st.session_state.get(f"ui_dur_{key_suffix}")
        if target_intervals is None: target_intervals = expected_intervals
        if target_duration is None: target_duration = expected_duration_min

        has_gps = False
        if 'position_lat' in df.columns and 'position_long' in df.columns:
            if not df['position_lat'].dropna().empty:
                has_gps = True

        if f"type_{key_suffix}" in st.session_state:
            detected_type = st.session_state[f"type_{key_suffix}"]
        else:
            if explicit_type:
                detected_type = explicit_type
            elif "rennradfahren" in filename.lower() or "radfahren" in filename.lower():
                detected_type = "Draußen"
            elif any(kw in filename.lower() for kw in ["draußen", "velotour", "rtf", "fahrt"]):
                detected_type = "Draußen"
            elif has_gps:
                detected_type = "Draußen"
            else:
                detected_type = "UNKNOWN"
                    
            if detected_type == "UNKNOWN":
                with perf_track(f"[render:{key_suffix}] fetch_workouts_from_db (Typ-Erkennung)"):
                    past_workouts = fetch_workouts_from_db(active_user_id)
                if not past_workouts.empty and 'int_count' in past_workouts.columns:
                    valid_past = past_workouts[past_workouts['type'].isin(["LIT", "MIT", "HIT", "HIT 40/20"])].dropna(subset=['int_count', 'int_length']).copy()
                    if not valid_past.empty:
                        exact = valid_past[(valid_past['int_count'] == expected_intervals) & (valid_past['int_length'] == expected_duration_min)]
                        if not exact.empty:
                            detected_type = exact['type'].mode()[0]
                        else:
                            valid_past['dist'] = abs(valid_past['int_count'] - expected_intervals) * 5 + abs(valid_past['int_length'] - expected_duration_min)
                            closest = valid_past.nsmallest(5, 'dist')
                            if not closest.empty:
                                detected_type = closest['type'].mode()[0]
                                
            st.session_state[f"type_{key_suffix}"] = detected_type

        is_ride_analysis = detected_type in ["GA", "RSH", "Draußen"]
        
        if is_ride_analysis:
            st.subheader("Advanced Ride Analyser")
        else:
            st.subheader("Advanced Intervall Analyzer")

        if not is_ride_analysis and not match_structure:
            st.warning("⚠️ Workout Nomenklatur fehlerhaft (Anzahl und Länge der Intervalle fehlen im Namen).")

        mode_type = st.session_state.get('erfassungs_modus', "Automatisch (Algorithmus)")

        if not is_ride_analysis:
            if is_admin:
                c1, c2, c3, _ = st.columns([0.5, 0.5, 0.5, 4.5])
                with c1: min_power = st.slider("Mindestleistung (Watt)", 50, 400, default_min_power, step=5, key=f"min_p_{key_suffix}")
                with c2: sg_win = st.slider("Filterfenster", 11, 121, 45, step=2, key=f"sg_{key_suffix}")
                with c3: edge_ignore_sec = st.slider("Rand-Ignorierung (Sek)", 0, 600, 120, step=10, key=f"edge_{key_suffix}")
            else:
                c1, _ = st.columns([0.5, 4.5])
                with c1: min_power = st.slider("Mindestleistung (Watt)", 50, 400, default_min_power, step=5, key=f"min_p_{key_suffix}")
                sg_win = 45
                edge_ignore_sec = 120
        else:
            c1, c2, _ = st.columns([0.5, 0.5, 4.0])
            with c1:
                pers_auto_chunks = st.session_state.get(f"auto_chunks_{key_suffix}", not has_gps)
                auto_chunks = st.checkbox("Fahrt in Abschnitte teilen", value=pers_auto_chunks, key=f"chunks_{key_suffix}")
                st.session_state[f"auto_chunks_{key_suffix}"] = auto_chunks
            with c2:
                if auto_chunks:
                    pers_equi = st.session_state.get(f"persistent_equi_{key_suffix}", False)
                    equidistant = st.checkbox("Äquidistant (Gleiche Zeitabschnitte)", value=pers_equi, key=f"equi_{key_suffix}")
                    st.session_state[f"persistent_equi_{key_suffix}"] = equidistant
                else:
                    equidistant = False
            expected_intervals = 1
            expected_duration_min = 60
            min_power = default_min_power
            sg_win = 45
            edge_ignore_sec = 120
        
        if 'power' in df.columns:
            with perf_track(f"[render:{key_suffix}] Signal-Filter (SG)"):
                df['p_clean'] = df['power'].fillna(0)
                df['p_sg'] = savgol_filter(df['p_clean'].rolling(7, center=True, min_periods=1).median(), window_length=sg_win, polyorder=2)

                p_deriv_raw = savgol_filter(df['p_clean'].rolling(7, center=True, min_periods=1).median(), window_length=sg_win, polyorder=2, deriv=1)
                deriv_win = 21 if len(df) >= 21 else (len(df) - 1 if len(df) % 2 == 0 else len(df))
                df['p_deriv'] = savgol_filter(p_deriv_raw, window_length=max(3, deriv_win), polyorder=2)

                df['power_roll_30'] = df['p_clean'].rolling(window=30, min_periods=1).mean()

            if mode_type == "Automatisch (Algorithmus)":
                if is_ride_analysis:
                    total_sec = len(df)
                    if auto_chunks:
                        if equidistant:
                            num_chunks = int((total_sec + 1800) // 3600)
                            if num_chunks < 1: num_chunks = 1
                            chunk_len = total_sec / num_chunks
                            if chunk_len < 1200 and num_chunks > 1:
                                num_chunks -= 1
                                chunk_len = total_sec / num_chunks
                            df['block_id'] = [int(i // chunk_len) + 1 for i in range(total_sec)]
                            df.loc[df['block_id'] > num_chunks, 'block_id'] = num_chunks
                        else:
                            num_full_chunks = total_sec // 3600
                            remainder = total_sec % 3600
                            if remainder > 0 and remainder < 1200 and num_full_chunks > 0:
                                chunk_len = total_sec / num_full_chunks
                                df['block_id'] = [int(i // chunk_len) + 1 for i in range(total_sec)]
                                df.loc[df['block_id'] > num_full_chunks, 'block_id'] = num_full_chunks
                            else:
                                chunk_len = 3600
                                df['block_id'] = [int(i // chunk_len) + 1 for i in range(total_sec)]
                            
                        df['is_interval'] = True
                        num_intervals = df['block_id'].nunique()
                        expected_intervals = num_intervals
                        df['highlight'] = df.apply(lambda row: row['power'] if row['block_id'] % 2 != 0 else None, axis=1)
                    else:
                        df['is_interval'] = False
                        df['block_id'] = 0
                        num_intervals = 0
                        expected_intervals = 0
                        df['highlight'] = None
                else:
                    deriv_data = df['p_deriv'].fillna(0).values.copy()
                    if edge_ignore_sec > 0 and len(deriv_data) > edge_ignore_sec * 2:
                        deriv_data[:edge_ignore_sec] = 0
                        deriv_data[-edge_ignore_sec:] = 0
                        
                    max_d = np.max(deriv_data)
                    min_d = np.min(deriv_data)
                    
                    best_intervals = []
                    best_thresh_pos = 0
                    best_thresh_neg = 0
                    best_score = -1
                    
                    if target_duration and target_duration > 0 and not match_4020:
                        # Hybrid-Methode: Erst grob über Rolling Window, dann Kanten per lokaler Ableitung finden
                        target_sec = int(target_duration * 60)
                        
                        # 1. Grobe Erkennung (Plateaus finden)
                        heavy_smooth = df['p_clean'].rolling(window=15, min_periods=1, center=True).mean()
                        roll_p = heavy_smooth.rolling(window=target_sec, min_periods=int(target_sec*0.5), center=True).mean()
                        
                        valid_centers = roll_p >= min_power
                        scores = roll_p.fillna(0).values
                        
                        candidates = []
                        for i in range(len(df)):
                            if not valid_centers.iloc[i]: continue
                            start_w = max(0, i - target_sec // 2)
                            end_w = min(len(df), i + target_sec // 2)
                            if scores[i] == np.max(scores[start_w:end_w]) and scores[i] > 0:
                                if not candidates or (i - candidates[-1]) > (target_sec * 0.8):
                                    candidates.append(i)
                                    
                        take_n = target_intervals if target_intervals and target_intervals > 0 else 1
                        candidates = sorted(candidates, key=lambda x: scores[x], reverse=True)[:take_n]
                        candidates.sort()
                        
                        # 2. Lokale Kantenerkennung mittels Ableitung (+/- Suchradius um die vermutete Kante)
                        search_radius = min(45, max(15, target_sec // 4)) 
                        
                        for center in candidates:
                            approx_left = max(0, center - (target_sec // 2))
                            approx_right = min(len(df) - 1, center + (target_sec // 2))
                            
                            # Linke Kante (pos. Ausschlag der Ableitung)
                            search_start_l = max(0, approx_left - search_radius)
                            search_start_r = min(len(df) - 1, approx_left + search_radius)
                            deriv_window_start = deriv_data[search_start_l:search_start_r]
                            
                            if len(deriv_window_start) > 0 and np.max(deriv_window_start) > max_d * 0.2:
                                left = search_start_l + np.argmax(deriv_window_start)
                            else:
                                left = approx_left
                                while left < center and heavy_smooth.iloc[left] < min_power - 20: left += 1
                                while left > 0 and heavy_smooth.iloc[left-1] >= min_power - 20: left -= 1
    
                            # Rechte Kante (neg. Ausschlag der Ableitung)
                            search_end_l = max(0, approx_right - search_radius)
                            search_end_r = min(len(df) - 1, approx_right + search_radius)
                            deriv_window_end = deriv_data[search_end_l:search_end_r]
                            
                            if len(deriv_window_end) > 0 and np.min(deriv_window_end) < min_d * 0.2:
                                right = search_end_l + np.argmin(deriv_window_end)
                            else:
                                right = approx_right
                                while right > center and heavy_smooth.iloc[right] < min_power - 20: right -= 1
                                while right < len(df) - 1 and heavy_smooth.iloc[right+1] >= min_power - 20: right += 1
                                
                            # Plausibilität: Wenn durch die Kanten das Intervall massiv beschnitten wird
                            if left >= right or (right - left) < (target_sec * 0.6):
                                left = max(0, center - target_sec // 2)
                                right = min(len(df) - 1, center + target_sec // 2)
                                
                            best_intervals.append((left, right))
                    else:
                        # Original-Logik (Globale Flankenerkennung) primär für 40/20 und "Unbekannt"
                        for factor in np.arange(0.35, 0.75, 0.05):
                            thresh_pos = max_d * factor
                            thresh_neg = min_d * factor
                            
                            is_above = deriv_data > thresh_pos
                            is_below = deriv_data < thresh_neg
                            
                            starts_in = np.where((~is_above[:-1]) & is_above[1:])[0]
                            starts_out = np.where(is_above[:-1] & (~is_above[1:]))[0]
                            ends_in = np.where((~is_below[:-1]) & is_below[1:])[0]
                            ends_out = np.where(is_below[:-1] & (~is_below[1:]))[0]
                            
                            start_points = []
                            for s_in in starts_in:
                                s_out_cands = starts_out[starts_out > s_in]
                                if len(s_out_cands) > 0: start_points.append((s_in + s_out_cands[0]) // 2)
                                    
                            end_points = []
                            for e_in in ends_in:
                                e_out_cands = ends_out[ends_out > e_in]
                                if len(e_out_cands) > 0: end_points.append((e_in + e_out_cands[0]) // 2)
                                    
                            current_intervals = []
                            last_end = -1
                            for sp in start_points:
                                if sp <= last_end: continue
                                possible_ends = [ep for ep in end_points if ep > sp]
                                if possible_ends:
                                    ep = possible_ends[0]
                                    next_starts = [nsp for nsp in start_points if nsp > sp]
                                    if not next_starts or ep < next_starts[0]:
                                        current_intervals.append((sp, ep))
                                        last_end = ep
                                        
                            if current_intervals:
                                durations = [ep - sp for sp, ep in current_intervals]
                                cv = np.std(durations) / np.mean(durations) if len(durations) > 1 else 0
                                score = len(current_intervals) * (1 - cv)
                                
                                if target_intervals and target_intervals > 0 and not match_4020:
                                    count_diff = abs(len(current_intervals) - target_intervals)
                                    if count_diff == 0:
                                        score += 1000
                                    else:
                                        score -= count_diff * 10
                                        
                                if target_duration and target_duration > 0 and not match_4020:
                                    target_sec = target_duration * 60
                                    avg_dur = np.mean(durations)
                                    if abs(avg_dur - target_sec) > (target_sec * 0.3):
                                        score -= 50
    
                                if score > best_score:
                                    best_score = score
                                    best_intervals = current_intervals
                                    best_thresh_pos = thresh_pos
                                    best_thresh_neg = thresh_neg
                    
                    st.session_state[f'best_thresh_pos_{key_suffix}'] = best_thresh_pos
                    st.session_state[f'best_thresh_neg_{key_suffix}'] = best_thresh_neg
                    
                    # Wenn Algorithmus zu viele Intervalle findet, die Schwächsten verwerfen
                    if best_intervals and target_intervals and target_intervals > 0 and not match_4020:
                        if len(best_intervals) > target_intervals:
                            scored_intervals = []
                            for sp, ep in best_intervals:
                                mean_p = df['p_clean'].iloc[sp:ep+1].mean()
                                scored_intervals.append((mean_p, sp, ep))
                            scored_intervals.sort(key=lambda x: x[0], reverse=True)
                            best_intervals = [(sp, ep) for _, sp, ep in scored_intervals[:target_intervals]]
                            best_intervals.sort(key=lambda x: x[0])
    
                    interval_nums = {}
                    is_micro = False
                    if best_intervals:
                        if detected_type == "HIT 40/20" or match_4020:
                            macro_blocks = []
                            current_macro = [best_intervals[0]]
                            for sp, ep in best_intervals[1:]:
                                if (sp - current_macro[-1][1]) <= 45:
                                    current_macro.append((sp, ep))
                                else:
                                    macro_blocks.append(current_macro)
                                    current_macro = [(sp, ep)]
                            macro_blocks.append(current_macro)
                            is_micro = len(macro_blocks) < len(best_intervals)
                        else:
                            macro_blocks = [[intv] for intv in best_intervals]
                            
                        b_id = 1
                        for m_idx, macro in enumerate(macro_blocks, start=1):
                            for u_idx, (sp, ep) in enumerate(macro, start=1):
                                interval_nums[b_id] = (m_idx * 100 + u_idx) if is_micro else b_id
                                b_id += 1
                                
                    is_int = [False] * len(df)
                    block = [0] * len(df)
                    b_id_counter = 1
                    if best_intervals:
                        for macro in macro_blocks:
                            for sp, ep in macro:
                                for j in range(sp, ep + 1):
                                    is_int[j] = True
                                    block[j] = b_id_counter
                                b_id_counter += 1
                    
                    df['is_interval'] = is_int
                    df['block_id'] = block
                    
                    if is_micro:
                        expected_intervals = len(macro_blocks)
                        macro_durs = [(macro[-1][1] - macro[0][0]) for macro in macro_blocks]
                        st.session_state[f'auto_dur_sec_{key_suffix}'] = int(sum(macro_durs) / len(macro_durs))
                        expected_duration_min = max(1, int((st.session_state[f'auto_dur_sec_{key_suffix}'] + 30) // 60))
                    else:
                        expected_intervals = len(best_intervals)
                        st.session_state[f'auto_dur_sec_{key_suffix}'] = int(round(sum(ep - sp for sp, ep in best_intervals) / len(best_intervals))) if best_intervals else 0
                        expected_duration_min = max(1, int((st.session_state[f'auto_dur_sec_{key_suffix}'] + 30) // 60))
                        
                    num_intervals = expected_intervals
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
            
                expected_intervals = num_intervals
                if num_intervals > 0:
                    avg_dur_sec = sum((e_t - s_t).total_seconds() for s_t, e_t in st.session_state['manual_intervals']) / num_intervals
                    expected_duration_min = max(1, int((avg_dur_sec + 30) // 60))
            
            # --- SETUP LAYOUT FOR UI ---
            type_options = ["LIT", "MIT", "HIT", "HIT 40/20", "GA", "RSH", "Draußen", "UNKNOWN"]
            idx = type_options.index(detected_type) if detected_type in type_options else 0
            
            col_t, col_i, col_btn_save, col_btn_adj = st.columns([0.45, 0.8, 1.25, 7.5])
            with col_t:
                detected_type = st.selectbox("Workout Typ", type_options, index=idx, key=f"type_{key_suffix}")

            if not is_ride_analysis:
                if detected_type == "HIT 40/20":
                    b_count = len(macro_blocks) if (mode_type == "Automatisch (Algorithmus)" and locals().get('is_micro', False)) else expected_intervals
                    m_count = (len(best_intervals) // b_count) if (mode_type == "Automatisch (Algorithmus)" and locals().get('is_micro', False) and b_count > 0) else (name_micro if match_4020 else 0)
                    workout_structure = f"{b_count}x{m_count}x40/20" if b_count > 0 else "Manuell"
                else:
                    workout_structure = f"{expected_intervals}x{expected_duration_min}" if expected_intervals > 0 else "Manuell"
            else:
                workout_structure = "Ride"
                
            # --- CALCULATE METRICS ---
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

            # --- BERECHNUNG DER KENNWERTE ---
            intervals_calculated = []
            unique_blocks = []
            if num_intervals > 0:
                unique_blocks = df[df['is_interval']]['block_id'].unique()
                
            stats = {}
            if active_user_id:
                try:
                    conn = get_db_connection()
                    user_data = conn.query("SELECT api_key, intervals_id FROM users WHERE id = :uid", params={"uid": active_user_id}, ttl=3600)
                    if not user_data.empty:
                        tmp_api = user_data.iloc[0]['api_key']
                        tmp_int = user_data.iloc[0].get('intervals_id', '0')
                        with perf_track(f"[render:{key_suffix}] get_athlete_stats_from_intervals"):
                            stats = get_athlete_stats_from_intervals(tmp_api, tmp_int)
                except Exception:
                    pass

                has_max_hr = str(stats.get('Max HR', '')).isdigit() and int(stats['Max HR']) > 0
                max_hr_val = int(stats['Max HR']) if has_max_hr else None
                hr_threshold = max_hr_val * 0.88 if max_hr_val else None
                
                macro_block_b_ids = {}
                if locals().get('is_micro', False):
                    for b_id in unique_blocks:
                        int_num = interval_nums.get(b_id, b_id)
                        if int_num > 100:
                            m_idx = int_num // 100
                            if m_idx not in macro_block_b_ids: macro_block_b_ids[m_idx] = []
                            macro_block_b_ids[m_idx].append(b_id)

                for idx_blk, b_id in enumerate(unique_blocks, start=1):
                    block_df = df[df['block_id'] == b_id]
                    if block_df.empty: continue
                    
                    avg_p = int(block_df['power'].mean())
                    np_val = (block_df['power_roll_30'] ** 4).mean() ** 0.25 if not block_df.empty else 0
                    if pd.isna(np_val): np_val = 0

                    if 'heart_rate' in block_df.columns:
                        hr_data = block_df['heart_rate'].dropna()
                    else:
                        hr_data = pd.Series(dtype=float)
                        
                    avg_hr = int(hr_data.mean()) if not hr_data.empty else 0
                    max_hr = int(hr_data.max()) if not hr_data.empty else 0
                    std_hr = float(round(hr_data.std(), 1)) if not hr_data.empty else 0
                    
                    efficiency = np_val / avg_hr if avg_hr > 0 else 0
                    
                    if not hr_data.empty:
                        lower_p = hr_data.quantile(0.20)
                        upper_p = hr_data.quantile(0.80)
                        avg_hr_p = int(hr_data[(hr_data >= lower_p) & (hr_data <= upper_p)].mean())
                    else: 
                        avg_hr_p = 0
                        
                    time_above_88 = 0
                    if hr_threshold and not hr_data.empty:
                        time_above_88 = len(hr_data[hr_data >= hr_threshold])
                        
                    eff_loss = 0.0
                    if not hr_data.empty and not block_df['power'].empty:
                        roll_p = block_df['power'].rolling('20s').mean()
                        roll_hr = hr_data.rolling('20s').mean()
                        roll_eff = (roll_p / roll_hr).replace([np.inf, -np.inf], np.nan).fillna(0)
                        valid = roll_eff > 0
                        if valid.sum() > 10: 
                            t_sec = (block_df.index - block_df.index[0]).total_seconds().values
                            slope, _ = np.polyfit(t_sec[valid], roll_eff[valid], 1)
                            eff_loss = slope * 60
                        
                    int_num_display = int(interval_nums.get(idx_blk, idx_blk)) if not is_ride_analysis and mode_type == "Automatisch (Algorithmus)" else int(idx_blk)
                        
                    intervals_calculated.append({
                        "Intervall": int_num_display, 
                        "Ø Watt": avg_p, 
                        "Ø HF": avg_hr, 
                        "Efficiency": float(round(efficiency, 2)),
                        "EffLoss": float(round(eff_loss, 4)),
                        "NP": float(round(np_val, 1)),
                        "Max HF": max_hr, 
                        "Δ HF+-": std_hr, 
                        "Ø HF_P (20-80)": avg_hr_p, 
                        "Dauer (mm:ss)": f"{int(len(block_df) // 60):02d}:{int(len(block_df) % 60):02d}",
                        "Dauer_sec": len(block_df),
                        ">= 88% HF (s)": time_above_88
                    })
                    
                    if locals().get('is_micro', False) and int_num_display > 100:
                        m_idx = int_num_display // 100
                        if b_id == macro_block_b_ids[m_idx][-1]:
                            first_b_id = macro_block_b_ids[m_idx][0]
                            last_b_id = macro_block_b_ids[m_idx][-1]
                            
                            start_time = df[df['block_id'] == first_b_id].index.min()
                            end_time = df[df['block_id'] == last_b_id].index.max()
                            
                            macro_df = df[(df.index >= start_time) & (df.index <= end_time)]
                            
                            mac_avg_p = int(macro_df['power'].mean()) if not macro_df['power'].empty else 0
                            mac_np_val = (macro_df['power_roll_30'] ** 4).mean() ** 0.25 if 'power_roll_30' in macro_df.columns and not macro_df.empty else 0
                            if pd.isna(mac_np_val): mac_np_val = 0
                            
                            mac_hr_data = macro_df['heart_rate'].dropna() if 'heart_rate' in macro_df.columns else pd.Series(dtype=float)
                            mac_avg_hr = int(mac_hr_data.mean()) if not mac_hr_data.empty else 0
                            mac_max_hr = int(mac_hr_data.max()) if not mac_hr_data.empty else 0
                            mac_std_hr = float(round(mac_hr_data.std(), 1)) if not mac_hr_data.empty else 0
                            
                            mac_efficiency = mac_np_val / mac_avg_hr if mac_avg_hr > 0 else 0
                            
                            mac_avg_hr_p = 0
                            if not mac_hr_data.empty:
                                mac_lower_p = mac_hr_data.quantile(0.20)
                                mac_upper_p = mac_hr_data.quantile(0.80)
                                mac_avg_hr_p = int(mac_hr_data[(mac_hr_data >= mac_lower_p) & (mac_hr_data <= mac_upper_p)].mean())
                            
                            mac_time_above_88 = 0
                            if hr_threshold and not mac_hr_data.empty:
                                mac_time_above_88 = len(mac_hr_data[mac_hr_data >= hr_threshold])
                                
                            mac_eff_loss = 0.0
                            if not mac_hr_data.empty and not macro_df['power'].empty:
                                m_roll_p = macro_df['power'].rolling('20s').mean()
                                m_roll_hr = mac_hr_data.rolling('20s').mean()
                                m_roll_eff = (m_roll_p / m_roll_hr).replace([np.inf, -np.inf], np.nan).fillna(0)
                                m_valid = m_roll_eff > 0
                                if m_valid.sum() > 10:
                                    m_t_sec = (macro_df.index - macro_df.index[0]).total_seconds().values
                                    m_slope, _ = np.polyfit(m_t_sec[m_valid], m_roll_eff[m_valid], 1)
                                    mac_eff_loss = m_slope * 60
                                
                            intervals_calculated.append({
                                "Intervall": m_idx * 100,
                                "Ø Watt": mac_avg_p,
                                "Ø HF": mac_avg_hr,
                                "Efficiency": float(round(mac_efficiency, 2)),
                                "EffLoss": float(round(mac_eff_loss, 4)),
                                "NP": float(round(mac_np_val, 1)),
                                "Max HF": mac_max_hr,
                                "Δ HF+-": mac_std_hr,
                                "Ø HF_P (20-80)": mac_avg_hr_p,
                                "Dauer (mm:ss)": f"{int(len(macro_df) // 60):02d}:{int(len(macro_df) % 60):02d}",
                                "Dauer_sec": len(macro_df),
                                ">= 88% HF (s)": mac_time_above_88
                            })
            
            real_ints = [i for i in intervals_calculated if not (i["Intervall"] >= 100 and i["Intervall"] % 100 == 0)]
            if not real_ints: real_ints = intervals_calculated
            
            n_ints = len(real_ints)
            int_avg_power = int(round(sum(i["Ø Watt"] for i in real_ints) / n_ints)) if n_ints > 0 else None
            int_avg_hr = int(round(sum(i["Ø HF"] for i in real_ints) / n_ints)) if n_ints > 0 else None
            int_avg_eff = float(round(sum(i["Efficiency"] for i in real_ints) / n_ints, 2)) if n_ints > 0 else None
            
            workout_date = df.index.min().strftime('%Y-%m-%d')
            
            if not is_ride_analysis and detected_type != "HIT 40/20" and n_ints > 0:
                durations_min = [max(1, int(round(i["Dauer_sec"] / 60.0))) for i in real_ints]
                if len(set(durations_min)) > 1:
                    workout_structure = f"{n_ints}x{'-'.join(map(str, durations_min))}"
                else:
                    workout_structure = f"{n_ints}x{durations_min[0]}"
            
            if not is_ride_analysis:
                final_filename = f"{detected_type} {workout_structure}"
            else:
                final_filename = f"{detected_type} Ride"
                ignore_names = ["fahrt", "ride", "morning ride", "afternoon ride", "lunch ride", "evening ride", "radfahren", "rennradfahren", "unbenannte fahrt", "unbekannt"]
                if filename and not any(ign in filename.lower() for ign in ignore_names):
                    final_filename += f" - {filename}"
            
            metadata = {
                "filename": final_filename, 
                "date": workout_date, 
                "type": detected_type, 
                "structure": workout_structure, 
                "avg_power": overall_avg_p, 
                "max_power": overall_max_p,
                "intervals_activity_id": selected_activity_id,
                "user_id": active_user_id,
                "int_avg_power": int_avg_power,
                "int_avg_hr": int_avg_hr,
                "int_avg_eff": int_avg_eff,
                "int_count": n_ints if not is_ride_analysis and n_ints > 0 else None,
                "int_length": int(round(sum(durations_min)/len(durations_min))) if not is_ride_analysis and n_ints > 0 and 'durations_min' in locals() else (expected_duration_min if not is_ride_analysis and n_ints > 0 else None)
            }

            # --- RENDER INTERVAL UI & BUTTONS ---
            with col_i:
                if not is_ride_analysis:
                    if detected_type == "HIT 40/20":
                        if not match_4020:
                            color = "#FFD700"
                            status_text = "Keine 40/20 Vorgaben im Namen"
                        elif b_count == name_blocks and m_count == name_micro:
                            color = "#33CC33"
                            status_text = "Passt zum Namen"
                        else:
                            color = "#FF3333"
                            status_text = f"Abweichung (Name: {name_blocks}x{name_micro})"
                            
                        st.markdown(f'<div style="font-size: 14px; color: rgb(163, 168, 184); margin-bottom: 2px;">Blöcke erkannt ({status_text})</div><div style="font-size: 1.65rem; font-weight: bold; line-height: 1.2; color: {color};">{workout_structure}</div>', unsafe_allow_html=True)
                    else:
                        if not match_structure:
                            color = "#FFD700"
                            status_text = "Keine Vorgaben im Namen"
                        elif expected_intervals == name_intervals and expected_duration_min == name_duration_min:
                            color = "#33CC33"
                            status_text = "Passt zum Namen"
                        else:
                            color = "#FF3333"
                            status_text = f"Abweichung (Name: {name_intervals}x{name_duration_min}m)"
                            
                        avg_dur_sec = st.session_state.get(f'auto_dur_sec_{key_suffix}', 0)
                        exact_dur_display = f"{int(avg_dur_sec // 60):02d}:{int(avg_dur_sec % 60):02d} mm:ss"
                        
                        if f"ui_int_{key_suffix}" not in st.session_state:
                            st.session_state[f"ui_int_{key_suffix}"] = int(expected_intervals if expected_intervals > 0 else 1)
                        if f"ui_dur_{key_suffix}" not in st.session_state:
                            st.session_state[f"ui_dur_{key_suffix}"] = int(expected_duration_min if expected_duration_min > 0 else 1)
                            
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            ui_intervals = st.number_input("Anzahl", min_value=1, step=1, key=f"ui_int_{key_suffix}")
                        with cc2:
                            ui_duration = st.number_input("Dauer (Min)", min_value=1, step=1, key=f"ui_dur_{key_suffix}")
                            
                        st.markdown(f'<div style="font-size: 13px; color: {color}; margin-top: 4px;">Status: {status_text} | Ø {exact_dur_display}</div>', unsafe_allow_html=True)
                        
                        expected_intervals = ui_intervals
                        expected_duration_min = ui_duration
                        workout_structure = f"{expected_intervals}x{expected_duration_min}"
                else:
                    st.markdown(f'<div style="font-size: 14px; color: rgb(163, 168, 184); margin-bottom: 2px;">Modus</div><div style="font-size: 1.65rem; font-weight: bold; line-height: 1.2; color: #3399FF;">Ride Analysis</div>', unsafe_allow_html=True)
            
            with col_btn_save:
                if metadata:
                    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
                    if st.session_state.get('overwrite_warning'):
                        st.warning("Bereits in DB. Überschreiben?")
                        if st.button("Ja", type="primary", use_container_width=True, key=f"yes_ow_{key_suffix}"):
                            success, msg = save_workout_to_db(metadata, intervals_calculated, overwrite_id=st.session_state['workout_to_overwrite'])
                            if success: 
                                st.success(msg)
                                st.session_state['overwrite_warning'] = False
                                if is_bulk:
                                    st.session_state['bulk_index'] += 1
                                    st.rerun()
                            else: st.error(msg)
                        if st.button("Abbrechen", use_container_width=True, key=f"no_ow_{key_suffix}"): 
                            st.session_state['overwrite_warning'] = False
                            st.rerun()
                    elif st.session_state.get('interval_mismatch_warning'):
                        st.warning(f"Achtung Abweichung. Speichern?")
                        if st.button("Ja", type="primary", key=f"yes_mis_{key_suffix}", use_container_width=True):
                            st.session_state['interval_mismatch_warning'] = False
                            dup_id = check_duplicate_workout(workout_date, detected_type, metadata['structure'], active_user_id, selected_activity_id)
                            if dup_id: 
                                st.session_state['overwrite_warning'] = True
                                st.session_state['workout_to_overwrite'] = dup_id
                                st.rerun()
                            else: 
                                success, message = save_workout_to_db(metadata, intervals_calculated)
                                if success: 
                                    st.success(message)
                                    if is_bulk:
                                        st.session_state['bulk_index'] += 1
                                        st.rerun()
                                else: st.error(message)
                        if st.button("Abbrechen", key=f"no_mis_{key_suffix}", use_container_width=True): 
                            st.session_state['interval_mismatch_warning'] = False
                            st.rerun()
                    else:
                        btn_label = "💾 In DB übernehmen (Bulk)" if is_bulk else "💾 In Datenbank übernehmen"
                        if st.button(btn_label, use_container_width=True, key=f"save_{key_suffix}"):
                            mismatch = False
                            if not is_ride_analysis:
                                if detected_type == "HIT 40/20" and match_4020:
                                    mismatch = (b_count != name_blocks or m_count != name_micro)
                                elif not match_4020 and match_structure:
                                    mismatch = (expected_intervals != name_intervals)
                                    
                            if mismatch:
                                st.session_state['interval_mismatch_warning'] = True
                                st.rerun()
                            else:
                                dup_id = check_duplicate_workout(workout_date, detected_type, metadata['structure'], active_user_id, selected_activity_id)
                                if dup_id: 
                                    st.session_state['overwrite_warning'] = True
                                    st.session_state['workout_to_overwrite'] = dup_id
                                    st.rerun()
                                else: 
                                    success, message = save_workout_to_db(metadata, intervals_calculated)
                                    if success: 
                                        st.success(message)
                                        if is_bulk:
                                            st.session_state['bulk_index'] += 1
                                            st.rerun()
                                    else: st.error(message)
                        
                        if is_bulk:
                            st.markdown("<div style='margin-top: 1.0rem;'></div>", unsafe_allow_html=True)
                            if st.button("⏭️ Überspringen & Nächstes", use_container_width=True, key=f"skip_{key_suffix}"):
                                st.session_state['bulk_index'] += 1
                                st.rerun()
                            if st.button("⏹️ Bulk Einlesen Abbrechen", use_container_width=True, key=f"cancel_bulk_{key_suffix}"):
                                st.session_state['bulk_active'] = False
                                clear_bulk_targets()
                                st.rerun()

            with col_btn_adj:
                if True:
                    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
                    if mode_type == "Automatisch (Algorithmus)" and is_admin:
                        auto_blocks_timestamps = [(df[df['block_id'] == b].index.min(), df[df['block_id'] == b].index.max()) for b in df[df['is_interval']]['block_id'].unique() if b > 0]
                        if is_ride_analysis:
                            st.button("⚙️ Manuelle Intervalle markieren", on_click=transfer_to_manual, args=(auto_blocks_timestamps,), use_container_width=True, key=f"adj_{key_suffix}")
                        else:
                            st.button("⚙️ Intervalle nachjustieren", on_click=transfer_to_manual, args=(auto_blocks_timestamps,), use_container_width=True, key=f"adj_{key_suffix}")
                    elif mode_type == "Manuell (Grafische Auswahl)":
                        st.button("🔄 Zurück zur Automatik", on_click=transfer_to_auto, use_container_width=True, key=f"adj_auto_{key_suffix}")

            # --- RENDER TABLES AND PLOTS ---
            st.markdown(f"""
            <style>
            .small-metric {{ font-size: 1.65rem !important; font-weight: bold; line-height: 1.2; }}
            .small-metric-label {{ font-size: 14px; color: rgb(163, 168, 184); margin-bottom: 2px; }}
            </style>
            <div style="font-size: 14px; color: rgb(163, 168, 184); margin-bottom: 5px;">Workoutdaten</div>
            <table style="border-collapse: collapse; border: none; text-align: left; width: auto; background: transparent;">
                <tr style="border: none; background: transparent;">
                    <td style="padding: 0 30px 0 0; border: none;"><div class="small-metric-label">Ø Leistung</div><div class="small-metric">{overall_avg_p} W</div></td>
                    <td style="padding: 0 30px 0 0; border: none;"><div class="small-metric-label">Normalized Power</div><div class="small-metric">{overall_np} W</div></td>
                    <td style="padding: 0 30px 0 0; border: none;"><div class="small-metric-label">Max Leistung</div><div class="small-metric">{overall_max_p} W</div></td>
                    <td style="padding: 0 30px 0 0; border: none;"><div class="small-metric-label">Ø HF</div><div class="small-metric">{overall_avg_hr} bpm</div></td>
                    <td style="padding: 0 30px 0 0; border: none;"><div class="small-metric-label">Max HF</div><div class="small-metric">{overall_max_hr} bpm</div></td>
                    <td style="padding: 0 30px 0 0; border: none;"><div class="small-metric-label">TSS Score</div><div class="small-metric">{overall_tss}</div></td>
                </tr>
            </table>
            <br>
            """, unsafe_allow_html=True)
            
            if intervals_calculated:
                col_tab1, col_tab2 = st.columns([2, 1])
                with col_tab1:
                    df_res = pd.DataFrame(intervals_calculated)
                    display_df = df_res.drop(columns=['Dauer_sec'])
                    
                    display_df['Intervall'] = display_df['Intervall'].apply(
                        lambda x: f"Block {x//100} Average" if (isinstance(x, (int, float)) and x >= 100 and x % 100 == 0) else (f"Block {x//100} - {x%100:02d}" if (isinstance(x, (int, float)) and x > 100) else str(x))
                    )
                    
                    display_df = display_df.rename(columns={"Intervall": "Kennwerte pro Intervall"})
                    
                    if '>= 88% HF (s)' in display_df.columns:
                        display_df['>= 88% HF'] = display_df['>= 88% HF (s)'].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}")
                        display_df = display_df.drop(columns=['>= 88% HF (s)'])
                    
                    avg_dur_sec_total = sum(i["Dauer_sec"] for i in real_ints) / len(real_ints)
                    avg_88_total = sum(i.get(">= 88% HF (s)", 0) for i in real_ints) / len(real_ints)
                    
                    avg_row = {
                        "Kennwerte pro Intervall": "Averages",
                        "Ø Watt": metadata["int_avg_power"],
                        "Ø HF": metadata["int_avg_hr"],
                        "Efficiency": metadata["int_avg_eff"],
                        "EffLoss": float(round(sum(i["EffLoss"] for i in real_ints) / len(real_ints), 4)) if real_ints else 0.0,
                        "NP": float(round(sum(i["NP"] for i in real_ints) / len(real_ints), 1)),
                        "Max HF": int(round(sum(i["Max HF"] for i in real_ints) / len(real_ints))),
                        "Δ HF+-": float(round(sum(i["Δ HF+-"] for i in real_ints) / len(real_ints), 1)),
                        "Ø HF_P (20-80)": int(round(sum(i["Ø HF_P (20-80)"] for i in real_ints) / len(real_ints))),
                        "Dauer (mm:ss)": f"{int(avg_dur_sec_total // 60):02d}:{int(avg_dur_sec_total % 60):02d}",
                        ">= 88% HF": f"{int(avg_88_total // 60):02d}:{int(avg_88_total % 60):02d}"
                    }
                    display_df = pd.concat([display_df, pd.DataFrame([avg_row])], ignore_index=True)
                    
                    styled_df = display_df.style.format({
                        "Ø Watt": "{:.0f}", "Ø HF": "{:.0f}", "Efficiency": "{:.2f}", "EffLoss": "{:.4f}", "NP": "{:.1f}",
                        "Max HF": "{:.0f}", "Δ HF+-": "{:.1f}", "Ø HF_P (20-80)": "{:.0f}"
                    }).set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                    st.dataframe(
                        styled_df, 
                        use_container_width=False, 
                        hide_index=True,
                        column_config={
                            "Kennwerte pro Intervall": st.column_config.Column("Kennwerte pro Intervall", width=120),
                            "Ø Watt": st.column_config.Column("Ø Watt", width=40),
                            "Ø HF": st.column_config.Column("Ø HF", width=40),
                            "Efficiency": st.column_config.Column("Efficiency", width=40),
                            "EffLoss": st.column_config.Column("EffLoss", width=40),
                            "NP": st.column_config.Column("NP", width=40),
                            "Max HF": st.column_config.Column("Max HF", width=40),
                            "Δ HF+-": st.column_config.Column("Δ HF+-", width=40),
                            "Ø HF_P (20-80)": st.column_config.Column("Ø HF_P (20-80)", width=60),
                            "Dauer (mm:ss)": st.column_config.Column("Dauer (mm:ss)", width=60),
                            ">= 88% HF": st.column_config.Column(">= 88% HF", width=60)
                        }
                    )
                
            st.subheader("Trainingsdaten & Analyse")
            
            has_hr = 'heart_rate' in df.columns and not df['heart_rate'].isna().all()
            alt_col = 'enhanced_altitude' if 'enhanced_altitude' in df.columns else 'altitude' if 'altitude' in df.columns else None
            has_alt = bool(alt_col and not df[alt_col].isna().all())
            
            num_rows = 1
            if has_hr: num_rows += 1
            if has_alt: num_rows += 1
            if is_admin: num_rows += 1
            
            fig_main = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04)
            current_row = 1
            
            fig_main.add_trace(go.Scatter(x=df.index, y=df.get('power'), mode='lines', name='Rohleistung', line=dict(color='rgba(150, 150, 150, 0.4)', width=1)), row=current_row, col=1)
            fig_main.add_trace(go.Scatter(x=df.index, y=df.get('p_sg'), mode='lines', name='Sav-Gol Trend', line=dict(color='rgba(51, 153, 255, 0.6)', width=1)), row=current_row, col=1)
            fig_main.add_trace(go.Scatter(x=df.index, y=df.get('highlight'), mode='lines', name='Intervall (Power)', line=dict(color='#FFA500', width=2.5)), row=current_row, col=1)
            
            if has_hr:
                current_row += 1
                df['hr_clean'] = df['heart_rate'].ffill().bfill()
                if is_ride_analysis and mode_type == "Automatisch (Algorithmus)":
                    df['hr_highlight'] = df.apply(lambda row: row['hr_clean'] if row['block_id'] % 2 != 0 else None, axis=1)
                else:
                    df['hr_highlight'] = df.apply(lambda row: row['hr_clean'] if row['is_interval'] else None, axis=1)
                fig_main.add_trace(go.Scatter(x=df.index, y=df['hr_clean'], mode='lines', name='Herzfrequenz', line=dict(color='rgba(255, 102, 102, 0.5)', width=1.5)), row=current_row, col=1)
                fig_main.add_trace(go.Scatter(x=df.index, y=df['hr_highlight'], mode='lines', name='Intervall (HF)', line=dict(color='#FF3333', width=2.5)), row=current_row, col=1)
                
            if has_alt:
                current_row += 1
                df['alt_clean'] = df[alt_col].ffill().bfill()
                if is_ride_analysis and mode_type == "Automatisch (Algorithmus)":
                    df['alt_highlight'] = df.apply(lambda row: row['alt_clean'] if row['block_id'] % 2 != 0 else None, axis=1)
                else:
                    df['alt_highlight'] = df.apply(lambda row: row['alt_clean'] if row['is_interval'] else None, axis=1)
                fig_main.add_trace(go.Scatter(x=df.index, y=df['alt_clean'], mode='lines', name='Höhe', line=dict(color='rgba(0, 204, 255, 0.4)', width=1.5)), row=current_row, col=1)
                fig_main.add_trace(go.Scatter(x=df.index, y=df['alt_highlight'], mode='lines', name='Intervall (Höhe)', line=dict(color='#00CCFF', width=2.5)), row=current_row, col=1)
                
            if is_admin:
                current_row += 1
                fig_main.add_trace(go.Scatter(x=df.index, y=df.get('p_deriv'), mode='lines', name='Steigung (Ableitung)', line=dict(color='#33CC33', width=1.2)), row=current_row, col=1)

                if not is_ride_analysis:
                    b_pos = st.session_state.get(f'best_thresh_pos_{key_suffix}', 0)
                    b_neg = st.session_state.get(f'best_thresh_neg_{key_suffix}', 0)
                    if b_pos != 0:
                        fig_main.add_hline(y=b_pos, line_dash="dot", line_color="rgba(255, 255, 255, 0.5)", row=current_row, col=1, annotation_text=f"Pos ({b_pos:.1f})", annotation_position="top left")
                        fig_main.add_hline(y=b_neg, line_dash="dot", line_color="rgba(255, 255, 255, 0.5)", row=current_row, col=1, annotation_text=f"Neg ({b_neg:.1f})", annotation_position="bottom left")

            plot_height = max(550, 300 + (num_rows - 1) * 200)
            drag_behavior = "select" if mode_type == "Manuell (Grafische Auswahl)" else "zoom"
            fig_main.update_layout(template="plotly_dark", height=plot_height, hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0), legend=dict(yanchor="top", y=1), dragmode=drag_behavior)
            
            selected_data = st.plotly_chart(fig_main, width='stretch', on_select="rerun", key=f"chart_{key_suffix}")
            
            has_gps = False
            if 'position_lat' in df.columns and 'position_long' in df.columns:
                lat_clean = df['position_lat'].dropna()
                if not lat_clean.empty:
                    # Semicircles (Garmin/FIT Standard) zu regulären Koordinaten konvertieren falls nötig
                    if lat_clean.abs().max() > 90:
                        df['lat'] = df['position_lat'] * (180.0 / (2**31))
                        df['lon'] = df['position_long'] * (180.0 / (2**31))
                    else:
                        df['lat'] = df['position_lat']
                        df['lon'] = df['position_long']
                    has_gps = not df['lat'].isna().all()
                    
            if has_gps:
                with st.expander("🗺️ GPS Route & Karte (Intervalle markiert)", expanded=False):
                    df_map = df.dropna(subset=['lat', 'lon']).copy()
                    
                    fig_map = go.Figure()
                    
                    # 1. Normale Strecke (als blaue Basislinie)
                    fig_map.add_trace(go.Scattermapbox(
                        lat=df_map['lat'], lon=df_map['lon'], mode='lines',
                        line=dict(width=3, color='rgba(51, 153, 255, 0.7)'),
                        name='Route',
                        hoverinfo='text',
                        hovertext=df_map.index.strftime('%H:%M:%S') + '<br>Leistung: ' + df_map['power'].fillna(0).astype(int).astype(str) + ' W'
                    ))
                    
                    # 2. Intervalle (als dicke orange Markierungen darüber)
                    df_intervals = df_map[df_map['is_interval']]
                    if not df_intervals.empty:
                        fig_map.add_trace(go.Scattermapbox(
                            lat=df_intervals['lat'], lon=df_intervals['lon'], mode='markers',
                            marker=dict(size=6, color='#FFA500'),
                            name='Intervalle',
                            hoverinfo='text',
                            hovertext=df_intervals.index.strftime('%H:%M:%S') + '<br>Leistung: ' + df_intervals['power'].fillna(0).astype(int).astype(str) + ' W'
                        ))
                        
                    fig_map.update_layout(mapbox_style="open-street-map", mapbox=dict(center=dict(lat=df_map['lat'].mean(), lon=df_map['lon'].mean()), zoom=10), margin=dict(l=0, r=0, t=0, b=0), height=450, showlegend=False)
                    st.plotly_chart(fig_map, width='stretch', key=f"map_{key_suffix}")
                if has_hr:
                    with st.expander("📈 Herzfrequenz-Kurve (Time in Zone)", expanded=False):
                        hr_data_full = df['heart_rate'].dropna()
                        if not hr_data_full.empty:
                            hr_sorted = np.sort(hr_data_full)[::-1]
                            time_min = np.arange(1, len(hr_sorted) + 1) / 60.0
                            
                            fig_hr_curve = go.Figure()
                            fig_hr_curve.add_trace(go.Scatter(x=time_min, y=hr_sorted, mode='lines', name='HF Kurve', line=dict(color='#FF3333', width=2)))
                            fig_hr_curve.update_layout(
                                template="plotly_dark",
                                xaxis_title="Kumulierte Zeit (logarithmisch)",
                                yaxis_title="Herzfrequenz (bpm)",
                                margin=dict(l=0, r=0, t=10, b=0),
                                height=450,
                                hovermode="x unified"
                            )
                        fig_hr_curve.update_xaxes(type="log", tickvals=[0.1, 1, 5, 10, 30, 60, 120, 240], ticktext=["6s", "1m", "5m", "10m", "30m", "1h", "2h", "4h"])
                        st.plotly_chart(fig_hr_curve, width='stretch', key=f"hr_curve_{key_suffix}")
            
            if mode_type == "Manuell (Grafische Auswahl)":
                st.markdown("### Manuelle Intervall-Bearbeitung")
                st.info("💡 **Tipp:** Ziehe mit der Maus direkt im Graphen ein Rechteck über den gewünschten Zeitbereich, um ein neues Intervall zu markieren.")
                start_t, end_t = None, None
                if selected_data:
                    box = selected_data.get("selection", {}).get("box", None) or selected_data.get("box", None)
                    if box and isinstance(box, list): box = box[0]
                    if isinstance(box, dict) and "x" in box: 
                        t1, t2 = box["x"][0], box["x"][1]
                        start_t, end_t = min(t1, t2), max(t1, t2)
                
                if start_t and end_t:
                    st.write(f"Auswahl: {start_t} bis {end_t}")
                    if st.button("Bereich als Intervall hinzufügen", key=f"add_{key_suffix}"):
                        new_s = pd.to_datetime(start_t)
                        new_e = pd.to_datetime(end_t)
                        
                        updated_intervals = []
                        for s, e in st.session_state['manual_intervals']:
                            if e < new_s or s > new_e:
                                updated_intervals.append((s, e))
                            elif s < new_s and e > new_e:
                                updated_intervals.append((s, new_s - pd.Timedelta(seconds=1)))
                                updated_intervals.append((new_e + pd.Timedelta(seconds=1), e))
                            elif s < new_s and e >= new_s and e <= new_e:
                                updated_intervals.append((s, new_s - pd.Timedelta(seconds=1)))
                            elif s >= new_s and s <= new_e and e > new_e:
                                updated_intervals.append((new_e + pd.Timedelta(seconds=1), e))
                                
                        updated_intervals.append((new_s, new_e))
                        updated_intervals = [(s, e) for s, e in updated_intervals if s <= e]
                        
                        st.session_state['manual_intervals'] = updated_intervals
                        st.rerun()
                        
                if st.session_state['manual_intervals']:
                    intervals_to_keep = []
                    for idx, (s, e) in enumerate(st.session_state['manual_intervals'], start=1):
                        col_text, col_btn = st.columns([0.8, 0.2])
                        col_text.write(f"Intervall {idx}: {s.strftime('%H:%M:%S')} bis {e.strftime('%H:%M:%S')}")
                        if col_btn.button("Löschen", key=f"del_manual_{idx}_{key_suffix}"): pass 
                        else: intervals_to_keep.append((s, e))
                    if len(intervals_to_keep) != len(st.session_state['manual_intervals']):
                        st.session_state['manual_intervals'] = intervals_to_keep
                        st.rerun()
        else:
            st.warning("⚠️ Keine Leistungsdaten in diesem Datensatz gefunden. Eine Auswertung ist nicht möglich.")
            if is_bulk:
                if st.button("⏭️ Workout Überspringen", key=f"skip_nopow_{key_suffix}"):
                    st.session_state['bulk_index'] += 1
                    st.rerun()
                if st.button("⏹️ Abbrechen", key=f"cancel_nopow_{key_suffix}"):
                    st.session_state['bulk_active'] = False
                    clear_bulk_targets()
                    st.rerun()   
    except Exception as e: 
        st.error(f"Fehler in der Analyse: {e}")
        if is_bulk:
            if st.button("⏭️ Fehlerhaftes Workout überspringen", key=f"err_skip_{key_suffix}"):
                st.session_state['bulk_index'] += 1
                st.rerun()
            if st.button("⏹️ Bulk Einlesen Abbrechen", key=f"err_cancel_{key_suffix}"):
                st.session_state['bulk_active'] = False
                clear_bulk_targets()
                st.rerun()

# --- LOGIN FUNKTION ---
def check_login(username, password):
    conn = get_db_connection()
    try:
        result = conn.query("SELECT id, password_hash, role, approved FROM users WHERE name = :name", params={"name": username}, ttl=0)
    except Exception:
        result = conn.query("SELECT id, password_hash, role FROM users WHERE name = :name", params={"name": username}, ttl=0)
        
    if not result.empty:
        user = result.iloc[0]
        stored_hash = user['password_hash']
        role = str(user['role']).lower().strip() if user['role'] else 'user'
        user_id = int(user['id'])
        is_approved = True if pd.isna(user.get('approved')) else bool(user.get('approved'))
        
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        if input_hash == stored_hash:
            if not is_approved and role != 'admin':
                return "not_approved", role, user_id
            return "success", role, user_id
    return "invalid", None, None

# --- SESSION STATES ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'role' not in st.session_state: st.session_state['role'] = None
if 'user' not in st.session_state: st.session_state['user'] = None
if 'user_id' not in st.session_state: st.session_state['user_id'] = None

# --- LOKALER DEV-MODUS (AUTO-LOGIN) ---
AUTO_LOGIN = True  # <--- Auf False setzen, bevor du den Code produktiv stellst!
AUTO_LOGIN_USERNAME = "Bastian"  # <--- Trage hier deinen Datenbank-Benutzernamen ein
PROFILING = True   # <--- Auf False setzen für Produktion; zeigt Performance-Monitor in der Sidebar

if PROFILING:
    st.session_state["_perf"] = []  # Reset bei jedem Script-Run

if AUTO_LOGIN and not st.session_state['logged_in']:
    conn = get_db_connection()
    res_user = conn.query("SELECT id, role FROM users WHERE name = :name", params={"name": AUTO_LOGIN_USERNAME}, ttl=0)
    if not res_user.empty:
        st.session_state['logged_in'] = True
        st.session_state['user'] = AUTO_LOGIN_USERNAME
        st.session_state['role'] = str(res_user.iloc[0]['role']).lower().strip() if res_user.iloc[0]['role'] else 'user'
        st.session_state['user_id'] = int(res_user.iloc[0]['id'])
        st.rerun()
    else:
        st.warning(f"Auto-Login fehlgeschlagen: Benutzer '{AUTO_LOGIN_USERNAME}' existiert nicht.")

# --- SESSION STATES (Rest) ---
if 'manual_intervals' not in st.session_state: st.session_state['manual_intervals'] = []
if 'erfassungs_modus' not in st.session_state: st.session_state['erfassungs_modus'] = "Automatisch (Algorithmus)"
if 'overwrite_warning' not in st.session_state: st.session_state['overwrite_warning'] = False
if 'interval_mismatch_warning' not in st.session_state: st.session_state['interval_mismatch_warning'] = False
if 'workout_to_overwrite' not in st.session_state: st.session_state['workout_to_overwrite'] = None
if 'df' not in st.session_state: st.session_state['df'] = pd.DataFrame()    

# --- DATENBANK HELFER (PostgreSQL) ---
def update_user_password(user_id, new_password):
    pwd_hash = hashlib.sha256(new_password.encode()).hexdigest()
    conn = get_db_connection()
    with conn.session as s:
        s.execute(text("UPDATE users SET password_hash = :pwd WHERE id = :uid"), {"pwd": pwd_hash, "uid": user_id})
        s.commit()
    st.cache_data.clear()
    return True

def add_new_athlete(name, api_key, password, intervals_id, approved=True):
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_db_connection()
    
    with conn.session as s:
        try:
            s.execute(text("""
                INSERT INTO users (name, api_key, password_hash, role, intervals_id, approved) 
                VALUES (:name, :api_key, :pwd, 'user', :iid, :approved)
            """), {"name": name.strip(), "api_key": api_key.strip(), "pwd": pwd_hash, "iid": intervals_id.strip(), "approved": approved})
        except Exception:
            s.rollback()
            # Fallback, falls die SQL-Anweisung in Supabase noch nicht durchgeführt wurde
            s.execute(text("""
                INSERT INTO users (name, api_key, password_hash, role, intervals_id) 
                VALUES (:name, :api_key, :pwd, 'user', :iid)
            """), {"name": name.strip(), "api_key": api_key.strip(), "pwd": pwd_hash, "iid": intervals_id.strip()})
        s.commit()
    st.cache_data.clear()
    return True, f"Athlet '{name}' angelegt!"

def get_authorized_athletes(current_user_name, role, user_id):
    conn = get_db_connection()
    try:
        if role == 'admin':
            return conn.query("SELECT * FROM users", ttl=0)
        elif role == 'trainer':
            try:
                # Versuche die trainer_id Spalte abzufragen
                return conn.query("SELECT * FROM users WHERE trainer_id = :tid OR name = :name", 
                                  params={"tid": user_id, "name": current_user_name}, ttl=0)
            except Exception:
                # Fallback, falls die Spalte 'trainer_id' gar nicht in der Tabelle existiert
                return conn.query("SELECT * FROM users WHERE name = :name", 
                                  params={"name": current_user_name}, ttl=0)
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
            return int(result.iloc[0]['id'])
            
    # 2. Fallback für rein lokale FIT-Dateien (ohne Cloud-ID)
    if user_id:
        result = conn.query("SELECT id FROM workouts WHERE date = :date AND type = :type AND structure = :structure AND user_id = :uid",
                            params={"date": date, "type": workout_type, "structure": structure, "uid": user_id}, ttl=0)
    else:
        result = conn.query("SELECT id FROM workouts WHERE date = :date AND type = :type AND structure = :structure AND user_id IS NULL",
                            params={"date": date, "type": workout_type, "structure": structure}, ttl=0)
    return int(result.iloc[0]['id']) if not result.empty else None

def save_workout_to_db(metadata, interval_list, overwrite_id=None):
    conn = get_db_connection()
    with conn.session as s:
        if overwrite_id:
            s.execute(text("DELETE FROM workouts WHERE id = :id"), {"id": int(overwrite_id)})
        
        # Workout einfügen
        res = s.execute(text("""
            INSERT INTO workouts (filename, date, type, structure, avg_power, max_power, intervals_activity_id, user_id, int_avg_power, int_avg_hr, int_avg_eff, int_count, int_length) 
            VALUES (:filename, :date, :type, :structure, :avg_p, :max_p, :act_id, :uid, :int_avg_p, :int_avg_hr, :int_avg_eff, :int_count, :int_length) RETURNING id
        """), {
            "filename": metadata['filename'], "date": metadata['date'], "type": metadata['type'], 
            "structure": metadata['structure'], "avg_p": metadata['avg_power'], "max_p": metadata['max_power'],
            "act_id": metadata.get('intervals_activity_id'), "uid": metadata.get('user_id'),
            "int_avg_p": metadata.get('int_avg_power'), "int_avg_hr": metadata.get('int_avg_hr'), "int_avg_eff": metadata.get('int_avg_eff'),
            "int_count": metadata.get('int_count'), "int_length": metadata.get('int_length')
        })
        workout_id = res.scalar()
        for row in interval_list:
            # Konvertiere jeden Wert sicher in einen Python-Typ (float oder int)
            s.execute(text("""
                                INSERT INTO intervals (workout_id, interval_num, avg_power, avg_hr, max_hr, duration_sec, std_hr, avg_hr_p, "NP_int", intervall_eff, time_above_88_hr, eff_loss) 
                                VALUES (:wid, :num, :ap, :ahr, :mhr, :dur, :std, :ahrp, :np, :eff, :t88, :eff_loss)
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
                                "eff": float(row['Efficiency']),
                "t88": int(row.get('>= 88% HF (s)', 0)),
                "eff_loss": float(row.get('EffLoss', 0.0))
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
    return conn.query(f"SELECT i.*, w.date, w.type, w.filename, w.intervals_activity_id FROM intervals i JOIN workouts w ON i.workout_id = w.id WHERE i.workout_id IN ({ids_string})", ttl=0)

def transfer_to_manual(timestamps):
    st.session_state['manual_intervals'] = timestamps
    st.session_state['erfassungs_modus'] = "Manuell (Grafische Auswahl)"
    st.session_state['overwrite_warning'] = False
    st.session_state['interval_mismatch_warning'] = False

def clear_bulk_targets():
    if 'bulk_temp_targets' in st.session_state:
        del st.session_state['bulk_temp_targets']

def transfer_to_auto():
    st.session_state['erfassungs_modus'] = "Automatisch (Algorithmus)"
    st.session_state['overwrite_warning'] = False
    st.session_state['interval_mismatch_warning'] = False

# --- LOGIN-TOR ---
if not st.session_state['logged_in']:
    st.title("🔒 Powerdata Dashboard")
    
    tab_login, tab_reg, tab_forgot = st.tabs(["🔑 Anmelden", "📝 Registrieren", "❓ Passwort vergessen"])
    
    with tab_login:
        user_in = st.text_input("Benutzername", key="log_user")
        pass_in = st.text_input("Passwort", type="password", key="log_pass")
        if st.button("Anmelden", use_container_width=True):
            status, role, user_id = check_login(user_in, pass_in)
            if status == "success":
                st.session_state['logged_in'] = True
                st.session_state['user'] = user_in
                st.session_state['role'] = role
                st.session_state['user_id'] = user_id
                st.rerun()
            elif status == "not_approved":
                st.error("Dein Account wurde noch nicht freigegeben. Bitte warte auf die Aktivierung durch einen Administrator!")
            else: st.error("Benutzername oder Passwort falsch!")
            
    with tab_reg:
        st.markdown("**Neu hier? Lege dir ein Profil an.**")
        reg_name = st.text_input("Benutzername", key="reg_name")
        reg_api = st.text_input("Intervals.icu API Key", key="reg_api", type="password")
        reg_id = st.text_input("Intervals.icu ID (z.B. 75948)", key="reg_id")
        reg_pass = st.text_input("Passwort", type="password", key="reg_pass")
        reg_pass2 = st.text_input("Passwort bestätigen", type="password", key="reg_pass2")
        if st.button("Registrieren", use_container_width=True):
            if reg_name and reg_api and reg_id and reg_pass:
                if reg_pass == reg_pass2:
                    conn = get_db_connection()
                    res = conn.query("SELECT id FROM users WHERE name = :name", params={"name": reg_name}, ttl=0)
                    if not res.empty:
                        st.error("Dieser Benutzername existiert leider bereits!")
                    else:
                        add_new_athlete(reg_name, reg_api, reg_pass, reg_id, approved=False)
                        st.success("Erfolgreich registriert! Dein Profil muss jedoch noch von einem Administrator freigegeben werden, bevor du dich einloggen kannst.")
                else: st.error("Die Passwörter stimmen nicht überein!")
            else: st.warning("Bitte fülle alle Felder aus.")
            
    with tab_forgot:
        st.markdown("### Passwort vergessen?")
        st.info("Da im Profil aktuell keine E-Mail-Adressen hinterlegt werden, ist ein automatischer Reset per E-Mail aus Sicherheitsgründen nicht möglich.")
        st.markdown("Bitte wende dich direkt an deinen **Trainer oder Administrator**, um dein Passwort zurücksetzen zu lassen.")
        
    st.stop()

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

@st.cache_data(ttl=3600, show_spinner=False)
def get_activity_df(act_id, api_key):
    """Lädt DataFrame bevorzugt aus Parquet. Falls nicht, via FIT-Download oder als Fallback über die Streams API."""
    _t0 = time.perf_counter()
    clean_act_id = str(act_id).strip()
    if clean_act_id.endswith(".0"): clean_act_id = clean_act_id[:-2]
    if clean_act_id in ["", "None", "nan"]: return pd.DataFrame()

    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        try:
            _t_sb = time.perf_counter()
            sb = get_supabase_client()
            _t_sb_conn = time.perf_counter()
            res_bytes = sb.storage.from_("workouts").download(f"workout_{clean_act_id}.parquet")
            _t_sb_dl = time.perf_counter()
            result = pd.read_parquet(io.BytesIO(res_bytes))
            print(f"[PERF] get_activity_df → Supabase-Client: {(_t_sb_conn-_t_sb)*1000:.0f}ms | Download: {(_t_sb_dl-_t_sb_conn)*1000:.0f}ms | read_parquet: {(time.perf_counter()-_t_sb_dl)*1000:.0f}ms | GESAMT: {(time.perf_counter()-_t0)*1000:.0f}ms [HIT]")
            return result
        except Exception as e:
            print(f"[PERF] get_activity_df → Supabase MISS ({e.__class__.__name__}: {e}) nach {(time.perf_counter()-_t0)*1000:.0f}ms")
            pass

    df_fit = pd.DataFrame()
    _t_fit = time.perf_counter()
    bin_data, _ = download_original_fit_file(api_key, clean_act_id)
    print(f"[PERF] get_activity_df → download_original_fit_file: {(time.perf_counter()-_t_fit)*1000:.0f}ms")
    if bin_data:
        try:
            _t_parse = time.perf_counter()
            _fit_fields = {'timestamp', 'power', 'heart_rate', 'cadence', 'speed', 'distance', 'altitude', 'position_lat', 'position_long'}
            fitfile = fitparse.FitFile(io.BytesIO(bin_data))
            records = [{k: v for k, v in r.get_values().items() if k in _fit_fields} for r in fitfile.get_messages('record')]
            df_fit = pd.DataFrame(records)
            print(f"[PERF] get_activity_df → FIT-Parse ({len(records)} records): {(time.perf_counter()-_t_parse)*1000:.0f}ms")
            if not df_fit.empty and 'timestamp' in df_fit.columns:
                df_fit['timestamp'] = pd.to_datetime(df_fit['timestamp'])
                df_fit.set_index('timestamp', inplace=True)
        except Exception:
            pass

    # Fallback auf Streams API
    if df_fit.empty:
        url = f"https://intervals.icu/api/v1/activity/{clean_act_id}/streams"
        try:
            params = {"types": "time,watts,heartrate,latlng,altitude"}
            _t_streams = time.perf_counter()
            response = requests.get(url, params=params, auth=HTTPBasicAuth('API_KEY', api_key), timeout=15)
            if response.status_code == 200:
                streams = response.json()
                stream_dict = {}
                for s in streams:
                    stype = s.get("type")
                    if stype in ["time", "watts", "heartrate", "latlng", "altitude"]:
                        if stype == "latlng":
                            coords = s.get("data", [])
                            stream_dict["position_lat"] = [c[0] if len(c)>0 else None for c in coords]
                            stream_dict["position_long"] = [c[1] if len(c)>1 else None for c in coords]
                        else:
                            stream_dict[stype] = s.get("data")
                if "time" in stream_dict:
                    df_fit = pd.DataFrame(stream_dict)
                    if "watts" in df_fit.columns: df_fit.rename(columns={"watts": "power"}, inplace=True)
                    if "heartrate" in df_fit.columns: df_fit.rename(columns={"heartrate": "heart_rate"}, inplace=True)
                    df_fit['timestamp'] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df_fit['time'], unit='s')
                    df_fit.set_index('timestamp', inplace=True)
                    print(f"[PERF] get_activity_df → Streams-API: {(time.perf_counter()-_t_streams)*1000:.0f}ms")
        except Exception:
            pass

    if not df_fit.empty and "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        try:
            for col in df_fit.columns:
                if df_fit[col].dtype == 'object': df_fit[col] = df_fit[col].astype(str)
            parquet_buffer = io.BytesIO()
            df_fit.to_parquet(parquet_buffer, engine='pyarrow', index=True)
            parquet_buffer.seek(0)
            sb = get_supabase_client()
            _t_up = time.perf_counter()
            sb.storage.from_("workouts").upload(f"workout_{clean_act_id}.parquet", parquet_buffer.read(), {"content-type": "application/octet-stream"})
            print(f"[PERF] get_activity_df → Supabase-Upload: {(time.perf_counter()-_t_up)*1000:.0f}ms")
        except Exception: pass

    print(f"[PERF] get_activity_df → GESAMT (MISS-Pfad): {(time.perf_counter()-_t0)*1000:.0f}ms")
    return df_fit

@st.cache_data(ttl=60, show_spinner=False)
def fetch_upcoming_events(api_key, athlete_id):
    if pd.isna(athlete_id) or str(athlete_id).strip() in ["", "None", "nan"]:
        clean_id = "0"
    else:
        clean_id = str(athlete_id).strip()
        
    if clean_id != "0" and not clean_id.lower().startswith("i"):
        clean_id = f"i{clean_id}"
    url = f"https://intervals.icu/api/v1/athlete/{clean_id}/events"
    
    start_str = datetime.now().strftime('%Y-%m-%d')
    end_str = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
    params = {"oldest": start_str, "newest": end_str}
    
    try:
        response = requests.get(url, params=params, auth=HTTPBasicAuth('API_KEY', api_key), timeout=10)
        if response.status_code == 200:
            events = response.json()
            # Filtere nach "RACE" Kategorien (A, B, C) und Typ "Ride"
            races = [e for e in events if str(e.get('category', '')).upper() in ['RACE', 'RACE_A', 'RACE_B', 'RACE_C'] and e.get('type') == 'Ride']
            races.sort(key=lambda x: x.get('start_date_local', '9999'))
            return races
    except Exception: pass
    return []

def upload_workout_to_intervals(api_key, athlete_id, date, name, description, workout_text):
    if pd.isna(athlete_id) or str(athlete_id).strip() in ["", "None", "nan"]:
        clean_id = "0"
    else:
        clean_id = str(athlete_id).strip()
        
    if clean_id != "0" and not clean_id.lower().startswith("i"):
        clean_id = f"i{clean_id}"
        
    url = f"https://intervals.icu/api/v1/athlete/{clean_id}/events"
    
    # Intervals wertet die Description als Grundlage für die strukturierten Intervalle aus
    full_description = f"{description}\n\n{workout_text}" if description else workout_text
    
    payload = {
        "start_date_local": f"{date.strftime('%Y-%m-%d')}T00:00:00",
        "type": "Ride",
        "category": "WORKOUT",
        "name": name,
        "description": full_description
    }
    
    try:
        response = requests.post(url, json=payload, auth=HTTPBasicAuth('API_KEY', api_key), timeout=10)
        if response.status_code == 200:
            return True, "Workout erfolgreich in Intervals.icu hochgeladen!"
        else:
            return False, f"Fehler {response.status_code}: {response.text}"
    except Exception as e:
        return False, f"Verbindungsfehler: {e}"

# --- SURFACE LAYOUT ---
col_title, col_logout = st.columns([8, 1])
with col_title:
    st.markdown("<h2 style='margin-top: 0rem; font-size: 2.2rem;'>Powerdata Dashboard</h2>", unsafe_allow_html=True)
with col_logout:
    if st.session_state.get('logged_in'):
        if st.button("🚪 Ausloggen", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['user'] = None
            st.session_state['role'] = None
            st.session_state['user_id'] = None
            st.rerun()

nav_options = ["Training einlesen", "Daten & Auswertung", "Fatigue Resistance", "Trendanalyse", "🧠 Smart Workout Builder", "⚙️ Einstellungen"]
if st.session_state.get('role') == 'admin':
    nav_options.append("👤 Athleten verwalten")
    nav_options.append("Bulk Data Analyser")

tabs = st.tabs(nav_options)

with tabs[4]:
    st.subheader("🧠 Smart Workout Builder")
    st.markdown("Definiere deine Rahmenbedingungen. Der Builder findet alle mathematisch möglichen Intervall-Kombinationen, die deinen Vorgaben entsprechen.")
    
    col_wb_left, col_wb_right = st.columns([1, 1])
    
    with col_wb_left:
        auth_users = get_authorized_athletes(st.session_state['user'], st.session_state['role'], st.session_state.get('user_id'))
        wb_athlete = None
        default_ftp = 250
        
        if not auth_users.empty:
            opts_wb = auth_users['name'].tolist()
            def_idx_wb = opts_wb.index(st.session_state['user']) if st.session_state['user'] in opts_wb else 0
            c_ath1, c_ath2 = st.columns(2)
            with c_ath1:
                wb_athlete = st.selectbox("Ziel-Account (Upload & FTP)", opts_wb, index=def_idx_wb, key="smart_ath_sel")
            ath_row = auth_users[auth_users['name'] == wb_athlete].iloc[0]
            
            wb_uid = int(ath_row['id'])
            if 'ftp_cache' not in st.session_state: 
                st.session_state['ftp_cache'] = {}
            if wb_uid not in st.session_state['ftp_cache']:
                stats = get_athlete_stats_from_intervals(ath_row['api_key'], ath_row.get('intervals_id', '0'))
                st.session_state['ftp_cache'][wb_uid] = stats.get('FTP', 0)
                
            ftp_val = st.session_state['ftp_cache'].get(wb_uid, 0)
            if ftp_val and str(ftp_val) != "-" and float(ftp_val) > 0:
                default_ftp = int(float(ftp_val))
                
            with c_ath2:
                wb_ftp = st.number_input("Referenz-FTP (Watt, via Intervals)", value=default_ftp, disabled=True, key="smart_ftp")
        else:
            wb_ftp = st.number_input("Referenz-FTP (Watt)", value=default_ftp, key="smart_ftp")
            
        builder_type = st.selectbox("Vorauswahl Intervalltyp", ["Stair", "RSH", "HIT", "LIT", "MIT"])
        
        sel = None
        if builder_type != "Stair":
            st.info(f"🚧 Der Intervalltyp '{builder_type}' befindet sich noch in der Entwicklung (TBD).")
        else:
            with st.form("smart_builder_form"):
                st.markdown("##### 1. Block-Zeiten & Intervalle")
                c1, c2, c3, c4 = st.columns(4)
                t_ges_target = c1.number_input("Ziel Gesamtzeit (Min) ±3", 1, 300, 50)
                n_min = c2.number_input("Min Anzahl Intervalle", 1, 50, 5)
                n_max = c2.number_input("Max Anzahl", 1, 50, 9)
                
                time_opts = [f"{m:02d}:{s:02d}" for m in range(61) for s in (0, 30)][:-1]
                t_p_min_str = c3.selectbox("Min Pause (mm:ss)", time_opts, index=time_opts.index("01:00"))
                t_p_max_str = c3.selectbox("Max Pause (mm:ss)", time_opts, index=time_opts.index("01:00"))
                t_i_min_str = c4.selectbox("Min Intervall (mm:ss)", time_opts, index=time_opts.index("00:30"))
                t_i_max_str = c4.selectbox("Max Intervall (mm:ss)", time_opts, index=time_opts.index("20:00"))
                
                def parse_mmss(val):
                    m, s = val.split(':')
                    return int(m) * 60 + int(s)
                    
                t_p_min = parse_mmss(t_p_min_str)
                t_p_max = parse_mmss(t_p_max_str)
                t_i_min = parse_mmss(t_i_min_str)
                t_i_max = parse_mmss(t_i_max_str)
                
                st.markdown("##### 2. Leistungsvorgaben (Watt)")
                c5, c6, c7, c8 = st.columns(4)
                w_i1_min = c5.number_input("Min Start-Watt (1. Int)", 50, 1500, 250)
                w_i1_max = c5.number_input("Max Start-Watt", 50, 1500, 250)
                g_min = c6.number_input("Min Steigerung pro Int. (G)", -50, 50, 5)
                g_max = c6.number_input("Max Steigerung pro Int.", -50, 50, 5)
                w_p_min = c7.number_input("Min Pausen-Watt", 30, 500, 140)
                w_p_max = c7.number_input("Max Pausen-Watt", 30, 500, 160)
                w_avg_min = c8.number_input("Min Ø-Watt (Gesamtblock)", 50, 500, 218)
                w_avg_max = c8.number_input("Max Ø-Watt", 50, 500, 222)
                
                st.markdown("##### 3. Extras")
                c9, c10 = st.columns(2)
                wu_dur = c9.number_input("Warmup davor (Min)", 0, 60, 10)
                cd_dur = c10.number_input("Cooldown danach (Min)", 0, 60, 10)
                
                submit_search = st.form_submit_button("🔍 Lösungsraum berechnen", type="primary")
                
            if submit_search:
                import time
                start_time = time.time()
                sols = []
                max_reached = False
                
                # Sicherstellen, dass Min <= Max
                _n_min, _n_max = min(n_min, n_max), max(n_min, n_max)
                _t_ges_min, _t_ges_max = max(1, t_ges_target - 3), t_ges_target + 3
                _t_p_min, _t_p_max = min(t_p_min, t_p_max), max(t_p_min, t_p_max)
                _t_i_min, _t_i_max = min(t_i_min, t_i_max), max(t_i_min, t_i_max)
                _w_i1_min, _w_i1_max = min(w_i1_min, w_i1_max), max(w_i1_min, w_i1_max)
                _g_min, _g_max = min(g_min, g_max), max(g_min, g_max)
                _w_p_min, _w_p_max = min(w_p_min, w_p_max), max(w_p_min, w_p_max)
                _w_avg_min, _w_avg_max = min(w_avg_min, w_avg_max), max(w_avg_min, w_avg_max)
                
                for n in range(int(_n_min), int(_n_max) + 1):
                    if max_reached or time.time() - start_time > 3.0: break
                    for t_ges in range(int(_t_ges_min), int(_t_ges_max) + 1):
                        if max_reached: break
                        t_ges_sec = t_ges * 60
                        for t_p in range(int(_t_p_min), int(_t_p_max) + 1):
                            if t_p % 5 != 0: continue
                            t_i = (t_ges_sec - (n - 1) * t_p) / n if n > 1 else t_ges_sec
                            
                            if t_i < _t_i_min or t_i > _t_i_max: continue
                            t_i_rnd = round(t_i, 1)
                            if t_i_rnd % 5 != 0: continue
                                
                            step_w1 = 5 if _w_i1_max > _w_i1_min else 1
                            for w_i1 in range(int(_w_i1_min), int(_w_i1_max) + 1, step_w1):
                                if max_reached: break
                                step_g = 1 if _g_max > _g_min else 1
                                for G in range(int(_g_min), int(_g_max) + 1, step_g):
                                    step_wp = 5 if _w_p_max > _w_p_min else 1
                                    for w_p in range(int(_w_p_min), int(_w_p_max) + 1, step_wp):
                                        if n > 1 and t_p > 0:
                                            W_int = n * w_i1 * t_i + G * t_i * n * (n - 1) / 2
                                            W_pause = (n - 1) * w_p * t_p
                                            w_avg_calc = (W_int + W_pause) / t_ges_sec
                                        elif n == 1:
                                            w_avg_calc = w_i1
                                            
                                        if _w_avg_min <= w_avg_calc <= _w_avg_max:
                                            score = 0
                                            if t_i_rnd % 30 == 0: score += 4
                                            elif t_i_rnd % 15 == 0: score += 2
                                            elif t_i_rnd % 10 == 0: score += 1
                                            if t_p % 30 == 0: score += 4
                                            elif t_p % 15 == 0: score += 2
                                            elif t_p % 10 == 0: score += 1
                                            if w_i1 % 10 == 0: score += 2
                                            elif w_i1 % 5 == 0: score += 1
                                            if w_p % 10 == 0: score += 2
                                            elif w_p % 5 == 0: score += 1
                                            
                                            sols.append({'n': n, 't_ges': t_ges, 't_i': t_i_rnd, 't_p': t_p, 'w_i1': w_i1, 'G': G, 'w_p': w_p, 'w_avg': int(round(w_avg_calc, 0)), 'score': score})
                                            if len(sols) >= 1000:
                                                max_reached = True
                                                break
                sols.sort(key=lambda x: x['score'], reverse=True)
                st.session_state['smart_sols'] = sols[:100]
                st.session_state['smart_searched'] = True
                
            if st.session_state.get('smart_searched'):
                sols = st.session_state.get('smart_sols', [])
                if not sols:
                    st.warning("⚠️ Keine Lösung gefunden! Bitte lockere die Toleranzen (z.B. größere Spanne bei Ø-Watt oder Gesamtzeit).")
                else:
                    st.success(f"✅ {len(sols)} mögliche Kombinationen gefunden (Zeige max. 100). Bitte wähle eine aus der Tabelle aus:")
                    df_sols = pd.DataFrame(sols)
                    df_sols = df_sols.drop(columns=['score'], errors='ignore')
                    
                    df_sols['t_i_fmt'] = df_sols['t_i'].apply(lambda x: f"{int(round(x))//60:02d}:{int(round(x))%60:02d}")
                    df_sols['t_p_fmt'] = df_sols['t_p'].apply(lambda x: f"{int(round(x))//60:02d}:{int(round(x))%60:02d}")
                    
                    df_sols = df_sols[['t_ges', 'n', 't_i_fmt', 't_p_fmt', 'w_i1', 'G', 'w_p', 'w_avg']]
                    df_sols = df_sols.rename(columns={
                        't_ges': 'Blockzeit (Min)', 'n': 'Anzahl', 't_i_fmt': 'Intervall', 
                        't_p_fmt': 'Pause', 'w_i1': 'Start-Watt', 'G': '+ Watt/Int', 
                        'w_p': 'Pausen-Watt', 'w_avg': 'Ø-Watt'
                    })
                    
                    sel = st.dataframe(df_sols, selection_mode="single-row", on_select="rerun", use_container_width=True, hide_index=True)
                    
    if builder_type == "Stair" and st.session_state.get('smart_searched') and sel and sel.get("selection", {}).get("rows"):
        with col_wb_right:
            idx = sel["selection"]["rows"][0]
            best_sol = sols[idx]
            
            t_i_str = f"{int(round(best_sol['t_i']))//60:02d}:{int(round(best_sol['t_i']))%60:02d}"
            st.markdown(f"#### 📊 Vorschau: {best_sol['n']}x {t_i_str} Intervalle")
            
            x_vals = [0]
            y_vals = [0]
            c_t = 0
            
            wu_w = int(wb_ftp * 0.6)
            if wu_dur > 0:
                x_vals.extend([c_t, c_t + wu_dur*60])
                y_vals.extend([wu_w, wu_w])
                c_t += wu_dur*60
            
            for i in range(best_sol['n']):
                w_i = best_sol['w_i1'] + i * best_sol['G']
                x_vals.extend([c_t, c_t + best_sol['t_i']])
                y_vals.extend([w_i, w_i])
                c_t += best_sol['t_i']
                
                if i < best_sol['n'] - 1:
                    x_vals.extend([c_t, c_t + best_sol['t_p']])
                    y_vals.extend([best_sol['w_p'], best_sol['w_p']])
                    c_t += best_sol['t_p']
                    
            cd_w = int(wb_ftp * 0.4)
            if cd_dur > 0:
                x_vals.extend([c_t, c_t + cd_dur*60])
                y_vals.extend([cd_w, cd_w])
                c_t += cd_dur*60
                
            fig = px.line(x=x_vals, y=y_vals)
            fig.update_traces(line_shape='hv', line=dict(color='#FFA500', width=2))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Zeit (Sekunden)", yaxis_title="Watt")
            st.plotly_chart(fig, width='stretch')
            
            st.markdown("#### 💾 Exportieren")
            c_exp1, c_exp2 = st.columns([1, 1])
            wb_name = st.text_input("Workout Name", "Smart Workout")
            
            def build_smart_zwo():
                lines = []
                lines.append('<workout_file>')
                lines.append('  <author>Powerdata Dashboard</author>')
                lines.append(f'  <name>{wb_name}</name>')
                lines.append('  <sportType>bike</sportType>')
                lines.append('  <tags></tags>')
                lines.append('  <workout>')
                if wu_dur > 0:
                    lines.append(f'    <Warmup Duration="{wu_dur*60}" PowerLow="0.4" PowerHigh="0.6" />')
                
                for i in range(best_sol['n']):
                    w_i = best_sol['w_i1'] + i * best_sol['G']
                    lines.append(f'    <SteadyState Duration="{int(best_sol["t_i"])}" Power="{w_i/wb_ftp:.2f}" />')
                    if i < best_sol['n'] - 1:
                        lines.append(f'    <SteadyState Duration="{int(best_sol["t_p"])}" Power="{best_sol["w_p"]/wb_ftp:.2f}" />')
                        
                if cd_dur > 0:
                    lines.append(f'    <Cooldown Duration="{cd_dur*60}" PowerLow="0.6" PowerHigh="0.4" />')
                    
                lines.append('  </workout>')
                lines.append('</workout_file>')
                return "\n".join(lines)
            
            c_exp1.download_button("💾 Als Zwift (.zwo) speichern", data=build_smart_zwo(), file_name=f"{wb_name.replace(' ', '_')}.zwo", mime="application/xml", use_container_width=True)
            
            if wb_athlete:
                if c_exp2.button("☁️ In Intervals.icu hochladen", use_container_width=True):
                    with st.spinner("Lade hoch..."):
                        t_lines = []
                        if wu_dur > 0: t_lines.append(f"- {wu_dur}m {int(wb_ftp*0.4)}-{int(wb_ftp*0.6)}W")
                        
                        for i in range(best_sol['n']):
                            w_i = best_sol['w_i1'] + i * best_sol['G']
                            ti_str = f"{int(round(best_sol['t_i']))//60:02d}:{int(round(best_sol['t_i']))%60:02d}"
                            t_lines.append(f"- {ti_str} {w_i}W")
                            if i < best_sol['n'] - 1:
                                tp_str = f"{int(round(best_sol['t_p']))//60:02d}:{int(round(best_sol['t_p']))%60:02d}"
                                t_lines.append(f"- {tp_str} {best_sol['w_p']}W")
                                
                        if cd_dur > 0: t_lines.append(f"- {cd_dur}m {int(wb_ftp*0.6)}-{int(wb_ftp*0.4)}W")
                        
                        i_text = "\n".join(t_lines)
                        ok, msg = upload_workout_to_intervals(ath_row['api_key'], ath_row.get('intervals_id', '0'), datetime.now() + timedelta(days=1), wb_name, "Erstellt mit dem Smart Builder", i_text)
                        if ok: st.success(msg)
                        else: st.error(msg)

with tabs[5]:
    st.subheader("⚙️ Einstellungen")
    st.markdown("### 🔑 Passwort ändern")
    with st.form("change_pwd_form", clear_on_submit=True):
        old_pass = st.text_input("Aktuelles Passwort", type="password")
        new_pass = st.text_input("Neues Passwort", type="password")
        new_pass2 = st.text_input("Neues Passwort bestätigen", type="password")
        if st.form_submit_button("Passwort aktualisieren"):
            if old_pass and new_pass and new_pass2:
                status, _, _ = check_login(st.session_state['user'], old_pass)
                if status == "success":
                    if new_pass == new_pass2:
                        update_user_password(st.session_state['user_id'], new_pass)
                        st.success("Dein Passwort wurde erfolgreich geändert!")
                    else:
                        st.error("Die neuen Passwörter stimmen nicht überein.")
                else:
                    st.error("Das aktuelle Passwort ist falsch.")
            else:
                st.warning("Bitte fülle alle Felder aus.")

# --- ADMIN-CHECK ---
if len(tabs) > 5:
    with tabs[6]:
        # WICHTIG: Alles ab hier muss eingerückt sein!
        if st.session_state.get('role') == 'admin':
            st.subheader("🛠️ Admin: Athleten verwalten")
            
            # --- AUSSTEHENDE FREIGABEN ---
            conn = get_db_connection()
            try:
                unapproved = conn.query("SELECT id, name FROM users WHERE approved = FALSE", ttl=0)
                if not unapproved.empty:
                    st.markdown("### ⏳ Ausstehende Freigaben")
                    for _, row in unapproved.iterrows():
                        col_u1, col_u2 = st.columns([0.8, 0.2])
                        col_u1.info(f"**{row['name']}** hat sich neu registriert und wartet auf Freigabe.")
                        if col_u2.button("✅ Freigeben", key=f"approve_{row['id']}", use_container_width=True):
                            with conn.session as s:
                                s.execute(text("UPDATE users SET approved = TRUE WHERE id = :uid"), {"uid": int(row['id'])})
                                s.commit()
                            st.success(f"{row['name']} wurde erfolgreich freigegeben!")
                            st.rerun()
                    st.markdown("---")
            except Exception:
                pass
                
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
                        if name_in and key_in and id_in and pwd_in:
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
            
            st.markdown("---")
            st.markdown("### ⚡ Migration: Historische Daten zu Parquet")
            st.info("Lädt fehlende FIT-Dateien über die Intervals-API, parst sie und speichert sie als blitzschnelle Parquet-Dateien im Supabase Storage Bucket 'workouts'.")
            
            if st.button("Start Parquet Migration", use_container_width=True):
                if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets:
                    st.error("Bitte füge SUPABASE_URL und SUPABASE_KEY in deine st.secrets (secrets.toml) ein!")
                else:
                    sb = get_supabase_client()
                    
                    conn = get_db_connection()
                    workouts = conn.query("SELECT id, intervals_activity_id, user_id FROM workouts WHERE intervals_activity_id IS NOT NULL", ttl=0)
                    
                    if workouts.empty:
                        st.info("Keine Workouts mit Cloud-IDs gefunden.")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            existing_files_res = sb.storage.from_("workouts").list()
                            existing_files = [f["name"] for f in existing_files_res] if existing_files_res else []
                        except Exception as e:
                            existing_files = []
                            st.warning(f"Konnte bestehende Dateien nicht abrufen (Bucket 'workouts' nicht gefunden?): {e}")

                        success_count = 0
                        error_count = 0
                        df_all_users = load_all_athletes()
                        api_keys = dict(zip(df_all_users['id'], df_all_users['api_key']))
                        total = len(workouts)
                        
                        for idx, row in workouts.iterrows():
                            act_id = str(row['intervals_activity_id'])
                            uid = row['user_id']
                            file_name = f"workout_{act_id}.parquet"
                            
                            if file_name in existing_files:
                                success_count += 1
                            else:
                                status_text.write(f"Verarbeite Workout {idx+1}/{total} (ID: {act_id})...")
                                api_k = api_keys.get(uid)
                                if api_k:
                                    bin_data, _ = download_original_fit_file(api_k, act_id)
                                    if bin_data:
                                        try:
                                            fitfile = fitparse.FitFile(io.BytesIO(bin_data))
                                            df_fit = pd.DataFrame([r.get_values() for r in fitfile.get_messages('record')])
                                            if not df_fit.empty and 'timestamp' in df_fit.columns:
                                                df_fit['timestamp'] = pd.to_datetime(df_fit['timestamp'])
                                                df_fit.set_index('timestamp', inplace=True)
                                                
                                                # Fix für PyArrow: Alle komplexen Objekte in Strings umwandeln
                                                for col in df_fit.columns:
                                                    if df_fit[col].dtype == 'object':
                                                        df_fit[col] = df_fit[col].astype(str)
                                                
                                                parquet_buffer = io.BytesIO()
                                                df_fit.to_parquet(parquet_buffer, engine='pyarrow', index=True)
                                                parquet_buffer.seek(0)
                                                
                                                sb.storage.from_("workouts").upload(file_name, parquet_buffer.read(), {"content-type": "application/octet-stream"})
                                                success_count += 1
                                            else: 
                                                st.warning(f"Workout {act_id} hat keine Zeitstempel (Leer).")
                                                error_count += 1
                                        except Exception as e: 
                                            st.error(f"Fehler bei Workout {act_id}: {e}")
                                            error_count += 1
                                    else: 
                                        st.warning(f"Konnte FIT-Datei für {act_id} nicht herunterladen.")
                                        error_count += 1
                                else: 
                                    st.warning(f"Kein API Key für User ID {uid} gefunden.")
                                    error_count += 1
                            progress_bar.progress((idx + 1) / total)
                            
                        status_text.write(f"✅ Migration abgeschlossen! {success_count} erfolgreich, {error_count} Fehler.")
        else:
            st.error("Zugriff verweigert! Nur für den Administrator.")

if len(tabs) > 6:
    with tabs[7]:
        if st.session_state.get('role') == 'admin':
            st.subheader("🚀 Bulk Data Analyser")
            
            if st.session_state.get('bulk_active'):
                b_idx = st.session_state.get('bulk_index', 0)
                b_targets = st.session_state.get('bulk_targets', [])
                
                if b_idx < len(b_targets):
                    current_target = b_targets[b_idx]
                    
                    st.progress((b_idx) / len(b_targets), text=f"Bearbeite Workout {b_idx + 1} von {len(b_targets)}: {current_target['Name']}")
                    st.markdown(f"### ⚙️ {current_target['Name']} ({current_target['Datum']})")
                    
                    if st.session_state.get('bulk_loaded_idx') != b_idx:
                        with st.spinner("Lade Workout..."):
                            b_df = get_activity_df(current_target['ID'], st.session_state['bulk_api_k'])
                            if not b_df.empty:
                                try:
                                    st.session_state['bulk_df'] = b_df
                                    st.session_state['bulk_loaded_idx'] = b_idx
                                    st.session_state['erfassungs_modus'] = "Automatisch (Algorithmus)"
                                    st.session_state['manual_intervals'] = []
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Fehler beim Parsen der FIT-Datei: {e}")
                                    st.session_state['bulk_index'] += 1
                                    st.rerun()
                            else:
                                st.error("Fehler beim Laden der Daten. Überspringe...")
                                st.session_state['bulk_index'] += 1
                                st.rerun()
                                
                    b_df = st.session_state.get('bulk_df', pd.DataFrame())
                    if not b_df.empty:
                        render_analysis_ui(b_df, current_target['Name'], st.session_state['bulk_uid'], current_target['ID'], st.session_state.get('bulk_min_power', 185), st.session_state.get('bulk_ftp', 250), True, key_suffix="bulk", is_bulk=True)
                    
                else:
                    st.success("🎉 Bulk Analyse abgeschlossen!")
                    if st.button("Beenden"):
                        st.session_state['bulk_active'] = False
                        clear_bulk_targets()
                        st.rerun()
            else:
                df_all_users = load_all_athletes()
                if not df_all_users.empty:
                    b_col1, b_col2, b_col3, b_col4 = st.columns([2, 1.5, 1.5, 5])
                    with b_col1:
                        opts_bulk = df_all_users["name"].tolist()
                        def_idx_bulk = opts_bulk.index(st.session_state['user']) if st.session_state['user'] in opts_bulk else 0
                        bulk_user_select = st.selectbox("Athlet", options=opts_bulk, index=def_idx_bulk, key="bulk_user", on_change=clear_bulk_targets)
                    
                    bulk_user_row = df_all_users.loc[df_all_users["name"] == bulk_user_select].iloc[0]
                    bulk_api_k = bulk_user_row["api_key"]
                    bulk_uid = int(bulk_user_row["id"])
                    
                    with b_col2:
                        bulk_start = st.date_input("Start-Datum", datetime.now() - timedelta(days=60), format="DD-MM-YYYY", key="bulk_start", on_change=clear_bulk_targets)
                    with b_col3:
                        bulk_end = st.date_input("End-Datum", datetime.now(), format="DD-MM-YYYY", key="bulk_end", on_change=clear_bulk_targets)
                        
                    b_tags = st.multiselect("Intensität (Bulk):", ["HIT", "MIT", "LIT", "HIT 40/20", "GA", "RSH", "Draußen"], default=[], key="bulk_tags", on_change=clear_bulk_targets)
                    
                    if st.button("Workouts suchen"):
                        with st.spinner("Lade Aktivitäten..."):
                            b_events, b_err = fetch_calendar_events(bulk_api_k, bulk_start, bulk_end)
                            
                        if b_events:
                            conn = get_db_connection()
                            try:
                                existing_workouts = conn.query("SELECT intervals_activity_id FROM workouts WHERE user_id = :uid AND intervals_activity_id IS NOT NULL", params={"uid": bulk_uid}, ttl=0)
                                existing_ids = existing_workouts['intervals_activity_id'].astype(str).tolist() if not existing_workouts.empty else []
                            except Exception:
                                existing_ids = []
                                
                            bulk_targets = []
                            for e in b_events:
                                if not e.get("id") or str(e.get("id")) in existing_ids: continue
                                
                                ev_type = e.get("type", "UNKNOWN")
                                if ev_type not in ["Ride", "VirtualRide", "IndoorRide"]: continue
                                
                                ev_name = e.get("name") or "Fahrt"
                                
                                # Laktattests / Tests beim Bulk-Import ignorieren
                                if any(kw in ev_name.lower() for kw in ["test", "laktat", "lactate"]):
                                    continue
                                
                                matches_tag = False
                                if not b_tags: matches_tag = True
                                else:
                                    for tag in b_tags:
                                        if tag == "Draußen":
                                            if any(kw in ev_name.lower() for kw in ["draußen", "rennradfahren", "radfahren", "velotour", "rtf", "fahrt"]):
                                                matches_tag = True; break
                                        elif tag in ev_name:
                                            matches_tag = True; break
                                            
                                if matches_tag:
                                    date_raw = e.get("start_date_local", "0000-00-00")[:10]
                                    bulk_targets.append({
                                        "Datum": date_raw,
                                        "Name": ev_name,
                                        "ID": str(e.get("id"))
                                    })
                            
                            st.session_state['bulk_temp_targets'] = bulk_targets
                            
                    if st.session_state.get('bulk_temp_targets') is not None:
                        bulk_targets = st.session_state['bulk_temp_targets']
                        st.info(f"Es wurden **{len(bulk_targets)}** neue Workouts gefunden, die den Filtern entsprechen und noch nicht in der Datenbank sind.")
                        
                        if len(bulk_targets) > 0:
                            if st.button("🚀 Bulk Analyse Starten"):
                                st.session_state['bulk_active'] = True
                                st.session_state['bulk_targets'] = bulk_targets
                                st.session_state['bulk_index'] = 0
                                st.session_state['bulk_uid'] = bulk_uid
                                st.session_state['bulk_api_k'] = bulk_api_k
                                st.session_state['bulk_loaded_idx'] = -1
                                
                                stats = get_athlete_stats_from_intervals(bulk_api_k, bulk_user_row.get("intervals_id", "0"))
                                ftp_val = stats.get("FTP", 0)
                                default_min_p = 185
                                try:
                                    if ftp_val and str(ftp_val) != "-":
                                        calc_power = int(float(ftp_val) * 0.7)
                                        default_min_p = max(50, min(calc_power, 400))
                                except Exception: pass
                                st.session_state['bulk_min_power'] = default_min_p
                                st.session_state['bulk_ftp'] = float(ftp_val) if ftp_val and str(ftp_val) != "-" else 250
                                
                                st.rerun()

with tabs[0]:
    if 'df' not in st.session_state: st.session_state['df'] = pd.DataFrame()
    
    df_all_users = get_authorized_athletes(st.session_state['user'], st.session_state['role'], st.session_state.get('user_id'))
    selected_activity_id = None
    filename = ""
    api_k = None
    ftp_val = 0
    active_user_id = None
    default_min_power = 185

    options_list = ["Lokal (.fit-Datei)"] + (df_all_users["name"].tolist() if not df_all_users.empty else [])
    
    default_index = 0
    if st.session_state['user'] in options_list:
        default_index = options_list.index(st.session_state['user'])
        
    col_u1, col_u2, col_u3, col_u4 = st.columns([0.8, 0.6, 0.6, 7.0])
    with col_u1:
        user_select = st.selectbox("Wer hat trainiert?", options=options_list, index=default_index, key="user_select_key")

    if user_select != "Lokal (.fit-Datei)" and not df_all_users.empty:
        active_user_row = df_all_users.loc[df_all_users["name"] == user_select].iloc[0]
        api_k = active_user_row["api_key"]
        active_user_id = int(active_user_row["id"])
        intervals_id = active_user_row.get("intervals_id", "0")
        
        with col_u2:
            start_date = st.date_input("Start-Datum", datetime.now() - timedelta(days=60), format="DD-MM-YYYY")
        with col_u3:
            end_date = st.date_input("End-Datum", datetime.now(), format="DD-MM-YYYY")

        if 'ftp_cache' not in st.session_state: st.session_state['ftp_cache'] = {}
        if active_user_id not in st.session_state['ftp_cache']:
            with perf_track("[tabs[0]] get_athlete_stats_from_intervals"):
                stats = get_athlete_stats_from_intervals(api_k, intervals_id)
            st.session_state['ftp_cache'][active_user_id] = stats.get("FTP", 0)
            
        ftp_val = st.session_state['ftp_cache'][active_user_id]
        try:
            if ftp_val and str(ftp_val) != "-":
                calc_power = int(float(ftp_val) * 0.7)
                default_min_power = max(50, min(calc_power, 400)) # Zwischen 50 und 400 Watt deckeln
        except Exception:
            pass

        with st.spinner("Synchronisiere Aktivitäten..."):
            with perf_track("[tabs[0]] fetch_calendar_events"):
                events, err = fetch_calendar_events(api_k, start_date, end_date)
            
        if events:
            conn = get_db_connection()
            try:
                # ttl=0 erzwingt den Live-Abgleich mit Supabase (umgeht den Cache)
                existing_workouts = conn.query("SELECT intervals_activity_id FROM workouts WHERE user_id = :uid AND intervals_activity_id IS NOT NULL", params={"uid": active_user_id}, ttl=0)
                existing_ids = existing_workouts['intervals_activity_id'].astype(str).tolist() if not existing_workouts.empty else []
            except Exception:
                existing_ids = []
                
            st.markdown("#### Intervals.icu Trainingdata")
            
            col_table, col_filter, _ = st.columns([4.5, 1.0, 4.5])
            
            with col_filter:
                st.markdown("##### 🔍 Filter")
                preselected_tags = st.multiselect("Intensität:", ["HIT", "MIT", "LIT", "GA", "RSH", "Draußen"], default=[])
                custom_search = st.text_input("Freitext-Suche:", value="")
                
            filtered_events = []
            for e in events:
                if not e.get("id"): continue
                
                ev_type = e.get("type", "UNKNOWN")
                if ev_type not in ["Ride", "VirtualRide", "IndoorRide"]: continue
                
                ev_name = e.get("name") or "Fahrt"
                # Exakte Suche der Tags (Case-Sensitive), um z.B. das Wort "mit" von "MIT" zu unterscheiden
                matches_tag = any(tag in ev_name for tag in preselected_tags) if preselected_tags else True
                matches_custom = custom_search.lower() in ev_name.lower() if custom_search else True
                if matches_tag and matches_custom:
                    date_raw = e.get("start_date_local", "0000-00-00")[:10]
                    date_str = f"{date_raw[8:10]}-{date_raw[5:7]}-{date_raw[0:4]}" if len(date_raw) == 10 else date_raw
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
                        use_container_width=True, 
                        on_select="rerun", 
                        selection_mode="single-row", 
                        hide_index=True, 
                        column_config={
                            "ID": None,
                            "Datum": st.column_config.TextColumn("Datum", width=45),
                            "Name": st.column_config.TextColumn("Name", width=108),
                            "Status": st.column_config.TextColumn("Status", width=45)
                        }
                    )
                    if sel and len(sel.get("selection", {}).get("rows", [])) > 0:
                        idx = sel["selection"]["rows"][0]
                        selected_activity_id = df_cockpit.iloc[idx]["ID"]
                        filename = df_cockpit.iloc[idx]["Name"]
                        with st.spinner("Lade Workout..."):
                            with perf_track("[tabs[0]] get_activity_df"):
                                df_fetched = get_activity_df(selected_activity_id, api_k)
                            if not df_fetched.empty:
                                st.session_state['df'] = df_fetched
                else:
                    st.info("Keine Aktivitäten passend zu den Filtern gefunden.")
        elif err:
            st.error(f"API-Fehler: {err}")
        else:
            st.info("Keine Aktivitäten gefunden.")
    elif user_select == "Lokal (.fit-Datei)":
        up = st.file_uploader("Datei wählen", type=["fit"])
        if up:
            fitfile = fitparse.FitFile(up)
            st.session_state['df'] = pd.DataFrame([r.get_values() for r in fitfile.get_messages('record')])
            st.session_state['df']['timestamp'] = pd.to_datetime(st.session_state['df']['timestamp'])
            st.session_state['df'].set_index('timestamp', inplace=True)
            filename = up.name

    df = st.session_state['df']
        
    # --- VERARBEITUNG & ALGORITHMUS ---
    if not df.empty:
        is_admin = st.session_state.get('role') == 'admin'
        with perf_track("[tabs[0]] render_analysis_ui"):
            render_analysis_ui(df, filename, active_user_id, selected_activity_id, default_min_power, ftp_val, is_admin, key_suffix="main")
    else:
        st.info("Bitte Athlet wählen oder Datei hochladen.")

with tabs[2]:
    st.subheader("🔋 Fatigue Resistance & Efficiency Loss")
    
    authorized_athletes_fr = get_authorized_athletes(st.session_state['user'], st.session_state['role'], st.session_state.get('user_id'))
    
    if authorized_athletes_fr.empty:
        st.warning("Keine Athleten gefunden.")
    else:
        if 'fr_selected_ids' not in st.session_state:
            st.session_state['fr_selected_ids'] = []
            
        def toggle_fr_workout(wid):
            if wid in st.session_state['fr_selected_ids']:
                st.session_state['fr_selected_ids'].remove(wid)
            else:
                st.session_state['fr_selected_ids'].append(wid)

        c1, c2 = st.columns([2, 6])
        
        with c1:
            c_sel1, c_sel2 = st.columns(2)
            opts_fr = authorized_athletes_fr["name"].tolist()
            def_idx_fr = opts_fr.index(st.session_state['user']) if st.session_state['user'] in opts_fr else 0
            with c_sel1:
                selected_name_fr = st.selectbox("Athlet wählen:", options=opts_fr, index=def_idx_fr, key="fr_athlete_selector")
            athlete_row_fr = authorized_athletes_fr[authorized_athletes_fr["name"] == selected_name_fr].iloc[0]
            with c_sel2:
                filter_type_fr = st.selectbox("Typ-Filter", ["ALLE", "LIT", "MIT", "HIT", "HIT 40/20", "GA", "RSH", "Draußen"], key="fr_filter_type")
            
            c_d1, c_d2 = st.columns(2)
            with c_d1: fr_start = st.date_input("Von", datetime.now() - timedelta(days=14), format="DD.MM.YYYY", key="fr_start")
            with c_d2: fr_end = st.date_input("Bis", datetime.now(), format="DD.MM.YYYY", key="fr_end")
            search_query_fr = st.text_input("Suche:", key="fr_search_input")

        uid_val_fr = int(athlete_row_fr['id'])
        with perf_track("[tabs[2]] fetch_workouts_from_db"):
            df_workouts_fr = fetch_workouts_from_db(uid_val_fr)
        
        if not df_workouts_fr.empty:
            conn_fr = get_db_connection()
            try:
                valid_eff_workouts = conn_fr.query("SELECT DISTINCT workout_id FROM intervals WHERE eff_loss IS NOT NULL AND workout_id IN (SELECT id FROM workouts WHERE user_id = :uid)", params={"uid": uid_val_fr}, ttl=0)
                valid_eff_ids = valid_eff_workouts['workout_id'].tolist() if not valid_eff_workouts.empty else []
                df_workouts_fr = df_workouts_fr[df_workouts_fr['id'].isin(valid_eff_ids)]
            except Exception:
                df_workouts_fr = pd.DataFrame()
        
        if df_workouts_fr.empty: 
            st.info(f"Keine Trainingsdaten mit berechnetem 'Efficiency Loss' für {selected_name_fr} gefunden.")
            st.info("💡 Lade alte Workouts unter dem Reiter 'Training einlesen' kurz erneut aus der Cloud rein und speichere sie ab, um die Kennzahl nachträglich zu generieren!")
        else:
            df_workouts_fr['date_dt'] = pd.to_datetime(df_workouts_fr['date'], errors='coerce')
            df_workouts_fr = df_workouts_fr.dropna(subset=['date_dt'])
            df_workouts_fr = df_workouts_fr.sort_values(by='date_dt', ascending=False)
            df_all_user_workouts_fr = df_workouts_fr.copy()

            if filter_type_fr != "ALLE": 
                df_workouts_fr = df_workouts_fr[df_workouts_fr['type'] == filter_type_fr]
            
            mask_date_fr = (df_workouts_fr['date_dt'].dt.date >= fr_start) & (df_workouts_fr['date_dt'].dt.date <= fr_end)
            df_workouts_fr = df_workouts_fr[mask_date_fr]

            if search_query_fr:
                df_workouts_fr = df_workouts_fr[df_workouts_fr['filename'].str.contains(search_query_fr, case=False, na=False) | df_workouts_fr['date'].str.contains(search_query_fr, case=False, na=False)]

            df_all_user_workouts_fr = df_all_user_workouts_fr.sort_values(by='date_dt')
            df_all_user_workouts_fr['date_fmt'] = df_all_user_workouts_fr['date_dt'].dt.strftime('%d.%m.%Y')
            all_possible_workouts_fr = (df_all_user_workouts_fr['date_fmt'] + " (" + df_all_user_workouts_fr['type'] + ")").unique()
            all_possible_workouts_fr = list(dict.fromkeys(all_possible_workouts_fr))
            plotly_colors_fr = px.colors.qualitative.Plotly
            extended_colors_fr = plotly_colors_fr * (len(all_possible_workouts_fr) // len(plotly_colors_fr) + 1)
            global_color_map_fr = {w: extended_colors_fr[i] for i, w in enumerate(all_possible_workouts_fr)}

            with c2:
                c_list_fr, c_sel_fr = st.columns([1.5, 1])
                with c_list_fr:
                    st.markdown("##### 🗂️ Suchergebnis (Workouts)")
                    if df_workouts_fr.empty:
                        st.info("Keine Workouts passend zu Filter & Zeitraum gefunden.")
                    else:
                        for idx, row in df_workouts_fr.iterrows():
                            date_str = row['date_dt'].strftime('%d.%m.%Y')
                            is_sel = row['id'] in st.session_state['fr_selected_ids']
                            st.checkbox(f"{date_str} | {row['type']} ({row['structure']}) | {row['filename']}", value=is_sel, key=f"fr_check_{row['id']}", on_change=toggle_fr_workout, args=(row['id'],))

                with c_sel_fr:
                    st.markdown("##### 📌 Ausgewählt zum Vergleich")
                    if not st.session_state['fr_selected_ids']:
                        st.info("Noch keine Workouts markiert.")
                    else:
                        sel_rows_fr = df_all_user_workouts_fr[df_all_user_workouts_fr['id'].isin(st.session_state['fr_selected_ids'])]
                        sel_rows_fr = sel_rows_fr.sort_values(by='date_dt', ascending=False)
                        for _, srow in sel_rows_fr.iterrows():
                            s_date_str = srow['date_dt'].strftime('%d.%m.%Y')
                            st.markdown(f"- **{s_date_str}**: {srow['type']} ({srow['structure']})")
                        if st.button("🗑️ Auswahl aufheben", use_container_width=True, key="fr_clear_sel"):
                            st.session_state['fr_selected_ids'] = []
                            st.rerun()

            selected_ids_to_render_fr = st.session_state['fr_selected_ids']
            if len(selected_ids_to_render_fr) >= 1:
                st.markdown("---")
                df_compare_fr = fetch_compare_from_db(selected_ids_to_render_fr)
                
                if not df_compare_fr.empty:
                    if 'eff_loss' not in df_compare_fr.columns:
                        df_compare_fr['eff_loss'] = None
                        
                    missing_eff = df_compare_fr['eff_loss'].isna().any()
                    if missing_eff:
                        st.warning("⚠️ Eines oder mehrere der ausgewählten Workouts haben noch keine berechneten Werte für 'Efficiency Loss'.")
                        st.info("💡 Lade diese Workouts einfach unter dem Reiter 'Training einlesen' kurz erneut aus der Cloud rein und speichere sie ab. Der Algorithmus berechnet die neue Kennzahl dann automatisch mit!")

                    has_micro_fr = (df_compare_fr['interval_num'] > 100).any()
                    show_micro_fr = False
                    if has_micro_fr:
                        show_micro_fr = st.checkbox("Einzelansicht (Micro-Intervalle)", value=False, key="fr_show_micro")
                        
                    df_compare_fr['date_dt'] = pd.to_datetime(df_compare_fr['date'], errors='coerce')
                    df_compare_fr['date_fmt'] = df_compare_fr['date_dt'].dt.strftime('%d.%m.%Y')
                    df_compare_fr['Workout'] = df_compare_fr['date_fmt'] + " (" + df_compare_fr['type'] + ")"
                    
                    if has_micro_fr and not show_micro_fr:
                        normal_df = df_compare_fr[df_compare_fr['interval_num'] <= 100].copy()
                        macro_averages = df_compare_fr[(df_compare_fr['interval_num'] > 100) & (df_compare_fr['interval_num'] % 100 == 0)].copy()
                        micro_df = df_compare_fr[(df_compare_fr['interval_num'] > 100) & (df_compare_fr['interval_num'] % 100 != 0)].copy()
                        micro_df['macro_num'] = micro_df['interval_num'] // 100
                        
                        existing_macros = set(zip(macro_averages['Workout'], macro_averages['interval_num'] // 100))
                        fallback_micros = micro_df[~micro_df.apply(lambda row: (row['Workout'], row['macro_num']) in existing_macros, axis=1)]
                        
                        if not fallback_micros.empty:
                            agg_dict = {'avg_power': 'mean', 'avg_hr': 'mean', 'max_hr': 'max', 'duration_sec': 'sum', 'date': 'first', 'type': 'first', 'date_dt': 'first'}
                            if 'filename' in fallback_micros.columns: agg_dict['filename'] = 'first'
                            if 'intervals_activity_id' in fallback_micros.columns: agg_dict['intervals_activity_id'] = 'first'
                            if 'workout_id' in fallback_micros.columns: agg_dict['workout_id'] = 'first'
                            for col in ['std_hr', 'avg_hr_p', 'intervall_eff', 'NP_int', 'time_above_88_hr', 'eff_loss']:
                                if col in fallback_micros.columns: agg_dict[col] = 'mean'
                            grouped = fallback_micros.groupby(['Workout', 'macro_num'], as_index=False).agg(agg_dict)
                            grouped.rename(columns={'macro_num': 'interval_num'}, inplace=True)
                            df_compare_fr = pd.concat([normal_df, macro_averages, grouped], ignore_index=True)
                        else:
                            df_compare_fr = pd.concat([normal_df, macro_averages], ignore_index=True)
                            
                    df_compare_fr = df_compare_fr.sort_values(['date_dt', 'interval_num'])
                    df_compare_fr['int_label'] = df_compare_fr['interval_num'].apply(lambda x: f"Block {x//100} Average" if x >= 100 and x % 100 == 0 else (f"B{x//100}.{x%100:02d}" if x > 100 else str(x)))
                    
                    workout_dates_fr = df_compare_fr[['Workout', 'date_dt']].drop_duplicates(subset=['Workout'])
                    workout_dates_fr = workout_dates_fr.sort_values(by='date_dt', ascending=True)
                    sorted_workouts_fr = workout_dates_fr['Workout'].tolist()
                    
                    st.markdown("#### Fatigue Resistance / Efficiency Loss")
                    
                    c_p1, c_p2 = st.columns(2)
                    with c_p1: 
                        fig1 = px.scatter(df_compare_fr, x="int_label", y="eff_loss", color="Workout", title="Efficiency Loss (W*min/bpm)", color_discrete_map=global_color_map_fr, category_orders={"Workout": sorted_workouts_fr})
                        fig1.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        if not show_micro_fr: fig1.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig1, width='stretch')
                    with c_p2:
                        def determine_int_type(row):
                            combined = str(row.get('type', '')).upper() + " " + str(row.get('filename', '')).upper()
                            if any(x in combined for x in ['HIT', 'RSH', '40/20']): return 'HIT'
                            if 'MIT' in combined: return 'MIT'
                            if any(x in combined for x in ['LIT', 'GA']): return 'LIT'
                            return 'Andere'
                            
                        df_compare_fr['Int_Type'] = df_compare_fr.apply(determine_int_type, axis=1)
                        
                        env_map = {}
                        for w_id in df_compare_fr['workout_id'].dropna().unique():
                            w_rows = df_compare_fr[df_compare_fr['workout_id'] == w_id]
                            if w_rows.empty: continue
                            act_id = w_rows['intervals_activity_id'].iloc[0] if 'intervals_activity_id' in w_rows.columns else None
                            w_type = w_rows['type'].iloc[0] if 'type' in w_rows.columns else ''
                            w_fname = w_rows['filename'].iloc[0] if 'filename' in w_rows.columns else ''
                            combined_env = str(w_type).lower() + " " + str(w_fname).lower()
                            
                            clean_act_id = None
                            if pd.notna(act_id) and str(act_id).strip() not in ["", "None", "nan"]:
                                clean_act_id = str(act_id).strip()
                                if clean_act_id.endswith(".0"): clean_act_id = clean_act_id[:-2]
                                
                            if any(kw in combined_env for kw in ['indoor', 'zwift', 'virtual', 'trainerroad', 'rouvy', 'ergdb']):
                                env_map[w_id] = 'Indoor'
                            elif any(kw in combined_env for kw in ['draußen', 'outdoor', 'draussen']):
                                env_map[w_id] = 'Outdoor'
                            elif clean_act_id:
                                try:
                                    df_temp = get_activity_df(clean_act_id, athlete_row_fr['api_key'])
                                    has_gps = False
                                    if not df_temp.empty and 'position_lat' in df_temp.columns:
                                        if not df_temp['position_lat'].dropna().empty: has_gps = True
                                    env_map[w_id] = 'Outdoor' if has_gps else 'Indoor'
                                except Exception:
                                    env_map[w_id] = 'Indoor'
                            else:
                                env_map[w_id] = 'Outdoor' if any(kw in combined_env for kw in ['ride', 'fahrt', 'radfahren', 'rennradfahren']) else 'Indoor'
                                
                        df_compare_fr['Environment'] = df_compare_fr['workout_id'].map(env_map)
                        
                        color_map = {'LIT': '#FFD700', 'MIT': '#FFA500', 'HIT': '#FF3333', 'Andere': '#A9A9A9'}
                        fig2 = px.scatter(df_compare_fr, x="avg_power", y="eff_loss", color="Int_Type", symbol="Environment", color_discrete_map=color_map, symbol_map={'Indoor': 'circle', 'Outdoor': 'star'}, hover_data={'Workout': True, 'int_label': True, 'Int_Type': False, 'Environment': False}, title="Efficiency Loss vs. Ø Leistung (Intervall)")
                        fig2.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=1, color='rgba(255,255,255,0.3)')))
                        fig2.update_layout(template="plotly_dark", xaxis_title="Ø Watt (Intervall)", yaxis_title="Efficiency Loss (W*min/bpm)", legend_title="Typ & Umgebung")
                        st.plotly_chart(fig2, width='stretch')

with tabs[1]:
    st.subheader("📊 Daten & Auswertung")
    
    authorized_athletes = get_authorized_athletes(st.session_state['user'], st.session_state['role'], st.session_state.get('user_id'))
    
    if authorized_athletes.empty:
        st.warning("Keine Athleten gefunden.")
    else:
        if 'eval_selected_ids' not in st.session_state:
            st.session_state['eval_selected_ids'] = []
            
        def toggle_eval_workout(wid):
            if wid in st.session_state['eval_selected_ids']:
                st.session_state['eval_selected_ids'].remove(wid)
            else:
                st.session_state['eval_selected_ids'].append(wid)

        # Layout: 3 Spalten
        c1, c2, c3 = st.columns([2, 1.5, 4.5])
        
        with c1:
            c_sel1, c_sel2 = st.columns(2)
            opts_eval = authorized_athletes["name"].tolist()
            def_idx_eval = opts_eval.index(st.session_state['user']) if st.session_state['user'] in opts_eval else 0
            with c_sel1:
                selected_name = st.selectbox("Athlet wählen:", options=opts_eval, index=def_idx_eval, key="data_eval_athlete_selector")
            athlete_row = authorized_athletes[authorized_athletes["name"] == selected_name].iloc[0]
            with c_sel2:
                filter_type = st.selectbox("Typ-Filter", ["ALLE", "LIT", "MIT", "HIT", "HIT 40/20", "GA", "RSH", "Draußen"], key="data_eval_filter_type")
            
            c_d1, c_d2 = st.columns(2)
            with c_d1: eval_start = st.date_input("Von", datetime.now() - timedelta(days=14), format="DD-MM-YYYY", key="eval_start")
            with c_d2: eval_end = st.date_input("Bis", datetime.now(), format="DD-MM-YYYY", key="eval_end")
            search_query = st.text_input("Suche:", key="data_eval_search_input")

        with c2:
            st.markdown("##### 👤 Profil")

            # Dynamische Abfrage mittels Helper-Funktion
            with perf_track("[tabs[1]] get_athlete_stats_from_intervals"):
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
            
            profile_html = f"""
            <table style="width: 75%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 1rem;">
                <thead>
                    <tr>
                        <th style="text-align: left; padding-bottom: 8px; border-bottom: 1px solid #444;">{selected_name}</th>
                        <th style="text-align: left; padding-bottom: 8px; border-bottom: 1px solid #444;">Stand: {datetime.now().strftime('%d-%m-%Y')}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td style="padding: 4px 0;">Alter: {stats['Age']}</td><td style="padding: 4px 0;">Gewicht: {str_weight}</td></tr>
                    <tr><td style="padding: 4px 0;">FTP: {str_ftp}</td><td style="padding: 4px 0;">{w_kg}</td></tr>
                    <tr><td style="padding: 4px 0;">Max HF: {str_hr}</td><td style="padding: 4px 0;"></td></tr>
                </tbody>
            </table>
            """
            st.markdown(profile_html, unsafe_allow_html=True)
            
            # --- ANSTEHENDE EVENTS ---
            st.markdown("##### 📅 Anstehende Events")
            with perf_track("[tabs[1]] fetch_upcoming_events"):
                upcoming_races = fetch_upcoming_events(athlete_row['api_key'], athlete_row.get('intervals_id', '0'))
            if upcoming_races:
                events_html = "<ul style='margin-top: 0; padding-left: 20px; line-height: 1.4; font-size: 0.9rem;'>"
                for r in upcoming_races[:3]:
                    r_date = r.get('start_date_local', '')[:10]
                    r_date_str = f"{r_date[8:10]}-{r_date[5:7]}-{r_date[0:4]}" if len(r_date) == 10 else r_date
                    r_name = r.get('name', 'Rennen')
                    
                    dist_m = r.get('distance')
                    elev_m = r.get('total_elevation_gain') or r.get('elevation_gain')
                    
                    details = []
                    if dist_m and float(dist_m) > 0: details.append(f"{float(dist_m)/1000:.1f} km")
                    if elev_m and float(elev_m) > 0: details.append(f"{int(float(elev_m))} hm")
                        
                    detail_str = f" <i>({', '.join(details)})</i>" if details else ""
                    events_html += f"<li style='margin-bottom: 4px;'><strong>{r_date_str}</strong>: {r_name}{detail_str}</li>"
                events_html += "</ul>"
                st.markdown(events_html, unsafe_allow_html=True)
            else:
                st.info("Keine Rennen oder Events im Kalender gefunden.")

        # --- WORKOUT LOGIK ---
        uid_val = int(athlete_row['id'])
        with perf_track("[tabs[1]] fetch_workouts_from_db"):
            df_workouts = fetch_workouts_from_db(uid_val)
        
        if df_workouts.empty: 
            st.info(f"Keine Trainingsdaten für {selected_name} gefunden.")
        else:
            df_workouts['date_dt'] = pd.to_datetime(df_workouts['date'], errors='coerce')
            df_workouts = df_workouts.dropna(subset=['date_dt'])
            df_workouts = df_workouts.sort_values(by='date_dt', ascending=False)
            df_all_user_workouts = df_workouts.copy()

            if filter_type != "ALLE": 
                df_workouts = df_workouts[df_workouts['type'] == filter_type]
            
            # Datum-Filter anwenden
            mask_date = (df_workouts['date_dt'].dt.date >= eval_start) & (df_workouts['date_dt'].dt.date <= eval_end)
            df_workouts = df_workouts[mask_date]

            if search_query:
                df_workouts = df_workouts[df_workouts['filename'].str.contains(search_query, case=False, na=False) | df_workouts['date'].str.contains(search_query, case=False, na=False)]

            # Feste Farbzuordnung für alle Workouts erstellen, chronologisch sortiert
            df_all_user_workouts = df_all_user_workouts.sort_values(by='date_dt')
            df_all_user_workouts['date_fmt'] = df_all_user_workouts['date_dt'].dt.strftime('%d-%m-%Y')
            all_possible_workouts = (df_all_user_workouts['date_fmt'] + " (" + df_all_user_workouts['type'] + ")").unique()
            all_possible_workouts = list(dict.fromkeys(all_possible_workouts))
            plotly_colors = px.colors.qualitative.Plotly
            extended_colors = plotly_colors * (len(all_possible_workouts) // len(plotly_colors) + 1)
            global_color_map = {w: extended_colors[i] for i, w in enumerate(all_possible_workouts)}

            delete_id = None
            with c3:
                c_list, c_sel = st.columns([1.5, 1])
                with c_list:
                    st.markdown("##### 🗂️ Suchergebnis (Workouts)")
                    if df_workouts.empty:
                        st.info("Keine Workouts passend zu Filter & Zeitraum gefunden.")
                    else:
                        for idx, row in df_workouts.iterrows():
                            col_check, col_del = st.columns([0.85, 0.15])
                            date_str = row['date_dt'].strftime('%d-%m-%Y')
                            is_sel = row['id'] in st.session_state['eval_selected_ids']
                            with col_check:
                                st.checkbox(
                                    f"{date_str} | {row['type']} ({row['structure']}) | {row['filename']}", 
                                    value=is_sel, 
                                    key=f"eval_check_{row['id']}",
                                    on_change=toggle_eval_workout,
                                    args=(row['id'],)
                                )
                            with col_del:
                                if st.button("🗑️", key=f"eval_del_{row['id']}"):
                                    delete_id = row['id']

                with c_sel:
                    st.markdown("##### 📌 Ausgewählt zum Vergleich")
                    if not st.session_state['eval_selected_ids']:
                        st.info("Noch keine Workouts markiert.")
                    else:
                        sel_rows = df_all_user_workouts[df_all_user_workouts['id'].isin(st.session_state['eval_selected_ids'])]
                        sel_rows = sel_rows.sort_values(by='date_dt', ascending=False)
                        for _, srow in sel_rows.iterrows():
                            s_date_str = srow['date_dt'].strftime('%d-%m-%Y')
                            st.markdown(f"- **{s_date_str}**: {srow['type']} ({srow['structure']})")
                        
                        if st.button("🗑️ Auswahl aufheben", use_container_width=True):
                            st.session_state['eval_selected_ids'] = []
                            st.rerun()
                        
            if delete_id is not None:
                if delete_id in st.session_state['eval_selected_ids']:
                    st.session_state['eval_selected_ids'].remove(delete_id)
                conn = get_db_connection()
                with conn.session as s:
                    s.execute(text("DELETE FROM workouts WHERE id = :id"), {"id": int(delete_id)})
                    s.commit()
                st.cache_data.clear()
                st.rerun()

            selected_ids_to_render = st.session_state['eval_selected_ids']
            if len(selected_ids_to_render) >= 1:
                st.markdown("---")
                df_compare = fetch_compare_from_db(selected_ids_to_render)
                
                if not df_compare.empty:
                    has_micro = (df_compare['interval_num'] > 100).any()
                    
                    if has_micro:
                        show_micro = st.checkbox("Einzelansicht (Micro-Intervalle)", value=False)
                    else:
                        show_micro = False
                        
                    df_compare['date_dt'] = pd.to_datetime(df_compare['date'], errors='coerce')
                    df_compare['date_fmt'] = df_compare['date_dt'].dt.strftime('%d-%m-%Y')
                    df_compare['Workout'] = df_compare['date_fmt'] + " (" + df_compare['type'] + ")"
                    
                    if has_micro and not show_micro:
                        normal_df = df_compare[df_compare['interval_num'] <= 100].copy()
                        macro_averages = df_compare[(df_compare['interval_num'] > 100) & (df_compare['interval_num'] % 100 == 0)].copy()
                        
                        micro_df = df_compare[(df_compare['interval_num'] > 100) & (df_compare['interval_num'] % 100 != 0)].copy()
                        micro_df['macro_num'] = micro_df['interval_num'] // 100
                        
                        existing_macros = set(zip(macro_averages['Workout'], macro_averages['interval_num']))
                        fallback_micros = micro_df[~micro_df.apply(lambda row: (row['Workout'], row['macro_num']) in existing_macros, axis=1)]
                        
                        if not fallback_micros.empty:
                            agg_dict = {'avg_power': 'mean', 'avg_hr': 'mean', 'max_hr': 'max', 'duration_sec': 'sum', 'date': 'first', 'type': 'first', 'date_dt': 'first'}
                            if 'workout_id' in fallback_micros.columns: agg_dict['workout_id'] = 'first'
                            if 'intervals_activity_id' in fallback_micros.columns: agg_dict['intervals_activity_id'] = 'first'
                            for col in ['std_hr', 'avg_hr_p', 'intervall_eff', 'NP_int', 'time_above_88_hr']:
                                if col in fallback_micros.columns: agg_dict[col] = 'mean'
                                
                            grouped = fallback_micros.groupby(['Workout', 'macro_num'], as_index=False).agg(agg_dict)
                            grouped.rename(columns={'macro_num': 'interval_num'}, inplace=True)
                            
                            df_compare = pd.concat([normal_df, macro_averages, grouped], ignore_index=True)
                        else:
                            df_compare = pd.concat([normal_df, macro_averages], ignore_index=True)
                        
                    # DataFrame verlässlich chronologisch (Alt -> Neu) sortieren
                    df_compare = df_compare.sort_values(['date_dt', 'interval_num'])
                        
                    df_compare['int_label'] = df_compare['interval_num'].apply(lambda x: f"Block {x//100} Average" if x >= 100 and x % 100 == 0 else (f"B{x//100}.{x%100:02d}" if x > 100 else str(x)))
                    
                    # Workouts in echter chronologischer Reihenfolge als Liste extrahieren
                    workout_dates = df_compare[['Workout', 'date_dt']].drop_duplicates(subset=['Workout'])
                    workout_dates = workout_dates.sort_values(by='date_dt', ascending=True)
                    sorted_workouts = workout_dates['Workout'].tolist()
                    
                    st.markdown("#### Intervall-Werte")
                    
                    # Dynamische Spaltenauswahl
                    optional_cols = st.multiselect(
                        "Zusätzliche Tabellenspalten anzeigen:",
                        options=["Eff.", "Max HF %", "Dauer", ">= 88% HF"],
                        default=["Eff.", "Max HF %", "Dauer", ">= 88% HF"]
                    )
                    
                    unique_workouts = sorted_workouts
                    
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
                            display_df['Int.'] = sub_df['interval_num'].apply(
                                lambda x: f"Block {x//100} Avg" if isinstance(x, (int, float)) and x >= 100 and x % 100 == 0 
                                else (f"B{x//100}.{x%100:02d}" if isinstance(x, (int, float)) and x > 100 
                                else str(x))
                            )
                            display_df['Ø W'] = sub_df['avg_power']
                            
                            has_max_hr = str(stats['Max HR']).isdigit() and int(stats['Max HR']) > 0
                            max_hr_val = int(stats['Max HR']) if has_max_hr else 1
                            
                            if has_max_hr:
                                display_df['Ø HF %'] = sub_df['avg_hr'] / max_hr_val * 100
                            else:
                                display_df['Ø HF'] = sub_df['avg_hr']
                            
                            if "Eff." in optional_cols: display_df['Eff.'] = sub_df['intervall_eff'] if 'intervall_eff' in sub_df.columns else 0
                            if "Max HF %" in optional_cols: 
                                if has_max_hr:
                                    display_df['Max HF %'] = sub_df['max_hr'] / max_hr_val * 100
                                else:
                                    display_df['Max HF'] = sub_df['max_hr']
                            if "Dauer" in optional_cols: display_df['Dauer'] = sub_df['duration_sec'].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}")
                            if ">= 88% HF" in optional_cols:
                                total_88_sec_w = sub_df['time_above_88_hr'].sum() if 'time_above_88_hr' in sub_df.columns else 0
                                if pd.isna(total_88_sec_w): total_88_sec_w = 0
                                total_88_str_w = f"{int(total_88_sec_w // 60):02d}:{int(total_88_sec_w % 60):02d}"
                                new_header_88_w = f">= 88% HF (Σ {total_88_str_w})"
                                display_df[new_header_88_w] = sub_df['time_above_88_hr'].apply(lambda x: f"{int((x if pd.notnull(x) else 0) // 60):02d}:{int((x if pd.notnull(x) else 0) % 60):02d}") if 'time_above_88_hr' in sub_df.columns else "00:00"
                            
                            # Average Row hinzufügen
                            avg_row = {"Int.": "Ø Gesamt"}
                            avg_row["Ø W"] = sub_df['avg_power'].mean()
                            
                            if has_max_hr:
                                avg_row["Ø HF %"] = display_df['Ø HF %'].mean()
                            else:
                                avg_row["Ø HF"] = sub_df['avg_hr'].mean()
                                
                            if "Eff." in optional_cols:
                                avg_row["Eff."] = sub_df['intervall_eff'].mean() if 'intervall_eff' in sub_df.columns else 0
                            if "Max HF %" in optional_cols:
                                if has_max_hr:
                                    avg_row["Max HF %"] = display_df['Max HF %'].mean()
                                else:
                                    avg_row["Max HF"] = sub_df['max_hr'].mean()
                            if "Dauer" in optional_cols:
                                mean_dur = sub_df['duration_sec'].mean()
                                avg_row["Dauer"] = f"{int(mean_dur // 60):02d}:{int(mean_dur % 60):02d}"
                            if ">= 88% HF" in optional_cols:
                                mean_88 = sub_df['time_above_88_hr'].mean() if 'time_above_88_hr' in sub_df.columns else 0
                                if pd.isna(mean_88): mean_88 = 0
                                avg_row[new_header_88_w] = f"{int(mean_88 // 60):02d}:{int(mean_88 % 60):02d}"
                                
                            display_df = pd.concat([display_df, pd.DataFrame([avg_row])], ignore_index=True)

                            format_dict = {"Ø W": "{:.0f}"}
                            if has_max_hr:
                                format_dict["Ø HF %"] = "{:.0f} %"
                            else:
                                format_dict["Ø HF"] = "{:.0f}"
                                
                            if "Eff." in optional_cols: format_dict["Eff."] = "{:.2f}"
                            if "Max HF %" in optional_cols: 
                                if has_max_hr:
                                    format_dict["Max HF %"] = "{:.0f} %"
                                else:
                                    format_dict["Max HF"] = "{:.0f}"
                            
                            styled_df = display_df.style.format(format_dict).set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                            
                            st.dataframe(styled_df, hide_index=True, use_container_width=True)

                    st.markdown("---")
                    show_graphs = st.checkbox("Graphansicht für diese Workouts aktivieren", value=False)
                    if show_graphs:
                        c_p3, c_p4 = st.columns(2)
                        with c_p3: 
                            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                            for w in sorted_workouts:
                                sub = df_compare[df_compare['Workout'] == w]
                                c = global_color_map.get(w, '#ffffff')
                                fig3.add_trace(go.Scatter(x=sub['int_label'], y=sub['max_hr'], name=w, mode='lines+markers', line=dict(color=c)), secondary_y=False)
                                if str(stats['Max HR']).isdigit() and int(stats['Max HR']) > 0:
                                    fig3.add_trace(go.Scatter(x=sub['int_label'], y=sub['max_hr'] / int(stats['Max HR']) * 100, showlegend=False, hoverinfo='skip', mode='lines', line=dict(color='rgba(0,0,0,0)')), secondary_y=True)
                            fig3.update_layout(title="Max HF", template="plotly_dark", margin=dict(r=20))
                            fig3.update_yaxes(title_text="bpm", secondary_y=False)
                            if str(stats['Max HR']).isdigit() and int(stats['Max HR']) > 0: fig3.update_yaxes(title_text="% Max HF", secondary_y=True, showgrid=False)
                            if not show_micro: fig3.update_xaxes(tick0=1, dtick=1)
                            st.plotly_chart(fig3, width='stretch')
                        with c_p4: 
                            fig4 = px.scatter(df_compare, x="int_label", y="intervall_eff", color="Workout", title="Efficiency (W/bpm)", color_discrete_map=global_color_map, category_orders={"Workout": sorted_workouts})
                            fig4.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                            if not show_micro: fig4.update_xaxes(tick0=1, dtick=1)
                            st.plotly_chart(fig4, width='stretch')

                    st.markdown("---")
                    st.markdown("#### 📈 Herzfrequenz-Kurven Vergleich (Live aus der Cloud)")
                    st.info("💡 Die kumulierte Herzfrequenz-Kurve wird in Echtzeit aus den hochauflösenden Cloud-Daten berechnet.")
                    
                    if st.button("📈 HF-Kurven für ausgewählte Workouts laden", use_container_width=True, key="load_live_hr_curves"):
                        st.session_state['eval_show_hr_curves'] = True
                        
                    if st.session_state.get('eval_show_hr_curves'):
                        with st.spinner("Berechne Herzfrequenz-Kurven..."):
                            fig_hr_comp = go.Figure()
                            curves_found = False
                            for w in unique_workouts:
                                sub_df = df_compare[df_compare['Workout'] == w]
                                act_id = sub_df['intervals_activity_id'].iloc[0] if 'intervals_activity_id' in sub_df.columns else None
                                
                                clean_act_id = None
                                if pd.notna(act_id) and str(act_id).strip() not in ["", "None", "nan"]:
                                    clean_act_id = str(act_id).strip()
                                    if clean_act_id.endswith(".0"): clean_act_id = clean_act_id[:-2]
                                    
                                if clean_act_id:
                                    df_live = get_activity_df(clean_act_id, athlete_row['api_key'])
                                    if not df_live.empty and 'heart_rate' in df_live.columns:
                                        hr_data = df_live['heart_rate'].dropna()
                                        if not hr_data.empty:
                                            hr_sorted = np.sort(hr_data)[::-1]
                                            time_min = np.arange(1, len(hr_sorted) + 1) / 60.0
                                            color = global_color_map.get(w, '#FF3333')
                                            fig_hr_comp.add_trace(go.Scatter(x=time_min, y=hr_sorted, mode='lines', name=w, line=dict(color=color, width=2)))
                                            curves_found = True
                                        else:
                                            st.warning(f"⚠️ Das Workout '{w}' enthält keine Herzfrequenz-Daten in der Cloud.")
                                    elif not df_live.empty:
                                        st.warning(f"⚠️ Das Workout '{w}' enthält keine Herzfrequenz-Daten in der Cloud.")
                                    else:
                                        st.warning(f"❌ Die Rohdaten für '{w}' konnten nicht geladen werden (Weder als Parquet noch über die Cloud-API).")
                                else:
                                    st.info(f"ℹ️ '{w}' hat keine verknüpfte Cloud-ID (evtl. manuell importiert).")
                            
                            if curves_found:
                                fig_hr_comp.update_layout(
                                    template="plotly_dark",
                                    xaxis_title="Kumulierte Zeit (logarithmisch)",
                                    yaxis_title="Herzfrequenz (bpm)",
                                    margin=dict(l=0, r=0, t=10, b=0),
                                    height=500,
                                    hovermode="x unified"
                                )
                                fig_hr_comp.update_xaxes(type="log", tickvals=[0.1, 1, 5, 10, 30, 60, 120, 240], ticktext=["6s", "1m", "5m", "10m", "30m", "1h", "2h", "4h"])
                                st.plotly_chart(fig_hr_comp, width='stretch', key="hr_curve_comp_chart")
                            else:
                                st.warning("Für die ausgewählten Workouts konnten keine Herzfrequenz-Daten in der Cloud gefunden werden.")

                    st.markdown("---")
                    st.markdown("#### 🗺️ GPS Routen (Live aus der Cloud)")
                    st.info("💡 GPS-Daten verbrauchen extrem viel Speicherplatz und werden deshalb nicht in deiner Datenbank abgelegt. Stattdessen zieht das Dashboard die Karte in Echtzeit basierend auf der Workout-ID aus der Cloud (Intervals.icu)!")
                    
                    if st.button("📍 Karten für ausgewählte Workouts laden", use_container_width=True, key="load_live_maps"):
                        st.session_state['eval_show_live_maps'] = True
                        
                    if st.session_state.get('eval_show_live_maps'):
                        with st.spinner("Lade GPS-Daten aus der Cloud..."):
                            for w in unique_workouts:
                                sub_df = df_compare[df_compare['Workout'] == w]
                                act_id = sub_df['intervals_activity_id'].iloc[0] if 'intervals_activity_id' in sub_df.columns else None
                                
                                clean_act_id = None
                                if pd.notna(act_id) and str(act_id).strip() not in ["", "None", "nan"]:
                                    clean_act_id = str(act_id).strip()
                                    if clean_act_id.endswith(".0"): clean_act_id = clean_act_id[:-2]
                                    
                                if clean_act_id:
                                    df_map_live = get_activity_df(clean_act_id, athlete_row['api_key'])
                                    if not df_map_live.empty:
                                        try:
                                            
                                            if 'position_lat' in df_map_live.columns and 'position_long' in df_map_live.columns:
                                                df_map_live = df_map_live.dropna(subset=['position_lat', 'position_long'])
                                                if not df_map_live.empty:
                                                    if df_map_live['position_lat'].abs().max() > 90:
                                                        df_map_live['lat'] = df_map_live['position_lat'] * (180.0 / (2**31))
                                                        df_map_live['lon'] = df_map_live['position_long'] * (180.0 / (2**31))
                                                    else:
                                                        df_map_live['lat'] = df_map_live['position_lat']
                                                        df_map_live['lon'] = df_map_live['position_long']
                                                    
                                                    fig_map_live = go.Figure()
                                                    fig_map_live.add_trace(go.Scattermapbox(
                                                        lat=df_map_live['lat'], lon=df_map_live['lon'], mode='lines',
                                                        line=dict(width=3, color=global_color_map.get(w, 'rgba(51, 153, 255, 0.7)')),
                                                        name=w
                                                    ))
                                                    fig_map_live.update_layout(mapbox_style="open-street-map", mapbox=dict(center=dict(lat=df_map_live['lat'].mean(), lon=df_map_live['lon'].mean()), zoom=10), margin=dict(l=0, r=0, t=30, b=0), height=400, title=f"Route: {w}")
                                                    st.plotly_chart(fig_map_live, width='stretch', key=f"map_live_{act_id}_{w}")
                                                else:
                                                    st.warning(f"Keine verwertbaren GPS-Koordinaten in {w} gefunden.")
                                            else:
                                                st.warning(f"Das Workout {w} enthält keine GPS-Spuren.")
                                        except Exception as e:
                                            st.error(f"Fehler beim Laden der Karte für {w}: {e}")
                                    else:
                                        st.warning(f"❌ Die Rohdaten für '{w}' konnten nicht geladen werden (Weder als Parquet noch über die Cloud-API).")
                                else:
                                    st.info(f"Das Workout '{w}' hat keine verknüpfte Cloud-ID (wurde vermutlich lokal importiert).")
                else:
                    st.warning("⚠️ Zu diesem Workout wurden keine Intervall-Daten gefunden (vermutlich ein alter/fehlerhafter Speicherstand). Bitte lösche das Workout über den 🗑️-Button und speichere es neu aus der Cloud ein.")

with tabs[3]:
    st.subheader("📈 Trendanalyse")
    
    authorized_athletes_trend = get_authorized_athletes(st.session_state['user'], st.session_state['role'], st.session_state.get('user_id'))
    
    if authorized_athletes_trend.empty:
        st.warning("Keine Athleten gefunden.")
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        
        with c1:
            selected_name_trend = st.selectbox("Athlet wählen:", options=authorized_athletes_trend["name"], key="trend_athlete_selector")
            athlete_row_trend = authorized_athletes_trend[authorized_athletes_trend["name"] == selected_name_trend].iloc[0]
            filter_type_trend = st.selectbox("Typ-Filter", ["ALLE", "LIT", "MIT", "HIT", "GA", "RSH", "Draußen"], key="trend_filter_type")

        with c2:
            st.markdown("##### 👤 Profil")
            with perf_track("[tabs[3]] get_athlete_stats_from_intervals"):
                stats_trend = get_athlete_stats_from_intervals(athlete_row_trend['api_key'], athlete_row_trend.get('intervals_id', '0'))
            
            ftp_t = stats_trend["FTP"]
            weight_t = stats_trend["Weight"]
            try: w_kg_t = f"{round(float(ftp_t) / float(weight_t), 2)} W/kg" if float(weight_t) > 0 else "-"
            except (ValueError, TypeError): w_kg_t = "-"
                
            str_weight_t = f"{float(weight_t):.1f} kg" if str(weight_t) != "-" else "-"
            str_ftp_t = f"{ftp_t} W" if str(ftp_t) != "-" else "-"
            str_hr_t = f"{stats_trend['Max HR']} bpm" if str(stats_trend['Max HR']) != "-" else "-"
            
            profile_html_trend = f"""
            <table style="width: 75%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 1rem;">
                <thead>
                    <tr>
                        <th style="text-align: left; padding-bottom: 8px; border-bottom: 1px solid #444;">{selected_name_trend}</th>
                        <th style="text-align: left; padding-bottom: 8px; border-bottom: 1px solid #444;">Stand: {datetime.now().strftime('%d-%m-%Y')}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td style="padding: 4px 0;">Alter: {stats_trend['Age']}</td><td style="padding: 4px 0;">Gewicht: {str_weight_t}</td></tr>
                    <tr><td style="padding: 4px 0;">FTP: {str_ftp_t}</td><td style="padding: 4px 0;">{w_kg_t}</td></tr>
                    <tr><td style="padding: 4px 0;">Max HF: {str_hr_t}</td><td style="padding: 4px 0;"></td></tr>
                </tbody>
            </table>
            """
            st.markdown(profile_html_trend, unsafe_allow_html=True)
            
        uid_val_trend = int(athlete_row_trend['id'])
        with perf_track("[tabs[3]] fetch_workouts_from_db"):
            df_workouts_trend_raw = fetch_workouts_from_db(uid_val_trend)
        
        if df_workouts_trend_raw.empty:
            st.info(f"Keine Trainingsdaten für {selected_name_trend} gefunden.")
        else:
            conn = get_db_connection()
            try:
                df_all_ints = conn.query("SELECT workout_id, duration_sec, interval_num FROM intervals WHERE workout_id IN (SELECT id FROM workouts WHERE user_id = :uid)", params={"uid": uid_val_trend}, ttl=0)
            except Exception:
                df_all_ints = pd.DataFrame()
                
            def build_real_struct_from_ints(w_id, w_type, fallback_struct):
                if w_type == 'HIT 40/20': 
                    m = re.search(r'(\d+x\d+x40/20)', str(fallback_struct))
                    return m.group(1) if m else '40/20'
                    
                if df_all_ints.empty: 
                    m = re.search(r'(\d+x\d+(?:-\d+)*)', str(fallback_struct))
                    return m.group(1) if m else "Manuell"
                    
                w_ints = df_all_ints[df_all_ints['workout_id'] == int(w_id)].sort_values('interval_num')
                if w_ints.empty: 
                    m = re.search(r'(\d+x\d+(?:-\d+)*)', str(fallback_struct))
                    return m.group(1) if m else "Manuell"
                
                real_ints = w_ints[~((w_ints['interval_num'] >= 100) & (w_ints['interval_num'] % 100 == 0))]
                if real_ints.empty: 
                    return "Manuell"
                
                n = len(real_ints)
                durs = [max(1, int(round(float(d) / 60.0))) for d in real_ints['duration_sec'] if pd.notna(d)]
                if len(set(durs)) > 1:
                    return f"{n}x{'-'.join(map(str, durs))}"
                else:
                    return f"{n}x{durs[0]}" if durs else "Manuell"

            def get_real_int_time(w_id, w_type, fallback_count, fallback_len):
                if w_type == 'HIT 40/20' or df_all_ints.empty:
                    return float(fallback_count or 0) * float(fallback_len or 0)
                w_ints = df_all_ints[df_all_ints['workout_id'] == int(w_id)]
                if w_ints.empty:
                    return float(fallback_count or 0) * float(fallback_len or 0)
                real_ints = w_ints[~((w_ints['interval_num'] >= 100) & (w_ints['interval_num'] % 100 == 0))]
                return float(pd.to_numeric(real_ints['duration_sec'], errors='coerce').sum() / 60.0)

            df_workouts_trend_raw['real_structure'] = df_workouts_trend_raw.apply(
                lambda r: build_real_struct_from_ints(r['id'], r['type'], r.get('structure', '')), axis=1
            )
            df_workouts_trend_raw['real_time'] = df_workouts_trend_raw.apply(
                lambda r: get_real_int_time(r['id'], r['type'], r.get('int_count', 0), r.get('int_length', 0)), axis=1
            )
            df_workouts_trend_raw['clean_name'] = df_workouts_trend_raw['type'] + " " + df_workouts_trend_raw['real_structure']
            
            df_workouts_trend = df_workouts_trend_raw.copy()
            if filter_type_trend != "ALLE":
                df_workouts_trend = df_workouts_trend[df_workouts_trend['type'] == filter_type_trend]
            
            if 'int_avg_eff' not in df_workouts_trend.columns:
                st.warning("Die Workouts in der Datenbank enthalten noch keine Metriken für Efficiency.")
            else:
                df_trend = df_workouts_trend.dropna(subset=['int_avg_eff', 'date']).copy()
                df_trend = df_trend[df_trend['int_avg_eff'] > 0]
                
                if df_trend.empty:
                    st.info(f"Keine ausreichenden Daten (Average Efficiency) für den Filter '{filter_type_trend}' vorhanden.")
                else:
                    df_trend['date_parsed'] = pd.to_datetime(df_trend['date'])
                    df_trend = df_trend.sort_values('date_parsed')
                    
                    # Lineare Ausgleichsgerade (Regression) berechnen
                    x_num = df_trend['date_parsed'].map(datetime.toordinal)
                    y = df_trend['int_avg_eff'].astype(float)
                    
                    if len(df_trend) > 1 and df_trend['date_parsed'].nunique() > 1:
                        z = np.polyfit(x_num, y, 1)
                        p = np.poly1d(z)
                        df_trend['trend'] = p(x_num)
                    else:
                        df_trend['trend'] = y
                        
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=df_trend['date_parsed'], y=df_trend['int_avg_eff'], mode='markers', name='Avg Efficiency', marker=dict(size=10, color='#3399FF'), text=df_trend['clean_name'], hovertemplate="<b>%{text}</b><br>Datum: %{x}<br>Efficiency: %{y:.2f}<extra></extra>"))
                    fig_trend.add_trace(go.Scatter(x=df_trend['date_parsed'], y=df_trend['trend'], mode='lines', name='Trend (Linear)', line=dict(color='#FF3333', dash='dash')))
                    
                    fig_trend.update_layout(title=f"Entwicklung der Intervall-Efficiency ({filter_type_trend})", xaxis_title="Datum", yaxis_title="Efficiency (W/bpm)", template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
                    fig_trend.update_xaxes(tickformat="%d-%m-%Y")
                    st.plotly_chart(fig_trend, width='stretch')
                    
            st.markdown("---")
            st.markdown("#### 📊 Workout Verteilung & Zeiten")
            
            col_d1, col_d2, _ = st.columns([1, 1, 2])
            with col_d1:
                trend_start = st.date_input("Von (Verteilung)", datetime.now() - timedelta(days=90), format="DD.MM.YYYY", key="trend_dist_start")
            with col_d2:
                trend_end = st.date_input("Bis (Verteilung)", datetime.now(), format="DD.MM.YYYY", key="trend_dist_end")
            
            df_dist = df_workouts_trend_raw.copy()
            df_dist['date_parsed'] = pd.to_datetime(df_dist['date'], errors='coerce')
            mask_date = (df_dist['date_parsed'].dt.date >= trend_start) & (df_dist['date_parsed'].dt.date <= trend_end)
            df_dist = df_dist[mask_date]
            
            target_types = ["LIT", "MIT", "HIT", "HIT 40/20"]
            df_dist = df_dist[df_dist['type'].isin(target_types)]
            
            if df_dist.empty:
                st.info("Keine Workouts im gewählten Zeitraum für LIT, MIT, HIT oder HIT 40/20 gefunden.")
            else:
                
                c_chart1, c_chart2 = st.columns(2)
                
                with c_chart1:
                    df_counts = df_dist['type'].value_counts().reindex(target_types, fill_value=0).reset_index()
                    df_counts.columns = ['Workout Typ', 'Anzahl']
                    
                    fig_counts = px.bar(df_counts, x='Workout Typ', y='Anzahl', title="Häufigkeit der Workout Typen", 
                                        color='Workout Typ', category_orders={"Workout Typ": target_types},
                                        color_discrete_sequence=px.colors.qualitative.Plotly)
                    fig_counts.update_layout(template="plotly_dark", showlegend=False, margin=dict(t=40, b=0, l=0, r=0), height=350)
                    fig_counts.update_traces(textposition='auto', texttemplate='%{y}')
                    st.plotly_chart(fig_counts, width='stretch')
                    
                with c_chart2:
                    df_time = df_dist.groupby('type')['real_time'].sum().reindex(target_types, fill_value=0).reset_index()
                    df_time.columns = ['Workout Typ', 'Gesamtzeit (Min)']
                    
                    fig_time = px.bar(df_time, x='Workout Typ', y='Gesamtzeit (Min)', title="Summierte Intervall-/Blockzeiten", 
                                      color='Workout Typ', category_orders={"Workout Typ": target_types},
                                      color_discrete_sequence=px.colors.qualitative.Plotly)
                    fig_time.update_layout(template="plotly_dark", showlegend=False, margin=dict(t=40, b=0, l=0, r=0), height=350)
                    fig_time.update_traces(textposition='auto', texttemplate='%{y} Min')
                    st.plotly_chart(fig_time, width='stretch')
                    
                st.markdown("##### Verteilung nach Struktur (z.B. 4x6, 6x3)")
                
                c_sub = st.columns(4)
                for idx, w_type in enumerate(target_types):
                    with c_sub[idx]:
                        df_sub = df_dist[df_dist['type'] == w_type]
                        if not df_sub.empty:
                            df_sub_counts = df_sub['real_structure'].value_counts().reset_index()
                            df_sub_counts.columns = ['Struktur', 'Anzahl']
                            df_sub_counts = df_sub_counts.sort_values('Anzahl', ascending=False)
                            
                            fig_sub = px.bar(df_sub_counts, x='Struktur', y='Anzahl', title=f"{w_type}",
                                             color_discrete_sequence=[px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]])
                            fig_sub.update_layout(template="plotly_dark", showlegend=False, xaxis_title=None, yaxis_title=None, margin=dict(l=0, r=0, t=30, b=0), height=300)
                            fig_sub.update_traces(textposition='auto', texttemplate='%{y}')
                            st.plotly_chart(fig_sub, width='stretch')
                        else:
                            st.info(f"Keine {w_type} Workouts.")

# --- PERFORMANCE MONITOR SIDEBAR ---
if PROFILING:
    perf_log = st.session_state.get("_perf", [])
    with st.sidebar.expander("⏱ Performance Monitor", expanded=True):
        if not perf_log:
            st.caption("Noch keine Timings – interagiere mit dem Dashboard.")
        else:
            total_ms = sum(e for _, e in perf_log) * 1000
            st.caption(f"**Σ instrumentiert: {total_ms:.0f} ms**")
            st.markdown("---")
            for label, elapsed in sorted(perf_log, key=lambda x: x[1], reverse=True):
                ms = elapsed * 1000
                if ms >= 1000:
                    color = "🔴"
                elif ms >= 300:
                    color = "🟡"
                else:
                    color = "🟢"
                st.markdown(f"{color} `{label}`  \n&nbsp;&nbsp;&nbsp;&nbsp;→ **{ms:.0f} ms**")
