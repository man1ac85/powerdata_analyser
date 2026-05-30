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
        match_structure = re.search(r'(\d+)[xX](\d+)', filename)

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

        if f"type_{key_suffix}" in st.session_state:
            detected_type = st.session_state[f"type_{key_suffix}"]
        else:
            if explicit_type:
                detected_type = explicit_type
            elif "rennradfahren" in filename.lower() or "radfahren" in filename.lower():
                detected_type = "Draußen"
            elif any(kw in filename.lower() for kw in ["draußen", "velotour", "rtf", "fahrt"]):
                detected_type = "Draußen"
            else:
                detected_type = "UNKNOWN"
                    
            if detected_type == "UNKNOWN":
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
                c1, c2, c3, _ = st.columns([1, 1, 1, 3])
                with c1: min_power = st.slider("Mindestleistung (Watt)", 50, 400, default_min_power, step=5, key=f"min_p_{key_suffix}")
                with c2: sg_win = st.slider("Filterfenster", 11, 121, 45, step=2, key=f"sg_{key_suffix}")
                with c3: edge_ignore_sec = st.slider("Rand-Ignorierung (Sek)", 0, 600, 120, step=10, key=f"edge_{key_suffix}")
            else:
                c1, _ = st.columns([1, 4])
                with c1: min_power = st.slider("Mindestleistung (Watt)", 50, 400, default_min_power, step=5, key=f"min_p_{key_suffix}")
                sg_win = 45
                edge_ignore_sec = 120
        else:
            c1, _ = st.columns([1, 4])
            with c1:
                pers_equi = st.session_state.get(f"persistent_equi_{key_suffix}", False)
                equidistant = st.checkbox("Äquidistant (Gleiche Zeitabschnitte)", value=pers_equi, key=f"equi_{key_suffix}")
                st.session_state[f"persistent_equi_{key_suffix}"] = equidistant
            expected_intervals = 1
            expected_duration_min = 60
            min_power = default_min_power
            sg_win = 45
            edge_ignore_sec = 120
        
        if 'power' in df.columns:
            df['p_clean'] = df['power'].fillna(0)
            df['p_sg'] = savgol_filter(df['p_clean'].rolling(7, center=True, min_periods=1).median(), window_length=sg_win, polyorder=2)
            
            p_deriv_raw = savgol_filter(df['p_clean'].rolling(7, center=True, min_periods=1).median(), window_length=sg_win, polyorder=2, deriv=1)
            deriv_win = 21 if len(df) >= 21 else (len(df) - 1 if len(df) % 2 == 0 else len(df))
            df['p_deriv'] = savgol_filter(p_deriv_raw, window_length=max(3, deriv_win), polyorder=2)
            
            df['power_roll_30'] = df['p_clean'].rolling(window=30, min_periods=1).mean()
            
            if is_ride_analysis:
                total_sec = len(df)
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
                
            elif mode_type == "Automatisch (Algorithmus)":
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
                        
                        # Algorithmus an User-Regler koppeln
                        if target_intervals and target_intervals > 0 and not match_4020:
                            count_diff = abs(len(current_intervals) - target_intervals)
                            if count_diff == 0:
                                score += 1000 # Massiver Boost, wenn die Anzahl exakt stimmt!
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
                if best_intervals:
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
                    b_id = 1
                    for m_idx, macro in enumerate(macro_blocks, start=1):
                        for u_idx, (sp, ep) in enumerate(macro, start=1):
                            interval_nums[b_id] = (m_idx * 100 + u_idx) if is_micro else m_idx
                            b_id += 1
                else:
                    is_micro = False
                            
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
            
            col_t, col_i, col_btn_save, col_btn_adj = st.columns([1.5, 2.5, 2.5, 3.5])
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
            if num_intervals > 0:
                unique_blocks = df[df['is_interval']]['block_id'].unique()
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
                        
                    intervals_calculated.append({
                        "Intervall": int(interval_nums.get(idx_blk, idx_blk)) if not is_ride_analysis and mode_type == "Automatisch (Algorithmus)" else int(idx_blk), 
                        "Ø Watt": avg_p, 
                        "Ø HF": avg_hr, 
                        "Efficiency": float(round(efficiency, 2)),
                        "NP": float(round(np_val, 1)),
                        "Max HF": max_hr, 
                        "Δ HF+-": std_hr, 
                        "Ø HF_P (20-80)": avg_hr_p, 
                        "Dauer (mm:ss)": f"{int(len(block_df) // 60):02d}:{int(len(block_df) % 60):02d}",
                        "Dauer_sec": len(block_df) 
                    })
            
            metadata = None
            if intervals_calculated:
                n_ints = len(intervals_calculated)
                int_avg_power = int(round(sum(i["Ø Watt"] for i in intervals_calculated) / n_ints))
                int_avg_hr = int(round(sum(i["Ø HF"] for i in intervals_calculated) / n_ints))
                int_avg_eff = float(round(sum(i["Efficiency"] for i in intervals_calculated) / n_ints, 2))
                
                workout_date = df.index.min().strftime('%Y-%m-%d')
                
                final_filename = filename
                if not match_structure:
                    final_filename = f"{detected_type} {workout_structure}"
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
                    "int_count": expected_intervals if not is_ride_analysis else None,
                    "int_length": expected_duration_min if not is_ride_analysis else None
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
                        st.markdown(f'<div style="font-size: 14px; color: rgb(163, 168, 184); margin-bottom: 2px;">Intervalle ({status_text} | Ø {exact_dur_display})</div>', unsafe_allow_html=True)
                        
                        if f"ui_int_{key_suffix}" not in st.session_state:
                            st.session_state[f"ui_int_{key_suffix}"] = int(expected_intervals if expected_intervals > 0 else 1)
                        if f"ui_dur_{key_suffix}" not in st.session_state:
                            st.session_state[f"ui_dur_{key_suffix}"] = int(expected_duration_min if expected_duration_min > 0 else 1)
                            
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            ui_intervals = st.number_input("Anzahl", min_value=1, step=1, key=f"ui_int_{key_suffix}")
                        with cc2:
                            ui_duration = st.number_input("Dauer (Min)", min_value=1, step=1, key=f"ui_dur_{key_suffix}")
                            
                        expected_intervals = ui_intervals
                        expected_duration_min = ui_duration
                        workout_structure = f"{expected_intervals}x{expected_duration_min}"
                else:
                    st.markdown(f'<div style="font-size: 14px; color: rgb(163, 168, 184); margin-bottom: 2px;">Modus</div><div style="font-size: 1.65rem; font-weight: bold; line-height: 1.2; color: #3399FF;">Ride Analysis</div>', unsafe_allow_html=True)
            
            with col_btn_save:
                if metadata and intervals_calculated:
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
                            dup_id = check_duplicate_workout(workout_date, detected_type, workout_structure, active_user_id, selected_activity_id)
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
                                dup_id = check_duplicate_workout(workout_date, detected_type, workout_structure, active_user_id, selected_activity_id)
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
                if intervals_calculated:
                    st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
                    if mode_type == "Automatisch (Algorithmus)" and is_admin and not is_ride_analysis:
                        auto_blocks_timestamps = [(df[df['block_id'] == b].index.min(), df[df['block_id'] == b].index.max()) for b in df[df['is_interval']]['block_id'].unique()]
                        st.button("⚙️ Intervalle nachjustieren", on_click=transfer_to_manual, args=(auto_blocks_timestamps,), use_container_width=True, key=f"adj_{key_suffix}")

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
                    
                    # Intervall-Nummern formatieren für 40/20 (Micro-Intervalle 101 -> Block 1 - 01)
                    display_df['Intervall'] = display_df['Intervall'].apply(
                        lambda x: f"Block {x//100} - {x%100:02d}" if (isinstance(x, (int, float)) and x > 100) else str(x)
                    )
                    
                    display_df = display_df.rename(columns={"Intervall": "Kennwerte pro Intervall"})
                    
                    avg_dur_sec_total = sum(i["Dauer_sec"] for i in intervals_calculated) / len(intervals_calculated)
                    avg_row = {
                        "Kennwerte pro Intervall": "Averages",
                        "Ø Watt": metadata["int_avg_power"],
                        "Ø HF": metadata["int_avg_hr"],
                        "Efficiency": metadata["int_avg_eff"],
                        "NP": float(round(sum(i["NP"] for i in intervals_calculated) / len(intervals_calculated), 1)),
                        "Max HF": int(round(sum(i["Max HF"] for i in intervals_calculated) / len(intervals_calculated))),
                        "Δ HF+-": float(round(sum(i["Δ HF+-"] for i in intervals_calculated) / len(intervals_calculated), 1)),
                        "Ø HF_P (20-80)": int(round(sum(i["Ø HF_P (20-80)"] for i in intervals_calculated) / len(intervals_calculated))),
                        "Dauer (mm:ss)": f"{int(avg_dur_sec_total // 60):02d}:{int(avg_dur_sec_total % 60):02d}"
                    }
                    display_df = pd.concat([display_df, pd.DataFrame([avg_row])], ignore_index=True)
                    
                    styled_df = display_df.style.format({
                        "Ø Watt": "{:.0f}", "Ø HF": "{:.0f}", "Efficiency": "{:.2f}", "NP": "{:.1f}",
                        "Max HF": "{:.0f}", "Δ HF+-": "{:.1f}", "Ø HF_P (20-80)": "{:.0f}"
                    }).set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
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
                fig_main.add_trace(go.Scatter(x=df.index, y=df.get('p_deriv'), mode='lines', name='Steigung (Ableitung)', line=dict(color='#33CC33', width=1.2)), row=3, col=1)

                if not is_ride_analysis:
                    b_pos = st.session_state.get(f'best_thresh_pos_{key_suffix}', 0)
                    b_neg = st.session_state.get(f'best_thresh_neg_{key_suffix}', 0)
                    if b_pos != 0:
                        fig_main.add_hline(y=b_pos, line_dash="dot", line_color="rgba(255, 255, 255, 0.5)", row=3, col=1, annotation_text=f"Pos ({b_pos:.1f})", annotation_position="top left")
                        fig_main.add_hline(y=b_neg, line_dash="dot", line_color="rgba(255, 255, 255, 0.5)", row=3, col=1, annotation_text=f"Neg ({b_neg:.1f})", annotation_position="bottom left")

            plot_height = 800 if is_admin else 550
            fig_main.update_layout(template="plotly_dark", height=plot_height, hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0), legend=dict(yanchor="top", y=1))
            
            selected_data = st.plotly_chart(fig_main, use_container_width=True, on_select="rerun", key=f"chart_{key_suffix}")
            
            if mode_type == "Manuell (Grafische Auswahl)":
                st.markdown("### Manuelle Intervall-Bearbeitung")
                start_t, end_t = None, None
                if selected_data:
                    box = selected_data.get("selection", {}).get("box", None) or selected_data.get("box", None)
                    if box and isinstance(box, list): box = box[0]
                    if isinstance(box, dict) and "x" in box: start_t, end_t = box["x"][0], box["x"][1]
                
                if start_t and end_t:
                    st.write(f"Auswahl: {start_t} bis {end_t}")
                    if st.button("Bereich als Intervall hinzufügen", key=f"add_{key_suffix}"):
                        st.session_state['manual_intervals'].append((pd.to_datetime(start_t), pd.to_datetime(end_t)))
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
    result = conn.query("SELECT id, password_hash, role FROM users WHERE name = :name", params={"name": username}, ttl=0)
    if not result.empty:
        user = result.iloc[0]
        stored_hash = user['password_hash']
        role = str(user['role']).lower().strip() if user['role'] else 'user'
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

# --- LOKALER DEV-MODUS (AUTO-LOGIN) ---
AUTO_LOGIN = False  # <--- Auf False setzen, bevor du den Code produktiv stellst!
AUTO_LOGIN_USERNAME = "Bastian"  # <--- Trage hier deinen Datenbank-Benutzernamen ein

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

def clear_bulk_targets():
    if 'bulk_temp_targets' in st.session_state:
        del st.session_state['bulk_temp_targets']

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

nav_options = ["Training einlesen", "Daten & Auswertung", "Trendanalyse"]
if st.session_state.get('role') == 'admin':
    nav_options.append("👤 Athleten verwalten")
    nav_options.append("Bulk Data Analyser")

tabs = st.tabs(nav_options)

# --- ADMIN-CHECK ---
if len(tabs) > 3:
    with tabs[3]:
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

if len(tabs) > 4:
    with tabs[4]:
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
                            bin_data, _ = download_original_fit_file(st.session_state['bulk_api_k'], current_target['ID'])
                            if bin_data:
                                try:
                                    fitfile = fitparse.FitFile(io.BytesIO(bin_data))
                                    records = [r.get_values() for r in fitfile.get_messages('record')]
                                    b_df = pd.DataFrame(records)
                                    if not b_df.empty and 'timestamp' in b_df.columns:
                                        b_df['timestamp'] = pd.to_datetime(b_df['timestamp'])
                                        b_df.set_index('timestamp', inplace=True)
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
                                st.error("Fehler beim Download. Überspringe...")
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
                        bulk_start = st.date_input("Start-Datum", datetime.now() - timedelta(days=60), format="DD.MM.YYYY", key="bulk_start", on_change=clear_bulk_targets)
                    with b_col3:
                        bulk_end = st.date_input("End-Datum", datetime.now(), format="DD.MM.YYYY", key="bulk_end", on_change=clear_bulk_targets)
                        
                    b_tags = st.multiselect("Intensität (Bulk):", ["HIT", "MIT", "LIT", "HIT 40/20", "GA", "RSH", "Draußen"], default=["HIT", "MIT", "LIT", "HIT 40/20", "GA", "RSH", "Draußen"], key="bulk_tags", on_change=clear_bulk_targets)
                    
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
            start_date = st.date_input("Start-Datum", datetime.now() - timedelta(days=60), format="DD.MM.YYYY")
        with col_u3:
            end_date = st.date_input("End-Datum", datetime.now(), format="DD.MM.YYYY")

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
                
            st.markdown("#### Intervals.icu Trainingdata")
            
            col_table, col_filter, _ = st.columns([4.5, 2.5, 3])
            
            with col_filter:
                st.markdown("##### 🔍 Filter")
                preselected_tags = st.multiselect("Intensität:", ["HIT", "MIT", "LIT", "GA", "RSH", "Draußen"], default=["HIT", "MIT", "LIT", "GA", "RSH", "Draußen"])
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
                        use_container_width=True, 
                        on_select="rerun", 
                        selection_mode="single-row", 
                        hide_index=True, 
                        column_config={
                            "ID": None,
                            "Datum": st.column_config.TextColumn("Datum", width=90),
                            "Name": st.column_config.TextColumn("Name", width=180),
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
        render_analysis_ui(df, filename, active_user_id, selected_activity_id, default_min_power, ftp_val, is_admin, key_suffix="main")
    else:
        st.info("Bitte Athlet wählen oder Datei hochladen.")

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
        c1, c2, c3 = st.columns([1.2, 1, 1.8])
        
        with c1:
            selected_name = st.selectbox("Athlet wählen:", options=authorized_athletes["name"], key="data_eval_athlete_selector")
            athlete_row = authorized_athletes[authorized_athletes["name"] == selected_name].iloc[0]
            filter_type = st.selectbox("Typ-Filter", ["ALLE", "LIT", "MIT", "HIT", "GA", "RSH", "Draußen"], key="data_eval_filter_type")
            c_d1, c_d2 = st.columns(2)
            with c_d1: eval_start = st.date_input("Von", datetime.now() - timedelta(days=14), format="DD.MM.YYYY", key="eval_start")
            with c_d2: eval_end = st.date_input("Bis", datetime.now(), format="DD.MM.YYYY", key="eval_end")
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

            # Feste Farbzuordnung für alle Workouts erstellen, damit Farben beim Auswählen gleich bleiben
            all_possible_workouts = (df_all_user_workouts['date'].str.slice(0, 10) + " (" + df_all_user_workouts['type'] + ")").unique()
            all_possible_workouts = sorted(all_possible_workouts)
            plotly_colors = px.colors.qualitative.Plotly
            extended_colors = plotly_colors * (len(all_possible_workouts) // len(plotly_colors) + 1)
            global_color_map = {w: extended_colors[i] for i, w in enumerate(all_possible_workouts)}

            st.markdown("---")
            c_list, c_sel = st.columns([1.5, 1])
            
            delete_id = None
            with c_list:
                st.markdown("##### 🗂️ Suchergebnis (Workouts)")
                if df_workouts.empty:
                    st.info("Keine Workouts passend zu Filter & Zeitraum gefunden.")
                else:
                    for idx, row in df_workouts.iterrows():
                        col_check, col_del = st.columns([0.85, 0.15])
                        date_str = row['date_dt'].strftime('%d.%m.%Y')
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
                        s_date_str = srow['date_dt'].strftime('%d.%m.%Y')
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
                    
                    col_sw1, col_sw2 = st.columns([3, 1])
                    with col_sw2:
                        show_micro = st.checkbox("Einzelansicht (Micro-Intervalle)", value=False) if has_micro else False
                        
                    df_compare['Workout'] = df_compare['date'].str.slice(0, 10) + " (" + df_compare['type'] + ")"
                    
                    if has_micro and not show_micro:
                        micro_df = df_compare[df_compare['interval_num'] > 100].copy()
                        normal_df = df_compare[df_compare['interval_num'] <= 100].copy()
                        
                        micro_df['macro_num'] = micro_df['interval_num'] // 100
                        agg_dict = {'avg_power': 'mean', 'avg_hr': 'mean', 'max_hr': 'max', 'duration_sec': 'sum', 'date': 'first', 'type': 'first'}
                        for col in ['std_hr', 'avg_hr_p', 'intervall_eff', 'NP_int']:
                            if col in micro_df.columns: agg_dict[col] = 'mean'
                            
                        grouped = micro_df.groupby(['Workout', 'macro_num'], as_index=False).agg(agg_dict)
                        grouped.rename(columns={'macro_num': 'interval_num'}, inplace=True)
                        
                        df_compare = pd.concat([normal_df, grouped], ignore_index=True)
                        df_compare = df_compare.sort_values(['Workout', 'interval_num'])
                        
                    df_compare['int_label'] = df_compare['interval_num'].apply(lambda x: f"B{x//100}.{x%100:02d}" if x > 100 else str(x))
                    
                    c_p1, c_p2 = st.columns(2)
                    c_p3, c_p4 = st.columns(2)
                    
                    with c_p1: 
                        fig1 = px.scatter(df_compare, x="int_label", y="avg_power", color="Workout", title="Ø Watt", color_discrete_map=global_color_map)
                        fig1.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        if not show_micro: fig1.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig1, use_container_width=True)
                    with c_p2:
                        fig_hr = go.Figure()
                        for w in df_compare['Workout'].unique():
                            sub = df_compare[df_compare['Workout'] == w]
                            c = global_color_map.get(w, '#ffffff')
                            fig_hr.add_trace(go.Scatter(x=sub['int_label'], y=sub['avg_hr'], name=f"{w} (Ø)", 
                                                        error_y=dict(type='data', array=sub['std_hr'], visible=True),
                                                        mode='lines+markers', line=dict(color=c), marker=dict(color=c)))
                            fig_hr.add_trace(go.Scatter(x=sub['int_label'], y=sub['avg_hr_p'], name=f"{w} (20-80%)", 
                                                        mode='markers', marker=dict(size=8, color=c, symbol='diamond', line=dict(color='white', width=1))))
                        fig_hr.update_layout(title="Ø Herzfrequenz (+- StdDev)", template="plotly_dark")
                        if not show_micro: fig_hr.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig_hr, use_container_width=True)
                    with c_p3: 
                        fig3 = px.scatter(df_compare, x="int_label", y="max_hr", color="Workout", title="Max HF", color_discrete_map=global_color_map)
                        fig3.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        if not show_micro: fig3.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig3, use_container_width=True)
                    with c_p4: 
                        fig4 = px.scatter(df_compare, x="int_label", y="intervall_eff", color="Workout", title="Efficiency (W/bpm)", color_discrete_map=global_color_map)
                        fig4.update_traces(mode='lines+markers').update_layout(template="plotly_dark")
                        if not show_micro: fig4.update_xaxes(tick0=1, dtick=1)
                        st.plotly_chart(fig4, use_container_width=True)

                    st.markdown("---")
                    st.markdown("#### Intervall-Werte")
                    
                    # Dynamische Spaltenauswahl
                    optional_cols = st.multiselect(
                        "Zusätzliche Tabellenspalten anzeigen:",
                        options=["Eff.", "Max HF", "± HF", "Dauer"],
                        default=["Eff.", "Max HF", "± HF", "Dauer"]
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
                            display_df['Int.'] = sub_df['interval_num'].apply(lambda x: f"Block {x//100} - {x%100:02d}" if x > 100 else str(x))
                            display_df['Ø W'] = sub_df['avg_power']
                            display_df['Ø HF'] = sub_df['avg_hr']
                            
                            if "Eff." in optional_cols: display_df['Eff.'] = sub_df['intervall_eff'] if 'intervall_eff' in sub_df.columns else 0
                            if "Max HF" in optional_cols: display_df['Max HF'] = sub_df['max_hr']
                            if "± HF" in optional_cols: display_df['± HF'] = sub_df['std_hr'] if 'std_hr' in sub_df.columns else 0
                            if "Dauer" in optional_cols: display_df['Dauer'] = sub_df['duration_sec'].apply(lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}")
                            
                            format_dict = {"Ø W": "{:.0f}", "Ø HF": "{:.0f}"}
                            if "Eff." in optional_cols: format_dict["Eff."] = "{:.2f}"
                            if "Max HF" in optional_cols: format_dict["Max HF"] = "{:.0f}"
                            
                            styled_df = display_df.style.format(format_dict).set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
                            
                            st.dataframe(styled_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("⚠️ Zu diesem Workout wurden keine Intervall-Daten gefunden (vermutlich ein alter/fehlerhafter Speicherstand). Bitte lösche das Workout über den 🗑️-Button und speichere es neu aus der Cloud ein.")

with tabs[2]:
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
            stats_trend = get_athlete_stats_from_intervals(athlete_row_trend['api_key'], athlete_row_trend.get('intervals_id', '0'))
            
            ftp_t = stats_trend["FTP"]
            weight_t = stats_trend["Weight"]
            try: w_kg_t = f"{round(float(ftp_t) / float(weight_t), 2)} W/kg" if float(weight_t) > 0 else "-"
            except (ValueError, TypeError): w_kg_t = "-"
                
            str_weight_t = f"{float(weight_t):.1f} kg" if str(weight_t) != "-" else "-"
            str_ftp_t = f"{ftp_t} W" if str(ftp_t) != "-" else "-"
            str_hr_t = f"{stats_trend['Max HR']} bpm" if str(stats_trend['Max HR']) != "-" else "-"
            
            profile_df_trend = pd.DataFrame([
                [f"{selected_name_trend}", f"Stand: {datetime.now().strftime('%d.%m.%y')}"],
                [f"Alter: {stats_trend['Age']}", f"Gewicht: {str_weight_t}"],
                [f"FTP: {str_ftp_t}", w_kg_t],
                [f"Max HF: {str_hr_t}", ""]
            ])
            st.table(profile_df_trend.set_axis([' ', '  '], axis=1))
            
        uid_val_trend = int(athlete_row_trend['id'])
        df_workouts_trend = fetch_workouts_from_db(uid_val_trend)
        
        if df_workouts_trend.empty:
            st.info(f"Keine Trainingsdaten für {selected_name_trend} gefunden.")
        else:
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
                    fig_trend.add_trace(go.Scatter(x=df_trend['date_parsed'], y=df_trend['int_avg_eff'], mode='markers', name='Avg Efficiency', marker=dict(size=10, color='#3399FF'), text=df_trend['filename'], hovertemplate="<b>%{text}</b><br>Datum: %{x}<br>Efficiency: %{y:.2f}<extra></extra>"))
                    fig_trend.add_trace(go.Scatter(x=df_trend['date_parsed'], y=df_trend['trend'], mode='lines', name='Trend (Linear)', line=dict(color='#FF3333', dash='dash')))
                    
                    fig_trend.update_layout(title=f"Entwicklung der Intervall-Efficiency ({filter_type_trend})", xaxis_title="Datum", yaxis_title="Efficiency (W/bpm)", template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_trend, use_container_width=True)
