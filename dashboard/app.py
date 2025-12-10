"""
Streamlit Dashboard for Ensemble Hill Climbing Training
Monitors SQLite database with auto-refresh for live training visualization
"""

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Vibrant, high-contrast color scheme (colorblind-safe)
COLORS = {
    'primary': '#0077BB',      # Bright Blue
    'secondary': '#EE3377',    # Magenta/Pink
    'tertiary': '#009988',     # Teal
    'quaternary': '#EE7733',   # Orange
    'accepted': '#009988',     # Teal for accepted
    'rejected': '#EE3377',     # Magenta for rejected
    'stage1': '#0077BB',       # Bright Blue for stage 1
    'stage2': '#009988',       # Teal for stage 2
    'fill': 'rgba(0, 119, 187, 0.2)',  # Transparent blue for fills
    'line_light': 'rgba(0, 119, 187, 0.6)',  # Light blue for lines
}

# Database path relative to dashboard directory
DB_PATH = Path(__file__).parent.parent / 'data' / 'ensemble_training.db'

# Page configuration
st.set_page_config(
    page_title="Ensemble training monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable automatic reruns to reduce flicker
if 'auto_refresh_count' not in st.session_state:
    st.session_state.auto_refresh_count = 0

# Cache database queries with 60s TTL
@st.cache_data(ttl=60)
def get_ensemble_data():
    """Retrieve all ensemble iteration data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM ensemble_log ORDER BY iteration_num", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_stage2_data():
    """Retrieve all stage 2 training data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM stage2_log ORDER BY ensemble_id, epoch", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=5)  # Refresh every 5 seconds for live batch tracking
def get_batch_status():
    """Retrieve current batch status"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM batch_status ORDER BY worker_id", conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=5)  # Refresh every 5 seconds for live batch tracking
def get_batch_summary():
    """Get batch summary statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'timeout' THEN 1 ELSE 0 END) as timeout,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error,
                AVG(CASE WHEN status = 'completed' THEN runtime_sec ELSE NULL END) as avg_runtime,
                MAX(CASE WHEN status = 'completed' THEN runtime_sec ELSE NULL END) as max_runtime,
                MAX(batch_num) as batch_num
            FROM batch_status
        ''')
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_workers': result[0] or 0,
            'running': result[1] or 0,
            'completed': result[2] or 0,
            'timeout': result[3] or 0,
            'error': result[4] or 0,
            'avg_runtime': result[5] or 0.0,
            'max_runtime': result[6] or 0.0,
            'batch_num': result[7] or 0
        }
    except Exception as e:
        return {
            'total_workers': 0,
            'running': 0,
            'completed': 0,
            'timeout': 0,
            'error': 0,
            'avg_runtime': 0.0,
            'max_runtime': 0.0,
            'batch_num': 0
        }

@st.cache_data(ttl=60)
def get_summary_stats(df):
    """Calculate summary statistics from ensemble data"""
    if df.empty:
        return {
            'total_iterations': 0,
            'accepted_count': 0,
            'rejected_count': 0,
            'ensemble_size': 0,
            'best_stage2_auc': 0.0,
            'current_temp': 0.0,
            'last_update': None,
            'aggregation_method': 'N/A',
            'models_per_hour': 0.0
        }
    
    accepted = df[df['accepted'] == 1]
    
    # Get ensemble size from most recent accepted model's num_models field
    if len(accepted) > 0 and 'num_models' in accepted.columns:
        ensemble_size = accepted.iloc[-1]['num_models']
    else:
        ensemble_size = len(accepted)
    
    # Determine current aggregation method
    if ensemble_size < 10:
        agg_method = "Simple Mean"
    else:
        agg_method = "DNN Weighted"
    
    # Calculate models per hour (successful candidates only)
    models_per_hour = 0.0
    if 'timestamp' in df.columns and len(df) > 1:
        try:
            first_time = datetime.fromisoformat(df['timestamp'].iloc[0])
            last_time = datetime.fromisoformat(df['timestamp'].iloc[-1])
            hours_elapsed = (last_time - first_time).total_seconds() / 3600
            if hours_elapsed > 0:
                # Count successful training runs (exclude timeouts)
                successful_runs = len(df[df.get('timeout', 0) == 0])
                models_per_hour = successful_runs / hours_elapsed
        except (ValueError, TypeError):
            models_per_hour = 0.0
    
    return {
        'total_iterations': len(df),
        'accepted_count': len(accepted),
        'rejected_count': len(df[df['accepted'] == 0]),
        'ensemble_size': ensemble_size,
        'best_stage2_auc': df['stage2_val_auc'].max() if 'stage2_val_auc' in df.columns else 0.0,
        'current_temp': df['temperature'].iloc[-1] if 'temperature' in df.columns else 0.0,
        'last_update': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None,
        'aggregation_method': agg_method,
        'models_per_hour': models_per_hour
    }

def check_database_exists():
    """Check if database file exists"""
    return Path(DB_PATH).exists()

# Main dashboard
st.title("Ensemble hill climbing training monitor")
st.markdown("---")

# Check database existence
if not check_database_exists():
    st.error(f"Database not found at: `{DB_PATH}`")
    st.info("Start the training notebook to create the database.")
    st.stop()

# Load data
ensemble_df = get_ensemble_data()
stage2_df = get_stage2_data()

# Check if data exists
if ensemble_df.empty:
    st.warning("No training data found in database. Waiting for first iteration...")
    st.info("The dashboard will auto-refresh every 60 seconds.")
    st.stop()

# Calculate summary statistics
stats = get_summary_stats(ensemble_df)

# Header metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total Iterations", stats['total_iterations'])
    
with col2:
    st.metric("Ensemble Size", stats['ensemble_size'])
    
with col3:
    st.metric("Best Stage 2 AUC", f"{stats['best_stage2_auc']:.4f}")
    
with col4:
    st.metric("Temperature", f"{stats['current_temp']:.4f}")
    
with col5:
    st.metric("Models/Hour", f"{stats['models_per_hour']:.1f}")
    
with col6:
    if stats['last_update']:
        last_update = datetime.fromisoformat(stats['last_update'])
        time_diff = datetime.now() - last_update
        minutes_ago = int(time_diff.total_seconds() / 60)
        st.metric("Last Update", f"{minutes_ago}m ago")
    else:
        st.metric("Last Update", "N/A")

st.markdown("---")

# Memory usage metrics (if available)
if 'training_memory_mb' in ensemble_df.columns and not ensemble_df['training_memory_mb'].isna().all():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_training_mem = ensemble_df['training_memory_mb'].mean()
        st.metric("Avg training memory", f"{avg_training_mem:.1f} MB")
    
    with col2:
        max_training_mem = ensemble_df['training_memory_mb'].max()
        st.metric("Peak training memory", f"{max_training_mem:.1f} MB")
    
    with col3:
        if 'stage2_memory_mb' in ensemble_df.columns:
            stage2_mem = ensemble_df['stage2_memory_mb'].dropna()
            if not stage2_mem.empty:
                avg_stage2_mem = stage2_mem.mean()
                st.metric("Avg stage 2 memory", f"{avg_stage2_mem:.1f} MB")
            else:
                st.metric("Avg stage 2 memory", "N/A")
        else:
            st.metric("Avg stage 2 memory", "N/A")
    
    st.markdown("---")

# Acceptance rate metric
accept_rate = (stats['accepted_count'] / stats['total_iterations'] * 100) if stats['total_iterations'] > 0 else 0
st.markdown(f"""
<div style="margin-bottom: 1rem;">
    <div style="font-size: 0.875rem; margin-bottom: 0.5rem;">
        Acceptance rate: {accept_rate:.1f}% ({stats['accepted_count']} accepted / {stats['rejected_count']} rejected)
    </div>
    <div style="background-color: rgba(255, 255, 255, 0.1); border-radius: 0.25rem; overflow: hidden; height: 0.5rem;">
        <div style="background-color: {COLORS['stage2']}; height: 100%; width: {accept_rate}%; transition: width 0.3s ease;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Timeout rate metric (if data available)
if 'timeout' in ensemble_df.columns:
    timeout_count = ensemble_df['timeout'].sum()
    timeout_rate = (timeout_count / stats['total_iterations'] * 100) if stats['total_iterations'] > 0 else 0
    st.markdown(f"""
<div style="margin-bottom: 1rem;">
    <div style="font-size: 0.875rem; margin-bottom: 0.5rem;">
        Timeout rate: {timeout_rate:.1f}% ({int(timeout_count)} timeouts / {stats['total_iterations']} iterations)
    </div>
    <div style="background-color: rgba(255, 255, 255, 0.1); border-radius: 0.25rem; overflow: hidden; height: 0.5rem;">
        <div style="background-color: {COLORS['stage2']}; height: 100%; width: {timeout_rate}%; transition: width 0.3s ease;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ====================
# SIDEBAR
# ====================
st.sidebar.title("Navigation")

# Use query parameters to persist page selection across refreshes
if 'page' not in st.query_params:
    st.query_params['page'] = "Current batch status"

# Page selection with query params
page_options = ["Current batch status", "Performance", "Diversity", "Stage 2 DNN", "Memory usage", "Timing"]
current_page = st.query_params.get('page', 'Current batch status')

# If stored page is not in current options (e.g., old "Batch Status"), default to first option
if current_page not in page_options:
    current_page = page_options[0]
    st.query_params['page'] = current_page

page = st.sidebar.radio(
    "Select page",
    page_options,
    label_visibility="collapsed",
    index=page_options.index(current_page)
)

# Update query params when page changes
if page != st.query_params.get('page'):
    st.query_params['page'] = page

# Conditional auto-refresh based on page
# Batch status page: 5 seconds (live worker tracking)
# Other pages: 60 seconds (less frequent updates)
# Use limit parameter to reduce excessive refreshing
if page == "Current batch status":
    refresh_count = st_autorefresh(interval=5000, limit=None, key="refresh_batch_status")
else:
    refresh_count = st_autorefresh(interval=60000, limit=None, key="refresh_other_pages")

# Store refresh count in session state
st.session_state.auto_refresh_count = refresh_count

# ====================
# CURRENT BATCH STATUS PAGE
# ====================
if page == "Current batch status":
    st.subheader("Current batch status")
    
    # Get batch status data
    batch_df = get_batch_status()
    batch_summary = get_batch_summary()
    
    if batch_summary['total_workers'] == 0:
        st.info("No active batch. Waiting for training to start or next batch...")
    else:
        # Batch summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Batch #", batch_summary['batch_num'])
        
        with col2:
            st.metric("Total Workers", batch_summary['total_workers'])
        
        with col3:
            st.metric("Running", batch_summary['running'], 
                     delta_color="off")
        
        with col4:
            st.metric("Completed", batch_summary['completed'],
                     delta_color="normal")
        
        with col5:
            st.metric("Timeout", batch_summary['timeout'],
                     delta_color="inverse" if batch_summary['timeout'] > 0 else "off")
        
        with col6:
            st.metric("Error", batch_summary['error'],
                     delta_color="inverse" if batch_summary['error'] > 0 else "off")
        
        st.markdown("---")
        
        # Progress bar
        if batch_summary['total_workers'] > 0:
            progress_pct = (batch_summary['completed'] + batch_summary['timeout'] + batch_summary['error']) / batch_summary['total_workers'] * 100
            st.markdown(f"""
            <div style="margin-bottom: 1.5rem;">
                <div style="font-size: 0.875rem; margin-bottom: 0.5rem;">
                    Batch progress: {progress_pct:.0f}% ({batch_summary['completed'] + batch_summary['timeout'] + batch_summary['error']}/{batch_summary['total_workers']} workers finished)
                </div>
                <div style="background-color: rgba(255, 255, 255, 0.1); border-radius: 0.25rem; overflow: hidden; height: 1rem;">
                    <div style="background-color: {COLORS['tertiary']}; height: 100%; width: {progress_pct}%; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Runtime metrics
        col1, col2 = st.columns(2)
        
        with col1:
            if batch_summary['avg_runtime'] > 0:
                avg_runtime = batch_summary['avg_runtime']
                if avg_runtime < 60:
                    runtime_display = f"{int(avg_runtime)}s"
                else:
                    runtime_display = f"{int(avg_runtime // 60)}m"
                st.metric("Average runtime (completed)", runtime_display)
        
        with col2:
            if batch_summary['max_runtime'] > 0:
                max_runtime = batch_summary['max_runtime']
                if max_runtime < 60:
                    runtime_display = f"{int(max_runtime)}s"
                else:
                    runtime_display = f"{int(max_runtime // 60)}m"
                st.metric("Max runtime (completed)", runtime_display)
        
        st.markdown("---")
        
        # Worker status table
        if not batch_df.empty:
            st.markdown("### Worker details")
            
            # Calculate runtime for display
            display_df = batch_df.copy()
            
            # Parse timestamps and calculate runtime
            for idx, row in display_df.iterrows():
                if pd.notna(row['start_time']):
                    start = datetime.fromisoformat(row['start_time'])
                    
                    if pd.notna(row['end_time']):
                        end = datetime.fromisoformat(row['end_time'])
                        runtime = (end - start).total_seconds()
                    elif row['status'] == 'running':
                        # Calculate current runtime for running workers
                        runtime = (datetime.now() - start).total_seconds()
                    else:
                        runtime = row['runtime_sec'] if pd.notna(row['runtime_sec']) else 0
                    
                    # Format runtime: seconds if < 60, minutes if >= 60
                    if runtime < 60:
                        display_df.at[idx, 'runtime_display'] = f"{int(runtime)}s"
                    else:
                        display_df.at[idx, 'runtime_display'] = f"{int(runtime // 60)}m"
                else:
                    display_df.at[idx, 'runtime_display'] = "N/A"
            
            # Add status emoji/icon
            status_icons = {
                'running': '🔄',
                'completed': '✅',
                'timeout': '⏱️',
                'error': '❌'
            }
            display_df['status_icon'] = display_df['status'].map(status_icons)
            
            # Select and rename columns for display
            display_columns = {
                'worker_id': 'Worker',
                'iteration_num': 'Iteration',
                'status_icon': '⚡',
                'status': 'Status',
                'classifier_type': 'Classifier',
                'runtime_display': 'Runtime',
                'last_update': 'Last Update'
            }
            
            # Format last update times (12-hour clock)
            display_df['last_update_display'] = display_df['last_update'].apply(
                lambda x: datetime.fromisoformat(x).strftime('%I:%M:%S %p') if pd.notna(x) else 'N/A'
            )
            display_df['last_update'] = display_df['last_update_display']
            
            table_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
            
            # Style the dataframe
            def highlight_status(row):
                if row['Status'] == 'running':
                    return ['background-color: rgba(0, 119, 187, 0.2)'] * len(row)
                elif row['Status'] == 'completed':
                    return ['background-color: rgba(0, 153, 136, 0.2)'] * len(row)
                elif row['Status'] == 'timeout':
                    return ['background-color: rgba(238, 119, 51, 0.2)'] * len(row)
                elif row['Status'] == 'error':
                    return ['background-color: rgba(238, 51, 119, 0.2)'] * len(row)
                return [''] * len(row)
            
            styled_table = table_df.style.apply(highlight_status, axis=1)
            st.dataframe(styled_table, width='stretch', hide_index=True)
            
            # Auto-refresh message
            st.caption("🔄 This page auto-refreshes every 5 seconds to show live worker status")
        else:
            st.warning("No worker data available for current batch")

# ====================
# PERFORMANCE PAGE
# ====================
elif page == "Performance":
    st.subheader("Performance metrics over time")
    
    if not ensemble_df.empty:
        # Combined Stage 1 and Stage 2 validation AUC plot
        fig_combined = go.Figure()
        
        # Use only accepted models for both Stage 1 and Stage 2
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1].sort_values('iteration_num')
        
        # Calculate rolling statistics for Stage 1 AUC (window of 10 accepted models)
        window_size = 10
        stage1_rolling_mean = accepted_df['stage1_val_auc'].rolling(window=window_size, min_periods=1).mean()
        stage1_rolling_std = accepted_df['stage1_val_auc'].rolling(window=window_size, min_periods=1).std()
        
        # Add Stage 1 shaded region (mean ± std)
        fig_combined.add_trace(go.Scatter(
            x=accepted_df['iteration_num'],
            y=stage1_rolling_mean + stage1_rolling_std,
            mode='lines',
            line=dict(width=0, shape='hv'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=accepted_df['iteration_num'],
            y=stage1_rolling_mean - stage1_rolling_std,
            mode='lines',
            line=dict(width=0, shape='hv'),
            fillcolor='rgba(238, 119, 51, 0.2)',
            fill='tonexty',
            name='Stage 1 std dev',
            hoverinfo='skip'
        ))
        
        # Add Stage 1 mean line
        fig_combined.add_trace(go.Scatter(
            x=accepted_df['iteration_num'],
            y=stage1_rolling_mean,
            mode='lines+markers',
            line=dict(color='rgba(238, 119, 51, 0.6)', width=2, dash='dash', shape='hv'),
            marker=dict(size=8, color=COLORS['quaternary']),
            name='Stage 1 mean (individual models)',
            hovertemplate='Iter %{x}<br>Stage 1 Mean: %{y:.4f}<extra></extra>'
        ))
        
        # Add Stage 2 line (ensemble performance)
        fig_combined.add_trace(go.Scatter(
            x=accepted_df['iteration_num'],
            y=accepted_df['stage2_val_auc'],
            mode='lines+markers',
            line=dict(color='rgba(0, 153, 136, 0.6)', width=3, dash='dash', shape='hv'),
            marker=dict(size=8, color=COLORS['tertiary']),
            name='Stage 2 (ensemble)',
            hovertemplate='Iter %{x}<br>Stage 2 AUC: %{y:.4f}<extra></extra>'
        ))
        
        # Add batch boundaries (every 10 accepted models) using num_models column
        # Filter to rows where num_models is a multiple of 10
        batch_rows = accepted_df[accepted_df['num_models'] % 10 == 0]
        batch_iterations = batch_rows['iteration_num'].values
        
        for batch_iter in batch_iterations:
            fig_combined.add_vline(
                x=batch_iter, 
                line_dash="dash", 
                line_color="gray",
                opacity=0.3,
                annotation_text=f"DNN retrain",
                annotation_position="top"
            )
        
        fig_combined.update_layout(
            title=dict(
                text="Validation AUC over time: Stage 1 (individual models) vs Stage 2 (ensemble)",
                y=0.98  # Move title up slightly
            ),
            xaxis_title="Iteration number",
            yaxis_title="Validation AUC",
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,  # Increased from 1.02 to create more space
                xanchor="right",
                x=1
            ),
            margin=dict(t=100)  # Add top margin for legend spacing
        )
        
        st.plotly_chart(fig_combined, width="stretch")
        
        # Stage 1 vs Stage 2 AUC scatter plot
        accepted_df_scatter = ensemble_df[ensemble_df['accepted'] == 1].copy()
        
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=accepted_df_scatter['stage1_val_auc'],
            y=accepted_df_scatter['stage2_val_auc'],
            mode='markers',
            marker=dict(size=8, color=COLORS['primary']),
            text=accepted_df_scatter['iteration_num'],
            hovertemplate='Iteration %{text}<br>Stage 1 AUC: %{x:.4f}<br>Stage 2 AUC: %{y:.4f}<extra></extra>',
            showlegend=False
        ))
        
        fig_scatter.update_layout(
            title="Stage 1 vs stage 2 AUC",
            xaxis_title="Stage 1 mean AUC",
            yaxis_title="Stage 2 ensemble AUC",
            height=400
        )
        
        st.plotly_chart(fig_scatter, width="stretch")
        
        # Ensemble size over time
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1].copy()
        accepted_df['cumulative_ensemble_size'] = range(1, len(accepted_df) + 1)
        
        fig_size = px.line(
            accepted_df,
            x='iteration_num',
            y='cumulative_ensemble_size',
            title="Ensemble size growth",
            labels={'iteration_num': 'Iteration number', 'cumulative_ensemble_size': 'Ensemble size'}
        )
        fig_size.update_traces(
            mode='lines+markers',
            line=dict(dash='dash', shape='hv'),
            marker=dict(size=8),
            line_color='rgba(238, 51, 119, 0.6)',
            marker_color=COLORS['secondary']
        )
        
        st.plotly_chart(fig_size, width="stretch")

# ====================
# DIVERSITY PAGE
# ====================
elif page == "Diversity":
    st.subheader("Diversity metrics")
    
    if not ensemble_df.empty and 'diversity_score' in ensemble_df.columns:
        # Diversity score over iterations
        fig_div = go.Figure()
        
        # Add accepted models
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1]
        fig_div.add_trace(go.Scatter(
            x=accepted_df['iteration_num'],
            y=accepted_df['diversity_score'],
            mode='markers',
            marker=dict(size=6, color=COLORS['accepted'], symbol='circle'),
            name='Accepted',
            text=accepted_df.apply(lambda row: f"Iter {row['iteration_num']}: {row['diversity_score']:.4f}", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add rejected models
        rejected_df = ensemble_df[ensemble_df['accepted'] == 0]
        fig_div.add_trace(go.Scatter(
            x=rejected_df['iteration_num'],
            y=rejected_df['diversity_score'],
            mode='markers',
            marker=dict(size=6, color=COLORS['rejected'], symbol='circle'),
            name='Rejected',
            text=rejected_df.apply(lambda row: f"Iter {row['iteration_num']}: {row['diversity_score']:.4f}", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig_div.update_layout(
            title=dict(text="Diversity score over iterations", y=0.98),
            xaxis_title="Iteration number",
            yaxis_title="Diversity score",
            hovermode='closest',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100)
        )
        
        st.plotly_chart(fig_div, width="stretch")
        
        # Diversity vs Stage 2 AUC scatter
        fig_scatter = px.scatter(
            ensemble_df,
            x='diversity_score',
            y='stage2_val_auc',
            title="Diversity vs stage 2 validation AUC (ensemble performance)",
            labels={'diversity_score': 'Diversity score', 'stage2_val_auc': 'Stage 2 validation AUC'},
            hover_data=['iteration_num', 'accepted']
        )
        
        st.plotly_chart(fig_scatter, width="stretch")
        
        # Composition Analysis - Accepted Models
        st.subheader("Ensemble composition")
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1]
        
        # Classifier type distribution in ensemble
        if 'classifier_type' in ensemble_df.columns and not ensemble_df['classifier_type'].isna().all():
            classifier_counts = accepted_df['classifier_type'].value_counts().sort_values(ascending=True)
            
            fig_classifiers = px.bar(
                x=classifier_counts.values,
                y=classifier_counts.index,
                orientation='h',
                title="Classifiers",
                labels={'x': 'Count', 'y': 'Classifier type'},
                color_discrete_sequence=[COLORS['stage1']]
            )
            
            st.plotly_chart(fig_classifiers, width="stretch")
        
        # Transformer usage frequency
        if 'transformers_used' in accepted_df.columns:
            # Parse comma-separated transformers
            all_transformers = []
            for trans_str in accepted_df['transformers_used'].dropna():
                # Strip whitespace from each transformer name
                transformers = [t.strip() for t in trans_str.split(',') if t.strip()]
                # If no transformers after filtering, it means empty/none
                if not transformers:
                    all_transformers.append('None')
                else:
                    all_transformers.extend(transformers)
            
            if all_transformers:
                transformer_counts = pd.Series(all_transformers).value_counts().sort_values(ascending=True)
                
                fig_trans = px.bar(
                    x=transformer_counts.values,
                    y=transformer_counts.index,
                    orientation='h',
                    title="Feature engineering",
                    labels={'x': 'Count', 'y': 'Transformer'},
                    color_discrete_sequence=[COLORS['stage2']]
                )
                
                st.plotly_chart(fig_trans, width="stretch")
        
        # Dimensionality reduction usage
        if 'pca_components' in accepted_df.columns:
            # Create dimensionality reduction type column
            dim_red_types = []
            for idx, row in accepted_df.iterrows():
                if pd.isna(row['pca_components']) or row.get('use_pca', 0) == 0:
                    dim_red_types.append('None')
                else:
                    # Try to determine the type from pca_components value
                    pca_comp = row['pca_components']
                    if isinstance(pca_comp, str):
                        dim_red_types.append(pca_comp)
                    else:
                        # Numeric value - it's likely PCA or similar
                        # Check if it's a variance ratio (0-1) or component count
                        if pca_comp < 1.0:
                            dim_red_types.append('PCA (variance)')
                        else:
                            dim_red_types.append('PCA/Other')
            
            accepted_df_copy = accepted_df.copy()
            accepted_df_copy['dim_reduction_type'] = dim_red_types
            dim_red_counts = accepted_df_copy['dim_reduction_type'].value_counts().sort_values(ascending=True)
            
            fig_dim_red = px.bar(
                x=dim_red_counts.values,
                y=dim_red_counts.index,
                orientation='h',
                title="Dimensionality reduction",
                labels={'x': 'Count', 'y': 'Technique'},
                color_discrete_sequence=[COLORS['primary']]
            )
            
            st.plotly_chart(fig_dim_red, width="stretch")

# ====================
# STAGE 2 DNN PAGE
# ====================
elif page == "Stage 2 DNN":
    st.subheader("Stage 2 DNN training progress")
    
    if not stage2_df.empty:
        # Get unique ensemble IDs (batches)
        ensemble_ids = sorted(stage2_df['ensemble_id'].unique())
        
        if ensemble_ids:
            # Dropdown to select batch
            selected_id = st.selectbox(
                "Select training batch",
                options=ensemble_ids,
                format_func=lambda x: f"Batch {x.split('_')[-1]}" if '_' in x else x
            )
            
            # Filter data for selected batch
            batch_data = stage2_df[stage2_df['ensemble_id'] == selected_id]
            
            # Training curves
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss curves
                fig_loss = go.Figure()
                
                fig_loss.add_trace(go.Scatter(
                    x=batch_data['epoch'],
                    y=batch_data['train_loss'],
                    mode='lines+markers',
                    name='Train Loss',
                    line=dict(color=COLORS['primary'], width=3),
                    marker=dict(size=6)
                ))
                
                fig_loss.add_trace(go.Scatter(
                    x=batch_data['epoch'],
                    y=batch_data['val_loss'],
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color=COLORS['secondary'], width=3),
                    marker=dict(size=6)
                ))
                
                fig_loss.update_layout(
                    title="Training loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    height=400
                )
                
                st.plotly_chart(fig_loss, width="stretch")
            
            with col2:
                # AUC curves
                fig_auc = go.Figure()
                
                fig_auc.add_trace(go.Scatter(
                    x=batch_data['epoch'],
                    y=batch_data['train_auc'],
                    mode='lines+markers',
                    name='Train AUC',
                    line=dict(color=COLORS['primary'], width=3),
                    marker=dict(size=6)
                ))
                
                fig_auc.add_trace(go.Scatter(
                    x=batch_data['epoch'],
                    y=batch_data['val_auc'],
                    mode='lines+markers',
                    name='Validation AUC',
                    line=dict(color=COLORS['secondary'], width=3),
                    marker=dict(size=6)
                ))
                
                fig_auc.update_layout(
                    title="Training AUC",
                    xaxis_title="Epoch",
                    yaxis_title="AUC",
                    height=400
                )
                
                st.plotly_chart(fig_auc, width="stretch")
            
            # Batch summary
            st.markdown("### Batch summary")
            final_epoch = batch_data[batch_data['epoch'] == batch_data['epoch'].max()].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final train loss", f"{final_epoch['train_loss']:.4f}")
            with col2:
                st.metric("Final val loss", f"{final_epoch['val_loss']:.4f}")
            with col3:
                st.metric("Final train AUC", f"{final_epoch['train_auc']:.4f}")
            with col4:
                st.metric("Final val AUC", f"{final_epoch['val_auc']:.4f}")
            
            # Show transfer learning info
            try:
                batch_num = int(selected_id.split('_')[-1]) if '_' in selected_id else 0
                if batch_num > 10:
                    st.info(f"Transfer learning: This DNN was initialized from the previous batch ({batch_num-10} models) and expanded to {batch_num} inputs.")
            except ValueError:
                # Handle non-numeric ensemble IDs
                pass
        
        # Input dimension growth visualization
        if len(ensemble_ids) > 1:
            st.markdown("### DNN architecture growth")
            
            # Extract batch numbers from ensemble IDs
            batch_numbers = []
            for eid in ensemble_ids:
                try:
                    if '_' in eid:
                        batch_numbers.append(int(eid.split('_')[-1]))
                except ValueError:
                    # Skip non-numeric IDs
                    continue
            
            if batch_numbers:
                fig_growth = px.line(
                    x=range(len(batch_numbers)),
                    y=batch_numbers,
                    title="DNN input dimension growth (transfer learning)",
                    labels={'x': 'Training round', 'y': 'Number of inputs (ensemble size)'},
                    markers=True
                )
                
                st.plotly_chart(fig_growth, width="stretch")
    else:
        st.info("No Stage 2 training data yet. DNN training starts at 10 accepted models.")

# ====================
# MEMORY USAGE PAGE
# ====================
elif page == "Memory usage":
    st.subheader("Memory usage analysis")
    
    if 'training_memory_mb' in ensemble_df.columns and not ensemble_df['training_memory_mb'].isna().all():
        # Training memory over iterations
        fig_train_mem = go.Figure()
        
        # Add memory trace
        fig_train_mem.add_trace(go.Scatter(
            x=ensemble_df['iteration_num'],
            y=ensemble_df['training_memory_mb'],
            mode='markers',
            marker=dict(
                size=6,
                color=ensemble_df['accepted'].map({1: 'green', 0: 'red'}),
                symbol=ensemble_df['accepted'].map({1: 'circle', 0: 'x'})
            ),
            name='Training Memory',
            text=ensemble_df.apply(lambda row: f"Iter {row['iteration_num']}: {row['training_memory_mb']:.1f} MB ({row['classifier_type']})", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add average line
        avg_mem = ensemble_df['training_memory_mb'].mean()
        fig_train_mem.add_trace(go.Scatter(
            x=ensemble_df['iteration_num'],
            y=[avg_mem] * len(ensemble_df),
            mode='lines',
            line=dict(dash='dash', color=COLORS['primary']),
            name=f'Average: {avg_mem:.1f} MB',
            hoverinfo='skip',
            showlegend=True
        ))
        
        fig_train_mem.update_layout(
            title=dict(text="Pipeline training memory usage", y=0.98),
            xaxis_title="Iteration number",
            yaxis_title="Memory usage (MB)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.08,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100),
            height=400
        )
        
        st.plotly_chart(fig_train_mem, width="stretch")
        
        # Memory by classifier type
        # Create box plot for memory distribution by classifier type
        fig_classifier_mem = go.Figure()
        
        # Calculate median for each classifier type and sort
        classifier_medians = ensemble_df.groupby('classifier_type')['training_memory_mb'].median().sort_values(ascending=False)
        
        # Add box plots in sorted order (highest median first)
        for clf_type in classifier_medians.index:
            clf_data = ensemble_df[ensemble_df['classifier_type'] == clf_type]['training_memory_mb']
            fig_classifier_mem.add_trace(go.Box(
                y=clf_data,
                name=clf_type,
                fillcolor=COLORS['primary'],
                line=dict(color=COLORS['quaternary'], width=2)
            ))
        
        fig_classifier_mem.update_layout(
            title="Memory usage distribution by classifier type",
            xaxis_title="Classifier type",
            yaxis_title="Memory usage (MB)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_classifier_mem, width="stretch")
        
        # Stage 2 memory (if available)
        if 'stage2_memory_mb' in ensemble_df.columns:
            stage2_mem_df = ensemble_df[ensemble_df['stage2_memory_mb'].notna()].copy()
            
            if not stage2_mem_df.empty:
                fig_stage2_mem = go.Figure()
                
                fig_stage2_mem.add_trace(go.Bar(
                    x=stage2_mem_df['iteration_num'],
                    y=stage2_mem_df['stage2_memory_mb'],
                    name='Stage 2 Memory',
                    text=stage2_mem_df['stage2_memory_mb'].apply(lambda x: f"{x:.1f} MB"),
                    textposition='outside',
                    marker_color=COLORS['tertiary']
                ))
                
                fig_stage2_mem.update_layout(
                    title="Stage 2 DNN training memory usage",
                    xaxis_title="Iteration number (DNN retraining events)",
                    yaxis_title="Memory usage (MB)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_stage2_mem, width="stretch")
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Stage 2 Memory", f"{stage2_mem_df['stage2_memory_mb'].min():.1f} MB")
                with col2:
                    st.metric("Avg Stage 2 Memory", f"{stage2_mem_df['stage2_memory_mb'].mean():.1f} MB")
                with col3:
                    st.metric("Max Stage 2 Memory", f"{stage2_mem_df['stage2_memory_mb'].max():.1f} MB")
    else:
        st.info("Memory tracking data not available. This feature requires running with the updated training code.")

# ====================
# TIMING PAGE
# ====================
elif page == "Timing":
    st.header("Timing analysis")
    
    if ensemble_df.empty or 'training_time_sec' not in ensemble_df.columns:
        st.info("No timing data available yet. Start training to collect metrics.")
    else:
        # Timeout statistics (if available)
        if 'timeout' in ensemble_df.columns:
            timeout_df = ensemble_df[ensemble_df['timeout'] == 1]
            total_attempts = len(ensemble_df)
            timeout_count = len(timeout_df)
            timeout_pct = (timeout_count / total_attempts * 100) if total_attempts > 0 else 0
            
            # Timeout summary
            st.subheader("Timeout summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total timeouts", timeout_count)
            with col2:
                st.metric("Timeout rate", f"{timeout_pct:.1f}%")
            with col3:
                successful_count = total_attempts - timeout_count
                st.metric("Successful runs", successful_count)
            
            # Timeout by classifier type
            if not timeout_df.empty and 'classifier_type' in timeout_df.columns:
                st.markdown("### Timeouts by classifier type")
                
                # Calculate timeout counts per classifier
                # Exclude special types (dnn_retrain, exception) from analysis
                timeout_df_real = timeout_df[~timeout_df['classifier_type'].isin(['dnn_retrain', 'exception'])]
                ensemble_df_real = ensemble_df[~ensemble_df['classifier_type'].isin(['dnn_retrain', 'exception', 'timeout'])]
                
                if not timeout_df_real.empty and not ensemble_df_real.empty:
                    classifier_timeouts = timeout_df_real['classifier_type'].value_counts()
                    classifier_totals = ensemble_df_real['classifier_type'].value_counts()
                    classifier_timeout_rates = (classifier_timeouts / classifier_totals * 100).fillna(0)
                    
                    # Create combined dataframe
                    timeout_summary = pd.DataFrame({
                        'Total Attempts': classifier_totals,
                        'Timeouts': classifier_timeouts.reindex(classifier_totals.index, fill_value=0),
                        'Timeout Rate (%)': classifier_timeout_rates.reindex(classifier_totals.index, fill_value=0)
                    }).sort_values('Timeout Rate (%)', ascending=False)
                    
                    # Bar chart - Timeout Rate
                    timeout_summary_sorted = timeout_summary.sort_values('Timeout Rate (%)', ascending=True)
                    
                    fig_timeout_classifier = px.bar(
                        x=timeout_summary_sorted['Timeout Rate (%)'],
                        y=timeout_summary_sorted.index,
                        orientation='h',
                        title='Timeout rate by classifier type',
                        labels={'x': 'Timeout rate (%)', 'y': 'Classifier type'},
                        color_discrete_sequence=[COLORS['secondary']]
                    )
                    st.plotly_chart(fig_timeout_classifier, width="stretch")
                else:
                    st.info("No timeout data available for classifier type breakdown.")
            
            st.markdown("---")
        
        # Filter for rows with timing data
        df_time = ensemble_df[ensemble_df['training_time_sec'].notna()].copy()
        
        if df_time.empty:
            st.info("No timing data available yet.")
        else:
            # Header metrics
            st.subheader("Training time summary")
            col1, col2 = st.columns(2)
            
            with col1:
                avg_training_time = df_time['training_time_sec'].mean()
                st.metric("Avg batch training time", f"{avg_training_time:.1f}s")
            
            with col2:
                total_training_time = df_time['training_time_sec'].sum()
                st.metric("Total batch training time", f"{total_training_time/60:.1f} min")
            
            # Chart 1: Training time distribution
            fig_time_hist = px.histogram(
                df_time,
                x='training_time_sec',
                nbins=30,
                title='Training time distribution',
                labels={'training_time_sec': 'Time (seconds)', 'count': 'Frequency'}
            )
            fig_time_hist.update_traces(marker_color=COLORS['quaternary'])
            st.plotly_chart(fig_time_hist, width="stretch")
            
            # Boxplot: Runtime distribution by classifier type
            if 'classifier_type' in df_time.columns:
                # Calculate median for each classifier type and sort
                time_medians = df_time.groupby('classifier_type')['training_time_sec'].median().sort_values(ascending=False)
                
                fig_time_box = go.Figure()
                
                # Add box plots in sorted order (highest median first)
                for clf_type in time_medians.index:
                    clf_data = df_time[df_time['classifier_type'] == clf_type]['training_time_sec']
                    fig_time_box.add_trace(go.Box(
                        y=clf_data,
                        name=clf_type,
                        fillcolor=COLORS['primary'],
                        line=dict(color=COLORS['quaternary'], width=2)
                    ))
                
                fig_time_box.update_layout(
                    title="Training time distribution by classifier type",
                    xaxis_title="Classifier type",
                    yaxis_title="Time (seconds)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_time_box, width="stretch")
            
            # Chart 3: Stage 2 DNN training time (if available)
            if 'stage2_time_sec' in df_time.columns:
                df_stage2_time = df_time[df_time['stage2_time_sec'].notna()].copy()
                if not df_stage2_time.empty:
                    st.subheader("Stage 2 DNN training time")
                    fig_stage2_time = px.line(
                        df_stage2_time,
                        x='iteration_num',
                        y='stage2_time_sec',
                        title='Time for stage 2 DNN training',
                        labels={'iteration_num': 'Iteration', 'stage2_time_sec': 'Time (seconds)'}
                    )
                    fig_stage2_time.update_traces(mode='lines+markers', marker_color=COLORS['tertiary'], line=dict(width=3))
                    st.plotly_chart(fig_stage2_time, width="stretch")

