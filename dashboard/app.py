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

# Database path relative to dashboard directory
DB_PATH = Path(__file__).parent.parent / 'data' / 'ensemble_training.db'

# Page configuration
st.set_page_config(
    page_title="Ensemble Training Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

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
        st.metric("Avg Training Memory", f"{avg_training_mem:.1f} MB")
    
    with col2:
        max_training_mem = ensemble_df['training_memory_mb'].max()
        st.metric("Peak Training Memory", f"{max_training_mem:.1f} MB")
    
    with col3:
        if 'stage2_memory_mb' in ensemble_df.columns:
            stage2_mem = ensemble_df['stage2_memory_mb'].dropna()
            if not stage2_mem.empty:
                avg_stage2_mem = stage2_mem.mean()
                st.metric("Avg Stage 2 Memory", f"{avg_stage2_mem:.1f} MB")
            else:
                st.metric("Avg Stage 2 Memory", "N/A")
        else:
            st.metric("Avg Stage 2 Memory", "N/A")
    
    st.markdown("---")

# Acceptance rate metric
accept_rate = (stats['accepted_count'] / stats['total_iterations'] * 100) if stats['total_iterations'] > 0 else 0
st.progress(accept_rate / 100, text=f"Acceptance Rate: {accept_rate:.1f}% ({stats['accepted_count']} accepted / {stats['rejected_count']} rejected)")

# Timeout rate metric (if data available)
if 'timeout' in ensemble_df.columns:
    timeout_count = ensemble_df['timeout'].sum()
    timeout_rate = (timeout_count / stats['total_iterations'] * 100) if stats['total_iterations'] > 0 else 0
    st.progress(timeout_rate / 100, text=f"Timeout Rate: {timeout_rate:.1f}% ({int(timeout_count)} timeouts / {stats['total_iterations']} iterations)")

st.markdown("---")

# ====================
# SIDEBAR
# ====================
st.sidebar.title("Navigation")

# Page selection
page = st.sidebar.radio(
    "Select page",
    ["Performance", "Diversity", "Stage 2 DNN", "Memory Usage", "Timing"],
    label_visibility="collapsed"
)

# ====================
# PERFORMANCE PAGE
# ====================
if page == "Performance":
    st.subheader("Performance metrics over time")
    
    if not ensemble_df.empty:
        # Combined Stage 1 and Stage 2 validation AUC plot
        fig_combined = go.Figure()
        
        # Calculate rolling statistics for Stage 1 AUC (window of 10 iterations)
        window_size = 10
        ensemble_df_sorted = ensemble_df.sort_values('iteration_num')
        stage1_rolling_mean = ensemble_df_sorted['stage1_val_auc'].rolling(window=window_size, min_periods=1).mean()
        stage1_rolling_std = ensemble_df_sorted['stage1_val_auc'].rolling(window=window_size, min_periods=1).std()
        
        # Add Stage 1 shaded region (mean ± std)
        fig_combined.add_trace(go.Scatter(
            x=ensemble_df_sorted['iteration_num'],
            y=stage1_rolling_mean + stage1_rolling_std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig_combined.add_trace(go.Scatter(
            x=ensemble_df_sorted['iteration_num'],
            y=stage1_rolling_mean - stage1_rolling_std,
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 114, 196, 0.2)',
            fill='tonexty',
            name='Stage 1 ±1 StdDev',
            hoverinfo='skip'
        ))
        
        # Add Stage 1 mean line
        fig_combined.add_trace(go.Scatter(
            x=ensemble_df_sorted['iteration_num'],
            y=stage1_rolling_mean,
            mode='lines',
            line=dict(color='rgba(68, 114, 196, 0.8)', width=2),
            name='Stage 1 Mean (Individual Models)',
            hovertemplate='Iter %{x}<br>Stage 1 Mean: %{y:.4f}<extra></extra>'
        ))
        
        # Add Stage 2 line (ensemble performance)
        # Use only accepted models for cleaner line
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1].sort_values('iteration_num')
        fig_combined.add_trace(go.Scatter(
            x=accepted_df['iteration_num'],
            y=accepted_df['stage2_val_auc'],
            mode='lines+markers',
            line=dict(color='rgb(34, 139, 34)', width=3),
            marker=dict(size=6, color='rgb(34, 139, 34)'),
            name='Stage 2 (Ensemble)',
            hovertemplate='Iter %{x}<br>Stage 2 AUC: %{y:.4f}<extra></extra>'
        ))
        
        # Add batch boundaries (every 10 accepted models)
        batch_iterations = accepted_df[accepted_df.index % 10 == 9]['iteration_num'].values
        
        for batch_iter in batch_iterations:
            fig_combined.add_vline(
                x=batch_iter, 
                line_dash="dash", 
                line_color="gray",
                opacity=0.3,
                annotation_text=f"DNN Retrain",
                annotation_position="top"
            )
        
        fig_combined.update_layout(
            title="Validation AUC Over Time: Stage 1 (Individual Models) vs Stage 2 (Ensemble)",
            xaxis_title="Iteration Number",
            yaxis_title="Validation AUC",
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)
        
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
        fig_size.update_traces(mode='lines+markers')
        
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
            marker=dict(size=6, color='green', symbol='circle'),
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
            marker=dict(size=6, color='red', symbol='x'),
            name='Rejected',
            text=rejected_df.apply(lambda row: f"Iter {row['iteration_num']}: {row['diversity_score']:.4f}", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig_div.update_layout(
            title="Diversity score over iterations",
            xaxis_title="Iteration number",
            yaxis_title="Diversity score",
            hovermode='closest',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_div, width="stretch")
        
        # Diversity vs Stage 2 AUC scatter
        fig_scatter = px.scatter(
            ensemble_df,
            x='diversity_score',
            y='stage2_val_auc',
            title="Diversity vs Stage 2 Validation AUC (Ensemble Performance)",
            labels={'diversity_score': 'Diversity score', 'stage2_val_auc': 'Stage 2 validation AUC'},
            hover_data=['iteration_num', 'accepted']
        )
        
        st.plotly_chart(fig_scatter, width="stretch")
        
        # Composition Analysis - Accepted Models
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1]
        
        # Transformer usage frequency
        if 'transformers_used' in accepted_df.columns:
            # Parse comma-separated transformers
            all_transformers = []
            for trans_str in accepted_df['transformers_used'].dropna():
                all_transformers.extend(trans_str.split(','))
            
            if all_transformers:
                transformer_counts = pd.Series(all_transformers).value_counts().sort_values(ascending=True)
                
                fig_trans = px.bar(
                    x=transformer_counts.values,
                    y=transformer_counts.index,
                    orientation='h',
                    title="Transformer usage frequency in accepted models",
                    labels={'x': 'Count', 'y': 'Transformer'}
                )
                
                st.plotly_chart(fig_trans, width="stretch")
        
        # Classifier type distribution in ensemble
        if 'classifier_type' in ensemble_df.columns and not ensemble_df['classifier_type'].isna().all():
            classifier_counts = accepted_df['classifier_type'].value_counts().sort_values(ascending=True)
            
            fig_classifiers = px.bar(
                x=classifier_counts.values,
                y=classifier_counts.index,
                orientation='h',
                title="Classifier types in ensemble",
                labels={'x': 'Count', 'y': 'Classifier type'}
            )
            
            st.plotly_chart(fig_classifiers, width="stretch")
        
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
                title="Dimensionality reduction technique usage in accepted models",
                labels={'x': 'Count', 'y': 'Technique'}
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
                "Select Training Batch",
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
                    line=dict(color='blue')
                ))
                
                fig_loss.add_trace(go.Scatter(
                    x=batch_data['epoch'],
                    y=batch_data['val_loss'],
                    mode='lines+markers',
                    name='Validation Loss',
                    line=dict(color='red')
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
                    line=dict(color='blue')
                ))
                
                fig_auc.add_trace(go.Scatter(
                    x=batch_data['epoch'],
                    y=batch_data['val_auc'],
                    mode='lines+markers',
                    name='Validation AUC',
                    line=dict(color='red')
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
                st.metric("Final Train Loss", f"{final_epoch['train_loss']:.4f}")
            with col2:
                st.metric("Final Val Loss", f"{final_epoch['val_loss']:.4f}")
            with col3:
                st.metric("Final Train AUC", f"{final_epoch['train_auc']:.4f}")
            with col4:
                st.metric("Final Val AUC", f"{final_epoch['val_auc']:.4f}")
            
            # Show transfer learning info
            try:
                batch_num = int(selected_id.split('_')[-1]) if '_' in selected_id else 0
                if batch_num > 10:
                    st.info(f"Transfer Learning: This DNN was initialized from the previous batch ({batch_num-10} models) and expanded to {batch_num} inputs.")
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
elif page == "Memory Usage":
    st.subheader("Memory usage analysis")
    
    if 'training_memory_mb' in ensemble_df.columns and not ensemble_df['training_memory_mb'].isna().all():
        # Training memory over iterations
        st.markdown("### Pipeline training memory")
        
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
        fig_train_mem.add_hline(
            y=avg_mem,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Average: {avg_mem:.1f} MB",
            annotation_position="right"
        )
        
        fig_train_mem.update_layout(
            title="Pipeline training memory usage",
            xaxis_title="Iteration number",
            yaxis_title="Memory usage (MB)",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig_train_mem, width="stretch")
        
        # Memory by classifier type
        st.markdown("### Memory usage by classifier type")
        
        classifier_memory = ensemble_df.groupby('classifier_type')['training_memory_mb'].agg(['mean', 'max', 'count'])
        classifier_memory = classifier_memory.sort_values('mean', ascending=False)
        
        fig_classifier_mem = go.Figure(data=[
            go.Bar(
                x=classifier_memory.index,
                y=classifier_memory['mean'],
                name='Average Memory',
                error_y=dict(
                    type='data',
                    array=classifier_memory['max'] - classifier_memory['mean'],
                    visible=True
                ),
                text=classifier_memory['count'].apply(lambda x: f"n={x}"),
                textposition='outside'
            )
        ])
        
        fig_classifier_mem.update_layout(
            title="Average memory usage by classifier type",
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
                st.markdown("### Stage 2 DNN training memory")
                
                fig_stage2_mem = go.Figure()
                
                fig_stage2_mem.add_trace(go.Bar(
                    x=stage2_mem_df['iteration_num'],
                    y=stage2_mem_df['stage2_memory_mb'],
                    name='Stage 2 Memory',
                    text=stage2_mem_df['stage2_memory_mb'].apply(lambda x: f"{x:.1f} MB"),
                    textposition='outside',
                    marker_color='purple'
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
                st.metric("Total Timeouts", timeout_count)
            with col2:
                st.metric("Timeout Rate", f"{timeout_pct:.1f}%")
            with col3:
                successful_count = total_attempts - timeout_count
                st.metric("Successful Runs", successful_count)
            
            # Timeout by classifier type
            if not timeout_df.empty and 'classifier_type' in timeout_df.columns:
                st.markdown("### Timeouts by classifier type")
                
                # Calculate timeout counts per classifier (excluding 'timeout' pseudo-type)
                timeout_df_real = timeout_df[timeout_df['classifier_type'] != 'timeout']
                ensemble_df_real = ensemble_df[ensemble_df['classifier_type'] != 'timeout']
                
                classifier_timeouts = timeout_df_real['classifier_type'].value_counts()
                classifier_totals = ensemble_df_real['classifier_type'].value_counts()
                classifier_timeout_rates = (classifier_timeouts / classifier_totals * 100).fillna(0)
                
                # Create combined dataframe
                timeout_summary = pd.DataFrame({
                    'Total Attempts': classifier_totals,
                    'Timeouts': classifier_timeouts.reindex(classifier_totals.index, fill_value=0),
                    'Timeout Rate (%)': classifier_timeout_rates.reindex(classifier_totals.index, fill_value=0)
                }).sort_values('Timeout Rate (%)', ascending=False)
                
                # Display as table
                st.dataframe(timeout_summary, use_container_width=True)
                
                # Bar chart - Timeout Rate
                # Sort by timeout rate for horizontal display
                timeout_summary_sorted = timeout_summary.sort_values('Timeout Rate (%)', ascending=True)
                
                fig_timeout_classifier = px.bar(
                    x=timeout_summary_sorted['Timeout Rate (%)'],
                    y=timeout_summary_sorted.index,
                    orientation='h',
                    title='Timeout rate by classifier type',
                    labels={'x': 'Timeout rate (%)', 'y': 'Classifier type'}
                )
                st.plotly_chart(fig_timeout_classifier, use_container_width=True)
                
                # Bar chart - Timeout Count
                timeout_count_sorted = timeout_summary.sort_values('Timeouts', ascending=True)
                
                fig_timeout_count = px.bar(
                    x=timeout_count_sorted['Timeouts'],
                    y=timeout_count_sorted.index,
                    orientation='h',
                    title='Number of timeouts by classifier type',
                    labels={'x': 'Number of timeouts', 'y': 'Classifier type'}
                )
                st.plotly_chart(fig_timeout_count, use_container_width=True)
            
            st.markdown("---")
        
        # Filter for rows with timing data
        df_time = ensemble_df[ensemble_df['training_time_sec'].notna()].copy()
        
        if df_time.empty:
            st.info("No timing data available yet.")
        else:
            # Header metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_training_time = df_time['training_time_sec'].mean()
                st.metric("Avg Training Time", f"{avg_training_time:.1f}s")
            
            with col2:
                total_training_time = df_time['training_time_sec'].sum()
                st.metric("Total Training Time", f"{total_training_time/60:.1f} min")
            
            with col3:
                if 'stage2_time_sec' in df_time.columns and df_time['stage2_time_sec'].notna().any():
                    avg_stage2_time = df_time[df_time['stage2_time_sec'].notna()]['stage2_time_sec'].mean()
                    st.metric("Avg Stage 2 Time", f"{avg_stage2_time:.1f}s")
                else:
                    st.metric("Avg Stage 2 Time", "N/A")
            
            with col4:
                if 'stage2_time_sec' in df_time.columns and df_time['stage2_time_sec'].notna().any():
                    total_stage2_time = df_time[df_time['stage2_time_sec'].notna()]['stage2_time_sec'].sum()
                    st.metric("Total Stage 2 Time", f"{total_stage2_time/60:.1f} min")
                else:
                    st.metric("Total Stage 2 Time", "N/A")
            
            # Chart 1: Training time over iterations
            st.subheader("Training time over iterations")
            fig_time_iter = px.line(
                df_time,
                x='iteration_num',
                y='training_time_sec',
                title='Parallel training time per iteration',
                labels={'iteration_num': 'Iteration', 'training_time_sec': 'Time (seconds)'}
            )
            fig_time_iter.update_traces(mode='lines+markers', marker_color='green')
            st.plotly_chart(fig_time_iter, use_container_width=True)
            
            # Chart 2: Time by classifier type
            st.subheader("Training time by classifier type")
            time_by_classifier = df_time.groupby('classifier_type').agg({
                'training_time_sec': ['mean', 'max', 'min', 'count']
            }).reset_index()
            time_by_classifier.columns = ['classifier_type', 'avg_time', 'max_time', 'min_time', 'count']
            
            fig_time_classifier = go.Figure()
            fig_time_classifier.add_trace(go.Bar(
                x=time_by_classifier['classifier_type'],
                y=time_by_classifier['avg_time'],
                name='Average Time',
                marker_color='lightgreen',
                error_y=dict(
                    type='data',
                    array=time_by_classifier['max_time'] - time_by_classifier['avg_time'],
                    arrayminus=time_by_classifier['avg_time'] - time_by_classifier['min_time']
                )
            ))
            fig_time_classifier.update_layout(
                title='Training time by classifier type (with min/max range)',
                xaxis_title='Classifier type',
                yaxis_title='Time (seconds)'
            )
            st.plotly_chart(fig_time_classifier, use_container_width=True)
            
            # Boxplot: Runtime distribution by classifier type
            if 'classifier_type' in df_time.columns:
                st.subheader("Runtime distribution by classifier type")
                fig_time_box = px.box(
                    df_time,
                    x='classifier_type',
                    y='training_time_sec',
                    title='Training time distribution by classifier type',
                    labels={'classifier_type': 'Classifier type', 'training_time_sec': 'Time (seconds)'},
                    points='all'  # Show all individual points
                )
                st.plotly_chart(fig_time_box, use_container_width=True)
            
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
                    fig_stage2_time.update_traces(mode='lines+markers', marker_color='purple')
                    st.plotly_chart(fig_stage2_time, use_container_width=True)
            
            # Chart 4: Time efficiency (time per AUC improvement)
            st.subheader("Time efficiency")
            if len(df_time) > 1:
                df_time['auc_improvement'] = df_time['stage2_val_auc'].diff()
                df_time['time_per_improvement'] = df_time['training_time_sec'] / df_time['auc_improvement'].abs()
                df_time_eff = df_time[df_time['time_per_improvement'].notna() & 
                                      ~df_time['time_per_improvement'].isin([float('inf'), -float('inf')])].copy()
                
                if not df_time_eff.empty:
                    fig_time_eff = px.scatter(
                        df_time_eff,
                        x='iteration_num',
                        y='time_per_improvement',
                        title='Training time per AUC improvement (lower is better)',
                        labels={'iteration_num': 'Iteration', 'time_per_improvement': 'Seconds per 0.01% AUC'},
                        hover_data=['classifier_type', 'training_time_sec', 'stage2_val_auc']
                    )
                    st.plotly_chart(fig_time_eff, use_container_width=True)
                else:
                    st.info("Not enough data yet to calculate time efficiency.")

