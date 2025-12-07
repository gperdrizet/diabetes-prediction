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
            'aggregation_method': 'N/A'
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
    
    return {
        'total_iterations': len(df),
        'accepted_count': len(accepted),
        'rejected_count': len(df[df['accepted'] == 0]),
        'ensemble_size': ensemble_size,
        'best_stage2_auc': df['stage2_val_auc'].max() if 'stage2_val_auc' in df.columns else 0.0,
        'current_temp': df['temperature'].iloc[-1] if 'temperature' in df.columns else 0.0,
        'last_update': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None,
        'aggregation_method': agg_method
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
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Iterations", stats['total_iterations'])
    
with col2:
    st.metric("Ensemble Size", stats['ensemble_size'])
    
with col3:
    st.metric("Best Stage 2 AUC", f"{stats['best_stage2_auc']:.4f}")
    
with col4:
    st.metric("Temperature", f"{stats['current_temp']:.4f}")
    
with col5:
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

st.markdown("---")

# ====================
# SIDEBAR
# ====================
st.sidebar.title("Navigation")

# Page selection
page = st.sidebar.radio(
    "Select page",
    ["Performance", "Diversity", "Composition", "Stage 2 DNN", "Memory Usage", "Timing"],
    label_visibility="collapsed"
)

# ====================
# PERFORMANCE PAGE
# ====================
if page == "Performance":
    st.subheader("Performance metrics over time")
    
    if not ensemble_df.empty:
        # Stage 2 validation AUC over iterations
        fig_stage2 = go.Figure()
        
        # Add all points
        fig_stage2.add_trace(go.Scatter(
            x=ensemble_df['iteration_num'],
            y=ensemble_df['stage2_val_auc'],
            mode='markers',
            marker=dict(
                size=6,
                color=ensemble_df['accepted'].map({1: 'green', 0: 'red'}),
                symbol=ensemble_df['accepted'].map({1: 'circle', 0: 'x'})
            ),
            name='All Iterations',
            text=ensemble_df.apply(lambda row: f"Iter {row['iteration_num']}: {row['stage2_val_auc']:.4f} ({'Accepted' if row['accepted'] else 'Rejected'})", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Add batch boundaries (every 10 accepted models)
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1]
        batch_iterations = accepted_df[accepted_df.index % 10 == 9]['iteration_num'].values
        
        for batch_iter in batch_iterations:
            fig_stage2.add_vline(
                x=batch_iter, 
                line_dash="dash", 
                line_color="blue",
                opacity=0.3,
                annotation_text=f"Batch {int((accepted_df[accepted_df['iteration_num'] <= batch_iter].shape[0]) / 10 * 10)}",
                annotation_position="top"
            )
        
        fig_stage2.update_layout(
            title="Stage 2 validation AUC (ensemble performance)",
            xaxis_title="Iteration number",
            yaxis_title="Stage 2 validation AUC",
            hovermode='closest',
            height=400
        )
        
        st.plotly_chart(fig_stage2, width="stretch")
        
        # Stage 1 validation AUC over iterations
        fig_stage1 = go.Figure()
        
        fig_stage1.add_trace(go.Scatter(
            x=ensemble_df['iteration_num'],
            y=ensemble_df['stage1_val_auc'],
            mode='markers',
            marker=dict(
                size=6,
                color=ensemble_df['accepted'].map({1: 'green', 0: 'red'}),
                symbol=ensemble_df['accepted'].map({1: 'circle', 0: 'x'})
            ),
            name='All Iterations',
            text=ensemble_df.apply(lambda row: f"Iter {row['iteration_num']}: {row['stage1_val_auc']:.4f} ({'Accepted' if row['accepted'] else 'Rejected'})", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig_stage1.update_layout(
            title="Stage 1 validation AUC (individual model performance)",
            xaxis_title="Iteration number",
            yaxis_title="Stage 1 validation AUC",
            hovermode='closest',
            height=400
        )
        
        st.plotly_chart(fig_stage1, width="stretch")
        
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
        
        fig_div.add_trace(go.Scatter(
            x=ensemble_df['iteration_num'],
            y=ensemble_df['diversity_score'],
            mode='markers',
            marker=dict(
                size=6,
                color=ensemble_df['accepted'].map({1: 'green', 0: 'red'}),
                symbol=ensemble_df['accepted'].map({1: 'circle', 0: 'x'})
            ),
            text=ensemble_df.apply(lambda row: f"Iter {row['iteration_num']}: {row['diversity_score']:.4f}", axis=1),
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig_div.update_layout(
            title="Diversity score over iterations",
            xaxis_title="Iteration number",
            yaxis_title="Diversity score",
            hovermode='closest',
            height=400
        )
        
        st.plotly_chart(fig_div, width="stretch")
        
        # Diversity vs Stage 1 AUC scatter
        fig_scatter = px.scatter(
            ensemble_df,
            x='stage1_val_auc',
            y='diversity_score',
            color='accepted',
            color_discrete_map={1: 'green', 0: 'red'},
            title="Diversity vs stage 1 validation AUC",
            labels={'stage1_val_auc': 'Stage 1 validation AUC', 'diversity_score': 'Diversity score'},
            hover_data=['iteration_num']
        )
        
        st.plotly_chart(fig_scatter, width="stretch")
        
        # Classifier type distribution in ensemble
        if 'classifier_type' in ensemble_df.columns and not ensemble_df['classifier_type'].isna().all():
            accepted_df = ensemble_df[ensemble_df['accepted'] == 1]
            classifier_counts = accepted_df['classifier_type'].value_counts()
            
            fig_classifiers = px.bar(
                x=classifier_counts.index,
                y=classifier_counts.values,
                title="Classifier types in ensemble",
                labels={'x': 'Classifier type', 'y': 'Count'}
            )
            
            st.plotly_chart(fig_classifiers, width="stretch")

# ====================
# COMPOSITION PAGE
# ====================
elif page == "Composition":
    st.subheader("Ensemble composition")
    
    if not ensemble_df.empty:
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1]
        
        # Transformer usage frequency
        if 'transformers_used' in accepted_df.columns:
            # Parse comma-separated transformers
            all_transformers = []
            for trans_str in accepted_df['transformers_used'].dropna():
                all_transformers.extend(trans_str.split(','))
            
            if all_transformers:
                transformer_counts = pd.Series(all_transformers).value_counts()
                
                fig_trans = px.bar(
                    x=transformer_counts.index,
                    y=transformer_counts.values,
                    title="Transformer usage frequency in accepted models",
                    labels={'x': 'Transformer', 'y': 'Count'}
                )
                fig_trans.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig_trans, width="stretch")
        
        # Classifier type distribution
        # Note: classifier_type column may not exist in old database schema
        if 'classifier_type' in accepted_df.columns and not accepted_df['classifier_type'].isna().all():
            classifier_counts = accepted_df['classifier_type'].value_counts()
            
            fig_pie = px.pie(
                values=classifier_counts.values,
                names=classifier_counts.index,
                title="Classifier type distribution"
            )
            
            st.plotly_chart(fig_pie, width="stretch")
        
        # PCA usage statistics
        # Note: use_pca and pca_components columns may not exist in old database schema
        if 'use_pca' in accepted_df.columns:
            pca_counts = accepted_df['use_pca'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Models with PCA", pca_counts.get(1, 0))
                st.metric("Models without PCA", pca_counts.get(0, 0))
            
            with col2:
                if 1 in pca_counts.index:
                    pca_models = accepted_df[accepted_df['use_pca'] == 1]
                    if 'pca_components' in pca_models.columns and not pca_models['pca_components'].isna().all():
                        avg_components = pca_models['pca_components'].mean()
                        st.metric("Avg PCA Components", f"{avg_components:.1f}")

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
        
        # Memory efficiency analysis
        st.markdown("### Memory efficiency")
        
        # Calculate memory per AUC point gained
        accepted_df = ensemble_df[ensemble_df['accepted'] == 1].copy()
        if len(accepted_df) > 1:
            accepted_df['auc_improvement'] = accepted_df['stage2_val_auc'].diff()
            accepted_df['memory_efficiency'] = accepted_df['training_memory_mb'] / (accepted_df['auc_improvement'] * 10000)
            
            # Remove inf and nan values
            efficiency_df = accepted_df[accepted_df['memory_efficiency'].notna() & ~np.isinf(accepted_df['memory_efficiency'])]
            
            if not efficiency_df.empty:
                fig_efficiency = go.Figure()
                
                fig_efficiency.add_trace(go.Scatter(
                    x=efficiency_df['iteration_num'],
                    y=efficiency_df['memory_efficiency'],
                    mode='lines+markers',
                    name='Memory Efficiency',
                    line=dict(color='orange'),
                    text=efficiency_df.apply(lambda row: f"Iter {row['iteration_num']}: {row['memory_efficiency']:.2f} MB per 0.01% AUC", axis=1),
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                fig_efficiency.update_layout(
                    title="Memory efficiency (lower is better)",
                    xaxis_title="Iteration number",
                    yaxis_title="MB per 0.01% AUC improvement",
                    height=400
                )
                
                st.plotly_chart(fig_efficiency, width="stretch")
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
                x='iteration',
                y='training_time_sec',
                title='Parallel training time per iteration',
                labels={'iteration': 'Iteration', 'training_time_sec': 'Time (seconds)'}
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
            
            # Chart 3: Stage 2 DNN training time (if available)
            if 'stage2_time_sec' in df_time.columns:
                df_stage2_time = df_time[df_time['stage2_time_sec'].notna()].copy()
                if not df_stage2_time.empty:
                    st.subheader("Stage 2 DNN training time")
                    fig_stage2_time = px.line(
                        df_stage2_time,
                        x='iteration',
                        y='stage2_time_sec',
                        title='Time for stage 2 DNN training',
                        labels={'iteration': 'Iteration', 'stage2_time_sec': 'Time (seconds)'}
                    )
                    fig_stage2_time.update_traces(mode='lines+markers', marker_color='purple')
                    st.plotly_chart(fig_stage2_time, use_container_width=True)
            
            # Chart 4: Time efficiency (time per AUC improvement)
            st.subheader("Time efficiency")
            if len(df_time) > 1:
                df_time['auc_improvement'] = df_time['stage2_auc_roc'].diff()
                df_time['time_per_improvement'] = df_time['training_time_sec'] / df_time['auc_improvement'].abs()
                df_time_eff = df_time[df_time['time_per_improvement'].notna() & 
                                      ~df_time['time_per_improvement'].isin([float('inf'), -float('inf')])].copy()
                
                if not df_time_eff.empty:
                    fig_time_eff = px.scatter(
                        df_time_eff,
                        x='iteration',
                        y='time_per_improvement',
                        color='classifier_type',
                        title='Training time per AUC improvement (lower is better)',
                        labels={'iteration': 'Iteration', 'time_per_improvement': 'Seconds per 0.01% AUC'},
                        hover_data=['training_time_sec', 'stage2_auc_roc']
                    )
                    st.plotly_chart(fig_time_eff, use_container_width=True)
                else:
                    st.info("Not enough data yet to calculate time efficiency.")

