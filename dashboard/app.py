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

# Hardcoded database path
DB_PATH = '/workspaces/diabetes-prediction/data/ensemble_training.db'

# Page configuration
st.set_page_config(
    page_title="Ensemble Training Monitor",
    page_icon="📊",
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
st.title("🎯 Ensemble Hill Climbing Training Monitor")
st.markdown("---")

# Check database existence
if not check_database_exists():
    st.error(f"❌ Database not found at: `{DB_PATH}`")
    st.info("Start the training notebook to create the database.")
    st.stop()

# Load data
ensemble_df = get_ensemble_data()
stage2_df = get_stage2_data()

# Check if data exists
if ensemble_df.empty:
    st.warning("⚠️ No training data found in database. Waiting for first iteration...")
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
    st.metric("Aggregation", stats['aggregation_method'])
    
with col6:
    if stats['last_update']:
        last_update = datetime.fromisoformat(stats['last_update'])
        time_diff = datetime.now() - last_update
        minutes_ago = int(time_diff.total_seconds() / 60)
        st.metric("Last Update", f"{minutes_ago}m ago")
    else:
        st.metric("Last Update", "N/A")

st.markdown("---")

# Acceptance rate metric
accept_rate = (stats['accepted_count'] / stats['total_iterations'] * 100) if stats['total_iterations'] > 0 else 0
st.progress(accept_rate / 100, text=f"Acceptance Rate: {accept_rate:.1f}% ({stats['accepted_count']} accepted / {stats['rejected_count']} rejected)")

st.markdown("---")

# Tabbed interface
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance", "🔀 Diversity", "🧩 Composition", "🧠 Stage 2 DNN"])

# ====================
# PERFORMANCE TAB
# ====================
with tab1:
    st.subheader("Performance Metrics Over Time")
    
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
            title="Stage 2 Validation AUC (Ensemble Performance)",
            xaxis_title="Iteration Number",
            yaxis_title="Stage 2 Validation AUC",
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
            title="Stage 1 Validation AUC (Individual Model Performance)",
            xaxis_title="Iteration Number",
            yaxis_title="Stage 1 Validation AUC",
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
            title="Ensemble Size Growth",
            labels={'iteration_num': 'Iteration Number', 'cumulative_ensemble_size': 'Ensemble Size'}
        )
        fig_size.update_traces(mode='lines+markers')
        
        st.plotly_chart(fig_size, width="stretch")

# ====================
# DIVERSITY TAB
# ====================
with tab2:
    st.subheader("Diversity Metrics")
    
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
            title="Diversity Score Over Iterations",
            xaxis_title="Iteration Number",
            yaxis_title="Diversity Score",
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
            title="Diversity vs Stage 1 Validation AUC",
            labels={'stage1_val_auc': 'Stage 1 Validation AUC', 'diversity_score': 'Diversity Score'},
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
                title="Classifier Types in Ensemble",
                labels={'x': 'Classifier Type', 'y': 'Count'}
            )
            
            st.plotly_chart(fig_classifiers, width="stretch")

# ====================
# COMPOSITION TAB
# ====================
with tab3:
    st.subheader("Ensemble Composition")
    
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
                    title="Transformer Usage Frequency in Accepted Models",
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
                title="Classifier Type Distribution"
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
# STAGE 2 DNN TAB
# ====================
with tab4:
    st.subheader("Stage 2 DNN Training Progress")
    
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
                    title="Training Loss",
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
            st.markdown("### Batch Summary")
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
                    st.info(f"ℹ️ Transfer Learning: This DNN was initialized from the previous batch ({batch_num-10} models) and expanded to {batch_num} inputs.")
            except ValueError:
                # Handle non-numeric ensemble IDs
                pass
        
        # Input dimension growth visualization
        if len(ensemble_ids) > 1:
            st.markdown("### DNN Architecture Growth")
            
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
                    title="DNN Input Dimension Growth (Transfer Learning)",
                    labels={'x': 'Training Round', 'y': 'Number of Inputs (Ensemble Size)'},
                    markers=True
                )
                
                st.plotly_chart(fig_growth, width="stretch")
    else:
        st.info("⏳ No Stage 2 training data yet. DNN training starts at 10 accepted models.")

# ====================
# SIDEBAR
# ====================
st.sidebar.title("⚙️ Dashboard Controls")

# Auto-refresh status
st.sidebar.info("🔄 Auto-refresh: Every 60 seconds")

# CSV Export
st.sidebar.markdown("### 📥 Export Data")

if st.sidebar.button("Download Ensemble Log CSV"):
    csv = ensemble_df.to_csv(index=False)
    st.sidebar.download_button(
        label="💾 Download ensemble_log.csv",
        data=csv,
        file_name=f"ensemble_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if not stage2_df.empty:
    if st.sidebar.button("Download Stage 2 Log CSV"):
        csv = stage2_df.to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Download stage2_log.csv",
            data=csv,
            file_name=f"stage2_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Database Reset
st.sidebar.markdown("### 🗑️ Database Reset")
st.sidebar.warning("⚠️ **DESTRUCTIVE OPERATION**")
st.sidebar.markdown("This will permanently delete all training data!")

reset_confirmation = st.sidebar.text_input(
    "Type 'DELETE DATABASE' to enable reset:",
    key="reset_confirm"
)

if reset_confirmation == "DELETE DATABASE":
    if st.sidebar.button("🔴 Confirm Reset Database", type="primary"):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS ensemble_log")
            cursor.execute("DROP TABLE IF EXISTS stage2_log")
            conn.commit()
            conn.close()
            st.sidebar.success("✅ Database reset complete. Restart training to reinitialize.")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Reset failed: {e}")
else:
    st.sidebar.button("🔴 Confirm Reset Database", disabled=True)

# Database info
st.sidebar.markdown("### 📊 Database Info")
st.sidebar.text(f"Path: {DB_PATH}")
if check_database_exists():
    db_size = Path(DB_PATH).stat().st_size / (1024 * 1024)  # MB
    st.sidebar.text(f"Size: {db_size:.2f} MB")
