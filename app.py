import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import json

# Configure page
st.set_page_config(
    page_title="News Dataset Explorer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the news dataset
@st.cache_data
def load_news_dataset():
    """Load the combined news labeled dataset"""
    try:
        with open('combined_news_labeled.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading news dataset: {str(e)}")
        return pd.DataFrame()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_news_dataset()
if 'original_data' not in st.session_state:
    st.session_state.original_data = st.session_state.data.copy() if st.session_state.data is not None else pd.DataFrame()
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = st.session_state.data.copy() if st.session_state.data is not None else pd.DataFrame()



def get_column_stats(df, column):
    """Get statistical information for a column"""
    stats = {}
    
    if df[column].dtype in ['int64', 'float64']:
        stats['type'] = 'Numeric'
        stats['count'] = df[column].count()
        stats['null_count'] = df[column].isnull().sum()
        stats['mean'] = df[column].mean()
        stats['median'] = df[column].median()
        stats['std'] = df[column].std()
        stats['min'] = df[column].min()
        stats['max'] = df[column].max()
    else:
        stats['type'] = 'Categorical'
        stats['count'] = df[column].count()
        stats['null_count'] = df[column].isnull().sum()
        stats['unique_count'] = df[column].nunique()
        stats['most_frequent'] = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else 'N/A'
    
    return stats

def filter_data(df, filters):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    for column, filter_config in filters.items():
        if filter_config['type'] == 'numeric':
            min_val, max_val = filter_config['range']
            filtered_df = filtered_df[
                (filtered_df[column] >= min_val) & 
                (filtered_df[column] <= max_val)
            ]
        elif filter_config['type'] == 'categorical':
            if filter_config['values']:
                filtered_df = filtered_df[filtered_df[column].isin(filter_config['values'])]
        elif filter_config['type'] == 'text':
            if filter_config['search']:
                filtered_df = filtered_df[
                    filtered_df[column].astype(str).str.contains(
                        filter_config['search'], case=False, na=False
                    )
                ]
    
    return filtered_df

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    """Convert dataframe to Excel for download"""
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Data')
    writer.close()
    output.seek(0)
    return output.getvalue()

# Main application
def main():
    st.title("📰 News Dataset Explorer")
    st.markdown("Explore labeled news articles with interactive features!")
    
    # Sidebar for dataset info
    with st.sidebar:
        st.header("📊 Dataset Information")
        data = st.session_state.data
        
        if data is not None and len(data) > 0:
            st.success("✅ News dataset loaded successfully!")
            st.info(f"📈 Total Articles: {data.shape[0]:,}")
            st.info(f"📋 Features: {data.shape[1]}")
            
            # Show dataset statistics
            st.subheader("Quick Stats")
            st.metric("Labels", data['label'].nunique())
            st.metric("Sources", data['source'].nunique())
            st.metric("Date Range", f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}")
            
            # Label distribution
            st.subheader("Article Labels")
            label_counts = data['label'].value_counts()
            for label, count in label_counts.items():
                st.text(f"{label}: {count}")
                
            # Source distribution  
            st.subheader("Top Sources")
            source_counts = data['source'].value_counts().head(5)
            for source, count in source_counts.items():
                st.text(f"{source}: {count}")
    
    # Main content area
    if st.session_state.data is not None and len(st.session_state.data) > 0:
        data = st.session_state.data
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data View", 
            "📊 Summary", 
            "🔍 Column Explorer", 
            "🔎 Search & Filter", 
            "📥 Export"
        ])
        
        with tab1:
            st.header("Dataset Overview")
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{data.shape[0]:,}")
            with col2:
                st.metric("Total Columns", data.shape[1])
            with col3:
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("Missing Values", f"{data.isnull().sum().sum():,}")
            
            # Data types
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': data.columns,
                'Data Type': data.dtypes.astype(str),
                'Non-Null Count': data.count(),
                'Null Count': data.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            # Display data with pagination
            st.subheader("Dataset Preview")
            
            # Pagination controls
            rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
            total_pages = (len(data) - 1) // rows_per_page + 1
            
            if total_pages > 1:
                page = st.number_input(
                    f"Page (1-{total_pages})", 
                    min_value=1, 
                    max_value=total_pages, 
                    value=1
                )
                start_idx = (page - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, len(data))
                st.dataframe(
                    data.iloc[start_idx:end_idx], 
                    use_container_width=True,
                    height=400
                )
                st.caption(f"Showing rows {start_idx + 1}-{end_idx} of {len(data)}")
            else:
                st.dataframe(data, use_container_width=True, height=400)
        
        with tab2:
            st.header("Statistical Summary")
            
            # Numeric columns summary
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.subheader("Numeric Columns")
                st.dataframe(data[numeric_cols].describe(), use_container_width=True)
                
                # Correlation matrix for numeric columns
                if len(numeric_cols) > 1:
                    st.subheader("Correlation Matrix")
                    corr_matrix = data[numeric_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Matrix",
                        color_continuous_scale="RdBu"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Categorical columns summary
            categorical_cols = data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.subheader("Categorical Columns")
                for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                    with st.expander(f"📊 {col}"):
                        value_counts = data[col].value_counts().head(10)
                        fig = px.bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            title=f"Top 10 values in {col}",
                            labels={'x': col, 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show value counts table
                        st.dataframe(
                            pd.DataFrame({
                                'Value': value_counts.index,
                                'Count': value_counts.values,
                                'Percentage': ((value_counts.values / len(data)) * 100).round(2)
                            }),
                            use_container_width=True
                        )
        
        with tab3:
            st.header("Column-wise Exploration")
            
            # Column selector
            selected_column = st.selectbox("Select a column to explore", data.columns)
            
            if selected_column:
                col_data = data[selected_column]
                stats = get_column_stats(data, selected_column)
                
                # Display column statistics
                st.subheader(f"Statistics for '{selected_column}'")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json({
                        "Data Type": stats['type'],
                        "Total Count": int(stats['count']),
                        "Null Count": int(stats['null_count']),
                        "Null Percentage": f"{(stats['null_count'] / len(data) * 100):.2f}%"
                    })
                
                with col2:
                    if stats['type'] == 'Numeric':
                        st.json({
                            "Mean": f"{stats['mean']:.2f}",
                            "Median": f"{stats['median']:.2f}",
                            "Std Dev": f"{stats['std']:.2f}",
                            "Min": f"{stats['min']:.2f}",
                            "Max": f"{stats['max']:.2f}"
                        })
                    else:
                        st.json({
                            "Unique Values": int(stats['unique_count']),
                            "Most Frequent": str(stats['most_frequent'])
                        })
                
                # Visualization
                st.subheader("Visualization")
                
                if stats['type'] == 'Numeric':
                    # Histogram
                    fig = px.histogram(
                        data, 
                        x=selected_column,
                        title=f"Distribution of {selected_column}",
                        nbins=30
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot
                    fig = px.box(
                        data,
                        y=selected_column,
                        title=f"Box Plot of {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Bar chart for categorical
                    value_counts = col_data.value_counts().head(20)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Value Counts for {selected_column}",
                        labels={'x': selected_column, 'y': 'Count'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("Search and Filter Data")
            
            # Initialize session state for filters if not exists
            if 'selected_labels' not in st.session_state:
                st.session_state.selected_labels = list(data['label'].unique())
            if 'selected_sources' not in st.session_state:
                st.session_state.selected_sources = []
            
            # Main filter controls
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Filter by Label")
                all_labels = sorted(data['label'].unique())
                selected_labels = st.multiselect(
                    "Select multiple labels:",
                    all_labels,
                    default=st.session_state.selected_labels,
                    key="label_filter"
                )
                st.session_state.selected_labels = selected_labels
                
            with col2:
                st.subheader("📰 Filter by Source")
                all_sources = sorted(data['source'].unique())
                
                # Create toggle buttons for sources
                st.write("Click to toggle sources:")
                
                # Control buttons
                control_col1, control_col2 = st.columns(2)
                with control_col1:
                    if st.button("✅ Select All Sources"):
                        st.session_state.selected_sources = list(all_sources)
                        st.rerun()
                with control_col2:
                    if st.button("❌ Clear All Sources"):
                        st.session_state.selected_sources = []
                        st.rerun()
                
                # Source toggle buttons
                source_cols = st.columns(min(3, len(all_sources)))
                
                for i, source in enumerate(all_sources):
                    col_idx = i % 3
                    with source_cols[col_idx]:
                        is_selected = source in st.session_state.selected_sources
                        button_style = "🔘" if is_selected else "⚪"
                        
                        if st.button(f"{button_style} {source}", key=f"source_{source}"):
                            if is_selected:
                                st.session_state.selected_sources.remove(source)
                            else:
                                st.session_state.selected_sources.append(source)
                            st.rerun()
                
                # Show selected sources
                if st.session_state.selected_sources:
                    st.success(f"**Selected sources:** {', '.join(st.session_state.selected_sources)}")
                else:
                    st.info("**No sources selected** - showing all articles")
                
                selected_sources = st.session_state.selected_sources
            
            # Date range filter
            st.subheader("📅 Filter by Date Range")
            min_date = data['date'].min().date()
            max_date = data['date'].max().date()
            
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            with date_col2:
                end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
            
            # Global search
            st.subheader("🔍 Text Search")
            search_col1, search_col2 = st.columns([3, 1])
            with search_col1:
                search_term = st.text_input("Search in title, description, or body:", placeholder="Enter keywords to search...")
            with search_col2:
                st.write("")  # Add spacing
                if st.button("🔄 Reset All Filters", help="Clear all filters and show all articles"):
                    st.session_state.selected_labels = list(data['label'].unique())
                    st.session_state.selected_sources = []
                    st.rerun()
            
            # Apply filters
            filtered_data = data.copy()
            
            # Filter by label
            if selected_labels:
                filtered_data = filtered_data[filtered_data['label'].isin(selected_labels)]
            
            # Filter by source (only if sources are selected)
            if selected_sources:
                filtered_data = filtered_data[filtered_data['source'].isin(selected_sources)]
            
            # Filter by date range
            start_date_ts = pd.Timestamp(start_date)
            end_date_ts = pd.Timestamp(end_date)
            
            filtered_data = filtered_data[
                (filtered_data['date'] >= start_date_ts) & 
                (filtered_data['date'] <= end_date_ts)
            ]
            
            # Apply text search
            if search_term and search_term.strip():
                search_mask = (
                    filtered_data['title'].str.contains(search_term, case=False, na=False) |
                    filtered_data['description'].str.contains(search_term, case=False, na=False) |
                    filtered_data['body'].str.contains(search_term, case=False, na=False)
                )
                filtered_data = filtered_data[search_mask]
            
            st.session_state.filtered_data = filtered_data
            
            # Display filter summary
            st.subheader("📊 Filter Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Total Articles", f"{len(filtered_data):,}")
                st.metric("Original Total", f"{len(data):,}")
                
            with summary_col2:
                if len(filtered_data) > 0:
                    filtered_labels = filtered_data['label'].value_counts()
                    st.write("**Labels in Results:**")
                    for label, count in filtered_labels.items():
                        st.text(f"{label}: {count}")
                        
            with summary_col3:
                if len(filtered_data) > 0:
                    filtered_sources = filtered_data['source'].value_counts()
                    st.write("**Sources in Results:**")
                    for source, count in filtered_sources.head(5).items():
                        st.text(f"{source}: {count}")
            
            # Display filtered results
            st.subheader("🔍 Filtered Articles")
            if len(filtered_data) == 0:
                st.warning("No articles match your current filters. Try adjusting your selection.")
            else:
                st.success(f"Found {len(filtered_data):,} articles matching your filters")
            
            if len(filtered_data) > 0:
                # Display options
                display_col1, display_col2 = st.columns([2, 1])
                with display_col1:
                    display_mode = st.radio("Display format:", ["Article Cards", "Data Table"], horizontal=True)
                with display_col2:
                    articles_per_page = st.selectbox("Articles per page:", [5, 10, 15, 20], index=1)
                
                total_pages = (len(filtered_data) - 1) // articles_per_page + 1
                
                if total_pages > 1:
                    page = st.number_input(
                        f"Page (1-{total_pages})",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        key="filtered_page"
                    )
                    start_idx = (page - 1) * articles_per_page
                    end_idx = min(start_idx + articles_per_page, len(filtered_data))
                    page_data = filtered_data.iloc[start_idx:end_idx]
                    st.caption(f"Showing articles {start_idx + 1}-{end_idx} of {len(filtered_data)} filtered results")
                else:
                    page_data = filtered_data
                
                if display_mode == "Article Cards":
                    # Display as article cards
                    for idx, row in page_data.iterrows():
                        with st.container():
                            st.markdown("---")
                            
                            # Header with label and source
                            header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
                            with header_col1:
                                st.markdown(f"**📅 {row['date'].strftime('%Y-%m-%d')}**")
                            with header_col2:
                                st.markdown(f"**🏷️ {row['label']}**")
                            with header_col3:
                                st.markdown(f"**📰 {row['source']}**")
                            
                            # Title and description
                            st.markdown(f"### {row['title']}")
                            st.markdown(f"*{row['description']}*")
                            
                            # Body preview with expand option
                            with st.expander("📖 Read full article"):
                                st.markdown(row['body'])
                                st.markdown(f"**🔗 Source URL:** {row['url']}")
                else:
                    # Display as data table
                    st.dataframe(
                        page_data,
                        use_container_width=True,
                        height=600
                    )
        
        with tab5:
            st.header("Export Data")
            
            # Choose data to export
            export_data_choice = st.radio(
                "Choose data to export:",
                ["Original Dataset", "Filtered Dataset"]
            )
            
            export_data = data if export_data_choice == "Original Dataset" else st.session_state.filtered_data
            
            if export_data is not None and len(export_data) > 0:
                st.info(f"Ready to export {len(export_data):,} rows × {export_data.shape[1]} columns")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV export
                    csv_data = convert_df_to_csv(export_data)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv_data,
                        file_name=f"dataset_export_{export_data_choice.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Excel export
                    excel_data = convert_df_to_excel(export_data)
                    st.download_button(
                        label="📊 Download as Excel",
                        data=excel_data,
                        file_name=f"dataset_export_{export_data_choice.lower().replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col3:
                    # JSON export
                    json_data = export_data.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📋 Download as JSON",
                        data=json_data,
                        file_name=f"dataset_export_{export_data_choice.lower().replace(' ', '_')}.json",
                        mime="application/json"
                    )
                
                # Export summary
                st.subheader("Export Summary")
                summary_data = {
                    "Total Rows": len(export_data),
                    "Total Columns": export_data.shape[1],
                    "Export Type": export_data_choice,
                    "Columns": list(export_data.columns)
                }
                st.json(summary_data)
            else:
                st.warning("No data available for export.")
    
    else:
        # Error screen when data fails to load
        st.markdown("""
        ## ⚠️ News Dataset Not Found
        
        The news dataset (`combined_news_labeled.json`) could not be loaded.
        
        Please ensure the file exists and contains valid JSON data with news articles.
        
        ### Expected Dataset Structure:
        - **title**: Article headline
        - **content**: Article text content
        - **category**: News category (Business, Technology, etc.)
        - **sentiment**: Article sentiment (Positive, Negative, Neutral)
        - **source**: News source publication
        - **date**: Publication date
        - **word_count**: Number of words in article
        - **keywords**: Array of relevant keywords
        """)
        
        # Sample data structure guide
        with st.expander("📚 Supported Data Formats"):
            st.markdown("""
            **CSV Files**: Comma-separated values with headers
            ```
            Name,Age,City
            John,25,New York
            Jane,30,Los Angeles
            ```
            
            **Excel Files**: .xlsx or .xls format with data in the first sheet
            
            **JSON Files**: Array of objects or records format
            ```json
            [
                {"Name": "John", "Age": 25, "City": "New York"},
                {"Name": "Jane", "Age": 30, "City": "Los Angeles"}
            ]
            ```
            """)

if __name__ == "__main__":
    main()
