import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
import json
import numpy as np
import os
import ast
import pathlib

def safe_eval(data):
    """Safely evaluate string data that might be a list"""
    if pd.isna(data):
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, str):
        try:
            # Try to evaluate as Python literal
            return ast.literal_eval(data)
        except:
            # If that fails, try to parse as string list
            if data.startswith('[') and data.endswith(']'):
                # Remove brackets and split by comma
                items = data[1:-1].split(',')
                return [item.strip().strip("'\"") for item in items if item.strip()]
            else:
                # Return as single item list
                return [data] if data else []
    return []

def clean_tweet_id(tweet_id):
    """Remove _x suffix from tweet_id for matching"""
    if pd.isna(tweet_id):
        return tweet_id
    tweet_id_str = str(tweet_id)
    if tweet_id_str.endswith('_x'):
        return tweet_id_str[:-2]  # Remove _x suffix
    return tweet_id_str

def create_network_graph(df):
    """
    Create a network graph from Twitter data with proper referenced tweet processing
    """
    # Create directed graph
    G = nx.DiGraph()
    
    st.info(f"Processing {len(df)} tweets to build network...")
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Create a mapping from cleaned tweet_id to username for quick lookup
    df['cleaned_tweet_id'] = df['tweet_id'].apply(clean_tweet_id)
    tweet_to_user = df.set_index('cleaned_tweet_id')['username'].to_dict()
    
    # Debug: Show some tweet ID mappings
    st.sidebar.info(f"Tweet ID mapping sample: {len(tweet_to_user)} entries")
    
    # Precompute user engagement and max followers
    user_engagement = df.groupby('username')['Engagement'].sum().to_dict()
    user_followers = df.groupby('username')['followers_count'].max().to_dict()
    user_types = df.groupby('username')['user_type'].first().to_dict()
    user_companies = df.groupby('username')['company'].first().to_dict()
    user_names = df.groupby('username')['name'].first().to_dict()
    
    # Get all unique usernames in the dataset
    all_usernames = set(df['username'].dropna().unique())
    
    # Counters for debugging
    interaction_counts = {'mentions': 0, 'replies': 0, 'retweets': 0, 'quotes': 0}
    
    # Process each row to build the graph
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 100 == 0:  # Update progress every 100 rows
            progress_bar.progress(min((idx + 1) / len(df), 1.0))
        
        author_username = row['username']
        
        # Skip if author username is missing
        if pd.isna(author_username):
            continue
            
        # Add author node with attributes
        if author_username not in G:
            G.add_node(author_username, 
                      name=user_names.get(author_username, author_username),
                      followers_count=user_followers.get(author_username, 0),
                      user_type=user_types.get(author_username, 'Unknown'),
                      engagement=user_engagement.get(author_username, 0),
                      company=user_companies.get(author_username, 'not mentioned'),
                      is_in_dataset=True)  # Mark as existing in dataset
        
        # Process mentions with robust handling
        try:
            mentions = safe_eval(row.get('mentions', []))
            
            for mentioned_user in mentions:
                if mentioned_user and str(mentioned_user) != 'nan' and mentioned_user != author_username:
                    # Clean the mentioned user string
                    mentioned_user = str(mentioned_user).strip().lstrip('@')
                    if not mentioned_user:
                        continue
                    
                    # Check if mentioned user exists in our dataset
                    is_in_dataset = mentioned_user in all_usernames
                    
                    # Add mentioned user as node if not exists
                    if mentioned_user not in G:
                        G.add_node(mentioned_user, 
                                  name=mentioned_user,
                                  followers_count=0,
                                  user_type='Unknown',
                                  engagement=0,
                                  company='not mentioned',
                                  is_in_dataset=is_in_dataset)
                    
                    # Add or update edge for mention
                    if G.has_edge(author_username, mentioned_user):
                        G[author_username][mentioned_user]['mentions'] += 1
                        G[author_username][mentioned_user]['total_interactions'] += 1
                    else:
                        G.add_edge(author_username, mentioned_user, 
                                 mentions=1, replies=0, retweets=0, quotes=0, total_interactions=1)
                    
                    interaction_counts['mentions'] += 1
        except Exception as e:
            continue
        
        # Process referenced tweets (replies, retweets, quotes) - CORRECTED LOGIC
        try:
            ref_tweets = safe_eval(row.get('referenced_tweets', []))
            
            for ref_tweet in ref_tweets:
                if isinstance(ref_tweet, dict):
                    ref_type = ref_tweet.get('type', '')
                    parent_tweet_id = ref_tweet.get('id', '')
                else:
                    continue  # Skip if not a dictionary
                
                # Skip if no parent tweet ID
                if not parent_tweet_id:
                    continue
                
                # Clean the parent tweet ID (remove _x suffix if present in mapping)
                cleaned_parent_id = clean_tweet_id(parent_tweet_id)
                
                # Find the original author of the parent tweet
                if cleaned_parent_id in tweet_to_user:
                    target_user = tweet_to_user[cleaned_parent_id]
                    
                    # Skip if target user is same as author or missing
                    if (pd.isna(target_user) or target_user == author_username or 
                        str(target_user) == 'nan'):
                        continue
                    
                    target_user = str(target_user).strip()
                    
                    # Check if target user exists in our dataset
                    is_in_dataset = target_user in all_usernames
                    
                    # Add target user as node if not exists
                    if target_user not in G:
                        G.add_node(target_user, 
                                  name=user_names.get(target_user, target_user),
                                  followers_count=user_followers.get(target_user, 0),
                                  user_type=user_types.get(target_user, 'Unknown'),
                                  engagement=user_engagement.get(target_user, 0),
                                  company=user_companies.get(target_user, 'not mentioned'),
                                  is_in_dataset=is_in_dataset)
                    
                    # Add or update edge based on reference type
                    if G.has_edge(author_username, target_user):
                        if ref_type == 'replied_to':
                            G[author_username][target_user]['replies'] += 1
                            G[author_username][target_user]['total_interactions'] += 1
                            interaction_counts['replies'] += 1
                        elif ref_type == 'retweeted':
                            G[author_username][target_user]['retweets'] += 1
                            G[author_username][target_user]['total_interactions'] += 1
                            interaction_counts['retweets'] += 1
                        elif ref_type == 'quoted':
                            G[author_username][target_user]['quotes'] += 1
                            G[author_username][target_user]['total_interactions'] += 1
                            interaction_counts['quotes'] += 1
                    else:
                        edge_data = {
                            'mentions': 0, 
                            'replies': 0, 
                            'retweets': 0, 
                            'quotes': 0, 
                            'total_interactions': 1
                        }
                        if ref_type == 'replied_to':
                            edge_data['replies'] = 1
                            interaction_counts['replies'] += 1
                        elif ref_type == 'retweeted':
                            edge_data['retweets'] = 1
                            interaction_counts['retweets'] += 1
                        elif ref_type == 'quoted':
                            edge_data['quotes'] = 1
                            interaction_counts['quotes'] += 1
                        G.add_edge(author_username, target_user, **edge_data)
                else:
                    # Debug: Show unmatched parent tweets
                    if idx < 10:  # Only show first few for debugging
                        st.sidebar.warning(f"Parent tweet {cleaned_parent_id} not found in dataset")
                        
        except Exception as e:
            # Skip referenced tweets errors and continue
            continue
    
    progress_bar.progress(1.0)
    
    # Display interaction statistics
    st.success(f"**Interaction Statistics:**")
    st.success(f"- Mentions: {interaction_counts['mentions']}")
    st.success(f"- Replies: {interaction_counts['replies']}")
    st.success(f"- Retweets: {interaction_counts['retweets']}")
    st.success(f"- Quotes: {interaction_counts['quotes']}")
    
    # Also show edge-based statistics
    total_mentions = sum([G[u][v].get('mentions', 0) for u, v in G.edges()])
    total_replies = sum([G[u][v].get('replies', 0) for u, v in G.edges()])
    total_retweets = sum([G[u][v].get('retweets', 0) for u, v in G.edges()])
    total_quotes = sum([G[u][v].get('quotes', 0) for u, v in G.edges()])
    
    st.info(f"**Edge-based Statistics:**")
    st.info(f"- Mentions in edges: {total_mentions}")
    st.info(f"- Replies in edges: {total_replies}")
    st.info(f"- Retweets in edges: {total_retweets}")
    st.info(f"- Quotes in edges: {total_quotes}")
    
    return G

def plot_network_graph(G, layout='spring', max_nodes=None):
    """
    Plot the network graph using Plotly
    """
    if len(G.nodes()) == 0:
        st.error("No nodes to display in the graph")
        return None
    
    # If max_nodes is specified, filter the graph
    if max_nodes and len(G.nodes()) > max_nodes:
        # Get top nodes by followers count
        node_followers = [(node, G.nodes[node].get('followers_count', 0)) for node in G.nodes()]
        node_followers.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in node_followers[:max_nodes]]
        
        # Create subgraph with top nodes
        G = G.subgraph(top_nodes).copy()
        
    if len(G.nodes()) == 0:
        st.error("No nodes to display after filtering")
        return None
        
    try:
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
    except Exception as e:
        st.error(f"Error in graph layout: {e}")
        return None
    
    # Extract node positions
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_names = []
    
    for node in G.nodes():
        try:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            followers = G.nodes[node].get('followers_count', 1)
            name = G.nodes[node].get('name', node)
            user_type = G.nodes[node].get('user_type', 'Unknown')
            engagement = G.nodes[node].get('engagement', 0)
            company = G.nodes[node].get('company', 'not mentioned')
            is_in_dataset = G.nodes[node].get('is_in_dataset', False)
            
            node_text.append(
                f"Username: @{node}<br>"
                f"Name: {name}<br>"
                f"User Type: {user_type}<br>"
                f"Followers: {followers:,}<br>"
                f"Total Engagement: {engagement:,}<br>"
                f"Company: {company}<br>"
                f"In Dataset: {'Yes' if is_in_dataset else 'No'}"
            )
            
            # Size based on followers (log scale for better visualization)
            size_val = max(15, min(60, np.log(followers + 1) * 6))
            node_size.append(size_val)
            
            # Color based on whether user is in dataset
            if is_in_dataset:
                # Color by user type for users in dataset
                if user_type == 'HCP':
                    node_color.append('#1f77b4')  # Blue
                elif user_type == 'ANALYST':
                    node_color.append('#ff7f0e')  # Orange
                elif user_type == 'BLOGGER':
                    node_color.append('#2ca02c')  # Green
                elif user_type == 'MEDICAL COMMUNITY':
                    node_color.append('#9467bd')  # Purple
                else:
                    node_color.append('#d62728')  # Red
            else:
                node_color.append('#7f7f7f')  # Gray for users not in dataset
            
            node_names.append(node)
        except Exception as e:
            continue
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_text = []
    edge_width = []
    edge_colors = []
    
    for edge in G.edges():
        try:
            source, target = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge info
            edge_data = G[source][target]
            interactions = []
            total_weight = edge_data.get('total_interactions', 0)
            
            if edge_data.get('mentions', 0) > 0:
                interactions.append(f"Mentions: {edge_data['mentions']}")
            if edge_data.get('replies', 0) > 0:
                interactions.append(f"Replies: {edge_data['replies']}")
            if edge_data.get('retweets', 0) > 0:
                interactions.append(f"Retweets: {edge_data['retweets']}")
            if edge_data.get('quotes', 0) > 0:
                interactions.append(f"Quotes: {edge_data['quotes']}")
            
            edge_text.append(
                f"From: @{source}<br>"
                f"To: @{target}<br>"
                f"Total Interactions: {total_weight}<br>" + 
                "<br>".join(interactions)
            )
            
            # Edge width based on total interactions
            width_val = max(1, min(8, total_weight * 0.8))
            edge_width.append(width_val)
            
            # Edge color based on interaction type
            if edge_data.get('replies', 0) > 0:
                edge_colors.append('#ff0000')  # Red for replies
            elif edge_data.get('retweets', 0) > 0:
                edge_colors.append('#00ff00')  # Green for retweets
            elif edge_data.get('quotes', 0) > 0:
                edge_colors.append('#0000ff')  # Blue for quotes
            else:
                edge_colors.append('#888888')  # Gray for mentions
            
        except Exception as e:
            continue
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add edges with arrows
    if edge_x:  # Only add edges if they exist
        for i in range(0, len(edge_x), 3):
            if i + 2 < len(edge_x):
                fig.add_trace(go.Scatter(
                    x=[edge_x[i], edge_x[i+1], None],
                    y=[edge_y[i], edge_y[i+1], None],
                    line=dict(
                        width=edge_width[i//3], 
                        color=edge_colors[i//3]
                    ),
                    hoverinfo='text',
                    hovertext=edge_text[i//3],
                    mode='lines',
                    showlegend=False
                ))
    
    # Add nodes
    if node_x:  # Only add nodes if they exist
        fig.add_trace(go.Scatter(
            x=node_x, 
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            textposition="top center",
            text=[name[:15] + '...' if len(name) > 15 else name for name in node_names],
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            hovertext=node_text,
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='Twitter User Interaction Network<br><sub>Node size: Followers count | Node color: User type (Gray: Not in dataset) | Edge width: Interaction frequency</sub>',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=900,
        height=700
    )
    
    return fig

def main():
    st.set_page_config(page_title="Twitter Network Analysis", layout="wide")
    st.title("📊 Twitter User Interaction Network")
    
    # Define the file path
    #file_path = r"C:\Users\rajas\OneDrive\Desktop\GRE\Documents\ASU\Semester_4\CPT_Summer\HCP_Analysis\eha_Twitter_tagged_Final.xlsx"
    file_path = pathlib.Path(__file__).parent / "eha_Twitter_tagged_Final.xlsx"

    
    # Check if file exists
    if os.path.exists(file_path):
        try:
            # Read the Excel file
            st.info("Loading Excel file...")
            df = pd.read_excel(file_path)
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Display column information for debugging
            with st.expander("Dataset Info (Click to expand)"):
                st.write("**Columns in dataset:**", list(df.columns))
                st.write("**First few rows of referenced_tweets:**")
                st.write(df['referenced_tweets'].head(10))
                st.write("**First few rows of tweet_id:**")
                st.write(df['tweet_id'].head(10))
            
            # Show basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Users", df['username'].nunique())
            with col2:
                st.metric("Total Tweets", len(df))
            with col3:
                st.metric("User Types", df['user_type'].nunique())
            with col4:
                st.metric("Max Followers", f"{df['followers_count'].max():,}")
            
            # Sidebar controls
            st.sidebar.header("Network Settings")
            
            # User Type Filter
            user_types = ['All'] + sorted(df['user_type'].dropna().unique().tolist())
            selected_user_types = st.sidebar.multiselect(
                "Filter by User Type",
                options=user_types,
                default=['All'],
                help="Select user types to include in the network"
            )
            
            # Company Filter
            companies = ['All'] + sorted(df['company'].dropna().unique().tolist())
            selected_companies = st.sidebar.multiselect(
                "Filter by Company",
                options=companies,
                default=['All'],
                help="Select companies to include in the network"
            )
            
            # Number of nodes filter
            max_nodes = st.sidebar.slider(
                "Maximum Number of Nodes to Display",
                min_value=10,
                max_value=500,
                value=100,
                help="Limit the number of nodes for better visualization"
            )
            
            layout_option = st.sidebar.selectbox(
                "Layout Algorithm",
                ['spring', 'circular', 'kamada_kawai'],
                help="Choose how to arrange the nodes in the network"
            )
            
            min_followers = st.sidebar.slider(
                "Minimum Followers Filter",
                min_value=0,
                max_value=int(df['followers_count'].max()),
                value=0,
                help="Only show users with at least this many followers"
            )
            
            # Apply filters
            filtered_df = df.copy()
            
            # Apply user type filter
            if 'All' not in selected_user_types:
                filtered_df = filtered_df[filtered_df['user_type'].isin(selected_user_types)]
            
            # Apply company filter
            if 'All' not in selected_companies:
                filtered_df = filtered_df[filtered_df['company'].isin(selected_companies)]
            
            # Apply followers filter
            user_followers = filtered_df.groupby('username')['followers_count'].max()
            users_to_include = user_followers[user_followers >= min_followers].index
            filtered_df = filtered_df[filtered_df['username'].isin(users_to_include)]
            
            st.sidebar.info(f"Showing {len(users_to_include)} users after filtering")
            
            # Display filter summary
            st.subheader("Filter Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Selected User Types:** {', '.join(selected_user_types) if 'All' in selected_user_types else ', '.join(selected_user_types)}")
            with col2:
                st.write(f"**Selected Companies:** {', '.join(selected_companies) if 'All' in selected_companies else ', '.join(selected_companies)}")
            with col3:
                st.write(f"**Min Followers:** {min_followers:,}")
            with col4:
                st.write(f"**Max Nodes:** {max_nodes}")
            
            # Create network graph
            if len(filtered_df) > 0:
                with st.spinner("Building network graph..."):
                    try:
                        G = create_network_graph(filtered_df)
                        
                        if len(G.nodes) > 0:
                            # Display network statistics
                            st.subheader("Network Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Nodes", len(G.nodes()))
                            with col2:
                                st.metric("Edges", len(G.edges()))
                            with col3:
                                try:
                                    density = nx.density(G)
                                    st.metric("Density", f"{density:.4f}")
                                except:
                                    st.metric("Density", "N/A")
                            with col4:
                                try:
                                    avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                                    st.metric("Avg Degree", f"{avg_degree:.2f}")
                                except:
                                    st.metric("Avg Degree", "N/A")
                            
                            # Plot the graph
                            st.subheader("Interactive Network Graph")
                            fig = plot_network_graph(G, layout=layout_option, max_nodes=max_nodes)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add legend
                                st.markdown("""
                                **Legend:**
                                - **Node Colors:** Blue=HCP, Orange=Analyst, Green=Blogger, Purple=Medical Community, Red=Other, Gray=Not in dataset
                                - **Edge Colors:** Red=Replies, Green=Retweets, Blue=Quotes, Gray=Mentions
                                - **Node Size:** Based on followers count
                                - **Edge Width:** Based on interaction frequency
                                """)
                            else:
                                st.error("Could not generate the network graph")
                            
                            # Show node details
                            st.subheader("Node Details")
                            nodes_data = []
                            for node in G.nodes():
                                node_data = {
                                    'Username': node,
                                    'Name': G.nodes[node].get('name', 'N/A'),
                                    'Followers': G.nodes[node].get('followers_count', 0),
                                    'User Type': G.nodes[node].get('user_type', 'Unknown'),
                                    'Total Engagement': G.nodes[node].get('engagement', 0),
                                    'Company': G.nodes[node].get('company', 'not mentioned'),
                                    'In Dataset': G.nodes[node].get('is_in_dataset', False),
                                    'Degree': G.degree(node)
                                }
                                nodes_data.append(node_data)
                            
                            nodes_df = pd.DataFrame(nodes_data)
                            st.dataframe(
                                nodes_df.sort_values('Followers', ascending=False), 
                                use_container_width=True,
                                height=400
                            )
                            
                            # Show edge details (interactions)
                            st.subheader("Interaction Details")
                            edges_data = []
                            for edge in G.edges():
                                source, target = edge
                                edge_data = G[source][target]
                                total_interactions = edge_data.get('total_interactions', 0)
                                
                                if total_interactions > 0:
                                    edges_data.append({
                                        'Source': source,
                                        'Target': target,
                                        'Mentions': edge_data.get('mentions', 0),
                                        'Replies': edge_data.get('replies', 0),
                                        'Retweets': edge_data.get('retweets', 0),
                                        'Quotes': edge_data.get('quotes', 0),
                                        'Total Interactions': total_interactions
                                    })
                            
                            if edges_data:
                                edges_df = pd.DataFrame(edges_data)
                                st.dataframe(
                                    edges_df.sort_values('Total Interactions', ascending=False), 
                                    use_container_width=True,
                                    height=400
                                )
                            else:
                                st.info("No interactions found in the filtered data.")
                            
                        else:
                            st.warning("No nodes found in the network graph. Please check your data or adjust filters.")
                            
                    except Exception as e:
                        st.error(f"Error creating network graph: {str(e)}")
                        st.info("This might be due to issues with the data format. Check the dataset info above.")
            else:
                st.warning("No data matches the current filters. Please adjust your filter settings.")
                    
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.info("Please make sure the file path is correct and the Excel file is not corrupted.")
    else:
        st.error(f"File not found at: {file_path}")
        st.info("Please check if the file exists at the specified location.")

if __name__ == "__main__":
    main()
