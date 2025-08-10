import streamlit as st
import pandas as pd
import plotly.express as px
import time
import webbrowser
import base64
from streamlit.components.v1 import html
from datetime import datetime
import numpy as np
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ========== NLP Libraries ==========
from textblob import TextBlob
import spacy
from spacy import displacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ========== MUST BE FIRST COMMAND ==========
st.set_page_config(
    page_title="✨ Amazon Review Insight Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Open Amazon.in in Browser ==========
if not st.session_state.get('browser_opened'):
    webbrowser.open_new_tab('https://www.amazon.in')
    st.session_state.browser_opened = True

# ========== Review Scraping Functions ==========
def extract_all_visible_text(product_url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        st.info("🔄 Loading Amazon product page...")
        driver.get(product_url)
        time.sleep(5)

        body = driver.find_element("tag name", "body")
        full_text = body.text.strip()
        return full_text

    except Exception as e:
        st.error(f"Error loading page: {str(e)}")
        return ""
    finally:
        driver.quit()

def extract_reviews(text):
    start_keyword = "Top reviews from India"
    end_keyword = "See more reviews"
    try:
        start = text.index(start_keyword) + len(start_keyword)
        end = text.index(end_keyword, start)
        review_section = text[start:end].strip()
        return review_section
    except ValueError:
        return ""

def parse_review(review_text):
    try:
        # Initialize default values
        reviewer = "Unknown"
        rating = 0
        verified = False
        date = None
        review_body = ""
        helpful_votes = 0
        
        # Split into lines and process each line
        lines = [line.strip() for line in review_text.split('\n') if line.strip()]
        
        if not lines:
            return None
            
        # First line is reviewer name
        reviewer = lines[0]
        
        # Process remaining lines
        for i in range(1, len(lines)):
            line = lines[i]
            
            if "out of 5 stars" in line:
                # Extract rating
                rating_part = line.split("out of 5 stars")[0]
                rating = float(rating_part.split()[-1])
                
                # The rest of this line might be the start of the review
                review_part = line.split("out of 5 stars")[1].strip()
                if review_part:
                    review_body += review_part + " "
                    
            elif "Reviewed in India on " in line:
                try:
                    date_part = line.split("Reviewed in India on ")[1].strip()
                    date = datetime.strptime(date_part, "%d %B %Y").strftime("%Y-%m-%d")
                except:
                    pass
            elif "Verified Purchase" in line:
                verified = True
            elif "Helpful" in line:
                # Extract helpful votes if available
                if "person found this helpful" in line or "people found this helpful" in line:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            helpful_votes = int(part)
                            break
            elif "Report" in line or "Customer image" in line:
                continue  # Skip these lines
            else:
                # This is part of the review content
                review_body += line + " "
        
        # Clean up review body
        review_body = review_body.strip()
        
        return {
            "Reviewer": reviewer,
            "Rating": rating,
            "Verified Purchase": "Yes" if verified else "No",
            "Date": date,
            "Review": review_body,
            "Location": "India",
            "Helpful Votes": helpful_votes
        }
    except Exception as e:
        st.warning(f"Couldn't parse review: {str(e)}")
        return None

# ========== Enhanced NLP Analysis Functions ==========
def analyze_sentiment_nlp(review_text):
    """Advanced sentiment analysis using TextBlob and spaCy"""
    # TextBlob analysis
    blob = TextBlob(review_text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Determine sentiment category
    if polarity > 0.2:
        sentiment = "Positive"
        reason = f"Strong positive sentiment (polarity: {polarity:.2f})"
    elif polarity < -0.2:
        sentiment = "Negative"
        reason = f"Strong negative sentiment (polarity: {polarity:.2f})"
    else:
        sentiment = "Neutral"
        reason = f"Neutral sentiment (polarity: {polarity:.2f})"
    
    # Additional insights from spaCy
    doc = nlp(review_text)
    
    # Extract key aspects being discussed
    aspects = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and token.text.lower() not in stopwords.words('english'):
            aspects.append(token.text)
    
    # Count most frequent aspects
    aspect_counts = Counter(aspects)
    top_aspects = [aspect for aspect, count in aspect_counts.most_common(3)]
    
    if top_aspects:
        reason += f". Main aspects: {', '.join(top_aspects)}"
    
    return sentiment, reason, polarity, subjectivity

def categorize_review_content(review_text):
    """Categorize review into product aspects using NLP"""
    doc = nlp(review_text.lower())
    
    categories = {
        "Quality": ["quality", "material", "build", "durable", "sturdy", "flimsy", "cheap"],
        "Performance": ["performance", "speed", "fast", "slow", "efficient", "powerful"],
        "Design": ["design", "look", "appearance", "style", "color", "shape", "size"],
        "Value": ["price", "cost", "worth", "value", "expensive", "affordable"],
        "Features": ["feature", "function", "setting", "mode", "option", "specification"],
        "Usability": ["easy", "difficult", "simple", "complex", "user-friendly", "interface"],
        "Service": ["delivery", "service", "support", "warranty", "return", "refund"]
    }
    
    detected_categories = set()
    
    # Check for category keywords
    for token in doc:
        for category, keywords in categories.items():
            if token.text in keywords:
                detected_categories.add(category)
    
    # If no categories detected, try pattern matching
    if not detected_categories:
        for category, keywords in categories.items():
            if any(keyword in review_text for keyword in keywords):
                detected_categories.add(category)
    
    # If still no categories, use noun chunks
    if not detected_categories:
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        for chunk in noun_chunks:
            for category, keywords in categories.items():
                if any(keyword in chunk for keyword in keywords):
                    detected_categories.add(category)
    
    return ", ".join(detected_categories) if detected_categories else "General"

def analyze_linguistic_patterns(review_text):
    """Analyze linguistic patterns for authenticity detection"""
    doc = nlp(review_text)
    
    # Features to examine
    features = {
        "excessive_adjectives": 0,
        "first_person_pronouns": 0,
        "passive_voice": 0,
        "specific_details": 0,
        "emotional_language": 0,
        "generic_phrases": 0
    }
    
    # Common generic phrases
    generic_phrases = [
        "great product", "highly recommend", "value for money", 
        "must buy", "good quality", "as described", "happy with purchase"
    ]
    
    # Count features
    for token in doc:
        # Excessive adjectives
        if token.pos_ == "ADJ":
            features["excessive_adjectives"] += 1
        
        # First person pronouns
        if token.text.lower() in ["i", "me", "my", "mine", "we", "our"]:
            features["first_person_pronouns"] += 1
        
        # Passive voice detection (simplified)
        if token.dep_ == "auxpass":
            features["passive_voice"] += 1
    
    # Check for specific details (numbers, measurements, etc.)
    if any(token.like_num for token in doc):
        features["specific_details"] += 1
    
    # Check for emotional language
    emotional_words = ["love", "hate", "awesome", "terrible", "disappointed", "thrilled"]
    if any(token.text.lower() in emotional_words for token in doc):
        features["emotional_language"] += 1
    
    # Check for generic phrases
    features["generic_phrases"] = sum(phrase in review_text.lower() for phrase in generic_phrases)
    
    return features

def check_authenticity_nlp(review):
    """Enhanced authenticity check using NLP patterns"""
    text = review['Review'].lower()
    features = analyze_linguistic_patterns(text)
    
    # Calculate authenticity score
    score = 0
    
    # Positive indicators
    score += min(features["first_person_pronouns"], 5) * 2  # Max 10 points
    score += min(features["specific_details"], 3) * 5  # Max 15 points
    
    # Negative indicators
    score -= min(features["excessive_adjectives"], 10)  # Max -10 points
    score -= min(features["generic_phrases"], 5) * 3  # Max -15 points
    score -= min(features["emotional_language"], 5) * 2  # Max -10 points
    
    # Normalize score to 0-100 range
    normalized_score = max(0, min(100, 50 + score))
    
    # Determine authenticity
    if normalized_score >= 70:
        authenticity = "Genuine"
        reason = f"High authenticity score ({normalized_score}/100). Shows personal experience and specific details."
    elif normalized_score >= 40:
        authenticity = "Likely Genuine"
        reason = f"Moderate authenticity score ({normalized_score}/100). Some indicators of personal experience."
    else:
        authenticity = "Potentially Fake"
        reason = f"Low authenticity score ({normalized_score}/100). Contains patterns common in fake reviews."
    
    return authenticity, reason, normalized_score

# ========== Premium CSS Animation and Styling ==========
def inject_custom_css():
    st.markdown(f"""
    <style>
        /* Main styling */
        .stApp {{
            background-color: #f9fafc;
        }}
        
        /* Header animation */
        .header-animation {{
            animation: fadeIn 1s ease-in-out;
        }}
        
        /* Metric cards */
        .metric-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            animation: slideUp 0.5s ease-out;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            margin: 10px 0;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        /* Review cards */
        .review-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            animation: fadeIn 0.5s ease-out;
        }}
        
        .review-genuine {{
            border-left: 4px solid #00AA45;
        }}
        
        .review-fake {{
            border-left: 4px solid #FF4D4D;
        }}
        
        .review-tag {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-right: 10px;
        }}
        
        .tag-genuine {{
            background-color: #E6F7EE;
            color: #00AA45;
        }}
        
        .tag-fake {{
            background-color: #FFEEEE;
            color: #FF4D4D;
        }}
        
        /* Timeline in sidebar */
        .timeline {{
            position: relative;
            padding-left: 20px;
        }}
        
        .timeline:before {{
            content: '';
            position: absolute;
            left: 6px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #6E8EFB;
        }}
        
        .timeline-item {{
            position: relative;
            margin-bottom: 20px;
        }}
        
        .timeline-item:before {{
            content: '';
            position: absolute;
            left: -20px;
            top: 5px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #6E8EFB;
        }}
        
        /* Accordion in sidebar */
        .accordion {{
            margin: 10px 0;
        }}
        
        .accordion-item {{
            margin-bottom: 8px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .accordion-header {{
            padding: 12px 15px;
            background: #f5f7fa;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }}
        
        .accordion-header:hover {{
            background: #eef2f7;
        }}
        
        .accordion-content {{
            padding: 0 15px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            background: white;
        }}
        
        .accordion-item.active .accordion-content {{
            max-height: 200px;
            padding: 15px;
        }}
        
        /* Blob animations */
        .blob {{
            position: absolute;
            border-radius: 50%;
            filter: blur(40px);
            opacity: 0.7;
            z-index: -1;
            animation: blobFloat 15s infinite linear;
        }}
        
        .floating {{
            animation: blobFloat 20s infinite ease-in-out;
        }}
        
        /* Keyframes */
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        
        @keyframes slideUp {{
            from {{ transform: translateY(20px); opacity: 0; }}
            to {{ transform: translateY(0); opacity: 1; }}
        }}
        
        @keyframes blobFloat {{
            0% {{ transform: translate(0px, 0px) scale(1); }}
            33% {{ transform: translate(30px, -50px) scale(1.1); }}
            66% {{ transform: translate(-20px, 20px) scale(0.9); }}
            100% {{ transform: translate(0px, 0px) scale(1); }}
        }}
    </style>
    """, unsafe_allow_html=True)

# Inject custom CSS
inject_custom_css()

# ========== App Header with Animation ==========
def render_header():
    st.markdown("""
    <div class="header-animation">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 15px;">
                <path d="M20 40C31.0457 40 40 31.0457 40 20C40 8.9543 31.0457 0 20 0C8.9543 0 0 8.9543 0 20C0 31.0457 8.9543 40 20 40Z" fill="#6E8EFB"/>
                <path d="M26.6667 13.3333H13.3333C11.4924 13.3333 10 14.8257 10 16.6667V23.3333C10 25.1743 11.4924 26.6667 13.3333 26.6667H26.6667C28.5076 26.6667 30 25.1743 30 23.3333V16.6667C30 14.8257 28.5076 13.3333 26.6667 13.3333Z" fill="white"/>
                <path d="M20 23.3333C21.8409 23.3333 23.3333 21.8409 23.3333 20C23.3333 18.159 21.8409 16.6667 20 16.6667C18.159 16.6667 16.6667 18.159 16.6667 20C16.6667 21.8409 18.159 23.3333 20 23.3333Z" fill="#6E8EFB"/>
            </svg>
            <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Amazon Review Insight Pro</h1>
        </div>
        <p style="color: #666; font-size: 1.1rem; margin-bottom: 2rem; max-width: 800px;">
            Advanced sentiment analysis and authenticity detection for Amazon product reviews. 
            Gain deep insights into customer experiences with our AI-powered analytics platform.
        </p>
    </div>
    """, unsafe_allow_html=True)

render_header()

# ========== Main App Functionality ==========
def main():
    # URL Input with Floating Animation
    with st.container():
        st.markdown("""
        <div style="position: relative;">
            <div class="blob" style="top: -50px; right: -50px; width: 200px; height: 200px;"></div>
            <div class="blob" style="bottom: -50px; left: -50px; width: 150px; height: 150px;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input("", placeholder="🔍 Paste any Amazon product URL here...", key="url_input", 
                              help="Example: https://www.amazon.in/dp/B08L5V53ZP")
        with col2:
            analyze_btn = st.button("Analyze Reviews", key="analyze_btn", 
                                   help="Click to analyze the product reviews")

    if analyze_btn and url:
        with st.spinner("🔍 Scraping and analyzing Amazon reviews..."):
            try:
                # Step 1: Extract all visible text
                full_text = extract_all_visible_text(url)
                
                if not full_text:
                    st.error("Failed to extract text from the product page")
                    return
                
                # Step 2: Extract reviews section
                review_section = extract_reviews(full_text)
                
                if not review_section:
                    st.error("Could not find reviews section on the page")
                    return
                
                # Step 3: Split into individual reviews (separated by "Helpful" and "Report")
                raw_reviews = []
                current_review = []
                lines = review_section.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this line starts a new review (after "Report")
                    if line.startswith("Report") and current_review:
                        # Add the completed review
                        raw_reviews.append("\n".join(current_review))
                        current_review = []
                        continue
                    
                    # Check if this is the first review (after "Top reviews from India")
                    if not current_review and line and not any(x in line for x in ["Helpful", "Report", "Customer image"]):
                        current_review.append(line)
                    elif current_review:
                        current_review.append(line)
                
                # Add the last review if exists
                if current_review:
                    raw_reviews.append("\n".join(current_review))
                
                # Step 4: Process reviews into structured data
                reviews = []
                for raw_review in raw_reviews:
                    parsed = parse_review(raw_review)
                    if parsed:
                        reviews.append(parsed)
                
                if not reviews:
                    st.error("No reviews could be parsed from the page")
                    return
                
                # Step 5: Enhanced NLP Analysis
                progress_bar = st.progress(0)
                for i, review in enumerate(reviews):
                    # Update progress
                    progress = int((i + 1) / len(reviews) * 100)
                    progress_bar.progress(progress)
                    
                    # Analyze sentiment with NLP
                    sentiment, sentiment_reason, polarity, subjectivity = analyze_sentiment_nlp(review['Review'])
                    review['Sentiment'] = sentiment
                    review['Sentiment Analysis'] = sentiment_reason
                    review['Sentiment Polarity'] = polarity
                    review['Sentiment Subjectivity'] = subjectivity
                    
                    # Categorize review content
                    review['Category'] = categorize_review_content(review['Review'])
                    
                    # Check authenticity with NLP
                    authenticity, authenticity_reason, auth_score = check_authenticity_nlp(review)
                    review['Authenticity'] = authenticity
                    review['Authenticity Reason'] = authenticity_reason
                    review['Authenticity Score'] = auth_score
                
                # Create DataFrame
                df = pd.DataFrame(reviews)
                
                # Ensure all expected columns are present
                expected_columns = ['Product', 'Reviewer', 'Location', 'Date', 'Rating', 
                                  'Review', 'Verified Purchase', 'Helpful Votes', 'Category',
                                  'Sentiment', 'Sentiment Analysis', 'Sentiment Polarity', 'Sentiment Subjectivity',
                                  'Authenticity', 'Authenticity Reason', 'Authenticity Score']
                
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = "N/A"
                
                # Add product name (simplified)
                df['Product'] = url.split('/')[-1] if len(url.split('/')) > 3 else "Amazon Product"
                
                st.session_state.reviews_df = df
                st.session_state.analysis_done = True
                st.success(f"✅ Successfully analyzed {len(df)} reviews!")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.session_state.analysis_done = False

    if 'analysis_done' in st.session_state and st.session_state.analysis_done:
        df = st.session_state.reviews_df
        
        # Summary Metrics with Animation
        cols = st.columns(4)
        metrics = [
            ("Total Reviews", len(df), "#6e8efb"),
            ("Average Rating", round(df['Rating'].mean(), 1), "#a777e3"),
            ("Authentic Reviews", f"{len(df[df['Authenticity'] == 'Genuine'])} ({df[df['Authenticity'] == 'Genuine'].shape[0]/len(df)*100:.1f}%)", "#00AA45"),
            ("Avg Sentiment", f"{df['Sentiment Polarity'].mean():.2f}", "#FFA500")
        ]

        
        for i, (title, value, color) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="animation-delay: {i*0.1}s;">
                    <h3>{title}</h3>
                    <div class="metric-value" style="background: linear-gradient(135deg, {color} 0%, {color} 100%);">{value}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Rating Distribution
        st.markdown("## 📈 Rating Distribution")
        rating_dist = df['Rating'].value_counts().sort_index()
        fig_rating = px.bar(rating_dist, 
                           labels={'index': 'Star Rating', 'value': 'Count'},
                           color=rating_dist.index,
                           color_continuous_scale='Bluered')
        fig_rating.update_layout(showlegend=False)
        st.plotly_chart(fig_rating, use_container_width=True)
        
        # Sentiment Analysis Tabs
        st.markdown("## 🔍 Advanced NLP Analysis")
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Categories", "Patterns", "Linguistic Features"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                sentiment_dist = df['Sentiment'].value_counts()
                fig_sentiment = px.pie(sentiment_dist, 
                                      names=sentiment_dist.index,
                                      title="Review Sentiment Distribution",
                                      color=sentiment_dist.index,
                                      color_discrete_map={'Positive': '#00AA45', 'Negative': '#FF4D4D', 'Neutral': '#FFA500'})
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                auth_dist = df['Authenticity'].value_counts()
                fig_auth = px.pie(auth_dist, 
                                  names=auth_dist.index,
                                  title="Review Authenticity Distribution",
                                  color=auth_dist.index,
                                  color_discrete_map={'Genuine': '#00AA45', 'Likely Genuine': '#7FDBFF', 'Potentially Fake': '#FF4D4D'})
                st.plotly_chart(fig_auth, use_container_width=True)
                
            # Sentiment vs Authenticity
            st.markdown("### Sentiment vs Authenticity")
            fig_sent_auth = px.sunburst(df, path=['Authenticity', 'Sentiment'], 
                                       color='Sentiment',
                                       color_discrete_map={'Positive': '#00AA45', 'Negative': '#FF4D4D', 'Neutral': '#FFA500'})
            st.plotly_chart(fig_sent_auth, use_container_width=True)
        
        with tab2:
            # Category Distribution
            st.markdown("### Review Categories")
            category_dist = df['Category'].value_counts()
            fig_category = px.bar(category_dist, 
                                labels={'index': 'Category', 'value': 'Count'},
                                title="What Aspects Are Customers Talking About?")
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Category by Rating
            st.markdown("### Categories by Rating")
            fig_cat_rating = px.box(df, x='Category', y='Rating', 
                                  color='Category',
                                  title="How Do Ratings Vary by Category?")
            st.plotly_chart(fig_cat_rating, use_container_width=True)
        
        with tab3:
            fig_patterns = px.scatter(df, x='Rating', y='Authenticity Score', 
                                    color='Sentiment',
                                    size='Helpful Votes',
                                    color_discrete_map={'Positive': '#00AA45', 'Negative': '#FF4D4D', 'Neutral': '#FFA500'},
                                    hover_data=['Reviewer', 'Review', 'Category'],
                                    title="Rating vs Authenticity Score by Sentiment")
            st.plotly_chart(fig_patterns, use_container_width=True)
            
            # Sentiment Polarity vs Subjectivity
            st.markdown("### Sentiment Analysis")
            fig_sent_polar = px.scatter(df, x='Sentiment Polarity', y='Sentiment Subjectivity',
                                       color='Authenticity',
                                       hover_data=['Review'],
                                       title="Sentiment Polarity vs Subjectivity")
            st.plotly_chart(fig_sent_polar, use_container_width=True)
        
        with tab4:
            # Analyze linguistic features for a sample review
            st.markdown("### Linguistic Feature Analysis")
            sample_review = st.selectbox("Select a review to analyze:", 
                                       df['Review'].sample(min(5, len(df))).tolist())
            
            if sample_review:
                doc = nlp(sample_review)
                
                # Display entities
                st.markdown("#### Named Entities")
                html = displacy.render(doc, style="ent")
                st.markdown(html, unsafe_allow_html=True)
                
                # Display POS tags
                st.markdown("#### Part-of-Speech Tags")
                pos_tags = [(token.text, token.pos_, token.tag_) for token in doc]
                pos_df = pd.DataFrame(pos_tags, columns=["Token", "POS", "Tag"])
                st.dataframe(pos_df)
                
                # Display dependency parse
                st.markdown("#### Dependency Parse")
                dep_svg = displacy.render(doc, style="dep", options={'compact': True})
                st.image(f"data:image/svg+xml;base64,{base64.b64encode(dep_svg.encode()).decode()}")
        
        # Highlighted Reviews
        st.markdown("## 🔎 Highlighted Reviews")
        sample_reviews = df.sample(min(5, len(df)))
        
        for _, row in sample_reviews.iterrows():
            sentiment_class = "review-genuine" if row['Authenticity'] == "Genuine" else "review-fake"
            tag_class = "tag-genuine" if row['Authenticity'] == "Genuine" else "tag-fake"
            
            st.markdown(f"""
            <div class="review-card {sentiment_class}" style="animation-delay: 0.3s;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div>
                        <span class="review-tag {tag_class}">{row['Authenticity']}</span>
                        <span class="review-tag" style="background: #E6F7EE; color: #6e8efb;">{row['Category']}</span>
                        <span style="font-weight: 600; color: #2a3f5f;">{row['Reviewer']}</span>
                        <span style="color: #666; font-size: 0.9rem;">from {row['Location']}</span>
                    </div>
                    <div style="display: flex; align-items: center;">
                        {"★" * int(row['Rating'])}{"☆" * (5 - int(row['Rating']))}
                        <span style="margin-left: 10px; font-weight: 600; color: #2a3f5f;">{row['Rating']}/5</span>
                    </div>
                </div>
                <p style="margin: 10px 0; font-size: 1rem; line-height: 1.6;">"{row['Review']}"</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #666; font-size: 0.9rem;">{row['Date']} • {row['Verified Purchase']} Verified</span>
                    <span style="color: #666; font-size: 0.9rem;">{row['Helpful Votes']} helpful votes</span>
                </div>
                <div style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.03); border-radius: 8px;">
                    <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>Sentiment:</strong> {row['Sentiment']} (Polarity: {row['Sentiment Polarity']:.2f}, Subjectivity: {row['Sentiment Subjectivity']:.2f})</p>
                    <p style="margin: 5px 0 0 0; color: #666; font-size: 0.9rem;"><strong>Authenticity:</strong> Score: {row['Authenticity Score']}/100 - {row['Authenticity Reason']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Time Series Analysis
        if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
            st.markdown("## ⏳ Review Activity Over Time")
            time_series = df.groupby(['Date', 'Authenticity']).size().unstack().fillna(0)
            fig_time = px.line(time_series, 
                              labels={'value': 'Number of Reviews', 'Date': 'Date'},
                              color_discrete_map={'Genuine': '#00AA45', 'Likely Genuine': '#7FDBFF', 'Potentially Fake': '#FF4D4D'})
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Full Data Table
        st.markdown("## 📋 Complete Review Data")
        st.dataframe(df, use_container_width=True)
        
        # Export Section
        st.markdown("## 💾 Export Analysis Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="amazon_review_analysis.csv",
                mime="text/csv",
                key="download_csv"
            )
        
        with col2:
            # For Excel download, we need to save to a buffer first
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            excel_data = output.getvalue()
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name="amazon_review_analysis.xlsx",
                mime="application/vnd.ms-excel",
                key="download_excel"
            )
        
        with col3:
            st.button("Generate PDF Report", key="pdf_report")

# ========== Sidebar with Enhanced UI ==========
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="position: relative; display: inline-block;">
                <div class="blob" style="top: -20px; left: -20px; width: 100px; height: 100px;"></div>
                <svg width="100" height="100" viewBox="0 0 100 100" style="position: relative; z-index: 1;">
                    <circle cx="50" cy="50" r="45" fill="#6E8EFB"/>
                    <path d="M70 40L45 65L30 50" stroke="white" stroke-width="8" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
                </svg>
            </div>
            <h3 style="margin-top: 15px;">Review Insight Pro</h3>
            <p style="color: #666; font-size: 0.9rem;">Advanced Amazon Review Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔍 How It Works")
        st.markdown("""
        <div class="timeline">
            <div class="timeline-item">
                <h4 style="margin: 0 0 5px 0;">1. Paste Product URL</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Copy any Amazon product link and paste it above</p>
            </div>
            <div class="timeline-item">
                <h4 style="margin: 0 0 5px 0;">2. Analyze Reviews</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Our NLP models process all customer reviews</p>
            </div>
            <div class="timeline-item">
                <h4 style="margin: 0 0 5px 0;">3. Get Insights</h4>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Discover sentiment, authenticity, and key topics</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 NLP Features")
        st.markdown("""
        <div class="accordion">
            <div class="accordion-item">
                <div class="accordion-header">Sentiment Analysis</div>
                <div class="accordion-content">
                    Uses TextBlob to determine polarity (-1 to 1) and subjectivity (0 to 1) of each review.
                </div>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">Content Categorization</div>
                <div class="accordion-content">
                    Identifies key aspects like Quality, Performance, Design, Value, etc. using spaCy.
                </div>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">Authenticity Detection</div>
                <div class="accordion-content">
                    Analyzes linguistic patterns to detect potentially fake reviews.
                </div>
            </div>
            <div class="accordion-item">
                <div class="accordion-header">Linguistic Features</div>
                <div class="accordion-content">
                    Examines POS tags, named entities, and dependency trees for deeper analysis.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🔔 Tips for Best Results")
        st.markdown("""
        - Analyze products with at least 20 reviews
        - Compare with similar products
        - Look for verified purchase reviews
        - Check review dates for patterns
        - Examine the linguistic features tab for deeper insights
        """)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p>This tool provides estimates based on NLP analysis.</p>
            <p>Results should be considered as one of several factors.</p>
            <p style="margin-top: 20px;">© 2023 Review Insight Pro</p>
        </div>
        """, unsafe_allow_html=True)

# ========== Run the App ==========
if __name__ == "__main__":
    render_sidebar()
    main()
    
    # Add some floating elements for visual appeal
    st.markdown("""
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: -1;">
        <div class="blob floating" style="width: 150px; height: 150px; background: rgba(110,142,251,0.1);"></div>
    </div>
    <div style="position: fixed; top: 100px; left: -50px; z-index: -1;">
        <div class="blob floating" style="width: 200px; height: 200px; background: rgba(167,119,227,0.1); animation-delay: 2s;"></div>
    </div>
    """, unsafe_allow_html=True)