import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import panel as pn
import hvplot.pandas

# Enable Panel extensions and a responsive sizing mode for a more modern feel
pn.extension('tabulator', sizing_mode="stretch_width")

# Load the data
try:
    df = pd.read_csv('articles.csv')
except FileNotFoundError:
    print("articles.csv not found. Please run fetch_data.py first.")
    exit()

# --- Analysis and Visualization ---

# 1. Word frequency analysis for a given word
def plot_word_frequency(word):
    if not word:
        return pn.pane.Alert("Please enter a word to search.", alert_type="warning")
    
    word = word.lower()
    df['title_desc'] = (df['title'].fillna('').str.lower() + " " + 
                        df['description'].fillna('').str.lower())
    
    # Count occurrences of the word in each article's title and description
    df['word_count'] = df['title_desc'].apply(lambda x: x.count(word))
    
    # Group by source and sum the counts
    frequency_by_source = df.groupby('source')['word_count'].sum().reset_index()
    
    # Create the bar chart
    chart = frequency_by_source.hvplot.bar(
        x='source', 
        y='word_count', 
        height=400,
        title=f"Frequency of the word '{word}' by source",
        xlabel="Source",
        ylabel="Total Count"
    ).opts(xrotation=45)
    
    return chart

# 2. Top 10 most frequent words (excluding stopwords)
def get_top_words_by_source():
    # A simple list of Hungarian stopwords, can be extended
    stopwords = ['és', 'a', 'az', 'is', 'egy', 'vagy', 'nem', 'de', 'hogy', 'van', 'volt', 'lesz', 'meg', 'kell', 'csak', 'ez', 'azt', 'itt', 'ott', 'majd', 'mint', 'tovább', 'valamint', 'illetve', 'még', 'sem', 'sok', 'nagyon', 'több', 'így', 'úgy', 'akkor', 'amikor', 'ahol', 'ami', 'aki', 'amit', 'akik', 'annak', 'ennek', 'ezért', 'azért', 'mert', 'hanem', 'hiszen', 'pedig', 'szerint', 'alatt', 'fölött', 'között', 'nélkül', 'után', 'előtt', 'mellett', 'át', 'be', 'ki', 'le', 'fel', 'össze', 'vissza', 'el', 'megint', 'mindig', 'soha', 'talán', 'persze', 'biztos', 'valóban', 'tehát', 'végül', 'azaz', 'illetőleg', 'való', 'egyik', 'másik', 'minden', 'semmi', 'valaki', 'valami', 'senki', 'bár', 'habár', 'noha', 'jóllehet', 'bárcsak', 'vajon', 'ugye', 'íme', 'nos', 'igen', 'se']
    
    vectorizer = TfidfVectorizer(stop_words=stopwords, max_features=1000)
    
    # Combine title and description for analysis
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    
    top_words_cards = []
    
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        
        if source_df.empty or source_df['text'].str.strip().empty:
            continue
            
        try:
            tfidf_matrix = vectorizer.fit_transform(source_df['text'])
            feature_names = vectorizer.get_feature_names_out()
            
            # Sum TF-IDF scores for each word across all documents for the source
            sum_tfidf = tfidf_matrix.sum(axis=0)
            words_freq = [(word, sum_tfidf[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            
            # Get top 10 words
            top_10 = words_freq[:10]
            top_10_df = pd.DataFrame(top_10, columns=['Word', 'TF-IDF Score'])
            
            # Create a modern-looking card with a Tabulator grid
            try:
                table = pn.widgets.Tabulator(
                    top_10_df,
                    height=260,
                    widths={'Word': 180, 'TF-IDF Score': 140},
                    layout='fit_data_table',
                    selectable=False,
                )
            except Exception:
                # Fallback to simple DataFrame widget if Tabulator resources are unavailable
                table = pn.widgets.DataFrame(top_10_df, height=260, width=340)

            card = pn.Card(
                table,
                title=f"Top 10 words • {source}",
                collapsed=False,
                styles={"minWidth": "320px"}
            )
            top_words_cards.append(card)
        except ValueError:
            # This can happen if a source has no text after stopword removal
            continue
            
    if not top_words_cards:
        return pn.pane.Alert("No top words could be computed for the sources.", alert_type="warning")

    return pn.GridBox(*top_words_cards, ncols=3)


# --- UI Components ---
word_input = pn.widgets.TextInput(name='Word', placeholder='Search for a word…')

# Reactive views
word_freq_view = pn.bind(plot_word_frequency, word=word_input)
top_words_view = get_top_words_by_source()

# Modern template-based layout
template = pn.template.FastListTemplate(
    title="RSS Feed Analysis",
    theme="dark",  # set default theme to dark; values: 'default', 'dark', 'system'
    header_background="#0F172A",  # slate-900
    accent="#3B82F6",  # blue-500
    sidebar=[
        pn.pane.Markdown("""
        ### Controls
        Type a word to analyze its frequency across news sources.
        """),
        word_input,
    ],
    main=[
        pn.Card(
            word_freq_view,
            title="Word Frequency Across Sources",
            collapsed=False,
        ),
        pn.Card(
            top_words_view,
            title="Top 10 Most Frequent Words by Source (TF-IDF)",
            collapsed=False,
        ),
    ],
)

# To display the dashboard
template.servable()
