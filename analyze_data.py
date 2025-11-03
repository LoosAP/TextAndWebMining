import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import panel as pn
import hvplot.pandas

pn.extension()

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
        return pn.pane.Markdown("Please enter a word to search.")
    
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
        width=600,
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
    
    top_words_tables = []
    
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
            
            # Create a table for the source
            table = pn.Column(
                pn.pane.Markdown(f"### Top 10 words for {source}"),
                pn.widgets.DataFrame(top_10_df, height=250, width=300)
            )
            top_words_tables.append(table)
        except ValueError:
            # This can happen if a source has no text after stopword removal
            continue
            
    return pn.GridBox(*top_words_tables, ncols=3)


# --- UI Components ---
word_input = pn.widgets.TextInput(name='Word', placeholder='Enter a word...')

# Bind the function to the input widget
dynamic_chart = pn.bind(plot_word_frequency, word=word_input)

# Dashboard Layout
dashboard = pn.Column(
    pn.pane.Markdown("# RSS Feed Analysis"),
    pn.Row(
        pn.Column(
            "## Word Frequency Across Sources",
            word_input,
            dynamic_chart
        ),
    ),
    pn.layout.Divider(),
    pn.Column(
        "## Top 10 Most Frequent Words by Source (TF-IDF)",
        get_top_words_by_source()
    )
)

# To display the dashboard
dashboard.servable()
