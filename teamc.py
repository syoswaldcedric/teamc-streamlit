import streamlit as st
# navigation dependecy
from nltk.stem.wordnet import WordNetLemmatizer
from streamlit_option_menu import option_menu
import pandas as pd
import string
import joblib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import streamlit as st
import seaborn as sns
from wordcloud import WordCloud
from PIL import Image
import numpy as np

details = {'sdgLables': ["No poverty", "Zero Hunger", "Good Health and well-being",
                         "Quality Education", "Gender equality", "Clean water and sanitation",
                         "Affordable and clean energy", "Decent work and economic growth",
                         "Industry, Innovation and Infrustructure", "Reduced Inequality",
                         "Sustainable cites and communities", "Responsible consumption and production",
                         "Climate Action", "life below water", "Life on land", "Peace , Justice and strong institutions",
                         "Partnership for the goals"],
           'sdg': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
           'Description': [
               'SDG 1 seeks to ‘end poverty in all its forms everywhere’, specifically by ensuring that the poor are covered by social protection systems; by securing their rights to economic resources, access to basic services and property ownership; and by building their resilience to economic, social and environmental shocks. ',
               'SDG 2 seeks to ‘end hunger, achieve food security and nutrition and promote sustainable agriculture’',
               'SDG 3 seeks to ensure healthy lives and promote well-being for all at all ages',
               'SDG 4 seeks toEnsure inclusive and equitable quality education and promote lifelong learning opportunities for all',
               'SDG 5 seeks to Achieve gender equality and empower all women and girls',
               'SDG 6 seeks to Ensure availability and sustainable management of water and sanitation for all',
               'SDG 7 seeks to Ensure access to affordable, reliable, sustainable and modern energy for all',
               'SDG 8 seeks to Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all',
               'SDG 9 seeks to Build resilient infrastructure, promote inclusive and sustainable industrialization',
               'SDG 10 seeks to Reduce inequality within and among countries',
               'SDG 11 seeks to Reduce inequality within and among countries',
               'SDG 12 seeks to Ensure sustainable consumption and production patterns',
               'SDG 13 seeks to Take urgent action to combat climate changes and its impact',
               'SDG 14 seeks to Conserve and sustainably use the oceans, seas and marine resources for sustainable development',
               'SDG 15 seeks to Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification and half and reverse land degradation and halt biodiversity loss',
               'SDG 16 seeks to Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels',
               'SDG 17 seeks to Strengthen the means of implementation and revitalise the global partnership for sustainable development'],
           'Poster': ['https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/1.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/2.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/3.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/4.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/5.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/6.jpg?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/7.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/8.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/9.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/10.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/11.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/12.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/13.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/14.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/15.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/16.png?raw=true',
                      'https://github.com/macchime/SDG_LOGOS/blob/main/sdg_logos/17.png?raw=true'],
           }

# creating a Dataframe object
df = pd.DataFrame(details)
df.to_csv('sdg_details.csv')


@st.cache(hash_funcs={pd.DataFrame: lambda _: None})
def import_dataset(path: str) -> pd.DataFrame:
    """
    Data read_csvtion, importing `csv` file only
    Args:
        path: string like `file` address in the directory
    Return:
        df: (Dataframe) output imported csv file as pandas Dataframe
    """

    train_df = pd.read_csv(path, sep=',')
    return train_df


def clean(text: str):
    """Converts apostrophe suffixes to words, replace webpage links with url,
    annotate hashtags and mentions, remove a selection of punctuation, and convert all words to lower case.
    Args:
        df (DataFrame): dataframe containing 'text' column to convert
    Returns:
        df (DataFrame): dataframe with converted 'text' column
    """
    lemmatizer = WordNetLemmatizer()

    def word_lemma(words, lemmatizer):
        lemma = [lemmatizer.lemmatize(word) for word in words]
        return ''.join([l for l in lemma])

    def remove_extras(post):
        punc_numbers = string.punctuation + '0123456789'
        return ''.join([l for l in post if l not in punc_numbers])

    # Lower case
    text = text.lower()
    # Removal of Punctuation
    text = remove_extras(text)
    text_split = text.split()
    stops = set(stopwords.words("english"))
    clean_text = ' '.join(
        [t for t in text_split if t not in stops and len(t) > 2])
    text = word_lemma(clean_text, lemmatizer)
    return text


clf = joblib.load('linsvc.pkl')
vectorizer = joblib.load('vec.pkl')


def multilabel_hardCoding(df: pd.DataFrame, pred: np.array, sdglabel: dict, thres: float = 0.5,
                          coln: str = 'multilabel_sdg', str_typ: bool = True, nn: bool = True, list_typ: bool = None,
                          dict_typ: bool = None) -> pd.DataFrame:
    """
    Args:
        df: DataFrame we are trying to make a prediction on
        pred: numpy array from model prediction (likely from nueral network)
        sdgLables: a dictionay contain all the name of the sdg goals and their designated number
        thres: a float to represent percentage cut-off for a valid prediction in `pred` array
        coln: a string to name the new column that would be created for `multilabe
    Return:
        df: DataFrame with multilabel additional column
    """
    pred_list = []
    for x_1 in pred:
        result = np.where(x_1 >= thres)[0]
        count = 0
        for x_2 in result:
            count += x_1[x_2]
        count_list = []
        for x_3 in result:
            count_list.append(round(x_1[x_3] / count, 2))
        final_str = []
        final_list = []
        final_dict = {}
        for x_4, y in zip(result, count_list):
            if nn != True:
                x_4 += 1
            if list_typ == True:
                final_list.append((sdgLables[x_4], str(y)))
            elif dict_typ == True:
                final_dict[sdgLables[x_4]] = str(y)
            else:
                final_str.append((f'{sdgLables[x_4]}: {str(y)}'))
        if final_str:
            pred_list.append(', '.join(final_str))
        elif final_list:
            pred_list.append(final_list)
        else:
            pred_list.append(final_dict)

    df[coln] = pred_list

    return df


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def single_text_pred(text: str, vect=vectorizer, model=clf):
    """
    Args:
        text: (String) text/article to be predicted
        df: (DataFrame) to be use in vectorizing against transforming the `text`
        model: (best with Nueral Network) model to be use in actual text prediction
    Return:
        matplotlib visual displaying the proportion of each label of the text
        :param model:
        :param vect:
    """
    #
    clean_text = clean(text)
    text_vec = [clean_text]

    # vectorizing the test dataframe
    text_vec = vect.transform(text_vec).toarray()

    # predicting for the text
    predict_proba_dist = model.decision_function(text_vec)
    pred_text = []
    for eachArr in predict_proba_dist:
        pred_text.append(softmax(eachArr))
    # pred_text = model.predict(text_vec)

    # creating new dataFrame
    df_new = pd.DataFrame.from_dict(
        {1: text}, orient='index', columns=['text'])
    # hardCoding multilabel in `dict` format
    df_final = multilabel_hardCoding(
        df_new, pred_text, sdgLables, dict_typ=True, thres=0.09, nn=False)
    multilabel_sdg = df_final.multilabel_sdg.iloc[0]

    return multilabel_sdg


def multiple_text_pred(df: pd.DataFrame, vect=vectorizer, model=clf):
    """
    Args:
        text: (String) text/article to be predicted
        df: (DataFrame) to be use in vectorizing against transforming the `text`
        model: (best with Nueral Network) model to be use in actual text prediction
    Return:
        matplotlib visual displaying the proportion of each label of the text
        :param model:
        :param vect:
    """
    #
    df['clean_text'] = df['text'].apply(clean)

    # vectorizing the test dataframe
    text_vec = vect.transform(df['clean_text']).toarray()

    # predicting for the text
    predict_proba_dist = model.decision_function(text_vec)
    pred_text = []
    for eachArr in predict_proba_dist:
        pred_text.append(softmax(eachArr))

    # hardCoding multilabel in `dict` format
    pred_df = multilabel_hardCoding(
        data[['text']], pred_text, sdgLables, thres=0.09, nn=False)
    return pred_df


def Poster(pred, percentage):
    """Takes the Name or title of a movie and displays the poster, title, movie overview and mean rating
    with a scale of 0 to 10 of the selected movie
    Prameters
    ---------

    """
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'''
            <a href="https://sdgs.un.org/goals/goal{pred['sdg']}">
               <td><img src={pred['Poster']} style='border-radius:10px; width:200px; height:100px;'></td>
            </a>''',
                    unsafe_allow_html=True
                    )
    with col2:
        st.subheader(f"{pred['sdgLables']} with {float(percentage) * 100}%")
        st.write(pred['Description'])


def bag_of_words_count(words, word_dict={}):
    """ this function takes in a list of words and returns a dictionary
        with each word as a key, and the value represents the number of
        times that word appeared"""
    words = words.split()
    for word in words:
        if word in word_dict.keys():
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    return word_dict


# @st.cache
def title_extract(df: pd.DataFrame, target_coln: str = 'text'):
    """This use to compile all the word and there frequency
    in the individual category found in a particular column"""
    count = 0
    result = {}
    type_labels = df[target_coln].unique()
    texts = {}
    df_grp = df.groupby(target_coln)
    for pp in type_labels:
        texts[pp] = {}
        for row in df_grp.get_group(pp)['clean_text']:
            texts[pp] = bag_of_words_count(row, texts[pp])
    return texts


def target_distribution(df: pd.DataFrame, target_label: dict, text_title: str = 'Historical Data') -> plt.show:
    """

    """
    df['SDG_Labels'] = df['sdg'].map(sdgLables)
    count = df.groupby("SDG_Labels").count()["text"].reset_index()

    # labels and size for the classes
    labels, sizes = target_label.values(), count.sort_index()['text']
    labels = [x for x in labels]

    # Declare pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig, ax1 = plt.subplots(figsize=(20, 15))
    graph = sns.barplot(ax=ax1, x=sizes, y=labels)
    sns.set(font_scale=5)
    plt.title(f"Distribution of the SDGs for {text_title}")
    plt.ylabel('Frequency')
    plt.xlabel('SDGs')
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    return st.pyplot(fig, use_container_width=True)


def wordcloud(word_bag, sdg_value, sdg_picker='Show All', ):
    # wordcloud plot
    # Most occurring words of the train set
    if sdg_picker == 'Show All':
        wordcloud = WordCloud(background_color='black', width=3200,
                              height=1700, random_state=46).generate(str(word_bag))
    else:
        wordcloud = WordCloud(background_color='black', width=3200,
                              height=1700, random_state=46).generate(str(word_bag[sdg_value]))
    # Displaying the word cloud image:
    plt.title(f"Important keywords in the Articles for {sdg_picker}")
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return st.pyplot(plt, use_container_width=True)


def country_extract(df: pd.DataFrame, country_num: int = 10, text_title: str = 'Historical Data') -> plt.show:
    """
    Args:
        :param text_title:
        :param df: DataFrame where we want to extract countries from
        :param country_num: (integer) how many country do want to see there distribution
    return:
        a visual of countries and there frequency

    """
    temp_df = df.copy()
    # Implementation of the creating_country function
    temp_df = temp_df['country'].value_counts(
    ).to_frame().sort_values('country', ascending=False)
    # temp_df.drop('', inplace=True)
    temp_df.dropna(subset=['country'], how='all', inplace=True)
    temp_df = temp_df[:country_num]

    # A visual check of the distribution of the sdg between the classes
    fig, ax1 = plt.subplots(figsize=(30, 15))
    graph = sns.barplot(ax=ax1, x=temp_df.index, y='country', data=temp_df)
    # graph.set_xticklabels(graph.get_xticklabels())
    sns.set(font_scale=5)
    plt.title(f"Distribution of the countries involve in {text_title}")
    plt.ylabel('Frequency')
    plt.xlabel('Countries')
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=25)
    plt.tick_params(axis='x', rotation=65)
    return st.pyplot(fig, use_container_width=True)


# # Data loading
sdg_df = import_dataset('modified_train.csv')

# Media loading
logo = Image.open("sdg_logo.png")

# other variables
# creating a dictionary for our target variable
sdgLables = {1: "No poverty", 2: "Zero Hunger", 3: "Good Health and well-being", 4: "Quality Education",
             5: "Gender equality", 6: "Clean water and sanitation", 7: "Affordable and clean energy",
             9: "Industry, Innovation and Infrastructure", 8: "Decent work and economic growth",
             10: "Reduced Inequality", 13: "Climate Action", 11: "Sustainable cites and communities",
             12: "Responsible consumption and production", 14: "life below water", 15: "Life on land"}

res_sdgLables = dict((v, k) for k, v in sdgLables.items())
sdgs = ['Show All'] + [sdgLables[x] for x in sdgLables.keys()]
word_extract = title_extract(df=sdg_df, target_coln='sdg')

# team = st.radio(['Team A', 'Team B', 'Team C'])
team = st.sidebar.radio(
    "Teams", ["Team A", "Team B", "Team C"])
if team == "Team C":
    with st.sidebar:
        selection = option_menu(
            menu_title="Main Menu",
            options=["Prediction", "Visualisation",
                     "Contact us", "White Paper"],
            icons=["emoji-expressionless",
                   "robot", "phone", "book"],
            menu_icon="cast",
            default_index=0
        )
    if selection == 'Prediction':
        st.title('SHELL SDG GOALS PREDICTION')
        # Displays nme and image of sdgs
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal1">
                   <td><img src={df.iloc[0]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col2:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal2">
                   <td><img src={df.iloc[1]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col3:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal3">
                   <td><img src={df.iloc[2]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col4:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal4">
                   <td><img src={df.iloc[3]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col5:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal5">
                   <td><img src={df.iloc[4]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col6:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal6">
                   <td><img src={df.iloc[5]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )

        with col1:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal7">
                   <td><img src={df.iloc[6]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col2:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal8">
                   <td><img src={df.iloc[7]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col3:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal9">
                   <td><img src={df.iloc[8]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col4:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal10">
                   <td><img src={df.iloc[9]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col5:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal11">
                   <td><img src={df.iloc[10]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )

        with col6:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal12">
                   <td><img src={df.iloc[11]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )

        with col1:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal13">
                   <td><img src={df.iloc[12]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col2:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal14">
                   <td><img src={df.iloc[13]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col3:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal15">
                   <td><img src={df.iloc[14]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col4:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal16">
                   <td><img src={df.iloc[15]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )
        with col5:
            st.markdown(f'''
                <a href="https://sdgs.un.org/goals/goal17">
                   <td><img src={df.iloc[16]['Poster']} style='border-radius:10px; width:100px; height:100px;'></td>
                </a>''',
                        unsafe_allow_html=True
                        )

        st.write(
            'To understand the goals towards which Shell is working select on of the options')

        # prediction Option
        option = st.selectbox(
            "Choos an option for prediction",
            ['Enter a Text', 'Upload dataset',
             'Upload PDF', 'Collect from a website']
        )

        st.write()

        # text prediction
        if option == 'Enter a Text':
            # dictionary with list object in values
            text_input = st.text_area(
                "Type or paste a sdg related text or article in the box", )

            pred = single_text_pred(text_input, vectorizer, clf)

            if st.button('Make prediction'):
                for name, perc in pred.items():
                    st.write("")
                    pred_num = res_sdgLables[name]
                    df1 = df.iloc[pred_num - 1]
                    Poster(df1, perc)

        # upLoad doc
        if option == 'Upload PDF':
            data = st.file_uploader(
                'Load a pdf file containing the text to be predicted called TEXT ')
            pred = st.button('Predict')
            if pred:
                st.write('This is SDG 1: ')

        # Load data
        if option == 'Upload dataset':
            data = st.file_uploader(
                'Load a CSV file having the text to be predicted called TEXT ')
            if data is not None:
                data = pd.read_csv(data)
            pred = st.button('Predict')

            if pred:
                pred_df = multiple_text_pred(data)
                st.dataframe(pred_df.head())

                @st.cache
                def convert_df(df):
                    return df.to_csv().encode('utf-8')
                csv = convert_df(pred_df)

                st.download_button(
                    "Press to Download",
                    csv,
                    "file.csv",
                    "text/csv",
                    key='download-csv'
                )

        # Web scraping
        if option == 'Collect from a website':
            web = st.text_area(
                'Enter the website you want to collect the data from')
            pred = st.button('Predict')
            if pred:
                st.write('This is SDG 1: ')

    if selection == 'Visualisation':
        st.header(
            'UNITED NATIONS SUSTAINABLE DEVELOPMENT GOALS MULTI-LABEL CLASSIFIER')
        col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
        with col3:
            st.image(logo, width=400, use_column_width='never')

        st.header("Exploratory Data Analysis from Historical Survey Articles")
        target_distribution(df=sdg_df, target_label=sdgLables)

        # wordcloud plot
        sdg_picker = st.selectbox("pick genre category", sdgs)
        sdg_value = None
        if sdg_picker != 'Show All':
            sdg_value = [k for k, v in sdgLables.items() if v == sdg_picker][0]
        wordcloud(word_extract, sdg_value, sdg_picker=sdg_picker)

        num = st.slider("Number of Countries", 2, 25, 5)
        country_extract(df=sdg_df, country_num=num, text_title=sdg_picker)

    # system support
    if selection == "Contact us":
        st.title("Contact us")
        st.write("The feedback of your experience with our system matters")

        col1, col2 = st.columns(2)
        with col1:

            st.subheader("Contact info")
            st.write("Cola Street, Near ATTC,")
            st.write("Adjacent Sociate Generale, Head Office,")
            st.write("Kokomlemle, P.O. Box AN0000, Accra")
            st.write("Telephone:+233 20 913 9674")
            st.write("WhatsApp:+233 55 234 6333")
            st.write("Email: syoswaldcedric@gmail.com")
            st.write("Website: https://explore-datascience.net/")
        with col2:
            st.subheader("Send Email")
            email = st.text_input("Enter your email")
            message = st.text_area("Enter your message")
            st.button("Send")
