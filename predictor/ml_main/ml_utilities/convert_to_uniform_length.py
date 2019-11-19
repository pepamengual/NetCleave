from sklearn.feature_extraction.text import CountVectorizer

def convert_to_uniform_length(training_table_texts):
    print("Converting kmer words into uniform length numerical vectors that represent counts for every kmer in the vocabulary...")
    cv = CountVectorizer()
    print("--> Fitting")
    fit_cv = cv.fit(training_table_texts)
    print("--> Transforming")
    data = cv.transform(training_table_texts)
    print("---> Shape of transformed data looks like the following...")
    print(data.shape)
    return data
