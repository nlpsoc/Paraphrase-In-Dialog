"""
    legacy file
        not part of the final project
"""
from paraphrase import token_classifier
from paraphrase.interview_data import MediaSumProcessor

DEDUPLICATE_PATH = "/_model/deduplicate"  # seed 202
ALL_PATH = "/_model/all"  # seed 201

def main():
    # load the interviews that were annotated in the project (Total of 1304 unique interviews)
    interviews = MediaSumProcessor()
    interviews.load_interview_data()
    # print number of interviews
    print(f"Number of interviews: {len(interviews.interviews)}")

    # for each interview get the number of paraphrases according to our trained _model
    classifier = token_classifier.TokenClassifier(DEDUPLICATE_PATH)

    paraphrase_count = []
    average_len_guest = []
    total_len_guest = []
    total_len_host = []
    ratio_guest_host = []
    for index, interview in interviews.interviews.iterrows():
        print("Index: ", index)
        # if index > 100:
        #     break
        # get utterances, starting with the first guest utterance (hopefully after introductions)
        utterances = interviews.get_utterances(interview["id"])[3:-2]
        to_classify_list = []
        for i in range(0, len(utterances) - 1, 2):
            to_classify_list.append({
                "sentence1": utterances[i].split(),
                "sentence2": utterances[i + 1].split()
            })
        model_classifications = classifier.inference(to_classify_list)
        model_binary = [
            1 if (1 in model_classifications[i][0]) and (1 in model_classifications[i][1])  # both sentences highlighted
            else 0 for i in range(len(to_classify_list))]
        paraphrase_count.append(sum(model_binary))
        # calculate the average length of the second utterances
        average_len_guest.append(sum([len(utterance.split()) for utterance in utterances[0::2]]) / len(utterances[0::2]))
        # calculate the total length of the guest and host utterances
        total_len_host.append(sum([len(utterance.split()) for utterance in utterances[1::2]]))
        total_len_guest.append(sum([len(utterance.split()) for utterance in utterances[0::2]]))
        ratio_guest_host.append(total_len_guest[-1] / total_len_host[-1])

    print(f"Average number of paraphrases: {sum(paraphrase_count) / len(paraphrase_count)}")
    print(f"Average length of replies: {sum(average_len_guest) / len(average_len_guest)}")
    print(f"Average ratio of guest to host utterances: {sum(ratio_guest_host) / len(ratio_guest_host)}")

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    # Convert to NumPy array
    X = np.array(paraphrase_count).reshape(-1, 1)
    y = np.array(average_len_guest)

    # Split into train (60%), dev (20%), and test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Create and train the _model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on dev set
    y_dev_pred = model.predict(X_dev)
    # Predict on test set
    y_test_pred = model.predict(X_test)

    # Evaluation on dev set
    print("Development Set Evaluation")
    print("Mean Squared Error:", mean_squared_error(y_dev, y_dev_pred))
    print("R-squared Score:", r2_score(y_dev, y_dev_pred))

    # Plotting actual vs predicted for dev set
    plt.scatter(X_dev, y_dev, color='blue', label='Actual')
    plt.plot(X_dev, y_dev_pred, color='red', label='Predicted')
    plt.xlabel('Paraphrase Count')
    plt.ylabel('Average Length of Reply')
    plt.title('Actual vs Predicted (Dev Set)')
    plt.legend()
    plt.show()

    # Add a constant to the _model (intercept)
    X_train_sm = sm.add_constant(X_train)

    # Fit the _model
    model_sm = sm.OLS(y_train, X_train_sm).fit()

    # Print the summary
    print(model_sm.summary())


if __name__ == '__main__':
    main()
