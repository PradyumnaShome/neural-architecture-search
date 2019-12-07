import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from ast import literal_eval
import pandas as pd
import numpy as np
import logging
import constants


def get_model_word_count(arch, vocabulary):
    """
    Given a vocabulary (list of various neural net layers), return a vector of counts in a neural architecture description.
    """
    vectorizer = CountVectorizer(vocabulary=vocabulary)

    arch = arch.replace('leaky_relu', 'leaky')
    logging.debug(arch)

    # Pad vocabulary items with spaces
    for layer in vocabulary:
        arch = arch.replace(layer, f" {layer} ")

    # Convert text to list
    arch_list = [arch]

    # Counts for each word in the vocabulary
    word_counts = vectorizer.fit_transform(arch_list).toarray()

    return word_counts


def get_all_arch(train_data):
    corpus_word_counts = []
    global_word_counts = [0 * len(constants.KNOWN_LAYERS)]
    arch_list = train_data["arch_and_hp"]

    for arch in arch_list:
        word_counts = get_model_word_count(arch=arch,
                                           vocabulary=constants.KNOWN_LAYERS)

        global_word_counts.append(word_counts)
        logging.debug(f"Word Counts: {word_counts}")

        corpus_word_counts = [
            x + y for x, y in zip(word_counts, corpus_word_counts)
        ]

    logging.info(f"Vocabulary: {constants.KNOWN_LAYERS}")
    logging.info(f"Corpus Word Counts: {corpus_word_counts}")
    logging.info(f"Global Word Counts: {global_word_counts}")
    return corpus_word_counts, global_word_counts


def loss_regression(train_data):
    """
    Uses linear regression on train / test loss to predict the final loss.
    """
    count_samples = len(train_data)
    val_losses = np.zeros(shape=(count_samples,
                                 constants.COUNT_PROVIDED_EPOCHS, 1))
    train_losses = np.zeros(shape=(count_samples,
                                   constants.COUNT_PROVIDED_EPOCHS, 1))

    models = [(LinearRegression(), LinearRegression())
              for i in range(len(train_data))]
    inputs = np.arange(constants.COUNT_PROVIDED_EPOCHS,
                       dtype='float').reshape(constants.COUNT_PROVIDED_EPOCHS,
                                              1)

    absolute_error = 0.0
    percentage_error = 0.0

    for index, sample in train_data.iterrows():
        logging.debug(f"Index: {index}")
        logging.debug(f"Sample: {sample}")

        for epoch in range(constants.COUNT_PROVIDED_EPOCHS):
            val_losses[index][epoch] = sample[f"val_losses_{epoch}"]
            train_losses[index][epoch] = sample[f"train_losses_{epoch}"]

        logging.debug(f"val_losses: {val_losses[index]}")
        logging.debug(f"train_losses: {train_losses[index]}")

        train_error_model = models[index][0]
        val_error_model = models[index][1]

        count_epochs = sample["epochs"]
        logging.debug(
            f"Count Epochs: {count_epochs}, Type: {type(count_epochs)}")

        train_error_model = train_error_model.fit(inputs, train_losses[index])
        val_error_model = val_error_model.fit(inputs, val_losses[index])

        x_pred = np.array([count_epochs]).reshape(-1, 1)

        train_loss_score = train_error_model.predict(x_pred)
        val_loss_score = train_error_model.predict(x_pred)

        actual_train_error = sample["train_loss"]
        predicted_train_error = train_loss_score

        current_percentage_error = abs(
            (actual_train_error - predicted_train_error[0, 0]) /
            actual_train_error)
        current_absolute_error = abs(
            (actual_train_error - predicted_train_error))

        logging.info(f"Current Percentage Error: {current_percentage_error}")

        absolute_error += current_absolute_error
        percentage_error += current_percentage_error

    mean_absolute_error = absolute_error / count_samples
    mean_percentage_error = percentage_error / count_samples

    logging.info(f"Mean Absolute Error: {mean_absolute_error}")
    logging.info(f"Mean Percentage Error: {mean_percentage_error}")

    return models


def gather_input_features(df):

    COUNT_SAMPLES = len(df)

    train_accs = []
    train_losses = []

    val_accs = []
    val_losses = []

    init_params = np.zeros(shape=(COUNT_SAMPLES,
                                  constants.COUNT_PROVIDED_EPOCHS))
    for i in range(COUNT_SAMPLES):
        init_params[i][0] = df["epochs"][i]
        init_params[i][1] = df["number_parameters"][i]
        init_params[i][2] = len(df["arch_and_hp"][i])

    for i in range(constants.COUNT_PROVIDED_EPOCHS):
        train_accs.append(df['train_accs_' + str(i)].tolist())
        train_losses.append(df['train_losses_' + str(i)].tolist())
        val_accs.append(df['val_accs_' + str(i)].tolist())
        val_losses.append(df['val_losses_' + str(i)].tolist())

    train_accs = np.array(list(map(list, zip(*train_accs))))
    train_losses = np.array(list(map(list, zip(*train_losses))))
    val_accs = np.array(list(map(list, zip(*val_accs))))
    val_losses = np.array(list(map(list, zip(*val_losses))))

    print(f"train_accs.shape = {train_accs.shape}")
    print(f"init_params.shape = {init_params.shape}")

    # Put all the training data into one vector, containing losses, and
    # accuracies for each sample
    train_features = np.concatenate((train_accs, train_losses, init_params),
                                    axis=1)
    val_features = np.concatenate((val_accs, val_losses, init_params), axis=1)

    return train_features, val_features


def dataset_regression(df):
    """
    Performs regression on the training dataset df, and returns a trained model train_loss, and val_loss.
    """

    train_error = df['train_error'].tolist()
    val_error = df['val_error'].tolist()

    train_input, val_input = gather_input_features(df)

    COUNT_ESTIMATORS = 90

    # Use a regression model, to fit a vector containing 50 losses and
    # accuracies each to the final training and validation loss values
    train_regressor = RandomForestRegressor(n_estimators=COUNT_ESTIMATORS,
                                            oob_score=True,
                                            random_state=0)
    train_regressor.fit(train_input, train_error)

    validation_regressor = RandomForestRegressor(n_estimators=COUNT_ESTIMATORS,
                                                 oob_score=True,
                                                 random_state=0)
    validation_regressor.fit(val_input, val_error)

    y_pred = train_regressor.predict(train_input)
    y_pred1 = validation_regressor.predict(val_input)

    print('Mean Absolute Error:',
          metrics.mean_absolute_error(train_error, y_pred))
    print('Mean Squared Error:',
          metrics.mean_squared_error(train_error, y_pred))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(train_error, y_pred)))
    print()
    print()
    print('Mean Absolute Error:',
          metrics.mean_absolute_error(val_error, y_pred1))
    print('Mean Squared Error:',
          metrics.mean_squared_error(val_error, y_pred1))
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(val_error, y_pred1)))

    return train_regressor, validation_regressor


def predict_on_test_data(train_regressor, val_regressor, test_data):
    train_input, val_input = gather_input_features(test_data)

    output_rows = len(test_data) // 2
    submission = pd.DataFrame(data={'id': [], 'Predicted': []})

    for i in range(output_rows // 2):
        # print(i)
        # Val
        row1 = {
            'id': f"test_{i}_val_error",
            'Predicted': val_regressor.predict([val_input[i]])[0]
        }
        # Train
        row2 = {
            'id': f"test_{i}_train_error",
            'Predicted': train_regressor.predict([train_input[i]])[0]
        }

        submission = submission.append(row1, ignore_index=True).append(
            row2, ignore_index=True)

    return submission


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(relativeCreated)1d-%(threadName)s-%(message)s')

    train_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")

    train_regressor, val_regressor = dataset_regression(train_data)

    submission = predict_on_test_data(train_regressor, val_regressor,
                                      test_data)

    csv = submission.to_csv(index=False)

    with open("submission.csv", "w+") as file:
        file.write(csv)


if __name__ == "__main__":
    main()
