import random
import unittest
import operator
import statistics
import numpy as np
import pandas as pd


def tournament_selection_parent_bins_edited(bin_scores, random_seed):
    random.seed(random_seed)

    # Choosing a random sample of 5% of the bin population or if that would be too small, choosing a sample of 50%
    if round(0.05 * len(bin_scores)) < 2:
        samplekeys = random.sample(bin_scores.keys(), round(0.5 * len(bin_scores)))
    else:
        samplekeys = random.sample(bin_scores.keys(), round(0.05 * len(bin_scores)))

    sample = {}
    for key in samplekeys:
        sample[key] = bin_scores[key]

    # Sorting the bins from best score to worst score
    sorted_bin_scores = dict(sorted(sample.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())

    # Choosing the parent bins and adding them to a list of parent bins
    parent_bins = [sorted_bin_list[0], sorted_bin_list[1]]

    return parent_bins, samplekeys, sample


def tournament_selection_parent_bins_original(bin_scores):
    # Choosing a random sample of 5% of the bin population or if that would be too small, choosing a sample of 50%
    if round(0.05 * len(bin_scores)) < 2:
        samplekeys = random.sample(bin_scores.keys(), round(0.5 * len(bin_scores)))
    else:
        samplekeys = random.sample(bin_scores.keys(), round(0.05 * len(bin_scores)))

    sample = {}
    for key in samplekeys:
        sample[key] = bin_scores[key]

    # Sorting the bins from best score to worst score
    sorted_bin_scores = dict(sorted(sample.items(), key=lambda item: item[1], reverse=True))
    sorted_bin_list = list(sorted_bin_scores.keys())

    # Choosing the parent bins and adding them to a list of parent bins
    parent_bins = [sorted_bin_list[0], sorted_bin_list[1]]

    return parent_bins, samplekeys, sample


def regroup_feature_matrix_edited(feature_list, feature_matrix, label_name, feature_bin_list, random_seed):
    # First deleting any bins that are empty
    bins_deleted = [x for x in feature_bin_list if x == []]
    feature_bin_list = [x for x in feature_bin_list if x != []]

    # Checking each pair of bins, if the bins are duplicates then one of the copies will be deleted
    no_duplicates = []
    num_duplicates = 0
    for bin_var in feature_bin_list:
        if bin_var not in no_duplicates:
            no_duplicates.append(bin_var)
        else:
            num_duplicates += 1

    feature_bin_list = no_duplicates

    # Calculate average length of nonempty bins in the population
    bin_lengths = [len(x) for x in feature_bin_list if len(x) != 0]
    replacement_length = round(statistics.mean(bin_lengths))

    # Replacing each deleted bin with a bin with random features
    np.random.seed(random_seed)
    random_seeds = np.random.randint(len(feature_list) * len(feature_list),
                                     size=((len(bins_deleted) + num_duplicates) * 2))
    for i in range(0, len(bins_deleted) + num_duplicates):
        random.seed(random_seeds[i])
        replacement = random.sample(feature_list, replacement_length)

        random.seed(random_seeds[len(bins_deleted) + num_duplicates + i])
        random_seeds_replacement = np.random.randint(len(feature_list) * len(feature_list),
                                                     size=2 * len(feature_bin_list))
        counter = 0
        while replacement in feature_bin_list:
            random.seed(random_seeds_replacement[counter])
            replacement = random.sample(feature_list, replacement_length)
            counter = counter + 1

        feature_bin_list.append(replacement)

    # Creating an empty data frame for the feature matrix with bins
    bins_df = pd.DataFrame()

    # Creating a list of 0s, where the number of 0s is the number of instances in the original feature matrix
    zero_list = [0] * len(feature_matrix.index)

    # Creating a dummy data frame
    dummy_df = pd.DataFrame()
    dummy_df['Zeros'] = zero_list
    # The list and dummy data frame will be used for adding later

    # For each feature group/bin, the values of the features in the bin will be summed to create a value for the bin
    # This will be used to create a feature matrix for the bins and a dictionary of binned feature groups

    count = 0
    binned_feature_groups = {}

    for i in range(0, len(feature_bin_list)):
        sum_column = dummy_df['Zeros']
        for j in range(0, len(feature_bin_list[i])):
            sum_column = sum_column + feature_matrix[feature_bin_list[i][j]]
        count = count + 1
        bins_df["Bin " + str(count)] = sum_column
        binned_feature_groups["Bin " + str(count)] = feature_bin_list[i]

    # Adding the class label to the data frame
    bins_df['Class'] = feature_matrix[label_name]
    return bins_df, binned_feature_groups


def regroup_feature_matrix_original(feature_list, feature_matrix, label_name, feature_bin_list):
    # First deleting any bins that are empty
    # Creating a list of bins to delete
    bins_to_delete = []
    for i in feature_bin_list:
        if not i:
            bins_to_delete.append(i)
    for i in bins_to_delete:
        feature_bin_list.remove(i)

    # The length of the bin will be equal to the average length of nonempty bins in the population
    bin_lengths = []
    for i in feature_bin_list:
        if len(i) > 0:
            bin_lengths.append(len(i))
    replacement_length = round(statistics.mean(bin_lengths))

    # Replacing each deleted bin with a bin with random features
    for i in range(0, len(bins_to_delete)):
        replacement = random.sample(feature_list, replacement_length)
        feature_bin_list.append(replacement)

    # Checking each pair of bins, if the bins are duplicates then one of the copies will be deleted
    seen = set()
    unique = []
    for x in feature_bin_list:
        srtd = tuple(sorted(x))
        if srtd not in seen:
            unique.append(x)
            seen.add(srtd)

    # Replacing each deleted bin with a bin with random features
    replacement_number = len(feature_bin_list) - len(unique)
    feature_bin_list = unique.copy()

    for i in feature_bin_list:
        if len(i) > 0:
            bin_lengths.append(len(i))
    replacement_length = round(statistics.mean(bin_lengths))

    for i in range(0, replacement_number):
        replacement = random.sample(feature_list, replacement_length)
        feature_bin_list.append(replacement)

    # Deleting duplicate features in the same bin and replacing them with random features
    for Bin in range(0, len(feature_bin_list)):
        unique = []
        for a in range(0, len(feature_bin_list[Bin])):
            if feature_bin_list[Bin][a] not in unique:
                unique.append(feature_bin_list[Bin][a])

        replace_number = len(feature_bin_list[Bin]) - len(unique)

        features_not_in_offspring = [item for item in feature_list if item not in feature_bin_list[Bin]]

        bin_replacement = unique.copy()
        if len(features_not_in_offspring) > replace_number:
            replacements = random.sample(features_not_in_offspring, replace_number)
        else:
            replacements = features_not_in_offspring.copy()
        bin_replacement.extend(replacements)

        feature_bin_list[Bin] = bin_replacement.copy()

    # Creating an empty data frame for the feature matrix with bins
    bins_df = pd.DataFrame()

    # Creating a list of 0s, where the number of 0s is the number of instances in the original feature matrix
    zero_list = []
    for a in range(0, len(feature_matrix.index)):
        zero_list.append(0)

    # Creating a dummy data frame
    dummy_df = pd.DataFrame()
    dummy_df['Zeros'] = zero_list
    # The list and dummy data frame will be used for adding later

    # For each feature group/bin, the values of the features in the bin will be summed to create a value for the bin
    # This will be used to create a feature matrix for the bins and a dictionary of binned feature groups

    count = 0
    binned_feature_groups = {}

    for i in range(0, len(feature_bin_list)):
        sum_column = dummy_df['Zeros']
        for j in range(0, len(feature_bin_list[i])):
            sum_column = sum_column + feature_matrix[feature_bin_list[i][j]]
        count = count + 1
        bins_df["Bin " + str(count)] = sum_column
        binned_feature_groups["Bin " + str(count)] = feature_bin_list[i]

    # Adding the class label to the data frame
    bins_df['Class'] = feature_matrix[label_name]
    return bins_df, binned_feature_groups


def crossover_and_mutation_edited(parent1_features, parent2_features, feature_list, crossover_probability,
                                  mutation_probability):
    offspring_list = []
    # Creating two lists for the offspring bins
    offspring1 = []
    offspring2 = []

    # CROSSOVER
    # Each feature in the parent bin will cross over based on the given probability (uniform crossover)

    # Creating two df for parent features and probability of crossover
    randnums1 = list(np.random.randint(0, 101, len(parent1_features)))
    crossover_threshold1 = list([crossover_probability * 100] * len(parent1_features))
    parent1_df = pd.DataFrame(parent1_features, columns=['Features'])
    parent1_df['Threshold'] = crossover_threshold1
    parent1_df['Rand_prob'] = randnums1

    randnums2 = list(np.random.randint(0, 101, len(parent2_features)))
    crossover_threshold2 = list([crossover_probability * 100] * len(parent2_features))
    parent2_df = pd.DataFrame(parent2_features, columns=['Features'])
    parent2_df['Threshold'] = crossover_threshold2
    parent2_df['Rand_prob'] = randnums2

    # Features with random probability less than the crossover probability will go to offspring 1.
    # The rest will go to offspring 2 for parent 1 and vice versa for parent 2.
    offspring1.extend(list(parent1_df.loc[parent1_df['Threshold'] > parent1_df['Rand_prob']]['Features']))
    offspring2.extend(list(parent1_df.loc[parent1_df['Threshold'] <= parent1_df['Rand_prob']]['Features']))
    offspring2.extend(list(parent2_df.loc[parent2_df['Threshold'] > parent2_df['Rand_prob']]['Features']))
    offspring1.extend(list(parent2_df.loc[parent2_df['Threshold'] <= parent2_df['Rand_prob']]['Features']))

    # Remove repeated features within each offspring
    offspring1 = list(set(offspring1))
    offspring2 = list(set(offspring2))

    # MUTATION
    #  (deletion and addition) only occurs with a certain probability on each feature in the original feature space

    # Creating a probability for mutation (addition) that accounts for the ratio between the feature list and the
    # size of the bin
    if len(offspring1) > 0 and len(offspring1) != len(feature_list):
        mutation_addition_prob1 = mutation_probability * (len(offspring1)) / (len(feature_list) - len(offspring1))
    elif len(offspring1) == 0 and len(offspring1) != len(feature_list):
        mutation_addition_prob1 = mutation_probability
    elif len(offspring1) == len(feature_list):
        mutation_addition_prob1 = 0

    if len(offspring2) > 0 and len(offspring2) != len(feature_list):
        mutation_addition_prob2 = mutation_probability * (len(offspring2)) / (len(feature_list) - len(offspring2))
    elif len(offspring2) == 0 and len(offspring2) != len(feature_list):
        mutation_addition_prob2 = mutation_probability
    elif len(offspring2) == len(feature_list):
        mutation_addition_prob2 = 0

    # Mutation: Deletion occurs on features with probability equal to the mutation parameter
    offspring1_df = pd.DataFrame(offspring1, columns=['Features'])
    mutation_threshold1 = list([mutation_probability * 100] * len(offspring1))
    rand1 = list(np.random.randint(0, 101, len(offspring1)))
    offspring1_df['Threshold'] = mutation_threshold1
    offspring1_df['Rand_prob'] = rand1

    offspring2_df = pd.DataFrame(offspring2, columns=['Features'])
    mutation_threshold2 = list([mutation_probability * 100] * len(offspring2))
    rand2 = list(np.random.randint(0, 101, len(offspring2)))
    offspring2_df['Threshold'] = mutation_threshold2
    offspring2_df['Rand_prob'] = rand2

    offspring1_df = offspring1_df.loc[offspring1_df['Threshold'] < offspring1_df['Rand_prob']]
    offspring1 = list(offspring1_df['Features'])

    offspring2_df = offspring2_df.loc[offspring2_df['Threshold'] < offspring2_df['Rand_prob']]
    offspring2 = list(offspring2_df['Features'])

    # Mutation: Addition occurs on this feature with probability proportional to the mutation parameter
    # The probability accounts for the ratio between the feature list and the size of the bin

    features_not_in_offspring1 = [item for item in feature_list if item not in offspring1]
    features_not_in_offspring2 = [item for item in feature_list if item not in offspring2]

    features_not_in_offspring1_df = pd.DataFrame(features_not_in_offspring1, columns=['Features'])
    mutation_addition_threshold1 = list([mutation_addition_prob1 * 100] * len(features_not_in_offspring1_df))
    rand1 = list(np.random.randint(0, 101, len(features_not_in_offspring1)))
    features_not_in_offspring1_df['Threshold'] = mutation_addition_threshold1
    features_not_in_offspring1_df['Rand_prob'] = rand1

    features_not_in_offspring2_df = pd.DataFrame(features_not_in_offspring2, columns=['Features'])
    mutation_addition_threshold2 = list([mutation_addition_prob2 * 100] * len(features_not_in_offspring2_df))
    rand2 = list(np.random.randint(0, 101, len(features_not_in_offspring2)))
    features_not_in_offspring2_df['Threshold'] = mutation_addition_threshold2
    features_not_in_offspring2_df['Rand_prob'] = rand2

    features_to_add1 = list(features_not_in_offspring1_df.loc[
                                features_not_in_offspring1_df['Threshold'] >= features_not_in_offspring1_df[
                                    'Rand_prob']]['Features'])
    features_to_add2 = list(features_not_in_offspring2_df.loc[
                                features_not_in_offspring2_df['Threshold'] >= features_not_in_offspring2_df[
                                    'Rand_prob']]['Features'])

    offspring1.extend(features_to_add1)
    offspring2.extend(features_to_add2)

    # Ensuring that each of the offspring is no more than c times the size of the other offspring
    c_constraint = 2
    while len(offspring1) > c_constraint * len(offspring2) | len(offspring2) > c_constraint * len(offspring1):
        if len(offspring1) > c_constraint * len(offspring2):
            min_features = int((len(offspring1) + len(offspring2)) / (c_constraint + 1)) + 1
            min_to_move = min_features - len(offspring2)
            max_to_move = len(offspring1) - min_features
            num_to_move = np.random.randint(min_to_move, max_to_move + 1)
            features_to_move = list(random.sample(offspring1, num_to_move))
            offspring1 = [x for x in offspring1 if x not in features_to_move]
            offspring2.extend(features_to_move)
        elif len(offspring2) > c_constraint * len(offspring1):
            min_features = int((len(offspring1) + len(offspring2)) / (c_constraint + 1)) + 1
            min_to_move = min_features - len(offspring1)
            max_to_move = len(offspring2) - min_features
            num_to_move = np.random.randint(min_to_move, max_to_move + 1)
            features_to_move = random.sample(offspring2, num_to_move)
            offspring2 = [x for x in offspring2 if x not in features_to_move]
            offspring1.extend(features_to_move)
        offspring1 = list(set(offspring1))
        offspring2 = list(set(offspring2))

    offspring_list.extend([offspring1, offspring2])

    return offspring_list


# Defining a function for crossover and mutation that creates n offspring based on crossover of selected parents
# n is the max number of bins (but not all the offspring will carry on, as the worst will be deleted in Step 2a next
# time)
def crossover_and_mutation_original(parent1_features, parent2_features, feature_list, crossover_probability,
                                    mutation_probability):
    offspring_list = []
    # Creating two lists for the offspring bins
    offspring1 = []
    offspring2 = []

    # CROSSOVER
    # Each feature in the parent bin will cross over based on the given probability (uniform crossover)
    for j in range(0, len(parent1_features)):
        if crossover_probability > random.random():
            offspring2.append(parent1_features[j])
        else:
            offspring1.append(parent1_features[j])

    for k in range(0, len(parent2_features)):
        if crossover_probability > random.random():
            offspring1.append(parent2_features[k])
        else:
            offspring2.append(parent2_features[k])

    # Ensuring that each of the offspring is no more than twice the size of the other offspring
    while len(offspring1) > 2 * len(offspring2):
        switch = random.choice(offspring1)
        offspring1.remove(switch)
        offspring2.append(switch)

    while len(offspring2) > 2 * len(offspring1):
        switch = random.choice(offspring2)
        offspring2.remove(switch)
        offspring1.append(switch)

    # MUTATION
    #  only occurs with a certain probability on each feature in the original feature space

    # Applying the mutation operation to the first offspring
    # Creating a probability for adding a feature that
    # accounts for the ratio between the feature list and the size of the bin
    if len(offspring1) > 0 and len(offspring1) != len(feature_list):
        mutation_addition_prob = mutation_probability * (len(offspring1)) / (len(feature_list) - len(offspring1))
    elif len(offspring1) == 0 and len(offspring1) != len(feature_list):
        mutation_addition_prob = mutation_probability
    elif len(offspring1) == len(feature_list):
        mutation_addition_prob = 0

    deleted_list = []
    # Deletion form of mutation
    for var in range(0, len(offspring1)):
        # Mutation (deletion) occurs on this feature with probability equal to the mutation parameter
        if mutation_probability > random.random():
            deleted_list.append(offspring1[var])

    for var in range(0, len(deleted_list)):
        offspring1.remove(deleted_list[var])

    # Creating a list of features outside the offspring
    features_not_in_offspring = [item for item in feature_list if item not in offspring1]

    # Addition form of mutation
    for var in range(0, len(features_not_in_offspring)):
        # Mutation (addiiton) occurs on this feature with probability proportional to the mutation parameter
        # The probability accounts for the ratio between the feature list and the size of the bin
        if mutation_addition_prob > random.random():
            offspring1.append(features_not_in_offspring[var])

    # Applying the mutation operation to the second offspring
    # Creating a probability for adding a feature that
    # accounts for the ratio between the feature list and the size of the bin
    if len(offspring2) > 0 and len(offspring2) != len(feature_list):
        mutation_addition_prob = mutation_probability * (len(offspring2)) / (len(feature_list) - len(offspring2))
    elif len(offspring2) == 0 and len(offspring2) != len(feature_list):
        mutation_addition_prob = mutation_probability
    elif len(offspring2) == len(feature_list):
        mutation_addition_prob = 0

    deleted_list = []
    # Deletion form of mutation
    for var in range(0, len(offspring2)):
        # Mutation (deletion) occurs on this feature with probability equal to the mutation parameter
        if mutation_probability > random.random():
            deleted_list.append(offspring2[var])

    for var in range(0, len(deleted_list)):
        offspring2.remove(deleted_list[var])

    # Creating a list of features outside the offspring
    features_not_in_offspring = [item for item in feature_list if item not in offspring2]

    # Addition form of mutation
    for var in range(0, len(features_not_in_offspring)):
        # Mutation (addiiton) occurs on this feature with probability proportional to the mutation parameter
        # The probability accounts for the ratio between the feature list and the size of the bin
        if mutation_addition_prob > random.random():
            offspring2.append(features_not_in_offspring[var])

    # CLEANUP
    # Deleting any repeats of an amino acid in a bin
    # Removing duplicates of features in the same bin that may arise due to crossover
    unique = []
    for a in range(0, len(offspring1)):
        if offspring1[a] not in unique:
            unique.append(offspring1[a])

    # Adding random features from outside the bin to replace the deleted features in the bin
    replace_number = len(offspring1) - len(unique)
    features_not_in_offspring = [item for item in feature_list if item not in offspring1]
    offspring1 = unique.copy()
    if len(features_not_in_offspring) > replace_number:
        replacements = random.sample(features_not_in_offspring, replace_number)
    else:
        replacements = features_not_in_offspring.copy()
    offspring1.extend(replacements)

    unique = []
    for a in range(0, len(offspring2)):
        if offspring2[a] not in unique:
            unique.append(offspring2[a])

    # Adding random features from outside the bin to replace the deleted features in the bin
    replace_number = len(offspring2) - len(unique)
    features_not_in_offspring = [item for item in feature_list if item not in offspring2]
    offspring2 = unique.copy()
    if len(features_not_in_offspring) > replace_number:
        replacements = random.sample(features_not_in_offspring, replace_number)
    else:
        replacements = features_not_in_offspring.copy()
    offspring2.extend(replacements)

    offspring_list.append(offspring1)
    offspring_list.append(offspring2)

    return offspring_list


class TestRARERandomFunctions(unittest.TestCase):
    def testTournamentSelectionSmallNumBinsEdited(self):
        bin_scores = {"Bin 1": 0.07, "Bin 2": 0.0010, "Bin 3": 0.045, "Bin 4": 0.023}
        parent_bins, samplekeys, sample = tournament_selection_parent_bins_edited(bin_scores, 10)
        self.assertEqual(len(samplekeys), 2)

    def testTournamentSelectionLargeNumBinsEdited(self):
        bin_scores = {"Bin 1": 0.34, "Bin 2": 0.123, "Bin 3": 0.678, "Bin 4": 0.765, "Bin 5": 0.7557, "Bin 6": 0.56,
                      "Bin 7": 0.1233, "Bin 8": -0.435, "Bin 9": 0.7569, "Bin 10": 0.92112,
                      "Bin 11": -0.786, "Bin 12": 0.2465, "Bin 13": -0.657, "Bin 14": -0.25981, "Bin 15": 0.8979,
                      "Bin 16": 0.8912, "Bin 17": 0.0051, "Bin 18": 0.561, "Bin 19": 0.6712, "Bin 20": 0.6129,
                      "Bin 21": 0.419, "Bin 22": -0.2210, "Bin 23": -.0903, "Bin 24": 0.00004, "Bin 25": 0.993245,
                      "Bin 26": 0.12393, "Bin 27": -0.002, "Bin 28": 0.12387, "Bin 29": 0.31649, "Bin 30": -.008,
                      "Bin 31": 0.999, "Bin 32": 0.926, "Bin 33": 0.193, "Bin 34": -0.188, "Bin 35": -0.913,
                      "Bin 36": -0.97, "Bin 37": 0.10, "Bin 38": .45, "Bin 39": -.23, "Bin 40": .77,
                      "Bin 41": -0.999, "Bin 42": -0.926, "Bin 43": -0.193, "Bin 44": 0.188, "Bin 45": 0.913,
                      "Bin 46": 0.97, "Bin 47": -0.00010, "Bin 48": -.45, "Bin 49": .00023, "Bin 50": -.077,
                      "Bin 51": .0999, "Bin 52": 0.88, "Bin 53": 0.1, "Bin 54": -0.1, "Bin 55": -0.876, "Bin 56": -0.9,
                      "Bin 57": 0.11, "Bin 58": .4, "Bin 59": -.2, "Bin 60": .7}
        parent_bins, samplekeys, sample = tournament_selection_parent_bins_edited(bin_scores, 10)
        self.assertEqual(len(samplekeys), 3)

    def testTournamentSelectionTopParentsEdited(self):
        bin_scores = {"Bin 1": 1, "Bin 2": 1, "Bin 3": 0.678, "Bin 4": 0.765, "Bin 5": 0.7557, "Bin 6": 0.56,
                      "Bin 7": 0.1233, "Bin 8": -0.435, "Bin 9": 0.7569, "Bin 10": 0.92112,
                      "Bin 11": -0.786, "Bin 12": 0.2465, "Bin 13": -0.657, "Bin 14": -0.25981, "Bin 15": 0.8979,
                      "Bin 16": 0.8912, "Bin 17": 0.0051, "Bin 18": 0.561, "Bin 19": 0.6712, "Bin 20": 0.6129,
                      "Bin 21": 0.419, "Bin 22": -0.2210, "Bin 23": -.0903, "Bin 24": 0.00004, "Bin 25": 0.993245,
                      "Bin 26": 0.12393, "Bin 27": -0.002, "Bin 28": 0.12387, "Bin 29": 0.31649, "Bin 30": -.008,
                      "Bin 31": 0.999, "Bin 32": 0.926, "Bin 33": 0.193, "Bin 34": -0.188, "Bin 35": -0.913,
                      "Bin 36": -0.97, "Bin 37": 0.10, "Bin 38": .45, "Bin 39": -.23, "Bin 40": .77,
                      "Bin 41": -0.999, "Bin 42": -0.926, "Bin 43": -0.193, "Bin 44": 0.188, "Bin 45": 0.913,
                      "Bin 46": 0.97, "Bin 47": -0.00010, "Bin 48": -.45, "Bin 49": .00023, "Bin 50": -.077,
                      "Bin 51": .0999, "Bin 52": 0.88, "Bin 53": 0.1, "Bin 54": -0.1, "Bin 55": -0.876, "Bin 56": -0.9,
                      "Bin 57": 0.11, "Bin 58": .4, "Bin 59": -.2, "Bin 60": .7}
        parent_bins, samplekeys, sample = tournament_selection_parent_bins_edited(bin_scores, 10)
        expected_parent1 = max(sample.items(), key=operator.itemgetter(1))[0]
        sample.pop(expected_parent1)
        expected_parent2 = max(sample.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(parent_bins, [expected_parent1, expected_parent2])

    def testTournamentSelectionSmallNumBinsOrig(self):
        bin_scores = {"Bin 1": 0.07, "Bin 2": 0.0010, "Bin 3": 0.045, "Bin 4": 0.023}
        parent_bins, samplekeys, sample = tournament_selection_parent_bins_original(bin_scores)
        self.assertEqual(len(samplekeys), 2)

    def testTournamentSelectionLargeNumBinsOrig(self):
        bin_scores = {"Bin 1": 0.34, "Bin 2": 0.123, "Bin 3": 0.678, "Bin 4": 0.765, "Bin 5": 0.7557, "Bin 6": 0.56,
                      "Bin 7": 0.1233, "Bin 8": -0.435, "Bin 9": 0.7569, "Bin 10": 0.92112,
                      "Bin 11": -0.786, "Bin 12": 0.2465, "Bin 13": -0.657, "Bin 14": -0.25981, "Bin 15": 0.8979,
                      "Bin 16": 0.8912, "Bin 17": 0.0051, "Bin 18": 0.561, "Bin 19": 0.6712, "Bin 20": 0.6129,
                      "Bin 21": 0.419, "Bin 22": -0.2210, "Bin 23": -.0903, "Bin 24": 0.00004, "Bin 25": 0.993245,
                      "Bin 26": 0.12393, "Bin 27": -0.002, "Bin 28": 0.12387, "Bin 29": 0.31649, "Bin 30": -.008,
                      "Bin 31": 0.999, "Bin 32": 0.926, "Bin 33": 0.193, "Bin 34": -0.188, "Bin 35": -0.913,
                      "Bin 36": -0.97, "Bin 37": 0.10, "Bin 38": .45, "Bin 39": -.23, "Bin 40": .77,
                      "Bin 41": -0.999, "Bin 42": -0.926, "Bin 43": -0.193, "Bin 44": 0.188, "Bin 45": 0.913,
                      "Bin 46": 0.97, "Bin 47": -0.00010, "Bin 48": -.45, "Bin 49": .00023, "Bin 50": -.077,
                      "Bin 51": .0999, "Bin 52": 0.88, "Bin 53": 0.1, "Bin 54": -0.1, "Bin 55": -0.876, "Bin 56": -0.9,
                      "Bin 57": 0.11, "Bin 58": .4, "Bin 59": -.2, "Bin 60": .7}
        parent_bins, samplekeys, sample = tournament_selection_parent_bins_original(bin_scores)
        self.assertEqual(len(samplekeys), 3)

    def testTournamentSelectionTopParentsOrig(self):
        bin_scores = {"Bin 1": 1, "Bin 2": 1, "Bin 3": 0.678, "Bin 4": 0.765, "Bin 5": 0.7557, "Bin 6": 0.56,
                      "Bin 7": 0.1233, "Bin 8": -0.435, "Bin 9": 0.7569, "Bin 10": 0.92112,
                      "Bin 11": -0.786, "Bin 12": 0.2465, "Bin 13": -0.657, "Bin 14": -0.25981, "Bin 15": 0.8979,
                      "Bin 16": 0.8912, "Bin 17": 0.0051, "Bin 18": 0.561, "Bin 19": 0.6712, "Bin 20": 0.6129,
                      "Bin 21": 0.419, "Bin 22": -0.2210, "Bin 23": -.0903, "Bin 24": 0.00004, "Bin 25": 0.993245,
                      "Bin 26": 0.12393, "Bin 27": -0.002, "Bin 28": 0.12387, "Bin 29": 0.31649, "Bin 30": -.008,
                      "Bin 31": 0.999, "Bin 32": 0.926, "Bin 33": 0.193, "Bin 34": -0.188, "Bin 35": -0.913,
                      "Bin 36": -0.97, "Bin 37": 0.10, "Bin 38": .45, "Bin 39": -.23, "Bin 40": .77,
                      "Bin 41": -0.999, "Bin 42": -0.926, "Bin 43": -0.193, "Bin 44": 0.188, "Bin 45": 0.913,
                      "Bin 46": 0.97, "Bin 47": -0.00010, "Bin 48": -.45, "Bin 49": .00023, "Bin 50": -.077,
                      "Bin 51": .0999, "Bin 52": 0.88, "Bin 53": 0.1, "Bin 54": -0.1, "Bin 55": -0.876, "Bin 56": -0.9,
                      "Bin 57": 0.11, "Bin 58": .4, "Bin 59": -.2, "Bin 60": .7}
        parent_bins, samplekeys, sample = tournament_selection_parent_bins_original(bin_scores)
        expected_parent1 = max(sample.items(), key=operator.itemgetter(1))[0]
        sample.pop(expected_parent1)
        expected_parent2 = max(sample.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(parent_bins, [expected_parent1, expected_parent2])

    def testCrossoverMutationCorrectCrossoverEdited(self):
        parent1_features = ['P_1', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'P_2', 'P_3']
        parent2_features = ['P_2', 'R_1', 'R_6', 'P_3', 'P_4', 'R_5', 'R_2', 'R_3']
        crossover_probability = 0.8

        # Creating two lists for the offspring bins
        offspring1 = []
        offspring2 = []

        # CROSSOVER
        # Each feature in the parent bin will cross over based on the given probability (uniform crossover)

        # Creating two df for parent features and probability of crossover
        randnums1 = [0, 12, 56, 30, 40, 80, 99, 5]
        crossover_threshold1 = list([crossover_probability * 100] * len(parent1_features))
        parent1_df = pd.DataFrame(parent1_features, columns=['Features'])
        parent1_df['Threshold'] = crossover_threshold1
        parent1_df['Rand_prob'] = randnums1

        randnums2 = [0, 90, 56, 30, 4, 80, 99, 5]
        crossover_threshold2 = list([crossover_probability * 100] * len(parent2_features))
        parent2_df = pd.DataFrame(parent2_features, columns=['Features'])
        parent2_df['Threshold'] = crossover_threshold2
        parent2_df['Rand_prob'] = randnums2

        # Features with random probability less than the crossover probability will go to offspring 1.
        # The rest will go to offspring 2 for parent1 and vice versa for parent 2.
        offspring1.extend(list(parent1_df.loc[parent1_df['Threshold'] > parent1_df['Rand_prob']]['Features']))
        offspring2.extend(list(parent1_df.loc[parent1_df['Threshold'] <= parent1_df['Rand_prob']]['Features']))
        offspring2.extend(list(parent2_df.loc[parent2_df['Threshold'] > parent2_df['Rand_prob']]['Features']))
        offspring1.extend(list(parent2_df.loc[parent2_df['Threshold'] <= parent2_df['Rand_prob']]['Features']))

        self.assertEqual(offspring1, list(['P_1', 'R_1', 'R_2', 'R_3', 'R_4', 'P_3', 'R_1', 'R_5', 'R_2']))
        self.assertEqual(offspring2, list(['R_5', 'P_2', 'P_2', 'R_6', 'P_3', 'P_4', 'R_3']))

    def testCrossOverMutationRemoveRepeatsEdited(self):
        offspring1 = list(['P_1', 'R_1', 'R_2', 'R_3', 'R_4', 'P_3', 'R_1', 'R_5', 'R_2'])
        offspring2 = list(['R_5', 'P_2', 'P_2', 'R_6', 'P_3', 'P_4', 'R_3'])

        # Remove repeated features within each offspring
        offspring1 = set(offspring1)
        offspring2 = set(offspring2)

        self.assertEqual(offspring1, {'P_1', 'R_1', 'R_2', 'R_3', 'R_4', 'P_3', 'R_5'})
        self.assertEqual(offspring2, {'R_5', 'P_2', 'R_6', 'P_3', 'P_4', 'R_3'})

    def testCrossoverMutationDeletionEdited(self):
        offspring1 = ['P_1', 'R_1', 'R_2', 'R_3', 'R_4', 'P_3', 'R_5']
        offspring2 = ['R_5', 'P_2', 'R_6', 'P_3', 'P_4', 'R_3']

        mutation_probability = 0.3
        # MUTATION (deletion and addition) only occurs with a certain probability on each feature in the
        # original feature space

        # Mutation: Deletion occurs on features with probability equal to the mutation parameter
        offspring1_df = pd.DataFrame(offspring1, columns=['Features'])
        mutation_threshold1 = list([mutation_probability * 100] * len(offspring1))
        rand1 = [0, 90, 56, 29, 4, 80, 99]
        offspring1_df['Threshold'] = mutation_threshold1
        offspring1_df['Rand_prob'] = rand1

        offspring2_df = pd.DataFrame(offspring2, columns=['Features'])
        mutation_threshold2 = list([mutation_probability * 100] * len(offspring2))
        rand2 = [0, 90, 56, 30, 4, 80]
        offspring2_df['Threshold'] = mutation_threshold2
        offspring2_df['Rand_prob'] = rand2

        offspring1_df = offspring1_df.loc[offspring1_df['Threshold'] < offspring1_df['Rand_prob']]
        offspring1 = list(offspring1_df['Features'])

        offspring2_df = offspring2_df.loc[offspring2_df['Threshold'] < offspring2_df['Rand_prob']]
        offspring2 = list(offspring2_df['Features'])

        self.assertEqual(offspring1, ['R_1', 'R_2', 'P_3', 'R_5'])
        self.assertEqual(offspring2, ['P_2', 'R_6', 'R_3'])

    def testCrossoverMutationAdditionEdited(self):
        offspring1 = ['P_1', 'R_1', 'R_2', 'R_3', 'R_4', 'P_3', 'R_5']
        offspring2 = ['R_5', 'P_2', 'R_6', 'P_3', 'P_4', 'R_3']

        feature_list = ['P_1', 'P_2', 'P_3', 'P_4', 'R_1', 'R_2', 'R_3', 'R_4', 'R_5', 'R_6']
        mutation_probability = 0.3

        # Creating a probability for mutation (addition) that accounts for the ratio between the feature list and the
        # size of the bin
        if len(offspring1) > 0 and len(offspring1) != len(feature_list):
            mutation_addition_prob1 = mutation_probability * (len(offspring1)) / (
                (len(feature_list) - len(offspring1)))
        elif len(offspring1) == 0 and len(offspring1) != len(feature_list):
            mutation_addition_prob1 = mutation_probability
        elif len(offspring1) == len(feature_list):
            mutation_addition_prob1 = 0

        if len(offspring2) > 0 and len(offspring2) != len(feature_list):
            mutation_addition_prob2 = mutation_probability * (len(offspring2)) / (
                (len(feature_list) - len(offspring2)))
        elif len(offspring2) == 0 and len(offspring2) != len(feature_list):
            mutation_addition_prob2 = mutation_probability
        elif len(offspring2) == len(feature_list):
            mutation_addition_prob2 = 0

        # mutation_addition_prob1 = 0.7000000000000001
        # mutation_addition_prob2 = 0.44999999999999996

        offspring1 = ['R_1', 'R_2', 'P_3', 'R_5']
        offspring2 = ['P_2', 'R_6', 'R_3']

        # Mutation: Addition occurs on this feature with probability proportional to the mutation parameter
        # The probability accounts for the ratio between the feature list and the size of the bin

        features_not_in_offspring1 = [item for item in feature_list if item not in offspring1]
        features_not_in_offspring2 = [item for item in feature_list if item not in offspring2]

        features_not_in_offspring1_df = pd.DataFrame(features_not_in_offspring1, columns=['Features'])
        mutation_addition_threshold1 = list([mutation_addition_prob1 * 100] * len(features_not_in_offspring1_df))
        rand1 = [0, 90, 56, 29, 4, 80]
        features_not_in_offspring1_df['Threshold'] = mutation_addition_threshold1
        features_not_in_offspring1_df['Rand_prob'] = rand1

        features_not_in_offspring2_df = pd.DataFrame(features_not_in_offspring2, columns=['Features'])
        mutation_addition_threshold2 = list([mutation_addition_prob2 * 100] * len(features_not_in_offspring2_df))
        rand2 = [0, 90, 56, 29, 4, 80, 99]
        features_not_in_offspring2_df['Threshold'] = mutation_addition_threshold2
        features_not_in_offspring2_df['Rand_prob'] = rand2

        features_to_add1 = list(features_not_in_offspring1_df.loc[
                                    features_not_in_offspring1_df['Threshold'] >= features_not_in_offspring1_df[
                                        'Rand_prob']]['Features'])
        features_to_add2 = list(features_not_in_offspring2_df.loc[
                                    features_not_in_offspring2_df['Threshold'] >= features_not_in_offspring2_df[
                                        'Rand_prob']]['Features'])

        offspring1.extend(features_to_add1)
        offspring2.extend(features_to_add2)

        self.assertEqual(features_not_in_offspring1, ['P_1', 'P_2', 'P_4', 'R_3', 'R_4', 'R_6'])
        self.assertEqual(features_not_in_offspring2, ['P_1', 'P_3', 'P_4', 'R_1', 'R_2', 'R_4', 'R_5'])
        self.assertEqual(offspring1, ['R_1', 'R_2', 'P_3', 'R_5', 'P_1', 'P_4', 'R_3', 'R_4'])
        self.assertEqual(offspring2, ['P_2', 'R_6', 'R_3', 'P_1', 'R_1', 'R_2'])

    def testCrossoverMutationBinSizeLimitO1LongEdited(self):
        # Ensuring that each of the offspring is no more than c times the size of the other offspring

        offspring1 = ['R_1', 'R_2', 'P_3', 'R_5', 'P_1', 'P_4', 'R_3', 'R_4']
        offspring2 = ['P_2', 'R_6', 'R_3']

        min_to_move = 0
        max_to_move = 0
        c_constraint = 2
        while len(offspring1) > c_constraint * len(offspring2) or len(offspring2) > c_constraint * len(offspring1):
            if len(offspring1) > c_constraint * len(offspring2):
                min_features = int((len(offspring1) + len(offspring2)) / (c_constraint + 1)) + 1
                min_to_move = min_features - len(offspring2)
                max_to_move = len(offspring1) - min_features
                features_to_move = ['R_1', 'R_2', 'R_3']
                offspring1 = [x for x in offspring1 if x not in features_to_move]
                offspring2.extend(features_to_move)
            elif len(offspring2) > c_constraint * len(offspring1):
                min_features = int((len(offspring1) + len(offspring2)) / (c_constraint + 1)) + 1
                min_to_move = min_features - len(offspring1)
                max_to_move = len(offspring2) - min_features
                num_to_move = np.random.randint(min_to_move, max_to_move + 1)
                features_to_move = random.sample(offspring2, num_to_move)
                offspring2 = [x for x in offspring2 if x not in features_to_move]
                offspring1.extend(features_to_move)
            offspring1 = set(offspring1)
            offspring2 = set(offspring2)

        self.assertEquals(min_to_move, 1)
        self.assertEquals(max_to_move, 4)
        self.assertEqual(offspring1, {'P_3', 'R_5', 'P_1', 'P_4', 'R_4'})
        self.assertEqual(offspring2, {'P_2', 'R_6', 'R_3', 'R_1', 'R_2'})


if __name__ == '__main__':
    unittest.main()
