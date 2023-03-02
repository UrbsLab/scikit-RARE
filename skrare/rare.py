# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 20:52:06 2022

@author: ann72
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from RARE_Methods_08_22_2022.py import *


class RARE(BaseEstimator, TransformerMixin):
    def __init__(self, label_name="Class", duration_name="grf_yrs",
                 given_starting_point=False, amino_acid_start_point=None, amino_acid_bins_start_point=None,
                 iterations=1000, set_number_of_bins=1, min_features_per_group=1, max_number_of_groups_with_feature=1,
                 informative_cutoff=0.2, crossover_probability=0.5, mutation_probability=0.05, elitism_parameter=0.2,
                 scoring_method='Relief', score_based_on_sample=True, score_with_common_variables=False,
                 instance_sample_size=500,
                 random_seed=None, bin_size_variability_constraint=None, max_features_per_bin=None):

        # iterations
        if not self.checkIsInt(iterations):
            raise Exception("iterations param must be nonnegative integer")

        if iterations < 0:
            raise Exception("iterations param must be nonnegative integer")

        # set_number_of_bins
        if not self.checkIsInt(set_number_of_bins):
            raise Exception("set_number_of_bins param must be nonnegative integer")

        if set_number_of_bins < 1:
            raise Exception("set_number_of_bins param must be nonnegative integer 1 or greater")

        # min_features_per_group
        if not self.checkIsInt(min_features_per_group):
            raise Exception("min_features_per_group param must be nonnegative integer")

        if min_features_per_group < 0:
            raise Exception("min_features_per_group param must be nonnegative integer")

        # max_number_of_groups_with_feature
        if not self.checkIsInt(max_number_of_groups_with_feature):
            raise Exception("max_number_of_groups_with_feature param must be nonnegative integer")

        if max_number_of_groups_with_feature < 0:
            raise Exception("max_number_of_groups_with_feature param must be nonnegative integer")

        if max_number_of_groups_with_feature > set_number_of_bins:
            raise Exception(
                "max_number_of_groups_with_feature must be less than or equal to population size of candidate bins")

        # informative_cutoff
        if not self.checkIsFloat(informative_cutoff):
            raise Exception("informative_cutoff param must be float from 0 - 0.5")

        if informative_cutoff < 0 or informative_cutoff > 0.5:
            raise Exception("informative_cutoff param must be float from 0 - 0.5")

        # crossover_probability
        if not self.checkIsFloat(crossover_probability):
            raise Exception("crossover_probability param must be float from 0 - 1")

        if crossover_probability < 0 or crossover_probability > 1:
            raise Exception("crossover_probability param must be float from 0 - 1")

        # mutation_probability
        if not self.checkIsFloat(mutation_probability):
            raise Exception("mutation_probability param must be float from 0 - 1")

        if mutation_probability < 0 or mutation_probability > 1:
            raise Exception("mutation_probability param must be float from 0 - 1")

        # elitism_parameter
        if not self.checkIsFloat(elitism_parameter):
            raise Exception("elitism_parameter param must be float from 0 - 1")

        if elitism_parameter < 0 or elitism_parameter > 1:
            raise Exception("elitism_parameter param must be float from 0 - 1")

        # given_starting_point
        if not (isinstance(given_starting_point, bool)):
            raise Exception("given_starting_point param must be boolean True or False")
        elif given_starting_point:
            if amino_acid_start_point == None or amino_acid_bins_start_point == None:
                raise Exception(
                    "amino_acid_start_point param and amino_acid_bins_start_point param must be a list if expert "
                    "knowledge is being inputted")
            elif not (isinstance(amino_acid_start_point, list)):
                raise Exception("amino_acid_start_point param must be a list")
            elif not (isinstance(amino_acid_bins_start_point, list)):
                raise Exception("amino_acid_bins_start_point param must be a list")

        # label_name
        if not (isinstance(label_name, str)):
            raise Exception("label_name param must be str")

        # scoring_method
        if scoring_method != 'Relief' or scoring_method != 'Univariate' or scoring_method != 'Relief only on bin and ' \
                                                                                             'common features':
            raise Exception(
                "scoring_method param must be 'Relief' or 'Univariate' or 'Relief only on bin and common features'")

        # score_based_on_sample
        if not (isinstance(score_based_on_sample, bool)):
            raise Exception("score_based_on_sample param must be boolean True or False")

        # score_with_common_variables
        if not (isinstance(score_with_common_variables, bool)):
            raise Exception("score_with_common_variables param must be boolean True or False")

        # instance_sample_size
        if not instance_sample_size.checkIsInt:
            raise Exception("instance_sample_size param must be integer")
        if instance_sample_size > set_number_of_bins:
            raise Exception(
                "instance_sample_size param must be less than or equal to the number of bins, which is " + str(
                    set_number_of_bins) + " bins.")

        # bin_size_variability_constraint
        if not (bin_size_variability_constraint is None) or not bin_size_variability_constraint.checkIsFloat:
            raise Exception(
                "bin_size_variability_constraint param must be None, an integer 1 or greater, or a float 1.0 or greater")
        if not (bin_size_variability_constraint is None) and bin_size_variability_constraint < 1:
            raise Exception("bin_size_variability_constraint is less than 1")

        # max_features_per_bin
        if not (max_features_per_bin is None):
            if not self.checkIsInt(max_features_per_bin):
                raise Exception("max_features_per_bin param must be nonnegative integer")

            if min_features_per_group <= max_features_per_bin:
                raise Exception("max_features_per_bin param must be greater or equal to min_features_per_group param")

        self.given_starting_point = given_starting_point
        self.amino_acid_start_point = amino_acid_start_point
        self.amino_acid_bins_start_point = amino_acid_bins_start_point
        self.iterations = iterations
        self.label_name = label_name
        self.set_number_of_bins = set_number_of_bins
        self.min_features_per_group = min_features_per_group
        self.max_number_of_groups_with_feature = max_number_of_groups_with_feature
        self.informative_cutoff = informative_cutoff
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism_parameter = elitism_parameter
        self.instance_sample_size = instance_sample_size
        self.random_seed = random_seed
        self.bin_size_variability_constraint = bin_size_variability_constraint
        self.max_features_per_bin = max_features_per_bin

        # Reboot Population
        if self.reboot_filename is not None:
            self.rebootPopulation()
            self.hasTrained = True
        else:
            self.iterationCount = 0

        self.hasTrained = False

    def checkIsInt(self, num):
        return isinstance(num, int)

    def checkIsFloat(self, num):
        return isinstance(num, float)

    ##*************** Fit ****************
    def fit(self, original_feature_matrix, y=None):
        """Scikit-learn required: Supervised training of FIBERS
        Parameters
        X: array-like {n_samples, n_features} Training instances. ALL INSTANCE ATTRIBUTES MUST BE NUMERIC or NAN
        y: array-like {n_samples} Training labels. ALL INSTANCE PHENOTYPES MUST BE NUMERIC NOT NAN OR OTHER TYPE
        Returns self
        """
        # original_feature_matrix
        if not (isinstance(original_feature_matrix, pd.DataFrame)):
            raise Exception("original_feature_matrix must be pandas dataframe")

        # Check if original_feature_matrix and y are numeric
        try:
            for instance in original_feature_matrix:
                for value in instance:
                    if not (np.isnan(value)):
                        float(value)
            for value in y:
                float(value)

        except:
            raise Exception("X and y must be fully numeric")

        if not (self.label_name in original_feature_matrix.columns):
            raise Exception("label_name param must be a column in the dataset")

        self.original_feature_matrix = original_feature_matrix

        return self

    def transform(self, original_feature_matrix, y=None):

        if not (self.original_feature_matrix == original_feature_matrix):
            raise Exception("X param does not match fitted matrix. Fit needs to be first called on the same matrix.")

        bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, common_feature_MAF_dict, rare_feature_df, common_feature_df, MAF_0_features = RARE_v2(
            self.given_starting_point, self.amino_acid_start_point,
            self.amino_acid_bins_start_point, self.iterations, self.original_feature_matrix,
            self.label_name, self.rare_variant_MAF_cutoff, self.set_number_of_bins,
            self.min_features_per_group, self.max_number_of_groups_with_feature,
            self.scoring_method, self.score_based_on_sample, self.score_with_common_variables,
            self.instance_sample_size,
            self.crossover_probability, self.mutation_probability, self.elitism_parameter,
            self.random_seed, self.bin_size_variability_constraint, self.max_features_per_bin)

        return self, bin_feature_matrix, common_features_and_bins_matrix, amino_acid_bins, amino_acid_bin_scores, rare_feature_MAF_dict, common_feature_MAF_dict, rare_feature_df, common_feature_df, MAF_0_features
