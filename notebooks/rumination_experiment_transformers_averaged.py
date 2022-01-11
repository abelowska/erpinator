import scipy
import scipy.stats
import numpy as np

from scipy.signal import butter, lfilter
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# # indices for slicing epoch into ERN part and Pe part (in milisecons)
# start_ern = 100
# stop_ern = 250
# start_pe = 250
# stop_pe= 450

# # indices in timepoints
# start_ern_tp = int(signal_frequency * start_ern / 1000)
# stop_ern_tp = int(signal_frequency * stop_ern / 1000)
# start_pe_tp = int(signal_frequency * start_pe / 1000)
# stop_pe_tp = int(signal_frequency * stop_pe / 1000)


signal_frequency = 256

# require 3-D data: epochs x channels x timepoints
class LowpassFilter(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"IN LOWPASS FILTER")
        fs = signal_frequency
        cutoff = 40  # Hz
        B, A = butter(
            6, cutoff / (fs / 2), btype="low", analog=False
        )  # 6th order Butterworth low-pass

        # filter each signal piece with Butterworth filter
        filtred_signal = np.array(
            [
                np.array([lfilter(B, A, channel, axis=0) for channel in epoch])
                for epoch in X
            ]
        )

        print(f"IN BUTTERWORTH FILTER SHAPE: {filtred_signal.shape}")
        return filtred_signal


# require 4-D data: participants x epochs x channels x timepoints
# return 3-D data: averaged participat epochs x channels x timepoints
class AveragePerParticipant(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"IN AVERAGE X SHAPE: {X.shape}")
        averaged_paricipant_epochs = np.array(
            [np.mean(participant, axis=0) for participant in X]
        )
        print(f"IN AVERAGE RETURN SHAPE: {averaged_paricipant_epochs.shape}")
        print(averaged_paricipant_epochs.dtype)
        return averaged_paricipant_epochs


# require 3-D data: epochs x channels x timepoints
class SpatialFilterPreprocessing(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # join data from each epoch. Shape: channels (n_features) x timepoints*epochs (n_samples)
        timepoints_per_channel = np.concatenate(X, axis=1)

        # create input vector for spatial filter training: array-like of shape (n_samples, n_features)
        spatial_filter_input_data = timepoints_per_channel.T

        return spatial_filter_input_data


# X in spatial-filter shape: n_samples x n_features
# Recovers shape: epoch x channel(spatial_filter_component) x timepoints
class SpatialFilterPostprocessing(TransformerMixin, BaseEstimator):
    def __init__(self, timepoints_count):
        super().__init__()
        self.timepoints_count = timepoints_count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # reshape to n_features x n_samples
        X_transposed = X.T

        # get number of created components(n_features)
        spatial_filter_n_components = X.shape[1]

        # get number of epochs: n_samples = epochs*timepoints -> epochs = n_samples / timepoints
        n_epochs = int(X.shape[0] / self.timepoints_count)

        # retrieve shape of epochs x n_components x timepoints
        data_channel_wise = X_transposed.reshape(
            spatial_filter_n_components, n_epochs, self.timepoints_count
        )
        data_epoch_wise = np.transpose(data_channel_wise, (1, 0, 2))

        return np.array(data_epoch_wise)


# reshape data from (channels x epoch x features) to (epochs x channles x features)
# and then flatten it to (epoch x channels*features)
class PostprocessingTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectorized_data = np.stack(X, axis=1)
        epochs_per_channel_feature = vectorized_data.reshape(
            vectorized_data.shape[0], -1
        )
        print(f"POST SHAPE:{epochs_per_channel_feature.shape}")
        return epochs_per_channel_feature


# require 4-D data: participants x epochs x channels x timepoints
class ChannelExtraction(TransformerMixin, BaseEstimator):
    def __init__(self, channel_list):
        super().__init__()
        self.channel_list = channel_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        selected_data = []

        for participant_data in X:
            # order each participant data channel-wise instead epoch-wise
            participant_data_channel_wise = np.transpose(participant_data, (1, 0, 2))

            # select channels specified in channel list
            participant_selected_data = np.array(
                [
                    participant_data_channel_wise[channel]
                    for channel in self.channel_list
                ]
            )

            # reorder participant data epoch-wise back
            participant_selected_data_epoch_wise = np.transpose(
                participant_selected_data, (1, 0, 2)
            )

            selected_data.append(participant_selected_data_epoch_wise)
        selected_data = np.array(selected_data)
        print(f"EXTRACTION {selected_data.shape}")
        return selected_data


# swap channels and epochs axes: from epoch_channel_timepoints to channel_epoch_timepoints and vice versa
class ChannelDataSwap(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_channel_swaped = np.transpose(X, (1, 0, 2))
        print(f"SWAP shape: {data_channel_swaped.shape}")
        return data_channel_swaped


class BinTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def bin_epoch(self, epoch):
        new_channels = []
        for channel in epoch:
            bins_channel = []
            index = 0
            while index + self.step + 1 < len(channel):
                this_bin = np.mean(channel[index : index + self.step + 1])
                bins_channel.append(this_bin)
                index += self.step
            new_channels.append(bins_channel)
        return new_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        binned_data = np.array([self.bin_epoch(epoch) for epoch in X])
        print(f"IN BINS RETURN SHAPE: {binned_data.shape}")
        return binned_data


# indices in timepoints
start_ern_bin = 2
stop_ern_bin = 5
start_pe_bin = 5
stop_pe_bin = 9


class ErnTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        ern_data = np.array(
            [
                participant.take(indices=range(start_ern_bin, stop_ern_bin), axis=1)
                for participant in X
            ]
        )

        print(f"IN ERN RETURN SHAPE: {ern_data.shape}")
        return ern_data


class PeTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        pe_data = np.array(
            [
                participant.take(indices=range(start_pe_bin, stop_pe_bin), axis=1)
                for participant in X
            ]
        )

        print(f"IN PE RETURN SHAPE: {pe_data.shape}")
        return pe_data


class GetFeature(TransformerMixin, BaseEstimator):
    def __init__(self, dataset, feature_name):
        super().__init__()
        self.feature_name = feature_name
        self.dataset = dataset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feature = np.array(
            X[X["marker"] == self.dataset][self.feature_name].to_list()
        ).reshape(-1, 1)
        print(f"IN FEATURE RETURN SHAPE: {feature.shape}")

        return feature


class EEGdata(TransformerMixin, BaseEstimator):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_train = np.array(X[X["marker"] == self.dataset]["epochs"].tolist())

        return X_train
