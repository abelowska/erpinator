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
        # print(f"IN LOWPASS FILTER")
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

        # print(f"IN BUTTERWORTH FILTER SHAPE: {filtred_signal.shape}")
        return filtred_signal


# require 4-D data: participants x epochs x channels x timepoints
# return 3-D data: averaged participat epochs x channels x timepoints
class AveragePerParticipant(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # print(f"IN AVERAGE X SHAPE: {X.shape}")
        averaged_paricipant_epochs = np.array(
            [np.mean(participant, axis=0) for participant in X]
        )
        # print(f"IN AVERAGE RETURN SHAPE: {averaged_paricipant_epochs.shape}")
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
        # print(f"POST SHAPE:{epochs_per_channel_feature.shape}")
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
        selected_data = np.array(selected_data, dtype=object)
        # print(f"EXTRACTION {selected_data.shape}")
        return selected_data


# swap channels and epochs axes: from epoch_channel_timepoints to channel_epoch_timepoints and vice versa
class ChannelDataSwap(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_channel_swaped = np.transpose(X, (1, 0, 2))
        # print(f"SWAP shape: {data_channel_swaped.shape}")
        return data_channel_swaped


class BinTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, step=12):
        super().__init__()
        self.step = step

    def bin_epoch(self, epoch):
        new_channels = []
        for channel in epoch:
            bins_channel = []
            index = 0
            while index + self.step < len(channel):
                this_bin = np.mean(channel[index : index + self.step])
                bins_channel.append(this_bin)
                index += self.step
            new_channels.append(bins_channel)
        return new_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        binned_data = np.array([self.bin_epoch(epoch) for epoch in X])
        # print(f"IN BINS RETURN SHAPE: {binned_data.shape}")
        return binned_data


# indices in bins

tmin, tmax = -0.1, 0.6  # Start and end of the segments
signal_frequency = 256

step_in_ms = 50  # in miliseconds (?)
step_tp = int(signal_frequency * step_in_ms / 1000)  # in timepoints

# indices for slicing epoch into ERN part and Pe part (in sec)
start_ern = 0
stop_ern = 0.15
start_pe = 0.15
stop_pe = 0.35

# start_ern_bin = int((signal_frequency * (start_ern - tmin)) / step_tp) - 1
# stop_ern_bin = int(signal_frequency * (stop_ern - tmin) / step_tp)
# start_pe_bin = int(signal_frequency * (start_pe - tmin) / step_tp)
# stop_pe_bin = int(signal_frequency * (stop_pe - tmin) / step_tp)

# start_ern_bin = 0
# stop_ern_bin = 4
# start_pe_bin = 4
# stop_pe_bin = 7

# indices for regression_union_100-600_baselined_centered_ampl-2-pe
start_ern_bin = 0
stop_ern_bin = 3
start_pe_bin = 2
stop_pe_bin = 7


start_ern_tp = 101
stop_ern_tp = 138
start_pe_tp = 137
stop_pe_tp = 186


# tu wchodzi zbinowany
class CenteredSignal(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        search_start_bin = 2

        search_data = np.array(
            [
                participant.take(indices=range(search_start_bin, 5), axis=1)
                for participant in X
            ]
        )

        signal_max_positions = np.array(
            [search_start_bin + np.argmax(epoch[0]) for epoch in search_data]
        )

        # print(signal_max_positions)

        X_index_zip = zip(X, signal_max_positions)

        centered_data = []
        for participant, index in X_index_zip:
            # print(f"participant{participant}, index {index}")
            centered_data.append(
                participant.take(indices=range(index - 2, index - 2 + 13), axis=1)
            )
        centered_data = np.array(centered_data)
        # centered_data = np.array(
        #     [
        #         # print(f"participant{participant}, index {index}")
        #         participant.take(indices=range(index-2, 15), axis=1)
        #         # print(f"participant{participant}, index {index}")
        #         for participant, index in X_index_zip
        #     ]
        # )

        # print(centered_data)

        #         # print(f"IN ERN RETURN SHAPE: {ern_data.shape}")
        return centered_data


# tu wchodzi zbinowany
class CenteredSignalAfterBaseline(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        search_start_bin = 2

        search_data = np.array(
            [
                participant.take(indices=range(search_start_bin, 5), axis=1)
                for participant in X
            ]
        )

        signal_max_positions = np.array(
            [search_start_bin + np.argmax(epoch[0]) for epoch in search_data]
        )

        # print(signal_max_positions)

        X_index_zip = zip(X, signal_max_positions)

        centered_data = []
        for participant, index in X_index_zip:
            # print(f"participant{participant}, index {index}")
            centered_data.append(
                participant.take(indices=range(index - 1, index - 1 + 10), axis=1)
            )
        centered_data = np.array(centered_data)
        # centered_data = np.array(
        #     [
        #         # print(f"participant{participant}, index {index}")
        #         participant.take(indices=range(index-2, 15), axis=1)
        #         # print(f"participant{participant}, index {index}")
        #         for participant, index in X_index_zip
        #     ]
        # )

        # print(centered_data)

        #         # print(f"IN ERN RETURN SHAPE: {ern_data.shape}")
        return centered_data


class CenteredPeAfterBaseline(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # print(X.shape)

        pe_centrum = np.array([np.argmin(epoch[0][0:6]) for epoch in X])

        # print(pe_centrum)

        X_index_zip = zip(X, pe_centrum)

        centered_data = []
        for participant, index in X_index_zip:
            # print(f"participant{participant}, index {index}")
            centered_data.append(
                participant.take(indices=range(index - 2, index + 1), axis=1)
            )
        centered_data = np.array(centered_data)
        # centered_data = np.array(
        #     [
        #         # print(f"participant{participant}, index {index}")
        #         participant.take(indices=range(index-2, 15), axis=1)
        #         # print(f"participant{participant}, index {index}")
        #         for participant, index in X_index_zip
        #     ]
        # )

        # print(centered_data)

        #         # print(f"IN ERN RETURN SHAPE: {ern_data.shape}")
        return centered_data


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

        # print(f"IN ERN RETURN SHAPE: {ern_data.shape}")
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

        # print(f"IN PE RETURN SHAPE: {pe_data.shape}")
        return pe_data


class ErnTransformerTP(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        ern_data = np.array(
            [
                participant.take(indices=range(start_ern_tp, stop_ern_tp), axis=1)
                for participant in X
            ]
        )

        # print(f"IN ERN RETURN SHAPE: {ern_data.shape}")
        return ern_data


class PeTransformerTP(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        pe_data = np.array(
            [
                participant.take(indices=range(start_pe_tp, stop_pe_tp), axis=1)
                for participant in X
            ]
        )

        # print(f"IN PE RETURN SHAPE: {pe_data.shape}")
        return pe_data


class ErnMinMaxFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        ern_min_max = np.array(
            [np.array([[max(epoch)] for epoch in channel]) for channel in X]
        )

        # print(f"IN ERN min max RETURN SHAPE: {ern_min_max.shape}")
        return ern_min_max


class PeMinMaxFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        pe_min_max = np.array(
            [np.array([[min(epoch)] for epoch in channel]) for channel in X]
        )

        # print(f"IN PE min max RETURN SHAPE: {pe_min_max.shape}")
        return pe_min_max


class ErnAmplitude2(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        ern_amplitude = np.array(
            [
                np.array([[max(this_bin) - min(this_bin)] for this_bin in channel])
                for channel in X
            ]
        )

        # print(f"IN ERN min max RETURN SHAPE: {ern_min_max.shape}")
        return ern_amplitude


class PeAmplitude2(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        pe_amplitude = np.array(
            [
                np.array([[max(this_bin) - min(this_bin)] for this_bin in channel])
                for channel in X
            ]
        )

        # print(f"IN ERN min max RETURN SHAPE: {ern_min_max.shape}")
        return pe_amplitude


class ErnAmplitude3(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        ern_amplitude = np.array(
            [np.array([[this_bin[0]] for this_bin in channel]) for channel in X]
        )

        bins_baselined = X - ern_amplitude

        # print(f"IN ERN min max RETURN SHAPE: {ern_min_max.shape}")
        return bins_baselined


class PeAmplitude3(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        pe_amplitude = np.array(
            [np.array([[this_bin[0]] for this_bin in channel]) for channel in X]
        )

        bins_baselined = X - pe_amplitude

        # print(f"IN ERN min max RETURN SHAPE: {ern_min_max.shape}")
        return bins_baselined


class ErnBaselined(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        baseline = np.array(
            [np.array([[min(this_bin[1:4])] for this_bin in channel]) for channel in X]
        )

        bins_baselined = X - baseline

        # print(f"IN ERN min max RETURN SHAPE: {ern_min_max.shape}")
        return bins_baselined


# zakładam całość sygnału - niewycięty i niezbinowany
class ErnAmplitude(TransformerMixin, BaseEstimator):
    def __init__(self, step=12):
        self.step = step

    def bin_signal(self, signal):
        new_channels = []
        for channel in signal:
            bins_channel = []
            index = 0
            while index + self.step < len(channel):
                this_bin = np.mean(channel[index : index + self.step])
                bins_channel.append(this_bin)
                index += self.step
            new_channels.append(bins_channel)
        return new_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        baseline_start_tp = 10
        baseline_stop_tp = 50

        ern_start = 20
        ern_stop = 65

        # gest signal from positivity before ERN
        baseline_signal = np.array(
            [
                epoch.take(indices=range(baseline_start_tp, baseline_stop_tp), axis=1)
                for epoch in X
            ]
        )

        # gest signal from ERN
        ern_signal = np.array(
            [epoch.take(indices=range(ern_start, ern_stop), axis=1) for epoch in X]
        )

        binned_baseline = np.array(
            [self.bin_signal(epoch) for epoch in baseline_signal]
        )
        binned_ern = np.array([self.bin_signal(epoch) for epoch in ern_signal])

        #         baseline_max = np.array([
        #             np.array([[ min(this_bin)] for this_bin[0] in channel])
        #             for channel in binned_baseline])

        #         ern_max = np.array([
        #             np.array([[ max(this_bin)] for this_bin[0] in channel])
        #             for channel in binned_ern])

        baseline_max = np.array(
            [[min(channel[0])] for channel in binned_baseline]
        ).reshape(len(X), 1, -1)

        ern_max = np.array([[max(channel[0])] for channel in binned_ern]).reshape(
            len(X), 1, -1
        )

        ern_amplitude = ern_max - baseline_max

        # print(f"IN ERN ampl RETURN SHAPE: {ern_amplitude.shape}")
        return ern_amplitude


# zakładam całość sygnału - niewycięty i niezbinowany
class ErnAmplitudeTest(TransformerMixin, BaseEstimator):
    def __init__(self, step=12):
        self.step = step

    def bin_signal(self, signal):
        new_channels = []
        for channel in signal:
            bins_channel = []
            index = 0
            while index + self.step < len(channel):
                this_bin = np.mean(channel[index : index + self.step])
                bins_channel.append(this_bin)
                index += self.step
            new_channels.append(bins_channel)
        return new_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        baseline_start_tp = 10
        baseline_stop_tp = 50

        ern_start = 20
        ern_stop = 65

        # gest signal from positivity before ERN
        baseline_signal = np.array(
            [
                epoch.take(indices=range(baseline_start_tp, baseline_stop_tp), axis=1)
                for epoch in X
            ]
        )

        # gest signal from ERN
        ern_signal = np.array(
            [epoch.take(indices=range(ern_start, ern_stop), axis=1) for epoch in X]
        )

        binned_baseline = np.array(
            [self.bin_signal(epoch) for epoch in baseline_signal]
        )
        binned_ern = np.array([self.bin_signal(epoch) for epoch in ern_signal])

        #         baseline_max = np.array([
        #             np.array([[ min(this_bin)] for this_bin[0] in channel])
        #             for channel in binned_baseline])

        #         ern_max = np.array([
        #             np.array([[ max(this_bin)] for this_bin[0] in channel])
        #             for channel in binned_ern])

        baseline_max = np.array(
            [
                [
                    (
                        min(channel[0]),
                        (
                            baseline_start_tp + np.argmin(channel[0]) * self.step,
                            baseline_start_tp
                            + np.argmin(channel[0]) * self.step
                            + self.step,
                        ),
                    )
                ]
                for channel in binned_baseline
            ]
        )

        ern_max = np.array(
            [
                [
                    (
                        max(channel[0]),
                        (
                            ern_start + np.argmax(channel[0]) * self.step,
                            ern_start + np.argmin(channel[0]) * self.step + self.step,
                        ),
                    )
                ]
                for channel in binned_ern
            ]
        )

        # ern_amplitude = ern_max - baseline_max

        # print(f"IN ERN ampl RETURN SHAPE: {ern_amplitude.shape}")
        return (baseline_max, ern_max)


class PeAmplitude(TransformerMixin, BaseEstimator):
    def __init__(self, step=12):
        self.step = step

    def bin_signal(self, signal):
        new_channels = []
        for channel in signal:
            bins_channel = []
            index = 0
            while index + self.step < len(channel):
                this_bin = np.mean(channel[index : index + self.step])
                bins_channel.append(this_bin)
                index += self.step
            new_channels.append(bins_channel)
        return new_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        pe_start = 60
        pe_stop = 100

        #         # gest signal from positivity before ERN
        #         baseline_signal = np.array(
        #             [
        #                 epoch.take(indices=range(ern_start, ern_stop), axis=1)
        #                 for epoch in X
        #             ]
        #         )

        # gest signal from Pe
        pe_signal = np.array(
            [epoch.take(indices=range(pe_start, pe_stop), axis=1) for epoch in X]
        )

        # binned_baseline = np.array([self.bin_signal(epoch, step_tp) for epoch in baseline_signal])
        binned_pe = np.array([self.bin_signal(epoch) for epoch in pe_signal])

        # baseline_max = np.array([
        #     np.array([[ max(this_bin)] for this_bin in channel])
        #     for channel in binned_baseline])

        # pe_max = np.array([
        #     np.array([[ max(this_bin) - min(this_bin)] for this_bin in channel)
        #     for channel in binned_pe])

        pe_max = np.array(
            [[max(channel[0]) - min(channel[0])] for channel in binned_pe]
        ).reshape(len(X), 1, -1)

        # pe_amplitude = abs(pe_max - baseline_max)

        # print(f"IN Pe ampl RETURN SHAPE: {pe_max.shape}")
        return pe_max


# return non-binned signal centred around ERN
class CenteredERN(TransformerMixin, BaseEstimator):
    def __init__(self, step=12):
        self.step = step

    def bin_signal(self, signal):
        new_channels = []
        for channel in signal:
            bins_channel = []
            index = 0
            while index + self.step < len(channel):
                this_bin = np.mean(channel[index : index + self.step])
                bins_channel.append(this_bin)
                index += self.step
            new_channels.append(bins_channel)
        return new_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        search_start = 24
        search_stop = 65

        # gest signal from given indices
        serch_signal = np.array(
            [
                epoch.take(indices=range(search_start, search_stop), axis=1)
                for epoch in X
            ]
        )

        # bin signal
        # binned_signal = np.array([self.bin_signal(epoch) for epoch in serch_signal])

        # find position of ERN - index of bin with maximum (signal has inverted sign). Searching only in the first channel!
        signal_max_positions = np.array([np.argmax(epoch[0]) for epoch in serch_signal])

        # 100ms = 25 tp
        this_range = 20

        # bin_centrum_tp = search_start + index*self.step + int(self.step/2)
        # 100ms = 25 tp
        # this_range = 25
        # ern_indices = (bin_centrum_tp - this_range,bin_centrum_tp + this_range)

        # change index of bin on tuple of indices ERN range
        # ern_indices = np.array([
        #     (search_start + bin_position*self.step + int(self.step/2) - this_range, search_start + bin_position*self.step + int(self.step/2) + this_range)
        #     for bin_position in signal_max_positions])

        ern_indices = np.array(
            [
                (
                    search_start + position - this_range,
                    search_start + position + this_range,
                )
                for position in signal_max_positions
            ]
        )

        X_indices_zip = zip(X, ern_indices)

        ern_signal = np.array(
            [
                epoch.take(indices=range(position[0], position[1]), axis=1)
                for epoch, position in X_indices_zip
            ]
        )

        # print(f"IN ERN min max RETURN SHAPE: {ern_min_max.shape}")
        return ern_signal


# return non-binned signal centred around ERN
class CenteredPe(TransformerMixin, BaseEstimator):
    def __init__(self, step=12):
        self.step = step

    def bin_signal(self, signal):
        new_channels = []
        for channel in signal:
            bins_channel = []
            index = 0
            while index + self.step < len(channel):
                this_bin = np.mean(channel[index : index + self.step])
                bins_channel.append(this_bin)
                index += self.step
            new_channels.append(bins_channel)
        return new_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        search_start = 60
        search_stop = 100

        # gest signal from given indices
        serch_signal = np.array(
            [
                epoch.take(indices=range(search_start, search_stop), axis=1)
                for epoch in X
            ]
        )

        # bin signal
        # binned_signal = np.array([self.bin_signal(epoch) for epoch in serch_signal])

        # find position of Pe - index of bin with minimum (signal has inverted sign). Searching only in the first channel!
        signal_max_positions = np.array([np.argmin(epoch[0]) for epoch in serch_signal])

        # 100ms = 25 tp
        this_range = 24

        # bin_centrum_tp = search_start + index*self.step + int(self.step/2)
        # 100ms = 25 tp
        # this_range = 25
        # ern_indices = (bin_centrum_tp - this_range,bin_centrum_tp + this_range)

        # change index of bin on tuple of indices ERN range
        # pe_indices = np.array([
        #     (search_start + bin_position*self.step + int(self.step/2) - this_range, search_start + bin_position*self.step + int(self.step/2) + this_range)
        #     for bin_position in signal_max_positions])

        pe_indices = np.array(
            [
                (
                    search_start + position - this_range,
                    search_start + position + this_range,
                )
                for position in signal_max_positions
            ]
        )

        X_indices_zip = zip(X, pe_indices)

        pe_signal = np.array(
            [
                epoch.take(indices=range(position[0], position[1]), axis=1)
                for epoch, position in X_indices_zip
            ]
        )

        # print(f"IN ERN min max RETURN SHAPE: {ern_min_max.shape}")
        return pe_signal


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
        # print(f"IN FEATURE RETURN SHAPE: {feature.shape}")

        return feature


class EEGdata(TransformerMixin, BaseEstimator):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_train = np.array(
            X[X["marker"] == self.dataset]["epochs"].tolist(), dtype=object
        )

        return X_train


class NarrowIndices(TransformerMixin, BaseEstimator):
    def __init__(self, start, stop):
        super().__init__()
        self.start = start
        self.stop = stop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_train = np.array(
            [
                participant.take(indices=range(self.start, self.stop), axis=2)
                for participant in X
            ]
        )

        return X_train
