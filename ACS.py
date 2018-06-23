import csv

import pandas as pd
import numpy as np
from SignalAnalyzer import SignalAnalyzer

class ACS():
    def __init__(self, project_path, labels, technique_types, threshold_levels, sampling_rate
                 , motion_extraction_position, ground_truth, recorded_time_duration=10):
        self.project_path = project_path
        self.labels = labels
        self.technique_types = technique_types
        self.threshold_levels = threshold_levels
        self.sampling_rate = sampling_rate
        self.motion_extraction_position = motion_extraction_position
        self.recorded_time_duration = recorded_time_duration
        self.ground_truth = ground_truth

    def execute(self, is_apply_dwt=False):
        for label in self.labels:
            #person_id = label.split("_")[0]
            #threshold_levels_of_correct_label = self.threshold_levels[person_id]
            threshold_levels_of_correct_label = self.threshold_levels
            ground_truth = self.ground_truth[label]
            for technique_type, threshold_level in zip(self.technique_types, threshold_levels_of_correct_label):
                technique_type_and_label = label + technique_type
                dataset_location = project_path + "/data/extractedComponents/"+technique_type_and_label+".csv"
                final_result_storage_location = project_path + "/result/final_result/activity_result/"
                final_result_storage_location_filename \
                    = final_result_storage_location + technique_type_and_label+ ".pickle"
                number_of_pin_componenets=1
                signal_analyzer = SignalAnalyzer(technique_type_and_label, project_path, dataset_location
                                                 , self.motion_extraction_position, self.recorded_time_duration, self.sampling_rate)
                signal_analyzer.execute(number_of_pin_componenets, technique_type_and_label, is_init=True)
                peak_points, selector, selected_channel = signal_analyzer.select_the_best_component( start=0, end=0,
                                                          is_apply_dwt=is_apply_dwt, channel_number_to_plot=0,
                                                        theshold_level=threshold_level, is_plot=False)
                signal_analyzer.store_final_result(technique_type_and_label, final_result_storage_location_filename
                                                   , peak_points, selector, selected_channel, ground_truth)

    def calculate_accuray_based_on_activity(self, activity_list, component_analysis, name=None):
        accuracy_ = self.project_path + "/result/final_result/final_accuracy.csv"

        with open(accuracy_, 'w') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(["technique", "activity", "error"])
            for activity in activity_list:
                for technique in component_analysis:
                    result_file_location = self.project_path + "/result/final_result/final_result_" + activity + ".csv"
                    df = pd.read_csv(result_file_location)
                    df = df[df.technique == technique]
                    if name!=None:
                        df = df[df.name == name]
                    total_error = np.sqrt(df.ix[:, 4].sum())/len(df.ix[:, 4])
                    writer.writerow([technique, activity, total_error])
                    print ("Total error {} - {} --> {}".format(activity, technique, total_error))

    def calculate_final_result(self, technique_types):
        final_result_normal = self.project_path+"/result/final_result/final_result_normal.csv"
        final_result_physical = self.project_path + "/result/final_result/final_result_physical.csv"
        final_results = self.project_path + "/result/final_result/final_results.csv"

        with open(final_results, 'w') as result_file:
            df_normal = pd.read_csv(final_result_normal)
            df_physical = pd.read_csv(final_result_physical)
            writer = csv.writer(result_file, delimiter='&')
            for id in range(1, 18):
                id = "p" + str(id)
                df_normal__ = df_normal[df_normal.name == id]
                df_physical__ = df_physical[df_physical.name == id]
                data_row = []
                # data_row.append(id)
                ground_truth_normal = 0
                ground_truth_physical = 0
                for technique in technique_types:
                    df_normal_ = df_normal__[df_normal__.technique == technique]
                    ground_truth_normal = round(np.array(df_normal_)[0][3], 2)
                    data_row.append(round(np.array(df_normal_)[0][2], 2))
                data_row.append(ground_truth_normal)
                for technique in technique_types:
                    df_physical_ = df_physical__[df_physical__.technique == technique]
                    ground_truth_physical = round(np.array(df_physical_)[0][3], 2)
                    data_row.append(round(np.array(df_physical_)[0][2], 2))
                data_row.append(ground_truth_physical)
                writer.writerow(data_row)


    def analysis(self, is_plot, is_apply_dwt, label, technique_type, theshold_level, plot_init=False):
        technique_type_and_label = label + technique_type
        dataset_location = self.project_path + "/data/extractedComponents/" + technique_type_and_label + ".csv"
        number_of_pin_componenets = 1
        signal_analyzer = SignalAnalyzer(technique_type_and_label, project_path, dataset_location
                                         , self.motion_extraction_position, self.recorded_time_duration, self.sampling_rate)
        signal_analyzer.execute(number_of_pin_componenets, technique_type_and_label, is_init=True)
        if plot_init:
            signal_analyzer.plot_initial_signals(start=0, end=250, with_ssa=False)
            #signal_analyzer.plot_initial_signals(start=0, end=0, with_ssa=True)
        peak_points, selector, selected_channel = signal_analyzer.select_the_best_component(start=0, end=self.sampling_rate*self.recorded_time_duration,
                                                                                            is_apply_dwt=is_apply_dwt,
                                                                                            channel_number_to_plot=0,
                                                                                            theshold_level=theshold_level,
                                                                                            is_plot=is_plot)
        peak_points = np.array(peak_points)
        time = (peak_points[-1] - peak_points[0]) / 250
        pulse_rate2 = (60 / time) * len(peak_points)
        #print(peak_points)
        #print(selector)
        #print(pulse_rate)
        print(pulse_rate2)
        print("Number of points: {}".format(len(peak_points)))
        print("Selected channel: {}".format(selected_channel[0] + 1))



project_path = "/home/runge/project/pulse"
labels = []
technique_types = ["fica", "pca", "jade", "shibbs"]
#technique_types = ["jade"]
type_of_activities = ["normal", "physical"]
motion_extraction_position = [2, 12]
threshold_levels = [0.06, 0.08, 0.07, 0.10]
#threshold_levels = [0.06]
sampling_rate = 250
recorded_time_duration = 20

for activity_type in type_of_activities:
    for x in range(1,16):
        labels.append("p" + str(x) + "_"+activity_type+ "_")

ground_truth_file = project_path + "/data/ground_truths.csv"
ground_truth_data = list(np.loadtxt(ground_truth_file, str, delimiter='\n'))
ground_truth = {}
for detail in ground_truth_data:
    detail = detail.split(",")
    try:
        value = float(detail[1])
        ground_truth[detail[0] + "_"] = value
    except (ValueError, TypeError):
        pass


acs = ACS(project_path, labels, technique_types, threshold_levels, sampling_rate, motion_extraction_position,
          ground_truth, recorded_time_duration)
#acs.execute(is_apply_dwt=True)


#acs.analysis(is_plot=False, is_apply_dwt=False, label="p2_normal_", technique_type="jade", theshold_level=0.07, plot_init=True)

signalAnalyzer = SignalAnalyzer("", project_path, None, motion_extraction_position, sampling_rate=sampling_rate
                                , recorded_time_duration=recorded_time_duration)
#for activity in type_of_activities:
#    signalAnalyzer.concat_result_based_on_activity(activity)


acs.calculate_accuray_based_on_activity( type_of_activities, technique_types, None)
acs.calculate_final_result(technique_types)

