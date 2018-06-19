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
            person_id = label.split("_")[0] + "_" + label.split("_")[1]
            threshold_levels_of_correct_label = self.threshold_levels[person_id]
            ground_truth = self.ground_truth[label]
            for technique_type, threshold_level in zip(self.technique_types, threshold_levels_of_correct_label):
                technique_type_and_label = label + technique_type
                dataset_location = project_path + "/data/extractedComponents/"+technique_type_and_label+".csv"
                final_result_storage_location = project_path + "/build/result/activity_result/"
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
        accuracy_ = self.project_path + "/build/result/final_accuracy.csv"

        with open(accuracy_, 'w') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(["technique", "activity", "error"])
            for activity in activity_list:
                for technique in component_analysis:
                    result_file_location = self.project_path + "/build/result/final_result_" + activity + ".csv"
                    df = pd.read_csv(result_file_location)
                    df = df[df.technique == technique]
                    if name!=None:
                        df = df[df.name == name]
                    total_error = np.sqrt(df.ix[:, 3].sum())
                    writer.writerow([technique, activity, total_error])
                    print ("Total error {} - {} --> {}".format(activity, technique, total_error))


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
        import numpy as np
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
type_of_activities = ["normal", "high"]
motion_extraction_position = [2, 8, 16]
sampling_rate = 250
recorded_time_duration = 20

for activity_type in type_of_activities:
    for x in range(1,8):
        labels.append("person_" + str(x) + "_"+activity_type+ "_")

threshold_levels = {}
threshold_levels["person_1"] =  [0.06, 0.08, 0.07, 0.10]
threshold_levels["person_2"] = [0.06, 0.08, 0.07, 0.10]
threshold_levels["person_3"] = [0.06, 0.08, 0.07, 0.10]
threshold_levels["person_4"] = [0.06, 0.08, 0.07, 0.10]
threshold_levels["person_5"] = [0.06, 0.08, 0.07, 0.10]
threshold_levels["person_6"] = [0.06, 0.08, 0.07, 0.10]
threshold_levels["person_7"] = [0.06, 0.08, 0.07, 0.10]

ground_truth_file = project_path + "data/ECG/groundtruth.csv"
ground_truth_data = list(np.loadtxt(ground_truth_file, str, delimiter='\n'))
ground_truth = {}
for detail in ground_truth_data:
    detail = detail.split(":")
    ground_truth[detail[0]] = int(detail[1])

acs = ACS(project_path, labels, technique_types, threshold_levels, sampling_rate, motion_extraction_position,
          ground_truth, recorded_time_duration)
acs.execute(is_apply_dwt=True)
#acs.calculate_accuray_based_on_activity( type_of_activities, technique_types, None)


# acs.analysis(is_plot=False, is_apply_dwt=False, label="geesara_v1_", technique_type="jade", theshold_level=0.07, plot_init=False)

#signalAnalyzer = SignalAnalyzer("", project_path, None, motion_extraction_position, sampling_rate=sampling_rate, recorded_time_duration=recorded_time_duration)
#for activity in type_of_activities:
#    signalAnalyzer.concat_result_based_on_activity(activity)


