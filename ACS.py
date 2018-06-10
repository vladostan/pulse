from analyzer import SignalAnalyzer

class ACS():
    def __init__(self, project_path, labels, technique_types, threshold_levels, sampling_rate, motion_extraction_position, recorded_time_duration=10):
        self.project_path = project_path
        self.labels = labels
        self.technique_types = technique_types
        self.threshold_levels = threshold_levels
        self.sampling_rate = sampling_rate
        self.motion_extraction_position = motion_extraction_position
        self.recorded_time_duration = recorded_time_duration

    def execute(self):
        for label in self.labels:
            for technique_type, threshold_level in zip(self.technique_types, self.threshold_levels):
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
                                                          is_apply_dwt=True, channel_number_to_plot=0,
                                                        theshold_level=threshold_level, is_plot=False)
                signal_analyzer.store_final_result(technique_type_and_label, final_result_storage_location_filename
                                                   , peak_points, selector, selected_channel)

    def analysis(self, is_plot, is_apply_dwt, label, technique_type, theshold_level, plot_init=False):
        technique_type_and_label = label + technique_type
        dataset_location = project_path + "/data/extractedComponents/" + technique_type_and_label + ".csv"
        number_of_pin_componenets = 1
        signal_analyzer = SignalAnalyzer(technique_type_and_label, project_path, dataset_location
                                         , self.motion_extraction_position, self.sampling_rate, self.recorded_time_duration)
        signal_analyzer.execute(number_of_pin_componenets, technique_type_and_label, is_init=True)
        if plot_init:
            signal_analyzer.plot_initial_signals(start=0, end=250, with_ssa=True)
        peak_points, selector, selected_channel = signal_analyzer.select_the_best_component(start=0, end=0,
                                                                                            is_apply_dwt=is_apply_dwt,
                                                                                            channel_number_to_plot=0,
                                                                                            theshold_level=theshold_level,
                                                                                            is_plot=is_plot)
        print(peak_points)
        print(selector)
        print("Number of points: {}".format(len(peak_points)))
        print("Selected channel: {}".format(selected_channel[0] + 1))



project_path = "/home/runge/project/pulse"
#labels = ["vlad_v2_"]
labels = ["vlad_v2_", "geesara_v2_", "vlad_v1_", "geesara_v1_"]
technique_types = ["fica", "pca", "jade", "shibbs"]
threshold_levels = [0.10, 0.22, 0.18, 0.22]
motion_extraction_position = [2, 5, 8]
#technique_types = ["fica0.12", "pca0.22", "jade0.18", "shibbs0.22"]
#technique_types = ["shibbs"]
sampling_rate = 250
recorded_time_duration = 20

acs = ACS(project_path, labels, technique_types, threshold_levels, sampling_rate, motion_extraction_position,
          recorded_time_duration)
#acs.execute()

#acs.analysis(is_plot=True, is_apply_dwt=False, label="vlad_v1_", technique_type="jade", theshold_level=0.18
#             , plot_init=True)

signalAnalyzer = SignalAnalyzer("", project_path, None, motion_extraction_position
                                , sampling_rate=sampling_rate, recorded_time_duration=recorded_time_duration)
signalAnalyzer.concat_result_based_on_activity("v1")


