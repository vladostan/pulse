from analyzer import SignalAnalyzer

class ACS():
    def __init__(self, project_path, labels, technique_types, threshold_levels, sampling_rate):
        self.project_path = project_path
        self.labels = labels
        self.technique_types = technique_types
        self.threshold_levels = threshold_levels
        self.sampling_rate = sampling_rate

    def execute(self):
        for label in self.labels:
            for technique_type, threshold_level in zip(self.technique_types, self.threshold_levels):
                technique_type_and_label = label + technique_type
                dataset_location = project_path + "/data/extractedComponents/"+technique_type_and_label+".csv"
                final_result_storage_location = project_path + "/build/result/"
                final_result_storage_location_filename \
                    = final_result_storage_location + technique_type_and_label+ ".pickle"
                number_of_pin_componenets=1
                signal_analyzer = SignalAnalyzer(technique_type_and_label, project_path, dataset_location)
                signal_analyzer.execute(number_of_pin_componenets, technique_type_and_label, is_init=True)
                peak_points, selector, selected_channel = signal_analyzer.select_the_best_component(pattern_start_at=0
                                                        , pattern_end_at=250, start=0, end=0,
                                                          is_apply_dwt=False, channel_number_to_plot=0,
                                                        theshold_level=threshold_level, is_plot=False)
                signal_analyzer.store_final_result(technique_type_and_label, final_result_storage_location_filename
                                                   , peak_points, selector, selected_channel)

    def analysis(self, is_plot, is_apply_dwt, label, technique_type, theshold_level, plot_init=False):
        technique_type_and_label = label + technique_type
        dataset_location = project_path + "/data/extractedComponents/" + technique_type_and_label + ".csv"
        number_of_pin_componenets = 1
        signal_analyzer = SignalAnalyzer(technique_type_and_label, project_path, dataset_location)
        signal_analyzer.execute(number_of_pin_componenets, technique_type_and_label, is_init=True)
        if plot_init:
            signal_analyzer.plot_initial_signals(start=0, end=250, with_ssa=False)
        peak_points, selector, selected_channel = signal_analyzer.select_the_best_component(pattern_start_at=0
                                                                                            , pattern_end_at=self.sampling_rate,
                                                                                            start=0, end=0,
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
#technique_types = ["fica0.12", "pca0.22", "jade0.18", "shibbs0.22"]
#technique_types = ["shibbs"]
sampling_rate = 250

acs = ACS(project_path, labels, technique_types, threshold_levels, sampling_rate)
#acs.execute()

#acs.analysis(is_plot=True, is_apply_dwt=False, label="vlad_v1_", technique_type="jade", theshold_level=0.18
#             , plot_init=False)

signalAnalyzer = SignalAnalyzer("", project_path, None)
signalAnalyzer.concat_result_based_on_activity("v2")


