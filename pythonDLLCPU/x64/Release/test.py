mat_directory = "Z:/Data/Processed Correlations/For Sandy/"
data_directory = "Z:/Data/Correlation Data/"

from corrLib import g3ToFile,g2ToFile

data_folder = data_directory+"g2_benchmark/"
#data_folder = data_directory+"test_files/"
#mat_file = mat_directory+"g2_new"
max_time = 1e-6
bin_width = 10e-9
pulse_spacing = 100e-6
max_pulse_distance = 4
g3ToFile(data_folder,mat_directory+"g3_new",max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=False)
#g2ToFile(data_folder,mat_directory+"g2_new",max_time,bin_width,pulse_spacing,max_pulse_distance)