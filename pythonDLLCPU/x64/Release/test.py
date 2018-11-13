mat_directory = "Z:/Data/Processed Correlations/For Sandy/"
data_directory = "Z:/Data/Correlation Data/"
#data_directory = "C:/Users/Sandy/Google Drive/Rydberg Experiment/Python/Decay Model/"

from corrLib import g3ToFile,g2ToFile,g2ToFile_new,g3ToFile_new,g2ToFile_pulse

data_folder = data_directory+"g2_test/"
#data_folder = data_directory+"Test/"
#mat_file = mat_directory+"g2_new"
max_time = 1e-6
bin_width = 20e-9
pulse_spacing = 100e-6
max_pulse_distance = 4
#g2ToFile_new(data_folder,mat_directory+"test_new",max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True)
#g2ToFile(data_folder,mat_directory+"g2_new",max_time,bin_width,pulse_spacing,max_pulse_distance)