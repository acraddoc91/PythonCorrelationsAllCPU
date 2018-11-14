// pythonDLLCPU.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
//  Microsoft
#define DLLEXPORT extern "C" __declspec(dllexport)

#include <stdio.h>

#include <iostream>
#include <vector>
#include <map>

#include "H5Cpp.h"
#include <H5Exception.h>

#include <Python.h>

#include <omp.h>
#include <math.h>

//Define integers explicitly to prevent problems on different platforms
#define int8 __int8
#define int16 __int16
#define int32 __int32
#define int64 __int64

const int max_tags_length = 200000;
const int max_clock_tags_length = 5000;
const int max_channels = 3;
const size_t return_size = 3;
const int file_block_size = 16;
const double tagger_resolution = 82.3e-12;
const int num_gpu = 2;
const int threads_per_cuda_block_numer = 64;
//const int offset[3] = { 0, 219, 211 };
const int offset[3] = { 0, 51, 56 };
const int shared_mem_size = 4;

struct shotData {
	bool file_load_completed;
	std::vector<int16> channel_list;
	std::map<int16, int16> channel_map;
	std::vector<int64> start_tags;
	std::vector<int64> end_tags;
	std::vector<int64> photon_tags;
	std::vector<int64> clock_tags;
	std::vector<std::vector<int64>> sorted_photon_tags;
	std::vector<std::vector<int32>> sorted_photon_bins;
	std::vector<std::vector<int64>> sorted_clock_tags;
	std::vector<std::vector<int32>> sorted_clock_bins;
	std::vector<int32> sorted_photon_tag_pointers;
	std::vector<int32> sorted_clock_tag_pointers;

	shotData() : sorted_photon_tags(max_channels, std::vector<int64>(max_tags_length, 0)), sorted_photon_bins(max_channels, std::vector<int32>(max_tags_length, 0)), sorted_photon_tag_pointers(max_channels, 0), sorted_clock_tags(2, std::vector<int64>(max_clock_tags_length, 0)), sorted_clock_bins(2, std::vector<int32>(max_clock_tags_length, 0)), sorted_clock_tag_pointers(2, 0) {}
};

void calculateNumer_g3(shotData *shot_data, int32 *max_bin, int32 *pulse_spacing, int32 *max_pulse_distance, int32 *coinc, int32 shot_file_num) {

	int32 start_clock = shot_data->sorted_clock_bins[1][0];
	int32 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];
	for (int channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {
		for (int channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {
			for (int channel_3 = channel_2 + 1; channel_3 < shot_data->channel_list.size(); channel_3++) {
				std::vector<std::vector<int32>> dummy_2(shot_data->sorted_photon_tag_pointers[channel_1]);
				std::vector<std::vector<int32>> dummy_3(shot_data->sorted_photon_tag_pointers[channel_1]);
				int32 lower_pointer_2 = 0;
				int32 lower_pointer_3 = 0;
				for (int32 i = 0; i < shot_data->sorted_photon_tag_pointers[channel_1]; i++) {
					int out_window = (shot_data->sorted_photon_bins[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_clock)) || (shot_data->sorted_photon_bins[channel_1][i] > (end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing)));
					if (!out_window) {
						bool going = true;
						int32 j = lower_pointer_2;
						while (going) {
							if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
								j++;
								lower_pointer_2 = j;
							}
							else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
								going = false;
							}
							else {
								dummy_2[i].push_back(shot_data->sorted_photon_bins[channel_2][j] - shot_data->sorted_photon_bins[channel_1][i]);
								j++;
							}
							if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
								going = false;
							}
						}
						going = true;
						int32 k = lower_pointer_3;
						while (going) {
							if (shot_data->sorted_photon_bins[channel_3][k] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
								k++;
								lower_pointer_3 = k;
							}
							else if (shot_data->sorted_photon_bins[channel_3][k] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
								going = false;
							}
							else {
								dummy_3[i].push_back(shot_data->sorted_photon_bins[channel_3][k] - shot_data->sorted_photon_bins[channel_1][i]);
								k++;
							}
							if (k > shot_data->sorted_photon_tag_pointers[channel_3]) {
								going = false;
							}
						}
					}
				}
				int tot_coinc = 0;
				for (int32 i = 0; i < shot_data->sorted_photon_tag_pointers[channel_1]; i++) {
					if ((dummy_2[i].size() > 0) && (dummy_3[i].size() > 0)) {
						tot_coinc += dummy_2[i].size() * dummy_3[i].size();
						for (int j = 0; j < dummy_2[i].size(); j++) {
							for (int k = 0; k < dummy_3[i].size(); k++) {
								int32 id_x = dummy_2[i][j] + *max_bin;
								int32 id_y = dummy_3[i][k] + *max_bin;
								int32 tot_id = id_y * (2 * (*max_bin) + 1) + id_x;
								coinc[tot_id + shot_file_num * ((*max_bin * 2 + 1) * (*max_bin * 2 + 1) + (*max_pulse_distance * 2) * (*max_pulse_distance * 2))]++;
							}
						}
					}
				}
			}
		}
	}
}

void calculateDenom_g3(shotData *shot_data, int32 *max_bin, int32 *pulse_spacing, int32 *max_pulse_distance, int32 *denom, int32 shot_file_num) {
	
	int32 start_clock = shot_data->sorted_clock_bins[1][0];
	int32 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	for (int channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {
		for (int channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {
			for (int channel_3 = channel_2 + 1; channel_3 < shot_data->channel_list.size(); channel_3++) {
				std::vector<std::vector<int32>> denom_counts(*max_pulse_distance * 2 + 1, std::vector<int32>(*max_pulse_distance * 2 + 1,0));
				#pragma omp parallel for
				for (int32 pulse_dist_1 = -*max_pulse_distance; pulse_dist_1 <= *max_pulse_distance; pulse_dist_1++) {
					for (int32 pulse_dist_2 = -*max_pulse_distance; pulse_dist_2 <= *max_pulse_distance; pulse_dist_2++) {
						if ((pulse_dist_1 != 0) && (pulse_dist_2 != 0) && (pulse_dist_1 != pulse_dist_2)) {
							int32 tau_1 = *pulse_spacing * pulse_dist_1;
							int32 tau_2 = *pulse_spacing * pulse_dist_2;
							int i = 0;
							int j = 0;
							int k = 0;
							while ((i < shot_data->sorted_photon_tag_pointers[channel_1]) && (j < shot_data->sorted_photon_tag_pointers[channel_2]) && (k < shot_data->sorted_photon_tag_pointers[channel_3])) {
								//Check if we're outside the window of interest
								int out_window = (shot_data->sorted_photon_bins[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_clock)) || (shot_data->sorted_photon_bins[channel_1][i] > (end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing)));
								//chan_1 > chan_2
								int c1gc2 = shot_data->sorted_photon_bins[channel_1][i] > (shot_data->sorted_photon_bins[channel_2][j] - tau_1);
								//Chan_1 > chan_3
								int c1gc3 = shot_data->sorted_photon_bins[channel_1][i] > (shot_data->sorted_photon_bins[channel_3][k] - tau_2);
								//Chan_1 == chan_2
								int c1ec2 = shot_data->sorted_photon_bins[channel_1][i] == (shot_data->sorted_photon_bins[channel_2][j] - tau_1);
								//Chan_1 == chan_3
								int c1ec3 = shot_data->sorted_photon_bins[channel_1][i] == (shot_data->sorted_photon_bins[channel_3][k] - tau_2);

								//Increment the running total if all three are equal and we're in the window
								denom_counts[pulse_dist_1 + *max_pulse_distance][pulse_dist_2 + *max_pulse_distance] += !out_window && c1ec2 && c1ec3;

								//Increment j if chan_2 < chan_1 or all three are equal
								j += c1gc2 || (c1ec2 && c1ec3);
								//Increment k if chan_3 < chan_1 or all three are equal
								k += c1gc3 || (c1ec2 && c1ec3);

								//Increment i if we're out the window or if chan_1 <= chan_2 and chan_1 <= chan_3
								i += out_window || (!c1gc2 && !c1gc3);
							}
						}
					}
				}
				for (int32 pulse_dist_1 = -*max_pulse_distance; pulse_dist_1 <= *max_pulse_distance; pulse_dist_1++) {
					for (int32 pulse_dist_2 = -*max_pulse_distance; pulse_dist_2 <= *max_pulse_distance; pulse_dist_2++) {
						denom[0] += denom_counts[pulse_dist_1 + *max_pulse_distance][pulse_dist_2 + *max_pulse_distance];
					}
				}
			}
		}
	}

}

void calculateNumer_g2(shotData *shot_data, int32 *max_bin, int32 *pulse_spacing, int32 *max_pulse_distance, int32 *coinc, int32 shot_file_num) {

	int32 start_clock = shot_data->sorted_clock_bins[1][0];
	int32 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];
	for (int channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {
		for (int channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {
			std::vector<std::vector<int32>> dummy_2(shot_data->sorted_photon_tag_pointers[channel_1]);
			int32 lower_pointer_2 = 0;
			for (int32 i = 0; i < shot_data->sorted_photon_tag_pointers[channel_1]; i++) {
				int out_window = (shot_data->sorted_photon_bins[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_clock)) || (shot_data->sorted_photon_bins[channel_1][i] > (end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing)));
				if (!out_window) {
					bool going = true;
					int32 j = lower_pointer_2;
					while (going) {
						if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
							j++;
							lower_pointer_2 = j;
						}
						else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
							going = false;
						}
						else {
							dummy_2[i].push_back(shot_data->sorted_photon_bins[channel_2][j] - shot_data->sorted_photon_bins[channel_1][i]);
							j++;
						}
						if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
							going = false;
						}
					}
				}
			}
			for (int32 i = 0; i < shot_data->sorted_photon_tag_pointers[channel_1]; i++) {
				if ((dummy_2[i].size() > 0)) {
					for (int j = 0; j < dummy_2[i].size(); j++) {
						int32 id_x = dummy_2[i][j] + *max_bin;
						int32 tot_id = id_x;
						coinc[tot_id + shot_file_num * ((*max_bin * 2 + 1) + (*max_pulse_distance * 2))]++;
					}
				}
			}
		}
	}
}

int32 first_above_binary_search(std::vector<int32> *a, int32 N, int32 b) {
	int32 L = 0;
	int32 R = N - 1;
	int32 return_val;
	if ((*a)[N - 1] < b) {
		return_val = N;
	}
	else {
		while (L <= R) {
			int32 m = floor((L + R) / 2);
			if ((*a)[m] < b) {
				L = m + 1;
			}
			else if (m == 0) {
				return_val = m;
				L = R + 1;
			}
			else if (((*a)[m] > b) && !((*a)[m - 1] < b)) {
				R = m - 1;
			}
			else {
				return_val = m;
				L = R + 1;
			}
		}
	}
	return return_val;
}

int32 first_below_binary_search(std::vector<int32> *a, int32 N, int32 b) {
	int32 L = 0;
	int32 R = N - 1;
	int32 return_val;
	if ((*a)[0] > b) {
		return_val = -1;
	}
	else {
		while (L <= R) {
			int32 m = floor((L + R) / 2);
			if (m == N - 1) {
				return m;
				L = R + 1;
			}
			else if (((*a)[m] < b) && !((*a)[m + 1] > b)) {
				L = m + 1;
			}
			else if ((*a)[m] > b) {
				R = m - 1;
			}
			else {
				return_val = m;
				L = R + 1;
			}
		}
	}
	return return_val;
}

void calculateNumer_g2_new(shotData *shot_data, int32 *max_bin, int32 *pulse_spacing, int32 *max_pulse_distance, int32 *coinc, int32 shot_file_num, int32 num_cpu_threads_proc) {
	
	//Get the start and stop clock bin
	int32 start_clock = shot_data->sorted_clock_bins[1][0];
	int32 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	//Loop over all the channels to be channel 1
	for (int channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {

		//Let's find out which indicies from channel 1 we can discard due to them being outside of the window of interest
		int32 low_index;
		int32 high_index;
		//Figure out which indices in the first thread we can ignore
		#pragma omp parallel for
		for (int32 i = 0; i < 2; i++) {
			if (i == 0) {
				low_index = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], *max_bin + *max_pulse_distance * *pulse_spacing + start_clock);
			}
			else {
				high_index = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing));
			}
		}

		//Split the remaining indices between the work threads we have
		int32 indices_per_thread = (high_index - low_index) / num_cpu_threads_proc;
		for (int channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {

			//Vector to hold the relevant indices on channel 2, for each tag on channel 1, that falls with the max/min tau
			std::vector<std::vector<int32>> channel_2_indices(high_index - low_index + 1, std::vector<int32>(2));
			#pragma omp parallel for num_threads(num_cpu_threads_proc)
			for (int32 thread = 0; thread < num_cpu_threads_proc; thread++) {
				
				//Find out form this thread what the first and last indices to work on are
				int32 first_index = thread*indices_per_thread + low_index;
				int32 last_index;
				if (thread == num_cpu_threads_proc - 1) {
					last_index = high_index;
				}
				else {
					last_index = first_index + indices_per_thread-1;
				}

				//Do a binary search to find the first and last relevant tag on channel 2 for the first tag on channel 1 that the thread is working on
				int32 lower_pointer = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] - *max_bin);;
				int32 upper_pointer = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] + *max_bin);

				//Save the relevant tags on channel 2
				channel_2_indices[first_index-low_index][0] = lower_pointer;
				channel_2_indices[first_index-low_index][1] = upper_pointer;

				//Loop over all the channel 1 indices this thread needs to work on
				for (int32 i = first_index + 1; i <= last_index; i++) {

					//Find the first tag on channel 2 that is within the max/min tau of the current channel 1 tag
					bool going = true;
					int32 j = lower_pointer;
					while (going) {
						if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
							j++;
							lower_pointer = j;
						}
						else if (shot_data->sorted_photon_bins[channel_2][j] >= shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
							going = false;
							lower_pointer = j;
						}
						if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
							going = false;
							lower_pointer = j;
						}
					}
					//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
					j = upper_pointer;
					going = true;
					while (going) {
						if (shot_data->sorted_photon_bins[channel_2][j] <= shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
							j++;
							upper_pointer = j;
						}
						else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
							going = false;
							upper_pointer = j - 1;
						}
						if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
							going = false;
							upper_pointer = shot_data->sorted_photon_tag_pointers[channel_2] - 1;
						}
					}
					//Save the relevant tags to the vector for later
					channel_2_indices[i - low_index][0] = lower_pointer;
					channel_2_indices[i - low_index][1] = upper_pointer;
				}
				//Loop through all the tags on the thread has worked on to find the tau bin which the tags on channel 2 fall into
				for (int32 i = first_index; i <= last_index; i++) {
					for (int j = channel_2_indices[i - low_index][0]; j <= channel_2_indices[i - low_index][1]; j++) {
						int32 id_x = shot_data->sorted_photon_bins[channel_2][j] - shot_data->sorted_photon_bins[channel_1][i] + *max_bin;
						coinc[id_x + thread * ((*max_bin * 2 + 1)) + shot_file_num * num_cpu_threads_proc * ((*max_bin * 2 + 1))]++;
					}
				}
			}
		}
	}
}

void calculateNumer_g3_new(shotData *shot_data, int32 *max_bin, int32 *pulse_spacing, int32 *max_pulse_distance, int32 *coinc, int32 shot_file_num, int32 num_cpu_threads_proc) {

	//Get the start and stop clock bin
	int32 start_clock = shot_data->sorted_clock_bins[1][0];
	int32 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	for (int channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {

		//Let's find out which indicies from channel 1 we can discard due to them being outside of the window of interest
		int32 low_index;
		int32 high_index;
		//Figure out which indices in the first thread we can ignore
		#pragma omp parallel for
		for (int32 i = 0; i < 2; i++) {
			if (i == 0) {
				low_index = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], *max_bin + *max_pulse_distance * *pulse_spacing + start_clock);
			}
			else {
				high_index = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing));
			}
		}

		//Split the remaining indices between the work threads we have
		int32 indices_per_thread = (high_index - low_index) / num_cpu_threads_proc;

		for (int channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {
			for (int channel_3 = channel_2 + 1; channel_3 < shot_data->channel_list.size(); channel_3++) {

				//Vector to hold the relevant indices on channel 2 and 3, for each tag on channel 1, that falls with the max/min tau
				std::vector<std::vector<int32>> channel_2_indices(high_index - low_index + 1, std::vector<int32>(2));
				std::vector<std::vector<int32>> channel_3_indices(high_index - low_index + 1, std::vector<int32>(2));
				#pragma omp parallel for num_threads(num_cpu_threads_proc)
				for (int32 thread = 0; thread < num_cpu_threads_proc; thread++) {
					//Find out form this thread what the first and last indices to work on are
					int32 first_index = thread*indices_per_thread + low_index;
					int32 last_index;
					if (thread == num_cpu_threads_proc - 1) {
						last_index = high_index;
					}
					else {
						last_index = first_index + indices_per_thread - 1;
					}

					//Do a binary search to find the first and last relevant tag on channel 2 and 3 for the first tag on channel 1 that the thread is working on
					int32 lower_pointer_2 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] - *max_bin);
					int32 upper_pointer_2 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] + *max_bin);
					int32 lower_pointer_3 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_3]), shot_data->sorted_photon_tag_pointers[channel_3], shot_data->sorted_photon_bins[channel_1][first_index] - *max_bin);
					int32 upper_pointer_3 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_3]), shot_data->sorted_photon_tag_pointers[channel_3], shot_data->sorted_photon_bins[channel_1][first_index] + *max_bin);

					//Save the relevant tags on channel 2 and 3
					channel_2_indices[first_index - low_index][0] = lower_pointer_2;
					channel_2_indices[first_index - low_index][1] = upper_pointer_2;
					channel_3_indices[first_index - low_index][0] = lower_pointer_3;
					channel_3_indices[first_index - low_index][1] = upper_pointer_3;

					for (int32 i = first_index; i <= last_index; i++) {
						//Find the first tag on channel 2 that is within the max/min tau of the current channel 1 tag
						bool going = true;
						int32 j = lower_pointer_2;
						while (going) {
							if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
								j++;
								lower_pointer_2 = j;
							}
							else if (shot_data->sorted_photon_bins[channel_2][j] >= shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
								going = false;
								lower_pointer_2 = j;
							}
							if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
								going = false;
								lower_pointer_2 = j;
							}
						}
						//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
						j = upper_pointer_2;
						going = true;
						while (going) {
							if (shot_data->sorted_photon_bins[channel_2][j] <= shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
								j++;
								upper_pointer_2 = j;
							}
							else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
								going = false;
								upper_pointer_2 = j - 1;
							}
							if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
								going = false;
								upper_pointer_2 = shot_data->sorted_photon_tag_pointers[channel_2] - 1;
							}
						}
						//Save the relevant tags to the vector for later
						channel_2_indices[i - low_index][0] = lower_pointer_2;
						channel_2_indices[i - low_index][1] = upper_pointer_2;

						//Find the first tag on channel 3 that is within the max/min tau of the current channel 1 tag
						going = true;
						j = lower_pointer_3;
						while (going) {
							if (shot_data->sorted_photon_bins[channel_3][j] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
								j++;
								lower_pointer_3 = j;
							}
							else if (shot_data->sorted_photon_bins[channel_3][j] >= shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
								going = false;
								lower_pointer_3 = j;
							}
							if (j > shot_data->sorted_photon_tag_pointers[channel_3]) {
								going = false;
								lower_pointer_3 = j;
							}
						}
						//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
						j = upper_pointer_3;
						going = true;
						while (going) {
							if (shot_data->sorted_photon_bins[channel_3][j] <= shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
								j++;
								upper_pointer_3 = j;
							}
							else if (shot_data->sorted_photon_bins[channel_3][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
								going = false;
								upper_pointer_3 = j - 1;
							}
							if (j > shot_data->sorted_photon_tag_pointers[channel_3]) {
								going = false;
								upper_pointer_3 = shot_data->sorted_photon_tag_pointers[channel_3] - 1;
							}
						}
						//Save the relevant tags to the vector for later
						channel_3_indices[i - low_index][0] = lower_pointer_3;
						channel_3_indices[i - low_index][1] = upper_pointer_3;
					}
					//Loop through all the tags on the thread has worked on to find the tau bin which the tags on channel 2 fall into
					for (int32 i = first_index; i <= last_index; i++) {
						for (int j = channel_2_indices[i - low_index][0]; j <= channel_2_indices[i - low_index][1]; j++) {
							for (int k = channel_3_indices[i - low_index][0]; k <= channel_3_indices[i - low_index][1]; k++) {

								int32 id_x = shot_data->sorted_photon_bins[channel_2][j] - shot_data->sorted_photon_bins[channel_1][i] + *max_bin;
								int32 id_y = shot_data->sorted_photon_bins[channel_3][k] - shot_data->sorted_photon_bins[channel_1][i] + *max_bin;
								int32 tot_id = id_y * (2 * (*max_bin) + 1) + id_x;
								coinc[tot_id + thread * ((*max_bin * 2 + 1) * (*max_bin * 2 + 1)) + shot_file_num * num_cpu_threads_proc  * ((*max_bin * 2 + 1) * (*max_bin * 2 + 1))]++;

							}
						}
					}
				}
			}
		}
	}
}

void calculateNumer_g2_pulse(shotData *shot_data, int32 *max_bin, int32 *coinc, int32 shot_file_num) {
	//Get the start and stop clock bin
	int32 start_clock = shot_data->sorted_clock_bins[1][0];
	int32 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	//Let's find out which indicies from channel 1 we can discard due to them being outside of the window of interest
	int32 low_index;
	int32 high_index;
	//Figure out which indices in the first thread we can ignore
	#pragma omp parallel for
	for (int32 i = 0; i < 2; i++) {
		if (i == 0) {
			low_index = first_above_binary_search(&(shot_data->sorted_clock_bins[1]), shot_data->sorted_clock_tag_pointers[1], *max_bin + start_clock);
		}
		else {
			high_index = first_below_binary_search(&(shot_data->sorted_clock_bins[1]), shot_data->sorted_clock_tag_pointers[1], end_clock - (*max_bin));
		}
	}


	for (int channel_2 = 0; channel_2 < shot_data->channel_list.size(); channel_2++) {
		for (int channel_3 = channel_2 + 1; channel_3 < shot_data->channel_list.size(); channel_3++) {

			//Vector to hold the relevant indices on channel 2 and 3, for each tag on channel 1, that falls with the max/min tau
			std::vector<std::vector<int32>> channel_2_indices(high_index - low_index + 1, std::vector<int32>(2));
			std::vector<std::vector<int32>> channel_3_indices(high_index - low_index + 1, std::vector<int32>(2));
			//Find out form this thread what the first and last indices to work on are
			int32 first_index = low_index;
			int32 last_index = high_index;

			//Do a binary search to find the first and last relevant tag on channel 2 and 3 for the first tag on channel 1 that the thread is working on
			int32 lower_pointer_2 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_clock_bins[1][first_index] - *max_bin);
			int32 upper_pointer_2 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_clock_bins[1][first_index] + *max_bin);
			int32 lower_pointer_3 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_3]), shot_data->sorted_photon_tag_pointers[channel_3], shot_data->sorted_clock_bins[1][first_index] - *max_bin);
			int32 upper_pointer_3 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_3]), shot_data->sorted_photon_tag_pointers[channel_3], shot_data->sorted_clock_bins[1][first_index] + *max_bin);

			//Save the relevant tags on channel 2 and 3
			channel_2_indices[first_index - low_index][0] = lower_pointer_2;
			channel_2_indices[first_index - low_index][1] = upper_pointer_2;
			channel_3_indices[first_index - low_index][0] = lower_pointer_3;
			channel_3_indices[first_index - low_index][1] = upper_pointer_3;

			for (int32 i = first_index; i <= last_index; i++) {
				//Find the first tag on channel 2 that is within the max/min tau of the current channel 1 tag
				bool going = true;
				int32 j = lower_pointer_2;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_clock_bins[1][i] - *max_bin) {
						j++;
						lower_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] >= shot_data->sorted_clock_bins[1][i] - *max_bin) {
						going = false;
						lower_pointer_2 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						lower_pointer_2 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_2;
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] <= shot_data->sorted_clock_bins[1][i] + *max_bin) {
						j++;
						upper_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_clock_bins[1][i] + *max_bin) {
						going = false;
						upper_pointer_2 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						upper_pointer_2 = shot_data->sorted_photon_tag_pointers[channel_2] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_2_indices[i - low_index][0] = lower_pointer_2;
				channel_2_indices[i - low_index][1] = upper_pointer_2;

				//Find the first tag on channel 3 that is within the max/min tau of the current channel 1 tag
				going = true;
				j = lower_pointer_3;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_3][j] < shot_data->sorted_clock_bins[1][i] - *max_bin) {
						j++;
						lower_pointer_3 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_3][j] >= shot_data->sorted_clock_bins[1][i] - *max_bin) {
						going = false;
						lower_pointer_3 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_3]) {
						going = false;
						lower_pointer_3 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_3;
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_3][j] <= shot_data->sorted_clock_bins[1][i] + *max_bin) {
						j++;
						upper_pointer_3 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_3][j] > shot_data->sorted_clock_bins[1][i] + *max_bin) {
						going = false;
						upper_pointer_3 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_3]) {
						going = false;
						upper_pointer_3 = shot_data->sorted_photon_tag_pointers[channel_3] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_3_indices[i - low_index][0] = lower_pointer_3;
				channel_3_indices[i - low_index][1] = upper_pointer_3;
			}
			//Loop through all the tags on the thread has worked on to find the tau bin which the tags on channel 2 fall into
			for (int32 i = first_index; i <= last_index; i++) {
				for (int j = channel_2_indices[i - low_index][0]; j <= channel_2_indices[i - low_index][1]; j++) {
					for (int k = channel_3_indices[i - low_index][0]; k <= channel_3_indices[i - low_index][1]; k++) {

						int32 id_x = shot_data->sorted_photon_bins[channel_2][j] - shot_data->sorted_clock_bins[1][i] + *max_bin;
						int32 id_y = shot_data->sorted_photon_bins[channel_3][k] - shot_data->sorted_clock_bins[1][i] + *max_bin;
						int32 tot_id = id_y * (2 * (*max_bin) + 1) + id_x;
						coinc[tot_id + ((*max_bin * 2 + 1) * (*max_bin * 2 + 1)) * shot_file_num]++;

					}
				}
			}
		}
	}
}

void calculateNumer_g2_pulse_2(shotData *shot_data, int32 *min_bin_1, int32 *max_bin_1, int32 *min_bin_2, int32 *max_bin_2, int32 *coinc, int32 shot_file_num) {
	//Get the start and stop clock bin
	int32 start_clock = shot_data->sorted_clock_bins[1][0];
	int32 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	//Let's find out which indicies from channel 1 we can discard due to them being outside of the window of interest
	int32 low_index;
	int32 high_index;

	//Check which min and max bin is bigger
	int32 max_bin_check;
	int32 min_bin_check;
	if (*min_bin_1 < *min_bin_2) {
		min_bin_check = *min_bin_1;
	}
	else {
		min_bin_check = *min_bin_2;
	}
	if (*max_bin_1 < *max_bin_2) {
		max_bin_check = *max_bin_2;
	}
	else {
		max_bin_check = *min_bin_1;
	}

	//Figure out which indices in the first list we can ignore
	#pragma omp parallel for
	for (int32 i = 0; i < 2; i++) {
		if (i == 0) {
			low_index = first_above_binary_search(&(shot_data->sorted_clock_bins[1]), shot_data->sorted_clock_tag_pointers[1], -min_bin_check + start_clock);
		}
		else {
			high_index = first_below_binary_search(&(shot_data->sorted_clock_bins[1]), shot_data->sorted_clock_tag_pointers[1], end_clock - max_bin_check);
		}
	}

	for (int channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {
		for (int channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {

			//Vector to hold the relevant indices on channel 2 and 3, for each tag on channel 1, that falls with the max/min tau
			std::vector<std::vector<int32>> channel_1_indices(high_index - low_index + 1, std::vector<int32>(2));
			std::vector<std::vector<int32>> channel_2_indices(high_index - low_index + 1, std::vector<int32>(2));
			//Find out form this thread what the first and last indices to work on are
			int32 first_index = low_index;
			int32 last_index = high_index;

			//Do a binary search to find the first and last relevant tag on channel 2 and 3 for the first tag on channel 1 that the thread is working on
			int32 lower_pointer_1 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], shot_data->sorted_clock_bins[1][first_index] + *min_bin_1);
			int32 upper_pointer_1 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], shot_data->sorted_clock_bins[1][first_index] + *max_bin_1);
			int32 lower_pointer_2 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_clock_bins[1][first_index] + *min_bin_2);
			int32 upper_pointer_2 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_clock_bins[1][first_index] + *max_bin_2);

			//Save the relevant tags on channel 2 and 3
			channel_1_indices[first_index - low_index][0] = lower_pointer_1;
			channel_1_indices[first_index - low_index][1] = upper_pointer_1;
			channel_2_indices[first_index - low_index][0] = lower_pointer_2;
			channel_2_indices[first_index - low_index][1] = upper_pointer_2;

			for (int32 i = first_index; i <= last_index; i++) {
				//Find the first tag on channel 2 that is within the max/min tau of the current channel 1 tag
				bool going = true;
				int32 j = lower_pointer_1;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_1][j] < shot_data->sorted_clock_bins[1][i] + *min_bin_1) {
						j++;
						lower_pointer_1 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_1][j] >= shot_data->sorted_clock_bins[1][i] + *min_bin_1) {
						going = false;
						lower_pointer_1 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_1]) {
						going = false;
						lower_pointer_1 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_1;
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_1][j] <= shot_data->sorted_clock_bins[1][i] + *max_bin_1) {
						j++;
						upper_pointer_1 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_1][j] > shot_data->sorted_clock_bins[1][i] + *max_bin_1) {
						going = false;
						upper_pointer_1 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_1]) {
						going = false;
						upper_pointer_1 = shot_data->sorted_photon_tag_pointers[channel_1] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_1_indices[i - low_index][0] = lower_pointer_1;
				channel_1_indices[i - low_index][1] = upper_pointer_1;

				//Find the first tag on channel 3 that is within the max/min tau of the current channel 1 tag
				going = true;
				j = lower_pointer_2;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_clock_bins[1][i] + *min_bin_2) {
						j++;
						lower_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] >= shot_data->sorted_clock_bins[1][i] + *min_bin_2) {
						going = false;
						lower_pointer_2 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						lower_pointer_2 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_2;
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] <= shot_data->sorted_clock_bins[1][i] + *max_bin_2) {
						j++;
						upper_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_clock_bins[1][i] + *max_bin_2) {
						going = false;
						upper_pointer_2 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						upper_pointer_2 = shot_data->sorted_photon_tag_pointers[channel_2] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_2_indices[i - low_index][0] = lower_pointer_2;
				channel_2_indices[i - low_index][1] = upper_pointer_2;
			}
			//Loop through all the tags on the thread has worked on to find the tau bin which the tags on channel 2 fall into
			for (int32 i = first_index; i <= last_index; i++) {
				for (int j = channel_1_indices[i - low_index][0]; j <= channel_1_indices[i - low_index][1]; j++) {
					for (int k = channel_2_indices[i - low_index][0]; k <= channel_2_indices[i - low_index][1]; k++) {

						int32 id_x = shot_data->sorted_photon_bins[channel_1][j] - shot_data->sorted_clock_bins[1][i] - *min_bin_1;
						int32 id_y = shot_data->sorted_photon_bins[channel_2][k] - shot_data->sorted_clock_bins[1][i] - *min_bin_2;
						int32 tot_id = id_y * (*max_bin_1 - *min_bin_1+1) + id_x;
						coinc[tot_id + ((*max_bin_1 - *min_bin_1 + 1) * (*max_bin_2 - *min_bin_2 + 1)) * shot_file_num]++;

					}
				}
			}
		}
	}
}

void calculateDenom_g2(shotData *shot_data, int32 *max_bin, int32 *pulse_spacing, int32 *max_pulse_distance, int32 *denom, int32 shot_file_num) {

	int32 start_clock = shot_data->sorted_clock_bins[1][0];
	int32 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	for (int channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {
		for (int channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {
			std::vector<int32> denom_counts(*max_pulse_distance * 2 + 1, 0);
			#pragma omp parallel for
			for (int32 pulse_dist = -*max_pulse_distance; pulse_dist <= *max_pulse_distance; pulse_dist++) {
				if (pulse_dist != 0) {
					int32 tau = *pulse_spacing * pulse_dist;
					int i = 0;
					int j = 0;
					while ((i < shot_data->sorted_photon_tag_pointers[channel_1]) && (j < shot_data->sorted_photon_tag_pointers[channel_2])) {
						//Check if we're outside the window of interest
						int out_window = (shot_data->sorted_photon_bins[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_clock)) || (shot_data->sorted_photon_bins[channel_1][i] > (end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing)));
						//chan_1 > chan_2
						int c1gc2 = shot_data->sorted_photon_bins[channel_1][i] >(shot_data->sorted_photon_bins[channel_2][j] - tau);
						//Check if we have a common element increment
						int c1ec2 = shot_data->sorted_photon_bins[channel_1][i] == (shot_data->sorted_photon_bins[channel_2][j] - tau);
						//Increment running total if channel 1 equals channel 2
						denom_counts[pulse_dist + *max_pulse_distance] += !out_window && c1ec2;
						//Increment channel 1 if it is greater than channel 2, equal to channel 2 or ouside of the window
						i += (!c1gc2 || out_window);
						j += (c1gc2 || c1ec2);
					}
				}
			}
			for (int32 pulse_dist = -*max_pulse_distance; pulse_dist <= *max_pulse_distance; pulse_dist++) {
				denom[0] += denom_counts[pulse_dist + *max_pulse_distance];
			}
		}
	}
}

//Function grabs all tags and channel list from file
void fileToShotData(shotData *shot_data, char* filename) {
	//Open up file
	H5::H5File file(filename, H5F_ACC_RDONLY);
	//Open up "Tags" group
	H5::Group tag_group(file.openGroup("Tags"));
	//Find out how many tag sets there are, should be 4 if not something is fucky
	hsize_t numTagsSets = tag_group.getNumObjs();
	if (numTagsSets != 4) {
		printf("There should be 4 sets of Tags, found %i\n", numTagsSets);
		delete filename;
		exit;
	}
	//Read tags to shotData structure
	//First the clock tags
	H5::DataSet clock_dset(tag_group.openDataSet("ClockTags0"));
	H5::DataSpace clock_dspace = clock_dset.getSpace();
	hsize_t clock_length[1];
	clock_dspace.getSimpleExtentDims(clock_length, NULL);
	shot_data->clock_tags.resize(clock_length[0]);
	clock_dset.read(&(*shot_data).clock_tags[0u], H5::PredType::NATIVE_UINT64, clock_dspace);
	clock_dspace.close();
	clock_dset.close();
	//Then start tags
	H5::DataSet start_dset(tag_group.openDataSet("StartTag"));
	H5::DataSpace start_dspace = start_dset.getSpace();
	hsize_t start_length[1];
	start_dspace.getSimpleExtentDims(start_length, NULL);
	shot_data->start_tags.resize(start_length[0]);
	start_dset.read(&(*shot_data).start_tags[0u], H5::PredType::NATIVE_UINT64, start_dspace);
	start_dspace.close();
	start_dset.close();
	//Then end tags
	H5::DataSet end_dset(tag_group.openDataSet("EndTag"));
	H5::DataSpace end_dspace = end_dset.getSpace();
	hsize_t end_length[1];
	end_dspace.getSimpleExtentDims(end_length, NULL);
	shot_data->end_tags.resize(end_length[0]);
	end_dset.read(&(*shot_data).end_tags[0u], H5::PredType::NATIVE_UINT64, end_dspace);
	end_dspace.close();
	end_dset.close();
	//Finally photon tags
	H5::DataSet photon_dset(tag_group.openDataSet("TagWindow0"));
	H5::DataSpace photon_dspace = photon_dset.getSpace();
	hsize_t photon_length[1];
	photon_dspace.getSimpleExtentDims(photon_length, NULL);
	shot_data->photon_tags.resize(photon_length[0]);
	photon_dset.read(&(*shot_data).photon_tags[0u], H5::PredType::NATIVE_UINT64, photon_dspace);
	photon_dspace.close();
	photon_dset.close();
	//And close tags group
	tag_group.close();
	//Open up "Inform" group
	H5::Group inform_group(file.openGroup("Inform"));
	//Grab channel list
	H5::DataSet chan_dset(inform_group.openDataSet("ChannelList"));
	H5::DataSpace chan_dspace = chan_dset.getSpace();
	hsize_t chan_length[1];
	chan_dspace.getSimpleExtentDims(chan_length, NULL);
	shot_data->channel_list.resize(chan_length[0]);
	chan_dset.read(&(*shot_data).channel_list[0u], H5::PredType::NATIVE_UINT16, chan_dspace);
	chan_dspace.close();
	chan_dset.close();
	//Close Inform group
	inform_group.close();
	//Close file
	file.close();

	//Populate channel map
	for (int16 i = 0; i < shot_data->channel_list.size(); i++) {
		shot_data->channel_map[shot_data->channel_list[i]] = i;
	}
}

//Reads relevant information for a block of files into shot_block
void populateBlock(std::vector<shotData> *shot_block, std::vector<char *> *filelist, int block_num, int num_devices, int block_size) {
	//Loop over the block size
	for (int i = 0; i < block_size * num_devices; i++) {
		//Default to assuming the block is corrupted
		(*shot_block)[i].file_load_completed = false;
		//Figure out the file id within the filelist
		int file_id = block_num * block_size * num_devices + i;
		//Check the file_id isn't out of range of the filelist
		if (file_id < filelist->size()) {
			//Try to load file to shot_block
			try {
				fileToShotData(&(*shot_block)[i], (*filelist)[file_id]);
				(*shot_block)[i].file_load_completed = true;
			}
			//Will catch if the file is corrupted, print corrupted filenames to command window
			catch (...) {
				printf("%s appears corrupted\n", (*filelist)[file_id]);
			}
		}
	}
}

//Process the time tags, assigning them to the correct channel, binning them appropriately and removing tags which do not fall in the clock mask
void sortTags(shotData *shot_data) {
	int32 i;
	int high_count = 0;
	//Loop over all tags in clock_tags
	for (i = 0; i < shot_data->clock_tags.size(); i++) {
		//Check if clock tag is a high word
		if (shot_data->clock_tags[i] & 1) {
			//Up the high count
			high_count++;
		}
		else {
			//Determine whether it is the rising (start) or falling (end) slope
			int slope = ((shot_data->clock_tags[i] >> 28) & 1);
			//Put tag in appropriate clock tag vector and increment the pointer for said vector
			shot_data->sorted_clock_tags[slope][shot_data->sorted_clock_tag_pointers[slope]] = ((shot_data->clock_tags[i] >> 1) & 0x7FFFFFF) + (high_count << 27) - ((shot_data->start_tags[1] >> 1) & 0x7FFFFFF);
			shot_data->sorted_clock_tag_pointers[slope]++;
		}
	}
	high_count = 0;
	//Clock pointer
	int clock_pointer = 0;
	//Loop over all tags in photon_tags
	for (i = 0; i < shot_data->photon_tags.size(); i++) {
		//Check if photon tag is a high word
		if (shot_data->photon_tags[i] & 1) {
			//Up the high count
			high_count++;
		}
		else {
			//Figure out if it fits within the mask
			int64 time_tag = ((shot_data->photon_tags[i] >> 1) & 0x7FFFFFF) + (high_count << 27) - ((shot_data->start_tags[1] >> 1) & 0x7FFFFFF);
			bool valid = true;
			while (valid) {
				//printf("%i\t%i\t%i\t", time_tag, shot_data->sorted_clock_tags[1][clock_pointer], shot_data->sorted_clock_tags[0][clock_pointer - 1]);
				//Increment dummy pointer if channel tag is greater than current start tag
				if ((time_tag >= shot_data->sorted_clock_tags[1][clock_pointer]) & (clock_pointer < shot_data->sorted_clock_tag_pointers[1])) {
					//printf("up clock pointer\n");
					clock_pointer++;
				}
				//Make sure clock_pointer is greater than 0, preventing an underflow error
				else if (clock_pointer > 0) {
					//Check if tag is lower than previous end tag i.e. startTags[j-1] < channeltags[i] < endTags[j-1]
					if (time_tag <= shot_data->sorted_clock_tags[0][clock_pointer - 1]) {
						//printf("add tag tot data\n");
						//Determine the index for given tag
						int channel_index;
						//Bin tag and assign to appropriate vector
						channel_index = shot_data->channel_map.find(((shot_data->photon_tags[i] >> 29) & 7) + 1)->second;
						shot_data->sorted_photon_tags[channel_index][shot_data->sorted_photon_tag_pointers[channel_index]] = time_tag;
						shot_data->sorted_photon_tag_pointers[channel_index]++;
						//printf("%i\t%i\t%i\n", channel_index, time_tag, shot_data->sorted_photon_tag_pointers[channel_index]);
					}
					//Break the valid loop
					valid = false;
				}
				// If tag is smaller than the first start tag
				else {
					valid = false;
				}
			}
		}
	}
}

//Converts our tags to bins with a given bin width
void tagsToBins(shotData *shot_data, double bin_width) {
	int tagger_bins_per_bin_width = (int)round(bin_width / tagger_resolution);
#pragma omp parallel for
	for (int channel = 0; channel < shot_data->sorted_photon_bins.size(); channel++) {
#pragma omp parallel for
		for (int i = 0; i < shot_data->sorted_photon_tag_pointers[channel]; i++) {
			shot_data->sorted_photon_bins[channel][i] = (shot_data->sorted_photon_tags[channel][i] + offset[channel]) / tagger_bins_per_bin_width;
		}
	}
	for (int slope = 0; slope <= 1; slope++) {
#pragma omp parallel for
		for (int i = 0; i < shot_data->sorted_clock_tag_pointers[slope]; i++) {
			shot_data->sorted_clock_bins[slope][i] = shot_data->sorted_clock_tags[slope][i] / tagger_bins_per_bin_width;
		}
	}
}

//Sorts photons and bins them for each file in a block
void sortAndBinBlock(std::vector<shotData> *shot_block, double bin_width, int num_devices, int block_size) {
#pragma omp parallel for
	for (int shot_file_num = 0; shot_file_num < (block_size * num_devices); shot_file_num++) {
		if ((*shot_block)[shot_file_num].file_load_completed) {
			sortTags(&(*shot_block)[shot_file_num]);
			tagsToBins(&(*shot_block)[shot_file_num], bin_width);
		}
	}
}

/*DLLEXPORT void getG3Correlations(char **file_list, int file_list_length, double max_time, double bin_width, double pulse_spacing, int max_pulse_distance, PyObject *numer, int32 *denom, bool calc_norm, int num_cpu_threads) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}

	int max_bin = (int)round(max_time / bin_width);
	int bin_pulse_spacing = (int)round(pulse_spacing / bin_width);

	int32 *coinc;
	coinc = (int32*)malloc((((2 * (max_bin)+1) * (2 * (max_bin)+1)) + (max_pulse_distance * 2) * (max_pulse_distance * 2)) * num_cpu_threads * sizeof(int32));

	for (int id = 0; id < (((2 * (max_bin)+1) * (2 * (max_bin)+1)) + (max_pulse_distance * 2) * (max_pulse_distance * 2)) * num_cpu_threads; id++) {
		coinc[id] = 0;
	}

	int blocks_req;
	if (file_list_length < (num_cpu_threads)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads)+1;
	}

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Max Time\tBin Width\tPulse Spacing\tMax Pulse Distance\n");
	printf("%fus\t%fns\t%fus\t%i\n", max_time * 1e6, bin_width * 1e9, pulse_spacing * 1e6, max_pulse_distance);

	std::vector<int32> denom_counts(num_cpu_threads, 0);

	//Processes files in blocks
	for (int block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads);

		//Processes files
#pragma omp parallel for num_threads(num_cpu_threads)
		for (int shot_file_num = 0; shot_file_num < num_cpu_threads; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				calculateNumer_g3(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, coinc, shot_file_num);
				if (calc_norm){
					calculateDenom_g3(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &(denom_counts[shot_file_num]), shot_file_num);
				}
			}
		}
		printf("Finished block %i of %i\n", block_num+1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for (int i = 0; i < num_cpu_threads; i++) {
		for (int j = 0; j < (((2 * (max_bin)+1) * (2 * (max_bin)+1)) + (max_pulse_distance * 2) * (max_pulse_distance * 2)); j++) {
			if (j < ((2 * (max_bin)+1) * (2 * (max_bin)+1))) {
				PyList_SetItem(numer, j, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[j + i * (((2 * (max_bin)+1) * (2 * (max_bin)+1)) + (max_pulse_distance * 2) * (max_pulse_distance * 2))]));
			}
		}
		denom[0] += denom_counts[i];
	}
	free(coinc);
}
*/

DLLEXPORT void getG3Correlations(char **file_list, int file_list_length, double max_time, double bin_width, double pulse_spacing, int max_pulse_distance, PyObject *numer, int32 *denom, bool calc_norm, int num_cpu_threads_files, int num_cpu_threads_proc) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}

	int max_bin = (int)round(max_time / bin_width);
	int bin_pulse_spacing = (int)round(pulse_spacing / bin_width);

	int32 *coinc;
	coinc = (int32*)malloc((((2 * (max_bin)+1) * (2 * (max_bin)+1))) * num_cpu_threads_files * num_cpu_threads_proc * sizeof(int32));

	for (int id = 0; id < (((2 * (max_bin)+1) * (2 * (max_bin)+1))) * num_cpu_threads_files * num_cpu_threads_proc; id++) {
		coinc[id] = 0;
	}

	int blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	std::vector<int32> denom_counts(num_cpu_threads_files, 0);

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Max Time\tBin Width\tPulse Spacing\tMax Pulse Distance\n");
	printf("%fus\t%fns\t%fus\t%i\n", max_time * 1e6, bin_width * 1e9, pulse_spacing * 1e6, max_pulse_distance);

	//Processes files in blocks
	for (int block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads_files);

		//Processes files
		#pragma omp parallel for num_threads(num_cpu_threads_files)
		for (int shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				calculateNumer_g3_new(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, coinc, shot_file_num,num_cpu_threads_proc);
				if (calc_norm) {
					calculateDenom_g3(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &(denom_counts[shot_file_num]), shot_file_num);
				}
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for (int i = 0; i < num_cpu_threads_files; i++) {
		for (int thread = 0; thread < num_cpu_threads_proc; thread++) {
			for (int j = 0; j < (((2 * (max_bin)+1) * (2 * (max_bin)+1))); j++) {
				PyList_SetItem(numer, j, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[j + thread * ((2 * (max_bin)+1) * (2 * (max_bin)+1)) + i * num_cpu_threads_proc * ((2 * (max_bin)+1) * (2 * (max_bin)+1))]));
			}
		}
		denom[0] += denom_counts[i];
	}
	free(coinc);
}

/*DLLEXPORT void getG2Correlations(char **file_list, int file_list_length, double max_time, double bin_width, double pulse_spacing, int max_pulse_distance, PyObject *numer, int32 *denom, bool calc_norm, int num_cpu_threads) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}

	int max_bin = (int)round(max_time / bin_width);
	int bin_pulse_spacing = (int)round(pulse_spacing / bin_width);

	int32 *coinc;
	coinc = (int32*)malloc(((2 * (max_bin)+1) + (max_pulse_distance * 2)) * num_cpu_threads * sizeof(int32));

	for (int id = 0; id < ((2 * (max_bin)+1) + (max_pulse_distance * 2)) * num_cpu_threads; id++) {
		coinc[id] = 0;
	}

	int blocks_req;
	if (file_list_length < (num_cpu_threads)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads)+1;
	}

	std::vector<int32> denom_counts(num_cpu_threads, 0);

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Max Time\tBin Width\tPulse Spacing\tMax Pulse Distance\n");
	printf("%fus\t%fns\t%fus\t%i\n", max_time * 1e6, bin_width * 1e9, pulse_spacing * 1e6, max_pulse_distance);

	//Processes files in blocks
	for (int block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads);

		//Processes files
#pragma omp parallel for num_threads(num_cpu_threads)
		for (int shot_file_num = 0; shot_file_num < num_cpu_threads; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				calculateNumer_g2(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, coinc, shot_file_num);
				if (calc_norm) {
					calculateDenom_g2(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &(denom_counts[shot_file_num]), shot_file_num);
				}
			}
		}
		printf("Finished block %i of %i\n", block_num+1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for (int i = 0; i < num_cpu_threads; i++) {
		for (int j = 0; j < ((2 * (max_bin)+1) + (max_pulse_distance * 2)); j++) {
			if (j < (2 * (max_bin)+1)) {
				PyList_SetItem(numer, j, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[j + i * ((2 * (max_bin)+1) + (max_pulse_distance * 2))]));
			}
		}
		denom[0] += denom_counts[i];
	}
	free(coinc);

}
*/

DLLEXPORT void getG2Correlations(char **file_list, int file_list_length, double max_time, double bin_width, double pulse_spacing, int max_pulse_distance, PyObject *numer, int32 *denom, bool calc_norm, int num_cpu_threads_files, int num_cpu_threads_proc) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}

	int max_bin = (int)round(max_time / bin_width);
	int bin_pulse_spacing = (int)round(pulse_spacing / bin_width);

	int32 *coinc;
	coinc = (int32*)malloc(((2 * (max_bin)+1)) * num_cpu_threads_files * num_cpu_threads_proc * sizeof(int32));
	for (int id = 0; id < ((2 * (max_bin)+1)) * num_cpu_threads_files * num_cpu_threads_proc; id++) {
		coinc[id] = 0;
	}

	int blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Max Time\tBin Width\tPulse Spacing\tMax Pulse Distance\n");
	printf("%fus\t%fns\t%fus\t%i\n", max_time * 1e6, bin_width * 1e9, pulse_spacing * 1e6, max_pulse_distance);

	std::vector<int32> denom_counts(num_cpu_threads_files,0);

	//Processes files in blocks
	for (int block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads_files);


		//Processes files
		#pragma omp parallel for num_threads(num_cpu_threads_files)
		for (int shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				calculateNumer_g2_new(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, coinc, shot_file_num, num_cpu_threads_proc);
				if (calc_norm) {
					calculateDenom_g2(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &(denom_counts[shot_file_num]), shot_file_num);
				}
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for (int i = 0; i < num_cpu_threads_files; i++) {
		for (int thread = 0; thread < num_cpu_threads_proc; thread++) {
			for (int j = 0; j < ((2 * (max_bin)+1) + (max_pulse_distance * 2)); j++) {
				if (j < (2 * (max_bin)+1)) {
					PyList_SetItem(numer, j, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[j + thread * ((2 * (max_bin)+1)) + i * num_cpu_threads_proc * ((2 * (max_bin)+1))]));
				}
			}
		}
		denom[0] += denom_counts[i];
	}
	free(coinc);

}

/*DLLEXPORT void getG2Correlations_pulse(char **file_list, int file_list_length, double max_time_tau, double bin_width, PyObject *numer, int32 *denom, int num_cpu_threads_files) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}

	int max_bin_tau = (int)round(max_time_tau / bin_width);


	int blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Max Time\tBin Width\n");
	printf("%fus\t%fns\n", max_time_tau * 1e6, bin_width * 1e9);

	int32 *coinc;
	coinc = (int32*)malloc((2 * (max_bin_tau)+1) * (2 * (max_bin_tau)+1) * num_cpu_threads_files * sizeof(int32));
	for (int id = 0; id < ((2 * (max_bin_tau)+1) * (2 * (max_bin_tau)+1) * num_cpu_threads_files); id++) {
		coinc[id] = 0;
	}
	
	std::vector<int32> denom_counts(num_cpu_threads_files, 0);

	//Processes files in blocks
	for (int block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads_files);

		//Processes files
		#pragma omp parallel for num_threads(num_cpu_threads_files)
		for (int shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				calculateNumer_g2_pulse(&(shot_block[shot_file_num]), &max_bin_tau, coinc, shot_file_num);
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for (int i = 0; i < num_cpu_threads_files; i++) {
		for (int j = 0; j < (2 * (max_bin_tau)+1) * (2 * (max_bin_tau)+1); j++) {
			PyList_SetItem(numer, j, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[j + i * (2 * (max_bin_tau)+1) * (2 * (max_bin_tau)+1)]));
		}
	}
	free(coinc);

}
*/

DLLEXPORT void getG2Correlations_pulse(char **file_list, int file_list_length, double min_tau_1, double max_tau_1, double min_tau_2, double max_tau_2, double bin_width, PyObject *numer, int32 *denom, int num_cpu_threads_files) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}

	int min_bin_tau_1 = (int)round(min_tau_1 / bin_width);
	int max_bin_tau_1 = (int)round(max_tau_1 / bin_width);
	int min_bin_tau_2 = (int)round(min_tau_2 / bin_width);
	int max_bin_tau_2 = (int)round(max_tau_2 / bin_width);

	int blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Tau_1\tTau_2\tBin Width\n");
	printf("%f to %fus\t%f to %fus\t%fns\n", min_tau_1 * 1e6, max_tau_1 * 1e6, min_tau_2 * 1e6, max_tau_2 * 1e6, bin_width * 1e9);

	int32 *coinc;
	coinc = (int32*)malloc((max_bin_tau_1-min_bin_tau_1 + 1) * (max_bin_tau_2 - min_bin_tau_2 + 1) * num_cpu_threads_files * sizeof(int32));
	for (int id = 0; id < ((max_bin_tau_1 - min_bin_tau_1 + 1) * (max_bin_tau_2 - min_bin_tau_2 + 1) * num_cpu_threads_files); id++) {
		coinc[id] = 0;
	}

	std::vector<int32> denom_counts(num_cpu_threads_files, 0);

	//Processes files in blocks
	for (int block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads_files);

		//Processes files
#pragma omp parallel for num_threads(num_cpu_threads_files)
		for (int shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				calculateNumer_g2_pulse_2(&(shot_block[shot_file_num]), &min_bin_tau_1, &max_bin_tau_1, &min_bin_tau_2, &max_bin_tau_2, coinc, shot_file_num);
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for (int i = 0; i < num_cpu_threads_files; i++) {
		for (int j = 0; j < (max_bin_tau_1 - min_bin_tau_1 + 1) * (max_bin_tau_2 - min_bin_tau_2 + 1); j++) {
			PyList_SetItem(numer, j, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[j + i * (max_bin_tau_1 - min_bin_tau_1 + 1) * (max_bin_tau_2 - min_bin_tau_2 + 1)]));
		}
	}
	free(coinc);

}