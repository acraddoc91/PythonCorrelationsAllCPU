import ctypes
import _ctypes
import multiprocessing
import os
import numpy as np
import scipy.io
#import wurlitzer
import time

tagger_resolution = 82.3e-12*2
num_gpu = 2

working_directory = '/usr/local/lib'
lib_name = 'pythonDLLCPU.so'

def file_list_to_ctypes(file_list,folder):
    len_list = len(file_list)
    str_array_type = ctypes.c_char_p * len_list
    str_array = str_array_type()
    for i, file in enumerate(file_list):
        str_array[i] = (folder+file).encode('utf-8')
    return str_array

def g2ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int), ctypes.c_int,]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def g2ToFile_new(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_new.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_new(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,4,4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def g3ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG3Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG3Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list).reshape((2*max_bin + 1),(2*max_bin + 1)),'denom':denom_ctypes.value,'tau':tau})

def g3ToFile_new(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG3Correlations_new.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG3Correlations_new(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,4,4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list).reshape((2*max_bin + 1),(2*max_bin + 1)),'denom':denom_ctypes.value,'tau':tau})

def g2ToFile_pulse(folder_name, file_out_name, max_tau, bin_width):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution

    int_max_tau = round(max_tau / int_bin_width) * int_bin_width
    max_tau_bin  = int(round(int_max_tau/int_bin_width))

    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_tau_bin + 1)*(2*max_tau_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_pulse.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_pulse(ctypes_file_list, num_files, int_max_tau, int_bin_width, numer_list, ctypes.byref(denom_ctypes),4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_tau_bin,max_tau_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list).reshape((2*max_tau_bin + 1),(2*max_tau_bin + 1)),'tau':tau})

def g2ToFile_pulse_2(folder_name, file_out_name, min_tau_1, max_tau_1, min_tau_2, max_tau_2, bin_width):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution

    int_min_tau_1 = round(min_tau_1 / int_bin_width) * int_bin_width
    min_tau_1_bin  = int(round(int_min_tau_1/int_bin_width))
    
    int_min_tau_2 = round(min_tau_2 / int_bin_width) * int_bin_width
    min_tau_2_bin  = int(round(int_min_tau_2/int_bin_width))

    int_max_tau_1 = round(max_tau_1 / int_bin_width) * int_bin_width
    max_tau_1_bin  = int(round(int_max_tau_1/int_bin_width))

    int_max_tau_2 = round(max_tau_2 / int_bin_width) * int_bin_width
    max_tau_2_bin  = int(round(int_max_tau_2/int_bin_width))

    num_files = len(file_list)
    numer_list = [int(0)]*(max_tau_1_bin - min_tau_1_bin + 1)*(max_tau_2_bin - min_tau_2_bin + 1)

    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_pulse_2.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_pulse_2(ctypes_file_list, num_files, int_min_tau_1, int_max_tau_1, int_min_tau_2, int_max_tau_2, int_bin_width, numer_list, ctypes.byref(denom_ctypes),4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau_1 = np.arange(min_tau_1_bin,max_tau_1_bin+1) * int_bin_width
    tau_2 = np.arange(min_tau_2_bin,max_tau_2_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list).reshape(max_tau_2_bin-min_tau_2_bin+1,max_tau_1_bin-min_tau_1_bin+1),'tau_1':tau_1,'tau_2':tau_2})
    
