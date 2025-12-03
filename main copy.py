import ctypes


lib = ctypes.CDLL('./sensor_sim.so')
lib.simulate_sensor_data.restype = ctypes.POINTER(ctypes.c_float)
lib.simulate_sensor_data.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int]
