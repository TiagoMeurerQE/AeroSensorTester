all:
    gcc -shared -o sensor_sim.so -fPIC sensor_sim.c -lm