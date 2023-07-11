device_function = r"""
FLAMEGPU_AGENT_FUNCTION(device_function, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float prop_float = FLAMEGPU->environment.getProperty<float>("float");
    const int16_t prop_int16 = FLAMEGPU->environment.getProperty<int16_t>("int16_t");
    const uint64_t prop_uint64_0 = FLAMEGPU->environment.getProperty<uint64_t, 3>("uint64_t", 0);
    const uint64_t prop_uint64_1 = FLAMEGPU->environment.getProperty<uint64_t, 3>("uint64_t", 1);
    const uint64_t prop_uint64_2 = FLAMEGPU->environment.getProperty<uint64_t, 3>("uint64_t", 2);
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        printf("Agent Function[Thread 0]! Properties(Float: %g, int16: %hd, uint64[3]: {%llu, %llu, %llu})\n", prop_float, prop_int16, prop_uint64_0, prop_uint64_1, prop_uint64_2);
    }
    return flamegpu::ALIVE;
}
"""