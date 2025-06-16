# Digital Beamformer on GPU (CUDA Accelerated)

This project implements **real-time digital beamforming** using **CUDA and GPU parallelism**. Designed to process signals from phased array antennas, it simulates directional signal enhancement using both **time-domain** and **frequency-domain beamforming techniques**, leveraging the computational power of NVIDIA GPUs.

> Developed as part of the *ECE 612 – Real-Time Embedded Systems* course, George Mason University.

---

## Objective

To build a GPU-accelerated digital beamformer that:
- **Aligns signals from multiple antennas** using programmable phase shifts.
- Enhances signal strength in a specific direction while suppressing noise.
- Demonstrates both **time-domain** and **FFT-based frequency-domain** beamforming on CUDA.

---

## Technologies Used

- **CUDA (NVIDIA Compute Unified Device Architecture)**
- **cuFFT** for fast Fourier transforms
- **C/C++** with NVCC compiler
- **Visual Studio** (for Windows development)
- **Hopper GPU Cluster** (for Linux deployment and benchmarking)

---

## Beamforming Concepts

Beamforming works by adjusting the **phase** and **amplitude** of signals received from an antenna array to steer the resulting beam in a desired direction.

### Key Equations
- Signal Model:  
  `sin(2πf₀t + θ)` where `θ = n × D × T × 2π`  
- Phase Shift:  
  `φ_n = n × f₀ × t`

### Time-Domain Beamforming
- Delays are applied to align signals in time before combining them.

### Frequency-Domain Beamforming
- Signals are transformed using the FFT.
- Complex weights are applied per frequency bin to steer the beam.

---

## File Structure
```bash
DigitalBeamformerGPU/
├── beamformer.cu # Main CUDA implementation
├── Makefile # Build script for nvcc
├── report.pdf # Full project write-up
├── presentation.pptx # Final presentation
```

---

## How to Run

### Environment Setup
1. Install **CUDA Toolkit** from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. (Windows) Use Visual Studio with CUDA integration  
3. (Linux/Cluster) SSH into the system, run:
   ```bash
   module load cuda
   nvcc -o beamformer beamformer.cu
   ./beamformer <num_antennas> <carrier_freq> <time_step> <delay>

Example: ./beamformer 1024 2.4e9 0.001 0.000001

---

## CUDA Kernels

| Kernel                         | Description                                             |
|-------------------------------|---------------------------------------------------------|
| `generate_signals`            | Generates modulated input signals with phase delays     |
| `time_domain_beamforming`     | Applies programmable phase shifts directly in the time domain |
| `frequency_domain_beamforming`| Applies complex weights after FFT for directional control|

Also includes robust macros for error handling:
- `gpuErrorCheck()` – Validates CUDA runtime execution
- `cufftErrorCheck()` – Checks cuFFT API calls for successful execution

---

## Results

- **100 Antennas** successfully beamformed in both time and frequency domains
- **Execution time** (time-domain): `0.025 ms`
- **Kernel execution time**: `0.024 ms` on CUDA-enabled GPU
- **GPU Utilization**: Efficient even with 100+ element arrays
- The output waveform shows high clarity and directionality

---

## Applications

- Radar & sonar signal enhancement
- 5G/6G wireless base station beamforming
- Autonomous vehicle object tracking
- Sensor fusion in large IoT arrays

---

## License

This project is for academic and research demonstration purposes only.  


---


