# cuda-aware-mpi-cluster

This is a tutorial how to set up a CUDA-aware MPI (mini) cluster 
for testing and development.

If successfull, MPI API calls should have directed access to the GPU
memory, bypassing the need to copy the data to RAM with CPU.  That 
should significantly reduce simulation time of distributed quantum state.

# References

* [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)


# Hardware

Virtual machine on Google Cloud.

Specs:

* 12 CPU (x86_64)
* 78 GB RAM (64 GB to cover GPU memory + 12 GB)
* 4 NVIDIA Tesla T4 (16 GB GDDR6 memory each)
* 128 GB SSD 
* Ubuntu 22.04 LTS  

Estimated cost: 1.86 USD / hour (on demand).


# Dependencies

## C/C++ toolchain and CMake-3.22:

```bash
sudo apt install gcc g++ build-essential
sudo apt install cmake
```

Compiler version info:

```bash
gcc -v
```

gives:

```text
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/lib/gcc/x86_64-linux-gnu/11/lto-wrapper
OFFLOAD_TARGET_NAMES=nvptx-none:amdgcn-amdhsa
OFFLOAD_TARGET_DEFAULT=1
Target: x86_64-linux-gnu
Configured with: ../src/configure -v --with-pkgversion='Ubuntu 11.4.0-1ubuntu1~22.04' --with-bugurl=file:///usr/share/doc/gcc-11/README.Bugs --enable-languages=c,ada,c++,go,brig,d,fortran,objc,obj-c++,m2 --prefix=/usr --with-gcc-major-version-only --program-suffix=-11 --program-prefix=x86_64-linux-gnu- --enable-shared --enable-linker-build-id --libexecdir=/usr/lib --without-included-gettext --enable-threads=posix --libdir=/usr/lib --enable-nls --enable-bootstrap --enable-clocale=gnu --enable-libstdcxx-debug --enable-libstdcxx-time=yes --with-default-libstdcxx-abi=new --enable-gnu-unique-object --disable-vtable-verify --enable-plugin --enable-default-pie --with-system-zlib --enable-libphobos-checking=release --with-target-system-zlib=auto --enable-objc-gc=auto --enable-multiarch --disable-werror --enable-cet --with-arch-32=i686 --with-abi=m64 --with-multilib-list=m32,m64,mx32 --enable-multilib --with-tune=generic --enable-offload-targets=nvptx-none=/build/gcc-11-XeT9lY/gcc-11-11.4.0/debian/tmp-nvptx/usr,amdgcn-amdhsa=/build/gcc-11-XeT9lY/gcc-11-11.4.0/debian/tmp-gcn/usr --without-cuda-driver --enable-checking=release --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu --with-build-config=bootstrap-lto-lean --enable-link-serialization=2
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)
```

## CUDA

We follow the [official guide].  Quotes from the guide in italics.

*To verify that your GPU is CUDA-capable, go to your distribution’s equivalent of System Properties, or, from the command line, enter:*

```bash
lspci | grep -i nvidia
```

We get:

```text
00:04.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
00:05.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
00:06.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
00:07.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
```

*The version of the kernel your system is running can be found by running the following command:*

```bash
uname -m && cat /etc/*release
```

We get:

```text
6.5.0-1016-gcp
```

Download and install CUDA Toolkit.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

*To install the open kernel module flavor:*

```bash
sudo apt-get install -y nvidia-driver-550-open
sudo apt-get install -y cuda-drivers-550
```

To install CUDA, we follow *Distribution-specific instructions detail how to install CUDA* for Ubuntu.

*The kernel headers and development packages for the currently running kernel can be installed with:*

```bash
sudo apt-get install linux-headers-$(uname -r)
sudo apt-key del 7fa2af80
```

Download and install CUDA sources:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

```bash
sudo apt-get install cuda-toolkit
```

Finally:

```bash
sudo reboot
```


Add at the end of `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

and type:

```bash
source .bashrc
```

Get and build [CUDA Samples][CUDA-samples]:

```bash
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
make -j 8
```

*After compilation, find and run `deviceQuery`*.

Our output:

```text
./bin/x86_64/linux/release/deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 4 CUDA Capable device(s)

Device 0: "Tesla T4"
  CUDA Driver Version / Runtime Version          12.4 / 12.4
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 14918 MBytes (15642329088 bytes)
  (040) Multiprocessors, (064) CUDA Cores/MP:    2560 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 4
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

Device 1: "Tesla T4"
  CUDA Driver Version / Runtime Version          12.4 / 12.4
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 14918 MBytes (15642329088 bytes)
  (040) Multiprocessors, (064) CUDA Cores/MP:    2560 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 5
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

Device 2: "Tesla T4"
  CUDA Driver Version / Runtime Version          12.4 / 12.4
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 14918 MBytes (15642329088 bytes)
  (040) Multiprocessors, (064) CUDA Cores/MP:    2560 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 6
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

Device 3: "Tesla T4"
  CUDA Driver Version / Runtime Version          12.4 / 12.4
  CUDA Capability Major/Minor version number:    7.5
  Total amount of global memory:                 14918 MBytes (15642329088 bytes)
  (040) Multiprocessors, (064) CUDA Cores/MP:    2560 CUDA Cores
  GPU Max Clock rate:                            1590 MHz (1.59 GHz)
  Memory Clock rate:                             5001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1024
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 7
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
> Peer access from Tesla T4 (GPU0) -> Tesla T4 (GPU1) : Yes
> Peer access from Tesla T4 (GPU0) -> Tesla T4 (GPU2) : No
> Peer access from Tesla T4 (GPU0) -> Tesla T4 (GPU3) : No
> Peer access from Tesla T4 (GPU1) -> Tesla T4 (GPU0) : Yes
> Peer access from Tesla T4 (GPU1) -> Tesla T4 (GPU2) : No
> Peer access from Tesla T4 (GPU1) -> Tesla T4 (GPU3) : No
> Peer access from Tesla T4 (GPU2) -> Tesla T4 (GPU0) : No
> Peer access from Tesla T4 (GPU2) -> Tesla T4 (GPU1) : No
> Peer access from Tesla T4 (GPU2) -> Tesla T4 (GPU3) : Yes
> Peer access from Tesla T4 (GPU3) -> Tesla T4 (GPU0) : No
> Peer access from Tesla T4 (GPU3) -> Tesla T4 (GPU1) : No
> Peer access from Tesla T4 (GPU3) -> Tesla T4 (GPU2) : Yes

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.4, CUDA Runtime Version = 12.4, NumDevs = 4
Result = PASS
```

Install recommended dependencies:

```bash
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa-dev libfreeimage-dev libglfw3-dev
```


[official guide]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
[CUDA-samples]: https://github.com/nvidia/cuda-samples


## GDRCopy

Build and install [GDRCopy].

```bash
sudo apt install build-essential devscripts debhelper fakeroot pkg-config dkms
git clone https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
```

```bash
sudo make prefix=/usr/local/ CUDA=/usr/local/cuda all install
sudo ./insmod.sh
```

[GDRCopy]: https://github.com/NVIDIA/gdrcopy

## UCX

Build and install [UCX].

```bash
git clone https://github.com/openucx/ucx.git
cd ucx
```

```bash
./autogen.sh
./contrib/configure-release --prefix=/usr/local --with-cuda=/usr/local/cuda --with-gdrcopy=/usr/local
make -j8
sudo make install
```

[UCX]: https://github.com/openucx/ucx


## OpenMPI

Build and install [Open MPI]. 

*Note:* The latest version of Open MPI, v5.0, can't detect CUDA installation. We
fall back to v4.1.6.

Get the sources:

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar xzv < openmpi-4.1.6.tar.gz
cd openmpi-4.1.6/
```

Follow the official [Open MPI CUDA guide].

```bash
./configure --with-cuda=/usr/local/cuda/ --with-ucx=/usr/local/
make -j8
sudo make install
sudo ldconfig
```

Verify the installation following [this guide][ompi-cuda-faq].

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

Our output:

```text
mca:mpi:base:param:mpi_built_with_cuda_support:value:true
```

All info: `ompi_info`:

```text
                 Package: Open MPI mm@sn-gpu4t4 Distribution
                Open MPI: 4.1.6
  Open MPI repo revision: v4.1.6
   Open MPI release date: Sep 30, 2023
                Open RTE: 4.1.6
  Open RTE repo revision: v4.1.6
   Open RTE release date: Sep 30, 2023
                    OPAL: 4.1.6
      OPAL repo revision: v4.1.6
       OPAL release date: Sep 30, 2023
                 MPI API: 3.1.0
            Ident string: 4.1.6
                  Prefix: /usr/local
 Configured architecture: x86_64-pc-linux-gnu
          Configure host: sn-gpu4t4
           Configured by: mm
           Configured on: Fri Mar 22 00:43:58 UTC 2024
          Configure host: sn-gpu4t4
  Configure command line: '--with-cuda=/usr/local/cuda/'
                          '--with-ucx=/usr/local/'
                Built by: mm
                Built on: Fri Mar 22 00:49:14 UTC 2024
              Built host: sn-gpu4t4
              C bindings: yes
            C++ bindings: no
             Fort mpif.h: no
            Fort use mpi: no
       Fort use mpi size: deprecated-ompi-info-value
        Fort use mpi_f08: no
 Fort mpi_f08 compliance: The mpi_f08 module was not built
  Fort mpi_f08 subarrays: no
           Java bindings: no
  Wrapper compiler rpath: runpath
              C compiler: gcc
     C compiler absolute: /usr/bin/gcc
  C compiler family name: GNU
      C compiler version: 11.4.0
            C++ compiler: g++
   C++ compiler absolute: /usr/bin/g++
           Fort compiler: none
       Fort compiler abs: none
         Fort ignore TKR: no
   Fort 08 assumed shape: no
      Fort optional args: no
          Fort INTERFACE: no
    Fort ISO_FORTRAN_ENV: no
       Fort STORAGE_SIZE: no
      Fort BIND(C) (all): no
      Fort ISO_C_BINDING: no
 Fort SUBROUTINE BIND(C): no
       Fort TYPE,BIND(C): no
 Fort T,BIND(C,name="a"): no
            Fort PRIVATE: no
          Fort PROTECTED: no
           Fort ABSTRACT: no
       Fort ASYNCHRONOUS: no
          Fort PROCEDURE: no
         Fort USE...ONLY: no
           Fort C_FUNLOC: no
 Fort f08 using wrappers: no
         Fort MPI_SIZEOF: no
             C profiling: yes
           C++ profiling: no
   Fort mpif.h profiling: no
  Fort use mpi profiling: no
   Fort use mpi_f08 prof: no
          C++ exceptions: no
          Thread support: posix (MPI_THREAD_MULTIPLE: yes, OPAL support: yes,
                          OMPI progress: no, ORTE progress: yes, Event lib:
                          yes)
           Sparse Groups: no
  Internal debug support: no
  MPI interface warnings: yes
     MPI parameter check: runtime
Memory profiling support: no
Memory debugging support: no
              dl support: yes
   Heterogeneous support: no
 mpirun default --prefix: no
       MPI_WTIME support: native
     Symbol vis. support: yes
   Host topology support: yes
            IPv6 support: no
      MPI1 compatibility: no
          MPI extensions: affinity, cuda, pcollreq
   FT Checkpoint support: no (checkpoint thread: no)
   C/R Enabled Debugging: no
  MPI_MAX_PROCESSOR_NAME: 256
    MPI_MAX_ERROR_STRING: 256
     MPI_MAX_OBJECT_NAME: 64
        MPI_MAX_INFO_KEY: 36
        MPI_MAX_INFO_VAL: 256
       MPI_MAX_PORT_NAME: 1024
  MPI_MAX_DATAREP_STRING: 128
           MCA allocator: bucket (MCA v2.1.0, API v2.0.0, Component v4.1.6)
           MCA allocator: basic (MCA v2.1.0, API v2.0.0, Component v4.1.6)
           MCA backtrace: execinfo (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA btl: self (MCA v2.1.0, API v3.1.0, Component v4.1.6)
                 MCA btl: tcp (MCA v2.1.0, API v3.1.0, Component v4.1.6)
                 MCA btl: vader (MCA v2.1.0, API v3.1.0, Component v4.1.6)
                 MCA btl: smcuda (MCA v2.1.0, API v3.1.0, Component v4.1.6)
            MCA compress: gzip (MCA v2.1.0, API v2.0.0, Component v4.1.6)
            MCA compress: bzip (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA crs: none (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                  MCA dl: dlopen (MCA v2.1.0, API v1.0.0, Component v4.1.6)
               MCA event: libevent2022 (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
               MCA hwloc: hwloc201 (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                  MCA if: linux_ipv6 (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
                  MCA if: posix_ipv4 (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
         MCA installdirs: env (MCA v2.1.0, API v2.0.0, Component v4.1.6)
         MCA installdirs: config (MCA v2.1.0, API v2.0.0, Component v4.1.6)
              MCA memory: patcher (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA mpool: hugepage (MCA v2.1.0, API v3.0.0, Component v4.1.6)
             MCA patcher: overwrite (MCA v2.1.0, API v1.0.0, Component
                          v4.1.6)
                MCA pmix: flux (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA pmix: isolated (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA pmix: pmix3x (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA pstat: linux (MCA v2.1.0, API v2.0.0, Component v4.1.6)
              MCA rcache: gpusm (MCA v2.1.0, API v3.3.0, Component v4.1.6)
              MCA rcache: grdma (MCA v2.1.0, API v3.3.0, Component v4.1.6)
              MCA rcache: rgpusm (MCA v2.1.0, API v3.3.0, Component v4.1.6)
           MCA reachable: weighted (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA shmem: sysv (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA shmem: mmap (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA shmem: posix (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA timer: linux (MCA v2.1.0, API v2.0.0, Component v4.1.6)
              MCA errmgr: default_tool (MCA v2.1.0, API v3.0.0, Component
                          v4.1.6)
              MCA errmgr: default_orted (MCA v2.1.0, API v3.0.0, Component
                          v4.1.6)
              MCA errmgr: default_app (MCA v2.1.0, API v3.0.0, Component
                          v4.1.6)
              MCA errmgr: default_hnp (MCA v2.1.0, API v3.0.0, Component
                          v4.1.6)
                 MCA ess: env (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA ess: hnp (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA ess: slurm (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA ess: tool (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA ess: singleton (MCA v2.1.0, API v3.0.0, Component
                          v4.1.6)
                 MCA ess: pmi (MCA v2.1.0, API v3.0.0, Component v4.1.6)
               MCA filem: raw (MCA v2.1.0, API v2.0.0, Component v4.1.6)
             MCA grpcomm: direct (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA iof: orted (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA iof: hnp (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA iof: tool (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA odls: default (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA odls: pspawn (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA oob: tcp (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA plm: slurm (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA plm: rsh (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA plm: isolated (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA ras: simulator (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
                 MCA ras: slurm (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA regx: reverse (MCA v2.1.0, API v1.0.0, Component v4.1.6)
                MCA regx: naive (MCA v2.1.0, API v1.0.0, Component v4.1.6)
                MCA regx: fwd (MCA v2.1.0, API v1.0.0, Component v4.1.6)
               MCA rmaps: seq (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA rmaps: ppr (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA rmaps: mindist (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA rmaps: round_robin (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
               MCA rmaps: resilient (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
               MCA rmaps: rank_file (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
                 MCA rml: oob (MCA v2.1.0, API v3.0.0, Component v4.1.6)
              MCA routed: direct (MCA v2.1.0, API v3.0.0, Component v4.1.6)
              MCA routed: binomial (MCA v2.1.0, API v3.0.0, Component v4.1.6)
              MCA routed: radix (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA rtc: hwloc (MCA v2.1.0, API v1.0.0, Component v4.1.6)
              MCA schizo: orte (MCA v2.1.0, API v1.0.0, Component v4.1.6)
              MCA schizo: flux (MCA v2.1.0, API v1.0.0, Component v4.1.6)
              MCA schizo: ompi (MCA v2.1.0, API v1.0.0, Component v4.1.6)
              MCA schizo: slurm (MCA v2.1.0, API v1.0.0, Component v4.1.6)
              MCA schizo: jsm (MCA v2.1.0, API v1.0.0, Component v4.1.6)
               MCA state: novm (MCA v2.1.0, API v1.0.0, Component v4.1.6)
               MCA state: tool (MCA v2.1.0, API v1.0.0, Component v4.1.6)
               MCA state: app (MCA v2.1.0, API v1.0.0, Component v4.1.6)
               MCA state: orted (MCA v2.1.0, API v1.0.0, Component v4.1.6)
               MCA state: hnp (MCA v2.1.0, API v1.0.0, Component v4.1.6)
                 MCA bml: r2 (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: sync (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: cuda (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: sm (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: han (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: basic (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: adapt (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: self (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: tuned (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: libnbc (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA coll: monitoring (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
                MCA coll: inter (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                MCA fbtl: posix (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA fcoll: vulcan (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA fcoll: dynamic (MCA v2.1.0, API v2.0.0, Component v4.1.6)
               MCA fcoll: two_phase (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
               MCA fcoll: dynamic_gen2 (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
               MCA fcoll: individual (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
                  MCA fs: ufs (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                  MCA io: ompio (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                  MCA io: romio321 (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                  MCA op: avx (MCA v2.1.0, API v1.0.0, Component v4.1.6)
                 MCA osc: rdma (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA osc: sm (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA osc: ucx (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA osc: pt2pt (MCA v2.1.0, API v3.0.0, Component v4.1.6)
                 MCA osc: monitoring (MCA v2.1.0, API v3.0.0, Component
                          v4.1.6)
                 MCA pml: v (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA pml: ob1 (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA pml: monitoring (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
                 MCA pml: cm (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA pml: ucx (MCA v2.1.0, API v2.0.0, Component v4.1.6)
                 MCA rte: orte (MCA v2.1.0, API v2.0.0, Component v4.1.6)
            MCA sharedfp: sm (MCA v2.1.0, API v2.0.0, Component v4.1.6)
            MCA sharedfp: individual (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
            MCA sharedfp: lockedfile (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
                MCA topo: basic (MCA v2.1.0, API v2.2.0, Component v4.1.6)
                MCA topo: treematch (MCA v2.1.0, API v2.2.0, Component
                          v4.1.6)
           MCA vprotocol: pessimist (MCA v2.1.0, API v2.0.0, Component
                          v4.1.6)
```

[Open MPI]: https://www-lb.open-mpi.org/
[Open MPI CUDA guide]: https://www-lb.open-mpi.org/faq/?category=buildcuda
[ompi-cuda-faq]: https://www-lb.open-mpi.org/faq/?category=runcuda


# Benchmark

Follow an example from NVIDIA blog: *[Benchmarking CUDA-Aware MPI][NVIDIA blog]*.

Download the sources:

```bash
git clone https://github.com/NVIDIA-developer-blog/code-samples.git
cd code-samples/posts/cuda-aware-mpi-example/src/
```

Modify the file: `Jacobi.h`. Change the line:

```c
#define ENV_LOCAL_RANK               "MV2_COMM_WORLD_LOCAL_RANK"
```

to:

```c
#define ENV_LOCAL_RANK          "OMPI_COMM_WORLD_LOCAL_RANK"
```

Compile the source code:

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
export MPI_HOME=/usr/local/
make
cd ../bin
```

Run the program in normal MPI mode:

```bash
mpirun -n 4 ./jacobi_cuda_normal_mpi -t 2 2
```

Our output:

```text
Topology size: 2 x 2
Local domain size (current node): 4096 x 4096
Global domain size (all nodes): 8192 x 8192
Starting Jacobi run with 4 processes using "Tesla T4" GPUs (ECC enabled: 4 / 4):
Iteration: 0 - Residue: 0.250000
Iteration: 100 - Residue: 0.002397
Iteration: 200 - Residue: 0.001204
Iteration: 300 - Residue: 0.000804
Iteration: 400 - Residue: 0.000603
Iteration: 500 - Residue: 0.000483
Iteration: 600 - Residue: 0.000403
Iteration: 700 - Residue: 0.000345
Iteration: 800 - Residue: 0.000302
Iteration: 900 - Residue: 0.000269
Stopped after 1000 iterations with residue 0.000242
Total Jacobi run time: 2.8496 sec.
Average per-process communication time: 0.1287 sec.
Measured lattice updates: 23.54 GLU/s (total), 5.88 GLU/s (per process)
Measured FLOPS: 117.69 GFLOPS (total), 29.42 GFLOPS (per process)
Measured device bandwidth: 1.51 TB/s (total), 376.62 GB/s (per process)
```

And in CUDA-aware MPI mode:

```bash
mpirun -n 4 ./jacobi_cuda_aware_mpi -t 2 2
```

Output:

```text
Topology size: 2 x 2
Local domain size (current node): 4096 x 4096
Global domain size (all nodes): 8192 x 8192
Starting Jacobi run with 4 processes using "Tesla T4" GPUs (ECC enabled: 4 / 4):
Iteration: 0 - Residue: 0.250000
Iteration: 100 - Residue: 0.002397
Iteration: 200 - Residue: 0.001204
Iteration: 300 - Residue: 0.000804
Iteration: 400 - Residue: 0.000603
Iteration: 500 - Residue: 0.000483
Iteration: 600 - Residue: 0.000403
Iteration: 700 - Residue: 0.000345
Iteration: 800 - Residue: 0.000302
Iteration: 900 - Residue: 0.000269
Stopped after 1000 iterations with residue 0.000242
Total Jacobi run time: 3.0188 sec.
Average per-process communication time: 0.2865 sec.
Measured lattice updates: 22.22 GLU/s (total), 5.55 GLU/s (per process)
Measured FLOPS: 111.10 GFLOPS (total), 27.77 GFLOPS (per process)
Measured device bandwidth: 1.42 TB/s (total), 355.51 GB/s (per process)
```


What's important here is not the bandwidth (see the blog post for explanation),
but rather that the CUDA-aware program doesn't segfault.  If MPI didn't have
direct access to the GPU memory, the pointers to CUDA arrays would have been
invalid. 🎆 🎉


[NVIDIA blog]: https://developer.nvidia.com/blog/benchmarking-cuda-aware-mpi/


