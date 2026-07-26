<!-- ![CI](https://github.com/KULeuven-MICAS/snax_cluster/actions/workflows/ci.yml/badge.svg)-->
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

# Overview

This repository hosts the hardware and base software for paper MX-like Tensor Core for LLM Inference with NPU Integration via Hardware-Software Co-design Exploration. This work is based on SNAX, a high-efficiency compute cluster platform designed for multi-accelerator development. It provides a standard SNAX shell for conveniently integrating versatile accelerators. Moreover, it contains a variety of accelerators that are available for use. More details available on [KULeuven-MICAS/snax_cluster](https://github.com/KULeuven-MICAS/snax_cluster).


<!-- SNAX supports several design-time and run-time configurations to support a variety of accelerators. For example, some design-time configurations include customizing the memory sizes, the interconnect structure connecting the accelerators to memory, the number of Snitch cores controlling accelerators, and so much more. For run-time configurations, we provide data streamers and reshufflers to aid accelerators in organizing data layouts in memory and managing data access patterns.

* SNAX is being co-developed with an MLIR-based toolchain, available at [KULeuven-MICAS/snax-mlir](https://github.com/KULeuven-MICAS/snax-mlir).
* SNAX clusters are also integrated in a CVA6 host system available at [KULeuven-MICAS/HeMAia](https://github.com/KULeuven-MICAS/HeMAiA) -->

## Directory Structure

The project is organized as a monolithic repository. Both hardware and software are co-located. Standalone accelerators can have their own repositories and can be imported here using [Bender](https://github.com/pulp-platform/bender).

The file tree is visualized as follows:

```
├── hw
│   ├── chisel
│   │   ├── csr_manager
│   │   └── streamer
│   ├── snax_accelerator_1
│   ├── snax_accelerator_2
│   ├── mx_like_tensor_core(Ours)
│   ├── snitch_stuff
│   └── templates
├── sw
├── target
│   └── snitch_cluster
│       ├── cfg
│       ├── generated
│       └── sw
│           ├── apps
│           │   ├── snax_system_1
│           │   └── snax_system_2
│           │   └── snax_mx(Ours)
│           └── snax_lib 
│               ├── snax_system_1
│               └── snax_system_2
│           │   └── snax_mx(Ours)
└── util
    ├── clustergen
    └── wrappergen
```

The top-level is structured as follows:

* `docs`: [Documentation](documentation.md) of the generator and software.
  Contains additional user guides.
* `hw`: All hardware IP components. The source files are either specified by SystemVerilog, Chisel, or a template to generate these files.
* `sw`: Hardware independent software, libraries, runtimes, etc.
* `target`: Contains the testbench setup, cluster configuration specific hardware and software, libraries, runtimes, etc.
* `util`: Utility and helper scripts.

## Getting Started

### Detailed Guides of SNAX cluster
It is recommended to go through the SNAX guide first, as the co-design flow is built upon it. It is also very helpful to finish the small exercise at the end of the below tutorial. 

There is a detailed [getting started guide](https://kuleuven-micas.github.io/snax_cluster/ug/getting_started.html) that contains several information about installation and directory setup.

There is a [detailed tutorial](https://kuleuven-micas.github.io/snax_cluster/tutorial/introduction.html) on integrating a custom accelerator into the SNAX cluster platform. 


### Workflow

As a quick start, it is recommended to download SNAX pre-built docker container.

1 ) Clone the repository and go inside it after:

``` bash
git clone https://github.com/KULeuven-MICAS/snax_cluster.git --recurse-submodules
```

Note: If you cloned the repository already but without the recursed submodules, update the summodules inside the repository:

```bash
git submodule update --init --recursive
```

2 ) Download the docker:

``` bash
docker pull ghcr.io/kuleuven-micas/snax:main
```

3 ) Start the container, we recommend to run it in Linux and WSL:

```bash
podman run -it -v `pwd`:/repo -w /repo ghcr.io/kuleuven-micas/snax:main
```

Alternatively, you can use map the real path of your working `snax_cluster` directory:

```bash
podman run -it -v `pwd`:`pwd` -w `pwd` ghcr.io/kuleuven-micas/snax:main
```

4 ) Generate all RTL files of your chosen accelerator. First, configure your own accelerator and SNAX components under `snax_cluster/target/snitch_cluster/cfg/{your_accelerator}.hjson.` In this work, this file is auto-generated.

Then, run the below command to generate RTL: 

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson rtl-gen
```

All the generated components can be found under `snax_cluster/target/snitch_cluster/generated/.`

In here and below, you can replace snax_mx_cluster.hjson with your own settings.

5 ) Build the HW. The `-j` option parallelizes the build. The command below builds a Verilator binary for simulation. Before running this command, you should check Bender.yml to make sure all needed RTL are in the settings.

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson bin/snitch_cluster.vlt -j
```


6 ) Build the SW. The `-j` option parallelizes the build.

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson sw -j
```

7 ) Run the built SW of the accelerator on the built Verilator binary. The `--vcd` is optional to dump a vcd file to be viewed with `gtkwave`. A `sim.vcd` file gets dump from where you ran the command.

```bash
./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-alu/build/snax-mx.elf --vcd
```

You should see a log like:

```bash
VCD wave generation enabled
[fesvr] Wrote 36 bytes of bootrom to 0x1000
[fesvr] Wrote entry point 0x80000000 to bootloader slot 0x1020
[fesvr] Wrote 72 bytes of bootdata to 0x1024
[Tracer] Logging Hart          0 to logs/trace_chip_00_hart_00000.dasm
[Tracer] Logging Hart          1 to logs/trace_chip_00_hart_00001.dasm
Accelerator Done! 
Accelerator Cycles: xx 
Number of errors: x 
```

You can view the waveform with:

```bash
gtkwave sim.vcd
```

PS: note that the --vcd may slow down the process. And if you are using printf to debug, the vcd file might be several GBs depending on the workload.


# Publications
[TBD]  

# Acknowledgement
[TBD]
SNAX is an extension of the original [Snitch Cluster](https://github.com/pulp-platform/snitch_cluster) framework. SNAX got its name as "SNitch Acceleration eXtension".
