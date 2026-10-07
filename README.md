[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# MX-like Tensor Core for LLM Inference

## Overview

This repository contains the hardware and base software accompanying the paper *"MX-like Tensor Core for LLM Inference with NPU Integration via Hardware-Software Co-Design Exploration"*. This work builds on SNAX, a high-efficiency compute cluster platform designed for multi-accelerator development. SNAX provides a standard shell for integrating a variety of accelerators. For more information about the underlying platform, see [KULeuven-MICAS/snax_cluster](https://github.com/KULeuven-MICAS/snax_cluster).


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
│   ├── mx_like_tensor_core
│   ├── snax_mx_alu
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
│           │   ├── snax_system_2
│           │   └── snax-mx
│           └── snax
│               ├── snax_system_1
│               ├── snax_system_2
│               └── snax-mx
├── util
│   ├── clustergen
│   └── wrappergen
└── docs
```

The MX-specific components introduced by this work are located in
`hw/mx_like_tensor_core`, `hw/snax_mx_alu`,
`target/snitch_cluster/sw/apps/snax-mx`, and
`target/snitch_cluster/sw/snax/snax-mx`.

The top-level is structured as follows:

* `docs`: Project documentation and additional user guides.
  See the [documentation index](docs/index.md).
* `hw`: All hardware IP components. The source files are either specified by SystemVerilog, Chisel, or a template to generate these files.
* `sw`: Hardware independent software, libraries, runtimes, etc.
* `target`: Contains the testbench setup, cluster configuration specific hardware and software, libraries, runtimes, etc.
* `util`: Utility and helper scripts.

## Getting Started

### Detailed Guides for the SNAX Cluster

We recommend reading the SNAX guide first because this co-design flow builds on the SNAX platform. The exercise at the end of the tutorial provides a useful introduction to the accelerator integration flow.

The [getting started guide](https://kuleuven-micas.github.io/snax_cluster/ug/getting_started.html) provides information about installation and repository setup.

The [accelerator integration tutorial](https://kuleuven-micas.github.io/snax_cluster/tutorial/introduction.html) explains how to integrate a custom accelerator into the SNAX cluster platform.

### Workflow

For a quick start, we recommend using the pre-built SNAX Docker container.

1. Clone the repository and enter the project directory:

```bash
git clone --recurse-submodules \
  https://github.com/MargZhao/MX-like-Tensor-Core-for-LLM-Inference-with-NPU-Integration-via-HW-SW-Co-Design-Exploration.git \
  snax-mx
cd snax-mx
```
<!-- TODO(RELEASE MAINTAINER):
Replace the GitHub clone URL with the final institutional repository URL.
-->

If the repository was cloned without its submodules, initialize them with:

```bash
git submodule update --init --recursive
```

2. Pull the Docker image:

```bash
docker pull ghcr.io/kuleuven-micas/snax:main
```

3. Start the container from the repository root. Linux and WSL are the recommended environments:

```bash
docker run --rm -it \
  -v "$(pwd):/repo" \
  -w /repo \
  ghcr.io/kuleuven-micas/snax:main
```

To use Podman instead, replace `docker` with `podman` in both commands above.

4. Use the provided MX cluster configuration at
`target/snitch_cluster/cfg/snax_mx_cluster.hjson`.

Generate the RTL files from the repository root:

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson rtl-gen
```

The generated components are placed under `target/snitch_cluster/generated/`.

To use a custom configuration, place it under `target/snitch_cluster/cfg/` and replace `snax_mx_cluster.hjson` in the commands below with the corresponding file name.

5. Build the hardware. The `-j` option parallelizes the build. The command below builds a Verilator binary for simulation. When using a custom accelerator, ensure that all required RTL sources are listed in `Bender.yml`.

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson bin/snitch_cluster.vlt -j
```

6. Build the software. The `-j` option parallelizes the build.

```bash
make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_mx_cluster.hjson sw -j
```

7. Run the accelerator software on the Verilator simulator. The optional `--vcd` flag generates a `sim.vcd` waveform file in the current directory for inspection with GTKWave.

```bash
./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-mx/build/snax-mx.elf --vcd
```

You should see a log like:

```text
VCD wave generation enabled
[fesvr] Wrote 36 bytes of bootrom to 0x1000
[fesvr] Wrote entry point 0x80000000 to bootloader slot 0x1020
[fesvr] Wrote 72 bytes of bootdata to 0x1024
[Tracer] Logging Hart          0 to logs/trace_chip_00_hart_00000.dasm
[Tracer] Logging Hart          1 to logs/trace_chip_00_hart_00001.dasm
[some debug infos...]
Accelerator Done! 
Accelerator Cycles: xx 
Number of errors: x 
```

You can view the waveform with:

```bash
gtkwave sim.vcd
```

> **Note:** VCD generation can significantly slow down the simulation. When extensive `printf` debugging is enabled, the waveform file may grow to several gigabytes, depending on the workload.

### LLM Benchmark Example

A simple LLM benchmark example is provided under
`target/snitch_cluster/sw/apps/snax-mx/accuracy_test/`.

## Publications

This work has been accepted for publication at the
[32nd Asia and South Pacific Design Automation Conference
(ASP-DAC 2027)](https://www.aspdac.com/aspdac2027/):

Z. Zhao, R. Ao, S. Cuyckens, M. Verhelst, C. Fang,
“MX-like Tensor Core for LLM Inference with NPU Integration via
Hardware-Software Co-Design Exploration,” *ASP-DAC*, 2027 (to appear).

If you use this repository in your research or build upon this work,
please cite the following paper:

```bibtex
@inproceedings{zhao2027mx,
  author    = {Zhao, Zijun and Ao, Ruichu and Cuyckens, Stef and
               Verhelst, Marian and Fang, Chao},
  title     = {{MX-like Tensor Core for LLM Inference with NPU Integration
               via Hardware-Software Co-Design Exploration}},
  booktitle = {Proceedings of the 32nd Asia and South Pacific Design
               Automation Conference (ASP-DAC)},
  year      = {2027},
  note      = {To appear}
}
```

<!-- TODO(RELEASE MAINTAINER):
Replace the preliminary citation above with the final publication record
after the ASP-DAC 2027 proceedings become available. Add the DOI, page
numbers, and proceedings URL, and remove the "To appear" note.
-->

## Acknowledgements

SNAX is an extension of the original [Snitch Cluster](https://github.com/pulp-platform/snitch_cluster) framework. SNAX got its name as "SNitch Acceleration eXtension".

<!-- TODO(RELEASE MAINTAINER):
Add project-specific acknowledgements and funding information before release.
-->


> **Release maintainer note:** Before publishing this project to the
> institutional repository, update all repository-specific URLs, including
> the clone URL, badges, documentation links, and publication metadata.
> Remove this note after completing the institutional release.
