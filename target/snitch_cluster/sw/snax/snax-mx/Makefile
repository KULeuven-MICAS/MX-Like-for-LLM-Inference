# Usage of absolute paths is required to externally include this Makefile
MK_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
include $(MK_DIR)/../../toolchain.mk

ifeq ($(SELECT_RUNTIME), rtl)
RUNTIME_DIR := rtl
else
RUNTIME_DIR := rtl-generic
endif

ROOT     = $(abspath $(MK_DIR)/../../../../..)
SNRT_DIR = $(ROOT)/sw/snRuntime
BUILDDIR = $(abspath build)
SRC_DIR  = $(abspath src)

####################
## Build variables #
####################

INCDIRS += $(abspath include)
INCDIRS += $(SNRT_DIR)/src
INCDIRS += $(SNRT_DIR)/api
INCDIRS += $(SNRT_DIR)/src/omp
INCDIRS += $(SNRT_DIR)/api/omp
INCDIRS += $(SNRT_DIR)/vendor/riscv-opcodes
INCDIRS += $(ROOT)/target/snitch_cluster/sw/runtime/common

# math.h needed by snRuntime

INCDIRS += $(SNRT_DIR)/../math/arch/riscv64/
INCDIRS += $(SNRT_DIR)/../math/arch/generic
INCDIRS += $(SNRT_DIR)/../math/src/include
INCDIRS += $(SNRT_DIR)/../math/src/internal
INCDIRS += $(SNRT_DIR)/../math/include/bits
INCDIRS += $(SNRT_DIR)/../math/include

INCDIRS += $(ROOT)/target/snitch_cluster/sw/runtime/$(RUNTIME_DIR)/src
INCDIRS += $(ROOT)/target/snitch_cluster/sw/runtime/$(RUNTIME_DIR)/include

# 输出目标改成了你的 mx 库！
OBJS        = $(BUILDDIR)/snax-mx-lib.o
ALL_OUTPUTS = $(OBJS)

.PHONY: all clean
all: $(ALL_OUTPUTS)

clean:
	rm -rf $(BUILDDIR)

$(BUILDDIR):
	mkdir -p $@

$(BUILDDIR)/%.o: $(SRC_DIR)/%.c | $(BUILDDIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -Wno-shift-op-parentheses -c $< -o $@