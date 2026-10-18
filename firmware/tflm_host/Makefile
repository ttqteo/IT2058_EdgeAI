# Small host-only build for the pinned, independently built TFLite Micro
# library.  The tensor arena is supplied by runner.cc at runtime.

ROOT := $(abspath ../..)
TFLM_ROOT := $(ROOT)/artifacts/vendor/tflite-micro
TFLM_GEN := $(TFLM_ROOT)/gen/osx_arm64_default_gcc
TFLM_LIB := $(TFLM_GEN)/lib/libtensorflow-microlite.a
FLATBUFFERS_INCLUDE := $(TFLM_ROOT)/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include
BUILD_DIR ?= $(ROOT)/artifacts/tflm-host
TARGET := $(BUILD_DIR)/edgeai_tflm_host
OBJECT := $(BUILD_DIR)/runner.o

# GNU Make has a built-in CXX=g++.  On this host that is also an Apple Clang
# wrapper, but select clang++ explicitly unless the caller supplied CXX.
ifeq ($(origin CXX),default)
CXX := clang++
endif
ifeq ($(origin CXX),undefined)
CXX := clang++
endif
CPPFLAGS += -I$(TFLM_ROOT) -I$(FLATBUFFERS_INCLUDE)
# TFLM's C-compatible headers intentionally use a C99 flexible array and
# several compatibility no-op parameters.  Keep strict diagnostics for this
# runner while silencing only those upstream-header extensions.
CXXFLAGS += -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
	-Wno-c99-extensions -Wno-unused-parameter
LDLIBS += -lm

.PHONY: all clean FORCE

all: $(TARGET)

# Always rebuild the host object and link.  This keeps upstream header and
# archive changes from being hidden by stale local artifacts.
FORCE:

$(BUILD_DIR):
	mkdir -p $@

$(OBJECT): runner.cc FORCE | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(TARGET): FORCE $(OBJECT) $(TFLM_LIB)
	$(CXX) $(CXXFLAGS) $(OBJECT) -L$(TFLM_GEN)/lib -ltensorflow-microlite $(LDLIBS) -o $@

clean:
	rm -f $(TARGET) $(OBJECT)
