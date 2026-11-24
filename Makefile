CMAKE ?= cmake
BUILD_DIR ?= build

.PHONY: all configure build test clean

all: build

configure:
	"$(CMAKE)" -S . -B "$(BUILD_DIR)"

build: configure
	"$(CMAKE)" --build "$(BUILD_DIR)"

test: build
	"$(CMAKE)" --build "$(BUILD_DIR)" --target test

clean:
	"$(CMAKE)" -E remove_directory "$(BUILD_DIR)"
