# Basic Makefile for Metal Projects

# Compiler and Flags
SWIFTC = xcrun -sdk macosx swiftc
METALC = xcrun -sdk macosx metal
FRAMEWORKS = -framework AppKit -framework Metal -framework MetalKit
TARGET = music_visualizer

# Source Files
SWIFT_SOURCES = main.swift Renderer.swift Camera.swift MathLibrary.swift inputController.swift
SHADER_SOURCES = Shaders.metal

DEFAULT_LIB = default.metallib

#Default Target
all: $(TARGET)

# Linking final executable
$(TARGET): $(OBJECTS) $(DEFAULT_LIB)
	$(SWIFTC) $(SWIFT_SOURCES) -o $(TARGET) $(FRAMEWORKS)

# Compiling Metal shader files
$(DEFAULT_LIB): $(SHADER_SOURCES)
	$(METALC) -o $@ $<

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS) $(DEFAULT_LIB) data/processed/*/* data/processed/*.npz
