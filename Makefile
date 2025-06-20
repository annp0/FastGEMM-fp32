CMAKE := cmake
BUILD_DIR := build
EXECUTABLE := run

.PHONY: all build debug clean sync

all: build

build:
		@mkdir -p $(BUILD_DIR)
		@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
		@$(CMAKE) --build $(BUILD_DIR) --config Release

debug:
		@mkdir -p $(BUILD_DIR)
		@$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Debug
		@$(CMAKE) --build $(BUILD_DIR) --config Debug

clean:
		@rm -rf $(BUILD_DIR)

run: build
		@$(BUILD_DIR)/$(EXECUTABLE)

sync:
		@git add .
		@git commit -m "make automatic update"
		@git push --force