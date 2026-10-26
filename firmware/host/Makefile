CC = clang
CFLAGS = -std=c11 -O2 -Wall -Wextra -Werror -pedantic
CPPFLAGS = -I../portable
LDLIBS = -lm

BUILD_DIR = build
TARGET = $(BUILD_DIR)/frontend_host
OBJECTS = $(BUILD_DIR)/frontend_host.o $(BUILD_DIR)/frontend.o

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)

$(BUILD_DIR)/frontend_host.o: frontend_host.c ../portable/frontend.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/frontend.o: ../portable/frontend.c ../portable/frontend.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean:
	rm -f build/frontend_host build/frontend_host.o build/frontend.o
