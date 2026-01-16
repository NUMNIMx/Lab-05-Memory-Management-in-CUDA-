NVCC = nvcc
TARGET = lab5_memory
SRC = lab5_memory.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
