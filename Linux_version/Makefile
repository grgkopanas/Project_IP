CPP_FILES := $(wildcard ./*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
LIB_FLAGS := -L$(HOME)/build/lib -L/opt/intel/composerxe/lib/intel64/
LD_FLAGS :=-lglut -lGL -lopencv_core -lopencv_highgui -lopencv_imgproc -lANN -liomp5
INC_FLAGS:= -I$(HOME)/build/include
CC_FLAGS := -O3 

main: $(OBJ_FILES)
	icc -g -openmp $(LIB_FLAGS)   $^ -o $@  $(LD_FLAGS)
obj/%.o: %.cpp
	icc -g -openmp $(INC_FLAGS) $(CC_FLAGS) -c -o $@ $<
clean:
	rm -f obj/* main 
