TARGET= alzette

CC=g++
OPT=-m64 -std=c++11 -O3 -lm
INC=-I/$$GUROBI_HOME/include/
LIB=-L/$$GUROBI_HOME/lib -lgurobi_c++ -lgurobi110 -lm

SRC=$(TARGET).cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(OPT) -o $(TARGET)  $(SRC)  $(INC) $(LIB) 

clean:
	rm -f $(TARGET) 
	rm -f *.o *.obj
	rm -f *~ *.~*

tar:
	tar cvzf $(TARGET).tar.gz $(SRC) $(HED) makefile

#--------------------------------------------------

