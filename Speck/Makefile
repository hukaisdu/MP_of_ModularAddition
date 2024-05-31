CC=g++
OPT=-std=c++17 -O3
INC=-I/$$GUROBI_HOME/include/
LIB=-L/$$GUROBI_HOME/lib -lgurobi_c++ -lgurobi110 -lm -lpthread -L//usr/lib/x86_64-linux-gnu

all: speck_mt speck_bt

speck_mt: speck_mt.cpp 
	$(CC) $(OPT) -o speck_mt  speck_mt.cpp $(INC) $(LIB) 

speck_bt: speck_bt.cpp 
	$(CC) $(OPT) -o speck_bt  speck_bt.cpp $(INC) $(LIB) 

clean:
	rm -f speck_mt speck_bt
	rm -f *.o *.obj
	rm -f *~ *.~*

tar:
	tar cvzf $(TARGET).tar.gz $(SRC) $(HED) makefile

#--------------------------------------------------

