import copy
import os.path
import time

import numpy as np
from gurobipy import *
from gurobipy import GRB
from functools import partial
import concurrent.futures

class MILP_SCHWAEMM():
    def __init__(self, word_size=64):
        self.word_size = word_size

    def LeftRot(self, X, n ):
        return X[n:] + X[:n]

    def ModAdd(self, m, U, H, W, K, size = None): #L and K are part of the carry bit
        if size == None:
            size = self.word_size
        #LSB have xor
        m.addConstr(U[size-1] + H[size-1] - W[size-1]- 2*K[size-2] == 0)
        m.addConstr(K[size - 1] == 0)

        for i in range(size- 2, 0, -1):
            m.addConstr(U[i]+H[i]+K[i]-W[i]-2*K[i-1]==0)
        #MSB
        m.addConstr( U[0] + H[0] + K[0] - W[0] == 0)

    def XOR(self, m, U,H,W, size = None):
        if size == None:
            size = self.word_size
        for i in range(size):
            m.addConstr( U[i] + H[i] - W[i] == 0)

    def XORConst(self, m, U, h, W , size = None):
        if size == None:
            size = self.word_size
        for i in range(size):
            #print( i, U, W )
            if h >> ( self.word_size - 1 - i ) & 0x1:
                m.addConstr( W[i] >= U[i] )
            else:
                m.addConstr( W[i] == U[i] )
    
    def SPLIT(self, m, X, Y, Z,size = None):
        if size == None:
            size = self.word_size

        for i in range(size):
            m.addConstr( X[i] >= Y[i] )
            m.addConstr( X[i] >= Z[i] )
            m.addConstr( Y[i] + Z[i] >= X[i] )

    def SPLIT_triple(self, m, X, S1, S2, S3, size = None):
        if size == None:
            size = self.word_size
        for i in range(size):
            m.addConstr(X[i] >= S1[i])
            m.addConstr(X[i] >= S2[i])
            m.addConstr(X[i] >= S3[i])
            m.addConstr(S1[i] + S2[i] + S3[i] >= X[i])
    def XOR_triple(self, m, U, H, L, W, size = None):
        if size == None:
            size = self.word_size
        for i in range(size):
            m.addConstr( U[i] + H[i] + L[i] - W[i] == 0)

    def AlzetteSbox(self, m, constant_value, input_X,output_X, R_az = 4):
        size = len(input_X)//2
        X = [[m.addVar(vtype=GRB.BINARY) for i in range(2 * size)] for r in range(R_az + 1)]
        SY1 = [[m.addVar(vtype=GRB.BINARY) for i in range(size)] for r in range(R_az)]
        SY2 = [[m.addVar(vtype=GRB.BINARY) for i in range(size)] for r in range(R_az)]
        SX1 = [[m.addVar(vtype=GRB.BINARY) for i in range(size)] for r in range(R_az)]
        SX2 = [[m.addVar(vtype=GRB.BINARY) for i in range(size)] for r in range(R_az)]
        O1 = [[m.addVar(vtype=GRB.BINARY) for i in range(size)] for r in range(R_az)]
        C1 = [[m.addVar(vtype=GRB.BINARY) for i in range(size)] for r in range(R_az)]

        shift_value_1 = [31, 17, 0, 24]
        shift_value_2 = [24, 17, 31, 16]

        # set input and output constraint
        for i in range(2 * size):
            m.addConstr(X[0][i] == input_X[i])
            m.addConstr(X[R_az][i] == output_X[i])

        for r in range(R_az):
            self.SPLIT(m, X[r][32:64], SY1[r], SY2[r], size = size)
            self.ModAdd(m, X[r][0:32], self.LeftRot(SY1[r], size - shift_value_1[r % 4]), O1[r], C1[r], size = size)  # L and K are part of the carry bit
            self.SPLIT(m, O1[r], SX1[r], SX2[r], size = size)
            self.XORConst(m, SX2[r], constant_value, X[r + 1][0:32],size = size)
            self.XOR(m, SY2[r], self.LeftRot(SX1[r], size - shift_value_2[r % 4]), X[r + 1][32:64],size = size)

    def ell(self,m, x,ox):
        y = x[0:16]
        z = x[16:32]
        z_copy = [m.addVar(vtype=GRB.BINARY) for i in range(16)]
        self.SPLIT(m, z, z_copy, ox[0:16], size = 16)
        self.XOR(m, y, z_copy, ox[16:32], size = 16)


    def M3(self,m, CO_0, CO_1,CO_2, MO_0,MO_1, MO_2):
        x0 = CO_0[0:32]
        y0 = CO_0[32:64]
        x1 = CO_1[0:32]
        y1 = CO_1[32:64]
        x2 = CO_2[0:32]
        y2 = CO_2[32:64]


        u0 = MO_0[0:32]
        v0 = MO_0[32:64]
        u1 = MO_1[0:32]
        v1 = MO_1[32:64]
        u2 = MO_2[0:32]
        v2 = MO_2[32:64]

        x0_copy1 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        x0_copy2 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        x1_copy1 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        x1_copy2 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        x2_copy1 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        x2_copy2 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]

        y0_copy1 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        y0_copy2 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        y1_copy1 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        y1_copy2 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        y2_copy1 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        y2_copy2 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]

        Ox = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Lx = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Lx_copy1 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Lx_copy2 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Lx_copy3 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]

        Oy = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Ly = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Ly_copy1 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Ly_copy2 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Ly_copy3 = [m.addVar(vtype=GRB.BINARY) for i in range(32)]

        self.SPLIT(m, x0, x0_copy1, x0_copy2, size=32)
        self.SPLIT(m, x1, x1_copy1, x1_copy2, size=32)
        self.SPLIT(m, x2, x2_copy1, x2_copy2, size=32)

        self.SPLIT(m, y0, y0_copy1, y0_copy2, size=32)
        self.SPLIT(m, y1, y1_copy1, y1_copy2, size=32)
        self.SPLIT(m, y2, y2_copy1, y2_copy2, size=32)

        self.XOR_triple(m, x0_copy1, x1_copy1, x2_copy1, Ox, size=32)
        self.XOR_triple(m, y0_copy1, y1_copy1, y2_copy1, Oy, size=32)
        # self.XOR(m,x0_copy1,x1_copy1, Ox ,size = 32)
        # self.XOR(m,y0_copy1,y1_copy1, Oy ,size = 32)

        self.ell(m, Ox, Lx)
        self.ell(m, Oy, Ly)

        self.SPLIT_triple(m, Lx, Lx_copy1, Lx_copy2, Lx_copy3, size=32)
        self.SPLIT_triple(m, Ly, Ly_copy1, Ly_copy2, Ly_copy3, size=32)
        # self.SPLIT(m,Lx, Lx_copy1, Lx_copy2, size = 32)
        # self.SPLIT(m,Ly, Ly_copy1, Ly_copy2, size = 32)


        self.XOR(m,Ly_copy1, x0_copy2,u0,size=32)
        self.XOR(m,Ly_copy2, x1_copy2,u1,size=32)
        self.XOR(m,Ly_copy3, x2_copy2,u2,size=32)

        self.XOR(m,Lx_copy1, y0_copy2,v0,size=32)
        self.XOR(m,Lx_copy2, y1_copy2,v1,size=32)
        self.XOR(m,Lx_copy3, y2_copy2,v2,size=32)



    def genModel384(self, R=4, input_X= None, output_X= None, count_num_solution = False, AEAD_mode = False ):
        m = Model()
        m.setParam('LogToConsole', 0)
        # m.setParam( 'Presolve', 0 )
        if count_num_solution == True:
            m.setParam('PoolSearchMode', 2)
            m.setParam('PoolSolutions', 2000000000)
        m.setParam('TimeLimit', 691200)
        m.setParam('Threads', 16)

        # generate variables
        constant_values = [0xb7e15162, 0xbf715880, 0x38b4da56, 0x324e7738, 0xbb1185eb, 0x4f7c7b57,0xcfbfa1c8, 0xc2b3293d]

        X = [[m.addVar(vtype=GRB.BINARY) for i in range(6 * self.word_size)] for r in range(R + 1)]
        Y0_consta = [[m.addVar(vtype=GRB.BINARY) for i in range(32)] for r in range(R)]
        Y1_consta = [[m.addVar(vtype=GRB.BINARY) for i in range(32)] for r in range(R)]
        ALZOUT = [[m.addVar(vtype=GRB.BINARY) for i in range(6 * self.word_size)] for r in range(R)]
        CO_0 = [[m.addVar(vtype=GRB.BINARY) for i in range(self.word_size)] for r in range(R)]
        CO_1 = [[m.addVar(vtype=GRB.BINARY) for i in range(self.word_size)] for r in range(R)]
        CO_2 = [[m.addVar(vtype=GRB.BINARY) for i in range(self.word_size)] for r in range(R)]
        MO_0 = [[m.addVar(vtype=GRB.BINARY) for i in range(self.word_size)] for r in range(R)]
        MO_1 = [[m.addVar(vtype=GRB.BINARY) for i in range(self.word_size)] for r in range(R)]
        MO_2 = [[m.addVar(vtype=GRB.BINARY) for i in range(self.word_size)] for r in range(R)]
        Y0_consta_LAST = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        Y1_consta_LAST = [m.addVar(vtype=GRB.BINARY) for i in range(32)]
        ALZOUT_LAST = [m.addVar(vtype=GRB.BINARY) for i in range(6 * self.word_size)]
        for i in range(6*self.word_size):
            m.addConstr(X[0][i] == input_X[i])
            m.addConstr(ALZOUT_LAST[i] == output_X[i])


        for r in range(R):
            # self.XORConst(m, U, h, W, size=16)
            self.XORConst(m, X[r][32:64], constant_values[r % 8], Y0_consta[r], size=32)
            self.XORConst(m, X[r][96:128], r % (2 ** 32), Y1_consta[r], size=32)

            X0Y0 = X[r][0:32] + Y0_consta[r]
            X1Y1 = X[r][64:96] + Y1_consta[r]
            self.AlzetteSbox(m, constant_values[0], X0Y0,ALZOUT[r][0:64], R_az=4)
            self.AlzetteSbox(m, constant_values[1], X1Y1,ALZOUT[r][64:128], R_az=4)
            self.AlzetteSbox(m, constant_values[2], X[r][128:192],ALZOUT[r][128:192], R_az=4)
            self.AlzetteSbox(m, constant_values[3], X[r][192:256],ALZOUT[r][192:256], R_az=4)
            self.AlzetteSbox(m, constant_values[4], X[r][256:320],ALZOUT[r][256:320], R_az=4)
            self.AlzetteSbox(m, constant_values[5], X[r][320:384],ALZOUT[r][320:384], R_az=4)


            self.SPLIT(m, ALZOUT[r][0:64], CO_0[r], X[r+1][192:256], size=64) #(x_0,y_0)
            self.SPLIT(m, ALZOUT[r][64:128], CO_1[r], X[r+1][256:320], size=64)  # (x_1,y_1)
            self.SPLIT(m, ALZOUT[r][128:192], CO_2[r], X[r+1][320:384], size=64)  # (x_2,y_2)

            self.M3(m,CO_0[r],CO_1[r], CO_2[r],MO_0[r],MO_1[r], MO_2[r])
            self.XOR(m,MO_0[r],ALZOUT[r][192:256], X[r+1][128:192], size = 64)
            self.XOR(m,MO_1[r],ALZOUT[r][256:320], X[r+1][0:64], size = 64)
            self.XOR(m,MO_2[r],ALZOUT[r][320:384], X[r+1][64:128], size = 64)
        self.XORConst(m, X[R][32:64], constant_values[R % 8], Y0_consta_LAST, size=32)
        self.XORConst(m, X[R][96:128], R, Y1_consta_LAST, size=32)
        X0Y0 = X[R][0:32] + Y0_consta_LAST
        X1Y1 = X[R][64:96] + Y1_consta_LAST

        self.AlzetteSbox(m, constant_values[0], X0Y0, ALZOUT_LAST[0:64], R_az=4)
        self.AlzetteSbox(m, constant_values[1], X1Y1, ALZOUT_LAST[64:128], R_az=4)
        self.AlzetteSbox(m, constant_values[2], X[R][128:192], ALZOUT_LAST[128:192], R_az=4)
        self.AlzetteSbox(m, constant_values[3], X[R][192:256], ALZOUT_LAST[192:256], R_az=4)
        self.AlzetteSbox(m, constant_values[4], X[R][256:320], ALZOUT_LAST[256:320], R_az=4)
        self.AlzetteSbox(m, constant_values[5], X[R][320:384], ALZOUT_LAST[320:384], R_az=4)




        m.optimize()
        if count_num_solution == True:
            return m.getAttr( 'SolCount' )
        if m.status == GRB.INFEASIBLE:  # soluable
            # print("This is Infeasible")
            return -1
        elif m.status == GRB.OPTIMAL:
            # print("This is optimal")
            counter = m.getAttr('SolCount')
            # print("{} solutions...".format(counter))
            return int(counter)
        elif m.status == GRB.TIME_LIMIT:
            print("exceed time limit")







if __name__ == "__main__":
    word_size = 64
    milp = MILP_SCHWAEMM(word_size=word_size)
    Rd = 4
    sparkle384 = True

    # start_time = time.time()
    # 64 * 6 = 384

    if sparkle384 == True:
        print("------Sparkle384------")
        start_time = time.time()
        for index_output_bit in range(0, 384): #127 190
            input_X = ( [0] * 64 ) + ( [0] * 63 )+[0] + ( [1] * 64 ) + [ 1 for index_j_1 in range(192, 384) ] #128
            output_X = [0 for i in range(384)]
            output_X[index_output_bit] = 1

            res = milp.genModel384(R=Rd, input_X= input_X, output_X= output_X, count_num_solution = False)
            print( index_output_bit , res )
        end_time = time.time() - start_time
        print("elapse_time:", end_time)
        print("---------------" * 5)

