#ifndef __ALZETTE_H__
#define __ALZETTE_H__

#include<bitset>
#include<vector>

using namespace std;


const int MAX = 2000000000;

const int N = 64;

template<int N>
struct cmp
{
    bool operator()( const bitset<N> & a, const bitset<N> & b ) const
    {
        for ( int i = 0; i < N; i++ )
            if ( a[i] < b[i] ) return true;
            else if ( a[i] > b[i] ) return false;
        return false; // equal
    }
};

void Alzette_core( GRBModel & model, const vector<GRBVar> & X, const vector<GRBVar> & Y, const unsigned int c, int a, int b );

int sv1[4] = { 31,17, 0, 24 };
int sv2[4] = { 24,17, 31,16 };
unsigned int constant[8] = { 0xb7e15162, 0xbf715880, 0x38b4da56, 0x324e7738, 0xbb1185eb, 0x4f7c7b57,0xcfbfa1c8, 0xc2b3293d };

#endif
