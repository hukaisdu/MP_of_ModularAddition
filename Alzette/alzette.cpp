#include<iostream>
#include<cstdio>
#include<bitset>
#include<vector>
#include<set>
#include<map>
#include<cmath>
#include<fstream>
#include<chrono>
#include<tuple>
#include<cassert>
#include <algorithm> 
#include"gurobi_c++.h" 
#include"alzette.h"

using namespace std;

const int ROUND = 6;

void SPLIT ( GRBModel & model, const vector<GRBVar> & X, const vector<GRBVar> & Y, const vector<GRBVar> & Z ) 
{
    int size = X.size();
    for ( int i = 0; i < size; i++ )
    {
        model.addConstr( X[i] >= Y[i] );
        model.addConstr( X[i] >= Z[i] );
        model.addConstr( Y[i] + Z[i] >= X[i] );
    }
}

void XOR ( GRBModel & model, const vector<GRBVar> & X, const vector<GRBVar> & Y, const vector<GRBVar> & Z ) 
{
    int N = X.size();
    for ( int i = 0; i < N; i++ )
    {
        model.addConstr( X[i] + Y[i] == Z[i] );
    }
}

// left rotation
template< int N >
vector<GRBVar> LeftRot( vector<GRBVar> & x, int n )
{
    vector<GRBVar> tmp ( N );
    for ( int i = n; i < N; i++ )
        tmp[ i - n ] = x[i];

    for ( int i = 0; i < n; i++ )
        tmp[N - n + i] = x[i]; 

    return tmp;
}

// modular addition
template<int N> 
void ModAdd( GRBModel & model, const vector<GRBVar> & U, 
                               const vector<GRBVar> & H, 
                               const vector<GRBVar> & W ) 
{
    vector<GRBVar> K(N);
    for ( int i = 0; i < N; i++ )
        K[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    model.addConstr( U[N-1] + H[N-1] - W[N-1] - 2 * K[N-2] == 0 );
    model.addConstr( K[N-1] == 0 );
    for ( int i = N-2; i > 0; i-- )
        model.addConstr( U[i] + H[i] + K[i] - W[i] - 2 * K[i-1] == 0 ); 
    model.addConstr( U[0] + H[0] + K[0] - W[0] == 0 );
}

// Alzette core
void Alzette_core( GRBModel & model, const vector<GRBVar> & X, const vector<GRBVar> & Y, const unsigned int c, int a, int b )
{
    const int NX = 32;
    // SPLIT 
    vector<GRBVar> S0(NX);
    for ( int i = 0; i < NX; i++ )
        S0[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    vector<GRBVar> S1( NX );
    for ( int i = 0; i < NX; i++ )
        S1[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    // modular addition
    vector<GRBVar> S2(NX);
    for ( int i = 0; i < NX; i++ )
        S2[i] = model.addVar( 0, 1, 0, GRB_BINARY );


    // SPLIT 
    vector<GRBVar> S3(NX);
    for ( int i = 0; i < NX; i++ )
        S3[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    vector<GRBVar> S4(NX);
    for ( int i = 0; i < NX; i++ )
        S4[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    for ( int i = 0; i < NX; i++ )
    {
        model.addConstr( S0[i] <= X[NX + i] );
        model.addConstr( S1[i] <= X[NX + i] );
        model.addConstr( S0[i] + S1[i] >= X[NX + i] );
    }


    vector<GRBVar> XX ( X.begin(), X.begin() + NX );
    ModAdd<NX> ( model, XX, LeftRot<NX> ( S0, NX - a ), S2 );

    
    for ( int i = 0; i < NX; i++ )
    {
        model.addConstr( S3[i] <= S2[i] );
        model.addConstr( S4[i] <= S2[i] );
        model.addConstr( S4[i] + S3[i] >= S2[i] );
    }


    // const xor
    for ( int i = 0; i < NX; i++ )
        if ( c >> ( NX - 1 - i ) & 0x1 )
            model.addConstr( S4[i] <= Y[i] );
        else
            model.addConstr( S4[i] == Y[i] );

    // XOR
    for ( int i = 0; i < NX; i++ )
    {
        //cout << i << " " << N - b << endl;
        model.addConstr( S3[ ( i + (NX - b) ) % NX ] + S1[i] == Y[ NX + i ] );
    }
}

void Alzette( GRBModel & model, const vector<GRBVar> & IN, const vector<GRBVar> & OUT, int c ) 
{
    //Alzette_core( GRBModel & model, const vector<GRBVar> & X, const vector<GRBVar> & Y, const unsigned int c, int a, int b )
    vector<GRBVar> T0(64);
    for ( int i = 0; i < 64; i++ )
        T0[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    vector<GRBVar> T1(64);
    for ( int i = 0; i < 64; i++ )
        T1[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    vector<GRBVar> T2(64);
    for ( int i = 0; i < 64; i++ )
        T2[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    Alzette_core (  model, IN, T0, constant[c], sv1[0], sv2[0] );
    Alzette_core (  model, T0, T1, constant[c], sv1[1], sv2[1] );
    Alzette_core (  model, T1, T2, constant[c], sv1[2], sv2[2] );
    Alzette_core (  model, T2, OUT, constant[c], sv1[3], sv2[3] );
   //
   // Alzette_core (  model, IN, T0, constant[c], sv1[2], sv2[2] );
   // Alzette_core (  model, T0, OUT, constant[c], sv1[3], sv2[3] );
}


// Alzette function
long AlzetteCount( const int rounds, const bitset<N> & in, const bitset<N> & out )
{

    try
    {
    GRBEnv env = GRBEnv( true );

    env.set(GRB_IntParam_LogToConsole, 0);

    env.set(GRB_IntParam_PoolSearchMode, 2);//focus on finding additional solutions 
    env.set(GRB_IntParam_MIPFocus, 3);
    env.set(GRB_IntParam_PoolSolutions, 2000000000); // try to find 2000000
                                              //
    env.start();
    
    // Create the model
    GRBModel model = GRBModel(env);

    // Create variables
    vector<vector<GRBVar>> X;
    for ( int r = 0; r < rounds + 1; r++ )
    {
        vector<GRBVar> XX(N);
        for (int i = 0; i < N; i++)
           XX[i] = model.addVar(0, 1, 0, GRB_BINARY);
        X.push_back( XX );
    }

    for ( int r = 0; r < rounds; r ++ )
        Alzette_core (  model, X[r], X[r+1], constant[0], sv1[r % 4], sv2[r % 4] );

    for ( int i = 0; i < N; i++ )
    {
        if ( in[i] == 0 )
            model.addConstr( X[0][i] == 0 );
        else
            model.addConstr( X[0][i] == 1 );

        if ( out[i] == 0 )
            model.addConstr( X[rounds][i] == 0 );
        else
            model.addConstr( X[rounds][i] == 1 );
    }

    GRBLinExpr l = 0;

    for ( int i = 0; i < N; i++ )
        l += X[rounds / 2][i];

    model.setObjective( l, GRB_MAXIMIZE );
    
    model.optimize();

    return model.get( GRB_IntAttr_SolCount );

    /*
    auto status = model.get( GRB_IntAttr_Status ); 

    if ( status == GRB_OPTIMAL )
    {
        cout << model.get( GRB_IntAttr_SolCount ) << endl;
        return false;
    }
    else if ( status == GRB_INFEASIBLE )
        return true;
    else
    {
        cerr << "Wrong " << endl;
        exit( EXIT_FAILURE );
    }
    */

    }
    catch ( GRBException e )
    {
        cout << "Error Code " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        exit( EXIT_FAILURE );
    }
    catch( ... )
    {
        cout << "Error During Callback" << endl;
    }
    return -1;
}

// Alzette function
int AlzetteMaxDegree( const int rounds, const bitset<N> & out, bitset<N> & realin )
{
    try
    {
        GRBEnv env = GRBEnv( true );

        env.set(GRB_IntParam_LogToConsole, 0);

        //env.set(GRB_IntParam_PoolSearchMode, 2);//focus on finding additional solutions 
        //env.set(GRB_IntParam_MIPFocus, 3);
        //env.set(GRB_IntParam_PoolSolutions, 2000000000); // try to find 2000000
                                                  //
        env.start();
        
        // Create the model
        GRBModel model = GRBModel(env);

        // Create variables
        vector<vector<GRBVar>> X;
        for ( int r = 0; r < rounds + 1; r++ )
        {
            vector<GRBVar> XX(N);
            for (int i = 0; i < N; i++)
               XX[i] = model.addVar(0, 1, 0, GRB_BINARY);
            X.push_back( XX );
        }

        if ( rounds == 4 )
        {
            for ( int i = 0; i < 20 ; i++ )
                model.addConstr( X[0][i] == 1 );
        }

        /*
        for ( int i = 50; i < 64; i++ )
        {
            model.addConstr( X[0][i] == 1 );
        }
        */

        for ( int r = 0; r < rounds; r ++ )
            Alzette_core (  model, X[r], X[r+1], constant[0], sv1[r % 4], sv2[ r  % 4] );

        for ( int i = 0; i < N; i++ )
        {
            if ( out[i] == 0 )
                model.addConstr( X[rounds][i] == 0 );
            else
                model.addConstr( X[rounds][i] == 1 );
        }

        GRBLinExpr l = 0;

        for ( int i = 0; i < N; i++ )
            l += X[0][i];

        model.setObjective( l, GRB_MAXIMIZE );
        
        while ( true )
        {   
            model.update();
            model.optimize();

            auto status = model.get( GRB_IntAttr_Status ); 

            if ( status == GRB_OPTIMAL )
            {
                auto res = model.getObjective().getValue();

                if ( rounds == 3 )
                {
                    if ( res >= 63  )
                    {
                        cout << res << " ";
                        return 64;
                    }
                }

                bitset<N> in;

                for ( int i = 0; i < N; i++ )
                {
                    if ( int ( round ( X[0][i].get( GRB_DoubleAttr_X ) ) ) == 0 ) 
                        in[i] = 0;
                    else
                        in[i] = 1;
                }

                cout << "Candidate Found " << in << " | " << res << endl;

                // count
                auto num = AlzetteCount( rounds, in, out );

                cout << "Number " << num << endl;
                if ( num % 2 == 1 )
                {
                    for ( int i = 0; i < N; i++ )
                        realin[i] = in[i];
                    return res;
                }
                else
                {
                    GRBLinExpr xl = 0;
                    for ( int i = 0; i < N; i++ )
                    {
                        if ( in[i] == 1 )
                            xl += ( 1 - X[0][i] );
                        else
                            xl += X[0][i];
                    }
                    model.addConstr( xl >= 1 );
                }
            }
            //cout << model.get( GRB_IntAttr_SolCount ) << endl;
            //return false;
            else if ( status == GRB_INFEASIBLE )
                return -1;
            else
            {
                cerr << "Wrong " << endl;
                exit( EXIT_FAILURE );
            }
      }

    }
    catch ( GRBException e )
    {
        cout << "Error Code " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        exit( EXIT_FAILURE );
    }
    catch( ... )
    {
        cout << "Error During Callback" << endl;
    }
    return -1;
}

void PassSmallLinear( GRBModel & model, const vector<GRBVar> & in, const vector<GRBVar> & out )
{
    int Ineq[][17] = {{0, -1, 0, 0, -1, -1, 0, 0, -1, 1, 0, 0, 1, 1, 0, 0, 1}, {0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1}, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0}, {0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, {2, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0}, {0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0}, {2, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1}, {0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0}, {2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1}, {0, 1, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0}, {2, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, -1, -1, 0}, {0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0}, {1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}};
    for ( auto const & it : Ineq )
    {
        GRBLinExpr ln = it[0];
        for ( int i = 0; i < 8; i++ )
            ln += it[i + 1] * in[i];
        for ( int i = 0; i < 8; i++ )
            ln += it[i + 9] * out[i];

        model.addConstr( ln >= 0 );
    }

}


void PassLinear( GRBModel & model, const vector<GRBVar> & in, const vector<GRBVar> & out )
{
    for ( int start = 0; start < 16; start++ )
    {
        vector<GRBVar> IN;
        vector<GRBVar> OUT;
        for ( int i = start; i < 128; i += 16 ) 
            IN.push_back( in[i] );
        for ( int i = start; i < 128; i += 16 ) 
            OUT.push_back( out[i] );
        PassSmallLinear( model, IN, OUT );
    }
}

// main function
/*
int checkbit( int Round, int bit )
{
    auto res = Alzette( Round, bit );
}
*/

set<int> Sparkle( const vector<int> & Active  )
{
    set<int> S;
    for ( int i = 128; i < 256; i++ )
        S.insert( i );

    try
    {
        GRBEnv env = GRBEnv( true );
        env.set(GRB_IntParam_LogToConsole, 0);
        env.start();
        GRBModel model = GRBModel(env);

        vector<vector<GRBVar>> X;

        for ( int r = 0; r < ROUND; r++ )
        {
            vector<GRBVar> XX(256);
            for ( int i = 0; i < 256; i++ ) 
            {
                XX[i] = model.addVar(0, 1, 0, GRB_BINARY);
            }
            X.push_back( XX );
        }

        vector<vector<GRBVar>> Y;
        for ( int r = 0; r < ROUND; r++ )
        {
            vector<GRBVar> YY(256);
            for ( int i = 0; i < 256; i++ ) 
            {
                YY[i] = model.addVar(0, 1, 0, GRB_BINARY);
            }
            Y.push_back( YY );
        }

        // Alzette 
        //
        for ( int r = 0; r < ROUND; r++ )
        {
            for ( int i = 0; i < 4; i++ )
            {
                vector<GRBVar> T0;
                for ( int k = 0; k < 64; k++ )
                    T0.push_back( X[r][64 * i + k ] );

                vector<GRBVar> T1;
                for ( int k = 0; k < 64; k++ )
                    T1.push_back(  Y[r][64 * i + k ] );

                Alzette( model, T0, T1, i ); 
            }

            if ( r < ROUND - 1 )
            {
                vector<GRBVar> T0 ( Y[r].begin() + 0,   Y[r].begin() + 64 );
                vector<GRBVar> T1 ( Y[r].begin() + 64,  Y[r].begin() + 128 );
                vector<GRBVar> T2 ( Y[r].begin() + 128, Y[r].begin() + 192 );
                vector<GRBVar> T3 ( Y[r].begin() + 192, Y[r].begin() + 256 );

                vector<GRBVar> TT0 ( X[r + 1].begin() + 0,   X[r + 1].begin() + 64 );
                vector<GRBVar> TT1 ( X[r + 1].begin() + 64,  X[r + 1].begin() + 128 );
                vector<GRBVar> TT2 ( X[r + 1].begin() + 128, X[r + 1].begin() + 192 );
                vector<GRBVar> TT3 ( X[r + 1].begin() + 192, X[r + 1].begin() + 256 );

                vector<GRBVar> C0 (64);
                for ( int i = 0; i < 64; i++ )
                    C0[i] = model.addVar(0, 1, 0, GRB_BINARY);

                vector<GRBVar> C1 (64);
                for ( int i = 0; i < 64; i++ )
                    C1[i] = model.addVar(0, 1, 0, GRB_BINARY);

                vector<GRBVar> C2 (64);
                for ( int i = 0; i < 64; i++ )
                    C2[i] = model.addVar(0, 1, 0, GRB_BINARY);

                vector<GRBVar> C3 (64);
                for ( int i = 0; i < 64; i++ )
                    C3[i] = model.addVar(0, 1, 0, GRB_BINARY);

                SPLIT ( model,  T0, C0, TT2 );
                SPLIT ( model,  T1, C1, TT3 );
                XOR( model, T2, C2, TT1 );
                XOR( model, T3, C3, TT0 );

                vector<GRBVar> M0;
                vector<GRBVar> M1;

                //merge( C0.begin(), C0.end(), C1.begin(), C1.end(),
                //  M0.begin());
                for ( auto it : C0 ) M0.push_back( it );
                for ( auto it : C1 ) M0.push_back( it );

                for ( auto it : C2 ) M1.push_back( it );
                for ( auto it : C3 ) M1.push_back( it );

                PassLinear(  model, M0, M1 );
            }
        }

        for ( int i = 0; i < 256; i++ )
            if ( find( Active.begin(), Active.end(), i ) != Active.end() )
                model.addConstr( X[0][i] == 1 );
            else
                model.addConstr( X[0][i] == 0 );

        GRBLinExpr ln = 0;
        for ( int i = 0; i < 256; i++ )
        {
            if ( i < 128 )
                model.addConstr( Y[ROUND-1][i] == 0 ); 
            else
                ln += Y[ROUND-1][i];
        }

        model.setObjective( ln, GRB_MINIMIZE );


        while ( 1 )
        {
            cout << "S size " << S.size() << endl;

            model.update();

            model.optimize();

            auto status = model.get( GRB_IntAttr_Status ); 

            if ( status == GRB_OPTIMAL )
            {
                if ( model.getObjective().getValue() >= 2 )
                {
                    break;
                }
                else // = 1
                {
                    vector<int> V(256);
                    for ( int i = 0; i < 256; i++ )
                    {
                        if(  abs( int( round( Y[ROUND-1][i].get( GRB_DoubleAttr_X ) ) ) ) == 1 )
                        {
                            model.addConstr( Y[ROUND-1][i] == 0 );

                            S.erase( i );
                        }
                    }
                }
            }
            else if ( status == GRB_INFEASIBLE )
            {
                break;
            }
        }
    }
    catch ( GRBException e )
    {
        cout << "Error Code " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        exit( EXIT_FAILURE );
    }
    catch( ... )
    {
        cout << "Error During Callback" << endl;
    }

    return S;
}

int main()
{
    for ( int i = 31; i >= 30; i-- )
    {
        //cout << i << endl;
        bitset<N> out ( 1UL << i );
        bitset<N> realin;
        //cout << out << endl;
        int maxd = AlzetteMaxDegree( 4, out, realin );

        cout << i << " "  << maxd << endl;
    }
}

