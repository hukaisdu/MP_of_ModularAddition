#include<iostream>
#include<cstdio>
#include<string> 
#include<bitset>
#include<vector>
#include<set>
#include<map>
#include<cmath>
#include<fstream>
#include<chrono>
#include<future>
#include<mutex>
#include<boost/algorithm/string.hpp>

#include"gurobi_c++.h" 
#include"thread_pool.h"


using namespace std;
using namespace thread_pool;

const int MAX = 2000000000;

mutex v_mutex;

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

template<int N, int ALPHA, int BETA>
void speck_core( GRBModel & model, const vector<GRBVar> & X, const vector<GRBVar> & Y, const vector<GRBVar> & K )
{
    // SPLIT 
    vector<GRBVar> S0(N);
    for ( int i = 0; i < N; i++ )
        S0[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    vector<GRBVar> S1( N );
    for ( int i = 0; i < N; i++ )
        S1[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    for ( int i = 0; i < N; i++ )
    {
        model.addConstr( S0[i] <= X[N + i] );
        model.addConstr( S1[i] <= X[N + i] );
        model.addConstr( S0[i] + S1[i] >= X[N + i] );
    }

    // modular addition
    vector<GRBVar> S2(N);
    for ( int i = 0; i < N; i++ )
        S2[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    vector<GRBVar> XX ( X.begin(), X.begin() + N );

    ModAdd<N> ( model, LeftRot<N>( XX, N - ALPHA ), S0, S2 );

    // Key xor
    //
    vector<GRBVar> S3(N);
    for ( int i = 0; i < N; i++ )
        S3[i] = model.addVar( 0, 1, 0, GRB_BINARY );

    for ( int i = 0; i < N; i++ )
        model.addConstr( S2[i] + K[i] == S3[i] );

    // SPLIT 
    vector<GRBVar> S4(N);
    for ( int i = 0; i < N; i++ )
        S4[i] = model.addVar( 0, 1, 0, GRB_BINARY );
    
    for ( int i = 0; i < N; i++ )
    {
        model.addConstr( S4[i] <= S3[i] );
        model.addConstr( Y[i]  <= S3[i] );
        model.addConstr( S4[i] + Y[i] >= S3[i] );
    }

    // XOR
    for ( int i = 0; i < N; i++ )
        model.addConstr( S4[i] + S1[(i + BETA) % N] == Y[ N + i ] );
}

/* given a pattern, counter the number of solutions within a time limit, 
if the time is on, return -1, otherwise, return the number of solutions */
template<int N, int ALPHA, int BETA, int Round> 
int SolutionCounter( 
                      const int mid,
                      const bitset<N> & in, 
                      //const bitset<Round * N / 2> & key,
                      const vector<int> & key,
                      const bitset<N> & out, 

                      const vector<int> &  middle,

                      int thread, 

                      double timeLimit
                    )
{
    GRBEnv env = GRBEnv( true );

    env.set(GRB_IntParam_LogToConsole, 0);
    env.set(GRB_IntParam_Threads, thread);
    env.set(GRB_IntParam_PoolSearchMode, 2);//focus on finding additional solutions 
    env.set(GRB_IntParam_MIPFocus, 3);
    env.set(GRB_IntParam_PoolSolutions, MAX); // try to find 2000000
    env.set(GRB_DoubleParam_TimeLimit, timeLimit );

    env.start();
    
    // Create the model
    GRBModel model = GRBModel(env);

    // Create variables
    vector<vector<GRBVar>> X;
    for ( int r = 0; r < Round + 1; r++ )
    {
        vector<GRBVar> XX(N);
        for (int i = 0; i < N; i++)
           XX[i] = model.addVar(0, 1, 0, GRB_BINARY);
        X.push_back( XX );
    }

    vector<vector<GRBVar>> K;
    for ( int r = 0; r < Round; r++ )
    {
        vector<GRBVar> KK(N / 2);
        for (int i = 0; i < N / 2; i++)
           KK[i] = model.addVar(0, 1, 0, GRB_BINARY);
        K.push_back( KK );
    }

    for ( int r = 0; r < Round; r++ )
    {
        for (int i = 0; i < N / 2; i++)
            if ( key[ N/2 * r + i ] == 1 )
                model.addConstr( K[r][i] == 1 );
            else
                model.addConstr( K[r][i] == 0 );
    }

    for ( int r = 0; r < Round; r ++ )
        speck_core<N/2, ALPHA, BETA> (  model, X[r], X[r+1], K[r] );

    for ( int i = 0; i < N; i++ )
    {
        if ( in[i] == 1 )
            model.addConstr( X[0][i] == 1 );
        else
            model.addConstr( X[0][i] == 0 );

        if ( out[i] == 1 )
            model.addConstr( X[Round][i] == 1 );
        else
            model.addConstr( X[Round][i] == 0 );
    }

    for ( int i = 0; i < N; i++ )
    {
        if ( middle[i] == 0 )
            model.addConstr( X[mid][i] == 0 );
        else if ( middle[i] == 1 )
            model.addConstr( X[mid][i] == 1 );
    }

    GRBLinExpr l = 0;
    for ( int r = 0; r < Round; r++ )
        for ( int i = 0; i < N/2; i++ )
            l += K[r][i];

    model.setObjective( l, GRB_MAXIMIZE );
    
    model.optimize();

    auto status = model.get(GRB_IntAttr_Status );

    if ( status == GRB_TIME_LIMIT )
    {
        return -1;
    }
    else
    {
	    int solCount = model.get( GRB_IntAttr_SolCount );
        return solCount;
    }
}

/* given a pattern, counter the number of solutions within a time limit, 
if the time is on, return -1, otherwise, return the number of solutions */
template<int N, int ALPHA, int BETA> 
int SolutionCounter_plain(  const int Round,
                      const bitset<N> & in, 
                      const bitset<N> & out, 
                      const vector<int> & key
                    )
{
    GRBEnv env = GRBEnv( true );

    env.set(GRB_IntParam_LogToConsole, 0);
    env.set(GRB_IntParam_PoolSearchMode, 2);//focus on finding additional solutions 
    env.set(GRB_IntParam_MIPFocus, 3);
    env.set(GRB_IntParam_PoolSolutions, MAX); // try to find 2000000

    env.start();
    
    // Create the model
    GRBModel model = GRBModel(env);

    // Create variables
    vector<vector<GRBVar>> X;
    for ( int r = 0; r < Round + 1; r++ )
    {
        vector<GRBVar> XX(N);
        for (int i = 0; i < N; i++)
           XX[i] = model.addVar(0, 1, 0, GRB_BINARY);
        X.push_back( XX );
    }

    vector<vector<GRBVar>> K;
    for ( int r = 0; r < Round; r++ )
    {
        vector<GRBVar> KK(N / 2);
        for (int i = 0; i < N / 2; i++)
           KK[i] = model.addVar(0, 1, 0, GRB_BINARY);
        K.push_back( KK );
    }

    for ( int r = 0; r < Round; r++ )
    {
        for (int i = 0; i < N / 2; i++)
            if ( key[ N/2 * r + i ] == 1 )
                model.addConstr( K[r][i] == 1 );
            else
                model.addConstr( K[r][i] == 0 );
    }

    for ( int r = 0; r < Round; r ++ )
        speck_core<N/2, ALPHA, BETA> (  model, X[r], X[r+1], K[r] );

    for ( int i = 0; i < N; i++ )
    {
        if ( in[i] == 1 )
            model.addConstr( X[0][i] == 1 );
        else
            model.addConstr( X[0][i] == 0 );

        if ( out[i] == 1 )
            model.addConstr( X[Round][i] == 1 );
        else
            model.addConstr( X[Round][i] == 0 );
    }

    for ( int r = 0; r < Round; r++ )
      for ( int i = 0; i < N/2; i++ )
        model.addConstr( K[r][i] == key[ r * N/2 + i ] );

    GRBLinExpr l = 0;
    for ( int r = 0; r < Round; r++ )
        for ( int i = 0; i < N/2; i++ )
            l += K[r][i];

    model.setObjective( l, GRB_MAXIMIZE );
    
    model.optimize();

	int solCount = model.get( GRB_IntAttr_SolCount );

    return solCount;
}


/* a wrapper of the solution counter, for the multitreading process */
template<int N, int ALPHA, int BETA, int Round> 
void worker( 
             int mid,
             const bitset<N> & in, 
             //const bitset<Round * N / 2> & key,
             const vector<int> & key,
             const bitset<N> & out, 
             const vector<int> &  middle,
             int thread, 
             double timeLimit,

             long & solNumber,
             vector< vector<int> > & layer2
                    )
{
    auto res = SolutionCounter<N, ALPHA, BETA, Round>( mid, in, key, out, middle, thread, timeLimit );

    lock_guard<mutex> guard( v_mutex ); 

    if ( res == -1 ) // need to expand
    {
        layer2.push_back( middle );
    }
    else
    {
        solNumber += res;
        //cout << "Worker " << solNumber << endl;
    }
}

template<int N>
void expandVector( const vector<vector<int>> & layer2, vector< vector<int> > & layer1 )
{
    for ( auto it : layer2 )
    {
        int i;
        for ( i = 0; i < N; i++ )
            if ( it[i] < 2 )
                ;
            else
                break;

        it[i] = 0;
        layer1.push_back( it );
        it[i] = 1;
        layer1.push_back( it );

        
    }
}

template<int N, int ALPHA, int BETA, int Round> 
vector<int> 
    suggestKeyMonomial_Half(
            const bitset<N> & in, 
		    const bitset<N> & out,
            const set< vector<int> > & keyS,
            const vector<int> & kappa, 
            bool & isInfeasible
		    )
{
    //cout << "Suggest" << endl;

    GRBEnv env = GRBEnv( true );
    env.set(GRB_IntParam_LogToConsole, 0);
    env.start();
    
    // Create the model
    GRBModel model = GRBModel(env);

    // Create variables
    vector<vector<GRBVar>> X;
    for ( int r = 0; r < Round + 1; r++ )
    {
        vector<GRBVar> XX(N);
        for (int i = 0; i < N; i++)
           XX[i] = model.addVar(0, 1, 0, GRB_BINARY);
        X.push_back( XX );
    }

    vector<vector<GRBVar>> K;
    for ( int r = 0; r < Round; r++ )
    {
        vector<GRBVar> KK(N / 2);
        for (int i = 0; i < N / 2; i++)
           KK[i] = model.addVar(0, 1, 0, GRB_BINARY);
        K.push_back( KK );
    }

    for ( int r = 0; r < Round; r ++ )
        speck_core<N/2, ALPHA, BETA> (  model, X[r], X[r+1], K[r] );

    for ( int i = 0; i < N; i++ )
    {
        if ( in[i] == 1 )
            model.addConstr( X[0][i] == 1 );
        else
            model.addConstr( X[0][i] == 0 );

        if ( out[i] == 1 )
            model.addConstr( X[Round][i] == 1 );
        else
            model.addConstr( X[Round][i] == 0 );
    }

    for ( auto it : keyS )
    {
        GRBLinExpr ll = 0;

        for ( int r = 0; r < Round; r++ )
            for ( int i = 0; i < N/2; i++ )
            {
                if ( it[ r * N/2 + i ] == 0 )
                    ll += K[r][i];

                else if ( it[ r * N/2 + i] == 1 )
                    ll += 1 - K[r][i];
            }

        model.addConstr( ll >= 1 ); 
    }

  for ( int r = 0; r < Round; r++ )
    //int r = probeRound;
    for ( int i = 0; i < N/2; i++ )
    {
      if ( kappa[r * N/2 + i] == 0 )
        model.addConstr( K[r][i] == 0 );

      else if ( kappa[r * N/2 + i] == 1 )
        model.addConstr( K[r][i] == 1 );
    }

  GRBLinExpr l = 0;
  for ( int r = 0; r < Round; r++ )
    for ( int i = 0; i < N/2; i++ )
        l += K[r][i];

  //for ( int i = 0; i < N; i++ )
  //    l += X[probeRound][i];

  model.setObjective( l, GRB_MAXIMIZE );

  int count = 0;

  while( true )
  { 
      //cout << "Start to optimize " << count << endl;
      count += 1;

      model.optimize();

      if ( model.get( GRB_IntAttr_Status ) == GRB_OPTIMAL )
      {
          //bitset<N> xx;
          //bitset<N/2 * Round> kk;
          vector<int> kk ( N/2 * Round, 2 );

          for ( int r = 0; r < Round; r++ )
          for ( int j = 0; j < N/2; j++ )
          {
                if ( round( K[r][j].get( GRB_DoubleAttr_Xn ) ) == 1 )
                    kk[r * N/2 + j] = 1;
                else if( round( K[r][j].get( GRB_DoubleAttr_Xn ) ) == 0 )
                    kk[r * N/2 + j] = 0;
                else
                {
                    cerr << "Error " << endl;
                    exit(-1);
                }
          }

          //cout << "Found a solution, start to count " << endl;

          int sol = SolutionCounter_plain<N, ALPHA, BETA >  ( Round, in, out, kk );

          //cout << "Sol " << sol << endl;

          if ( sol == 0 )
              exit( EXIT_FAILURE );

          if ( sol % 2 == 1)
              return kk;
          else
          {
              GRBLinExpr ll = 0;

              for ( int r = 0; r < Round; r++ )
              for ( int i = 0; i < N/2; i++ )
              {
                     if ( kk[r * N/2 + i] == 0 )
                        ll += K[r][i];
                     else if ( kk[r * N/2 + i] == 1 )
                        ll += 1 - K[r][i];
              }

             model.addConstr( ll >= 1 ); 
          }

          for ( int r = 0; r < Round; r++ )
            //int r = probeRound;
            for ( int i = 0; i < N/2; i++ )
            {
              if ( kappa[r * N/2 + i] == 0 )
                model.addConstr( K[r][i] == 0 );

              else if ( kappa[r * N/2 + i] == 1 )
                model.addConstr( K[r][i] == 1 );
            }

          model.update();

          //cout << "Optimial " << endl;
	   }

      else
      {
          //cout << "Infeasible " << endl;
          isInfeasible = true;
          return vector<int> ( 0 );
      }

      /*
     if ( count > 400 )
     {
        isInfeasible = true;
        return vector<int> ( 0 );
     }
     */
  }
}

template<int N, int ALPHA, int BETA, int Round> 
bitset<N/2> 
    suggestKeyMonomial(
            const bitset<N> & in, 
		    const bitset<N> & out,
            const set< vector<int> > & keyS,
            int probeRound,
            const vector<int> & kappa, 
            bool & isInfeasible
		    )
{
    //cout << "Suggest" << endl;

    GRBEnv env = GRBEnv( true );
    env.set(GRB_IntParam_LogToConsole, 0);
    env.start();
    
    // Create the model
    GRBModel model = GRBModel(env);

    // Create variables
    vector<vector<GRBVar>> X;
    for ( int r = 0; r < Round + 1; r++ )
    {
        vector<GRBVar> XX(N);
        for (int i = 0; i < N; i++)
           XX[i] = model.addVar(0, 1, 0, GRB_BINARY);
        X.push_back( XX );
    }

    vector<vector<GRBVar>> K;
    for ( int r = 0; r < Round; r++ )
    {
        vector<GRBVar> KK(N / 2);
        for (int i = 0; i < N / 2; i++)
           KK[i] = model.addVar(0, 1, 0, GRB_BINARY);
        K.push_back( KK );
    }

    for ( int r = 0; r < Round; r ++ )
        speck_core<N/2, ALPHA, BETA> (  model, X[r], X[r+1], K[r] );

    for ( int i = 0; i < N; i++ )
    {
        if ( in[i] == 1 )
            model.addConstr( X[0][i] == 1 );
        else
            model.addConstr( X[0][i] == 0 );

        if ( out[i] == 1 )
            model.addConstr( X[Round][i] == 1 );
        else
            model.addConstr( X[Round][i] == 0 );
    }

    for ( auto it : keyS )
    {
        //cout << "exclude keyS " << endl;
        //for ( auto jt : it )
        //    cout << jt;
        //cout << endl;

        GRBLinExpr ll = 0;

        for ( int r = 0; r < Round; r++ )
            for ( int i = 0; i < N/2; i++ )
            {
                if ( it[ r * N/2 + i ] == 0 )
                    ll += K[r][i];

                else if ( it[ r * N/2 + i] == 1 )
                    ll += 1 - K[r][i];
            }

        model.addConstr( ll >= 1 ); 
      }

  for ( int r = 0; r < Round; r++ )
    //int r = probeRound;
    for ( int i = 0; i < N/2; i++ )
    {
      if ( kappa[r * N/2 + i] == 0 )
        model.addConstr( K[r][i] == 0 );

      else if ( kappa[r * N/2 + i] == 1 )
        model.addConstr( K[r][i] == 1 );
    }

  GRBLinExpr l = 0;
  for ( int r = 0; r < Round; r++ )
    for ( int i = 0; i < N/2; i++ )
        l += K[probeRound][i];

  //for ( int i = 0; i < N; i++ )
  //    l += X[probeRound][i];

  model.setObjective( l, GRB_MAXIMIZE );

  int count = 0;

  while( true )
  { 
      //cout << "Round " << probeRound << endl;
      //cout << "Start to optimize " << count << endl;
      count += 1;

      model.optimize();

      if ( model.get( GRB_IntAttr_Status ) == GRB_OPTIMAL )
      {
          bitset<N> xx;
          bitset<N/2> kk;

          for ( int j = 0; j < N/2; j++ )
          {
                if ( round( K[probeRound][j].get( GRB_DoubleAttr_Xn ) ) == 1 )
                    kk[j] = 1;
                else if( round( K[probeRound][j].get( GRB_DoubleAttr_Xn ) ) == 0 )
                    kk[j] = 0;
                else
                {
                    cerr << "Error " << endl;
                    exit(-1);
                }
          }

          for ( int j = 0; j < N; j++ )
          {
                if ( round( X[probeRound][j].get( GRB_DoubleAttr_Xn ) ) == 1 )
                    xx[j] = 1;
                else if ( round( X[probeRound][j].get( GRB_DoubleAttr_Xn ) ) == 0 )
                    xx[j] = 0;
                else
                {
                    cerr << "Error " << endl;
                    exit(-1);
                }
          }

          vector<int> kkk( N/2 * ( Round - probeRound ), 2 ); 

          //cout << "kk " << kk << endl;
          //cout << "xx " << xx << endl;

          for ( int i = 0; i < N/2; i++ )
            kkk[i] = kk[i];

          //cout << "kkk 1    ";

          //for ( auto it : kkk )
          //    cout << it;

          //cout << endl;

          

          //cout << "kappa " << endl;
          //for ( int i = 0; i < kappa.size(); i++ )
          //{
          //    if ( (i % ( N/2 ) == 0 ) && (i > 1) )
          //        cout << endl;
          //    cout << kappa[i];
          //}
          //cout << endl;
          
          for ( int r = 1; r < ( Round - probeRound ); r++ )
          {
            for ( int i = 0; i < N/2; i++ )
            {
                kkk[r * N/2 + i] = kappa[N/2 * ( probeRound + r) + i ];
            }
          }

          //cout << "kkk 2    ";

          //for ( auto it : kkk )
          //    cout << it;

          //cout << endl;

          //getchar();
         // cout << "Found a solution, start to count " << endl;

         // cout << probeRound << " " <<  "xx " << xx << " " << "out " << out << endl;

          int sol = SolutionCounter_plain<N, ALPHA, BETA >  ( Round - probeRound, xx, out, kkk );

          //cout << Round - probeRound << " " << xx << " " << out << " ";
          //cout << "probeRound " << probeRound << endl;
          //cout << "kkk ";

          //for ( auto it : kkk )
          //    cout << it;

          //cout << endl;


          //cout << "kappa " << endl;
          //for ( auto it : kappa )
          //    cout << it;
          //cout << endl;

          //cout << "Sol " << sol << endl;

          if ( sol == 0 )
              exit( EXIT_FAILURE );

          if ( sol % 2 == 1)
              return kk;
          else
          {
              GRBLinExpr ll = 0;

              for ( int i = 0; i < N/2; i++ )
              {
                     if ( kk[i] == 0 )
                        ll += K[probeRound][i];
                     else if ( kk[i] == 1 )
                        ll += 1 - K[probeRound][i];
              }


              for ( int i = 0; i < N; i++ )
              {
                     if ( xx[i] == 0 )
                        ll += X[probeRound][i];
                     else if ( xx[i] == 1 )
                        ll += 1 - X[probeRound][i];
              }

             model.addConstr( ll >= 1 ); 


          }

          model.update();

          //cout << "Optimial " << endl;
	   }

      else
      {
          //cout << "Infeasible " << endl;
          isInfeasible = true;
          return bitset<N/2> ( 0 );
      }

      /*
     if ( count > 400 )
     {
        isInfeasible = true;
        return bitset<N/2> ( 0 );
     }
     */
  }
}

template<int N, int ALPHA, int BETA, int Round>
long speck( const int Mid, 
            //const bitset<N/2 * Round> key, 
            const vector<int> & key, 
            const bitset<N> & IN, 
            const bitset<N> & OUT, 
            int thread )
{
    long solNumber = 0;

    vector< vector<int> > layer1; // before solver

    vector<int> sv;

    for ( int i = 0; i < N; i++ )
        sv.push_back( 2 );

    layer1.push_back( sv );

    vector< vector<int> > layer2; // before expanded

    int cl = 0;

    while ( layer1.size() > 0 )
    {
            double time = 0;

            if ( cl < 16  )
                time = 10;
            else if ( cl < 32 )
               time = 3600;
            else
               time = 24 * 3600 * 5;

            cl += 1;

            //cout << "layer " << cl << " " << " Time: " << time << endl;

            //cout << "layer1 size: " << layer1.size() << endl;

            layer2.clear();

            ThreadPool thread_pool{};
            vector< future<void>> futures;

            for ( auto & it : layer1 )
            {
                futures.emplace_back( thread_pool.Submit( 
                    worker<N, ALPHA, BETA, Round>,  Mid, IN, key, OUT, it, thread, time, ref( solNumber), ref(layer2) ) );
            }

            for (auto & it : futures)
                it.get();

            layer1.clear();
            expandVector<N>( layer2, layer1 );

    }

    return solNumber;
}

template<int N, int ALPHA, int BETA, int Round>
void getKeyMonomial( int in, int out, bitset<N/2*Round> & kmon )
{
    bitset<N> IN;

    for ( int i = 0 ; i < N; i++ )
        IN[i] = 1;

    IN[in] = 0;

    bitset<N> OUT;

    for ( int i = 0 ; i < N; i++ )
        OUT[i] = 0;

    OUT[ out ] = 1;

    //vector<int> fkappa ( Round * N/2, 2 );

    set< vector<int> > keyS;

    int count = 0;

    while ( true ) 
    {
        bool isInfeasible = false;

        vector<int> kappa ( Round * N/2, 2 );

        for ( int r = Round - 1 ; r > -1; r-- ) 
        {
            //cout << endl;

            //cout << "test r = " << r << endl;

            auto k = suggestKeyMonomial<N, ALPHA, BETA, Round> ( IN, OUT, keyS, r, kappa, isInfeasible );

            //cout << "k " << k << endl;

            if ( isInfeasible == false )
            {
                for ( int i = 0; i < N/2; i++ )
                    if ( k[i] == 0 )
                        kappa[ r * N/2 + i ] = 0;
                    else
                        kappa[ r * N/2 + i ] = 1;
                //keyS.insert( kappa );

                if ( r == 0 )
                {
                    for ( int i = 0; i < N/2 * Round; i++ )
                        kmon[i] = kappa[i];

                    cout << in << " " << out << " " << "finish" << endl;
                    return;
                }
            }
            else
            {
                if ( r == Round - 1 )
                {
                    cerr << " Balanced " << "In: " << in << " Out: " << out << endl;
                    exit( EXIT_FAILURE );
                }

                /*
                for ( auto it : keyS )
                {
                    for ( auto jt : it )
                        cout << jt;
                    cout << endl;
                }
                */

                keyS.insert( kappa );

                break;
                //continue;
                //cerr << " Error " << "In: " << in << " Out: " << out << endl;
                //exit( EXIT_FAILURE );
            }
        }

        /*
        cout << "test remaining r " << endl;

        auto kt = suggestKeyMonomial_Half <N, ALPHA, BETA, Round> ( IN, OUT, keyS, kappa, isInfeasible );
        */

        /*
        if ( isInfeasible == false )
        {
            for ( int i = 0; i < N/2 * Round; i++ )
                fkappa[i] = kt[i];
            break;
        }
        else
            keyS.insert( kappa );
        */
    }

}



int main()
{
    const int N = 96;

    const int ALPHA = 47;
    const int BETA = 46;

    const int Mid = 2;

    const int Thread = 4;

    const int Round = 6;

    auto start = chrono::high_resolution_clock::now();

    bitset<N/2* Round> keymonomial[N][N];

    ThreadPool thread_pool{ 16 };

    vector< future<void>> futures;

    for ( int in = 0; in < N; in++ )
    for ( int out = 0; out < N; out++ )
    //for ( int out = 0; out < N; out++ )
    {
        futures.emplace_back( thread_pool.Submit( 
                            getKeyMonomial<N, ALPHA, BETA, Round>, in, out, ref( keymonomial[in][out] ) ) );
    }

    for (auto & it : futures)
        it.get();

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds> ( end - start ).count();

    for ( int i = 0; i < N; i++ )
        for ( int j = 0; j < N; j++ )
            cout << i << " " << j << " " << keymonomial[i][j] << endl;

    cout << "Time: " << duration << endl;

}



 

