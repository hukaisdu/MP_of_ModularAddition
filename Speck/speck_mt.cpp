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

template<int B>
class mycallback: public GRBCallback
{
    public:
        vector<GRBVar> _X; 
        set<bitset<B>, cmp<B>> * _S;

        mycallback( const vector<GRBVar> & X, set<bitset<B>, cmp<B> > * S ) 
        {
            _X = X;
            _S = S;
        }
    protected:
        void callback () 
        {
          try 
          {
               if (where == GRB_CB_MIPSOL) 
               {
                  // MIP solution callback
                  bitset<B> s;

                  for ( int i = 0; i < B; i++ )
                  {
                      int x = abs ( int( round( getSolution( _X[i] ) ) ) );

                      if ( x == 0 )
                          s[i] = 0;
                      else
                          s[i] = 1;
                  }

                  _S -> insert( s );

                  GRBLinExpr l = 0;
                  for ( int i = 0; i < B; i++ )
                      if ( s[i] == 0 )
                          l += _X[i];
                      else
                          l += 1 - _X[i];
                  addLazy ( l >= 1 ); 
               } 
           }
          catch (GRBException e) 
           {
                cout << "Error number: " << e.getErrorCode() << endl;
                cout << e.getMessage() << endl;
            } 
            catch (...) 
            {
                cout << "Error during callback" << endl;
            }
          }
      
};

vector<string> split( const string & str, const string & delim )
{
	vector<string> res;
	boost::split( res, str, boost::is_any_of( delim ) );
	return res;
}

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

template<int N, int alpha, int beta>
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

    ModAdd<N> ( model, LeftRot<N>( XX, N - alpha ), S0, S2 );

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
        model.addConstr( S4[i] + S1[(i + beta) % N] == Y[ N + i ] );
}

template<int N, int alpha, int beta> 
void SolutionCounter( int rounds, 
        const bitset<N> & in, const bitset<N> & out, 
        int thread, int part, int cnt )
{
    GRBEnv env = GRBEnv( true );
    env.set(GRB_IntParam_LogToConsole, 0);

    env.set(GRB_IntParam_Threads, thread);

    //env.set(GRB_IntParam_Presolve, 0);
    env.set(GRB_IntParam_PoolSearchMode, 2);//focus on finding additional solutions 
    env.set(GRB_IntParam_MIPFocus, 3);
    env.set(GRB_IntParam_PoolSolutions, MAX); // try to find 2000000
    //env.set(GRB_DoubleParam_TimeLimit, 24 * 3600 );
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

    vector<vector<GRBVar>> K;
    for ( int r = 0; r < rounds; r++ )
    {
        vector<GRBVar> KK(N / 2);
        for (int i = 0; i < N / 2; i++)
           KK[i] = model.addVar(0, 1, 0, GRB_BINARY);
        K.push_back( KK );
    }

    for ( int r = 0; r < rounds; r ++ )
        speck_core<N/2, alpha, beta> (  model, X[r], X[r+1], K[r] );

    for ( int i = 0; i < N; i++ )
    {
        if ( in[i] == 1 )
            model.addConstr( X[0][i] == 1 );
        else
            model.addConstr( X[0][i] == 0 );

        if ( out[i] == 1 )
            model.addConstr( X[rounds][i] == 1 );
        else
            model.addConstr( X[rounds][i] == 0 );
    }

    GRBLinExpr l = 0;
    for ( int r = 0; r < rounds; r++ )
        for ( int i = 0; i < N/2; i++ )
            l += K[r][i];

    model.setObjective( l, GRB_MAXIMIZE );
    
    /*
    if ( keybit == -2 )
    {
        ;
    }
    else if ( keybit == -1 )
    {
        for ( int i = 0; i < N; i++ )
            model.addConstr( K[i/(N/2)][i%(N/2)] == 0 );
    }
    else
    {
        for ( int i = 0; i < N; i++ )
        {
            if ( i == keybit )
                model.addConstr( K[i/(N/2)][i%(N/2)] == 1 );
            else
                model.addConstr( K[i/(N/2)][i%(N/2)] == 0 );
        }
    }
    */
    
    model.optimize();

    lock_guard<mutex> guard( v_mutex );

    if ( model.get( GRB_IntAttr_Status ) == GRB_TIME_LIMIT )
    {
    	string filename = string( "Trails_TIME/vec_" ) + to_string( cnt ) + string( ".sol" ); 

    	ofstream os ( filename, ios::out );

	    os << in << " " << out << " " << rounds << endl; 

    	os.close();
    }
    else
    {
	    int solCount = model.get(GRB_IntAttr_SolCount);

	    map< bitset<N>, int, cmp<N>> counterMap;
	    for ( int e = 0; e < solCount; e++ )
	    {
		    bitset<N> v;
		    model.set(GRB_IntParam_SolutionNumber, e );
            for ( int r = 0; r < rounds; r++ )
            {
                for ( int j = 0; j < N/2; j++ )
                {
                if ( round( K[r][j].get( GRB_DoubleAttr_Xn ) ) == 1 )
                    v[N/2 * r + j] = 1;
                else if ( round( K[r][j].get( GRB_DoubleAttr_Xn ) ) == 0 )
                    v[N/2 * r + j] = 0;
                else
                {
                    cerr << "Error " << endl;
                    exit(-1);
                }
               }
            }
		    counterMap[v]++;
	    }

	    string filename;

	    if ( part == 0 )
		    filename = string( "Trails0/trail_" ) + to_string( cnt ) + string( ".sol" ); 
	    else
		    filename = string( "Trails1/trail_" ) + to_string( cnt ) + string( ".sol" ); 

	    ofstream os ( filename, ios::out );

	    for ( auto it : counterMap ) 
		    os << it.first << " " << it.second << endl;

	   // cout << part << " " << cnt << endl;

	    os.close();
	    //return solCount;
     }
}

template<int N, int alpha, int beta> 
set<bitset<N>, cmp<N> > extractMiddleNodes( int rounds, int mid,
        const bitset<N> & in, const bitset<N> & out )
{
    GRBEnv env = GRBEnv( true );

    //env.set(GRB_IntParam_OutputFlag, 0);
    env.set(GRB_IntParam_LogToConsole, 0);
    //env.set(GRB_IntParam_Threads, 16);

    //env.set(GRB_IntParam_Presolve, 0);
    env.set(GRB_IntParam_PoolSearchMode, 2);//focus on finding additional solutions 
    env.set(GRB_IntParam_MIPFocus, 3);
    env.set(GRB_IntParam_PoolSolutions, MAX); // try to find 2000000
    env.set(GRB_IntParam_LazyConstraints, 1 );

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

    vector<vector<GRBVar>> K;
    for ( int r = 0; r < rounds; r++ )
    {
        vector<GRBVar> KK(N / 2);
        for (int i = 0; i < N / 2; i++)
           KK[i] = model.addVar(0, 1, 0, GRB_BINARY);
        K.push_back( KK );
    }

    for ( int r = 0; r < rounds; r ++ )
        speck_core<N/2, alpha, beta> (  model, X[r], X[r+1], K[r] );

    for ( int i = 0; i < N; i++ )
    {
        if ( in[i] == 1 )
            model.addConstr( X[0][i] == 1 );
        else
            model.addConstr( X[0][i] == 0 );

        if ( out[i] == 1 )
            model.addConstr( X[rounds][i] == 1 );
        else
            model.addConstr( X[rounds][i] == 0 );
    }

    set< bitset<N>, cmp<N> > S;

    mycallback<N> cb( X[2], &S );

    model.setCallback(&cb);
    
    model.optimize();

    //int solCount = model.get(GRB_IntAttr_SolCount);

    //return solCount;
    return S;
}

template<int N, int ALPHA, int BETA>
map< bitset<2*N>,  int, cmp<2*N> > 
speck(  const int Round, const int Mid, const bitset<N> & IN, const bitset<N> & OUT, int thread )
{
    //int sol = SolutionCounter<N, ALPHA, BETA >( Round, IN, OUT );
    auto S = extractMiddleNodes<N, ALPHA, BETA >( Round, Mid, IN, OUT );

    map< bitset<2*N>, int, cmp<2*N> > MV;

    int sol = 0;

    ThreadPool thread_pool{};
    vector< future<void>> futures;

    int size = S.size();

    cout << size << endl;

    int count = 0;

    for ( auto & it : S )
    {
	    //cout << it << endl;

        futures.emplace_back( thread_pool.Submit( SolutionCounter<N, ALPHA, BETA>, 
                    Mid, IN, it, thread, 0, count  ) ); 

        futures.emplace_back( thread_pool.Submit( SolutionCounter<N, ALPHA, BETA>, 
                    Round - Mid, it, OUT, thread, 1, count  ) ); 
        count += 1;
    }
    
    for (auto & it : futures)
        it.get();

    map<bitset<N>, int, cmp<N>> M0[size];    
    map<bitset<N>, int, cmp<N>> M1[size];    

    for ( int c = 0; c < count; c++ )
    {
    	string filename = string( "Trails0/trail_" ) + to_string( c ) + string( ".sol" ); 
    	ifstream is ( filename, ios::in );

	string line;

	while( getline( is, line ) )
	{
        auto v = split( line, string(" ") );    
	    bitset<N> b ( v[0] );
	    int bv = stoi( v[1] );
	    M0[c][b] = bv;
	}
	is.close();

    string filename1 = string( "Trails1/trail_" ) + to_string( c ) + string( ".sol" ); 
    ifstream is1 ( filename1, ios::in );

	string line1;

	while( getline( is1, line1 ) )
	{
            auto v = split( line1, string(" ") );    
	    bitset<N> b ( v[0] );
	    int bv = stoi( v[1] );
	    M1[c][b]  = bv;
	}
	    is1.close();
    }    

    for ( int i = 0; i < size; i++ )
    {
        for ( auto it : M0[i] )
        {
           // cout << "M0 " << it.first << " " << it.second << endl;

            for ( auto jt : M1[i] )
            {
                bitset<2*N> v;
                for ( int i = 0; i < N; i++ )
                {
                    v[i] = it.first[i];
                    v[N + i] = jt.first[i];
                }

                MV[v] += it.second * jt.second;
            }
        }
    }

    //for ( auto it : MV )
    //    cout << it.first << " " << it.second << endl;

    //string filename = string( "Trail1/trail_" ) + to_string( count ) + string( ".sol" ); 

    return MV;
}

int main()
{
    const int N = 32;
    const int ALPHA = 7;
    const int BETA = 2;

    const int Mid = 2;
    const int Thread = 2;

    auto start = chrono::high_resolution_clock::now();

    bitset<N> IN;

    for ( int i = 0 ; i < N; i++ )
        IN[i] = 1;

    IN[25 ] = 0;
    IN[26] = 0;

    /* SPECK-32 */
    //IN[24] = 0;
    //IN[25] = 0;
    //IN[26] = 0;
    //cout << "25 " << "26 " << endl;

    /* SPECK-48 */
    //IN[39] = 0;
    //IN[40] = 0;
    //IN[41] = 0;
     
    /* SPECK- 64*/
    //IN[55] = 0;
    //IN[56] = 0;
    //IN[57] = 0;
    //IN[58] = 0;
     //IN[54] = 0;
    // IN[57] = 0;
    // IN[58] = 0;
//54 55 58 3
    //IN[58] = 0;


    /* SPECK-96 */
    //IN[86] = 0;
    //IN[87] = 0;
    //IN[88] = 0;


    // the most difficult one
    //IN[120] = 0;
    //IN[121] = 0;
    //IN[122] = 0;

    //cout << "119 " << "121 " << "122 " << endl;

    int Round;
    bitset<N> OUT;


    // Case 1
    Round = 5;
    for ( int i = 0 ; i < N; i++ )
    {
        OUT[i] = 0;
    }
    OUT[ N/2 - 1 - ALPHA ] = 1;

    auto M1 = speck<N, ALPHA, BETA> ( Round, Mid, IN, OUT, Thread );

    // Case 2
    Round = 4;
    for ( int i = 0 ; i < N; i++ )
    {
        OUT[i] = 0;
    }
    OUT[ N/2 - 1 - ALPHA ] = 1;
    auto M2 = speck<N, ALPHA, BETA> ( Round, Mid, IN, OUT, Thread );


    // Case 3
    Round = 4;
    for ( int i = 0 ; i < N; i++ )
    {
        OUT[i] = 0;
    }
    OUT[ N/2 + (N - 1 + BETA ) % N ] = 1;

    auto M3 = speck<N, ALPHA, BETA> ( Round, Mid, IN, OUT, Thread );

    // Case 4
    Round = 4;
    for ( int i = 0 ; i < N; i++ )
    {
        OUT[i] = 0;
    }
    OUT[ 2 * N -1 ] = 1;

    auto M4 = speck<N, ALPHA, BETA> ( Round, Mid, IN, OUT, Thread );


    auto end = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::seconds> ( end - start ).count();


    map< bitset<2*N>, int, cmp<2*N> > M;

    for ( auto it : M1 )
        M[ it.first ] += it.second;

    for ( auto it : M2 )
        M[ it.first ] += it.second;

    for ( auto it : M3 )
        M[ it.first ] += it.second;

    for ( auto it : M4 )
        M[ it.first ] += it.second;

    cout << "Monomial: " << endl;
    long long sol = 0;
    for ( auto & it : M )
    {
        //cout << it.first << " " << it.second << endl;
        sol += it.second;

        if ( it.second % 2 == 1 )
        {
            for ( int i = 0; i < 3 * N; i++ )
                if ( it.first[i] == true )
                    cout << "k^" << ( i / (N/2) ) << "_{" << ( i % (N/2) ) << "}";
            cout << endl;
        }
        //cout << it.second << endl;
        //
    }
    cout << "Time: " << duration << endl;
    cout << "Solution: " << sol << endl;
}



 

