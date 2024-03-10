#include <math.h>
#include <Mathplot++>


//covariance matrix 
Q[3][3] = {{1,0,0}, //var(x)
	   {0,1,0}, //var(y)
	   {0,0,1}}; //var(yaw)

R[3][3] = {{1,0,0},
	   {0,1,0},
	   {0,0,1}};
//noise parameter
input_noise = {{1,0},{0,5}};

//measurement matrix
H =        


