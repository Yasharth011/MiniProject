#include<stdlib.h>
#include<cmath>
#include<opencv4/opencv2/opencv.hpp>
#include<opencv4/opencv2/core/core.hpp>
#include<opencv4/opencv2/highgui/highgui.hpp>
#include<Eigen/Dense>
#include<librealsense2/h/rs_sensor.h>
#include<librealsense2/hpp/rs_pipeline.hpp>
#include<stack>
#include<tuple>

//namespace plt = matplotlibcpp;
using namespace std;
using namespace Eigen;
	
class EKF
{
	public:
	//Covariance Matrix
        Matrix<float, 4, 4> Q = {
         0.1, 0.0, 0.0, 0.0,
         0.0, 0.1, 0.0, 0.0,
         0.0, 0.0, 0.1, 0.0, 
	 0.0, 0.0, 0.0, 0.0}; 

        Matrix<float, 2, 2> R = {
     	1,0,
        0,1};

        //input noise 
        Matrix<float, 2, 2> ip_noise = {
        1.0, 0,
        0, (30*(3.14/180))};

        //measurement matrix
        Matrix<float, 2, 4> H = {
        1,0,0,0,
        0,1,0,0};

	float dt = 0.1;

	tuple<MatrixXf, MatrixXf> observation(MatrixXf xTrue, MatrixXf u)
	{	
		xTrue = state_model(xTrue, u);
		
		Matrix<float, 2, 1> ud;
		ud = u + ip_noise * MatrixXf::Random(2,1);

		return make_tuple(xTrue, ud);
	}
	
	MatrixXf state_model(MatrixXf x, MatrixXf u)
	{
		Matrix<float, 4, 4> A={
	            1,0,0,0,
		    0,1,0,0,
		    0,0,1,0,
		    0,0,0,0};

		Matrix<float, 4, 2> B={
		    (dt*cos(x.coeff(2,0))), 0,
		    (dt*sin(x.coeff(2,0))), 0,
		     0, dt,
		     1, 0};

		x = A * x + B * u;

		return x;
	}			

	MatrixXf jacob_f(MatrixXf x, MatrixXf u)
	{	
		float yaw = x.coeff(2,0);

		float v = u.coeff(0,0);

		Matrix<float, 4, 4> jF={
		     1.0, 0.0, (-dt*v*sin(yaw)), (dt*cos(yaw)),
		     0.0, 1.0, (dt*v*cos(yaw)), (dt*sin(yaw)),
		     0.0, 0.0, 1.0, 0.0,
		     0.0, 0.0, 0.0, 1.0};
		
		return jF;

	}

	MatrixXf observation_model(MatrixXf x)
	{
		Matrix<float, 2, 1> z;

		z = H * x;

		return z;
	}	

	tuple<MatrixXf, MatrixXf> ekf_estimation(MatrixXf xEst, MatrixXf PEst, MatrixXf z, MatrixXf u)
	{	
		//Predict 
		Matrix<float, 4, 1> xPred;
		xPred = state_model(xEst, u);

		//state vector
		Matrix<float, 4, 4> jF; 
		jF = jacob_f(xEst, u); 

		Matrix<float, 4, 4> PPred;
		PPred = jF*PEst*jF + Q;

		//Update
		Matrix<float, 2, 1> zPred;
		zPred = observation_model(xPred);

		Matrix<float, 2, 1> y;
		y = z - zPred; //measurement residual 
		
		Matrix<float, 2, 2> S;
		S = H*PPred*H.transpose() + R; //Innovation Covariance
		
		Matrix<float, 4, 1> K;
		K = PPred*H.transpose()*S.inverse(); //Kalman Gain
		
		xEst = xPred + K * y; //update step

		PEst = (MatrixXf::Identity(4,1) - (K*H)) * PPred;

		return make_tuple(xEst, PEst);
	}

};


int main()
{
	EKF obj;
  	
	Matrix<float, 1, 3> gyro;
	Matrix<float, 1, 3> accel; 
        //state vector
        Matrix<float, 4, 1> xEst = MatrixXf::Zero(4,1);
	Matrix<float, 4, 1> xTrue = MatrixXf::Zero(4,1);
	Matrix<float, 4, 1> PEst = MatrixXf::Identity(4,1);
	Matrix<float, 4, 1> ud = MatrixXf::Zero(2,1);
	Matrix<float, 2, 1> z = MatrixXf::Zero(2,1);
	//history 
	stack<MatrixXf> hxEst;
	stack<MatrixXf> hxTrue;

	hxEst.push(xEst);
	hxTrue.push(xTrue);

	while (true)
	{
		
		Matrix<float,1,3> accel = {};
		Matrix<float,1,3> gyro = {};
		
		//calculating net acceleration
		float accel_net = sqrt((pow(accel(0), 2) + pow(accel(1), 2)));
		
		accel_net = accel_net*obj.dt;

		//control input
		Matrix<float, 2, 1> u={accel_net, gyro(2)};
				
		float time = time + obj.dt;

		tie(xTrue, ud) = obj.observation(xTrue, u);
		
		z = obj.observation_model(xTrue);

		tie(xEst, PEst) = obj.ekf_estimation(xEst, PEst, z , ud);

		//store datat history
		hxEst.push(xEst);
		hxTrue.push(xTrue);
		
		if show_animation
		{
			plt.cla();

			//for stopping simulation with the esc key
			plt.gcf().canvas.mpl_connect("key release event", 
				[]{if(GetKeyState((int)"q"==1)) exit(0); 
				   else continue;});	

			//plotting actual state(represented by blue)
			plt.plot(hxTrue.coeff(0, seq(0, hxTrue.cols()), 
				 hxTrue.coeff(1, seq(1, hxTrue.cols()), "bo-");


			//plotting actual state(represented y red)
			plt.plot(hxEst.coeff(0, seq(0, hxEst.cols()), 
				 hxEst.coeff(1, seq(1, hxEst.cols()), "r-");

			plt.axis("equal");

			plt.grid(true);

			plt.pause(0.001);
		}
    		

	}
	return 0;
}




	     	









