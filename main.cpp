#include "windmill.hpp"
#include<Eigen/Core>
#include<ceres/ceres.h>    
#include<chrono>
#include<math.h>

using namespace std;
using namespace cv;

int N=5225;

bool compareContourAreas(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
    return cv::contourArea(contour1) < cv::contourArea(contour2);
}

std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

struct MyCostFunction
{
    MyCostFunction(double truevalues,double dt):truevalues(truevalues),dt(dt){}

	template<typename T>
	bool operator()(const T* const a,T* residual) const
	{   

        T time = T(dt);
        T real = T(truevalues);
        T angle = cos(a[3] * time + (a[0] / a[1]) * (ceres::cos(a[2]) - ceres::cos(a[1] * time + a[2])) );

		residual[0] = real-angle;

		return true;
	}

     double truevalues;
     double dt;
};

int main(){
    double result[5] = {0};

    for(int p=0;p<10;p++){

    double a[4]={0.789+0.5,1.884+.5,1.65+0.5,1.305+0.5};

    std::chrono::milliseconds tt= std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    double t1 = (double)tt.count();
    WINDMILL::WindMill wm(t1/1000);
   
    cv::Mat src;
    double real[10000]; 
    double Dt[10000];
    int n= 0; 

    while (n<N)
    {
        std::chrono::milliseconds ttt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double t2 = (double)ttt.count();

        src = wm.getMat(t2/1000);
        
    

        //==========================代码区========================//
    //convert to Gray
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	//binarization 
	Mat binary;
	cv::threshold(gray,binary,10,255,THRESH_BINARY);

	//find the contours
    std::vector<std::vector<cv::Point>> contours;  
    std::vector<cv::Vec4i> hierarchy;  
    cv::findContours(binary, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  


    std::sort(contours.begin(), contours.end(), compareContourAreas);

     Point2i R_Center;    
     if (contours[0].size() > 0){
            Rect bounding_rect = boundingRect(contours[0]);
            rectangle(src,bounding_rect,Scalar(0,255,0),1);
            R_Center = cv::Point2i(bounding_rect.x + bounding_rect.width / 2, bounding_rect.y + bounding_rect.height / 2);
            cv::circle(src, R_Center, 2, Scalar(0, 255, 0), 1);
        }
     Point2i boxCenter;
     if (contours[1].size() > 0) {
            Rect bounding_rect = boundingRect(contours[1]);
            rectangle(src,bounding_rect,Scalar(0,255,0),1);
           cv::Mat tempFind = binary(bounding_rect);
            vector<vector<Point>> tempFindContours;
            vector<Vec4i> tempFindHierarchy;
            findContours(tempFind, tempFindContours, tempFindHierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
            for (int j = 0; j < tempFindContours.size(); j++) {
                if (tempFindHierarchy[j][3] != -1) { 
                    Rect bounding_findbox = boundingRect(tempFindContours[j]);
                    bounding_findbox.x += bounding_rect.x;
                    bounding_findbox.y += bounding_rect.y;
                    boxCenter = cv::Point2i(bounding_findbox.x + bounding_findbox.width / 2, bounding_findbox.y + bounding_findbox.height / 2);
                    cv::circle(src, boxCenter, 4, Scalar(0, 255, 0), 3);
                }
            }
        }

          
            cv::Point2i vec =  boxCenter - R_Center;
            double r = sqrt(vec.x*vec.x + vec.y*vec.y);
        
            Dt[n]=(t2-t1)/1000;
            //cout << dt[n%10] << endl;
            real[n] = vec.x / r;
            n++;       
        //namedWindow("windmill",WINDOW_KEEPRATIO);
        //imshow("windmill", src);
            //cv::waitKey(1);  

    }
        
        ceres::Problem problem;
        for(size_t i = 0 ; i < N; i++){
            double newreal= real[i];
            double newdt = Dt[i];
            //ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunction, 1, 4>(new CostFunction(nowCos, nowDt));
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<MyCostFunction, 1, 4>(new MyCostFunction(newreal, newdt)), new ceres::CauchyLoss(1.0), a);
        }

        
        ceres::Solver::Options options;
        options.linear_solver_type=ceres::DENSE_QR;
      //  options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        
         chrono::steady_clock::time_point T1 = chrono::steady_clock::now();

        
        ceres::Solve(options, &problem, &summary);

         chrono::steady_clock::time_point T2 = chrono::steady_clock::now();
        chrono::duration<double> timeusing = chrono::duration_cast<chrono::duration<double>>( T2-T1 );

        if(a[0] < 0) a[0] = -a[0];
        if(a[1] < 0) a[1] = -a[1];
        while(a[2] < 0) {a[2] += 6.2831;}
        while(a[2] > 6.2831) {a[2] -= 6.2831;}

        std::cout << endl <<"总用时: " << Dt[n-1]+(double)timeusing.count() << endl;        
        for(int k = 0; k < 4; k++){result[k] += a[k];}
        result[4] += Dt[n-1]+(double)timeusing.count();


       if (abs(0.785-a[0])<0.05*0.785 &&abs(1.884-a[1])<0.05*1.884&& (abs(0.24-a[2])<0.05*0.24)||(abs(1.81-a[2])<0.05*1.81)||(abs(3.38-a[2])<0.05*3.38)&&abs(1.305-a[3])<0.05*1.305) {
            cout << "Good Done !" << endl;
            }   

       if(abs(0.785-a[0])<0.05*0.785){ std::cout << "A: " << a[0] << endl;}
       if(abs(1.884-a[1])<0.05*1.884){ std::cout << "w: " << a[1] << endl;}
       if((abs(0.24-a[2])<0.05*0.24)||(abs(1.81-a[2])<0.05*1.81)||(abs(3.38-a[2])<0.05*3.38)){ std::cout << "fai: " << a[2] << endl;}
       if(abs(1.305-a[3])<0.05*1.305){ std::cout << "A0: " << a[3] << endl;}

        //=======================================================//
    }
    cout<<endl<<"Average Time : "<<result[4]/10<<endl;
    //for(int k = 0; k < 5; k++){
      //  std::cout << endl << result[k]/10 << endl;
    //}
}
