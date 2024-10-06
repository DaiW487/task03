#include "windmill.hpp"
#include<Eigen/Core>
#include<ceres/ceres.h>
#include<chrono>
#include<math.h>


using namespace std;
using namespace cv;


int N=3000;


bool compareContourAreas(const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2) {
    return cv::contourArea(contour1) < cv::contourArea(contour2);
}


std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
double t0=0;



//	自定义残差计算模型
struct MyCostFunction
{
    MyCostFunction(double truevalues,double dt):truevalues(truevalues),dt(dt){}

    //	模板函数
	template<typename T>
	bool operator()(const T* const a,T* residual) const
	{   
        T time = T(dt);
        T real = T(truevalues);
        T angle = cos(a[3] * time + (a[0] / a[1]) * (ceres::cos(a[1]*T(t0)+a[2]) - ceres::cos(a[1] * (time+T(t0) ) + a[2])) );

		residual[0] = angle-real;

		return true;
	}

     double truevalues;
     double dt;
};

int main(){
    double result[5] = {0};

    for(int p=0;p<10;p++){

    double a[4]={0.789+0.5,1.884+0.5,1.65+0.5,1.305+0.5};

    std::chrono::milliseconds tt= std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    double t1 = (double)tt.count()/1000;
    WINDMILL::WindMill wm(t1);
   
    cv::Mat src;
    double real[10000]; 
    double Dt[10000];
    int count_frames = 0; 

    while (count_frames<N)
    {
        std::chrono::milliseconds ttt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double t2 = (double)ttt.count()/1000;

        src = wm.getMat(t2);
        
    

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


    




          // 计算圆心到圆上点的向量
            cv::Point2i vec =  boxCenter - R_Center;
            double r = sqrt(vec.x*vec.x + vec.y*vec.y);
        
            Dt[count_frames]=(t2-t1)/1000;
            //cout << dt[count_frames%10] << endl;
            real[count_frames] = vec.x / r;
            count_frames++;// 记录经过的帧数       
            //cv::imshow("windmill", src);
            //cv::waitKey(1);  

    }
        // STEP1：构建优化问题
        ceres::Problem problem;
        for(size_t i = 0 ; i < N; i++){
            // 添加CostFunction到Problem中
            double newreal= real[i];
            double newdt = Dt[i];
            //ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunction, 1, 4>(new CostFunction(nowCos, nowDt));
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<MyCostFunction, 1, 4>(new MyCostFunction(newreal, newdt)), new ceres::CauchyLoss(1.0), a);
        }

        
        

    
        // STEP4：配置求解器
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        //options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        
         chrono::steady_clock::time_point T1 = chrono::steady_clock::now();

        // STEP5：运行求解器
        ceres::Solve(options, &problem, &summary);

         chrono::steady_clock::time_point T2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( T2-T1 );

        if(a[0] < 0) a[0] = -a[0];
        if(a[1] < 0) a[1] = -a[1];
        while(a[2] < 0) {a[2] += 6.2831;}
        while(a[2] > 6.2831) {a[2] -= 6.2831;}
        std::cout << endl <<"总用时: " << Dt[count_frames-1]+(double)time_used.count() << endl;        
        for(int k = 0; k < 4; k++){result[k] += a[k];}
        result[4] += Dt[count_frames-1]+(double)time_used.count();


       if (abs(0.785-a[0])<0.05*0.785 && abs(1.884-a[1])<0.05*1.884 && abs(1.81-a[2])<0.05*1.81 &&abs(1.305-a[3])<0.05*1.305) {
            std::cout << "A_get: " << a[0] << endl;
            std::cout << "w_get: " << a[1] << endl;
            std::cout << "fai_get: " << a[2] << endl;
            std::cout << "b_get: " << a[3] << endl;
            }   

        //namedWindow("windmill",WINDOW_KEEPRATIO);
        //imshow("windmill", src);
        //=======================================================//
        
    }
    for(int k = 0; k < 5; k++){
        std::cout << endl << result[k]/10 << endl;
    }
}
