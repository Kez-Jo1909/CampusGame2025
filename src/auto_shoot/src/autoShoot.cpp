#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tdt_interface/msg/receive_data.hpp>
#include <tdt_interface/msg/send_data.hpp>
#include"./PreProcess/PreProcess.h"
#include"./Detect/Detect.h"
#include"./Armor/Armor.h"
#include"./Kalman/Kalman.h"
#include <cmath>

#include <opencv2/opencv.hpp>
#include <iostream>

class VideoSaver {
private:
    cv::VideoWriter videoWriter;
    bool isInitialized = false;
    int frameWidth;
    int frameHeight;
    int fps;

public:
    // 构造函数，设置帧宽、帧高和帧率
    VideoSaver()
        : frameWidth(1280), frameHeight(720), fps(30) {}

    // 保存帧到视频文件的函数
    void saveFrame(const cv::Mat& frame) {
        // 如果 VideoWriter 还没有初始化，进行初始化
        if (!isInitialized) {
            videoWriter.open("/home/kezjo/output_video.avi", 
                             cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                             fps, 
                             cv::Size(frameWidth, frameHeight));
            if (!videoWriter.isOpened()) {
                std::cerr << "Error: Could not open video file for writing." << std::endl;
                return;
            }
            isInitialized = true;
        }

        // 检查输入帧的大小是否与设置的大小一致
        if (frame.cols != frameWidth || frame.rows != frameHeight) {
            std::cout<<frame.cols<<" "<<frame.rows<<std::endl;
            std::cerr << "Error: Frame size does not match the initialized video size." << std::endl;
            return;
        }

        // 将帧写入视频文件
        videoWriter.write(frame);
    }

    // 释放资源
    void release() {
        if (isInitialized) {
            videoWriter.release();
            isInitialized = false;
        }
    }
};


/*截取数据集用
int n=1;
int MatSave(cv::Mat img) {
    std::string ImgName = "/home/kezjo/Matsave2/" + std::to_string(n) + ".jpg";

    if (cv::imwrite(ImgName, img)) {
        std::cout << "Image saved successfully!" << std::endl;
        n++;
    } else {
        std::cerr << "Failed to save image." << std::endl;
        return -1;
    }

    return 0;
}
*/
std::vector<cv::Point> findCorners(const std::vector<cv::Point>& path) {
    std::vector<cv::Point> corners;
    // 如果路径少于两个点，不可能有拐点
    if (path.size() < 2) return corners;

    // 遍历路径，查找拐点
    for (size_t i = 1; i < path.size() - 1; ++i) {
        int dx1 = path[i].x - path[i - 1].x;
        int dy1 = path[i].y - path[i - 1].y;
        int dx2 = path[i + 1].x - path[i].x;
        int dy2 = path[i + 1].y - path[i].y;

        // 检查方向是否发生变化
        if ((dx1 != 0 && dy1 == 0 && dy2 != 0) || 
            (dy1 != 0 && dx1 == 0 && dx2 != 0)) {
            corners.push_back(path[i]);  // 当前点是拐点
        }
    }

    return corners;
}

struct Node{
    cv::Point point;
    float g_cost;//起点到当前节点的代价
    float h_cost;//当前节点到目标的估计代价,使用曼哈顿距离
    Node* parent;
    float f_cost() const { return g_cost + h_cost;}
};

struct CompareNode {
    bool operator()(Node* a, Node* b) {
        return a->f_cost() > b->f_cost();
    }
};

class Navigation{
private:
    std::vector<cv::Point> NaviTarget;//[0,50],保存用，不实际使用
    std::vector<cv::Point> NaviTargetUse;
    cv::Mat map;//传入后保存，不使用
    cv::Mat map_show;
    cv::Point iniStart=cv::Point(1,1);//path传出记得除以10
    cv::Mat NaviMap;
    cv::Mat NaviMapUsed=cv::Mat::zeros(50, 50, CV_8UC1);
    int ini=0;
public:
    int whetherIni(){
        return ini;
    }
    float manhattanDistance(const cv::Point& a,const cv::Point& b){
        return (std::abs(a.x-b.x)+std::abs(a.y-b.y));
    }
    cv::Mat getMap(){
        return NaviMap;
    }
    std::vector<cv::Point> AStar(const cv::Point& start,const cv::Point& goal,cv::Mat map){//需要先对地图进行kernel=15的erode操作
        /*TODO：navi在这里*/
        /*A*具体实现*/
        //定义上下左右四个方向
        const std::vector<cv::Point> directions={
            cv::Point(0,1),cv::Point(1,0),cv::Point(0,-1),cv::Point(-1,0)
        };

        std::priority_queue<Node*,std::vector<Node*>,CompareNode>open_list;
        std::vector<cv::Point> closed_set;

        //创建起点节点
        Node* start_node=new Node{start,0,manhattanDistance(start,goal),nullptr};
        open_list.push(start_node);

        while(!open_list.empty()){
            Node* current_node=open_list.top();
            open_list.pop();

            if(current_node->point==goal){
                std::vector<cv::Point> path;
                while(current_node){
                    path.push_back(current_node->point);
                    current_node=current_node->parent;
                }
                std::reverse(path.begin(),path.end());
                return path;
            }

            closed_set.push_back(current_node->point);//将当前点添加到 closed_set

            for(const auto& direction:directions){
                cv::Point neighbour_point=current_node->point+direction;
                
                if(neighbour_point.x<0 || neighbour_point.x>=map.cols || neighbour_point.y<0 || neighbour_point.y>=map.rows || map.at<uchar>(cv::Point(neighbour_point)) == 0){
                    continue;
                }

                // 如果邻居已经在 closed_set 中，跳过
                if (std::find(closed_set.begin(), closed_set.end(), neighbour_point) != closed_set.end()) {
                    continue;
                }

                float new_g_cost=current_node->g_cost+1.0;
                float new_h_cost=manhattanDistance(neighbour_point,goal);

                //创建新节点
                Node* neighbour_node=new Node{neighbour_point,new_g_cost,new_h_cost,current_node};
                open_list.push(neighbour_node);
            }
        }
        return {};
    }
    void showPosition(float x,float y){
        /*实时显示当前位置*/
        cv::Mat frame;
        if(map.empty()){
            std::cerr << "showPosition Error: map is empty!" << std::endl;
            return;
        }
        cv::cvtColor(map,frame,cv::COLOR_GRAY2BGR);//转成3通道
        cv::circle(frame,cv::Point(x*10,y*10),3,cv::Scalar(0,255,0),-1);
        for(int i=0;i<NaviTarget.size();i++){
            cv::putText(frame,std::to_string(i+1),NaviTarget[i]*10,cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
            //cv::circle(frame,cv::Point(NaviTarget[i].x,NaviTarget[i].y)*10,3,cv::Scalar(0,255,0),-1);
        }
        cv::Mat flipped_frame;
        cv::flip(frame, flipped_frame, 0);
        cv::imshow("Real_Time_Map",flipped_frame);
    }
    cv::Mat erodeMap(int kernel_size)
    {
        /* 创建一个 kernel_size*kernel_size 的矩形结构元素作为卷积核 */
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
        cv::Mat eroded_map;
        // 进行腐蚀操作
        cv::erode(map, eroded_map, kernel);
        return eroded_map;
    }
    void SetNaviTarget(cv::Point2f setTarget){
        /*将手动传出的目标点坐标设置为优先目标点*/
        NaviTarget.insert(NaviTarget.begin(),setTarget*10);
    }
    void initialNavigation(cv::Mat map_img){
        ini=1;
        map=map_img;//map:1250*1250
        cv::resize(map,map,cv::Size(500,500));
        if(map.empty()){
            std::cerr << "initialNavi Error: map_image is empty!" << std::endl;
            return;
        }
        NaviMap=erodeMap(15);
        for(int i=0;i<500;i+=10){
            for(int j=0;j<500;j+=10){
                NaviMapUsed.at<uchar>(i/10, j/10)=NaviMap.at<uchar>(i,j);
            }
        }
        cv::cvtColor(map,map_show,cv::COLOR_GRAY2BGR);
        std::cout<<"Initial finished"<<std::endl;
    }
    void FindRectangles(){
        //先进行腐蚀，只留下目标区域,kernel_size=20,返回值为cvMat，不专门声明eroded_map,直接在底下调用
        //二值化图像（假设占用值为255的区域为前景，其他为背景）
        cv::Mat binary_map;
        cv::threshold(this->erodeMap(20), binary_map, 254, 255, cv::THRESH_BINARY);
    
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(binary_map, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Rect> rectangles;

        for (const auto& contour : contours)
        {
            cv::Rect rect = cv::boundingRect(contour);
            rectangles.push_back(rect);
        }

        // 按矩形面积从大到小排序
        std::sort(rectangles.begin(), rectangles.end(), [](const cv::Rect& r1, const cv::Rect& r2) {
            return r1.area() > r2.area();
        });

        for (int i = 0; i < rectangles.size();i++)
        {
            //下面这个/10是整个点坐标除以10
            NaviTarget.push_back(cv::Point((rectangles[i].x+rectangles[i].width/2),(rectangles[i].y+rectangles[i].height/2))/10);
            const cv::Rect& rect = rectangles[i];
            //std::cout<<NaviTarget[i]<<std::endl;
            cv::rectangle(map_show, rect, cv::Scalar(0, 255, 0), -1);  // 红色框，线宽2
        }
        cv::Mat map_show_flipped;
        cv::flip(map_show, map_show_flipped, 0);  // 参数0表示垂直翻转

        // 显示带有矩形框的地图
        cv::imshow("Top 5 Largest Rectangles", map_show_flipped);
        cv::waitKey(1);
    }
    std::vector<cv::Point> useNavi(cv::Point presentPoint){
        //std::cout<<"present Point:"<<presentPoint<<std::endl;
        std::vector<std::vector<cv::Point>> Path;
        if(NaviTargetUse.size()<=0){
            NaviTargetUse=NaviTarget;
        }
        for(int i=0;i<NaviTargetUse.size();i++){
            //std::cout<<"Navi State:"<<i<<std::endl;
            Path.push_back(AStar(presentPoint, NaviTargetUse[i], NaviMapUsed));
        }
        cv::Mat map_to_show;
        cv::cvtColor(map,map_to_show,cv::COLOR_GRAY2BGR);
        int minId=0;
        for(int i=1;i<Path.size();i++){
            //std::cout<<"Judge state:"<<i<<" size:"<<Path[i].size()<<std::endl;
            if(Path[i].size()<Path[minId].size()){
                minId=i;
            }
        }
        // std::cout<<"minId:"<<minId<<std::endl;
        for(int i=0;i<Path[minId].size();i++){
            cv::circle(map_to_show,Path[minId][i]*10,1,cv::Scalar(0,0,255),-1);
        }
        cv::flip(map_to_show, map_to_show, 0);
        cv::imshow("NaviMap",map_to_show);
        NaviTargetUse.erase(NaviTargetUse.begin()+minId);
        return findCorners(Path[minId]);
    }
};

class AutoShoot : public rclcpp::Node
{
public:
    AutoShoot()
        : Node("auto_shoot_node")
    {
    	//初始化一个卡尔曼
    	Kalman iniKalman;
    	kalmanList.push_back(iniKalman);
    	
        // 订阅相机图像
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera_image", 10,
            std::bind(&AutoShoot::imageCallback, this, std::placeholders::_1));

        // 订阅云台角度
        receive_data_sub_ = this->create_subscription<tdt_interface::msg::ReceiveData>(
            "/real_angles", 10,
            std::bind(&AutoShoot::receiveCallback, this, std::placeholders::_1));

        // 订阅栅格地图
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", 10,
            std::bind(&AutoShoot::mapCallback, this, std::placeholders::_1));

        // 订阅当前机器人位姿
        position_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/position", 10,
            std::bind(&AutoShoot::positionCallback, this, std::placeholders::_1));

        // 订阅当前真实速度
        real_speed_sub_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
            "/real_speed", 10,
            std::bind(&AutoShoot::realSpeedCallback, this, std::placeholders::_1));

        // 订阅目标点位姿
        goal_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&AutoShoot::goalPoseCallback, this, std::placeholders::_1));

        // 发布目标云台角度
        send_data_pub_ = this->create_publisher<tdt_interface::msg::SendData>("/target_angles", 10);

        // 发布目标速度
        speed_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>("/target_speed", 10);

        // 发布比赛开始信号
        game_start_pub_ = this->create_publisher<std_msgs::msg::Bool>("/game_start", 10);

        publishGameStartSignal();
    }

private:	
    void publishGameStartSignal()
    {
        auto msg = std::make_shared<std_msgs::msg::Bool>();
        msg->data = true;
        game_start_pub_->publish(*msg);
        RCLCPP_INFO(this->get_logger(), "Game start");
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
    	auto start = std::chrono::steady_clock::now();
        cv::Mat frame;
        std::vector<uint8_t> jpeg_data(msg->data.begin(), msg->data.end());
        frame = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);

        if (frame.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to decode image.");
            return;
        }
        //MatSave(frame);
        
        TargetFound=0;//默认未找到目标
        whetherShoot=false;
        
		cv::Mat imgDil=ImgPrePro.Process(frame,5);//阈值化预处理
		//cv::imshow("imgDil",imgDil);
		DetectedLight=Cons.getContours(imgDil);
		if(DetectedLight.size()>1){
			std::vector<ArmorToPair> DetectedArmor=lp.PairProcess(DetectedLight,frame);//灯条配对
			if(DetectedArmor.size()>0){
				int minDistanceId=0;
				double minDistance=10000;
				for(int i=0;i<DetectedArmor.size();i++){
					//DetectedArmor[i].ArmorDraw(frame);
					if(DetectedArmor[i].GetArmorInfo().distance<minDistance){
						minDistanceId=i;
						minDistance=DetectedArmor[i].GetArmorInfo().distance;
					}
				}
				Armor Target(DetectedArmor[minDistanceId].GetArmorInfo());
                TargetThisFrame=Target;
				Target.ArmorDraw(frame,1);
                emptyFrame=0;
				
				TargetFound=1;
                if(whetherLimit==1){
                    if(Target.getArmorInfo().center.x<=frame.cols/2+35 && Target.getArmorInfo().center.x>=frame.cols/2-35 && Target.getArmorInfo().center.y<=frame.rows/2+35 &&Target.getArmorInfo().center.x>=frame.rows/2-35)
                        whetherShoot=true;
                }
                else{
                    whetherShoot=true;
                }
				cv::Point TargetPoint;
				
				if(LastTarget.getArmorInfo().id!=-1){
					double dt=minDistance/2800.0;
					TargetPoint=usingKalman(kalmanList,fps,&Target,&LastTarget,dt);
					cv::circle(frame,TargetPoint,3,cv::Scalar(0,0,255),-1);
				}
				else{
					TargetPoint=Target.getArmorInfo().center;
				}
				
				//开始计算yaw角
				double delta_x=frame.cols/2-Target.getArmorInfo().center.x;
				double yaw_change = delta_x * (90.0 / frame.cols);
				yaw=yaw-yaw_change;

				
				//计算pitch
				double delta_y=frame.rows/2-Target.getArmorInfo().center.y;
				double pitch_change=delta_y*(60.0/frame.rows);
				pitch=pitch+pitch_change;
				
				LastTarget=Target;
			}
		}
        /****************************图像处理结束***************************/
        
        std::string PositionText="Position:("+std::to_string(real_position_x)+","+std::to_string(real_position_y)+")";
        std::string TargetPositionText="Target:("+std::to_string(target_x)+","+std::to_string(target_y)+")";
        int baseline=0;
        cv::Size textSize = cv::getTextSize(PositionText, cv::FONT_HERSHEY_SIMPLEX, 1, 1,&baseline);
        cv::putText(frame,TargetPositionText,cv::Point(10,frame.rows-10),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame,PositionText,cv::Point(10,frame.rows-15-textSize.height),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame,std::to_string(fps),cv::Point(10,30),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
        
        std::string textYaw="Yaw:"+std::to_string(fmod(yaw, 360.0));
        std::string textPitch="Pitch:"+std::to_string(pitch);
        cv::putText(frame,textPitch,cv::Point(10,frame.rows-20-2*textSize.height),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
        cv::putText(frame,textYaw,cv::Point(10,frame.rows-25-3*textSize.height),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
        cv::Scalar Color;
        std::string velocity="speed(x:"+std::to_string(real_linear_speed_x)+",y:"+std::to_string(real_linear_speed_y)+")";
        cv::putText(frame,velocity,cv::Point(10,frame.rows-30-4*textSize.height),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 1);
        if(TargetFound==0){
            emptyFrame++;
        	Color=cv::Scalar(0,0,255);
        }
        else if(TargetFound==1 && whetherShoot==1){
        	Color=cv::Scalar(0,255,0);
        }
        else if(TargetFound==1 && whetherShoot==0){
        	Color=cv::Scalar(255,255,0);
        }
        if(whetherShowUi==1){
            if(whetherLimit==1){
                //画框
                cv::rectangle(frame, cv::Point(frame.cols/2-35,frame.rows/2-35), cv::Point(frame.cols/2+35,frame.rows/2+35),Color, 2);
            }
            //画准星
            cv::line(frame,cv::Point(frame.cols/2,frame.rows/2-10),cv::Point(frame.cols/2,frame.rows/2-2),Color,2);
            cv::line(frame,cv::Point(frame.cols/2,frame.rows/2+10),cv::Point(frame.cols/2,frame.rows/2+2),Color,2);
            cv::line(frame,cv::Point(frame.cols/2+2,frame.rows/2),cv::Point(frame.cols/2+10,frame.rows/2),Color,2);
            cv::line(frame,cv::Point(frame.cols/2-2,frame.rows/2),cv::Point(frame.cols/2-10,frame.rows/2),Color,2);
        }
        vs.saveFrame(frame);
		cv::imshow("Camera Image", frame);
        /******************************************/        
        int ret=cv::waitKey(1);
        if(ret=='o'){
        	if(whetherShoot==true)
        		whetherShoot=false;
        	else
        		whetherShoot=true;
        }
        else if(ret=='h'){
        	if(TargetFound==1)
        		TargetFound=0;
        	else if(TargetFound==0)
        		TargetFound=1;
        }
        else if(ret=='l')
        	yaw+=5;
        else if(ret=='j')
        	yaw-=5;
        else if(ret=='i')
        	pitch+=5;
        else if(ret=='k')
        	pitch-=5;
        else if(ret=='u'){
        	if(whetherShowUi==1)
        		whetherShowUi=0;
        	else
        		whetherShowUi=1;
        }
        else if(ret=='r'){
            target_x=-1;
            target_y=-1;
        }
        else if(ret=='q')
            vs.release();
        else if(ret=='f'){
            needNavi=1;
        }
        else if(ret=='1'){
            std::cout<<"Shoot Mode Switch!"<<std::endl;
            if(whetherLimit==1)
                whetherLimit=0;
            else
                whetherLimit=1;
        }
        else if(ret=='e'){
            emptyFrame=120;
        }
        	


        /********************发布你应该发布的角度**************************/
        if(emptyFrame>=90){
        	yaw+=9;
            pitch=0;
        }
        auto send_data_msg = std::make_shared<tdt_interface::msg::SendData>();
        send_data_msg->pitch = pitch;
        send_data_msg->yaw = yaw;
        send_data_msg->if_shoot = whetherShoot;

        send_data_pub_->publish(*send_data_msg);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> elapsedTime = end - start;
        if (elapsedTime.count() > 0) {
            fps = 1000 / elapsedTime.count();
        }
    }

    void realSpeedCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {

        real_linear_speed_x = msg->twist.linear.x;
        real_linear_speed_y = msg->twist.linear.y;
        /****************处理回调速度************************/
        float v=0;
        float vx=0;
        float vy=0;
        if(TargetFound==0){
            if(emptyFrame>=120){
                //长时间没有检测到目标后，前往下一个目标点
                if(target_x==-1 || target_y==-1){
                    if(Path.size()>0){
                        target_x=Path[0].x;
                        target_y=Path[0].y;
                        Path.erase(Path.begin());
                    }
                    else{
                        v=0;
                        vx=0;
                        vy=0;
                        //needNavi=1;
                    }
                }
                //<<"tx:"<<target_x<<" ty:"<<target_y<<std::endl;
                if(target_x != -1 && target_y!=-1){
                    float dx=target_x-real_position_x;
                    float dy=target_y-real_position_y;
                    //std::cout<<dx<<" "<<dy<<std::endl;
                    float TargetPointDistance =sqrt(dx * dx + dy * dy);
                    float direction_x=dx/TargetPointDistance;
                    float direction_y=dy/TargetPointDistance;
                    if(TargetPointDistance>3)
                        v=TargetPointDistance*1.5;
                    else if(TargetPointDistance>1.5 && TargetPointDistance<=3)
                        v=3.5;
                    else if(TargetPointDistance<=1.5 && TargetPointDistance>0.25)
                        v=1;
                    else{
                        v=0;
                        target_x=-1;
                        target_y=-1;
                    }
                    vx=v*direction_x;
                    vy=v*direction_y;
                }
            }
        }
        else{//TargetFound==1
            if(TargetThisFrame.getDistance()>1600){
                v=2.0;
                float yaw_=fmod(yaw, 360.0);
                vx=-(v*sin(yaw_*M_PI/180));
                vy=-(v*cos(yaw_*M_PI/180));
            }
            else if(TargetThisFrame.getDistance()<1200){
                v=2.0;
                float yaw_=fmod(yaw, 360.0);
                vx=(v*sin(yaw_*M_PI/180));
                vy=(v*cos(yaw_*M_PI/180));
            }
            else if(TargetThisFrame.getDistance()>=800 && TargetThisFrame.getDistance()<=1600){
                v=0;
                vx=0;
                vy=0;
            }
        }
        /*******************发布期望速度***********************/
        //std::cout<<"vx:"<<vx<<" "<<"vy:"<<vy<<std::endl;
        auto target_speed_msg=std::make_shared<geometry_msgs::msg::TwistStamped>();
        target_speed_msg->twist.linear.x=vx;
        target_speed_msg->twist.linear.y=vy;
        target_speed_msg->header.stamp=this->get_clock()->now();
        speed_pub_->publish(*target_speed_msg);
    }

    // 云台角度回调
    void receiveCallback(const tdt_interface::msg::ReceiveData::SharedPtr msg)
    {
        pitch = msg->pitch;
        yaw = msg->yaw;
        //RCLCPP_INFO(this->get_logger(),"Real yaw:%f,real pitch %f]", yaw, pitch);
    }

    // 栅格地图回调
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
    {
        /*****************************保存或处理你的地图*****************************/
        int width = msg->info.width;
        int height = msg->info.height;
        cv::Mat map_image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int index = x + y * width;
                int8_t occupancy_value = msg->data[index];
                uint8_t pixel_value = 0;

                if (occupancy_value == 0)
                    pixel_value = 255;
                else if (occupancy_value == 100)
                    pixel_value = 0;
                else
                    pixel_value = 128;

                map_image.at<uint8_t>(y, x) = pixel_value;
            }
        }
        if(!map_image.empty()){
            if(navi.whetherIni()==0){
                navi.initialNavigation(map_image);
                navi.FindRectangles();
            }
        }
    }

    void positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        /***********************处理自身位置信息**************************/
        real_position_x=msg->pose.position.x;
        real_position_y=msg->pose.position.y;
        if(real_position_x==0){
            real_position_x=1;
        }
        if(real_position_y==0){
            real_position_y=1;
        }
        navi.showPosition(real_position_x,real_position_y);//实时显示自己位置
        if(needNavi==1){
            std::cout<<"Navi on"<<std::endl;
            Path=navi.useNavi(cv::Point(real_position_x,real_position_y));
            needNavi=0;
        }
        //RCLCPP_INFO(this->get_logger(), "Robot position: [x: %f, y: %f, z: %f]", msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    }

    void goalPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
    	target_x=msg->pose.position.x;
    	target_y=msg->pose.position.y;
    }


    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr real_speed_sub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr speed_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr game_start_pub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr position_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_sub_;
    rclcpp::Publisher<tdt_interface::msg::SendData>::SharedPtr send_data_pub_;
    rclcpp::Subscription<tdt_interface::msg::ReceiveData>::SharedPtr receive_data_sub_;

    float yaw;
    float pitch;

    float real_linear_speed_x=0;
    float real_linear_speed_y=0;

    float real_position_x=1;
    float real_position_y=1;

    float target_x=-1;
    float target_y=-1;

    int TargetFound=0;
    bool whetherShoot=false;
    int whetherShowUi=1;

    PreProcess ImgPrePro;
    Contours Cons;
    std::vector<LightRect> DetectedLight;
    LightPair lp;
    Armor LastTarget;

    std::vector<Kalman> kalmanList;
    double fps;

    Armor TargetThisFrame;
    int emptyFrame=120;
    int whetherLimit=1;

    Navigation navi;

    std::vector<cv::Point> Path;

    int needNavi=1;

    VideoSaver vs;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AutoShoot>());
    rclcpp::shutdown();
    return 0;
}
