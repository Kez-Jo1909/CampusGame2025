//
// Created by kezjo on 24-10-3.
//

#include "Armor.h"
#include<opencv2/opencv.hpp>
#include"../camera/Camera.h"

ArmorToPair::ArmorToPair(LightRect lr1, LightRect lr2) {
    if(lr1.rR.center.x <= lr2.rR.center.x) {
        leftLight = lr1;
        rightLight = lr2;
    }
    else {
        leftLight = lr2;
        rightLight = lr1;
    }
    width=std::abs(this->leftLight.rR.center.x - this->rightLight.rR.center.x);
    height=std::max(std::max(this->leftLight.rR.size.height,this->leftLight.rR.size.width),std::max(this->rightLight.rR.size.height,this->rightLight.rR.size.width));
    angle=(this->leftLight.rR.angle+this->rightLight.rR.angle)/2;
    ArmorCenter=(leftLight.rR.center+rightLight.rR.center)/2;
    predict=4;
}

void ArmorToPair::ArmorDraw(cv::Mat img) {
    cv::Scalar color=cv::Scalar(255,255,255);
    cv::circle(img, ArmorCenter, 5, color,-1);
    for (int j = 0; j < 4; j++) {
        cv::line(img, this->leftLight.vertices[j], this->leftLight.vertices[(j + 1) % 4], color, 2);
        cv::line(img, this->rightLight.vertices[j], this->rightLight.vertices[(j + 1) % 4], color, 2);
    }
    std::string PredictText=std::to_string(distance);
    cv::putText(img,PredictText,ArmorCenter,cv::FONT_HERSHEY_SIMPLEX,1,color,2);
    // std::string DistanceText=std::to_string(this->distance);
    // cv::putText(img,DistanceText,ArmorCenter,cv::FONT_HERSHEY_SIMPLEX,1,cv::Scalar(0,0,255),2);
    cv::Point2f points[4];
    rrect.points(points);
    for(int j=0;j<4;j++) {
        cv::line(img,points[j],points[(j + 1) % 4],color,3);
    }
}

cv::Mat ArmorToPair::getRoiOfInterest(cv::Mat img) {
    cv::Mat roi;

    int length=(this->height+this->width)*2/5;
    //int length=(this->height+this->width)/2;

    if(angle>60) {
        rrect=cv::RotatedRect(ArmorCenter,cv::Size(length*7/5*3/2,length*1.1),angle);
    }
    else {
        rrect=cv::RotatedRect(ArmorCenter,cv::Size(length*1.1,length*7/5*3/2),angle);
    }
    //std::cout<<angle<<std::endl;
    cv::Rect boundingRect=rrect.boundingRect();//找到外接矩形
    cv::Mat copyImg;
    img.copyTo(copyImg);
    cv::Mat rotatedImage;
    cv::Mat rotationMatrix;
    if(rrect.angle>45) {
        rotationMatrix=cv::getRotationMatrix2D(rrect.center,-(90-rrect.angle),1.0);
    }
    else {
        rotationMatrix=cv::getRotationMatrix2D(rrect.center,rrect.angle,1.0);
    }
    cv::warpAffine(copyImg, rotatedImage, rotationMatrix, copyImg.size(), cv::INTER_CUBIC);
    boundingRect &= cv::Rect(0, 0, rotatedImage.cols, rotatedImage.rows);
    roi = rotatedImage(boundingRect);
    //cv::imshow("video",img);
    //cv::resize(roi,roi,cv::Size(200,280));
    //cv::imshow("roi",roi);
    cv::resize(roi,roi,cv::Size(20,28));
    return roi;
}

double ArmorToPair::GetDistance(cv::Mat img) {


    //3d点坐标
    std::vector<cv::Point3f> Object_points ={
        {-67,27.5,0},//左上
        {-67,0,0},//左中
        {-67,-27.5,0},//左下
        {67,27.5,0},//右上
        {67,0,0},//右中
        {67,-27.5,0},//右下
   };

    std::vector<cv::Point2f> imagePoints={
        {this->leftLight.rR.center.x,this->leftLight.rR.center.y-this->height/2},//top left
        this->leftLight.rR.center, //left center
        {this->leftLight.rR.center.x,this->leftLight.rR.center.y+this->height/2}, //bottom left
        {this->rightLight.rR.center.x,this->rightLight.rR.center.y-this->height/2},
        this->rightLight.rR.center,
        {this->rightLight.rR.center.x,this->rightLight.rR.center.y+this->height/2},
    };

    bool Success=cv::solvePnP(Object_points, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_SQPNP);

    if(Success) {
        distance = cv::norm(tvec);
    }
    else {
        std::cerr<<"solvePnP failed"<<std::endl;
        return -1;
    }
    return distance;
}

ArmorInfo ArmorToPair::GetArmorInfo() {
    ArmorInfo armorInfo;
    armorInfo.center=ArmorCenter;
    armorInfo.distance=distance;
    armorInfo.rvec=rvec;
    armorInfo.tvec=tvec;
    armorInfo.id=predict;
    armorInfo.left=leftLight;
    armorInfo.right=rightLight;
    armorInfo.roi=rrect;
    return armorInfo;
}

Armor::Armor(ArmorInfo ArIf) {
    info=ArIf;
    isDetectedThisFrame=1;
}

void Armor::ArmorDraw(cv::Mat img,int whetherT) {
    if(isDetectedThisFrame==0) {
        std::cerr<<"Armor"<<info.id<<" not detected this frame"<<std::endl;
        return;
    }
    cv::Scalar color=cv::Scalar(0,255,255);
    if(whetherT==1) {
        color=cv::Scalar(0,255,0);
    }
    cv::circle(img, info.center, 5, color,-1);
    for (int j = 0; j < 4; j++) {
        cv::line(img, this->info.left.vertices[j], this->info.left.vertices[(j + 1) % 4], color, 2);
        cv::line(img, this->info.right.vertices[j], this->info.right.vertices[(j + 1) % 4], color, 2);
    }
    std::string PredictText=std::to_string(info.distance);
    cv::putText(img,PredictText,info.center,cv::FONT_HERSHEY_SIMPLEX,1,color,2);
    cv::Point2f points[4];
    info.roi.points(points);
    for(int j=0;j<4;j++) {
        cv::line(img,points[j],points[(j + 1) % 4],color,3);
    }
}

void Armor::renew() {
    isDetectedThisFrame=0;
}

double Armor::getDistance() {
    return info.distance;
}

int Armor::getID() {
    return info.id;
}

int ArmorToPair::getPre() {
    return predict;
}

void Armor::renewArmor(ArmorInfo ArIF) {
    info=ArIF;
    isDetectedThisFrame=1;
}

int Armor::whetherFound() {
    return isDetectedThisFrame;
}

Armor::Armor() {
    info.id=-1;
    isDetectedThisFrame=0;
    info.center.x=0;
    info.center.y=0;
}


void Armor::Calculate() {
    //卡尔曼
}

ArmorInfo Armor::getArmorInfo() {
    return info;
}




