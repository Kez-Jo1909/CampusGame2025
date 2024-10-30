//
// Created by kezjo on 24-10-6.
//

#include "Kalman.h"
#include<Eigen/Dense>
#include<iostream>
#include"../Armor/Armor.h"
#include"../camera/Camera.h"

void Kalman::Initialize(const Eigen::VectorXd& X_in) {
    X_=X_in;
}

void Kalman::setF(double dt) {
    F_<<1,0,0,dt,0,0,
        0,1,0,0,dt,0,
        0,0,1,0,0,dt,
        0,0,0,1,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1;
}

void Kalman::Predict() {
    if (X_.size() != 6 || F_.cols() != 6 || P_.cols() != 6) {
        std::cerr << "Error: Size mismatch in Predict() - X_: " << X_.size()
                  << ", F_: (" << F_.rows() << " x " << F_.cols()
                  << "), P_: (" << P_.rows() << " x " << P_.cols() << ")" << std::endl;
    }
    X_=F_*X_;
    Eigen::MatrixXd Ft=F_.transpose();
    P_=F_*P_*Ft+Q_;
}

Kalman::Kalman() {
    initialized = false;
    Q_.setIdentity(6,6);
    P_.setIdentity(6,6);
    F_.setIdentity(6,6);
    X_ = Eigen::VectorXd(6);
    H_ << 1,0,0,0,0,0,
         0,1,0,0,0,0,
         0,0,1,0,0,0;
    double Q_factor=0.1;
    double R_factor=0.001;
    Q_=Q_*Q_factor;
    R_=R_*R_factor;
    //std::cout<<Q_<<std::endl;
}

void Kalman::MeasurementUpdate(const Eigen::VectorXd &z) {
    Eigen::VectorXd y=z-H_*X_;
    Eigen::MatrixXd S=H_*P_*H_.transpose()+R_;
    Eigen::MatrixXd K=P_*H_.transpose()*S.inverse();//kalman增益
    X_=X_+K*y;
    int size=X_.size();
    Eigen::MatrixXd I=Eigen::MatrixXd::Identity(size,size);
    P_=(I-K*H_)*P_;
    //std::cout<<P_<<std::endl;
}

Eigen::VectorXd Kalman::getX() {
    return X_;
}

cv::Point usingKalman(std::vector<Kalman>& kl,double fps,Armor* TargetArmors,Armor* LastTarget,double deltaT) {
    Eigen::VectorXd X_in(6);
    double dt = 1 / fps;
    double vx = (TargetArmors->getArmorInfo().tvec.at<double>(0) - LastTarget->getArmorInfo().tvec.at<double>(0))*fps;
    double vy = (TargetArmors->getArmorInfo().tvec.at<double>(1) - LastTarget->getArmorInfo().tvec.at<double>(1))*fps;
    double vz = (TargetArmors->getArmorInfo().tvec.at<double>(2) - LastTarget->getArmorInfo().tvec.at<double>(2))*fps;

    X_in << LastTarget->getArmorInfo().tvec.at<double>(0), LastTarget->getArmorInfo().tvec.at<double>(1), LastTarget->getArmorInfo().tvec.at<double>(2), vx, vy, vz;
    kl[0].Initialize(X_in);
    kl[0].setF(dt);

    kl[0].Predict();

    Eigen::VectorXd z(3);
    z << TargetArmors->getArmorInfo().tvec.at<double>(0),
         TargetArmors->getArmorInfo().tvec.at<double>(1),
         TargetArmors->getArmorInfo().tvec.at<double>(2);

    kl[0].MeasurementUpdate(z);
    Eigen::VectorXd get_x = kl[0].getX();
    Eigen::VectorXd predictX(3);
    predictX<<(get_x(0)+get_x(3)*deltaT),(get_x(1)+get_x(4)*deltaT),(get_x(2)+get_x(5)*deltaT);
    cv::Point predictP=projectTo2D(predictX);
    return predictP;
}
