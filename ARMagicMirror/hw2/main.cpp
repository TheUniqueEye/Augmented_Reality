//  Created by Jing Yan(眼睛) on 2016/10/27.
//  Copyright © 2016年 Jing Yan(眼睛). All rights reserved.


/* Assignment 2: AR Magic Mirror
 
 0. Based on the code of Assignment 1 - camera calibration.
 1. Build a "magic mirror" application which augments the view from your laptop camera with 3D graphics.
 2. Implement computer vision techniques to recognize gestures from the user for interaction.
 
 Key Control
 Press r  - to toggle the yellow rectangle around face + 4 red circles
 Press o - to toggle the display of spinning object: teapot, cone, torus
 Press g - to toggle the display of threshold images
 Press f - to toggle the detection mode: background subtraction, optical flow
 
 Gesture Control
 Use your hand to touch the green and red button on the screen, to choose different spinning objects
 
 */

#include <iostream>
#include <stdio.h>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/background_segm.hpp"
#include <GLUT/glut.h>
#include <vector>


using namespace cv;
using namespace std;


VideoCapture cap(0); // open the default camera

//using the test video stream
//VideoCapture cap("/Users/EYE/Desktop/hw1/testcamera/checkerboard.mov");

int IMAGE_NUM = 11;
vector<Mat> img(IMAGE_NUM);
vector<Mat> gray_image(IMAGE_NUM);
int frameCount=0;
int touchCount=0;


bool patternfound;
Size patternSize(8,6); //interior number of corners
float squareSize = 1.f; // & "1 arbitrary unit"
vector<vector<Point2f> > imageCorners(IMAGE_NUM);
vector<vector<Point3f> > objectCorners(IMAGE_NUM);

vector<Mat> rotationVectors(IMAGE_NUM);
vector<Mat> translationVectors(IMAGE_NUM);
Mat distortionCoefficients;
Mat cameraMatrix;
Size imageSize=img[0].size();
double rms;

vector<float> reprojErrs;
double totalAvgErr;
Point2d principalPoint;
double focalLength,aspectRatio;
double fovx,fovy;

GLdouble fx; // get Fx,Fy,Cx,Cy from Camera Matrix
GLdouble fy;
GLdouble cx;
GLdouble cy;

vector<Point2f> imageCorners_temp;
vector<Point3f> objCorners_temp;

Mat frame,pre_frame,diff;
Mat tempframe;
Mat img_temp,img_temp_resized;
Mat rvec_temp = Mat::eye(3, 1, CV_64F);
Mat tvec_temp = Mat::eye(3, 1, CV_64F);
GLfloat transformMat[16];


float img_ratio = 1280.f / 720.f;  // camera image: 1280,720
float resize_ratio = 1280.f / 320.f; // resize ratio
float widthS = 320.f; // resize width
float heightS = 320.f/img_ratio; // resize height

Mat img_undistort;
Mat gray_temp,gray_temp_resized;
Mat img_back,img_diff,img_diff_resized,img_threshold,img_threshold_color;


//Mat img_flow = Mat::eye(widthS, widthS/img_ratio, CV_32FC2);
Mat mag = Mat::eye(widthS, widthS/img_ratio, CV_32F);
Mat img_flow;
//Mat mag;
Mat mag_resized, mag_threshold,mag_threshold_color;


GLboolean mode_wire=true; // mode switches
GLboolean mode_back=false;
GLboolean mode_thres=false;
GLboolean mode_detect=true;
GLboolean mode_dis=false;
bool touched=false;
int mode_obj=2;

String face_cascade_name = "/Users/EYE/Desktop/HW2/haarcascade_frontalface_default.xml";
String eyes_cascade_name = "/Users/EYE/Desktop/HW2/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

vector<Rect> faces;
float headHeight = 15;
float headWidth = 15;
float headDis;


float i=0; // turning degree

// Set lighting intensity and color
GLfloat qaAmbientLight[]    = {0.2, 0.2, 0.2, 1.0};
GLfloat qaDiffuseLight[]    = {1, 1, 1, 1.0};
GLfloat qaSpecularLight[]    = {1.0, 1.0, 1.0, 1.0};
// Light source position
GLfloat qaLightPosition[]    = {0, 0, 0, 1};// Positional Light
GLfloat qaLightDirection[]    = {1, 1, 1, 0};// Directional Light



vector<Point3f> Create3DChessboardCorners( Size boardSize, float squareSize);
void detectAndDisplay( Mat frame );

double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                 const vector<vector<Point2f> >& imagePoints,
                                 const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                 const Mat& cameraMatrix , const Mat& distCoeffs,
                                 vector<float>& perViewErrors);

/////////////////////////////////////////////////////////////////////////////////

// This function creates the 3D points of your chessboard in its own coordinate system
// reference: https://github.com/daviddoria/Examples/blob/master/c%2B%2B/OpenCV/DrawChessboardCorners/DrawChessboardCorners.cxx

vector< Point3f> Create3DChessboardCorners( Size boardSize, float squareSize){
    
    vector< Point3f> corners;
    
    for( int i = 0; i < boardSize.height; i++ ){
        for( int j = 0; j < boardSize.width; j++ ){
            corners.push_back( Point3f(float(j*squareSize),float(i*squareSize), 0));
        }
    }
    return corners;
}


/////////////////////////////////////////////////////////////////////////////////

// The function returns the average re-projection error.
// reference: http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html#source-code

double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                 const vector<vector<Point2f> >& imagePoints,
                                 const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                 const Mat& cameraMatrix , const Mat& distCoeffs,
                                 vector<float>& perViewErrors){
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());
    
    for( i = 0; i < (int)objectPoints.size(); ++i ){
        projectPoints( Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, // project
                      distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2); // norm - difference
        
        int n = (int)objectPoints[i].size(); // points per frame
        perViewErrors[i] = (float) sqrt(err*err/n); // save for this view &
        totalErr        += err*err;  // sum it up &
        totalPoints     += n;
    }
    return sqrt(totalErr/totalPoints); // calculate the arithmetical mean
}

/////////////////////////////////////////////////////////////////////////////////

// openGL - a useful function for displaying your coordinate system
// reference: http://www.cs.ucsb.edu/~holl/CS291A/opengl_cv.cpp

void drawAxes(float length)
{
    glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT) ;
    
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) ;
    glDisable(GL_LIGHTING) ;
    
    glBegin(GL_LINES) ;
    glColor3f(1,0,0) ;
    glVertex3f(0,0,0) ;
    glVertex3f(length,0,0);
    
    glColor3f(0,1,0) ;
    glVertex3f(0,0,0) ;
    glVertex3f(0,length,0);
    
    glColor3f(0,0,1) ;
    glVertex3f(0,0,0) ;
    glVertex3f(0,0,length);
    glEnd() ;
    
    glPopAttrib() ;
}

/////////////////////////////////////////////////////////////////////////////////

// Function for face detection http://docs.opencv.org/3.0-beta/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
// 0.Detecting face / 1.draw rectangle / 2.compute the distance from face to camera

void detectAndDisplay()
{
    
    // Downsample the camera image to speed up the classifier
   
    float offset = 10.f*resize_ratio;
    
    resize(gray_temp, gray_temp_resized, Size( widthS, heightS), 1.0, 1.0, INTER_CUBIC);
    
    
    equalizeHist( gray_temp_resized, gray_temp_resized );
  
    //-- Detect faces
    face_cascade.detectMultiScale( gray_temp_resized, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30) );
    
   
    
    // for each detected face, draw a rectangle cover the face
    for( size_t i = 0; i < faces.size(); i++ ){
        
        Point pt1(faces[i].x*resize_ratio,faces[i].y*resize_ratio);
        Point pt2((faces[i].x+faces[i].width)*resize_ratio,(faces[i].y+faces[i].height)*resize_ratio);
        if(mode_wire) {
        rectangle(img_undistort,pt1, pt2, Scalar( 0, 255, 255 ));
        }
        // Computing the distance from head to camera
        headDis = fy * headHeight /(faces[i].height*resize_ratio);

        
        // create 2D image points
        imageCorners_temp.clear();
        imageCorners_temp.push_back(Point2f(  faces[0].x*resize_ratio,  faces[0].y*resize_ratio ));
        // left bottom
        imageCorners_temp.push_back(Point2f(  faces[0].x*resize_ratio,  (faces[0].y+faces[0].height)*resize_ratio )); // left top
        imageCorners_temp.push_back(Point2f(  (faces[0].x+faces[0].width)*resize_ratio, (faces[0].y+faces[0].height)*resize_ratio )); // right top
        imageCorners_temp.push_back(Point2f(  (faces[0].x+faces[0].width)*resize_ratio , faces[0].y*resize_ratio  )); // right bottom
        
    }

}

/////////////////////////////////////////////////////////////////////////////////

// drawOnPixel function : change pixel value of an image

void drawOnPixel(int left, int right, int top, int bottom,
                 Mat img, double blue, double green, double red ){
    
    for(int i=left; i<right; i++){
        for(int j=top; j<bottom; j++){
            
            img.at<Vec3b>(j, i).val[0] = blue; // blue
            img.at<Vec3b>(j, i).val[1] = green; // green
            img.at<Vec3b>(j, i).val[2] = red; // red
        }
    }
    
    
}

/////////////////////////////////////////////////////////////////////////////////

//buttonTouched function : Count foreground pixels - to know if certain button is touched

bool buttonTouched(int left, int right, int top, int bottom, Mat img){
    
    int count=0;

    for(int i=left; i<right; i++){
        for(int j=top; j<bottom; j++){
            double blue = img.at<Vec3b>(j, i).val[0]; // green
            double green = img.at<Vec3b>(j, i).val[1]; // green
            double red = img.at<Vec3b>(j, i).val[2]; // red
            
            if(blue==255 && green==255 && red==255) count++;
            
            if(count>2000) touched =true;
            else touched =false;
        }
    }
    //cout<<"count = " << count<<endl;
    return touched;
}



/////////////////////////////////////////////////////////////////////////////////

void idle(){

    //------------------ get frame and mirror image ------------------
    
    cap >> frame; // get a new frame from camera
    
    if(!frame.data ){
        cout <<  "frame data not loaded properly" << endl ;
    }
    
    flip(frame, tempframe, 1); // mirror the camera image
    
    img_temp = tempframe.clone();

    
    //------------------ undistort  frame image  ------------------
    Mat map1, map2;
    initUndistortRectifyMap(cameraMatrix, distortionCoefficients, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, img_temp.size(), 1, img_temp.size(), 0),  img_temp.size(), CV_16SC2, map1, map2);
    
    undistort( img_temp, img_undistort, cameraMatrix, distortionCoefficients);
    remap(img_temp, img_undistort, map1, map2, INTER_LINEAR);
    
    cvtColor( img_undistort, gray_temp, CV_BGR2GRAY );
    

    //------------------ Face detection  ------------------
    detectAndDisplay();
    
    
    // ------------------ 3D-2D point correspondences  ------------------
    
    // get Fx,Fy,Cx,Cy from Camera Matrix
    fx= cameraMatrix.at<double>(0,0);
    fy = cameraMatrix.at<double>(1,1);
    cx = cameraMatrix.at<double>(0,2);
    cy = cameraMatrix.at<double>(1,2);


    // create 3D object points
  
    objCorners_temp.clear();
    objCorners_temp.push_back(Point3f(-headWidth/2,-headHeight/2,0)); // left bottom
    objCorners_temp.push_back(Point3f(-headWidth/2,headHeight/2,0)); // left top
    objCorners_temp.push_back(Point3f(headWidth/2,headHeight/2,0)); // right top
    objCorners_temp.push_back(Point3f(headWidth/2,-headHeight/2,0)); // right bottom
    
    // get rotation and translation vectors
    if(faces.size()!=0){
        solvePnP(objCorners_temp, imageCorners_temp, cameraMatrix, distortionCoefficients, rvec_temp, tvec_temp);
        
        // construct 4*4 transformation matrix
        transformMat[0] = -1;
        transformMat[1] = 0;
        transformMat[2] = 0;
        transformMat[3] = 0;
        transformMat[4] = 0;
        transformMat[5] = -1;
        transformMat[6] = 0;
        transformMat[7] = 0;
        transformMat[8] = 0;
        transformMat[9] = 0;
        transformMat[10] = 1;
        transformMat[11] = 0;
        transformMat[12] = tvec_temp.at<double>(0,0);
        transformMat[13] = tvec_temp.at<double>(0,1);
        transformMat[14] = -tvec_temp.at<double>(0,2);
        transformMat[15] = 1.0;
        
    }
    
    
    //------------------ motion detection / background subtraction  ------------------
    
    // store background img and compute per-pixel difference
    if(mode_back){
        img_back = gray_temp_resized.clone();
        mode_back = false;
    }
    
    if(frameCount==0) { // Init
        img_back = gray_temp_resized.clone();
        pre_frame = gray_temp_resized.clone();
        
        resize(img_temp, img_temp_resized, Size( widthS, heightS), 1.0, 1.0, INTER_CUBIC);
        diff = img_temp_resized.clone();
    }
    
    img_diff = abs(img_back - gray_temp_resized);
    
    // resize difference image
    resize(img_diff, img_diff_resized, Size(1280,720), 1.0, 1.0, INTER_CUBIC);
    threshold(img_diff_resized, img_threshold, 40., 255, THRESH_BINARY);
    frameCount++;
    
    // convert threshold image back to RGB then flip to display
    cvtColor( img_threshold, img_threshold_color, CV_GRAY2BGR );
    flip(img_threshold_color, img_threshold_color, 0);
    

    
    //------------------ motion detection / optical flow  ------------------
    
    // get a resized color image as container
    resize(img_temp, img_temp_resized, Size( widthS, heightS), 1.0, 1.0, INTER_CUBIC);
    diff = img_temp_resized.clone();
    
    // compute flow between previous and current frame
    calcOpticalFlowFarneback(pre_frame, gray_temp_resized, img_flow, 0.5, 1, 3, 1, 5, 1.1, 0);
    
    // compute the magnitude of flow vector at each pixel
    // store value to the R,G,B channel of the container
    for(int row=0; row<widthS; row++){
        for(int col=0; col<heightS ; col++){
            
            Point2d flow_point = img_flow.at<Point2d>(col,row/2); // horizontal: duplicate image
            diff.at<Vec3b>(col,row).val[0] = sqrt(flow_point.x*flow_point.x + flow_point.y*flow_point.y);
            diff.at<Vec3b>(col,row).val[1] = sqrt(flow_point.x*flow_point.x + flow_point.y*flow_point.y);
            diff.at<Vec3b>(col,row).val[2] = sqrt(flow_point.x*flow_point.x + flow_point.y*flow_point.y);
            
        }
    }
    
    flip(diff, diff, 0); // flip upside down
    
    // resize to large image and set threshold at 30
    resize(diff, diff, Size(1280,720), 1.0, 1.0, INTER_CUBIC);
    threshold(diff, diff, 30., 255, THRESH_BINARY);
    mag_threshold_color = diff.clone();
    
    pre_frame = gray_temp_resized.clone(); // previous frame
    
    
    //------------------ Button Touched  ------------------
    
    if(touchCount>12){
        
        if(mode_detect){
            
            if(buttonTouched(50,150,550,650,img_threshold_color)){
                touchCount = 0;
                if(mode_obj<4)
                    mode_obj+=1;
                else
                    mode_obj = 2;
            }
            
            if(buttonTouched(1050,1150,550,650,img_threshold_color)){
                touchCount = 0;
                if(mode_obj>2) mode_obj-=1;
                else mode_obj = 4;
            }
        }else{
            
            if(buttonTouched(50,150,550,650,mag_threshold_color)){
                touchCount = 0;
                if(mode_obj<4) mode_obj+=1;
                else mode_obj = 2;
            }
            
            if(buttonTouched(1050,1150,550,650,mag_threshold_color)){
                touchCount = 0;
                if(mode_obj>2) mode_obj-=1;
                else mode_obj = 4;
            }
        }
    }
    touchCount++;
}

/////////////////////////////////////////////////////////////////////////////////

void display(){
    
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the window
    
    Mat tempimage;
    flip(img_undistort, tempimage, 0); //mirror the image
    
    
    
    //------------------ Create Two Button & Change Button Color ------------------
    
    
    //Camera frame display
    if(buttonTouched(50,150,550,650,img_threshold_color) || buttonTouched(50,150,550,650,mag_threshold_color)) // left
        drawOnPixel(50,150,550,650,tempimage,0,255,255); // yellow
    else
        drawOnPixel(50,150,550,650,tempimage,0,255,0); // green
    
    if(buttonTouched(1050,1150,550,650,img_threshold_color) || buttonTouched(1050,1150,550,650,mag_threshold_color)) // right
        drawOnPixel(1050,1150,550,650,tempimage,0,255,255); // yellow
    else
        drawOnPixel(1050,1150,550,650,tempimage,0,0,255); // red
    
    
    // Background substraction display
    if(mode_detect){
        if(buttonTouched(50,150,550,650,img_threshold_color)) // left
            drawOnPixel(50,150,550,650,img_threshold_color,0,255,255); // yellow
        else
            drawOnPixel(50,150,550,650,img_threshold_color,0,255,0); // green
        
        if(buttonTouched(1050,1150,550,650,img_threshold_color)) // right
            drawOnPixel(1050,1150,550,650,img_threshold_color,0,255,255); // yellow
        else
            drawOnPixel(1050,1150,550,650,img_threshold_color,0,0,255); // red
    }
    
    // Optical flow display
    else{
        if(buttonTouched(50,150,550,650,mag_threshold_color)) // left
            drawOnPixel(50,150,550,650,mag_threshold_color,0,255,255); // yellow
        else
            drawOnPixel(50,150,550,650,mag_threshold_color,0,255,0); // green
        
        if(buttonTouched(1050,1150,550,650,mag_threshold_color)) // right
            drawOnPixel(1050,1150,550,650,mag_threshold_color,0,255,255); // yellow
        else
            drawOnPixel(1050,1150,550,650,mag_threshold_color,0,0,255); // red
    }
    
    
    //------------------ display camera image & threshold image  ------------------
    
    if(mode_thres){ // press "g" to display threshold image
        if(mode_detect)
        glDrawPixels( img_threshold_color.size().width, img_threshold_color.size().height, GL_BGR, GL_UNSIGNED_BYTE, img_threshold_color.ptr() );
        else
         glDrawPixels( mag_threshold_color.size().width, mag_threshold_color.size().height, GL_BGR, GL_UNSIGNED_BYTE, mag_threshold_color.ptr() );
        
    }else{ // display camera image
        glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    }
    
    
    //------------------ View & Projection  ------------------
        
    //set viewport
    glViewport(0, 0, tempimage.size().width, tempimage.size().height);
        
    //set projection matrix using intrinsic camera params
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
        
    
    // glFrustum — multiply the current matrix by a perspective matrix
    // reference: https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
    //http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        
    
    
    GLdouble near = 0.001;
    GLdouble far = 500;
        
    glFrustum(0-cx*(near/fx), // left
              (img_undistort.size().width-cx)*(near/fx), // right
              (img_undistort.size().height-cy)*(near/fy), // bottom
              0-cy*(near/fy), // top
              near,far);
        
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    glMultMatrixf(transformMat);
    
    
    
    //------------------ 3D Red Points  ------------------
    
    if(mode_wire){
    glDisable(GL_LIGHTING) ;
    glPushMatrix();
    glTranslated(-headWidth/2, -headHeight/2, 0);
    glColor3f(1., 0., 0.);
    glutSolidSphere(0.2, 50, 50);
    glPopMatrix();
    
    glPushMatrix();
    glTranslated(headWidth/2, -headHeight/2, 0);
    glColor3f(1., 0., 0.);
    glutSolidSphere(0.2, 50, 50);
    glPopMatrix();
    
    glPushMatrix();
    glTranslated(-headWidth/2, headHeight/2, 0);
    glColor3f(1., 0., 0.);
    glutSolidSphere(0.2, 50, 50);
    glPopMatrix();
    
    glPushMatrix();
    glTranslated(headWidth/2, headHeight/2, 0);
    glColor3f(1., 0., 0.);
    glutSolidSphere(0.2, 50, 50);
    glPopMatrix();
    }
  
    //------------------ Surface Occlusion  ------------------
    
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable( GL_DEPTH_TEST );
    glEnable(GL_LIGHTING) ;
    
    
    glPushMatrix();
    glColor3f(1., 0., 0.);
    
    // Draw a rectangle for occlusion
    glBegin(GL_POLYGON);
    glVertex2f(-headWidth/2,-headHeight/2);
    glVertex2f(-headWidth/2, headHeight/2);
    glVertex2f(headWidth/2, headHeight/2);
    glVertex2f(headWidth/2, -headHeight/2);
    glEnd();
    
    //glutSolidCube(10); // will be slightly shift from the center point
    
    glPopMatrix();
    
    
    //------------------ Spinning teapot/cone/donut  ------------------
    
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    
    if(faces.size()!=0){
        
        glColor3f(1., 1., 1.);
        glColorMaterial(GL_FRONT_AND_BACK,GL_DIFFUSE);
        
        
       
        
        // Rotation
        float x=0.0;
        float radius=(headWidth/2.0 + 4.0)*(headWidth/2.0 + 4.0);

        if(mode_dis){
            
      
        if(i<=360){
            glPushMatrix();
            glRotated(i, 0, 1, 0);
            
            if(x<=3){
                x=x+0.1;
            }else{
                x=0;
            }
            glTranslatef(x, 0, sqrt(radius-x*x));
            
            // changing mode: teapot/cone/donut
            //cout<<"mode = "<<mode_obj<<endl;
            
            if(mode_obj==2){
                glutSolidTeapot(2.5);
            }else if (mode_obj==3){
                glutSolidCone(2, 3, 50, 50);
            }else if (mode_obj==4){
                glutSolidTorus(1,2,50,50);
            }
            glPopMatrix();
            i=i+5;
            
        }else{
            i=0;
            glTranslatef(0, 0, 3);
            if(mode_obj==2){
                glutSolidTeapot(2.5);
            }else if (mode_obj==3){
                glutSolidCone(2, 3, 50, 50);
            }else if (mode_obj==4){
                glutSolidTorus(1,2,50,50);
            }
        }
    }
    
     glDisable(GL_DEPTH_TEST);
    
   }
    
    // show the rendering on the screen
    glutSwapBuffers();
    
    // post the next redisplay
    glutPostRedisplay();
}

/////////////////////////////////////////////////////////////////////////////////

// http://www.codemiles.com/c-opengl-examples/add-lighting-to-teapot-3d-t9132.html

void initLighting(){
    
    // Enable lighting
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    
    // Set lighting
    glLightfv(GL_LIGHT0, GL_AMBIENT, qaAmbientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, qaDiffuseLight);
    glLightfv(GL_LIGHT0, GL_POSITION, qaLightPosition);
    glLightfv(GL_LIGHT0, GL_SPECULAR, qaSpecularLight);
    
}

/////////////////////////////////////////////////////////////////////////////////

void keyboard( unsigned char key, int x, int y ){
    switch ( key ){
        case 'q':
            exit(0); // quit when q is pressed
            break;
            
        // showing the rectangle covering face
        case 'r':
            mode_wire = !mode_wire; // on&off
            break;
        
        // recording background frame
        case 'b':
            mode_back = true; // on for once
            break;
            
        // displaying the threshold image
        case 'g':
            mode_thres = !mode_thres; // on&off
            break;
        
            
        // switch the detection mode between: background substraction, optical flow
        case 'f':
            mode_detect = !mode_detect; // switches
            break;
       
        // displaying the spinning object
        case 'o':
            mode_dis = !mode_dis; // on&off
            break;
            
        // switching objects: teapot, cone, torus
        case '2':
            mode_obj = 2;
            break;
        case '3':
            mode_obj = 3;
            break;
        case '4':
            mode_obj = 4;
            break;
            
        default:
            break;
    }
}

/////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv )
{
    
    distortionCoefficients =  Mat::zeros(4, 1, CV_64F);
    cameraMatrix =  Mat::eye(3, 3, CV_64F);
    

    imageSize = Size((int) cap.get( CV_CAP_PROP_FRAME_WIDTH ),(int) cap.get( CV_CAP_PROP_FRAME_HEIGHT ));
    
    
    // [0. Load images, convert to gray scale]
    for (int index = 0; index < IMAGE_NUM; index++){
        
        img[index] = imread("/Users/EYE/Desktop/HW2/sampleImages/"+to_string(index+1)+".jpg");
        
        if (!img[index].data){
            cout << "Image data not loaded properly" << endl;
            cin.get();
            return -1;
        }
        
        cvtColor( img[index], gray_image[index], CV_BGR2GRAY );
    }
    
    if(!img.data() ){ // Check for invalid input
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }
    

    //------------------ camera calibration / sample set  ------------------
    
    // [1. obtain 2D locations of chessboard corners]
    
    for(int index = 0; index < IMAGE_NUM; index++){
        patternfound = findChessboardCorners(gray_image[index], patternSize, imageCorners[index],CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        
        if(patternfound)
            cornerSubPix(gray_image[index], imageCorners[index], Size(11, 11), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        
        drawChessboardCorners(img[index], patternSize, Mat(imageCorners[index]), patternfound);
        
        
    // [2. Specify 3D locations for the corners]
        objectCorners[index] = Create3DChessboardCorners(patternSize, squareSize);
        
    }
    
    // [3. Calculate the camera matrices and distortion coefficients.]
    
    int flags = 0;
    imageSize=img[0].size();
    
    rms = calibrateCamera(objectCorners, imageCorners, imageSize, cameraMatrix,distortionCoefficients, rotationVectors, translationVectors, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    
    cout << "RMS: " << rms << endl;
    cout << "Camera matrix: " << cameraMatrix << endl;
    cout << "Distortion _coefficients: " << distortionCoefficients << endl;
    
    // [4. calculate re-projection error]
    
    totalAvgErr = computeReprojectionErrors(objectCorners,imageCorners,rotationVectors, translationVectors, cameraMatrix, distortionCoefficients, reprojErrs);
    cout << "totalAvgErr: " << totalAvgErr << endl;
    
    // [5. Computes useful camera characteristics from the camera matrix ]
    calibrationMatrixValues(cameraMatrix,img[0].size(),0.0f,0.0f,fovx,fovy,focalLength, principalPoint,aspectRatio);
    
    cout << " field of view X: " << fovx << endl;
    cout << " field of view Y: " << fovy << endl;
    cout << " focal length: " << focalLength << endl;
    cout << " principal point: " << principalPoint << endl;
    cout << " aspect ratio: " << aspectRatio << endl;
    
    
    if(!cap.isOpened()){  // check the camera
        cout <<  "Could not open the camera" << endl ;
        return -1;
    }
    
    //------------------ openGL main  ------------------
    
    // initialize GLUT
    glutInit( &argc, &argv[0] );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE |GLUT_DEPTH);
    glutInitWindowPosition( 20, 20 );
    glutInitWindowSize(1280,720 );
    glutCreateWindow( "OpenGL teapot overlay" );
    
    // Enable lighting and color material
    glEnable(GL_COLOR_MATERIAL);
    initLighting();
    
    // set up GUI callback functions
    glutIdleFunc(idle);
    glutKeyboardFunc( keyboard );
    glutDisplayFunc( display );
    
    
    //------------------ load face detection classifier / cascades  ------------------

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)1Error loading eyes cascade\n"); return -1; };
    
    //------------------ initialize idle ------------------
    idle();
    
    //------------------ start GUI loop ------------------
    glutMainLoop();
    
    return 0;
}

