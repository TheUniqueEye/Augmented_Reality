//  Created by Jing Yan(眼睛) on 2016/10/27.
//  Copyright © 2016年 Jing Yan(眼睛). All rights reserved.


/* Assignment 2: AR Magic Mirror
 
 0. Based on the code of Assignment 1.
 1. Build a "magic mirror" application which augments the view from your laptop camera with 3D graphics.
 2. Implement computer vision techniques to recognize gestures from the user for interaction.
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
#include <GLUT/glut.h>



using namespace cv;
using namespace std;


VideoCapture cap(0); // open the default camera

int IMAGE_NUM = 11; // &
vector<Mat> img(IMAGE_NUM);
vector<Mat> gray_image(IMAGE_NUM);

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

Mat frame;
Mat tempframe;
Mat img_temp;
Mat rvec_temp = Mat::eye(3, 1, CV_64F);
Mat tvec_temp = Mat::eye(3, 1, CV_64F);

Mat gray_temp;
Mat img_undistort;

vector<Point2f> imageCorners_temp(1);
vector<Point3f> objCorners_temp(1);

int mode=3;

String face_cascade_name = "/Users/EYE/Desktop/HW2/haarcascade_frontalface_default.xml";
String eyes_cascade_name = "/Users/EYE/Desktop/HW2/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

vector<Rect> faces;


vector<Point3f> Create3DChessboardCorners( Size boardSize, float squareSize);
void detectAndDisplay( Mat frame );

double computeReprojectionErrors( const vector<vector<Point3f> >& objectPoints,
                                 const vector<vector<Point2f> >& imagePoints,
                                 const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                                 const Mat& cameraMatrix , const Mat& distCoeffs,
                                 vector<float>& perViewErrors);


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
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);   // norm - difference
        
        int n = (int)objectPoints[i].size(); // points per frame
        perViewErrors[i] = (float) sqrt(err*err/n);          // save for this view &
        totalErr        += err*err;                               // sum it up &
        totalPoints     += n;
    }
    return sqrt(totalErr/totalPoints); // calculate the arithmetical mean
}



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

// function for face detection
// http://docs.opencv.org/3.0-beta/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html

void detectAndDisplay()
{
    //vector<Rect> faces;
    //Mat gray_temp_flip;
    
    equalizeHist( gray_temp, gray_temp );
    //flip(gray_temp, gray_temp_flip, 1);
    
    //-- Detect faces
    face_cascade.detectMultiScale( gray_temp, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30) );
    
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( img_undistort, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        
    }
    
    //-- Show what you got
    //imshow( "Capture - Face detection", img_undistort );
}


void idle(){
    
    cap >> frame; // get a new frame from camera
    
    if(!frame.data ){
        cout <<  "frame data not loaded properly" << endl ;
        //return -1;
    }
    
    // based on the way cv::Mat stores data, you need to flip it before displaying it
    flip(frame, tempframe, 1); // 1 instead of 0
    
    // undistort realtime video stream using the calibration parameters
    
    img_temp = tempframe;

    
    
    // [undistort the frame image using camera matrix and distirtion coefficients]
    Mat map1, map2;
    initUndistortRectifyMap(cameraMatrix, distortionCoefficients, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, img_temp.size(), 1, img_temp.size(), 0),  img_temp.size(), CV_16SC2, map1, map2);
    
    undistort( img_temp, img_undistort, cameraMatrix, distortionCoefficients);
    remap(img_temp, img_undistort, map1, map2, INTER_LINEAR);
    
    cvtColor( img_undistort, gray_temp, CV_BGR2GRAY );
    
    detectAndDisplay();

}

void display()
{
    
  
    ////////////////////////////// openGL part ///////////////////////////
    
    // clear the window
    glClear( GL_COLOR_BUFFER_BIT );
    
    Mat tempimage;
    
    // show the current undistorted camera frame
    
    //based on the way cv::Mat stores data, you need to flip it before displaying it
    flip(img_undistort, tempimage, 0);
    glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
    

    if(faces.size()!=0){
        
        //////////////////////////////////////////////////////////////////////////////////
        // Here, set up new parameters to render a scene viewed from the camera.
        
        //set viewport
        glViewport(0, 0, tempimage.size().width, tempimage.size().height);
        
        //set projection matrix using intrinsic camera params
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        
        
        
        // glFrustum — multiply the current matrix by a perspective matrix
        // reference: https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
        //http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        
        
        // get Fx,Fy,Cx,Cy from Camera Matrix
        GLdouble fx= cameraMatrix.at<double>(0,0);
        GLdouble fy = cameraMatrix.at<double>(1,1);
        GLdouble cx = cameraMatrix.at<double>(0,2);
        GLdouble cy = cameraMatrix.at<double>(1,2);
        
        GLdouble near = 0.001;
        GLdouble far = 500;
        
        glFrustum(0-cx*(near/fx), // left
                  (img_undistort.size().width-cx)*(near/fx), // right
                  (img_undistort.size().height-cy)*(near/fy), // bottom
                  0-cy*(near/fy), // top
                  near,far);
        
        
        //gluPerspective(fovy, aspectRatio, near, far);
        
        //you will have to set modelview matrix using extrinsic camera params
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        //gluLookAt(0, 0, 15, 0, 0, 0, 0, 1, 0);
        
        
        /////////////////////////////////////////////////////////////////////////////////
      
        
        
        
        
        
        
        /*
        
        /////////////////////////////////////////////////////////////////////////////////
        // draw 3D shapes: Teapot / Spheres
        
        // convert rotation and translation paremeters from Matrix to Vector
        
        vector<float> Translation(3);
        Translation[0] = tvec_temp.at<double>(0,0);
        Translation[1] = tvec_temp.at<double>(0,1);
        Translation[2] = -tvec_temp.at<double>(0,2);
        
        
        Mat rmatrix = Mat::eye(3, 3, CV_64F);
        
        Rodrigues(rvec_temp,rmatrix);
        
        vector<float> Rmatrix(9);
        Rmatrix[0] = -rmatrix.at<double>(0,0);
        Rmatrix[1] = -rmatrix.at<double>(1,0);
        Rmatrix[2] = rmatrix.at<double>(2,0);
        Rmatrix[3] = -rmatrix.at<double>(0,1);
        Rmatrix[4] = -rmatrix.at<double>(1,1);
        Rmatrix[5] = rmatrix.at<double>(2,1);
        Rmatrix[6] = -rmatrix.at<double>(0,2);
        Rmatrix[7] = -rmatrix.at<double>(1,2);
        Rmatrix[8] = rmatrix.at<double>(2,2);
        
        GLfloat matrix[16] = {
            Rmatrix[0],Rmatrix[1],Rmatrix[2],0.0,
            Rmatrix[3],Rmatrix[4],Rmatrix[5],0.0,
            Rmatrix[6],Rmatrix[7],Rmatrix[8],0.0,
            Translation[0],Translation[1],Translation[2],1.0
        };
        
        glMultMatrixf(matrix);
        glPushMatrix();
        
        // draw teapot
        glScaled(-1,-1,1);
        if(mode==1){
            glPushMatrix();
            //glTranslated(4, 3, 0);
            glRotated(90,1.0,0.0,0.0);
            glutWireTeapot(3);
            glPopMatrix();
        }
        
        // draw spheres
        else if(mode==2){
            for(int i = 0; i < patternSize.width; i++ ){
                for(int j = 0; j < patternSize.height; j++ ){
                    glPushMatrix();
                    glTranslated(i,j,0);
                    glutSolidSphere(.3, 100, 100);
                    glPopMatrix();
                }
            }
        }
        
        drawAxes(1.0);
        glPopMatrix();
        */
    }
    
    
    
    
    
    // show the rendering on the screen
    glutSwapBuffers();
    
    // post the next redisplay
    glutPostRedisplay();
}


void keyboard( unsigned char key, int x, int y ){
    switch ( key ){
        case 'q':
            exit(0); // quit when q is pressed
            break;
            
        case '1':
            mode = 1;
            break;
            
        case '2':
            mode = 2;
            break;
            
        case '3':
            mode = 3;
            break;
            
        default:
            break;
    }
}


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
    
    
    
    //////// PART1 camera calibration using test images from the camera ///////
    
    // [1. obtain 2D locations of chessboard corners]
    // reference:http://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga6a10b0bb120c4907e5eabbcd22319022
    
    for(int index = 0; index < IMAGE_NUM; index++){
        patternfound = findChessboardCorners(gray_image[index], patternSize, imageCorners[index],CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        
        if(patternfound)
            cornerSubPix(gray_image[index], imageCorners[index], Size(11, 11), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        
        drawChessboardCorners(img[index], patternSize, Mat(imageCorners[index]), patternfound);
        
        
        // [2. Specify 3D locations for the corners]
        objectCorners[index] = Create3DChessboardCorners(patternSize, squareSize);
        
    }
    
    // [3. Calculate the camera matrices and distortion coefficients.]
    // reference: https://github.com/daviddoria/Examples/blob/master/c%2B%2B/OpenCV/DrawChessboardCorners/DrawChessboardCorners.cxx
    
    int flags = 0;
    imageSize=img[0].size();
    
    rms = calibrateCamera(objectCorners, imageCorners, imageSize, cameraMatrix,distortionCoefficients, rotationVectors, translationVectors, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    
    cout << "RMS: " << rms << endl;
    cout << "Camera matrix: " << cameraMatrix << endl;
    cout << "Distortion _coefficients: " << distortionCoefficients << endl;
    
    // [4. project the 3D checkerboard corners to 2D points for each image,
    // then calculate re-projection error]
    
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
    
     //detectAndDisplay( frame );

    
    
    ////////////////////////////// openGL part ///////////////////////////
    
    // initialize GLUT
    glutInit( &argc, &argv[0] );
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
    glutInitWindowPosition( 20, 20 );
    glutInitWindowSize(1080,720 );
    glutCreateWindow( "OpenGL teapot overlay" );
    
    // set up GUI callback functions
    glutIdleFunc(idle);
    glutDisplayFunc( display );
    glutKeyboardFunc( keyboard );
    
    
    
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)1Error loading eyes cascade\n"); return -1; };
    
    idle();  //init
    
    
    // start GUI loop
    glutMainLoop();
    

    
    return 0;
}

