/*===============================================================================
Name:Zhenyu Yang, Jing Yan, Junxiang Yap
Final Project - AR Navigation on Android

Reference: Vuforia example image target
===============================================================================*/

package com.vuforia.samples.VuforiaSamples.app.ImageTargets;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.Log;

import com.vuforia.Device;
import com.vuforia.Matrix44F;
import com.vuforia.State;
import com.vuforia.Tool;
import com.vuforia.Trackable;
import com.vuforia.TrackableResult;
import com.vuforia.Vuforia;
import com.vuforia.samples.SampleApplication.SampleAppRenderer;
import com.vuforia.samples.SampleApplication.SampleAppRendererControl;
import com.vuforia.samples.SampleApplication.SampleApplicationSession;
import com.vuforia.samples.SampleApplication.utils.Arrow;
import com.vuforia.samples.SampleApplication.utils.Button;
import com.vuforia.samples.SampleApplication.utils.CubeObject;
import com.vuforia.samples.SampleApplication.utils.CubeShaders;
import com.vuforia.samples.SampleApplication.utils.LoadingDialogHandler;
import com.vuforia.samples.SampleApplication.utils.SampleApplication3DModel;
import com.vuforia.samples.SampleApplication.utils.SampleUtils;
import com.vuforia.samples.SampleApplication.utils.Square;
import com.vuforia.samples.SampleApplication.utils.Teapot;
import com.vuforia.samples.SampleApplication.utils.Texture;
import com.vuforia.samples.SampleApplication.utils.WaitingCircle;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.params.BasicHttpParams;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Vector;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import static java.lang.Math.sqrt;

// The renderer class for the ImageTargets sample. 
public class ImageTargetRenderer extends Activity implements GLSurfaceView.Renderer, SampleAppRendererControl,SensorEventListener {


    //+++++++++++++++++++++++++++++  Zhenyu's globals  ++++++++++++++++++++++++++++++++=

    private static final String LOGTAG = "ImageTargetRenderer";

    private SampleApplicationSession vuforiaAppSession;
    private ImageTargets mActivity;
    private SampleAppRenderer mSampleAppRenderer;

    private Vector<Texture> mTextures;

    private int shaderProgramID;
    private int vertexHandle;
    private int textureCoordHandle;
    private int mvpMatrixHandle;
    private int texSampler2DHandle;

    private Teapot mTeapot;



    private float kBuildingScale = 12.0f;
    private SampleApplication3DModel mBuildingsModel;

    private boolean mIsActive = false;
    private boolean mModelIsLoaded = false;

    private static final float OBJECT_SCALE_FLOAT = 1.0f;

    float[] camera_location = new float[3];

int frameCount = 0;

    float[][] teapot_data = new float[4][4];
    float[][] teapot_data_rotation_only = new float[4][16];


    float[] modelViewProjectionMemory = new float[16];

    String place_location_destination = "";
    String place_id_destination = "";
    String place_keyWord_destination = "";

    GLText glText;                             // A GLText Instance

    private SensorManager mSensorManager;
    private float compassRead = 0.0f;


    // for acc sensor
    private float[] gravity = new float[3];
    private float[] linear_acceleration = new float[3];
    private float velocity_Y = 0;
    private float distance_Y = 0;


    private double EPSILON = 8.854e-12;

    //for gyro
    private static final float NS2S = 1.0f / 1000000000.0f;
    private  float[] gyroRotation = new float[3];
    private  float[] gyroRotation_qwt = new float[4];
    private  double[] gyroRotation_offset = new double[3];
    private float timestamp = 0.01f;


    private boolean ifAttached = false;
    private boolean ifPickupReset = false;



    //GPS locations
    float currentLocationX = 34.416095f;
    float currentLocationY = -119.844679f;
    float [] currentLocation = new float [2];
    float[][] locationList = new float[0][3];

    private Context context;


    boolean ifRunshowDirectionInGoogleMapsNextFrame = false;
    boolean isRecording = false;

    boolean ifShowArrow = true;
    boolean orientatedByCompass = false;
    boolean ifShowInformation = true;

    float setp = 5.0f;

    float ratio = 2000000.0f;

    //================================= end of Zhenyu's globals  =====================



    //+++++++++++++++++++++++++++++  Junxiang's globals  ++++++++++++++++++++++++++++++++=


    private int buttonDepth = 300;


    private Square mSquare;
    private Button mButton;
    private WaitingCircle mWaitingCircle;
    private CubeObject mCube;
    private Arrow mArrow;

    int waitingCounterNavigation = 0;
    int waitingCounterInformation = 0;

    //phase 1
    Cursor onScreen = new Cursor(0.0f,0.0f,0.00f,12);
    waitingCircleControl onScreenWaiting = new waitingCircleControl(0.0f,0.0f,-1.01f,4);
    Background bg = new Background(0.01f,5);
    private GLText glTextDestination; // A GLText Instance
    int augmentationStep = 60;
    int augmentationCounter = 0;
    int[] waitingCounterCursor = {0,0,0};
    boolean[] hoverOn = {false,false,false};
    int chosenNumber = 4; // there are only 3 strings in the view

    Cursor microphone = new Cursor(71.0f,120f,0.01f,13);
    int waitingCounterMicrophone = 0;
    boolean onMicrophone = false;
    boolean voiceControlOn = false;


    handMarkerPosition h_pos =  new handMarkerPosition(0,0); // hand position, which is the position of the marker

    boolean stringSelected = false;
    boolean missionStarted = false;


    private String[] destination = {
            "",
            "Please indicate your destination",
            ""
    };


    //phase 2
    // Navigation Button
    ButtonControl navigationOn = new ButtonControl(21.0f,120f,0.01f,0,0);
    ButtonControl navigationOff = new ButtonControl(21.0f,120f,0.01f,1,180);
    waitingCircleControl navigation = new waitingCircleControl(21.0f,120.0f,0.01f,4);

    // Information Button
    ButtonControl informationOn = new ButtonControl(71.0f,120f,0.01f,2,0);
    ButtonControl informationOff = new ButtonControl(71.0f,120f,0.01f,3,180);
    waitingCircleControl information = new waitingCircleControl(71.0f,120.0f,0.01f,4);

    boolean navigationSelected = false; // If the button is chosed to change the status
    boolean informationSelected = false; // If the button is chosed to change the status

    boolean tracking = false; // if still in tracking status
    boolean markerInFrame = false; // if the marker can be seen by the camera
    int outofFrameCounter = 0; // if the marker cannot be seen by the camera anymore, count 10 frames and change the tracking status

    float angleIncreased = 0; // Controlling rotation effect

    private float distNavigation = 0; // Storing the distance between marker and the button of navigation
    private float distInformation = 0; // Storing the distance between marker and the button of information

    boolean show = true;






    //================================= end of Junxiang's globals  =====================





    //+++++++++++++++++++++++++++++  Jing's globals  ++++++++++++++++++++++++++++++++=

    private static final String LOGTAG1 = "ImageTargetRenderer";
    // deal with the modelview and projection matrices
    float[] inverseMV = new float[16];
    float[] modelViewProjection = new float[16];
    float[] mvProjection = new float[16];
    // camera matrix
    float[] camPosition = new float[3];
    float[] camMatrix = new float[16];
    String keyword;
    JSONObject obj;
    JSONArray results;
    JSONArray type;
    String test;
    String pre_keyword = "-1";
    String name;
    String type0;
    String rating;
    String price_level;
    String photo_reference;
    String[][] database = new String[10][2]; // 10 space
    int libraryCounter=0;
    String[] image_set = new String[3];
//    private SampleApplicationSession vuforiaAppSession;
//    private ImageTargets mActivity;
//    private SampleAppRenderer mSampleAppRenderer;



    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // text
//    private Vector<Texture> mTextures;
//    private int shaderProgramID;
//    private int vertexHandle;


    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    private int textureCoordHandle;
//    private int mvpMatrixHandle;
//    private int texSampler2DHandle;
//    private Teapot mTeapot;
    private Plane p;
//    private boolean mIsActive = false;
//    private boolean mModelIsLoaded = false;
//    private float OBJECT_SCALE_FLOAT = 0.5f; // teapot scale
    private GLText glText1; // A GLText Instance
    private GLText glText2; // A GLText Instance

    //================================= end of Jing's globals  =====================



    public ImageTargetRenderer(ImageTargets activity, SampleApplicationSession session,Context contextInput)  throws JSONException {
        mActivity = activity;
        vuforiaAppSession = session;
        context = contextInput;

        // SampleAppRenderer used to encapsulate the use of RenderingPrimitives setting
        // the device mode AR/VR and stereo mode
        mSampleAppRenderer = new SampleAppRenderer(this, mActivity, Device.MODE.MODE_AR, false, 10f, 5000f);
    }


    // Called to draw the current frame.
    @Override
    public void onDrawFrame(GL10 gl) {
        if (!mIsActive)
            return;

        // Call our function to render content from SampleAppRenderer class
        mSampleAppRenderer.render();
    }


    public void setActive(boolean active) {
        mIsActive = active;

        if (mIsActive)
            mSampleAppRenderer.configureVideoBackground();
    }


    // Called when the surface is created or recreated.
    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        Log.d(LOGTAG, "GLRenderer.onSurfaceCreated");

        // Call Vuforia function to (re)initialize rendering after first use
        // or after OpenGL ES context was lost (e.g. after onPause/onResume):
        vuforiaAppSession.onSurfaceCreated();

        mSampleAppRenderer.onSurfaceCreated();



        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^JING^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // Set the background frame color
        GLES20.glClearColor( 0.5f, 0.5f, 0.5f, 1.0f );

        // Create the GLText
        glText1 = new GLText(this.context.getAssets());
        glText2 = new GLText(this.context.getAssets());

        glText1.load( "LucidaGrande.ttc", 55, 2, 2 );  // Create Font (Height: 55 Pixels / X+Y Padding 5 Pixels)
        glText2.load( "LucidaGrande.ttc", 20, 5, 10 );



        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    }


    // Called when the surface changed size.
    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        Log.d(LOGTAG, "GLRenderer.onSurfaceChanged");

        // Call Vuforia function to handle render surface size changes:
        vuforiaAppSession.onSurfaceChanged(width, height);

        // RenderingPrimitives to be updated when some rendering change is done
        mSampleAppRenderer.onConfigurationChanged(mIsActive);

        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^JING^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        try {
            initRendering();
        } catch (JSONException e) {
            e.printStackTrace();
        }
        // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//        initRendering();
    }



    @Override
    public void onSensorChanged(SensorEvent event) {

       // compassRead = event.values[0];
        //Log.i("Sensor", "compassRead = "+compassRead);
        //Log.i(TAG, "compassRead event.values[1]= "+event.values[1]);



        Sensor sensor = event.sensor;  //get the event sensor
        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            //TODO: get values
            //Log.v("blah blah", "TYPE_ACCELEROMETER");


            //pro-processing

            final float alpha = 0.8f;

            // Isolate the force of gravity with the low-pass filter.
            gravity[0] = alpha * gravity[0] + (1 - alpha) * event.values[0];
            gravity[1] = alpha * gravity[1] + (1 - alpha) * event.values[1];
            gravity[2] = alpha * gravity[2] + (1 - alpha) * event.values[2];

            // Remove the gravity contribution with the high-pass filter.
            linear_acceleration[0] = event.values[0] - gravity[0];
            linear_acceleration[1] = event.values[1] - gravity[1];
            linear_acceleration[2] = event.values[2] - gravity[2];

            //end of pre-processing


            //acc0Text.setText("acc0 = " + Float.toString(linear_acceleration[0]) + " units");
            //acc1Text.setText("acc0 = " + Float.toString(linear_acceleration[1]) + " units");
            //acc2Text.setText("acc0 = " + Float.toString(linear_acceleration[2]) + " units");

        } else if (sensor.getType() == Sensor.TYPE_ORIENTATION) {
            //TODO: get values
            //Log.v("blah blah", "TYPE_ORIENTATION3");
            //compassText.setText("Heading: " + Float.toString(event.values[0]) + " degrees");
            compassRead = event.values[0];
        }

        else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {

            //Log.v("blah blah", "TYPE_GYROSCOPE");
            float axisX = 0.0f;
            float axisY = 0.0f;
            float axisZ = 0.0f;

            //pre-processing


            ;
            // This timestep's delta rotation to be multiplied by the current rotation
            // after computing it from the gyro sample data.
            //Log.v("blah blah", "timestamp = " + Double.toString(timestamp) + " units");
            if (timestamp != 0) {
                final float dT = (event.timestamp - timestamp) * NS2S;
                // Axis of the rotation sample, not normalized yet.
                axisX = event.values[0];
                axisY = event.values[1];
                axisZ = event.values[2];

                // Calculate the angular speed of the sample
                double omegaMagnitude = sqrt(axisX * axisX + axisY*axisY + axisZ*axisZ);

                // Normalize the rotation vector if it's big enough to get the axis
                // (that is, EPSILON should represent your maximum allowable margin of error)
                //Log.v("blah blah", "omegaMagnitude = " + Double.toString(omegaMagnitude) + " units");
                //Log.v("blah blah", "axisX = " + Double.toString(axisX) + " units");
                if (omegaMagnitude > EPSILON) {
                    axisX /= omegaMagnitude;
                    axisY /= omegaMagnitude;
                    axisZ /= omegaMagnitude;
                }


                // Integrate around this axis with the angular speed by the timestep
                // in order to get a delta rotation from this sample over the timestep
                // We will convert this axis-angle representation of the delta rotation
                // into a quaternion before turning it into the rotation matrix.
                //Log.v("blah blah", "omegaMagnitude = " + Double.toString(omegaMagnitude) + " units");
                double thetaOverTwo = omegaMagnitude * dT / 2.0f;
                //double thetaOverTwo += omegaMagnitude * dT / 2.0f;
                //Log.v("blah blah", "thetaOverTwo = " + Double.toString(thetaOverTwo) + " units");
                double sinThetaOverTwo = Math.sin(thetaOverTwo);
                double cosThetaOverTwo = Math.cos(thetaOverTwo);
                //gyroRotation_qwt[0] += sinThetaOverTwo * axisX;
                //gyroRotation_qwt[1] += sinThetaOverTwo * axisY;
                //gyroRotation_qwt[2] += sinThetaOverTwo * axisZ;
                //gyroRotation_qwt[3] = (float)(cosThetaOverTwo);

                //gyroRotation = qwttoAngles(gyroRotation_qwt[0],gyroRotation_qwt[1],gyroRotation_qwt[2],gyroRotation_qwt[3]);

                gyroRotation[0] += thetaOverTwo * axisX*2.0f;
                gyroRotation[1] += thetaOverTwo * axisY*2.0f;
                gyroRotation[2] += thetaOverTwo * axisZ*2.0f;






                timestamp = event.timestamp;
                // User code should concatenate the delta rotation we computed with the current rotation
                // in order to get the updated rotation.
                // rotationCurrent = rotationCurrent * deltaRotationMatrix;
            }
            //end of pre-processing
            // Log.v("blah blah", "compassRead = "+compassRead);
            //Log.v("blah blah", "linear_acceleration[0] = "+linear_acceleration[0]+" linear_acceleration[1] = "+linear_acceleration[1]+" linear_acceleration[2] = "+linear_acceleration[2]);
            //Log.v("blah blah", "gyroRotation[0] = "+gyroRotation[0]+" gyroRotation[1] = "+gyroRotation[1]+" gyroRotation[2] = "+gyroRotation[2]);
            //TODO: get values
            //Log.v("blah blah", "TYPE_ORIENTATION3");
        }



    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // not in use
    }








    // Function for initializing the renderer.
    private void initRendering()  throws JSONException{
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, Vuforia.requiresAlpha() ? 0.0f
                : 1.0f);

        for (Texture t : mTextures) {
           // Log.d(LOGTAG, "t = "+t.toString());
            GLES20.glGenTextures(1, t.mTextureID, 0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, t.mTextureID[0]);
            GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                    GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameterf(GLES20.GL_TEXTURE_2D,
                    GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
                    t.mWidth, t.mHeight, 0, GLES20.GL_RGBA,
                    GLES20.GL_UNSIGNED_BYTE, t.mData);
        }

        shaderProgramID = SampleUtils.createProgramFromShaderSrc(
                CubeShaders.CUBE_MESH_VERTEX_SHADER,
                CubeShaders.CUBE_MESH_FRAGMENT_SHADER);

        vertexHandle = GLES20.glGetAttribLocation(shaderProgramID,
                "vertexPosition");
        textureCoordHandle = GLES20.glGetAttribLocation(shaderProgramID,
                "vertexTexCoord");
        mvpMatrixHandle = GLES20.glGetUniformLocation(shaderProgramID,
                "modelViewProjectionMatrix");
        texSampler2DHandle = GLES20.glGetUniformLocation(shaderProgramID,
                "texSampler2D");

        if (!mModelIsLoaded) {
            mTeapot = new Teapot();

//+++++   Junxiang's part  +++++++++
            mSquare = new Square();
            mArrow = new Arrow();
            mButton = new Button();
            mWaitingCircle = new WaitingCircle();
            mCube = new CubeObject();

            mButton.initData();
            mWaitingCircle.initData();


            navigationOn.initRotate();
            navigationOff.initRotate();

            informationOn.initRotate();
            informationOff.initRotate();


            // Create the GLText
            glTextDestination = new GLText(this.context.getAssets());
            Log.d("glText1", "load!!");
            glTextDestination.load( "HelveticaNeue.dfont", 40, 2, 2 );  // Create Font (Height: 14 Pixels / X+Y Padding 2 Pixels)

            //========  end of Junxiang's part  =============

            try {
                mBuildingsModel = new SampleApplication3DModel();
                mBuildingsModel.loadModel(mActivity.getResources().getAssets(),
                        "ImageTargets/Buildings.txt");
                mModelIsLoaded = true;
            } catch (IOException e) {
                Log.e(LOGTAG, "Unable to load buildings");
            }

            // Hide the Loading Dialog
            mActivity.loadingDialogHandler
                    .sendEmptyMessage(LoadingDialogHandler.HIDE_LOADING_DIALOG);
        }

        image_set[0] = "Habit";

        image_set[1] = "PandaExpress";

        image_set[2] = "Starbucks";

        //initializing teapots

        float gap = 30f;
        float yOffSet = 55f;

        teapot_data[0][0] = -3f * gap;  //x
        teapot_data[0][1] = yOffSet;  //y
        teapot_data[0][2] = 0f;  //z
        teapot_data[0][3] = 0f;  //if attached
        teapot_data_rotation_only[0] = makePureTranslationM(0, 0, 0);

        teapot_data[1][0] = -1f * gap;  //x
        teapot_data[1][1] = yOffSet;  //y
        teapot_data[1][2] = 0f;  //z
        teapot_data[1][3] = 0f;  //if attached
        teapot_data_rotation_only[1] = makePureTranslationM(0, 0, 0);

        teapot_data[2][0] = 1f * gap;  //x
        teapot_data[2][1] = yOffSet;  //y
        teapot_data[2][2] = 0f;  //z
        teapot_data[2][3] = 0f;  //if attached
        teapot_data_rotation_only[2] = makePureTranslationM(0, 0, 0);

        teapot_data[3][0] = 3f * gap;  //x
        teapot_data[3][1] = yOffSet;  //y
        teapot_data[3][2] = 0f;  //z
        teapot_data[3][3] = 0f;  //if attached
        teapot_data_rotation_only[3] = makePureTranslationM(0, 0, 0);

        modelViewProjectionMemory = makePureTranslationM(0, 0, 0);
        //end of initializing teapots



        //sensors
        mSensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_ORIENTATION), SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_GAME);
        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_GAME);

        //glText
        glText = new GLText(context.getAssets());
        // Load the font from file (set size + padding), creates the texture
        // NOTE: after a successful call to this the font is ready for rendering!
        glText.load("HelveticaNeue.dfont", 28, 1, 1);  // Create Font (Height: 14 Pixels / X+Y Padding 2 Pixels)



        //GPS locations
        currentLocation[0] = currentLocationX;
        currentLocation[1] = currentLocationY;


        //place_location_destination = getPlaceLocationFromName(currentLocation, place_keyWord_destination);
        //Log.i("JSON", "place_location_destination in ini");

        //String place_id_destination = getPlaceIdFromName(currentLocation,place_keyWord_destination);

       // locationList =  getOverviewLocationFromUrl("https://maps.googleapis.com/maps/api/directions/json?origin="+currentLocation[0]+","+currentLocation[1]+"&destination=place_id:"+place_id_destination+"&key=AIzaSyB2g2I8EjT0Uy_WCn1etvvyd_f629DoLRc");

         for(int i =0;i<locationList.length;i++){
            Log.i("JSON", "points["+i+"]  =  " + locationList[i][0]+", "+locationList[i][1]);
        }
        //test zone









/*
        // Acquire a reference to the system Location Manager
        LocationManager locationManager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);

        // Define a listener that responds to location updates
        LocationListener locationListener = new LocationListener() {
            public void onLocationChanged(Location location) {
                // Called when a new location is found by the network location provider.
                Log.i("JSON", "location  =  " + location.toString());
            }

            public void onStatusChanged(String provider, int status, Bundle extras) {}

            public void onProviderEnabled(String provider) {}

            public void onProviderDisabled(String provider) {}
        };

        // Register the listener with the Location Manager to receive location updates
        locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 0, 0, locationListener);
*/


        //end of test zone
    }




//*******************************JING_START****************************************



    // -----------------------------------

    // Construct a Transformation Matrix
    public float[] maketMatrix(float x, float y, float z)
    {
        float[] tMatrix = new float[16];
        for(int i=0; i<12; i++) {
            tMatrix[i] = 0;
        }
        for(int i=0; i<4; i++){
            tMatrix[i*5] = 1;
        }
        tMatrix[12]=x; // Translation x,y,z
        tMatrix[13]=y;
        tMatrix[14]=z;
        return tMatrix;
    }
    //-----------------------------------

    //-----------------------------------
//*******************************JING_END****************************************

    public void updateConfiguration() {
        mSampleAppRenderer.onConfigurationChanged(mIsActive);
    }

    public float[] updateConfiguration2() {
        mSampleAppRenderer.onConfigurationChanged(mIsActive);
        return camera_location;
    }

    public void renderFrame(State state, float[] projectionMatrix) {
        //place_keyWord_destination = "UCSB Elings hall";
        //frameCount++;
        //Log.i("JSON", "frameCount = "+frameCount);
        //Log.i("JSON", "ifShowArrow = "+ifShowArrow);
        if(ifRunshowDirectionInGoogleMapsNextFrame) {
            //place_keyWord_destination = "UCSB unversity center";
            place_location_destination = getPlaceLocationFromName(currentLocation, place_keyWord_destination);
            showDirectionInGoogleMaps();

            Log.i("JSON", "place_location_destination in renderFrame");
        }
        //++++++++++++ interaction controll +++++++++++
        //voiceControlOn

        if(voiceControlOn) {
            if(isRecording==false) {
                voiceRecIsCalled();
                //isRecording = true;
                //Log.i("str", "isRecording is set to true");
            }
            //else{
             //   voiceControlOn = false;
            //}
        }

       // Log.d("voiceControlOn", Boolean.toString(voiceControlOn));
        //Log.d("isRecording", Boolean.toString(isRecording));
       //============== interaction controll ===========


        //+++++   Junxiang's part  +++++++++
        distNavigation = 10000;
        distInformation = 10000;
        //========  end of Junxiang's part  =============


        // Renders video background replacing Renderer.DrawVideoBackground()
        mSampleAppRenderer.renderVideoBackground();

        GLES20.glEnable(GLES20.GL_DEPTH_TEST);

        // handle face culling, we need to detect if we are using reflection
        // to determine the direction of the culling
        GLES20.glEnable(GLES20.GL_CULL_FACE);
        GLES20.glCullFace(GLES20.GL_BACK);


        //+++++   Junxiang's part  +++++++++
        markerInFrame = false;


        //************************************************
        // NON_Marker Part
        //************************************************
        /*
        PHASE I
        TEXT SELECTION
         */

        // ^^^^^^^^^^^^^^^^^^^^^ rendering text ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if(!missionStarted){
            // enable texture + alpha blending
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA); // enable aptha
            Matrix.multiplyMM(bg.matrix4button, 0, projectionMatrix, 0, bg.mvMatrix, 0);
            Matrix.multiplyMM(bg.matrix4button, 0, projectionMatrix, 0, bg.mvMatrix, 0);
            Matrix.scaleM(bg.matrix4button, 0, 12*2,
                    19.2f*2, OBJECT_SCALE_FLOAT);
            glDrawSquare(bg.textureID+augmentationCounter,bg.matrix4button);
            GLES20.glDisable(GLES20.GL_BLEND);

            // ^^^^^^^^^^^^^^^^^^^^^ Microphone ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA); // enable aptha
            GLES20.glDisable(GLES20.GL_DEPTH_TEST); // disable depth test

            if(onMicrophone){
                Matrix.scaleM(microphone.mvMatrix, 0, 1.1f, 1.1f ,1.0f);
            }
            Matrix.multiplyMM(microphone.matrix4button, 0, projectionMatrix, 0, microphone.mvMatrix, 0);
            if(voiceControlOn){
                glDrawButton(microphone.textureID,microphone.matrix4button);
            }else{
                glDrawButton(microphone.textureID+1,microphone.matrix4button);
            }
            if(onMicrophone){
                Matrix.scaleM(microphone.mvMatrix, 0, 1/1.1f, 1/1.1f ,1.0f);
            }
            //Log.d("MICPHONE", Boolean.toString(onMicrophone));
            //Log.d("voiceControlOn", Boolean.toString(voiceControlOn));
            GLES20.glDisable(GLES20.GL_BLEND);

            GLES20.glEnable(GLES20.GL_DEPTH_TEST); // disable depth test

            // ^^^^^^^^^^^^^^^^^^^^^ rendering text ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA); // enable aptha

            GLES20.glDisable(GLES20.GL_DEPTH_TEST); // disable depth test

            // TEST: render the entire font texture
            //glText.drawTexture( 100/2, 100/2, modelViewProjection);

            // TEST: render some strings with the font
            float[] mvMatrix = new float[16];
            float[] mvp = new float[16];

            mvMatrix[0] = 0;
            mvMatrix[1] = -1;
            mvMatrix[2] = 0;
            mvMatrix[4] = -1;
            mvMatrix[5] = 0;
            mvMatrix[6] = 0;
            mvMatrix[8] = 0;
            mvMatrix[9] = 0;
            mvMatrix[10] = -1;
            mvMatrix[12] = 0;
            mvMatrix[13] = 0;
            mvMatrix[14] = 300f - augmentationStep * augmentationCounter;
            mvMatrix[15] = 1;
            Matrix.rotateM(mvMatrix, 0, -90, 0, 0, 1.0f);
            Matrix.scaleM(mvMatrix, 0, 0.3f, 0.3f ,1.0f);


            // Load the font from file (set size + padding), creates the texture

            //First String
            Matrix.translateM(mvMatrix, 0, 0f, -20f ,0f);
            if(hoverOn[0]){
                Matrix.scaleM(mvMatrix, 0, 1.1f, 1.1f ,1.0f);
                Matrix.translateM(mvMatrix, 0, 0f, -10f ,0f);
            }
            Matrix.multiplyMM(mvp, 0, projectionMatrix, 0, mvMatrix, 0);
            glTextDestination.begin( 1.0f, 1.0f, 1.0f, 1.0f, mvp ); // Begin Text Rendering (Set Color WHITE)
            glTextDestination.drawC(destination[0], 0f, 100f, 0f);
            glTextDestination.end();
            if(hoverOn[0]){
                Matrix.translateM(mvMatrix, 0, 0f, 10f ,0f);
                Matrix.scaleM(mvMatrix, 0, 1/1.1f, 1/1.1f ,1.0f);
            }


            //Second String
            Matrix.translateM(mvMatrix, 0, 0f, -80f ,0f);
            if(hoverOn[1]){
                Matrix.scaleM(mvMatrix, 0, 1.1f, 1.1f ,1.0f);
                Matrix.translateM(mvMatrix, 0, 0f, -10f ,0f);
            }
            Matrix.multiplyMM(mvp, 0, projectionMatrix, 0, mvMatrix, 0);
            glTextDestination.begin( 1.0f, 1.0f, 1.0f, 1.0f, mvp ); // Begin Text Rendering (Set Color WHITE)
            glTextDestination.drawC(destination[1], 0f, 100f, 0f);
            glTextDestination.end();
            if(hoverOn[1]){
                Matrix.translateM(mvMatrix, 0, 0f, 10f ,0f);
                Matrix.scaleM(mvMatrix, 0, 1/1.1f, 1/1.1f ,1.0f);
            }


            //Third String
            Matrix.translateM(mvMatrix, 0, 0f, -80f ,0f);
            if(hoverOn[2]){
                Matrix.scaleM(mvMatrix, 0, 1.1f, 1.1f ,1.0f);
                Matrix.translateM(mvMatrix, 0, 0f, -10f ,0f);
            }
            Matrix.multiplyMM(mvp, 0, projectionMatrix, 0, mvMatrix, 0);
            glTextDestination.begin( 1.0f, 1.0f, 1.0f, 1.0f, mvp ); // Begin Text Rendering (Set Color WHITE)
            glTextDestination.drawC(destination[2], 0f, 100f, 0f);
            glTextDestination.end();
            if(hoverOn[2]){
                Matrix.translateM(mvMatrix, 0, 0f, 10f ,0f);
                Matrix.scaleM(mvMatrix, 0, 1/1.1f, 1/1.1f ,1.0f);
            }

            if(stringSelected){
                if(augmentationCounter < 6){
                    augmentationCounter++;
                }else{
                    missionStarted = true;
                }
            }

            GLES20.glDisable(GLES20.GL_BLEND);
            GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        }

        //************************************************
        // Marker Part
        //************************************************
        // Did we find any trackables this frame?
        for (int tIdx = 0; tIdx < state.getNumTrackableResults(); tIdx++) {
            TrackableResult result = state.getTrackableResult(tIdx);
            Trackable trackable = result.getTrackable();
            printUserData(trackable);


//****************************************JING_START****************************************
            if(!trackable.getName().equalsIgnoreCase("stones")&&ifShowInformation&&missionStarted){
                printUserData(trackable);

//            String userData = (String) trackable.getUserData();

                Matrix44F modelViewMatrix_Vuforia = Tool
                        .convertPose2GLMatrix(result.getPose());
                float[] modelViewMatrix = modelViewMatrix_Vuforia.getData();

//            if(userData == "stones"){
//                modelViewMatrix = modelViewMatrix_Vuforia.getData();
//            }

//                int textureIndex = trackable.getName().equalsIgnoreCase("stones") ? 0
//                        : 1;
//                textureIndex = trackable.getName().equalsIgnoreCase("tarmac") ? 2
//                        : textureIndex;


                if (!mActivity.isExtendedTrackingActive()) {

                    // for drawing teapot

                    Matrix.translateM(modelViewMatrix, 0, 0.0f, 0.0f,
                            OBJECT_SCALE_FLOAT);
                    Matrix.scaleM(modelViewMatrix, 0, OBJECT_SCALE_FLOAT,
                            OBJECT_SCALE_FLOAT, OBJECT_SCALE_FLOAT);

                }

                Matrix.multiplyMM(modelViewProjection, 0, projectionMatrix, 0, modelViewMatrix, 0);

                //





// ^^^^^^^^^^^^^^^^^^^ get Json File from Google place API ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                // store info:
                if (keyword != null) {
                    Log.d(LOGTAG, "keyword =" + keyword);


                    double[] location = {34.4126719, -119.8481672};
                    float radius = 500;

                    String tokenKey = "AIzaSyAmXO64yIHzwcdU4fqqAHwBdeh-jzoO0iw";

                    boolean ifFound = false;
                    String currentTest = null;

                    for (int i = 0; i < 10; i++) {
                        if (keyword == database[i][0]) {
                            currentTest = database[i][1];
                            ifFound = true;
                            break;
                        }
                    }

                    if (ifFound == false) {
                        // getJson: get store info from Google Place API
                        test = getJSON("https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=" + location[0] + "," + location[1] + "&radius=" + radius + "&types=food&name=" + keyword + "&key=" + tokenKey);
                        Log.d(LOGTAG, "getJSON downloading");
                        database[libraryCounter + 1][1] = test;
                        database[libraryCounter + 1][0] = keyword;
                        libraryCounter++;
                        currentTest = test;
                    }


                    try {
                        obj = new JSONObject(currentTest);
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }


                    try {
                        results = obj.getJSONArray("results");
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                    try {
                        name = results.getJSONObject(0).getString("name");
                        Log.d(LOGTAG, "name = " + name.toString());
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }


                    try {
                        type = results.getJSONObject(0).getJSONArray("types");
                        type0 = type.getString(0);
                        Log.d(LOGTAG, "type0 = " + type0.toString());
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }


                    try {
                        rating = results.getJSONObject(0).getString("rating");
                        Log.d(LOGTAG, "rating = " + rating.toString());
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                    try {
                        price_level = results.getJSONObject(0).getString("price_level");
                        Log.d(LOGTAG, "price_level = " + price_level.toString());
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                    try {
                        photo_reference = results.getJSONObject(0).getJSONArray("photos").getJSONObject(0).getString("photo_reference");
                        String image = "https://maps.googleapis.com/maps/api/place/photo?maxwidth=70&photoreference=" + photo_reference + "&sensor=true&key=" + tokenKey;
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }


                    pre_keyword = keyword;
                    //Log.d(LOGTAG, "pre_keyword =" + keyword);

                }




                // ^^^^^^^^^^^^^^^^^^^^^ rendering Plane Background ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // enable texture + alpha blending

                GLES20.glEnable(GLES20.GL_BLEND);
                GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA); // enable aptha
                GLES20.glEnable(GLES20.GL_DEPTH_TEST);


                p = new Plane();

                GLES20.glUseProgram(shaderProgramID); //

                GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                        false, 0, p.getVertices());
                GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                        GLES20.GL_FLOAT, false, 0, p.getTexCoords());

                GLES20.glEnableVertexAttribArray(vertexHandle);
                GLES20.glEnableVertexAttribArray(textureCoordHandle);

                // activate texture 0, bind it, and pass to shader
                GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                        mTextures.get(15).mTextureID[0]);
                GLES20.glUniform1i(texSampler2DHandle, 0);

                float planeScale=1.0f;
                if(name!=null) {
                    planeScale = name.length()/16.0f;
                    Log.d(LOGTAG, "name =" + name);
                    Log.d(LOGTAG, "planeScale =" + planeScale);

                }

                Matrix.scaleM(modelViewProjection, 0, planeScale, 1, 1);

                float[] pMatrix = new float[16];
                Matrix.multiplyMM(pMatrix, 0, modelViewProjection, 0, maketMatrix(0, 0, 0), 0);


                // finally draw the teapot

                // pass the model view matrix to the shader
                GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                        modelViewProjection, 0);


                GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                        p.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                        p.getIndices());
                Matrix.scaleM(modelViewProjection, 0, 1 / planeScale,
                        1, 1);

                // disable the enabled arrays
                GLES20.glDisableVertexAttribArray(vertexHandle);
                GLES20.glDisableVertexAttribArray(textureCoordHandle);




                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                // ^^^^^^^^^^^^^^^^^^^^^ rendering Plane Background ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                GLES20.glDisable(GLES20.GL_BLEND);

                for(int i=0; i<3; i++) {
                    Log.d(LOGTAG, "keyword =" + keyword);
                    Log.d(LOGTAG, "image_set[i] =" + image_set[i]);

                    if (keyword.equals( image_set[i])) {


                        Log.d(LOGTAG, "draw image i =" + i);

                        for(int j=0; j<3; j++) {

                            Plane p2 = new Plane();

                            GLES20.glUseProgram(shaderProgramID); //

                            GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                                    false, 0, p2.getVertices());
                            GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                                    GLES20.GL_FLOAT, false, 0, p2.getTexCoords());

                            GLES20.glEnableVertexAttribArray(vertexHandle);
                            GLES20.glEnableVertexAttribArray(textureCoordHandle);

                            // activate texture 0, bind it, and pass to shader


                            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
                            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                                    mTextures.get(20+i*3+j).mTextureID[0]);
                            GLES20.glUniform1i(texSampler2DHandle, 0);

                            float planeScale2 = 0.3f;



                            Matrix.scaleM(modelViewProjection, 0, planeScale2, planeScale2, planeScale2);

                            float[] pMatrix2 = new float[16];
                            Matrix.multiplyMM(pMatrix2, 0, modelViewProjection, 0, maketMatrix(60.0f*planeScale+50.0f, -35.0f+j*35.0f, 0.1f), 0);


                            // finally draw the teapot

                            // pass the model view matrix to the shader
                            GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                                    pMatrix2, 0);


                            GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                                    p2.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                                    p2.getIndices());
                            Matrix.scaleM(modelViewProjection, 0, 1 / planeScale2,
                                    1 / planeScale2, 1 / planeScale2);

                            // disable the enabled arrays
                            GLES20.glDisableVertexAttribArray(vertexHandle);
                            GLES20.glDisableVertexAttribArray(textureCoordHandle);
                        }
                    }
                }
                GLES20.glEnable(GLES20.GL_BLEND);

                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                // ^^^^^^^^^^^^^^^^^^^^^ rendering text ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                GLES20.glDisable(GLES20.GL_DEPTH_TEST); // disable depth test

                if (keyword != null) {


                    float fontScale = 0.1f;

                    Matrix.scaleM(modelViewProjection, 0, fontScale,
                            fontScale, fontScale);

                    // Store Logo
                    if(name!=null) {
                        glText1.begin(1.0f, 1.0f, 1.0f, 1.0f, modelViewProjection); // Begin Text Rendering (Set Color WHITE)
                        glText1.drawC(name, 0f, 70f, 0f);
                        glText1.end();
                    }

                    // Store Info
                    if(type!=null) {
                        glText2.begin(1.0f, 1.0f, 1.0f, 1.0f, modelViewProjection);
                        glText2.drawC(type0, 0f, -10f, 0f);
                        glText2.drawC("rating  " + rating, 0f, -40f, 0f);
                        glText2.drawC("price level " + price_level, 0f, -70f, 0f);
                        //glText.drawC(photo_reference, 0f, 0f, 0f)； // photo reference
                        glText2.end();
                    }

                    Matrix.scaleM(modelViewProjection, 0, 1/fontScale,
                            1/fontScale, 1/fontScale);

                    GLES20.glDisable(GLES20.GL_BLEND);
                    GLES20.glEnable(GLES20.GL_DEPTH_TEST);

                    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                }
                Matrix.invertM(inverseMV,0,modelViewMatrix,0);

                // compute camera position in the world coordinate :
                camPosition[0]=inverseMV[12];
                camPosition[1]=inverseMV[13];
                camPosition[2]=inverseMV[14];

            }
//****************************************JING_END****************************************
            else if(trackable.getName().equalsIgnoreCase("stones")){
                Matrix44F modelViewMatrix_Vuforia = Tool
                        .convertPose2GLMatrix(result.getPose());
                float[] modelViewMatrix = modelViewMatrix_Vuforia.getData();
                float[] modelViewMatrixOrigin = modelViewMatrix.clone();
                float[] cameracoord4calc = new float[4];

                markerInFrame = true;

                cameracoord4calc = world2camera(modelViewMatrix,h_pos.x,h_pos.y,h_pos.z);
                h_pos.cameraX = cameracoord4calc[0];
                h_pos.cameraY = cameracoord4calc[1];
                h_pos.cameraZ = cameracoord4calc[2];


                //**************************************** Phase I **************************************************
                if(!missionStarted){
                    onScreen.mvMatrix[12] = buttonDepth*h_pos.cameraX/h_pos.cameraZ;
                    onScreen.mvMatrix[13] = buttonDepth*h_pos.cameraY/h_pos.cameraZ;
                    Matrix.multiplyMM(onScreen.matrix4button, 0, projectionMatrix, 0, onScreen.mvMatrix, 0);
                    Matrix.scaleM(onScreen.matrix4button, 0, 3.0f/8.0f,
                            3.0f/8.0f, OBJECT_SCALE_FLOAT);
                    GLES20.glEnable(GLES20.GL_BLEND);
                    GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA); // enable aptha
                    if(stringSelected == false) {
                        glDrawButton(onScreen.textureID, onScreen.matrix4button);
                    }
                    GLES20.glDisable(GLES20.GL_BLEND);

                    float cursorX = onScreen.mvMatrix[12];
                    float cursorY = onScreen.mvMatrix[13];
                    if(stringSelected == false){
                        if(cursorX < 50 && cursorX > -50){
                            if(cursorY < -15 && cursorY > -35){
                                if(destination[0] != "" && !voiceControlOn){
                                    waitingCounterCursor[0] +=3;
                                    glDrawWaiting(onScreenWaiting.textureID,onScreen.matrix4button,waitingCounterCursor[0]);
                                    waitingCounterCursor[1] = 0;
                                    waitingCounterCursor[2] = 0;
                                    hoverOn[0] = true;
                                    hoverOn[1] = false;
                                    hoverOn[2] = false;
                                }
                            }else if(cursorY < 10 && cursorY > -10){
                                if(destination[1] != "" && !voiceControlOn && destination[1] != "Please indicate your destination") {
                                    waitingCounterCursor[1] +=3;
                                    glDrawWaiting(onScreenWaiting.textureID,onScreen.matrix4button,waitingCounterCursor[1]);
                                    waitingCounterCursor[0] = 0;
                                    waitingCounterCursor[2] = 0;
                                    hoverOn[0] = false;
                                    hoverOn[1] = true;
                                    hoverOn[2] = false;
                                }
                            }else if(cursorY < 35 && cursorY > 15){
                                if(destination[2] != "" && !voiceControlOn){                       waitingCounterCursor[2] +=3;
                                    glDrawWaiting(onScreenWaiting.textureID,onScreen.matrix4button,waitingCounterCursor[2]);
                                    waitingCounterCursor[0] = 0;
                                    waitingCounterCursor[1] = 0;
                                    hoverOn[0] = false;
                                    hoverOn[1] = false;
                                    hoverOn[2] = true;
                                }
//                        waitingCounterCursor[2] +=3;
//                        glDrawWaiting(onScreenWaiting.textureID,onScreen.matrix4button,waitingCounterCursor[2]);
                            }
                        }else{
                            for(int i = 0; i < 3; ++i){
                                hoverOn[i] = false;
                                waitingCounterCursor[i] = 0;
                            }
                        }


                        if(cursorX > 105 && cursorX < 135 && cursorY > 51 && cursorY < 91){
                            onMicrophone = true;
                            waitingCounterMicrophone +=3;
                            glDrawWaiting(onScreenWaiting.textureID,onScreen.matrix4button,waitingCounterMicrophone);
                        }else{
                            onMicrophone = false;
                            waitingCounterMicrophone = 0;
                        }

                        if(waitingCounterMicrophone >= mWaitingCircle.getNumObjectIndex()){
                            waitingCounterMicrophone = 0;
                            voiceControlOn = true;
                        }
                        //  Log.d("h0:",Boolean.toString(hoverOn[0]));
                        //  Log.d("h1:",Boolean.toString(hoverOn[1]));
                        // Log.d("h2:",Boolean.toString(hoverOn[2]));

                        for(int i = 0; i < 3; ++i){
                            if(waitingCounterCursor[i] >= mWaitingCircle.getNumObjectIndex()){
                                waitingCounterCursor[i] = 0;
                                stringSelected = true;
                                chosenNumber = i;
                            }
                        }
                    }
                }


                if(place_keyWord_destination!=null&&chosenNumber!=4){
                    Log.d("dist:ON", "chosen");
                    place_keyWord_destination = destination[chosenNumber];

                    place_id_destination = getPlaceIdFromName(currentLocation,place_keyWord_destination);

                    locationList =  getOverviewLocationFromUrl("https://maps.googleapis.com/maps/api/directions/json?origin="+currentLocation[0]+","+currentLocation[1]+"&destination=place_id:"+place_id_destination+"&key=AIzaSyB2g2I8EjT0Uy_WCn1etvvyd_f629DoLRc");



                    chosenNumber = 4;
                }
                //Log.d("dist:ON", "chosenNumber = "+chosenNumber);
                //Log.d("dist:ON", "place_keyWord_destination = "+place_keyWord_destination);
                //*****************************************************************************************************


                //**************************************** Phase II **************************************************
                if(missionStarted){
                    //Calculate the distance between navigation button and marker
                    float distNavigationOn = (float) Math.sqrt(Math.pow(h_pos.cameraX+navigationOn.y, 2)
                            + Math.pow(h_pos.cameraY-navigationOn.x, 2)
                            + Math.pow(h_pos.cameraZ-navigationOn.z, 2));
                    float distNavigationOff = (float) Math.sqrt(Math.pow(h_pos.cameraX-navigationOff.y, 2)
                            + Math.pow(h_pos.cameraY-navigationOff.x, 2)
                            + Math.pow(h_pos.cameraZ-navigationOff.z, 2));
                    // Log.d("dist:ON", Float.toString(distNavigationOn));
                    // Log.d("dist:OFF", Float.toString(distNavigationOff));
                    if(distNavigationOn >= distNavigationOff){
                        distNavigation = distNavigationOff;
                    }else{
                        distNavigation = distNavigationOn;
                    }


                    //Calculate the distance between information button and marker
                    float distInformationOn = (float) Math.sqrt(Math.pow(h_pos.cameraX+informationOn.y, 2)
                            + Math.pow(h_pos.cameraY-informationOn.x, 2)
                            + Math.pow(h_pos.cameraZ-informationOn.z, 2));
                    float distInformationOff = (float) Math.sqrt(Math.pow(h_pos.cameraX-informationOff.y, 2)
                            + Math.pow(h_pos.cameraY-informationOff.x, 2)
                            + Math.pow(h_pos.cameraZ-informationOff.z, 2));
                    if(distInformationOn >= distInformationOff){
                        distInformation = distInformationOff;
                    }else{
                        distInformation = distInformationOn;
                    }

                    // In case the distance between marker & information is the same as the distance between the marker & navigation
                    if(distInformation < distNavigation){
                        distNavigation = 10000;
                    }else{
                        distInformation = 10000;
                    }
//                    if (dist < 60.0f && (System.currentTimeMillis() - startTime) > 2000) {
//                        startTime = System.currentTimeMillis();
//                        captured = true;
//                        if (chosenNum == 4) {
//                            chosenNum = i;
//                        } else {
//                            if (dist[i] < dist[chosenNum]) {
//                                chosenNum = i;
//                            }
//                        }
//                    }
//
//            } else {
//                dist[chosenNum] = modelViewMatrix[14];
//
//                if(t_pos[chosenNum].z <= OBJECT_SCALE_FLOAT && (System.currentTimeMillis() - startTime) > 2000){
//                    captured = false;
//                    chosenNum = 4;
//                    startTime = System.currentTimeMillis();
//                }
//            }
                    modelViewMatrix = modelViewMatrixOrigin.clone();
                    int textureIndex = 12;
//                int textureIndex = trackable.getName().equalsIgnoreCase("stones") ? 0
//                        : 1;
//                textureIndex = trackable.getName().equalsIgnoreCase("tarmac") ? 2
//                        : textureIndex;

                    // deal with the modelview and projection matrices
                    float[] modelViewProjection = new float[16];
                    float[] modelViewCamera = new float[16];
                    Matrix.multiplyMM(modelViewCamera, 0, modelViewMatrix, 0, h_pos.getWoordCoordMatrix(), 0);
                    Matrix.scaleM(modelViewCamera, 0, OBJECT_SCALE_FLOAT,
                            OBJECT_SCALE_FLOAT, OBJECT_SCALE_FLOAT);

//            Matrix.multiplyMM(modelViewProjection, 0, projectionMatrix, 0, modelViewCamera, 0);
                    Matrix.multiplyMM(modelViewProjection, 0, projectionMatrix, 0, modelViewMatrix, 0);
//                    if(show) {
//                    GLES20.glColorMask(false,false,false,false);
//                    glDrawButton(textureIndex, modelViewProjection);
//                    Matrix.scaleM(modelViewProjection, 0, 10f,
//                            10f, OBJECT_SCALE_FLOAT);
                        glDrawButton(textureIndex, modelViewProjection);
//                    GLES20.glColorMask(true,true,true,true);
                    }
//                }
                //*****************************************************************************************************

                tracking = true;
                outofFrameCounter = 0;
            }



            SampleUtils.checkGLError("Render Frame");
        }

        if(markerInFrame == false){
            for(int i = 0; i < 3; ++i){
                hoverOn[i] = false;
                waitingCounterCursor[i] = 0;
            }
        }




        //************************************************
        // NON_Marker Part
        //************************************************
        /*
        PHASE II
        BUTTON CONTROL
         */

        if (!markerInFrame && outofFrameCounter > 10){
            tracking = false;
        }
        outofFrameCounter++;

        //Log.d("markerInView",Boolean.toString(markerInFrame));
        //Log.d("non",Boolean.toString(navigationOn.inView));
        //Log.d("noff",Boolean.toString(navigationOff.inView));
        //Log.d("tracking",Boolean.toString(tracking));
        //Log.d("nonY",Float.toString(navigationOn.y));
        //Log.d("noffY",Float.toString(navigationOff.y));





        if(missionStarted){
            // Show or hide the button
            if(tracking == false){
                navigationOff.moveOut();
                navigationOn.moveOut();
                informationOff.moveOut();
                informationOn.moveOut();
            }
            if(tracking == true){
                navigationOff.moveIn();
                navigationOn.moveIn();
                informationOff.moveIn();
                informationOn.moveIn();
            }

            /***********************************************************
             * Navigation Button Control                               *
             ***********************************************************/

            if(navigationOn.inView && navigationOff.inView){
                if(distNavigation < 50 && navigationSelected == false){
                    // when hovered on, size increase
                    Matrix.scaleM(navigationOn.mvMatrix, 0, 1.1f,
                            1.1f, OBJECT_SCALE_FLOAT);
                    Matrix.multiplyMM(navigationOn.matrix4button, 0, projectionMatrix, 0, navigationOn.mvMatrix, 0);
                    glDrawButton(navigationOn.textureID,navigationOn.matrix4button);
                    Matrix.scaleM(navigationOn.mvMatrix, 0, 1/1.1f,
                            1/1.1f, OBJECT_SCALE_FLOAT);
                    Matrix.scaleM(navigationOff.mvMatrix, 0, 1.1f,
                            1.1f, OBJECT_SCALE_FLOAT);
                    Matrix.multiplyMM(navigationOff.matrix4button, 0, projectionMatrix, 0, navigationOff.mvMatrix, 0);
                    glDrawButton(navigationOff.textureID,navigationOff.matrix4button);
                    Matrix.scaleM(navigationOff.mvMatrix, 0, 1/1.1f,
                            1/1.1f, OBJECT_SCALE_FLOAT);

                    // Draw the waiting circle
                    Matrix.multiplyMM(navigation.matrix4button, 0, projectionMatrix, 0, navigation.mvMatrix, 0);
                    waitingCounterNavigation+=3;
                    if(waitingCounterNavigation > mWaitingCircle.getNumObjectIndex()){
                        waitingCounterNavigation -= mWaitingCircle.getNumObjectIndex();
                        navigationSelected = true;
                        ifShowArrow = !ifShowArrow;
                        Log.d("dist:ON", "ifShowArrow = !ifShowArrow;");
                        show = ! show;
                    }
                    glDrawWaiting(navigation.textureID,navigation.matrix4button,waitingCounterNavigation);

                }else if(navigationSelected == true){
                    navigationOn.rotate();
                    Matrix.multiplyMM(navigationOn.matrix4button, 0, projectionMatrix, 0, navigationOn.mvMatrix, 0);
                    glDrawButton(navigationOn.textureID,navigationOn.matrix4button);
                    navigationOff.rotate();
                    Matrix.multiplyMM(navigationOff.matrix4button, 0, projectionMatrix, 0, navigationOff.mvMatrix, 0);
                    glDrawButton(navigationOff.textureID,navigationOff.matrix4button);
                    angleIncreased += navigationOn.angleRotatingIncreasement;
                    if(angleIncreased >= 180){
                        navigationSelected = false;
                        angleIncreased = 0;
                    }
                }else{
                    waitingCounterNavigation = 0;
                    //        navigationOn.rotate();
                    Matrix.multiplyMM(navigationOn.matrix4button, 0, projectionMatrix, 0, navigationOn.mvMatrix, 0);
                    glDrawButton(navigationOn.textureID,navigationOn.matrix4button);
                    //        navigationOff.rotate();
                    Matrix.multiplyMM(navigationOff.matrix4button, 0, projectionMatrix, 0, navigationOff.mvMatrix, 0);
                    glDrawButton(navigationOff.textureID,navigationOff.matrix4button);
                }
            }else{
                waitingCounterNavigation = 0;
                //        navigationOn.rotate();
                Matrix.multiplyMM(navigationOn.matrix4button, 0, projectionMatrix, 0, navigationOn.mvMatrix, 0);
                glDrawButton(navigationOn.textureID,navigationOn.matrix4button);
                //        navigationOff.rotate();
                Matrix.multiplyMM(navigationOff.matrix4button, 0, projectionMatrix, 0, navigationOff.mvMatrix, 0);
                glDrawButton(navigationOff.textureID,navigationOff.matrix4button);
            }



            /***********************************************************
             * Information Button Control                              *
             ***********************************************************/

            if(informationOn.inView && informationOff.inView){
                if(distInformation < 50 && informationSelected == false){
                    // when hovered on, size increase
                    Matrix.scaleM(informationOn.mvMatrix, 0, 1.1f,
                            1.1f, OBJECT_SCALE_FLOAT);
                    Matrix.multiplyMM(informationOn.matrix4button, 0, projectionMatrix, 0, informationOn.mvMatrix, 0);
                    glDrawButton(informationOn.textureID,informationOn.matrix4button);
                    Matrix.scaleM(informationOn.mvMatrix, 0, 1/1.1f,
                            1/1.1f, OBJECT_SCALE_FLOAT);

                    Matrix.scaleM(informationOff.mvMatrix, 0, 1.1f,
                            1.1f, OBJECT_SCALE_FLOAT);
                    Matrix.multiplyMM(informationOff.matrix4button, 0, projectionMatrix, 0, informationOff.mvMatrix, 0);
                    glDrawButton(informationOff.textureID,informationOff.matrix4button);
                    Matrix.scaleM(informationOff.mvMatrix, 0, 1/1.1f,
                            1/1.1f, OBJECT_SCALE_FLOAT);

                    // Draw the waiting circle
                    Matrix.multiplyMM(information.matrix4button, 0, projectionMatrix, 0, information.mvMatrix, 0);
                    waitingCounterInformation+=3;
                    if(waitingCounterInformation > mWaitingCircle.getNumObjectIndex()){
                        waitingCounterInformation -= mWaitingCircle.getNumObjectIndex();
                        informationSelected = true;
                        show = ! show;
                    }
                    glDrawWaiting(information.textureID,information.matrix4button,waitingCounterInformation);

                }else if(informationSelected == true){
                    informationOn.rotate();
                    Matrix.multiplyMM(informationOn.matrix4button, 0, projectionMatrix, 0, informationOn.mvMatrix, 0);
                    glDrawButton(informationOn.textureID,informationOn.matrix4button);
                    informationOff.rotate();
                    Matrix.multiplyMM(informationOff.matrix4button, 0, projectionMatrix, 0, informationOff.mvMatrix, 0);
                    glDrawButton(informationOff.textureID,informationOff.matrix4button);
                    angleIncreased += informationOn.angleRotatingIncreasement;
                    if(angleIncreased >= 180){
                        informationSelected = false;
                        ifShowInformation = !ifShowInformation;
                        angleIncreased = 0;
                    }
                }else{
                    waitingCounterInformation = 0;
                    Matrix.multiplyMM(informationOn.matrix4button, 0, projectionMatrix, 0, informationOn.mvMatrix, 0);
                    glDrawButton(informationOn.textureID,informationOn.matrix4button);
                    Matrix.multiplyMM(informationOff.matrix4button, 0, projectionMatrix, 0, informationOff.mvMatrix, 0);
                    glDrawButton(informationOff.textureID,informationOff.matrix4button);
                }
            }else{
                waitingCounterInformation = 0;
                Matrix.multiplyMM(informationOn.matrix4button, 0, projectionMatrix, 0, informationOn.mvMatrix, 0);
                glDrawButton(informationOn.textureID,informationOn.matrix4button);
                Matrix.multiplyMM(informationOff.matrix4button, 0, projectionMatrix, 0, informationOff.mvMatrix, 0);
                glDrawButton(informationOff.textureID,informationOff.matrix4button);
            }
        }

        //这次得是draw waiting circle
//        Log.d("dist", Float.toString(distNavigation));




        //========  end of Junxiang's part  =============



        // Did we find any trackables this frame?
        for (int tIdx = 0; tIdx < state.getNumTrackableResults(); tIdx++) {
            TrackableResult result = state.getTrackableResult(tIdx);
            Trackable trackable = result.getTrackable();
            //Log.i("Mtrix", " state.getNumTrackableResults()  =  "+ state.getNumTrackableResults());
           // Log.i("Mtrix", "trackable  =  "+trackable.getName().toString());

            printUserData(trackable);
            Matrix44F modelViewMatrix_Vuforia = Tool
                    .convertPose2GLMatrix(result.getPose());
            float[] modelViewMatrix = modelViewMatrix_Vuforia.getData();

            int textureIndex = trackable.getName().equalsIgnoreCase("stones") ? 17
                    : 18;
            textureIndex = trackable.getName().equalsIgnoreCase("tarmac") ? 16
                    : textureIndex;


            //Log.i("Mtrix", "textureIndex  =  "+textureIndex);



            // deal with the modelview and projection matrices
            float[] modelViewProjection = new float[16];
            float[] modelViewProjection_backup = new float[16];
            float[] tempMatrix = new float[16];


            if (!mActivity.isExtendedTrackingActive()) {
                Matrix.translateM(modelViewMatrix, 0, 0.0f, 0.0f,
                        OBJECT_SCALE_FLOAT);
                Matrix.scaleM(modelViewMatrix, 0, OBJECT_SCALE_FLOAT,
                        OBJECT_SCALE_FLOAT, OBJECT_SCALE_FLOAT);
            } else {
                Matrix.rotateM(modelViewMatrix, 0, 90.0f, 1.0f, 0, 0);
                Matrix.scaleM(modelViewMatrix, 0, kBuildingScale,
                        kBuildingScale, kBuildingScale);
            }

            Matrix.multiplyMM(modelViewProjection, 0, projectionMatrix, 0, modelViewMatrix, 0);
            //printMatrix(projectionMatrix);

            Matrix.scaleM(modelViewProjection, 0, 0.5f, 0.5f, 0.5f);

            camera_location = getCamPose(modelViewMatrix);

            // activate the shader program and bind the vertex/normal/tex coords
            GLES20.glUseProgram(shaderProgramID);
            //backup the finalized projectionView
            modelViewProjection_backup = modelViewProjection.clone();


            //restore the finalized projectionView
            modelViewProjection = modelViewProjection_backup.clone();
            //draw a thing on the marker
            //drawMeBy2(0,0,0, textureIndex, modelViewProjection, makePureTranslationM(0,0,0));

            SampleUtils.checkGLError("Render Frame");

        }
        //above: only gets executed when image targets are detected.


        // activate the shader program and bind the vertex/normal/tex coords
        GLES20.glUseProgram(shaderProgramID);

        //draw arrow
        int textureIndex = 17;


        float [] arrow_rotation = makePureTranslationM(0,0,0);
        float [] arrow_rotation_offset = makePureTranslationM(0,0,0);
       // Matrix.setRotateEulerM(arrow_rotation,0,90,compassRead,0);









        GLES20.glEnable(GLES20.GL_DEPTH_TEST);

        //==================      revision from Junxiang =================
        //ARROW
        float [] default_projection = makePureTranslationM(0,0,0);  //hard coded

        float[] mvMatrix = new float[16];
        mvMatrix[0] = 0;
        mvMatrix[1] = -1;
        mvMatrix[2] = 0;
        mvMatrix[3] = 0;
        mvMatrix[4] = -1;
        mvMatrix[5] = 0;
        mvMatrix[6] = 0;
        mvMatrix[7] = 0;
        mvMatrix[8] = 0;
        mvMatrix[9] = 0;
        mvMatrix[10] = -1;
        mvMatrix[11] = 0;
        mvMatrix[12] = 0;
        mvMatrix[13] = 0;
        mvMatrix[14] = 300;//前移300
        mvMatrix[15] = 1;






        // 以camera为坐标原点，
        float[] mvMatrixCamera = new float[16];
        mvMatrixCamera[0] = 0;
        mvMatrixCamera[1] = -1;
        mvMatrixCamera[2] = 0;
        mvMatrixCamera[3] = 0;
        mvMatrixCamera[4] = -1;
        mvMatrixCamera[5] = 0;
        mvMatrixCamera[6] = 0;
        mvMatrixCamera[7] = 0;
        mvMatrixCamera[8] = 0;
        mvMatrixCamera[9] = 0;
        mvMatrixCamera[10] = -1;
        mvMatrixCamera[11] = 0;
        mvMatrixCamera[12] = 0;
        mvMatrixCamera[13] = 0;
        mvMatrixCamera[14] = 0;
        mvMatrixCamera[15] = 1;
        //==================    end of revision from Junxiang =================

        GLES20.glDisable(GLES20.GL_DEPTH_TEST);



        if(ifShowArrow&&missionStarted) {
            // ==================== drawing arrow ====================

            int lastestPointIndedx = getLatestPoint(locationList);
            //  if (lastestPointIndedx != -1)
            //      Log.i("Mtrix", "lastestPoint  =  " + locationList[lastestPointIndedx][0] + ", " + locationList[lastestPointIndedx][1]);
            float[] temp_p = new float[2];
            temp_p[0] = currentLocation[0] +0.00000f;
            temp_p[1] = currentLocation[1] +0.0004f;



            Matrix.rotateM(mvMatrix, 0, 90, 0, 1.0f, 0);
            if(orientatedByCompass) {
                Matrix.rotateM(mvMatrix, 0, compassRead, 0, 0, 1.0f);
            }
            else{

                if(lastestPointIndedx!=-1)
                    Matrix.rotateM(mvMatrix, 0, ((float)(gyroRotation_offset[0])-(float)(gyroRotation[0])+getOffAngle(currentLocation, locationList[lastestPointIndedx]))*180.0f/3.141592653f, 0, 0, 1.0f);
                else {

                    Matrix.rotateM(mvMatrix, 0, ((float)(gyroRotation_offset[0])-(float)(gyroRotation[0]))*180.0f/3.141592653f, 0, 0, 1.0f);
                    // Matrix.rotateM(mvMatrix, 0, ((float) (gyroRotation_offset[0]) - (float) (gyroRotation[0]) + getOffAngle(currentLocation, locationList[lastestPointIndedx])) * 180.0f / 3.141592653f, 0, 0, 1.0f);

                }


                // Matrix.rotateM(mvMatrix, 0, ((float)(gyroRotation_offset[1])-(float)(gyroRotation[1]))*180.0f/3.141592653f, 0, 1.0f, 0);
                //Matrix.rotateM(mvMatrix, 0, ((float)(gyroRotation_offset[2])-(float)(gyroRotation[2]))*180.0f3.141592653f, 1.0f, 0, 0);
            }
            Matrix.multiplyMM(default_projection, 0, projectionMatrix, 0, mvMatrix, 0);
            glDrawArrow(textureIndex, default_projection);





            // ==================== end of drawing arrow ====================

//        Matrix.rotateM(mvMatrix, 0, compassRead, 0, 0, 1.0f);
            Matrix.rotateM(mvMatrixCamera, 0, 90, 0, 1.0f, 0);
            if(orientatedByCompass) {
                Matrix.rotateM(mvMatrixCamera, 0, compassRead, 0, 0, 1.0f);
            }
            else{
                Matrix.rotateM(mvMatrixCamera, 0, ((float)(gyroRotation_offset[0])-(float)(gyroRotation[0]))*180.0f/3.141592653f, 0, 0, 1.0f);
                // Matrix.rotateM(mvMatrixCamera, 0, -((float)(gyroRotation_offset[1])-(float)(gyroRotation[1]))*180.0f/3.141592653f, 0, 1.0f, 0);
                //Matrix.rotateM(mvMatrixCamera, 0, ((float)(gyroRotation_offset[2])-(float)(gyroRotation[2]))*180.0f/3.141592653f, 1.0f, 0, 0);
            }

            float[] navi_modelViewProjection_origin = mvMatrixCamera.clone();
            float[] navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();


            // ==================== drawing landmarks ====================
        /*
        navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();

        Log.i("Mtrix", "locationList.length  =  "+locationList.length);
        int temp_textureIndex = 0;
        float approachingThres = 300.0f;
        for(int i = 0;i<locationList.length;i++){


           // Log.i("Mtrix", "getDistance2d  =  "+getDistance2d(currentLocation[0] * ratio+0,currentLocation[1] * ratio+ 200 * (compassRead - 180), locationList[i][0]*ratio,locationList[i][1]*ratio));

            //update landmarks
            if(getDistance2d(currentLocation[0] * ratio+0,currentLocation[1] * ratio+200 * (compassRead - 180), locationList[i][0]*ratio,locationList[i][1]*ratio)<approachingThres)  //0,0 or 0,0,0 is the location of the camera
                locationList[i][2] = 1.0f;

            if(locationList[i][2]>0)
                temp_textureIndex = 17;
            else
                temp_textureIndex = 18;

            drawMeBy2(locationList[i][0]*ratio, locationList[i][1]*ratio, 0, temp_textureIndex, navi_modelViewProjection_temp, makePureTranslationM(0, 0, 0));

        }

*/


            int temp_textureIndex = 17;
            float approachingThres = 100.0f;


            float[] translationM = makePureTranslationM(-currentLocation[0] * ratio, -currentLocation[1] * ratio, 0);
            //translationM = makePureTranslationM(0,0,0);

            //L
            navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();
            Matrix.multiplyMM(navi_modelViewProjection_temp, 0, navi_modelViewProjection_temp, 0, translationM, 0);
            Matrix.translateM(navi_modelViewProjection_temp, 0, 0, 400, 0);
            Matrix.multiplyMM(default_projection, 0, projectionMatrix, 0, navi_modelViewProjection_temp, 0);
            glDrawLandmark(textureIndex, default_projection);
            // Matrix.translateM(mvMatrixCamera, 0, 0, -400, 0);

            //F
            navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();
            Matrix.multiplyMM(navi_modelViewProjection_temp, 0, navi_modelViewProjection_temp, 0, translationM, 0);
            Matrix.translateM(navi_modelViewProjection_temp, 0, 400, 0, 0);
            Matrix.multiplyMM(default_projection, 0, projectionMatrix, 0, navi_modelViewProjection_temp, 0);
            glDrawLandmark(textureIndex, default_projection);
            //Matrix.translateM(mvMatrixCamera, 0, -400, 0, 0);

            //R
            navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();
            Matrix.multiplyMM(navi_modelViewProjection_temp, 0, navi_modelViewProjection_temp, 0, translationM, 0);
            Matrix.translateM(navi_modelViewProjection_temp, 0, 0, -400, 0);
            Matrix.multiplyMM(default_projection, 0, projectionMatrix, 0, navi_modelViewProjection_temp, 0);
            glDrawLandmark(textureIndex, default_projection);
            // Matrix.translateM(mvMatrixCamera, 0, 0, 400, 0);

            //B
            navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();
            Matrix.multiplyMM(navi_modelViewProjection_temp, 0, navi_modelViewProjection_temp, 0, translationM, 0);
            Matrix.translateM(navi_modelViewProjection_temp, 0, -400, 0, 0);
            Matrix.multiplyMM(default_projection, 0, projectionMatrix, 0, navi_modelViewProjection_temp, 0);
            glDrawLandmark(textureIndex, default_projection);
            // Matrix.translateM(mvMatrixCamera, 0, 400, 0, 0);


            //test
            navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();
            Matrix.multiplyMM(navi_modelViewProjection_temp, 0, navi_modelViewProjection_temp, 0, translationM, 0);
            Matrix.translateM(navi_modelViewProjection_temp, 0, (temp_p[0]) * ratio, (temp_p[1]) * ratio, 0);
            Matrix.multiplyMM(default_projection, 0, projectionMatrix, 0, navi_modelViewProjection_temp, 0);
            glDrawLandmark(18, default_projection);
            //Matrix.translateM(mvMatrixCamera, 0, 400, 0, 0);
            //34.415948f;
            //float currentLocationY = -119.844588f;

            for (int i = 0; i < locationList.length; i++) {


                // Log.i("Mtrix", "getDistance2d  =  "+getDistance2d(currentLocation[0] * ratio+0,currentLocation[1] * ratio+ 200 * (compassRead - 180), locationList[i][0]*ratio,locationList[i][1]*ratio));

                //update landmarks
                if (getDistance2d(currentLocation[0] * ratio + 0, currentLocation[1] * ratio + 200 * (compassRead - 180), locationList[i][0] * ratio, locationList[i][1] * ratio) < approachingThres)  //0,0 or 0,0,0 is the location of the camera
                    locationList[i][2] = 1.0f;

                if (locationList[i][2] > 0)
                    temp_textureIndex = 17;
                else
                    temp_textureIndex = 18;

                if(i==lastestPointIndedx)
                    temp_textureIndex = 16;

                //drawMeBy2(, , 0, temp_textureIndex, navi_modelViewProjection_temp, makePureTranslationM(0, 0, 0));
                navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();
                Matrix.multiplyMM(navi_modelViewProjection_temp, 0, navi_modelViewProjection_temp, 0, translationM, 0);
                Matrix.translateM(navi_modelViewProjection_temp, 0, locationList[i][0] * ratio, locationList[i][1] * ratio, 0);
                Matrix.multiplyMM(default_projection, 0, projectionMatrix, 0, navi_modelViewProjection_temp, 0);
                glDrawLandmark(temp_textureIndex, default_projection);

            }
            // ==================== end of drawing landmarks ====================





            //------------------drawing text------------------

            float [] HUD_modelViewProjection_origin = makePureTranslationM(1,4,3);
            float [] HUD_modelViewProjection_temp = makePureTranslationM(1,4,3);

            //draw HUD data display
            HUD_modelViewProjection_temp = HUD_modelViewProjection_origin.clone();
            Matrix.setRotateM(arrow_rotation_offset, 0, 90, 1, 0, 0);
            Matrix.multiplyMM(HUD_modelViewProjection_temp, 0, HUD_modelViewProjection_temp, 0, arrow_rotation_offset, 0);
            glText.begin(1.0f, 1.0f, 1.0f, 1.0f, HUD_modelViewProjection_temp);         // Begin Text Rendering (Set Color WHITE)

            glText.draw("Data = " + compassRead, -175, 120, 120, 90, 0, 0);                // Draw Test String
            glText.end();



            navi_modelViewProjection_temp = navi_modelViewProjection_origin.clone();
            glText.begin(1.0f, 1.0f, 1.0f, 1.0f, navi_modelViewProjection_temp);         // Begin Text Rendering (Set Color WHITE)
            glText.draw("I'm marker", -175, 120, 120, 90, 0, 0);                // Draw Test String
            glText.end();                                   // End Text Rendering

            //------------------end of drawing text------------------



        }


//        Matrix.multiplyMM(default_projection, 0, projectionMatrix, 0, mvMatrix, 0);







        //float [] navi_modelViewProjection_origin = makePureTranslationM(1,4,3);
        //float [] navi_modelViewProjection_temp = makePureTranslationM(1,4,3);






      //  int ratio = 2000000; //2000000

        //navi_modelViewProjection_origin =   moveToNaviWorldAt(0, 2 * (compassRead - 180), navi_modelViewProjection_origin);
        //navi_modelViewProjection_origin = moveToNaviWorldAt(currentLocation[0] * ratio+0, currentLocation[1] * ratio + 200 * (compassRead - 180), navi_modelViewProjection_origin);

        //Log.i("Mtrix", "currentLocation[1] * ratio + 200 * (compassRead - 180)  =  "+currentLocation[1] * ratio + 200 * (compassRead - 180));


        //Log.i("Mtrix", "lcurrentLocation  =  " + currentLocation[0] * ratio + ", " + currentLocation[1]*ratio);
        //Log.i("Mtrix", "locationList  =  "+locationList[1][0]*ratio+", "+locationList[1][1]*ratio);






        SampleUtils.checkGLError("Render Frame");



        //disable the depth test before drawing text
        GLES20.glDisable(GLES20.GL_DEPTH_TEST);





    }


    private float[]  moveToNaviWorldAt(float x, float y,float [] view){


        //moving virtual camera in the navigation world

        Matrix.multiplyMM(view, 0, view, 0, makePureTranslationM(-x, -y, 0), 0);  //remake the model view projection

        //end of moving virtual camera in the navigation world


        return view;
    }



    private void drawMeBy2(float x, float y, float z, int textureIndex, float[] modelViewProjection, float[] teapot_data_rotation) {
        float[] tempMatrix = new float[16];
        float[] modelViewProjection_temp = modelViewProjection.clone();
        //float[] modelViewProjection = new float[16];
        GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                false, 0, mTeapot.getVertices());
        GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                GLES20.GL_FLOAT, false, 0, mTeapot.getTexCoords());

        GLES20.glEnableVertexAttribArray(vertexHandle);
        GLES20.glEnableVertexAttribArray(textureCoordHandle);

        // activate texture 0, bind it, and pass to shader
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                mTextures.get(textureIndex).mTextureID[0]);
        GLES20.glUniform1i(texSampler2DHandle, 0);

        //do scaling  //just need once for all tespots
        //Matrix.scaleM(modelViewProjection, 0, 0.5f, 0.5f, 0.5f);


        tempMatrix = makePureTranslationM(x, y, z);
        Matrix.multiplyMM(modelViewProjection_temp, 0, modelViewProjection_temp, 0, tempMatrix, 0);
        Matrix.multiplyMM(modelViewProjection_temp, 0, modelViewProjection_temp, 0, teapot_data_rotation, 0);

        // pass the model view matrix to the shader
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                modelViewProjection_temp, 0);

        // finally draw the teapot
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                mTeapot.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                mTeapot.getIndices());

        // disable the enabled arrays
        GLES20.glDisableVertexAttribArray(vertexHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }

    private void printUserData(Trackable trackable) {
        String userData = (String) trackable.getUserData();
        //Log.d(LOGTAG, "UserData:Retreived User Data	\"" + userData + "\"");
        keyword = userData;
    }

    public void setTextures(Vector<Texture> textures) {
        mTextures = textures;
        Log.d(LOGTAG, "textures = "+textures.toString());

    }

    public void printMatrix(float[] matri) {
        Log.i("Mtrix", "===");
        Log.i("Mtrix", "" + matri[0] + " \t" + matri[1] + " \t" + matri[2] + " \t" + matri[3]);
        Log.i("Mtrix", "" + matri[4] + " \t" + matri[5] + " \t" + matri[6]+" \t" + matri[7]);
        Log.i("Mtrix", "" + matri[8] + " \t" + matri[9] + " \t" + matri[10] + " \t" + matri[11]);
        Log.i("Mtrix", "" + matri[12] + " \t" + matri[13] + " \t" + matri[14] + " \t" + matri[15]);
    }

    public float[] makePureTranslationM(float x,float y,float z) {
        float[] tempM = new float[16];
        tempM[0] = 1; tempM[1] = 0; tempM[2] = 0; tempM[3] = 0;
        tempM[4] = 0; tempM[5] = 1; tempM[6] = 0; tempM[7] = 0;
        tempM[8] = 0; tempM[9] = 0; tempM[10] = 1; tempM[11] = 0;
        tempM[12] = x; tempM[13] = y; tempM[14] = z; tempM[15] = 1;
        return tempM;
    }

    public float[] setTranslationM(float[] tempM,float x,float y,float z) {
        tempM[12] = x; tempM[13] = y; tempM[14] = z;
        return tempM;
    }

    public float getDistance(float x1, float y1, float z1, float x2, float y2, float z2 ){
        float distance = 0f;
        distance = (float)Math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
        return distance;
    }

    public float getDistance2d(float x1, float y1, float x2, float y2 ){
        float distance = 0f;
        distance = (float)Math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
        return distance;
    }

    public float[] getDistance4All(float[][] dataArray, float[] point) {

        float[] tempDis = new float [dataArray.length];

        for(int i = 0; i<tempDis.length;i++){
            tempDis[i] = getDistance(dataArray[i][0], dataArray[i][1], dataArray[i][2], point[0], point[1], point[2]);
        }
        return tempDis;
    }


    public float[] getCamPose(float [] mv){
        float [] pos = new float[3];
        float [] mv_inverse = new float[16];
        float [] mv_inverse_transpose = new float[16];
        Matrix.invertM(mv_inverse, 0, mv, 0);
        pos[0] = mv_inverse[12]*2.0f;
        pos[1] = mv_inverse[13]*2.0f;
        pos[2] = mv_inverse[14]*2.0f;
        return pos;
    }


    private float[][] decodePoly(String encoded){

        String poly = "";
        int index = 0, len = encoded.length();

        //println("len = " + len );

        ArrayList<float[]> v = new ArrayList<float[]>();
        int lat = 0, lng = 0;
        String p = "";


        while (index < len) {
            int b, shift = 0, result = 0;

            do {
                b = encoded.charAt(index++) - 63;
                result |= (b & 0x1f) << shift;
                shift += 5;
            } while (b >= 0x20);

            //println("result = " + result );

            int dlat = ((result & 1) != 0 ? ~(result >> 1) : (result >> 1));
            lat += dlat;

            shift = 0;
            result = 0;
            do {
                b = encoded.charAt(index++) - 63;
                result |= (b & 0x1f) << shift;
                shift += 5;
            } while (b >= 0x20);
            int dlng = ((result & 1) != 0 ? ~(result >> 1) : (result >> 1));
            lng += dlng;

            //p = "("+(int) (((double) lat / 1E5) * 1E6)+" , "+
            //   (int) (((double) lng / 1E5) * 1E6)+")";

            p = "("+ (((double) lat / 1E5) )+" , "+
                    (((double) lng / 1E5) )+")\n";


            //println("p = " + p );

            // println("counter = " + counter );
            poly+=p;
            //locations[counter][0] = (((double) lat / 1E5) );
            //locations[counter++][1] = (((double) lng / 1E5) );
            float[] temp_location = new float[3];
            temp_location[0] =  (float) (((float) lat / 1E5) );
            temp_location[1] = (float) (((float) lng / 1E5) );
            temp_location[2] = 0.0f;
            v.add(temp_location);

        }

        float[][] locations = new float[v.size()][3];

        for(int i =0;i<v.size();i++){
            //println("v = " + v.get(i)[1] );
            locations[i]=  v.get(i);
        }
        // println("v.size() = " + v.size() );
        return locations;
    }

    private String getJSON2(String url) {
String result = null;
        Log.i("Mtrix", "getJSON2() called");
        TestAsyncTask testAsyncTask = new TestAsyncTask(this, "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=34.415947,-119.84459&radius=500&keyword=&key=AIzaSyAmXO64yIHzwcdU4fqqAHwBdeh-jzoO0iw");
        testAsyncTask.execute();
        result  = testAsyncTask.getReulst();
        //Log.i("Mtrix", "result = "+result);
       // while(result==null){
            //Log.i("Mtrix", "waiting for result");
        //    result  = testAsyncTask.getReulst();

       // }

    return result;
    }


    private String getJSON(String url){

        Log.i("Mtrix", "getJSON() called");
        DefaultHttpClient   httpclient = new DefaultHttpClient(new BasicHttpParams());
        HttpPost httppost = new HttpPost(url);
// Depends on your web service
        httppost.setHeader("Content-type", "application/json");
        InputStream inputStream = null;
        String result = null;
        try {
            HttpResponse response = httpclient.execute(httppost);
            HttpEntity entity = response.getEntity();

            inputStream = entity.getContent();
            // json is UTF-8 by default
            BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"), 8);
            StringBuilder sb = new StringBuilder();

            String line = null;
            while ((line = reader.readLine()) != null)
            {
               // Log.i("Mtrix", "line = "+line);
                sb.append(line + "\n");
            }
            result = sb.toString();
        } catch (Exception e) {
            Log.i("Mtrix", "getJson error");
            // Oops
        }
        finally {
            try{if(inputStream != null)inputStream.close();}catch(Exception squish){}
        }

        return result;

    }


    private String getPointsAtStep(int step, String JSONString){
        String points = "";
        try {
            JSONObject jObject = new JSONObject(JSONString);
            JSONArray jArray = jObject.getJSONArray("routes");
            points =  jArray.getJSONObject(0).getJSONArray("legs").getJSONObject(0)
                    .getJSONArray("steps").getJSONObject(step).getJSONObject("polyline").getString("points").toString();
        } catch (JSONException e) {
            // Oops
        }
return points;
    }

    private int getNumbOfStep( String JSONString){
        int numbOfSteps = 0;
        try {
            JSONObject jObject = new JSONObject(JSONString);
            JSONArray jArray = jObject.getJSONArray("routes");
            numbOfSteps =  jArray.getJSONObject(0).getJSONArray("legs").getJSONObject(0)
                    .getJSONArray("steps").length();
        } catch (JSONException e) {
            // Oops
        }
        return numbOfSteps;
    }




    private float[][] getLocationAtStep(int step, String JSONString){
        String decodePoly =  getPointsAtStep(step, JSONString);
        float[][] result =  decodePoly(decodePoly);
        return result;
    }

    private float[][] getOverviewLocation( String JSONString){
        String points = "";
        try {
            JSONObject jObject = new JSONObject(JSONString);
            JSONArray jArray = jObject.getJSONArray("routes");
            points =  jArray.getJSONObject(0).getJSONObject("overview_polyline").getString("points").toString();
            //points =  jArray.getJSONObject(0).getJSONArray("overview_polyline").getJSONObject(0).getString("points").toString();
        } catch (JSONException e) {
            // Oops
        }
        float[][] result =  decodePoly(points);
       return result;
   }


    private float[][] getOverviewLocationFromUrl( String URL){
        Log.i("getJSON", "getOverviewLocationFromUrl");
        String JSONResult = getJSON(URL);
        Log.i("URL", "URL = "+URL);
        String points = "";
        try {
            JSONObject jObject = new JSONObject(JSONResult);
            JSONArray jArray = jObject.getJSONArray("routes");
            points =  jArray.getJSONObject(0).getJSONObject("overview_polyline").getString("points").toString();
            //points =  jArray.getJSONObject(0).getJSONArray("overview_polyline").getJSONObject(0).getString("points").toString();
        } catch (JSONException e) {
            // Oops
        }

        Log.i("getJSON", "overview_polyline = "+points);

        float[][] result =  decodePoly(points);
        return result;
    }


    private String getPlaceIdFromName(float [] gpsLocation,String keyword ){

        keyword = processJsonStringSapce(keyword);
        Log.i("getJSON", "getPlaceIdFromName");
        String URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location="+gpsLocation[0]+","+gpsLocation[1]+"&radius=500&keyword="+keyword+"&key=AIzaSyAmXO64yIHzwcdU4fqqAHwBdeh-jzoO0iw";
        Log.i("URL", "URL = "+URL);
        String JSONResult = getJSON(URL);
        //Log.i("getJSON", "JSONResult = "+JSONResult);
        String place_id = "";

        try {
            JSONObject jObject = new JSONObject(JSONResult);
            JSONArray jArray = jObject.getJSONArray("results");
            place_id =  jArray.getJSONObject(0).getString("place_id").toString();

            //points =  jArray.getJSONObject(0).getJSONArray("overview_polyline").getJSONObject(0).getString("points").toString();
        } catch (JSONException e) {
            // Oops
        }
        Log.i("getJSON", "place_id = "+place_id);
        return place_id;
    }


    private String getPlaceLocationFromName(float [] gpsLocation,String keyword ){

        keyword = processJsonStringSapce(keyword);
        Log.i("getJSON", "getPlaceLocationFromName");
        String URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?location="+gpsLocation[0]+","+gpsLocation[1]+"&radius=500&keyword="+keyword+"&key=AIzaSyAmXO64yIHzwcdU4fqqAHwBdeh-jzoO0iw";
        Log.i("URL", "URL = "+URL);

        String place_lcoation = "";


        String JSONResult = "";
        JSONResult = getJSON(URL);
       // Log.i("URL", "==========");
       // JSONResult  =getJSON2("");


        //Log.i("getJSON", "JSONResult = "+JSONResult);


        try {
            JSONObject jObject = new JSONObject(JSONResult);
            JSONArray jArray = jObject.getJSONArray("results");
            place_lcoation =  jArray.getJSONObject(0).getJSONObject("geometry").getJSONObject("location").getString("lat").toString();
            place_lcoation += ",";
            place_lcoation += jArray.getJSONObject(0).getJSONObject("geometry").getJSONObject("location").getString("lng").toString();
//.getJSONArray("location").getJSONObject(0).getString("lat")
            //points =  jArray.getJSONObject(0).getJSONArray("overview_polyline").getJSONObject(0).getString("points").toString();
        } catch (JSONException e) {
            // Oops
        }


        Log.i("getJSON", "place_lcoation = "+place_lcoation);
        return place_lcoation;
    }



private String processJsonStringSapce(String str){

    str = str.replaceAll("\\s","+");
    //Log.i("str", "processJsonStringSapce result = "+str);
    return str;

}


    public void voiceRecIsCalled(){

        Log.i("str", "voice is called");
        //img.test();


    }

    public void setIsRecording(boolean b){
        Log.i("str", "isRecording is set to "+b);
        isRecording = b;
    }
    public void setVoiceControlOn(boolean b){
        Log.i("str", "voiceControlOn is set to "+b);
        voiceControlOn = b;
    }
    public boolean getVoiceControlOn(){
return voiceControlOn;
    }

    public boolean getisRecording(){
        return isRecording;
    }


    public void setDestination(ArrayList<String> strArray){
int counter = 0;
        for (String result : strArray) {



            //Log.i("str", result);
            destination[counter] = result;
            counter++;

            if(counter==3)
                break;

        }

        if(counter<3) {

            for(int i = counter;i<3;i++){
                destination[i] = "";
        }
        }


        //vis
        for(int i = 0;i<3;i++){

            Log.i("str", "destination["+i+"] = "+destination[i]);

        }

    }



    public void receiveStringFromMain(ArrayList<String> strArray){

        Log.i("str", "Here are results from voice recognition.");
        for (String result : strArray)
            Log.i("str", result);

        setDestination(strArray);

        //startSpeechToText();
    }

public void runGoogleMapsNextFrame(){
    this.ifRunshowDirectionInGoogleMapsNextFrame = true;
}

    public void degubGoLeft(){
        this.currentLocation[0] += setp*ratio;
    }
    public void degubGoRight(){
        this.currentLocation[0] -= setp*ratio;
    }
    public void degubGoForward(){
        this.currentLocation[1] += setp*ratio;
    }
    public void degubGoBackward(){
        this.currentLocation[1] -= setp*ratio;
    }
    public void degubResetGyro(){
        this.gyroRotation_offset[0]  = gyroRotation[0];
        this.gyroRotation_offset[1]  = gyroRotation[1];
        this.gyroRotation_offset[2]  = gyroRotation[2];
    }
    public void changeOrientationMode(){
        this.orientatedByCompass = !this.orientatedByCompass;
    }


    public void showDirectionInGoogleMaps(){
        ifRunshowDirectionInGoogleMapsNextFrame = false;
        Log.i("str", "showDirectionInGoogleMaps, place_keyWord_destination = "+place_keyWord_destination);
        place_location_destination = getPlaceLocationFromName(currentLocation, place_keyWord_destination);


        // Create a Uri from an intent string. Use the result to create an Intent.
        Uri gmmIntentUri = Uri.parse("google.navigation:q=" + place_location_destination + "&mode=w");

// Create an Intent from gmmIntentUri. Set the action to ACTION_VIEW
        Intent mapIntent = new Intent(Intent.ACTION_VIEW, gmmIntentUri);
// Make the Intent explicit by setting the Google Maps package
        mapIntent.setPackage("com.google.android.apps.maps");

// Attempt to start an activity that can handle the Intent
        context.startActivity(mapIntent);


    }


    public int getLatestPoint(float [][] locationListInput){
        int indexCatcced = -1;
        for(int i = 0;i<locationListInput.length;i++){
            //Log.i("getLatestPoint", "locationListInput["+i+"][2] = "+locationListInput[i][2]);
            if(locationListInput[i][2]>0) {
                indexCatcced = i;
                break;
            }
        }

        return indexCatcced;
    }


    public float getOffAngle(float [] p1, float [] p2){
        float angle = 0.0f;
        angle = (float)(Math.atan2((double)(p1[1]-p2[1]),(double)(p1[0]-p2[0]))+Math.PI);
        Log.i("Mtrix", "getOffAngle  =  " + angle);
        return angle;
    }


    //==================================================================
//||
//||
//||
//||               UI  part
//||
//||
//||
//==================================================================

    //functions

    private void glDrawButton(int textureIndex,float[] modelViewProjection){
        // activate the shader program and bind the vertex/normal/tex coords
        GLES20.glUseProgram(shaderProgramID);

        GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                false, 0, mButton.getVertices());
        GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                GLES20.GL_FLOAT, false, 0, mButton.getTexCoords());


        GLES20.glEnableVertexAttribArray(vertexHandle);
        GLES20.glEnableVertexAttribArray(textureCoordHandle);

        // activate texture 0, bind it, and pass to shader
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                mTextures.get(textureIndex).mTextureID[0]);
        GLES20.glUniform1i(texSampler2DHandle, 0);

        // pass the model view matrix to the shader
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                modelViewProjection, 0);

        // finally draw the button
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                mButton.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                mButton.getIndices());
        // disable the enabled arrays
        GLES20.glDisableVertexAttribArray(vertexHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }

    // Draw the waiting circle, the difference between glDrawWaiting and glDraw button is the "int num" parameter.
    private void glDrawWaiting(int textureIndex,float[] modelViewProjection,int num){
        // activate the shader program and bind the vertex/normal/tex coords
        GLES20.glUseProgram(shaderProgramID);

        GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                false, 0, mWaitingCircle.getVertices());
        GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                GLES20.GL_FLOAT, false, 0, mWaitingCircle.getTexCoords());
        GLES20.glEnableVertexAttribArray(vertexHandle);
        GLES20.glEnableVertexAttribArray(textureCoordHandle);

        // activate texture 0, bind it, and pass to shader
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                mTextures.get(textureIndex).mTextureID[0]);
        GLES20.glUniform1i(texSampler2DHandle, 0);

        // pass the model view matrix to the shader
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                modelViewProjection, 0);

        // finally draw the button
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                num, GLES20.GL_UNSIGNED_SHORT,
                mWaitingCircle.getIndices());
        // disable the enabled arrays
        GLES20.glDisableVertexAttribArray(vertexHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }

    private void glDrawSquare(int textureIndex,float[] modelViewProjection){
        // activate the shader program and bind the vertex/normal/tex coords
        GLES20.glUseProgram(shaderProgramID);

        GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                false, 0, mSquare.getVertices());
        GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                GLES20.GL_FLOAT, false, 0, mSquare.getTexCoords());


        GLES20.glEnableVertexAttribArray(vertexHandle);
        GLES20.glEnableVertexAttribArray(textureCoordHandle);

        // activate texture 0, bind it, and pass to shader
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                mTextures.get(textureIndex).mTextureID[0]);
        GLES20.glUniform1i(texSampler2DHandle, 0);

        // pass the model view matrix to the shader
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                modelViewProjection, 0);

        // finally draw the button
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                mSquare.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                mSquare.getIndices());
        // disable the enabled arrays
        GLES20.glDisableVertexAttribArray(vertexHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }

    // world -> camera
    public float[] world2camera(float[] mvm, float x, float y, float z){
        float [] world = {x,y,z,1};
        float [] camera = new float[4];
        Matrix.multiplyMV(camera, 0, mvm, 0, world, 0);
        return camera;
    }
    // camera -> world
    public float[] camera2world(float[] mvm, float x, float y, float z){
        float [] camera = {x,y,z,1};
        float [] world = new float[4];
        float[] mvmIn = new float[16];
        Matrix.invertM(mvmIn,0,mvm,0);
        Matrix.multiplyMV(world, 0, mvmIn , 0, camera, 0);
        return world;
    }

    private void glDrawArrow(int textureIndex,float[] modelViewProjection){
        // activate the shader program and bind the vertex/normal/tex coords
        GLES20.glUseProgram(shaderProgramID);

        GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                false, 0, mTeapot.getVertices());
        GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                GLES20.GL_FLOAT, false, 0, mTeapot.getTexCoords());


        GLES20.glEnableVertexAttribArray(vertexHandle);
        GLES20.glEnableVertexAttribArray(textureCoordHandle);

        // activate texture 0, bind it, and pass to shader
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                mTextures.get(textureIndex).mTextureID[0]);
        GLES20.glUniform1i(texSampler2DHandle, 0);

        // pass the model view matrix to the shader
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                modelViewProjection, 0);

        // finally draw the button
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                mTeapot.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                mTeapot.getIndices());
        // disable the enabled arrays
        GLES20.glDisableVertexAttribArray(vertexHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }

    private void glDrawLandmark(int textureIndex,float[] modelViewProjection){
        // activate the shader program and bind the vertex/normal/tex coords
        GLES20.glUseProgram(shaderProgramID);

        GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                false, 0, mCube.getVertices());
        GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                GLES20.GL_FLOAT, false, 0, mCube.getTexCoords());


        GLES20.glEnableVertexAttribArray(vertexHandle);
        GLES20.glEnableVertexAttribArray(textureCoordHandle);

        // activate texture 0, bind it, and pass to shader
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                mTextures.get(textureIndex).mTextureID[0]);
        GLES20.glUniform1i(texSampler2DHandle, 0);

        // pass the model view matrix to the shader
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                modelViewProjection, 0);

        // finally draw the button
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                mCube.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                mCube.getIndices());
        // disable the enabled arrays
        GLES20.glDisableVertexAttribArray(vertexHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }

    //classes


    class handMarkerPosition {
        public float x;
        public float y;
        public float z;

        handMarkerPosition(float x, float y)
        {
            this.x = x;
            this.y = y;
            this.z = OBJECT_SCALE_FLOAT;
        }

        //using matrix rather than translateM
        public float[] getWoordCoordMatrix(){
            float[] worldMatrix = new float[16];
            Matrix.setIdentityM(worldMatrix,0);
            worldMatrix[12] = x;
            worldMatrix[13] = y;
            worldMatrix[14] = z;
            return worldMatrix;
        }

        public float cameraX;
        public float cameraY;
        public float cameraZ;
    }

    class Cursor {
        public float x;
        public float y;
        public float offset;
        public int textureID;
        public float[] mvMatrix = new float[16];
        public float[] mvMatrixOrigin = new float[16];
        public float[] matrix4button = new float[16];
        Cursor(float x, float y, float offset, int i)
        {
            this.x = x;
            this.y = y;
            this.offset = offset;
            textureID = i;
            mvMatrix[0] = 0;
            mvMatrix[1] = -1;
            mvMatrix[2] = 0;
            mvMatrix[4] = -1;
            mvMatrix[5] = 0;
            mvMatrix[6] = 0;
            mvMatrix[8] = 0;
            mvMatrix[9] = 0;
            mvMatrix[10] = -1;
            mvMatrix[12] = y;
            mvMatrix[13] = x;
            mvMatrix[14] = buttonDepth + offset;
            mvMatrix[15] = 1;
            mvMatrixOrigin = mvMatrix.clone();
        }
    }

    class Background {
        public float x;
        public float y;
        public float offset;
        public int textureID;
        public float[] mvMatrix = new float[16];
        public float[] mvMatrixOrigin = new float[16];
        public float[] matrix4button = new float[16];
        Background(float offset, int i)
        {
            this.x = 0;
            this.y = 0;
            this.offset = offset;
            textureID = i;
            mvMatrix[0] = 0;
            mvMatrix[1] = -1;
            mvMatrix[2] = 0;
            mvMatrix[4] = -1;
            mvMatrix[5] = 0;
            mvMatrix[6] = 0;
            mvMatrix[8] = 0;
            mvMatrix[9] = 0;
            mvMatrix[10] = -1;
            mvMatrix[12] = y;
            mvMatrix[13] = x;
            mvMatrix[14] = buttonDepth + 2.01f;
            mvMatrix[15] = 1;
            mvMatrixOrigin = mvMatrix.clone();
        }
    }

    class ButtonControl {
        public float x;
        public float y;
        public float z;
        public float homeX;
        public float homeY;
        public float hideX;
        public float hideY;
        public float offset;
        public float angle;
        public int textureID;
        public boolean inUse;
        public boolean inRotate;
        public boolean inMove;
        public boolean inView;
        public float[] mvMatrix = new float[16];
        public float[] mvMatrixOrigin = new float[16];
        public float[] matrix4button = new float[16];
        public float angleRotatingIncreasement = 10f;
        ButtonControl(float x, float y, float offset, int i, float angle) {
            this.offset = offset;
            this.z = buttonDepth;
            this.angle = angle;

            homeX = x;
            homeY = y;
            hideX = x;
            if(angle != 0){
                hideY = 55;
            }else{
                hideY = -185;
            }
            this.x = hideX;
            this.y = hideY;

            textureID = i;
            inUse = true;
            inRotate = false;
            inMove = false;
            inView = true;
            mvMatrix[0] = 0;
            mvMatrix[1] = -1;
            mvMatrix[2] = 0;
            mvMatrix[4] = -1;
            mvMatrix[5] = 0;
            mvMatrix[6] = 0;
            mvMatrix[8] = 0;
            mvMatrix[9] = 0;
            mvMatrix[10] = -1;
            if(angle != 0){
                mvMatrix[12] = hideY;
            }else{
                mvMatrix[12] = -1 * hideY;
                homeY = -1 * homeY;
            }
            mvMatrix[13] = hideX;
            mvMatrix[14] = buttonDepth;
            mvMatrix[15] = 1;
            mvMatrixOrigin = mvMatrix.clone();
        }
        void initRotate(){
            Matrix.rotateM(mvMatrix, 0, angle, 1.0f, 0, 0);
        }
        void rotate(){
            Matrix.rotateM(mvMatrix, 0, angleRotatingIncreasement, 1.0f, 0, 0);
            angle += angleRotatingIncreasement;
            if(angle >= 360){
                angle = 0;
            }
        }
        void moveOut(){
            inView = false;
            if(y > hideY) {
                Matrix.translateM(mvMatrix, 0, 0, -5, 0);
                y -= 5;
            }
            //修改boolean值
        }
        void moveIn(){
            if(y < homeY) {
                Matrix.translateM(mvMatrix, 0, 0, 5, 0);
                y += 5;
            }else{
                inView = true;
            }
        }

    }

    class waitingCircleControl {
        public float x;
        public float y;
        public float offset;
        public int textureID;
        public float[] mvMatrix = new float[16];
        public float[] mvMatrixOrigin = new float[16];
        public float[] matrix4button = new float[16];
        waitingCircleControl(float x, float y, float offset, int i)
        {
            this.x = x;
            this.y = y;
            this.offset = offset;
            textureID = i;
            mvMatrix[0] = 0;
            mvMatrix[1] = -1;
            mvMatrix[2] = 0;
            mvMatrix[4] = -1;
            mvMatrix[5] = 0;
            mvMatrix[6] = 0;
            mvMatrix[8] = 0;
            mvMatrix[9] = 0;
            mvMatrix[10] = -1;
            mvMatrix[12] = y;
            mvMatrix[13] = x;
            mvMatrix[14] = buttonDepth + 1 + offset;
            mvMatrix[15] = 1;
            mvMatrixOrigin = mvMatrix.clone();
        }
    }






}








