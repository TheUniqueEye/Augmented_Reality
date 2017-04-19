/*===============================================================================
Copyright (c) 2016 PTC Inc. All Rights Reserved.

Copyright (c) 2012-2014 Qualcomm Connected Experiences, Inc. All Rights Reserved.

Vuforia is a trademark of PTC Inc., registered in the United States and other 
countries.
===============================================================================*/

package com.vuforia.samples.VuforiaSamples.app.ImageTargets;

import java.io.IOException;
import java.util.Vector;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

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
import com.vuforia.samples.SampleApplication.utils.CubeShaders;
import com.vuforia.samples.SampleApplication.utils.LoadingDialogHandler;
import com.vuforia.samples.SampleApplication.utils.SampleApplication3DModel;
import com.vuforia.samples.SampleApplication.utils.SampleUtils;
import com.vuforia.samples.SampleApplication.utils.Teapot;
import com.vuforia.samples.SampleApplication.utils.Texture;



// The renderer class for the ImageTargets sample. 
public class ImageTargetRenderer implements GLSurfaceView.Renderer, SampleAppRendererControl
{
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
    
    private float OBJECT_SCALE_FLOAT = 0.7f; // teapot scale


    // deal with the modelview and projection matrices
    float[] inverseMV = new float[16];
    float[] modelViewProjection = new float[16];
    float[] mvProjection = new float[16];

    // camera matrix
    float[] camPosition = new float[3];
    float[] camMatrix = new float[16];


    boolean isAnyPickup = false;
    boolean reset = true;
    int index=0;
    float min_disbt=0;


    // teapot
    float[][] teapot = new float[4][4]; //  4 teapots | x,y,z,ispickedup

    // individual transformation matrix to 4 teapots
    float[] transforM = new float[16]; // 1 picked teapots | 4*4 transformation
    float[][] rotateMatrix = new float[4][16]; // 4 teapots | rotation matrix


    float[] tempMatrixUp = new float[16];
    float[] tempMatrixDown = new float[16];



    double yaw;
    int index_rotate=5;


    public ImageTargetRenderer(ImageTargets activity, SampleApplicationSession session)
    {
        mActivity = activity;
        vuforiaAppSession = session;
        // SampleAppRenderer used to encapsulate the use of RenderingPrimitives setting
        // the device mode AR/VR and stereo mode
        mSampleAppRenderer = new SampleAppRenderer(this, mActivity, Device.MODE.MODE_AR, false, 10f , 5000f);
    }
    
    
    // Called to draw the current frame.
    @Override
    public void onDrawFrame(GL10 gl)
    {
        if (!mIsActive)
            return;
        
        // Call our function to render content from SampleAppRenderer class
        mSampleAppRenderer.render();
    }
    

    public void setActive(boolean active)
    {
        mIsActive = active;

        if(mIsActive)
            mSampleAppRenderer.configureVideoBackground();
    }


    // Called when the surface is created or recreated.
    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config)
    {
        //Log.d(LOGTAG, "GLRenderer.onSurfaceCreated");
        
        // Call Vuforia function to (re)initialize rendering after first use
        // or after OpenGL ES context was lost (e.g. after onPause/onResume):
        vuforiaAppSession.onSurfaceCreated();

        mSampleAppRenderer.onSurfaceCreated();
    }
    
    
    // Called when the surface changed size.
    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        //Log.d(LOGTAG, "GLRenderer.onSurfaceChanged");
        
        // Call Vuforia function to handle render surface size changes:
        vuforiaAppSession.onSurfaceChanged(width, height);

        // RenderingPrimitives to be updated when some rendering change is done
        mSampleAppRenderer.onConfigurationChanged(mIsActive);

        initRendering();
    }
    
    
    // Function for initializing the renderer.
    private void initRendering()
    {
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, Vuforia.requiresAlpha() ? 0.0f
                : 1.0f);
        
        for (Texture t : mTextures)
        {
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

        if(!mModelIsLoaded) {
            mTeapot = new Teapot();

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

        // Initialization
        // teapot position in the world coordinate system

        // teapot Num.1
        //tMatrix1 = maketMatrix(-150,0,0); // transformMatrix: x,y,z
        teapot[0][0] = -150;
        teapot[0][1] = 0;
        teapot[0][2] = 0;
        teapot[0][3] = 0; // picked up == false

        // teapot Num.2
        //tMatrix2 = maketMatrix(-50,0,0);
        teapot[1][0] = -50;
        teapot[1][1] = 0;
        teapot[1][2] = 0;
        teapot[1][3] = 0;

        // teapot Num.3
        //tMatrix3 = maketMatrix(50,0,0);
        teapot[2][0] = 50;
        teapot[2][1] = 0;
        teapot[2][2] = 0;
        teapot[2][3] = 0;

        // teapot Num.4
        //tMatrix4 = maketMatrix(150,0,0);
        teapot[3][0] = 150;
        teapot[3][1] = 0;
        teapot[3][2] = 0;
        teapot[3][3] = 0;

        // rotation matrix
        for(int i=0; i<4; i++){
            rotateMatrix[i][0]=1;
            rotateMatrix[i][5]=1;
            rotateMatrix[i][10]=1;
            rotateMatrix[i][15]=1;
        }

        tempMatrixUp[0]=1;
        tempMatrixUp[5]=1;
        tempMatrixUp[10]=1;
        tempMatrixUp[15]=1;


        tempMatrixDown[0]=1;
        tempMatrixDown[5]=1;
        tempMatrixDown[10]=1;
        tempMatrixDown[15]=1;


    }

    //-----------------------------------

    // function for drawing teapot
    public void drawTeapot(float[] teap, int textureIndex)
    {


        float[] tMatrix = maketMatrix(teap[0],teap[1],teap[2]);
        mvProjection = modelViewProjection.clone();

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

        Matrix.multiplyMM(mvProjection, 0, mvProjection, 0, tMatrix, 0);

        // pass the model view matrix to the shader
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                mvProjection, 0);

        // finally draw the teapot
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                mTeapot.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                mTeapot.getIndices());

        // disable the enabled arrays
        GLES20.glDisableVertexAttribArray(vertexHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }

    //-----------------------------------

    // function for drawing rotated teapot
    public void drawRotateTeapot(float[] teap, float[] rot, int textureIndex)
    {

        float[] tMatrix = maketMatrix(teap[0],teap[1],teap[2]);
        mvProjection = modelViewProjection.clone();

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

        Matrix.multiplyMM(mvProjection, 0, mvProjection, 0, tMatrix, 0);
        Matrix.multiplyMM(mvProjection, 0, mvProjection, 0, rot, 0);


        // pass the model view matrix to the shader
        GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                mvProjection, 0);

        // finally draw the teapot
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                mTeapot.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                mTeapot.getIndices());

        // disable the enabled arrays
        GLES20.glDisableVertexAttribArray(vertexHandle);
        GLES20.glDisableVertexAttribArray(textureCoordHandle);
    }



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


    // distance between teapot and camera
    public int distance(float[] camPosition, float[][] teapot)
    {
        float[] dis = new float[4];
        float temp;
        int ind=0;

        for(int i=0; i<4; i++) {
            dis[i] = (float) Math.sqrt(
                    Math.pow(camPosition[0] - teapot[i][0], 2) +
                            Math.pow(camPosition[1] - teapot[i][1], 2)
                            + Math.pow(camPosition[2] - teapot[i][2], 2)
            );
        }

        //Log.d(LOGTAG, "dis1  = "+ dis[0]+" dis1  = "+ dis[1]+" dis1  = "+ dis[2]+" dis1  = "+ dis[3]);

        temp = dis[0];
        for(int i=0; i<4; i++) {
            if (temp > dis[i]) {
                temp = dis[i];
                ind = i;
            }
        }
        return ind;

    }

    // computing the min distance between picked teapot(camera) and others ---- for setting bounding
    public float distancebt(int index, float[] camPosition, float[][] teapot){
        float[] disbt = new float[4];
        float temp_dis= 10000;

        for(int i=0; i<4; i++){
            disbt[i] = (float) Math.sqrt(
                    Math.pow(camPosition[0] - teapot[i][0], 2) +
                            Math.pow(camPosition[1] - teapot[i][1], 2)
                            + Math.pow(camPosition[2] - teapot[i][2], 2)
            );
            if(i==index){
                disbt[i]=10000; // extreme large num
            }
        }

        for(int i=0; i<4; i++) {
            if (temp_dis > disbt[i]) {
                temp_dis = disbt[i];
            }
        }
        return temp_dis; // the least distance between the picked teapot and others
    }



    //-----------------------------------


    public void updateConfiguration()
    {
        mSampleAppRenderer.onConfigurationChanged(mIsActive);
    }

    // The render function called from SampleAppRendering by using RenderingPrimitives views.
    // The state is owned by SampleAppRenderer which is controlling it's lifecycle.
    // State should not be cached outside this method.
    public void renderFrame(State state, float[] projectionMatrix)
    {
        // Renders video background replacing Renderer.DrawVideoBackground()
        mSampleAppRenderer.renderVideoBackground();

        GLES20.glEnable(GLES20.GL_DEPTH_TEST);

        // handle face culling, we need to detect if we are using reflection
        // to determine the direction of the culling
        GLES20.glEnable(GLES20.GL_CULL_FACE);
        GLES20.glCullFace(GLES20.GL_BACK);

        // Did we find any trackables this frame?
        for (int tIdx = 0; tIdx < state.getNumTrackableResults(); tIdx++) {
            TrackableResult result = state.getTrackableResult(tIdx);
            Trackable trackable = result.getTrackable();
            printUserData(trackable);
            Matrix44F modelViewMatrix_Vuforia = Tool
                    .convertPose2GLMatrix(result.getPose());
            float[] modelViewMatrix = modelViewMatrix_Vuforia.getData();



            int textureIndex = trackable.getName().equalsIgnoreCase("stones") ? 0
                    : 1;
            textureIndex = trackable.getName().equalsIgnoreCase("tarmac") ? 2
                    : textureIndex;


            if (!mActivity.isExtendedTrackingActive()) {

                // for drawing teapot

                Matrix.translateM(modelViewMatrix, 0, 0.0f, 0.0f,
                        OBJECT_SCALE_FLOAT);
                Matrix.scaleM(modelViewMatrix, 0, OBJECT_SCALE_FLOAT,
                        OBJECT_SCALE_FLOAT, OBJECT_SCALE_FLOAT);

            } else {

                // for drawing buildings

                Matrix.rotateM(modelViewMatrix, 0, 90.0f, 1.0f, 0, 0);
                Matrix.scaleM(modelViewMatrix, 0, kBuildingScale,
                        kBuildingScale, kBuildingScale);
            }
            Matrix.multiplyMM(modelViewProjection, 0, projectionMatrix, 0, modelViewMatrix, 0);

            Matrix.invertM(inverseMV,0,modelViewMatrix,0);

            //Log.d(LOGTAG, "============");
            //Log.d(LOGTAG, "inverseMV= "+inverseMV[12]+" , "+inverseMV[13]+" , "+inverseMV[14]);


            // compute camera position in the world coordinate :
            camPosition[0]=inverseMV[12];
            camPosition[1]=inverseMV[13];
            camPosition[2]=inverseMV[14];


            // activate the shader program and bind the vertex/normal/tex coords
            GLES20.glUseProgram(shaderProgramID);




            //-------------- rendering teapot & UI-----------------------

            if (!mActivity.isExtendedTrackingActive()) {

                //Log.d(LOGTAG, "index \"" + index + "\""); // print index

                float minDis;
                float camHeight = camPosition[2];


                // -----reset to enable another round of pickup, leave the working zone------

                if(camHeight>150){
                    reset = true;
                }

                /*
                Log.d(LOGTAG, "pot 1 \"" + teapot[0][3] + "\"");
                Log.d(LOGTAG, "pot 2 \"" + teapot[1][3] + "\"");
                Log.d(LOGTAG, "pot 3 \"" + teapot[2][3] + "\"");
                Log.d(LOGTAG, "pot 4 \"" + teapot[3][3] + "\"");
                Log.d(LOGTAG, "============");
                Log.d(LOGTAG, "isAnyPickup = "+isAnyPickup);
                */


                // -------------for teapot picked up---------------
                if(isAnyPickup){

                    //Log.d(LOGTAG, "index =  "+(index+1)); // print index

                    min_disbt = distancebt(index,camPosition,teapot);
                    //Log.d(LOGTAG, "min_disbt = "+min_disbt);

                    // ------------- put teapot down---------------

                    // distance between camera and marker < 70 && > bounding distance 80

                    if(camHeight<70 && reset && min_disbt>80){

                        //Log.d(LOGTAG, "put down "+(index+1));

                        isAnyPickup = false;
                        reset = false;

                        teapot[index][0] = camPosition[0];
                        teapot[index][1] = camPosition[1];
                        teapot[index][2] = 0;
                        teapot[index][3] = 0;

                        // compute the camera rotation (yaw) from pickup to putdown > set as rotation Matrix
                        tempMatrixDown = modelViewMatrix.clone();
                        yaw = Math.atan2(tempMatrixDown[4],tempMatrixDown[0])-Math.atan2(tempMatrixUp[4],tempMatrixUp[0]);

                        yaw = -(yaw/Math.PI) * 180;
                        //Log.d(LOGTAG, "yaw 1 "+Math.atan2(tempMatrixDown[4],tempMatrixDown[0]));
                        Log.d(LOGTAG, "yaw 2 "+Math.atan2(tempMatrixUp[4],tempMatrixUp[0]));
                        Log.d(LOGTAG, "yaw  "+ yaw);

                        float[] tempRotate = new float[16];
                        //Matrix.setRotateM(rotateMatrix[index],0,(float)yaw,0,0,1);
                        Matrix.setRotateEulerM(tempRotate,0,0,0,(float)yaw);
                        Matrix.multiplyMM(rotateMatrix[index],0,rotateMatrix[index],0,tempRotate,0);

                    }else{

                        // ------------ render picked up teapot in front of camera---------------
                        for(int i=0; i<4; i++){
                            if(teapot[i][3]==0){
                                //drawTeapot(teapot[i],textureIndex);
                                drawRotateTeapot(teapot[i], rotateMatrix[i], textureIndex);
                            }else{
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

                                // 1. do not multiply the modelViewMatrix
                                // 2. translate a certain distance in front of the camera
                                //Matrix.multiplyMM(projectionMatrix, 0, projectionMatrix, 0, maketMatrix(0, 0, 40), 0); // *multiply order

                                // pass the transform matrix to the shader
                                GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                                        transforM, 0); // use the transforM matrix - to keep relative rotation+position


                                // finally draw the teapot
                                GLES20.glDrawElements(GLES20.GL_TRIANGLES,
                                        mTeapot.getNumObjectIndex(), GLES20.GL_UNSIGNED_SHORT,
                                        mTeapot.getIndices());

                                // disable the enabled arrays
                                GLES20.glDisableVertexAttribArray(vertexHandle);
                                GLES20.glDisableVertexAttribArray(textureCoordHandle);

                            }
                        }
                    }

                // ---------------for teapot not picked up---------------
                }else{

                    index = distance(camPosition,teapot);
                    //Log.d(LOGTAG, "index =  "+(index+1));

                    minDis = (float) Math.sqrt(
                            Math.pow(camPosition[0] - teapot[index][0], 2) +
                                    Math.pow(camPosition[1] - teapot[index][1], 2)
                                    + Math.pow(camPosition[2] - teapot[index][2], 2)
                    );



                    // ---------------pick teapot up---------------
                    if(minDis < 100 && reset) {
                        //Log.d(LOGTAG, "pick up "+(index+1));
                        isAnyPickup=true;
                        reset = false;
                        teapot[index][3]=1;

                        // store the relative transform matrix from camera to picked teapot
                        float[] tMatrix = maketMatrix(teapot[index][0],teapot[index][1],teapot[index][2]);
                        float[] tempMatrix = modelViewProjection.clone(); // stored camera transformation matrix
                        tempMatrixUp = modelViewMatrix.clone();

                        Matrix.multiplyMM(tempMatrix, 0, tempMatrix, 0, tMatrix, 0);
                        Matrix.multiplyMM(tempMatrix, 0, tempMatrix, 0, rotateMatrix[index], 0);

                        transforM = tempMatrix.clone();

                    //---------------render 4 teapots at their position on the marker---------------
                    }else{
                        for(int i=0; i<4; i++) {
                                drawRotateTeapot(teapot[i], rotateMatrix[i], textureIndex);
                        }
                    }
                }
            }



            //-------------- rendering Building-----------------------

            else {
                GLES20.glDisable(GLES20.GL_CULL_FACE);
                GLES20.glVertexAttribPointer(vertexHandle, 3, GLES20.GL_FLOAT,
                        false, 0, mBuildingsModel.getVertices());
                GLES20.glVertexAttribPointer(textureCoordHandle, 2,
                        GLES20.GL_FLOAT, false, 0, mBuildingsModel.getTexCoords());

                GLES20.glEnableVertexAttribArray(vertexHandle);
                GLES20.glEnableVertexAttribArray(textureCoordHandle);

                GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,
                        mTextures.get(3).mTextureID[0]);
                GLES20.glUniformMatrix4fv(mvpMatrixHandle, 1, false,
                        modelViewProjection, 0);
                GLES20.glUniform1i(texSampler2DHandle, 0);
                GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0,
                        mBuildingsModel.getNumObjectVertex());

                SampleUtils.checkGLError("Renderer DrawBuildings");
            }

            SampleUtils.checkGLError("Render Frame");

        }

        GLES20.glDisable(GLES20.GL_DEPTH_TEST);

    }

    private void printUserData(Trackable trackable)
    {
        String userData = (String) trackable.getUserData();
        //Log.d(LOGTAG, "UserData:Retreived User Data	\"" + userData + "\"");
    }
    
    
    public void setTextures(Vector<Texture> textures)
    {
        mTextures = textures;
        
    }
    
}
