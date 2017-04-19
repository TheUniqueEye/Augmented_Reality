package com.vuforia.samples.SampleApplication.utils;

import java.nio.Buffer;

/**
 * Created by Zhenyu on 2016-12-06.
 */

public class WaitingCircle extends MeshObject {
    private Buffer mVertBuff;
    private Buffer mTexCoordBuff;
    private Buffer mNormBuff;
    private Buffer mIndBuff;

    private int indicesNumber = 180;
    private int verticesNumber = 180;

    private float scale = 5.0f;
    private float r = 20.0f;


    public WaitingCircle() {
    }

    public double[] BUTTON_VERTS = new double[verticesNumber * 3];//顶点坐标数据
    public double[] BUTTON_TEX_COORDS = new double[verticesNumber * 2];//顶点纹理S,T坐标值数组
    public double[] BUTTON_NORMS = new double[verticesNumber * 3];
    public short[] BUTTON_INDICES = new short[verticesNumber];


    public void initData() {
        //坐标数据初始化
        int count = 0;
        int stCount = 0;
        int indexCount = 0;
        int angleIncrease = 6;
        for (int angdeg = 0; angdeg < 360; angdeg += 6) {
            double angrad = Math.toRadians(angdeg);//当前弧度
            double angradNext = Math.toRadians(angdeg + angleIncrease);//下一弧度
            // 中心点
            BUTTON_VERTS[count++] = 0;//顶点坐标
            BUTTON_VERTS[count++] = 0;
            BUTTON_VERTS[count++] = 3.0f;
            BUTTON_TEX_COORDS[stCount++] = 0.5f;//ST坐标
            BUTTON_TEX_COORDS[stCount++] = 0.5f;
            BUTTON_INDICES[indexCount] = (short) indexCount;
            indexCount++;
            //当前点
            BUTTON_VERTS[count++] = r * Math.cos(angrad);
            BUTTON_VERTS[count++] = r * Math.sin(angrad);//顶点坐标
            BUTTON_VERTS[count++] = 3.0f;
            BUTTON_TEX_COORDS[stCount++] = 0.5 + 0.5 * Math.cos(angrad);
            BUTTON_TEX_COORDS[stCount++] = 0.5 + 0.5 * Math.sin(angrad);//st坐标
            BUTTON_INDICES[indexCount] = (short) indexCount;
            indexCount++;
            //下一点
            BUTTON_VERTS[count++] = r * Math.cos(angradNext);
            BUTTON_VERTS[count++] = r * Math.sin(angradNext);//顶点坐标
            BUTTON_VERTS[count++] = 3.0f;
            BUTTON_TEX_COORDS[stCount++] = 0.5 + 0.5 * Math.cos(angradNext);
            BUTTON_TEX_COORDS[stCount++] = 0.5 + 0.5 * Math.sin(angradNext);//st坐标
            BUTTON_INDICES[indexCount] = (short) indexCount;
            indexCount++;
        }
        setVerts();
        setTexCoords();
        setNorms();
        setIndices();
    }

    private void setVerts() {
        mVertBuff = fillBuffer(BUTTON_VERTS);
        verticesNumber = BUTTON_VERTS.length / 3;
    }


    private void setTexCoords() {
//        double[] BUTTON_TEX_COORDS = {  };
        mTexCoordBuff = fillBuffer(BUTTON_TEX_COORDS);
    }


    private void setNorms() {
//        double[] BUTTON_NORMS = {  };
//        Vec3F v1,v2,v3,norm;
//        v1 = new Vec3F(BUTTON_VERTS[0], BUTTON_VERTS[1], BUTTON_VERTS[2]);
//        v2 = new Vec3F(BUTTON_VERTS[3]-BUTTON_VERTS[0], BUTTON_VERTS[4]-BUTTON_VERTS[1], BUTTON_VERTS[5]-BUTTON_VERTS[2]);
//        v3 = new Vec3F(BUTTON_VERTS[6]-BUTTON_VERTS[0], BUTTON_VERTS[7]-BUTTON_VERTS[1], BUTTON_VERTS[8]-BUTTON_VERTS[2]);
//        norm = SampleMath.Vec3FCross(v3,v2);
//        float[] normValue = norm.getData();
//        double normalized = Math.sqrt(normValue[0] * normValue[0] + normValue[1] * normValue[1] + normValue[2] * normValue[2]);
//
//        normValue[0] /= normalized;
//        normValue[1] /= normalized;
//        normValue[2] /= normalized;

        int counter = 0;
        for (int i = 0; i < verticesNumber; ++i) {
            BUTTON_NORMS[counter++] = 0;
            BUTTON_NORMS[counter++] = 0;
            BUTTON_NORMS[counter++] = 1.0;
        }

        mNormBuff = fillBuffer(BUTTON_NORMS);
    }


    private void setIndices() {
        mIndBuff = fillBuffer(BUTTON_INDICES);
        indicesNumber = BUTTON_INDICES.length;
    }


    public int getNumObjectIndex() {
        return indicesNumber;
    }


    @Override
    public int getNumObjectVertex() {
        return verticesNumber;
    }


    @Override
    public Buffer getBuffer(MeshObject.BUFFER_TYPE bufferType) {
        Buffer result = null;
        switch (bufferType) {
            case BUFFER_TYPE_VERTEX:
                result = mVertBuff;
                break;
            case BUFFER_TYPE_TEXTURE_COORD:
                result = mTexCoordBuff;
                break;
            case BUFFER_TYPE_NORMALS:
                result = mNormBuff;
                break;
            case BUFFER_TYPE_INDICES:
                result = mIndBuff;
            default:
                break;

        }

        return result;
    }
}
