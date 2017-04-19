package com.vuforia.samples.SampleApplication.utils;

import java.nio.Buffer;


public class Arrow extends MeshObject
{

    private Buffer mVertBuff;
    private Buffer mTexCoordBuff;
    private Buffer mNormBuff;
    private Buffer mIndBuff;

    private int indicesNumber = 0;
    private int verticesNumber = 0;

    public Arrow()
    {
        setVerts();
        setTexCoords();
        setNorms();
        setIndices();
    }

    private void setVerts()
    {
        double[] TEAPOT_VERTS = {
                30.0, 0.0, 0.0,
                -29.0,24.0, 0.0,
                -16.0,-12.0, 0.0,
                -16.0,12.0, 0.0,
                -29.0,-24.0, 0.0
        };
        mVertBuff = fillBuffer(TEAPOT_VERTS);
        verticesNumber = TEAPOT_VERTS.length / 3;
    }


    private void setTexCoords()
    {
        double[] TEAPOT_TEX_COORDS = {
                1.0,1.0,
                1.0,0.0,
                0.0,0.0,
                0.0,0.0,
                0.0,1.0
        };
        mTexCoordBuff = fillBuffer(TEAPOT_TEX_COORDS);

    }


    private void setNorms()
    {
        double[] TEAPOT_NORMS = {
                0.0,0.0,1.0,
                0.0,0.0,1.0,
                0.0,0.0,1.0,
                0.0,0.0,1.0,
                0.0,0.0,1.0
        };
        mNormBuff = fillBuffer(TEAPOT_NORMS);
    }


    private void setIndices()
    {

        short[] TEAPOT_INDICES = {
                2, 1, 0,
                3, 1, 0,
                4, 2, 0,
                4, 3, 0,
                3, 2, 1,
                2, 3, 4
        };
        mIndBuff = fillBuffer(TEAPOT_INDICES);
        indicesNumber = TEAPOT_INDICES.length;
    }


    public int getNumObjectIndex()
    {
        return indicesNumber;
    }


    @Override
    public int getNumObjectVertex()
    {
        return verticesNumber;
    }


    @Override
    public Buffer getBuffer(BUFFER_TYPE bufferType)
    {
        Buffer result = null;
        switch (bufferType)
        {
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