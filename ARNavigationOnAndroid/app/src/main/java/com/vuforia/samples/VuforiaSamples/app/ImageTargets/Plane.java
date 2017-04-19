package com.vuforia.samples.VuforiaSamples.app.ImageTargets;

import com.vuforia.samples.SampleApplication.utils.MeshObject;

import java.nio.Buffer;

/**
 * Created by EYE on 04/12/2016.
 */


public class Plane extends MeshObject
{

    private Buffer mVertBuff_p;
    private Buffer mTexCoordBuff_p;
    private Buffer mNormBuff_p;
    private Buffer mIndBuff_p;

    private int indicesNumber_p = 0;
    private int verticesNumber_p = 0;


    public Plane()
    {
        setVerts();
        setTexCoords();
        setNorms();
        setIndices();
    }


    private void setVerts()
    {
        double scale = 5.0;
        double[] TEAPOT_VERTS = {
                 5.0*scale, 3.0*scale, 0.0,
                 5.0*scale,-3.0*scale, 0.0,
                -5.0*scale,-3.0*scale, 0.0,
                -5.0*scale, 3.0*scale, 0.0
        };
        mVertBuff_p = fillBuffer(TEAPOT_VERTS);
        verticesNumber_p = TEAPOT_VERTS.length / 3;
    }


    private void setTexCoords()
    {
        double[] TEAPOT_TEX_COORDS = {
                1.0, 1.0, //
                1.0, 0.0, //
                0.0, 0.0, //
                0.0, 1.0, //
        };
        mTexCoordBuff_p = fillBuffer(TEAPOT_TEX_COORDS);

    }


    private void setNorms()
    {
        double[] TEAPOT_NORMS = {
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0,
                0.0, 0.0, 1.0,

        };
        mNormBuff_p = fillBuffer(TEAPOT_NORMS);
    }


    private void setIndices()
    {
        short[] TEAPOT_INDICES = {
                0, 3, 2, 0, 2, 1
        };
        mIndBuff_p = fillBuffer(TEAPOT_INDICES);
        indicesNumber_p = TEAPOT_INDICES.length;
    }


    public int getNumObjectIndex()
    {
        return indicesNumber_p;
    }


    @Override
    public int getNumObjectVertex()
    {
        return verticesNumber_p;
    }


    @Override
    public Buffer getBuffer(BUFFER_TYPE bufferType)
    {
        Buffer result = null;
        switch (bufferType)
        {
            case BUFFER_TYPE_VERTEX:
                result = mVertBuff_p;
                break;
            case BUFFER_TYPE_TEXTURE_COORD:
                result = mTexCoordBuff_p;
                break;
            case BUFFER_TYPE_NORMALS:
                result = mNormBuff_p;
                break;
            case BUFFER_TYPE_INDICES:
                result = mIndBuff_p;
            default:
                break;

        }

        return result;
    }

}
