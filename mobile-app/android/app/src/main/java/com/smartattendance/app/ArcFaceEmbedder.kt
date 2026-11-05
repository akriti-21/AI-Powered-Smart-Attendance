package com.smartattendance.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.sqrt

class ArcFaceEmbedder(
    private val environment: OrtEnvironment,
    private val session: OrtSession
) {
    companion object {
        private const val INPUT_SIZE = 112
        private const val FEATURE_DIM = 512
        private const val INPUT_MEAN = 127.5f
        private const val INPUT_STD = 127.5f
    }

    fun extract(alignedFace: Mat): FloatArray {
        require(!alignedFace.empty()) { "Aligned face is empty" }
        require(
            alignedFace.rows() == INPUT_SIZE &&
            alignedFace.cols() == INPUT_SIZE
        ) {
            "ArcFace requires a 112x112 image"
        }
        require(alignedFace.channels() == 3) {
            "ArcFace input must have three BGR channels"
        }

        val rgb = Mat()
        val floatImage = Mat()

        try {
            Imgproc.cvtColor(
                alignedFace,
                rgb,
                Imgproc.COLOR_BGR2RGB
            )

            rgb.convertTo(
                floatImage,
                CvType.CV_32FC3,
                1.0 / INPUT_STD.toDouble(),
                -INPUT_MEAN.toDouble() / INPUT_STD.toDouble()
            )

            val planeSize = INPUT_SIZE * INPUT_SIZE
            val inputData = FloatArray(3 * planeSize)
            val rowData = FloatArray(INPUT_SIZE * 3)

            for (y in 0 until INPUT_SIZE) {
                floatImage.get(y, 0, rowData)

                for (x in 0 until INPUT_SIZE) {
                    val pixel = x * 3
                    val index = y * INPUT_SIZE + x

                    inputData[index] = rowData[pixel]
                    inputData[planeSize + index] = rowData[pixel + 1]
                    inputData[2 * planeSize + index] = rowData[pixel + 2]
                }
            }

            val inputName = session.inputNames.first()
            val tensor = OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(inputData),
                longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            )

            try {
                val outputs = session.run(mapOf(inputName to tensor))

                try {
                    val batch = outputs[0].value as Array<FloatArray>

                    require(batch.size == 1) {
                        "Unexpected ArcFace batch size"
                    }

                    return normalize(batch[0])
                } finally {
                    outputs.close()
                }
            } finally {
                tensor.close()
            }
        } finally {
            rgb.release()
            floatImage.release()
        }
    }

    private fun normalize(embedding: FloatArray): FloatArray {
        require(embedding.size == FEATURE_DIM) {
            "ArcFace embedding must contain $FEATURE_DIM values"
        }

        var squaredNorm = 0.0

        for (value in embedding) {
            require(value.isFinite()) {
                "ArcFace embedding contains invalid values"
            }
            squaredNorm += value.toDouble() * value.toDouble()
        }

        val norm = sqrt(squaredNorm)

        require(norm.isFinite() && norm > 0.0) {
            "ArcFace produced an invalid zero embedding"
        }

        return FloatArray(FEATURE_DIM) { index ->
            (embedding[index] / norm).toFloat()
        }
    }
}
