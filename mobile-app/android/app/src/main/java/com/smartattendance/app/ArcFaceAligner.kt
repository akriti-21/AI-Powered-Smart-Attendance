package com.smartattendance.app

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

object ArcFaceAligner {

    private val referenceLandmarks = arrayOf(
        Point(38.2946, 51.6963),
        Point(73.5318, 51.5014),
        Point(56.0252, 71.7366),
        Point(41.5493, 92.3655),
        Point(70.7299, 92.2041)
    )

    /**
     * Aligns a face crop using five InsightFace landmarks.
     *
     * Landmark order:
     * left eye, right eye, nose, left mouth corner, right mouth corner.
     *
     * Returns a 112x112 BGR Mat.
     * The caller is responsible for releasing the returned Mat.
     */
    fun alignFace(
        faceCrop: Mat,
        landmarks: Array<Point>
    ): Mat {
        require(!faceCrop.empty()) { "Face crop is empty" }
        require(landmarks.size == 5) { "Exactly five landmarks are required" }

        // Solve:
        // x' = a*x - b*y + tx
        // y' = b*x + a*y + ty
        val aMatrix = Mat(10, 4, CvType.CV_64F)
        val bMatrix = Mat(10, 1, CvType.CV_64F)
        val solution = Mat()

        try {
            for (i in 0 until 5) {
                val x = landmarks[i].x
                val y = landmarks[i].y
                val target = referenceLandmarks[i]

                aMatrix.put(
                    2 * i, 0,
                    x, -y, 1.0, 0.0
                )
                aMatrix.put(
                    2 * i + 1, 0,
                    y, x, 0.0, 1.0
                )

                bMatrix.put(2 * i, 0, target.x)
                bMatrix.put(2 * i + 1, 0, target.y)
            }

            val solved = Core.solve(
                aMatrix,
                bMatrix,
                solution,
                Core.DECOMP_SVD
            )

            check(solved) { "Could not calculate face alignment transform" }

            val a = solution.get(0, 0)[0]
            val b = solution.get(1, 0)[0]
            val tx = solution.get(2, 0)[0]
            val ty = solution.get(3, 0)[0]

            val transform = Mat(2, 3, CvType.CV_64F)
            try {
                transform.put(
                    0, 0,
                    a, -b, tx,
                    b, a, ty
                )

                val aligned = Mat()
                Imgproc.warpAffine(
                    faceCrop,
                    aligned,
                    transform,
                    Size(112.0, 112.0),
                    Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT
                )
                return aligned
            } finally {
                transform.release()
            }
        } finally {
            aMatrix.release()
            bMatrix.release()
            solution.release()
        }
    }
}
