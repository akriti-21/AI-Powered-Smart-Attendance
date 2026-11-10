
package com.smartattendance.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

import android.util.Log

import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

import java.nio.FloatBuffer

import kotlin.math.max
import kotlin.math.min

data class RetinaFaceResult(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val landmarks: Array<Point>
) {
    val area: Float
        get() = max(0f, x2 - x1) * max(0f, y2 - y1)
}

class RetinaFaceDetector(
    private val environment: OrtEnvironment,
    private val session: OrtSession
) {

    companion object {
        private const val TAG = "RetinaFace"

        private const val INPUT_SIZE = 640
        private const val DETECTION_THRESHOLD = 0.50f
        private const val NMS_THRESHOLD = 0.40f

        private const val INPUT_MEAN = 127.5
        private const val INPUT_STD = 128.0

        private val STRIDES = intArrayOf(8, 16, 32)
        private const val NUM_ANCHORS = 2
    }

    private data class Candidate(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val landmarks: Array<Point>
    )

    // ==================================================
    // DETECT FACES
    // ==================================================

    fun detect(crop: Mat): List<RetinaFaceResult> {

        require(!crop.empty()) {
            "RetinaFace crop is empty"
        }

        require(crop.channels() == 3) {
            "RetinaFace input must have three BGR channels"
        }

        val originalWidth = crop.cols()
        val originalHeight = crop.rows()

        Log.d(
            TAG,
            "Starting detection. Crop: ${originalWidth}x${originalHeight}"
        )

        val scale = min(
            INPUT_SIZE.toFloat() / originalWidth,
            INPUT_SIZE.toFloat() / originalHeight
        )

        val resizedWidth = (originalWidth * scale).toInt()
        val resizedHeight = (originalHeight * scale).toInt()

        if (resizedWidth <= 0 || resizedHeight <= 0) {
            Log.w(TAG, "Invalid resized dimensions")
            return emptyList()
        }

        Log.d(
            TAG,
            "Resized dimensions: ${resizedWidth}x${resizedHeight}, scale=$scale"
        )

        val resized = Mat()
        val padded = Mat()
        val floatImage = Mat()

        try {

            // Resize image.

            Imgproc.resize(
                crop,
                resized,
                Size(
                    resizedWidth.toDouble(),
                    resizedHeight.toDouble()
                ),
                0.0,
                0.0,
                Imgproc.INTER_LINEAR
            )

            // SCRFD preprocessing:
            // Preserve BGR order.
            // Pad bottom and right with zeros.

            padded.create(
                INPUT_SIZE,
                INPUT_SIZE,
                CvType.CV_8UC3
            )

            padded.setTo(Scalar(0.0, 0.0, 0.0))

            val roi = padded.submat(
                0,
                resizedHeight,
                0,
                resizedWidth
            )

            try {
                resized.copyTo(roi)
            } finally {
                roi.release()
            }

            // Normalize:
            // (pixel - 127.5) / 128

            padded.convertTo(
                floatImage,
                CvType.CV_32FC3,
                1.0 / INPUT_STD,
                -INPUT_MEAN / INPUT_STD
            )

            val planeSize = INPUT_SIZE * INPUT_SIZE

            val inputData = FloatArray(3 * planeSize)
            val rowData = FloatArray(INPUT_SIZE * 3)

            for (y in 0 until INPUT_SIZE) {

                floatImage.get(y, 0, rowData)

                for (x in 0 until INPUT_SIZE) {

                    val pixel = x * 3
                    val index = y * INPUT_SIZE + x

                    // BGR channel order.

                    inputData[index] = rowData[pixel]

                    inputData[planeSize + index] =
                        rowData[pixel + 1]

                    inputData[2 * planeSize + index] =
                        rowData[pixel + 2]
                }
            }

            Log.d(
                TAG,
                "Input tensor prepared: [1, 3, $INPUT_SIZE, $INPUT_SIZE]"
            )

            // Create ONNX input tensor.

            val inputName = session.inputNames.first()

            val inputTensor = OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(inputData),
                longArrayOf(
                    1,
                    3,
                    INPUT_SIZE.toLong(),
                    INPUT_SIZE.toLong()
                )
            )

            try {

                Log.d(TAG, "Running RetinaFace ONNX inference")

                val outputs = session.run(
                    mapOf(inputName to inputTensor)
                )

                try {

                    require(outputs.size() >= 9) {
                        "Expected 9 RetinaFace outputs, got ${outputs.size()}"
                    }

                    Log.d(
                        TAG,
                        "ONNX outputs received: ${outputs.size()}"
                    )

                    val decoded = ArrayList<Candidate>()

                    var totalPredictions = 0
                    var finiteScores = 0
                    var aboveThreshold = 0
                    var invalidBoxes = 0
                    var invalidLandmarks = 0
                    var maxScore = 0f

                    // Decode predictions for strides 8, 16, 32.

                    for (level in STRIDES.indices) {

                        val stride = STRIDES[level]

                        val scores = outputRows(
                            outputs[level].value
                        )

                        val boxes = outputRows(
                            outputs[level + 3].value
                        )

                        val landmarks = outputRows(
                            outputs[level + 6].value
                        )

                        val featureHeight = INPUT_SIZE / stride
                        val featureWidth = INPUT_SIZE / stride

                        val expectedCount =
                            featureHeight *
                            featureWidth *
                            NUM_ANCHORS

                        Log.d(
                            TAG,
                            "Stride $stride: scores=${scores.size}, " +
                                "boxes=${boxes.size}, " +
                                "landmarks=${landmarks.size}"
                        )

                        require(scores.size == expectedCount) {
                            "Unexpected score count at stride $stride: ${scores.size}"
                        }

                        require(boxes.size == expectedCount) {
                            "Unexpected box count at stride $stride: ${boxes.size}"
                        }

                        require(landmarks.size == expectedCount) {
                            "Unexpected landmark count at stride $stride: ${landmarks.size}"
                        }

                        for (index in 0 until expectedCount) {

                            totalPredictions++

                            val scoreRow = scores[index]

                            if (scoreRow.isEmpty()) {
                                continue
                            }

                            val score = scoreRow[0]

                            if (score.isFinite()) {

                                finiteScores++

                                maxScore = max(
                                    maxScore,
                                    score
                                )
                            }

                            if (
                                !score.isFinite() ||
                                score < DETECTION_THRESHOLD
                            ) {
                                continue
                            }

                            aboveThreshold++

                            val location = index / NUM_ANCHORS

                            val gridY = location / featureWidth
                            val gridX = location % featureWidth

                            val centerX = gridX * stride.toFloat()
                            val centerY = gridY * stride.toFloat()

                            val box = boxes[index]

                            if (box.size < 4) {
                                invalidBoxes++
                                continue
                            }

                            // Decode bounding box.

                            val x1 =
                                (centerX - box[0] * stride) / scale

                            val y1 =
                                (centerY - box[1] * stride) / scale

                            val x2 =
                                (centerX + box[2] * stride) / scale

                            val y2 =
                                (centerY + box[3] * stride) / scale

                            val landmarkValues = landmarks[index]

                            if (landmarkValues.size < 10) {
                                invalidLandmarks++
                                continue
                            }

                            // Decode five facial landmarks.

                            val points = Array(5) { pointIndex ->

                                Point(
                                    (
                                        centerX +
                                        landmarkValues[pointIndex * 2] *
                                        stride
                                    ).toDouble() / scale,

                                    (
                                        centerY +
                                        landmarkValues[pointIndex * 2 + 1] *
                                        stride
                                    ).toDouble() / scale
                                )
                            }

                            if (
                                x1.isFinite() &&
                                y1.isFinite() &&
                                x2.isFinite() &&
                                y2.isFinite() &&
                                x2 > x1 &&
                                y2 > y1 &&
                                points.all {
                                    it.x.isFinite() && it.y.isFinite()
                                }
                            ) {

                                decoded.add(
                                    Candidate(
                                        x1,
                                        y1,
                                        x2,
                                        y2,
                                        score,
                                        points
                                    )
                                )

                            } else {

                                invalidBoxes++
                            }
                        }
                    }

                    // Diagnostic information.

                    Log.d(
                        TAG,
                        "--------------------------------"
                    )

                    Log.d(
                        TAG,
                        "Crop dimensions: ${originalWidth}x${originalHeight}"
                    )

                    Log.d(
                        TAG,
                        "Total predictions: $totalPredictions"
                    )

                    Log.d(
                        TAG,
                        "Finite scores: $finiteScores"
                    )

                    Log.d(
                        TAG,
                        "Scores above threshold: $aboveThreshold"
                    )

                    Log.d(
                        TAG,
                        "Maximum confidence: $maxScore"
                    )

                    Log.d(
                        TAG,
                        "Invalid boxes: $invalidBoxes"
                    )

                    Log.d(
                        TAG,
                        "Invalid landmarks: $invalidLandmarks"
                    )

                    Log.d(
                        TAG,
                        "Decoded valid candidates: ${decoded.size}"
                    )

                    Log.d(
                        TAG,
                        "--------------------------------"
                    )

                    // Apply NMS.

                    val results = applyNms(decoded).map {

                        RetinaFaceResult(
                            it.x1,
                            it.y1,
                            it.x2,
                            it.y2,
                            it.score,
                            it.landmarks
                        )
                    }

                    Log.d(
                        TAG,
                        "Faces after NMS: ${results.size}"
                    )

                    return results

                } finally {
                    outputs.close()
                }

            } finally {
                inputTensor.close()
            }

        } finally {

            resized.release()
            padded.release()
            floatImage.release()
        }
    }

    // ==================================================
    // DETECT LARGEST FACE
    // ==================================================

    fun detectLargest(crop: Mat): RetinaFaceResult? {

        val results = detect(crop)

        val largest = results.maxByOrNull {
            it.area
        }

        Log.d(
            TAG,
            "Largest face result: ${largest != null}"
        )

        return largest
    }

    // ==================================================
    // CONVERT ONNX OUTPUTS
    // ==================================================

    private fun outputRows(value: Any): Array<FloatArray> {

        if (value is Array<*> && value.isNotEmpty()) {

            if (value[0] is FloatArray) {

                @Suppress("UNCHECKED_CAST")
                return value as Array<FloatArray>
            }

            if (value[0] is Array<*>) {

                @Suppress("UNCHECKED_CAST")
                val batch = value as Array<Array<FloatArray>>

                require(batch.size == 1) {
                    "Expected RetinaFace batch size 1"
                }

                return batch[0]
            }
        }

        throw IllegalArgumentException(
            "Unsupported RetinaFace output type: ${value.javaClass.name}"
        )
    }

    // ==================================================
    // NON-MAXIMUM SUPPRESSION
    // ==================================================

    private fun applyNms(
        candidates: List<Candidate>
    ): List<Candidate> {

        if (candidates.isEmpty()) {
            return emptyList()
        }

        val sorted = candidates.sortedByDescending {
            it.score
        }

        val kept = ArrayList<Candidate>()

        for (candidate in sorted) {

            var keep = true

            for (previous in kept) {

                if (
                    calculateIou(candidate, previous) >
                    NMS_THRESHOLD
                ) {
                    keep = false
                    break
                }
            }

            if (keep) {
                kept.add(candidate)
            }
        }

        return kept
    }

    // ==================================================
    // IOU CALCULATION
    // ==================================================

    private fun calculateIou(
        a: Candidate,
        b: Candidate
    ): Float {

        val xx1 = max(a.x1, b.x1)
        val yy1 = max(a.y1, b.y1)

        val xx2 = min(a.x2, b.x2)
        val yy2 = min(a.y2, b.y2)

        val width = max(0f, xx2 - xx1 + 1f)
        val height = max(0f, yy2 - yy1 + 1f)

        val intersection = width * height

        val areaA =
            (a.x2 - a.x1 + 1f) *
            (a.y2 - a.y1 + 1f)

        val areaB =
            (b.x2 - b.x1 + 1f) *
            (b.y2 - b.y1 + 1f)

        val union = areaA + areaB - intersection

        return if (union > 0f) {
            intersection / union
        } else {
            0f
        }
    }
}
