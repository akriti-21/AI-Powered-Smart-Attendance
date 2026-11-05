package com.smartattendance.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Core
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
        private const val INPUT_SIZE = 640
        private const val DETECTION_THRESHOLD = 0.50f
        private const val NMS_THRESHOLD = 0.40f
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

    /**
     * Runs RetinaFace on a BGR face crop.
     * Returns all detections after InsightFace-style NMS.
     * Coordinates and landmarks are relative to the original crop.
     */
    fun detect(crop: Mat): List<RetinaFaceResult> {
        require(!crop.empty()) { "RetinaFace crop is empty" }
        require(crop.channels() == 3) {
            "RetinaFace input must have three BGR channels"
        }

        val originalWidth = crop.cols()
        val originalHeight = crop.rows()

        val imageRatio = originalHeight.toDouble() / originalWidth
        val modelRatio = 1.0

        val resizedWidth: Int
        val resizedHeight: Int

        if (imageRatio > modelRatio) {
            resizedHeight = INPUT_SIZE
            resizedWidth = (resizedHeight / imageRatio).toInt()
        } else {
            resizedWidth = INPUT_SIZE
            resizedHeight = (resizedWidth * imageRatio).toInt()
        }

        if (resizedWidth <= 0 || resizedHeight <= 0) {
            return emptyList()
        }

        val scale = resizedHeight.toFloat() / originalHeight

        val resized = Mat()
        val padded = Mat()
        val rgb = Mat()
        val floatImage = Mat()

        try {
            Imgproc.resize(
                crop,
                resized,
                Size(resizedWidth.toDouble(), resizedHeight.toDouble())
            )

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

            Imgproc.cvtColor(padded, rgb, Imgproc.COLOR_BGR2RGB)
            rgb.convertTo(
                floatImage,
                CvType.CV_32FC3,
                1.0 / 128.0,
                -127.5 / 128.0
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
            val inputTensor = OnnxTensor.createTensor(
                environment,
                FloatBuffer.wrap(inputData),
                longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            )

            try {
                val outputs = session.run(mapOf(inputName to inputTensor))

                try {
                    val decoded = ArrayList<Candidate>()

                    for (level in STRIDES.indices) {
                        val stride = STRIDES[level]
                        val scores = outputRows(outputs[level].value)
                        val boxes = outputRows(outputs[level + 3].value)
                        val landmarks = outputRows(outputs[level + 6].value)

                        val featureHeight = INPUT_SIZE / stride
                        val featureWidth = INPUT_SIZE / stride
                        val expectedCount =
                            featureHeight * featureWidth * NUM_ANCHORS

                        require(scores.size == expectedCount) {
                            "Unexpected RetinaFace score count at stride $stride"
                        }
                        require(boxes.size == expectedCount) {
                            "Unexpected RetinaFace box count at stride $stride"
                        }
                        require(landmarks.size == expectedCount) {
                            "Unexpected RetinaFace landmark count at stride $stride"
                        }

                        for (index in 0 until expectedCount) {
                            val score = scores[index][0]

                            if (!score.isFinite() ||
                                score < DETECTION_THRESHOLD
                            ) {
                                continue
                            }

                            val location = index / NUM_ANCHORS
                            val anchor = index % NUM_ANCHORS
                            val gridY = location / featureWidth
                            val gridX = location % featureWidth

                            val centerX = gridX * stride.toFloat()
                            val centerY = gridY * stride.toFloat()

                            val box = boxes[index]
                            val x1 = (centerX - box[0] * stride) / scale
                            val y1 = (centerY - box[1] * stride) / scale
                            val x2 = (centerX + box[2] * stride) / scale
                            val y2 = (centerY + box[3] * stride) / scale

                            val landmarkValues = landmarks[index]
                            val points = Array(5) { pointIndex ->
                                Point(
                                    ((
                                        centerX +
                                        landmarkValues[pointIndex * 2] * stride
                                    ) / scale).toDouble(),
                                    ((
                                        centerY +
                                        landmarkValues[pointIndex * 2 + 1] * stride
                                    ) / scale).toDouble()
                                )
                            }

                            if (
                                x1.isFinite() && y1.isFinite() &&
                                x2.isFinite() && y2.isFinite() &&
                                x2 > x1 && y2 > y1 &&
                                points.all {
                                    it.x.isFinite() && it.y.isFinite()
                                }
                            ) {
                                decoded.add(
                                    Candidate(
                                        x1, y1, x2, y2, score, points
                                    )
                                )
                            }
                        }
                    }

                    return applyNms(decoded).map {
                        RetinaFaceResult(
                            it.x1,
                            it.y1,
                            it.x2,
                            it.y2,
                            it.score,
                            it.landmarks
                        )
                    }
                } finally {
                    outputs.close()
                }
            } finally {
                inputTensor.close()
            }
        } finally {
            resized.release()
            padded.release()
            rgb.release()
            floatImage.release()
        }
    }

    /**
     * Selects the largest face, as FaceEngine.detect_faces does
     * when an expanded YOLO crop contains multiple faces.
     */
    fun detectLargest(crop: Mat): RetinaFaceResult? {
        return detect(crop).maxByOrNull { it.area }
    }

    private fun outputRows(value: Any): Array<FloatArray> {
        // The current det_10g.onnx model exports rank-two tensors:
        // [N, 1], [N, 4], and [N, 10].
        if (value is Array<*> && value.isNotEmpty()) {
            if (value[0] is FloatArray) {
                @Suppress("UNCHECKED_CAST")
                return value as Array<FloatArray>
            }

            // Also support a rank-three batched output: [1, N, K].
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

    private fun applyNms(
        candidates: List<Candidate>
    ): List<Candidate> {
        if (candidates.isEmpty()) return emptyList()

        val sorted = candidates.sortedByDescending { it.score }
        val kept = ArrayList<Candidate>()

        for (candidate in sorted) {
            var keep = true

            for (previous in kept) {
                if (calculateIou(candidate, previous) > NMS_THRESHOLD) {
                    keep = false
                    break
                }
            }

            if (keep) kept.add(candidate)
        }

        return kept
    }

    /**
     * InsightFace RetinaFace NMS uses inclusive pixel dimensions (+1).
     */
    private fun calculateIou(a: Candidate, b: Candidate): Float {
        val xx1 = max(a.x1, b.x1)
        val yy1 = max(a.y1, b.y1)
        val xx2 = min(a.x2, b.x2)
        val yy2 = min(a.y2, b.y2)

        val width = max(0f, xx2 - xx1 + 1f)
        val height = max(0f, yy2 - yy1 + 1f)
        val intersection = width * height

        val areaA =
            (a.x2 - a.x1 + 1f) * (a.y2 - a.y1 + 1f)
        val areaB =
            (b.x2 - b.x1 + 1f) * (b.y2 - b.y1 + 1f)

        val union = areaA + areaB - intersection

        return if (union > 0f) intersection / union else 0f
    }
}
