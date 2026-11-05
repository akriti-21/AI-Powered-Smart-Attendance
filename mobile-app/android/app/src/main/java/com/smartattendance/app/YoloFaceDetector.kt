package com.smartattendance.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class FaceBox(
    val x1: Int,
    val y1: Int,
    val x2: Int,
    val y2: Int,
    val confidence: Float
)

class YoloFaceDetector(
    private val environment: OrtEnvironment,
    private val session: OrtSession
) {

    companion object {
        private const val INPUT_SIZE = 1600
        private const val CONFIDENCE_THRESHOLD = 0.15f
        private const val IOU_THRESHOLD = 0.70f
        private const val MAX_DETECTIONS = 100
        private const val MAX_NMS_CANDIDATES = 30000
    }

    private data class Candidate(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val confidence: Float
    )

    fun detect(image: Mat): List<FaceBox> {
        require(!image.empty()) { "Input image is empty" }
        require(image.channels() == 3) {
            "Input image must have three BGR channels"
        }

        val originalWidth = image.cols()
        val originalHeight = image.rows()

        val scale = min(
            INPUT_SIZE.toDouble() / originalHeight,
            INPUT_SIZE.toDouble() / originalWidth
        )

        val resizedWidth = Math.rint(originalWidth * scale).toInt()
        val resizedHeight = Math.rint(originalHeight * scale).toInt()

        val horizontalPadding = INPUT_SIZE - resizedWidth
        val verticalPadding = INPUT_SIZE - resizedHeight

        val left = Math.rint(horizontalPadding / 2.0 - 0.1).toInt()
        val right = Math.rint(horizontalPadding / 2.0 + 0.1).toInt()
        val top = Math.rint(verticalPadding / 2.0 - 0.1).toInt()
        val bottom = Math.rint(verticalPadding / 2.0 + 0.1).toInt()

        val resized = Mat()
        val padded = Mat()
        val rgb = Mat()
        val floatImage = Mat()

        try {
            Imgproc.resize(
                image,
                resized,
                Size(resizedWidth.toDouble(), resizedHeight.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_LINEAR
            )

            Core.copyMakeBorder(
                resized,
                padded,
                top,
                bottom,
                left,
                right,
                Core.BORDER_CONSTANT,
                Scalar(114.0, 114.0, 114.0)
            )

            Imgproc.cvtColor(padded, rgb, Imgproc.COLOR_BGR2RGB)
            rgb.convertTo(floatImage, CvType.CV_32FC3, 1.0 / 255.0)

            val planeSize = INPUT_SIZE * INPUT_SIZE
            val inputData = FloatArray(3 * planeSize)
            val rowData = FloatArray(INPUT_SIZE * 3)

            for (y in 0 until INPUT_SIZE) {
                floatImage.get(y, 0, rowData)

                for (x in 0 until INPUT_SIZE) {
                    val pixelIndex = x * 3
                    val index = y * INPUT_SIZE + x

                    inputData[index] = rowData[pixelIndex]
                    inputData[planeSize + index] = rowData[pixelIndex + 1]
                    inputData[2 * planeSize + index] = rowData[pixelIndex + 2]
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
                    val output = outputs[0].value as Array<Array<FloatArray>>

                    require(output.size == 1 && output[0].size == 5) {
                        "Unexpected YOLO output shape"
                    }
                    require(output[0][0].size == 52500) {
                        "Unexpected YOLO prediction count"
                    }

                    val candidates = decode(output[0])

                    return applyNms(candidates)
                        .take(MAX_DETECTIONS)
                        .mapNotNull { candidate ->
                            val x1 = (
                                (candidate.x1 - left) / scale
                            ).toInt().coerceIn(0, originalWidth)

                            val y1 = (
                                (candidate.y1 - top) / scale
                            ).toInt().coerceIn(0, originalHeight)

                            val x2 = (
                                (candidate.x2 - left) / scale
                            ).toInt().coerceIn(0, originalWidth)

                            val y2 = (
                                (candidate.y2 - top) / scale
                            ).toInt().coerceIn(0, originalHeight)

                            if (x2 <= x1 || y2 <= y1) {
                                null
                            } else {
                                FaceBox(
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                    candidate.confidence
                                )
                            }
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

    private fun decode(output: Array<FloatArray>): List<Candidate> {
        val count = output[0].size
        val candidates = ArrayList<Candidate>()

        for (i in 0 until count) {
            val confidence = output[4][i]

            if (!confidence.isFinite() ||
                confidence <= CONFIDENCE_THRESHOLD
            ) {
                continue
            }

            val centerX = output[0][i]
            val centerY = output[1][i]
            val width = output[2][i]
            val height = output[3][i]

            if (!centerX.isFinite() ||
                !centerY.isFinite() ||
                !width.isFinite() ||
                !height.isFinite() ||
                width <= 0f ||
                height <= 0f
            ) {
                continue
            }

            candidates.add(
                Candidate(
                    centerX - width / 2f,
                    centerY - height / 2f,
                    centerX + width / 2f,
                    centerY + height / 2f,
                    confidence
                )
            )
        }

        return if (candidates.size > MAX_NMS_CANDIDATES) {
            candidates.sortedByDescending { it.confidence }
                .take(MAX_NMS_CANDIDATES)
        } else {
            candidates
        }
    }

    private fun applyNms(
        candidates: List<Candidate>
    ): List<Candidate> {

        if (candidates.isEmpty()) return emptyList()

        val sorted = candidates.sortedByDescending { it.confidence }
        val kept = ArrayList<Candidate>()

        for (candidate in sorted) {
            var shouldKeep = true

            for (previous in kept) {
                if (calculateIou(candidate, previous) > IOU_THRESHOLD) {
                    shouldKeep = false
                    break
                }
            }

            if (shouldKeep) {
                kept.add(candidate)

                if (kept.size == MAX_DETECTIONS) {
                    break
                }
            }
        }

        return kept
    }

    private fun calculateIou(a: Candidate, b: Candidate): Float {
        val intersectionX1 = max(a.x1, b.x1)
        val intersectionY1 = max(a.y1, b.y1)
        val intersectionX2 = min(a.x2, b.x2)
        val intersectionY2 = min(a.y2, b.y2)

        val intersectionWidth = max(0f, intersectionX2 - intersectionX1)
        val intersectionHeight = max(0f, intersectionY2 - intersectionY1)

        val intersection = intersectionWidth * intersectionHeight

        if (intersection == 0f) return 0f

        val areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
        val areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
        val union = areaA + areaB - intersection

        return if (union > 0f) intersection / union else 0f
    }
}
