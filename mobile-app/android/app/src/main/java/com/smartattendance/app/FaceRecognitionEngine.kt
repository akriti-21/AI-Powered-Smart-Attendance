package com.smartattendance.app

import org.opencv.core.Mat

data class RecognizedFaceFeature(
    val bbox: FaceBox,
    val embedding: FloatArray
)

class FaceRecognitionEngine(
    private val yolo: YoloFaceDetector,
    private val retinaFace: RetinaFaceDetector,
    private val arcFace: ArcFaceEmbedder
) {
    companion object {
        private const val CROP_EXPANSION = 0.25f
    }

    fun extractFaces(image: Mat): List<RecognizedFaceFeature> {
        require(!image.empty()) { "Input image is empty" }
        require(image.channels() == 3) {
            "Input image must be a three-channel BGR image"
        }

        val imageWidth = image.cols()
        val imageHeight = image.rows()
        val yoloFaces = yolo.detect(image)
        val results = ArrayList<RecognizedFaceFeature>()

        for (face in yoloFaces) {
            val boxWidth = face.x2 - face.x1
            val boxHeight = face.y2 - face.y1

            val expandX = (boxWidth * CROP_EXPANSION).toInt()
            val expandY = (boxHeight * CROP_EXPANSION).toInt()

            val cropX1 = (face.x1 - expandX).coerceAtLeast(0)
            val cropY1 = (face.y1 - expandY).coerceAtLeast(0)
            val cropX2 = (face.x2 + expandX).coerceAtMost(imageWidth)
            val cropY2 = (face.y2 + expandY).coerceAtMost(imageHeight)

            if (cropX2 <= cropX1 || cropY2 <= cropY1) {
                continue
            }

            val crop = image.submat(
                cropY1,
                cropY2,
                cropX1,
                cropX2
            )

            try {
                val retinaResult = try {
                    retinaFace.detectLargest(crop)
                } catch (_: Exception) {
                    null
                } ?: continue

                val aligned = ArcFaceAligner.alignFace(
                    crop,
                    retinaResult.landmarks
                )

                try {
                    val embedding = arcFace.extract(aligned)

                    results.add(
                        RecognizedFaceFeature(
                            bbox = face,
                            embedding = embedding
                        )
                    )
                } finally {
                    aligned.release()
                }
            } finally {
                crop.release()
            }
        }

        return results
    }
}
