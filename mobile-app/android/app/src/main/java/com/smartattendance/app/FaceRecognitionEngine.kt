
package com.smartattendance.app

import android.util.Log
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
        private const val TAG = "FaceRecognition"

        private const val CROP_EXPANSION = 0.25f

        private const val RETINAFACE_CONFIDENCE_THRESHOLD = 0.40f

        private const val DUPLICATE_IOU_THRESHOLD = 0.40f

        private const val EMBEDDING_SIZE = 512
    }

    // ---------------------------------------------------------
    // MAIN FACE EXTRACTION
    // ---------------------------------------------------------

    fun extractFaces(image: Mat): List<RecognizedFaceFeature> {

        require(!image.empty()) {
            "Input image is empty"
        }

        require(image.channels() == 3) {
            "Input image must be a three-channel BGR image"
        }

        val imageWidth = image.cols()
        val imageHeight = image.rows()

        val results = ArrayList<RecognizedFaceFeature>()

        Log.d(
            TAG,
            "Starting face extraction: ${imageWidth}x${imageHeight}"
        )

        // -----------------------------------------------------
        // STEP 1: YOLO DETECTION
        // -----------------------------------------------------

        val yoloFaces = try {
            yolo.detect(image)
        } catch (e: Exception) {
            Log.e(TAG, "YOLO detection failed", e)
            emptyList()
        }

        Log.d(TAG, "YOLO detected ${yoloFaces.size} faces")

        // -----------------------------------------------------
        // STEP 2: PROCESS YOLO DETECTIONS
        // -----------------------------------------------------

        yoloFaces.forEachIndexed { index, face ->

            Log.d(
                TAG,
                "YOLO FACE $index: " +
                    "x1=${face.x1}, y1=${face.y1}, " +
                    "x2=${face.x2}, y2=${face.y2}, " +
                    "confidence=${face.confidence}"
            )

            val boxWidth = face.x2 - face.x1
            val boxHeight = face.y2 - face.y1

            if (boxWidth <= 0 || boxHeight <= 0) {
                Log.w(TAG, "YOLO face $index has invalid dimensions")
                return@forEachIndexed
            }

            val expandX = (boxWidth * CROP_EXPANSION).toInt()
            val expandY = (boxHeight * CROP_EXPANSION).toInt()

            val cropX1 = (face.x1 - expandX).coerceAtLeast(0)
            val cropY1 = (face.y1 - expandY).coerceAtLeast(0)

            val cropX2 = (face.x2 + expandX).coerceAtMost(imageWidth)
            val cropY2 = (face.y2 + expandY).coerceAtMost(imageHeight)

            if (cropX2 <= cropX1 || cropY2 <= cropY1) {
                Log.w(TAG, "YOLO face $index has invalid crop")
                return@forEachIndexed
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
                } catch (e: Exception) {
                    Log.e(TAG, "RetinaFace crop detection failed: $index", e)
                    null
                }

                if (retinaResult == null) {
                    Log.w(
                        TAG,
                        "YOLO face $index: RetinaFace returned null"
                    )
                    return@forEachIndexed
                }

                val aligned = try {
                    ArcFaceAligner.alignFace(
                        crop,
                        retinaResult.landmarks
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Alignment failed for YOLO face $index", e)
                    null
                }

                if (aligned == null) {
                    return@forEachIndexed
                }

                try {

                    val embedding = arcFace.extract(aligned)

                    if (!isValidEmbedding(embedding)) {
                        Log.e(
                            TAG,
                            "YOLO face $index produced invalid embedding"
                        )
                        return@forEachIndexed
                    }

                    results.add(
                        RecognizedFaceFeature(
                            bbox = face,
                            embedding = embedding
                        )
                    )

                    Log.d(
                        TAG,
                        "YOLO face $index embedding generated"
                    )

                } catch (e: Exception) {
                    Log.e(
                        TAG,
                        "YOLO face $index embedding extraction failed",
                        e
                    )
                } finally {
                    aligned.release()
                }

            } catch (e: Exception) {
                Log.e(
                    TAG,
                    "YOLO face $index processing failed",
                    e
                )
            } finally {
                crop.release()
            }
        }

        // -----------------------------------------------------
        // STEP 3: FULL-IMAGE RETINAFACE DETECTION
        // Always runs, even when YOLO found faces.
        // -----------------------------------------------------

        Log.d(TAG, "Starting full-image RetinaFace detection")

        try {

            val directFaces = retinaFace.detect(image)

            Log.d(
                TAG,
                "Full-image RetinaFace detected ${directFaces.size} faces"
            )

            directFaces.forEachIndexed { index, face ->

                Log.d(
                    TAG,
                    "RETINAFACE FACE $index: " +
                        "confidence=${face.confidence}"
                )

                if (
                    face.confidence <
                    RETINAFACE_CONFIDENCE_THRESHOLD
                ) {
                    Log.d(
                        TAG,
                        "RetinaFace face $index rejected: low confidence"
                    )
                    return@forEachIndexed
                }

                val x1 = face.x1.toInt().coerceIn(0, imageWidth)
                val y1 = face.y1.toInt().coerceIn(0, imageHeight)
                val x2 = face.x2.toInt().coerceIn(0, imageWidth)
                val y2 = face.y2.toInt().coerceIn(0, imageHeight)

                if (x2 <= x1 || y2 <= y1) {
                    Log.w(
                        TAG,
                        "RetinaFace face $index has invalid bounding box"
                    )
                    return@forEachIndexed
                }

                val candidateBox = FaceBox(
                    x1,
                    y1,
                    x2,
                    y2,
                    face.confidence
                )

                // Avoid duplicate detections.
                val duplicate = results.any { existing ->

                    calculateIoU(
                        existing.bbox,
                        candidateBox
                    ) > DUPLICATE_IOU_THRESHOLD
                }

                if (duplicate) {
                    Log.d(
                        TAG,
                        "RetinaFace face $index skipped: duplicate"
                    )
                    return@forEachIndexed
                }

                // Align directly from the original image.
                val aligned = try {

                    ArcFaceAligner.alignFace(
                        image,
                        face.landmarks
                    )

                } catch (e: Exception) {

                    Log.e(
                        TAG,
                        "RetinaFace face $index alignment failed",
                        e
                    )

                    null
                }

                if (aligned == null) {
                    return@forEachIndexed
                }

                try {

                    val embedding = arcFace.extract(aligned)

                    if (!isValidEmbedding(embedding)) {
                        Log.e(
                            TAG,
                            "RetinaFace face $index has invalid embedding"
                        )
                        return@forEachIndexed
                    }

                    results.add(
                        RecognizedFaceFeature(
                            bbox = candidateBox,
                            embedding = embedding
                        )
                    )

                    Log.d(
                        TAG,
                        "RetinaFace face $index embedding generated"
                    )

                } catch (e: Exception) {

                    Log.e(
                        TAG,
                        "RetinaFace face $index embedding failed",
                        e
                    )

                } finally {
                    aligned.release()
                }
            }

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Full-image RetinaFace detection failed",
                e
            )
        }

        // -----------------------------------------------------
        // STEP 4: FINAL RESULT
        // -----------------------------------------------------

        Log.d(
            TAG,
            "YOLO detections: ${yoloFaces.size}"
        )

        Log.d(
            TAG,
            "Final unique face embeddings: ${results.size}"
        )

        return results
    }

    // ---------------------------------------------------------
    // EMBEDDING VALIDATION
    // ---------------------------------------------------------

    private fun isValidEmbedding(
        embedding: FloatArray
    ): Boolean {

        return embedding.size == EMBEDDING_SIZE &&
            embedding.all { it.isFinite() }
    }

    // ---------------------------------------------------------
    // INTERSECTION OVER UNION
    // Used to avoid duplicate detections.
    // ---------------------------------------------------------

    private fun calculateIoU(
        a: FaceBox,
        b: FaceBox
    ): Float {

        val intersectionX1 = maxOf(a.x1, b.x1)
        val intersectionY1 = maxOf(a.y1, b.y1)

        val intersectionX2 = minOf(a.x2, b.x2)
        val intersectionY2 = minOf(a.y2, b.y2)

        val intersectionWidth =
            maxOf(0, intersectionX2 - intersectionX1)

        val intersectionHeight =
            maxOf(0, intersectionY2 - intersectionY1)

        val intersectionArea =
            intersectionWidth.toFloat() *
                intersectionHeight.toFloat()

        val areaA =
            (a.x2 - a.x1).toFloat() *
                (a.y2 - a.y1).toFloat()

        val areaB =
            (b.x2 - b.x1).toFloat() *
                (b.y2 - b.y1).toFloat()

        val unionArea =
            areaA + areaB - intersectionArea

        return if (unionArea <= 0f) {
            0f
        } else {
            intersectionArea / unionArea
        }
    }
}
