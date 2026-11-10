
package com.smartattendance.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.util.Log

import com.facebook.react.bridge.*
import com.facebook.react.module.annotations.ReactModule

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

import java.io.File

@ReactModule(name = FaceInferenceModule.NAME)
class FaceInferenceModule(
    reactContext: ReactApplicationContext
) : ReactContextBaseJavaModule(reactContext) {

    companion object {
        const val NAME = "FaceInference"
        private const val MODEL_FOLDER = "models"
        private const val TAG = "FaceInference"

        private val MODEL_FILES = listOf(
            "yolov11n-face.onnx",
            "det_10g.onnx",
            "w600k_r50.onnx"
        )
    }

    private val environment = OrtEnvironment.getEnvironment()
    private val engineLock = Any()

    private var yoloSession: OrtSession? = null
    private var detectorSession: OrtSession? = null
    private var arcfaceSession: OrtSession? = null

    private var faceEngine: FaceRecognitionEngine? = null
    private var openCvReady = false

    override fun getName(): String = NAME

    // ==================================================
    // COPY OFFLINE MODELS TO LOCAL APP STORAGE
    // ==================================================

    private fun copyModel(
        context: Context,
        filename: String
    ): String {

        val directory = File(
            context.filesDir,
            MODEL_FOLDER
        )

        if (!directory.exists() && !directory.mkdirs()) {
            throw IllegalStateException(
                "Could not create model directory"
            )
        }

        val destination = File(directory, filename)

        if (!destination.exists() || destination.length() == 0L) {
            context.assets.open(
                "$MODEL_FOLDER/$filename"
            ).use { input ->

                destination.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }

        return destination.absolutePath
    }

    // ==================================================
    // CREATE ONNX SESSION
    // ==================================================

    private fun createSession(path: String): OrtSession {

        val options = OrtSession.SessionOptions()

        return try {
            environment.createSession(path, options)
        } finally {
            options.close()
        }
    }

    // ==================================================
    // CORRECT IMAGE ORIENTATION USING EXIF
    // ==================================================

    private fun correctImageOrientation(
        bitmap: Bitmap,
        uri: Uri
    ): Bitmap {

        val orientation = try {

            reactApplicationContext.contentResolver
                .openInputStream(uri)
                ?.use { input ->

                    ExifInterface(input).getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )
                } ?: ExifInterface.ORIENTATION_NORMAL

        } catch (e: Exception) {

            Log.e(
                TAG,
                "Could not read image orientation",
                e
            )

            ExifInterface.ORIENTATION_NORMAL
        }

        Log.d(TAG, "EXIF orientation: $orientation")

        val matrix = Matrix()

        when (orientation) {

            ExifInterface.ORIENTATION_ROTATE_90 ->
                matrix.postRotate(90f)

            ExifInterface.ORIENTATION_ROTATE_180 ->
                matrix.postRotate(180f)

            ExifInterface.ORIENTATION_ROTATE_270 ->
                matrix.postRotate(270f)

            ExifInterface.ORIENTATION_FLIP_HORIZONTAL ->
                matrix.preScale(-1f, 1f)

            ExifInterface.ORIENTATION_FLIP_VERTICAL ->
                matrix.preScale(1f, -1f)

            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1f, 1f)
            }

            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1f, 1f)
            }
        }

        if (matrix.isIdentity) {
            return bitmap
        }

        val correctedBitmap = Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )

        if (correctedBitmap !== bitmap) {
            bitmap.recycle()
        }

        return correctedBitmap
    }

    // ==================================================
    // LOAD ALL OFFLINE MODELS
    // ==================================================

    @ReactMethod
    fun loadModels(promise: Promise) {

        var newYolo: OrtSession? = null
        var newDetector: OrtSession? = null
        var newArcface: OrtSession? = null

        try {

            synchronized(engineLock) {

                if (!openCvReady) {

                    openCvReady = OpenCVLoader.initLocal()

                    if (!openCvReady) {
                        throw IllegalStateException(
                            "OpenCV initialization failed"
                        )
                    }
                }
            }

            val context = reactApplicationContext

            val yoloPath = copyModel(
                context,
                MODEL_FILES[0]
            )

            val detectorPath = copyModel(
                context,
                MODEL_FILES[1]
            )

            val arcfacePath = copyModel(
                context,
                MODEL_FILES[2]
            )

            newYolo = createSession(yoloPath)
            newDetector = createSession(detectorPath)
            newArcface = createSession(arcfacePath)

            val readyYolo = newYolo
                ?: throw IllegalStateException("YOLO model failed to load")

            val readyDetector = newDetector
                ?: throw IllegalStateException("RetinaFace model failed to load")

            val readyArcface = newArcface
                ?: throw IllegalStateException("ArcFace model failed to load")

            val newEngine = FaceRecognitionEngine(
                YoloFaceDetector(environment, readyYolo),
                RetinaFaceDetector(environment, readyDetector),
                ArcFaceEmbedder(environment, readyArcface)
            )

            synchronized(engineLock) {

                yoloSession?.close()
                detectorSession?.close()
                arcfaceSession?.close()

                yoloSession = readyYolo
                detectorSession = readyDetector
                arcfaceSession = readyArcface

                faceEngine = newEngine

                newYolo = null
                newDetector = null
                newArcface = null
            }

            val result = Arguments.createMap()

            result.putBoolean("success", true)
            result.putString(
                "message",
                "Offline models loaded successfully."
            )

            result.putString("yolo", "loaded")
            result.putString("faceDetector", "loaded")
            result.putString("arcface", "loaded")
            result.putBoolean("openCVReady", true)

            promise.resolve(result)

            Log.d(
                TAG,
                "All offline face recognition models loaded successfully"
            )

        } catch (e: Exception) {

            try {
                newYolo?.close()
                newDetector?.close()
                newArcface?.close()
            } catch (_: Exception) {
            }

            Log.e(TAG, "Model loading failed", e)

            promise.reject(
                "MODEL_LOAD_ERROR",
                e.message,
                e
            )
        }
    }

    // ==================================================
    // GET RUNTIME STATUS
    // ==================================================

    @ReactMethod
    fun getRuntimeStatus(promise: Promise) {

        try {

            val result = Arguments.createMap()

            synchronized(engineLock) {

                result.putString(
                    "runtime",
                    "ONNX Runtime"
                )

                result.putString(
                    "version",
                    "1.20.0"
                )

                result.putBoolean(
                    "available",
                    true
                )

                result.putBoolean(
                    "modelsLoaded",
                    faceEngine != null &&
                        yoloSession != null &&
                        detectorSession != null &&
                        arcfaceSession != null
                )

                result.putBoolean(
                    "openCVReady",
                    openCvReady
                )
            }

            promise.resolve(result)

        } catch (e: Exception) {

            promise.reject(
                "RUNTIME_ERROR",
                e.message,
                e
            )
        }
    }

    // ==================================================
    // EXTRACT FACES AND GENERATE EMBEDDINGS
    // ==================================================

    @ReactMethod
    fun extractFaces(
        imageUri: String,
        promise: Promise
    ) {

        Thread {

            var bitmap: Bitmap? = null

            val rgba = Mat()
            val bgr = Mat()

            try {

                Log.d(TAG, "--------------------------------")
                Log.d(TAG, "STARTING FACE EXTRACTION")
                Log.d(TAG, "Image URI: $imageUri")

                if (imageUri.isBlank()) {
                    throw IllegalArgumentException(
                        "Image URI is empty"
                    )
                }

                synchronized(engineLock) {

                    if (
                        faceEngine == null ||
                        !openCvReady
                    ) {
                        throw IllegalStateException(
                            "Load models before extracting faces"
                        )
                    }
                }

                val uri = Uri.parse(imageUri)

                // Decode local image.
                val decodedBitmap =
                    reactApplicationContext.contentResolver
                        .openInputStream(uri)
                        ?.use { stream ->

                            BitmapFactory.decodeStream(stream)

                        } ?: throw IllegalArgumentException(
                            "Could not decode image URI: $imageUri"
                        )

                Log.d(
                    TAG,
                    "Decoded bitmap: ${decodedBitmap.width} x ${decodedBitmap.height}"
                )

                Log.d(
                    TAG,
                    "Decoded bitmap config: ${decodedBitmap.config}"
                )

                // Correct image orientation.
                val orientedBitmap =
                    correctImageOrientation(
                        decodedBitmap,
                        uri
                    )

                bitmap = orientedBitmap

                Log.d(
                    TAG,
                    "Corrected bitmap width: ${orientedBitmap.width}"
                )

                Log.d(
                    TAG,
                    "Corrected bitmap height: ${orientedBitmap.height}"
                )

                Log.d(
                    TAG,
                    "Corrected bitmap config: ${orientedBitmap.config}"
                )

                // Convert bitmap to OpenCV Mat.
                Utils.bitmapToMat(
                    orientedBitmap,
                    rgba
                )

                if (rgba.empty()) {
                    throw IllegalStateException(
                        "Image conversion failed: RGBA Mat is empty"
                    )
                }

                Log.d(
                    TAG,
                    "RGBA Mat width: ${rgba.cols()}"
                )

                Log.d(
                    TAG,
                    "RGBA Mat height: ${rgba.rows()}"
                )

                Log.d(
                    TAG,
                    "RGBA Mat channels: ${rgba.channels()}"
                )

                // Convert RGBA to BGR.
                Imgproc.cvtColor(
                    rgba,
                    bgr,
                    Imgproc.COLOR_RGBA2BGR
                )

                if (bgr.empty()) {
                    throw IllegalStateException(
                        "BGR conversion failed"
                    )
                }

                // DIAGNOSTIC LOGS
                Log.d(TAG, "Image URI: $imageUri")
                Log.d(TAG, "Bitmap width: ${orientedBitmap.width}")
                Log.d(TAG, "Bitmap height: ${orientedBitmap.height}")
                Log.d(TAG, "Bitmap config: ${orientedBitmap.config}")
                Log.d(TAG, "OpenCV Mat width: ${bgr.cols()}")
                Log.d(TAG, "OpenCV Mat height: ${bgr.rows()}")
                Log.d(TAG, "OpenCV Mat channels: ${bgr.channels()}")
                Log.d(TAG, "OpenCV Mat empty: ${bgr.empty()}")

                Log.d(
                    TAG,
                    "Starting YOLO + RetinaFace + ArcFace pipeline"
                )

                // Run complete offline recognition pipeline.
                val features = synchronized(engineLock) {

                    val engine = faceEngine
                        ?: throw IllegalStateException(
                            "Face recognition engine is not initialized"
                        )

                    engine.extractFaces(bgr)
                }

                Log.d(
                    TAG,
                    "Face extraction completed. Faces: ${features.size}"
                )

                // Prepare React Native response.
                val result = Arguments.createMap()
                val facesArray = Arguments.createArray()

                for (feature in features) {

                    val faceMap = Arguments.createMap()

                    // Bounding box.
                    val box = Arguments.createMap()

                    box.putInt("x1", feature.bbox.x1)
                    box.putInt("y1", feature.bbox.y1)
                    box.putInt("x2", feature.bbox.x2)
                    box.putInt("y2", feature.bbox.y2)

                    faceMap.putMap("bbox", box)

                    faceMap.putDouble(
                        "confidence",
                        feature.bbox.confidence.toDouble()
                    )

                    // ArcFace embedding.
                    val embeddingArray = Arguments.createArray()

                    for (value in feature.embedding) {
                        embeddingArray.pushDouble(
                            value.toDouble()
                        )
                    }

                    faceMap.putArray(
                        "embedding",
                        embeddingArray
                    )

                    facesArray.pushMap(faceMap)
                }

                result.putBoolean("success", true)
                result.putInt("faceCount", features.size)
                result.putArray("faces", facesArray)

                Log.d(
                    TAG,
                    "Returning ${features.size} detected faces to React Native"
                )

                promise.resolve(result)

            } catch (e: Exception) {

                Log.e(
                    TAG,
                    "Face extraction failed",
                    e
                )

                promise.reject(
                    "FACE_EXTRACTION_ERROR",
                    e.message,
                    e
                )

            } finally {

                bitmap?.recycle()

                rgba.release()
                bgr.release()

                Log.d(TAG, "Face extraction resources released")
            }

        }.start()
    }

    // ==================================================
    // CLEAN UP NATIVE RESOURCES
    // ==================================================

    override fun invalidate() {

        synchronized(engineLock) {

            faceEngine = null

            try {

                yoloSession?.close()
                detectorSession?.close()
                arcfaceSession?.close()

            } catch (e: Exception) {

                Log.e(
                    TAG,
                    "Error closing ONNX sessions",
                    e
                )
            }

            yoloSession = null
            detectorSession = null
            arcfaceSession = null
        }

        super.invalidate()
    }
}
