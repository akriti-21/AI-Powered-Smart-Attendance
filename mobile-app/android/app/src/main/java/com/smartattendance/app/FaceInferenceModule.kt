package com.smartattendance.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri

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

    private fun copyModel(context: Context, filename: String): String {
        val directory = File(context.filesDir, MODEL_FOLDER)

        if (!directory.exists() && !directory.mkdirs()) {
            throw IllegalStateException("Could not create model directory")
        }

        val destination = File(directory, filename)

        if (!destination.exists() || destination.length() == 0L) {
            context.assets.open("$MODEL_FOLDER/$filename").use { input ->
                destination.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        }

        return destination.absolutePath
    }

    private fun createSession(path: String): OrtSession {
        val options = OrtSession.SessionOptions()

        return try {
            environment.createSession(path, options)
        } finally {
            options.close()
        }
    }

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

            val yoloPath = copyModel(context, MODEL_FILES[0])
            val detectorPath = copyModel(context, MODEL_FILES[1])
            val arcfacePath = copyModel(context, MODEL_FILES[2])

            newYolo = createSession(yoloPath)
            newDetector = createSession(detectorPath)
            newArcface = createSession(arcfacePath)

            val newEngine = FaceRecognitionEngine(
                YoloFaceDetector(environment, newYolo),
                RetinaFaceDetector(environment, newDetector),
                ArcFaceEmbedder(environment, newArcface)
            )

            synchronized(engineLock) {
                yoloSession?.close()
                detectorSession?.close()
                arcfaceSession?.close()

                yoloSession = newYolo
                detectorSession = newDetector
                arcfaceSession = newArcface
                faceEngine = newEngine

                newYolo = null
                newDetector = null
                newArcface = null
            }

            val result = Arguments.createMap()
            result.putBoolean("success", true)
            result.putString(
                "message",
                "Models loaded and face recognition engine initialized."
            )
            result.putString("yolo", "loaded")
            result.putString("faceDetector", "loaded")
            result.putString("arcface", "loaded")
            result.putBoolean("openCVReady", true)

            promise.resolve(result)

        } catch (e: Exception) {
            try {
                newYolo?.close()
                newDetector?.close()
                newArcface?.close()
            } catch (_: Exception) {
            }

            promise.reject("MODEL_LOAD_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun getRuntimeStatus(promise: Promise) {
        try {
            val result = Arguments.createMap()

            synchronized(engineLock) {
                result.putString("runtime", "ONNX Runtime")
                result.putString("version", "1.20.0")
                result.putBoolean("available", true)

                result.putBoolean(
                    "modelsLoaded",
                    faceEngine != null &&
                        yoloSession != null &&
                        detectorSession != null &&
                        arcfaceSession != null
                )

                result.putBoolean("openCVReady", openCvReady)
            }

            promise.resolve(result)

        } catch (e: Exception) {
            promise.reject("RUNTIME_ERROR", e.message, e)
        }
    }

    @ReactMethod
    fun extractFaces(imageUri: String, promise: Promise) {
        Thread {
            var bitmap: Bitmap? = null
            val rgba = Mat()
            val bgr = Mat()

            try {
                if (imageUri.isBlank()) {
                    throw IllegalArgumentException("Image URI is empty")
                }

                synchronized(engineLock) {
                    if (faceEngine == null || !openCvReady) {
                        throw IllegalStateException(
                            "Load models before extracting faces"
                        )
                    }
                }

                val uri = Uri.parse(imageUri)

                bitmap = reactApplicationContext.contentResolver
                    .openInputStream(uri)
                    ?.use { stream -> BitmapFactory.decodeStream(stream) }
                    ?: throw IllegalArgumentException(
                        "Could not decode image URI: $imageUri"
                    )

                Utils.bitmapToMat(bitmap, rgba)

                Imgproc.cvtColor(
                    rgba,
                    bgr,
                    Imgproc.COLOR_RGBA2BGR
                )

                val features = synchronized(engineLock) {
                    val engine = faceEngine
                        ?: throw IllegalStateException(
                            "Face recognition engine is not initialized"
                        )

                    engine.extractFaces(bgr)
                }

                val result = Arguments.createMap()
                val facesArray = Arguments.createArray()

                for (feature in features) {
                    val faceMap = Arguments.createMap()

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

                    val embeddingArray = Arguments.createArray()

                    for (value in feature.embedding) {
                        embeddingArray.pushDouble(value.toDouble())
                    }

                    faceMap.putArray("embedding", embeddingArray)
                    facesArray.pushMap(faceMap)
                }

                result.putBoolean("success", true)
                result.putInt("faceCount", features.size)
                result.putArray("faces", facesArray)

                promise.resolve(result)

            } catch (e: Exception) {
                promise.reject("FACE_EXTRACTION_ERROR", e.message, e)

            } finally {
                bitmap?.recycle()
                rgba.release()
                bgr.release()
            }
        }.start()
    }

    override fun invalidate() {
        synchronized(engineLock) {
            faceEngine = null

            try {
                yoloSession?.close()
                detectorSession?.close()
                arcfaceSession?.close()
            } catch (_: Exception) {
            }

            yoloSession = null
            detectorSession = null
            arcfaceSession = null
        }

        super.invalidate()
    }
}
