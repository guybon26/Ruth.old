package com.ruth.mobile

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.Constraints
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkRequest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class RuthWorker(appContext: Context, workerParams: WorkerParameters) :
    CoroutineWorker(appContext, workerParams) {

    companion object {
        // Load the native library containing the JNI implementation
        init {
            System.loadLibrary("ruth_jni")
        }

        fun scheduleWork(context: Context) {
            val constraints = Constraints.Builder()
                .setRequiresCharging(true)
                .setRequiredNetworkType(NetworkType.UNMETERED)
                .setRequiresDeviceIdle(true)
                .build()

            val workRequest: WorkRequest = OneTimeWorkRequestBuilder<RuthWorker>()
                .setConstraints(constraints)
                .build()

            WorkManager.getInstance(context).enqueue(workRequest)
        }
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        // 1. Thermal Check
        if (isDeviceTooHot()) {
            // Return retry to reschedule when conditions might be better
            return@withContext Result.retry()
        }

        // 2. Run Training
        // Paths would typically come from inputData or internal storage
        val modelPath = applicationContext.filesDir.absolutePath + "/model.pte"
        val dataPath = applicationContext.filesDir.absolutePath + "/data.bin"

        val success = runTrainingStep(modelPath, dataPath)

        if (success) {
            Result.success()
        } else {
            Result.failure()
        }
    }

    private fun isDeviceTooHot(): Boolean {
        val intent = applicationContext.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )
        
        // Temperature is an integer in tenths of a degree Celsius
        val temp = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        
        // 400 tenths = 40.0 degrees Celsius
        return temp > 400
    }

    /**
     * JNI Bridge to C++ ExecuTorch Runtime.
     * 
     * @param modelPath Path to the .pte model file.
     * @param dataPath Path to the training data.
     * @return True if training step succeeded, False otherwise.
     */
    external fun runTrainingStep(modelPath: String, dataPath: String): Boolean
}
