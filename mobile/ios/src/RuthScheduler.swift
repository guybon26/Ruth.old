import Foundation
import BackgroundTasks
import os.log

// Mock C++ Wrapper for Training Loop
class RuthTrainer {
    static let shared = RuthTrainer()
    
    func train(completion: @escaping (Bool) -> Void) {
        // Simulate training
        print("Starting training...")
        DispatchQueue.global().asyncAfter(deadline: .now() + 2.0) {
            print("Training finished.")
            completion(true)
        }
    }
    
    func stop() {
        print("Stopping training...")
    }
}

class RuthScheduler {
    static let shared = RuthScheduler()
    
    let taskIdentifier = "com.ruth.train"
    
    private init() {}
    
    func register() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: taskIdentifier, using: nil) { task in
            if let processingTask = task as? BGProcessingTask {
                self.handleAppRefresh(task: processingTask)
            }
        }
    }
    
    func schedule() {
        let request = BGProcessingTaskRequest(identifier: taskIdentifier)
        request.requiresNetworkConnectivity = false // Local training doesn't strictly need network, aggregation does
        request.requiresExternalPower = true
        
        // Schedule for 15 minutes from now (earliest begin date)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)
        
        do {
            try BGTaskScheduler.shared.submit(request)
            os_log("Task scheduled successfully", type: .info)
        } catch {
            os_log("Could not schedule task: %@", type: .error, error.localizedDescription)
        }
    }
    
    func handleAppRefresh(task: BGProcessingTask) {
        // 1. Expiration Handler
        task.expirationHandler = {
            // Save checkpoint, cleanup
            RuthTrainer.shared.stop()
            task.setTaskCompleted(success: false)
        }
        
        // 2. Thermal Check
        let thermalState = ProcessInfo.processInfo.thermalState
        if thermalState == .critical || thermalState == .serious {
            os_log("Thermal state too high, skipping training", type: .error)
            task.setTaskCompleted(success: false)
            return
        }
        
        // 3. Run Training
        RuthTrainer.shared.train { success in
            task.setTaskCompleted(success: success)
            
            // Re-schedule
            self.schedule()
        }
    }
}
