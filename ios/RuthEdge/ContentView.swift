import SwiftUI
import Foundation

// Assuming RuthRunner is bridged via bridging header
// If compiling this file directly without the project context, RuthRunner might be undefined.
// In a real Xcode project, the bridging header makes it available.

class TrainingViewModel: ObservableObject {
    @Published var logs: String = "Ready to train.\n"
    @Published var thermalState: String = "Nominal"
    @Published var isTraining: Bool = false
    
    private var runner: RuthRunner?
    
    init() {
        setupThermalMonitoring()
        loadModel()
    }
    
    private func setupThermalMonitoring() {
        // Initial state
        updateThermalState(ProcessInfo.processInfo.thermalState)
        
        // Observer
        NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.updateThermalState(ProcessInfo.processInfo.thermalState)
        }
    }
    
    private func updateThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal: thermalState = "Nominal"
        case .fair: thermalState = "Fair"
        case .serious: thermalState = "Serious"
        case .critical: thermalState = "Critical"
        @unknown default: thermalState = "Unknown"
        }
    }
    
    private func loadModel() {
        // Attempt to load model from bundle
        if let modelPath = Bundle.main.path(forResource: "ruth_llama_1b_mps", ofType: "pte") {
            runner = RuthRunner(modelPath: modelPath)
            if runner != nil {
                appendLog("Model loaded successfully from bundle.")
            } else {
                appendLog("Error: Failed to initialize RuthRunner.")
            }
        } else {
            appendLog("Error: Model file 'ruth_llama_1b_mps.pte' not found in bundle.")
        }
    }
    
    func startTraining() {
        guard !isTraining else { return }
        isTraining = true
        appendLog("Starting training loop...")
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            let inputs = [
                "The quick brown fox jumps over the lazy dog.",
                "Ruth is a privacy-preserving federated learning system.",
                "Training on device ensures data stays local.",
                "Causal discovery helps identify robust features."
            ]
            
            var step = 0
            while self.isTraining {
                // 1. Thermal Gate
                let state = ProcessInfo.processInfo.thermalState
                if state == .serious || state == .critical {
                    DispatchQueue.main.async {
                        self.appendLog("Thermal throttle active. Cooling down (10s)...")
                    }
                    sleep(10)
                    continue
                }
                
                // 2. Train Step
                if let runner = self.runner {
                    let input = inputs[step % inputs.count]
                    // Note: Objective-C method is trainStepWithInput:
                    // Swift imports it as trainStep(withInput:) or similar depending on naming conventions
                    // In RuthRunner.h: - (float)trainStepWithInput:(NSString *)input;
                    // Swift: trainStep(withInput: String)
                    let loss = runner.trainStep(withInput: input)
                    
                    DispatchQueue.main.async {
                        self.appendLog("Step \(step): Loss/Grad = \(String(format: "%.4f", loss))")
                    }
                } else {
                    DispatchQueue.main.async {
                        self.appendLog("Error: Runner not initialized.")
                        self.isTraining = false
                    }
                    break
                }
                
                step += 1
                // Simulate some delay between steps to not hammer the CPU/GPU too hard in this loop
                usleep(100000) // 0.1s
            }
        }
    }
    
    func stopTraining() {
        isTraining = false
        appendLog("Training stopped.")
    }
    
    private func appendLog(_ message: String) {
        // Keep log size manageable
        if logs.count > 10000 {
            logs = String(logs.suffix(5000))
        }
        logs += "\(DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)): \(message)\n"
    }
}

struct ContentView: View {
    @StateObject var viewModel = TrainingViewModel()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("RuthEdge Trainer")
                .font(.largeTitle)
                .bold()
            
            HStack {
                Text("Thermal State:")
                Text(viewModel.thermalState)
                    .foregroundColor(thermalColor(viewModel.thermalState))
                    .bold()
            }
            .padding(.top)
            
            ScrollView {
                Text(viewModel.logs)
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .background(Color(UIColor.secondarySystemBackground))
            .cornerRadius(10)
            .frame(maxHeight: .infinity)
            
            Button(action: {
                if viewModel.isTraining {
                    viewModel.stopTraining()
                } else {
                    viewModel.startTraining()
                }
            }) {
                Text(viewModel.isTraining ? "STOP TRAINING" : "START TRAINING")
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(viewModel.isTraining ? Color.red : Color.blue)
                    .cornerRadius(10)
            }
            .padding(.bottom)
        }
        .padding()
    }
    
    func thermalColor(_ state: String) -> Color {
        switch state {
        case "Nominal": return .green
        case "Fair": return .yellow
        case "Serious": return .orange
        case "Critical": return .red
        default: return .gray
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
