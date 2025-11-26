# RuthEdge iOS Project Setup

## Info.plist Configuration

To support background training and motion data access, add the following keys to your `Info.plist`:

1.  **Background Tasks**:
    - Key: `BGTaskSchedulerPermittedIdentifiers`
    - Type: `Array`
    - Value: `["com.ruth.train"]`

2.  **Privacy - Motion Usage**:
    - Key: `NSMotionUsageDescription`
    - Type: `String`
    - Value: "Ruth uses motion data to learn causal links."

## Build Settings

- **Enable C++ Exceptions**: Yes
- **Enable C++ RTTI**: Yes
- **Other Linker Flags**: `-all_load` (may be required for some ExecuTorch backends)
