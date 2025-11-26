#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * RuthRunner
 * Bridge between Swift/Obj-C and the C++ RuthTrainer/ExecuTorch runtime.
 */
@interface RuthRunner : NSObject

/**
 * Initializes the runner with the path to the .pte model file.
 * @param modelPath The absolute path to the exported ExecuTorch model.
 */
- (instancetype)initWithModelPath:(NSString *)modelPath;

/**
 * Performs a single training step.
 * @param input The input string (e.g., tokenized text or raw text to be
 * tokenized).
 * @return The loss value (scalar) from the training step.
 */
- (float)trainStepWithInput:(NSString *)input;

@end

NS_ASSUME_NONNULL_END
