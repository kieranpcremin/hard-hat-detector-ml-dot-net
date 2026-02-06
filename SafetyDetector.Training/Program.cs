using Microsoft.ML;
using Microsoft.ML.Vision;
using SafetyDetector.Shared.DataModels;

// =============================================================================
// CONFIGURATION
// =============================================================================

string dataDir = args.Length > 0 ? args[0] : Path.Combine("..", "data");
string modelOutputPath = args.Length > 1 ? args[1] : Path.Combine("..", "SafetyDetector.Web", "MLModel", "model.zip");
int epochs = 15;
int batchSize = 10;

Console.WriteLine("============================================================");
Console.WriteLine("  HARD HAT DETECTOR - ML.NET Training");
Console.WriteLine("============================================================");
Console.WriteLine($"  Data directory:  {Path.GetFullPath(dataDir)}");
Console.WriteLine($"  Model output:    {Path.GetFullPath(modelOutputPath)}");
Console.WriteLine($"  Epochs:          {epochs}");
Console.WriteLine($"  Batch size:      {batchSize}");
Console.WriteLine($"  Architecture:    ResNet V2 50");
Console.WriteLine("============================================================\n");

// =============================================================================
// STEP 1: Load images from folder structure
// =============================================================================
// Folder-per-class convention: data/hard_hat/*.jpg, data/no_hard_hat/*.jpg

Console.WriteLine("Loading images from folder structure...");

var images = new List<ImageData>();
var supportedExtensions = new[] { ".jpg", ".jpeg", ".png" };

foreach (var classDir in Directory.GetDirectories(dataDir))
{
    string label = Path.GetFileName(classDir);

    var imageFiles = Directory.GetFiles(classDir)
        .Where(f => supportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
        .ToList();

    foreach (var file in imageFiles)
    {
        images.Add(new ImageData { ImagePath = Path.GetFullPath(file), Label = label });
    }

    Console.WriteLine($"  {label}: {imageFiles.Count} images");
}

Console.WriteLine($"  Total: {images.Count} images\n");

if (images.Count == 0)
{
    Console.WriteLine("ERROR: No images found. Please check your data directory.");
    Console.WriteLine("Expected structure:");
    Console.WriteLine("  data/");
    Console.WriteLine("    hard_hat/     (JPEG/PNG images)");
    Console.WriteLine("    no_hard_hat/  (JPEG/PNG images)");
    return;
}

// =============================================================================
// STEP 2: Create ML.NET pipeline
// =============================================================================

var mlContext = new MLContext(seed: 42);

// Load into IDataView
IDataView fullData = mlContext.Data.LoadFromEnumerable(images);

// Shuffle to avoid ordering bias
fullData = mlContext.Data.ShuffleRows(fullData, seed: 42);

// Preprocessing pipeline:
// 1. MapValueToKey - convert string label to numeric key
// 2. LoadRawImageBytes - read image files into byte arrays
var preprocessPipeline = mlContext.Transforms.Conversion
    .MapValueToKey("LabelAsKey", "Label")
    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "Image",
        imageFolder: null,       // Paths are absolute
        inputColumnName: "ImagePath"));

// Fit and transform the preprocessing
Console.WriteLine("Preprocessing images (loading bytes)...");
IDataView preprocessedData = preprocessPipeline.Fit(fullData).Transform(fullData);

// =============================================================================
// STEP 3: Train/Validation split (80/20)
// =============================================================================

var trainTestSplit = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.2, seed: 42);
IDataView trainData = trainTestSplit.TrainSet;
IDataView valData = trainTestSplit.TestSet;

Console.WriteLine($"Train set: {trainData.GetRowCount()} images");
Console.WriteLine($"Val set:   {valData.GetRowCount()} images\n");

// =============================================================================
// STEP 4: Define the training pipeline
// =============================================================================
// ImageClassificationTrainer uses transfer learning with a pre-trained
// ResNet V2 50 backbone. It:
// 1. Computes bottleneck features (output of frozen backbone layers)
// 2. Caches them to disk for faster subsequent epochs
// 3. Trains only the final classification layer

Console.WriteLine("Configuring ImageClassification trainer...");
Console.WriteLine("  Architecture: ResNet V2 50 (transfer learning)");
Console.WriteLine("  This will download pre-trained weights on first run (~100MB)\n");

var trainerOptions = new ImageClassificationTrainer.Options
{
    FeatureColumnName = "Image",
    LabelColumnName = "LabelAsKey",
    ValidationSet = valData,
    Arch = ImageClassificationTrainer.Architecture.ResnetV250,
    Epoch = epochs,
    BatchSize = batchSize,
    LearningRate = 0.01f,
    MetricsCallback = (metrics) =>
    {
        if (metrics.Train != null)
        {
            Console.WriteLine($"  Epoch {metrics.Train.Epoch:D3} | " +
                              $"Train Acc: {metrics.Train.Accuracy * 100:F1}% | " +
                              $"Train Loss: {metrics.Train.CrossEntropy:F4}");
        }
    },
    ReuseTrainSetBottleneckCachedValues = true,
    ReuseValidationSetBottleneckCachedValues = true
};

var trainingPipeline = mlContext.MulticlassClassification.Trainers
    .ImageClassification(trainerOptions)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

// =============================================================================
// STEP 5: Train the model
// =============================================================================

Console.WriteLine("Training started...\n");
var stopwatch = System.Diagnostics.Stopwatch.StartNew();

ITransformer trainedModel = trainingPipeline.Fit(trainData);

stopwatch.Stop();
Console.WriteLine($"\nTraining completed in {stopwatch.Elapsed.TotalMinutes:F1} minutes\n");

// =============================================================================
// STEP 6: Evaluate on validation set
// =============================================================================

Console.WriteLine("Evaluating on validation set...");

IDataView predictions = trainedModel.Transform(valData);
var metrics = mlContext.MulticlassClassification.Evaluate(
    predictions,
    labelColumnName: "LabelAsKey",
    predictedLabelColumnName: "PredictedLabel",
    scoreColumnName: "Score");

Console.WriteLine("\n============================================================");
Console.WriteLine("  EVALUATION RESULTS");
Console.WriteLine("============================================================");
Console.WriteLine($"  Macro Accuracy:    {metrics.MacroAccuracy * 100:F2}%");
Console.WriteLine($"  Micro Accuracy:    {metrics.MicroAccuracy * 100:F2}%");
Console.WriteLine($"  Log Loss:          {metrics.LogLoss:F4}");
Console.WriteLine($"  Log Loss Reduction:{metrics.LogLossReduction:F4}");
Console.WriteLine("============================================================\n");

// Print confusion matrix
Console.WriteLine("Confusion Matrix:");
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

// =============================================================================
// STEP 7: Save the model
// =============================================================================

// Ensure output directory exists
string? outputDir = Path.GetDirectoryName(Path.GetFullPath(modelOutputPath));
if (outputDir != null)
    Directory.CreateDirectory(outputDir);

mlContext.Model.Save(trainedModel, trainData.Schema, modelOutputPath);

Console.WriteLine($"Model saved to: {Path.GetFullPath(modelOutputPath)}");
Console.WriteLine($"Model file size: {new FileInfo(modelOutputPath).Length / 1024.0 / 1024.0:F1} MB");
Console.WriteLine("\nTo run the web app:");
Console.WriteLine("  dotnet run --project SafetyDetector.Web");
