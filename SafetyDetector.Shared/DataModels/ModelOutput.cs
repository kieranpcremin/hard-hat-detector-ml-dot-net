using Microsoft.ML.Data;

namespace SafetyDetector.Shared.DataModels;

/// <summary>
/// Output schema from ML.NET prediction.
/// PredictedLabel is the human-readable class name.
/// Score contains probability for each class.
/// </summary>
public class ModelOutput
{
    public string ImagePath { get; set; } = string.Empty;

    public string Label { get; set; } = string.Empty;

    public string PredictedLabel { get; set; } = string.Empty;

    public float[] Score { get; set; } = Array.Empty<float>();
}
