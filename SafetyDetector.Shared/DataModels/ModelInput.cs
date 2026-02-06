using Microsoft.ML.Data;

namespace SafetyDetector.Shared.DataModels;

/// <summary>
/// Input schema for the ML.NET image classification trainer.
/// The Image column contains raw bytes loaded by LoadRawImageBytes.
/// LabelAsKey is the numeric encoding of the string label.
/// </summary>
public class ModelInput
{
    public string ImagePath { get; set; } = string.Empty;

    public string Label { get; set; } = string.Empty;

    [ColumnName("LabelAsKey")]
    public uint LabelAsKey { get; set; }

    [ColumnName("Image")]
    public byte[] Image { get; set; } = Array.Empty<byte>();
}
