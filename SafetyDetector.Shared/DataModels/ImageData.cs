namespace SafetyDetector.Shared.DataModels;

/// <summary>
/// Represents a raw image loaded from disk.
/// Used to build the initial dataset from folder structure.
/// Folder name becomes the Label (e.g., "hard_hat", "no_hard_hat").
/// </summary>
public class ImageData
{
    public string ImagePath { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
}
