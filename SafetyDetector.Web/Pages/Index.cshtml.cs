using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.Extensions.ML;
using SafetyDetector.Shared.DataModels;

namespace SafetyDetector.Web.Pages;

public class IndexModel : PageModel
{
    private readonly PredictionEnginePool<ModelInput, ModelOutput> _predictionEnginePool;

    public IndexModel(PredictionEnginePool<ModelInput, ModelOutput> predictionEnginePool)
    {
        _predictionEnginePool = predictionEnginePool;
    }

    // Prediction results (null until a prediction is made)
    public string? PredictedLabel { get; set; }
    public float Confidence { get; set; }
    public Dictionary<string, float>? Probabilities { get; set; }
    public string? UploadedImageBase64 { get; set; }
    public string? ErrorMessage { get; set; }
    public bool HasPrediction => PredictedLabel != null;
    public bool IsHardHat => PredictedLabel == "hard_hat";

    public void OnGet()
    {
    }

    public async Task<IActionResult> OnPostAsync(IFormFile? imageFile)
    {
        if (imageFile == null || imageFile.Length == 0)
        {
            ErrorMessage = "Please select an image to upload.";
            return Page();
        }

        // Validate file type
        var allowedExtensions = new[] { ".jpg", ".jpeg", ".png" };
        var extension = Path.GetExtension(imageFile.FileName).ToLowerInvariant();
        if (!allowedExtensions.Contains(extension))
        {
            ErrorMessage = "Only JPEG and PNG images are supported.";
            return Page();
        }

        try
        {
            // Read image bytes
            using var memoryStream = new MemoryStream();
            await imageFile.CopyToAsync(memoryStream);
            var imageBytes = memoryStream.ToArray();

            // Store base64 for displaying the uploaded image
            UploadedImageBase64 = $"data:{imageFile.ContentType};base64,{Convert.ToBase64String(imageBytes)}";

            // Check if model exists
            var modelPath = Path.Combine(Directory.GetCurrentDirectory(), "MLModel", "model.zip");
            if (!System.IO.File.Exists(modelPath))
            {
                ErrorMessage = "Model not found. Please train the model first using SafetyDetector.Training.";
                return Page();
            }

            // Create prediction input
            var input = new ModelInput { Image = imageBytes };

            // Make prediction
            var prediction = _predictionEnginePool.Predict(modelName: "HardHatDetector", example: input);

            PredictedLabel = prediction.PredictedLabel;

            // Build probabilities dictionary
            // ML.NET Score array order matches the key encoding order
            string[] classNames = { "hard_hat", "no_hard_hat" };
            Probabilities = new Dictionary<string, float>();

            if (prediction.Score != null)
            {
                Confidence = prediction.Score.Max();

                for (int i = 0; i < Math.Min(prediction.Score.Length, classNames.Length); i++)
                {
                    Probabilities[classNames[i]] = prediction.Score[i];
                }
            }
        }
        catch (Exception ex)
        {
            ErrorMessage = $"Prediction failed: {ex.Message}";
        }

        return Page();
    }
}
