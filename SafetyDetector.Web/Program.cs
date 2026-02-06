using Microsoft.Extensions.ML;
using SafetyDetector.Shared.DataModels;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddRazorPages();

// Register ML.NET PredictionEnginePool (thread-safe for web apps)
var modelPath = Path.Combine(builder.Environment.ContentRootPath, "MLModel", "model.zip");
builder.Services.AddPredictionEnginePool<ModelInput, ModelOutput>()
    .FromFile(modelName: "HardHatDetector", filePath: modelPath, watchForChanges: true);

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseRouting();
app.UseAuthorization();

app.MapStaticAssets();
app.MapRazorPages()
   .WithStaticAssets();

app.Run();
