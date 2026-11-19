using CodeAgent.Agent;
using CodeAgent.Llm;
using Serilog;
using Serilog.Extensions.Logging;
using Serilog.Formatting.Compact;

static void DisplayLogo()
{
    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine("C# Coding Agent");
    Console.ResetColor();
}

// Clear terminal screen on startup
Console.Clear();

DisplayLogo();
Console.WriteLine();

// --- 0. Logging: Serilog is the provider, but everything else in the app
// only sees Microsoft.Extensions.Logging.ILogger — the standard .NET pattern
// that keeps the logging library swappable. Logs go to files next to the app
// (never into the user's project, and never to the console, which the agent
// UI owns): a human-readable .log and a machine-readable .clef (newline-
// delimited JSON) that structured-log tooling can query by property.
var logsFolder = Path.Combine(AppContext.BaseDirectory, "logs");
var serilog = new LoggerConfiguration()
    .MinimumLevel.Debug()
    // Carries BeginScope properties (like TaskId) onto every event.
    .Enrich.FromLogContext()
    .WriteTo.File(Path.Combine(logsFolder, "agent-.log"),
        rollingInterval: RollingInterval.Day, retainedFileCountLimit: 14)
    .WriteTo.File(new CompactJsonFormatter(), Path.Combine(logsFolder, "agent-.clef"),
        rollingInterval: RollingInterval.Day, retainedFileCountLimit: 14)
    .CreateLogger();
// dispose: true flushes buffered log events when the factory is disposed.
using var loggerFactory = new SerilogLoggerFactory(serilog, dispose: true);

// --- 1. Ask for the folder the agent is allowed to work in ---
string rootFolder;
while (true)
{
    Console.Write("Input folder (the agent will only read/write inside this folder, or press Enter for current directory): ");
    var input = Console.ReadLine()?.Trim().Trim('"');
    
    // Empty string means use the current directory "."
    if (string.IsNullOrWhiteSpace(input))
    {
        rootFolder = Directory.GetCurrentDirectory();
        break;
    }

    var fullPath = Path.GetFullPath(input);
    if (!Directory.Exists(fullPath))
    {
        Console.WriteLine($"Folder not found: {fullPath}");
        continue;
    }

    rootFolder = fullPath;
    break;
}

// --- 2. Pick the model server ---
Console.WriteLine("Model servers:");
foreach (var b in Backend.All)
{
    Console.WriteLine($"  {b.Name,-8} {b.Description}");
}
Backend backend;
while (true)
{
    Console.Write("Model server [ollama]: ");
    var backendInput = Console.ReadLine()?.Trim();
    if (Backend.Find(string.IsNullOrWhiteSpace(backendInput) ? "ollama" : backendInput) is { } found)
    {
        backend = found;
        break;
    }
    Console.WriteLine($"Unknown model server: {backendInput}");
}
var client = backend.Create(loggerFactory);

// --- 3. Pick the model (needs native tool-calling support: qwen, llama3.1+, ...) ---
Console.Write($"Model to use [{backend.DefaultModel}]: ");
var modelInput = Console.ReadLine()?.Trim();
var model = string.IsNullOrWhiteSpace(modelInput) ? backend.DefaultModel : modelInput;

// --- 4. Context window: sent to Ollama as num_ctx and used for trimming ---
// (OpenAI-compatible servers fix the window at startup, e.g. 'embla serve
// --context'; enter the same value so trimming keeps the prompt under it.)
// Small local models degrade (and their KV cache balloons) long before huge
// windows; 32k plus transcript trimming beats a nominal 262k in practice.
Console.Write("Context window in tokens [32768]: ");
var contextInput = Console.ReadLine()?.Trim();
var contextTokens = int.TryParse(contextInput, out var parsed) && parsed > 0 ? parsed : 32768;

Console.WriteLine();
Console.WriteLine($"Working folder: {rootFolder}");
Console.WriteLine($"Model server:   {backend.Name}");
Console.WriteLine($"Model:          {model}");
Console.WriteLine($"Context:        {contextTokens:N0} tokens");
Console.WriteLine($"Logging to:     {logsFolder}");
Console.WriteLine();
Console.WriteLine("Welcome to the Coding Agent! Enter your task below, or use these commands:");
Console.WriteLine("-----------------------------------------------");
Console.WriteLine("");
Console.WriteLine("<task>              - Give the agent a specific coding task to execute.");
Console.WriteLine("  plan <task>       - Investigate and propose a detailed plan without modifying files.");
Console.WriteLine("  go                - Execute the most recently created or approved plan step by step.");
Console.WriteLine("                    (Use 'plan' first, then 'go' when you're ready to apply changes.)");
Console.WriteLine("");
Console.WriteLine("model <name>        - Show the models the server offers, or switch to another model.");
Console.WriteLine("  clear             - Reset the conversation history and start fresh.");
Console.WriteLine("  exit              - Exit the application.");
Console.WriteLine();

var agent = new CodingAgent(rootFolder, model, contextTokens, loggerFactory, client);

// Ctrl+C while a task runs cancels the task instead of killing the app;
// Ctrl+C at the prompt exits as usual.
CancellationTokenSource? runningTask = null;
Console.CancelKeyPress += (_, e) =>
{
    if (runningTask is { IsCancellationRequested: false } cts)
    {
        e.Cancel = true;
        cts.Cancel();
    }
};

// --- 5. The REPL: one task per line, agent keeps history between tasks ---
while (true)
{
    Console.Write("> ");
    var task = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(task))
    {
        continue;
    }
    if (task.Equals("exit", StringComparison.OrdinalIgnoreCase))
    {
        break;
    }
    if (task.Equals("clear", StringComparison.OrdinalIgnoreCase))
    {
        agent = new CodingAgent(rootFolder, model, contextTokens, loggerFactory, client);
        Console.WriteLine("[conversation cleared]");
        continue;
    }
    if (task.Equals("model", StringComparison.OrdinalIgnoreCase)
        || task.StartsWith("model ", StringComparison.OrdinalIgnoreCase))
    {
        var requested = task.Length > 5 ? task[6..].Trim() : "";

        // The installed list validates switches and powers the overview;
        // if the server can't be reached we still allow switching blind.
        List<string> installed = [];
        try
        {
            installed = await client.ListModelsAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[could not list models from {client.Name}: {ex.Message}]");
        }

        if (requested.Length == 0)
        {
            Console.WriteLine($"Current model: {model}");
            foreach (var name in installed)
            {
                Console.WriteLine($"  {(name == model ? "*" : " ")} {name}");
            }
            Console.WriteLine("Switch with: model <name>");
            continue;
        }

        if (installed.Count > 0 && !installed.Contains(requested))
        {
            Console.WriteLine($"Model '{requested}' is not installed. Type 'model' to see the installed list.");
            continue;
        }

        // Takes effect on the next request; conversation history is kept.
        model = requested;
        agent.Model = requested;
        Console.WriteLine($"[model switched to {model}]");
        continue;
    }

    using var cts = new CancellationTokenSource();
    runningTask = cts;
    try
    {
        if (task.StartsWith("plan ", StringComparison.OrdinalIgnoreCase))
        {
            await agent.PlanAsync(task[5..].Trim(), cts.Token);
        }
        else if (task.Equals("go", StringComparison.OrdinalIgnoreCase))
        {
            // The plan lives in the conversation history; approval is just
            // another user message, now with the full toolset available.
            await agent.RunAsync("The plan is approved. Execute it now, step by step.", cancellationToken: cts.Token);
        }
        else
        {
            await agent.RunAsync(task, cancellationToken: cts.Token);
        }
    }
    finally
    {
        runningTask = null;
    }
}
