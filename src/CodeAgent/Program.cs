using CodeAgent.Agent;
using CodeAgent.Ollama;

static void DisplayLogo()
{
    Console.ForegroundColor = ConsoleColor.Blue;
    Console.WriteLine("Mantares Coding Agent");
    Console.ResetColor();
}

DisplayLogo();
Console.WriteLine();

// --- 1. Ask for the folder the agent is allowed to work in ---
string rootFolder;
while (true)
{
    Console.Write("Input folder (the agent will only read/write inside this folder): ");
    var input = Console.ReadLine()?.Trim().Trim('"');
    if (string.IsNullOrWhiteSpace(input))
    {
        continue;
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

// --- 2. Pick the model (needs native tool-calling support: qwen, llama3.1+, ...) ---
Console.Write("Model to use [qwen3.5:9b]: ");
var modelInput = Console.ReadLine()?.Trim();
var model = string.IsNullOrWhiteSpace(modelInput) ? "qwen3.5:9b" : modelInput;

// --- 3. Context window: sent to Ollama as num_ctx and used for trimming ---
Console.Write("Context window in tokens [262144]: ");
var contextInput = Console.ReadLine()?.Trim();
var contextTokens = int.TryParse(contextInput, out var parsed) && parsed > 0 ? parsed : 262144;

Console.WriteLine();
Console.WriteLine($"Working folder: {rootFolder}");
Console.WriteLine($"Model:          {model}");
Console.WriteLine($"Context:        {contextTokens:N0} tokens");
Console.WriteLine();
Console.WriteLine("Welcome to the Coding Agent! Enter your task below, or use these commands:");
Console.WriteLine("-----------------------------------------------");
Console.WriteLine("");
Console.WriteLine("<task>              - Give the agent a specific coding task to execute.");
Console.WriteLine("  plan <task>       - Investigate and propose a detailed plan without modifying files.");
Console.WriteLine("  go                - Execute the most recently created or approved plan step by step.");
Console.WriteLine("                    (Use 'plan' first, then 'go' when you're ready to apply changes.)");
Console.WriteLine("");
Console.WriteLine("model <name>        - Show installed models from Ollama, or switch to another model.");
Console.WriteLine("  clear             - Reset the conversation history and start fresh.");
Console.WriteLine("  exit              - Exit the application.");
Console.WriteLine();

var agent = new CodingAgent(rootFolder, model, contextTokens);

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

// --- 4. The REPL: one task per line, agent keeps history between tasks ---
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
        agent = new CodingAgent(rootFolder, model, contextTokens);
        Console.WriteLine("[conversation cleared]");
        continue;
    }
    if (task.Equals("model", StringComparison.OrdinalIgnoreCase)
        || task.StartsWith("model ", StringComparison.OrdinalIgnoreCase))
    {
        var requested = task.Length > 5 ? task[6..].Trim() : "";

        // The installed list validates switches and powers the overview;
        // if Ollama can't be reached we still allow switching blind.
        List<string> installed = [];
        try
        {
            installed = await new OllamaClient().ListModelsAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[could not list models from Ollama: {ex.Message}]");
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
