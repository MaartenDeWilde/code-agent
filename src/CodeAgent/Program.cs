using CodeAgent.Agent;

static void DisplayLogo()
{
    Console.ForegroundColor = ConsoleColor.Blue;    
    Console.WriteLine("      Mantares Coding Agent");
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
Console.Write("Model to use [gemma4:12b-mlx]: ");
var modelInput = Console.ReadLine()?.Trim();
var model = string.IsNullOrWhiteSpace(modelInput) ? "gemma4:12b-mlx" : modelInput;

// --- 3. Context window: sent to Ollama as num_ctx and used for trimming ---
Console.Write("Context window in tokens [262144]: ");
var contextInput = Console.ReadLine()?.Trim();
var contextTokens = int.TryParse(contextInput, out var parsed) && parsed > 0 ? parsed : 262144;

Console.WriteLine();
Console.WriteLine($"Working folder: {rootFolder}");
Console.WriteLine($"Model:          {model}");
Console.WriteLine($"Context:        {contextTokens:N0} tokens");
Console.WriteLine();
Console.WriteLine("Commands:");
Console.WriteLine("  <task>       give the agent a task to execute");
Console.WriteLine("  plan <task>  investigate and propose a plan without changing files");
Console.WriteLine("  go           execute the most recent plan");
Console.WriteLine("  clear        reset the conversation");
Console.WriteLine("  exit         quit");
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
