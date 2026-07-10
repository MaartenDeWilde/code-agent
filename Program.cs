Console.WriteLine("=== Simple Local Coding Agent ===");
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

Console.WriteLine();
Console.WriteLine($"Working folder: {rootFolder}");
Console.WriteLine($"Model:          {model}");
Console.WriteLine("Type a task for the agent, 'clear' to reset the conversation, or 'exit' to quit.");
Console.WriteLine();

var agent = new CodingAgent(rootFolder, model);

// --- 3. The REPL: one task per line, agent keeps history between tasks ---
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
        agent = new CodingAgent(rootFolder, model);
        Console.WriteLine("[conversation cleared]");
        continue;
    }

    await agent.RunAsync(task);
}
