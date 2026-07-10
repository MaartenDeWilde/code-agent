using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

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

Console.Write("Model to use [gemma4:12b-mlx]: ");
var modelInput = Console.ReadLine()?.Trim();
var model = string.IsNullOrWhiteSpace(modelInput) ? "gemma4:12b-mlx" : modelInput;

Console.WriteLine();
Console.WriteLine($"Working folder: {rootFolder}");
Console.WriteLine($"Model:          {model}");
Console.WriteLine("Type a task for the agent, or 'exit' to quit.");
Console.WriteLine();

var agent = new CodingAgent(rootFolder, model);

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

    await agent.RunAsync(task);
}

// ======================================================================
// Agent
// ======================================================================

class CodingAgent
{
    private const int MaxSteps = 15;

    private readonly string _rootFolder;
    private readonly string _model;
    private readonly OllamaClient _ollama = new();
    private readonly List<ChatMessage> _history = new();

    public CodingAgent(string rootFolder, string model)
    {
        _rootFolder = rootFolder;
        _model = model;
        _history.Add(new ChatMessage("system", BuildSystemPrompt(rootFolder)));
    }

    public async Task RunAsync(string task)
    {
        _history.Add(new ChatMessage("user", task));

        for (var step = 0; step < MaxSteps; step++)
        {
            string raw;
            try
            {
                raw = await Spinner.RunAsync(() => _ollama.ChatAsync(_model, _history));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[error talking to Ollama: {ex.Message}]");
                return;
            }

            _history.Add(new ChatMessage("assistant", raw));

            AgentAction? action;
            try
            {
                action = ParseAction(raw);
            }
            catch (Exception)
            {
                Console.WriteLine("[agent returned invalid JSON, asking it to retry]");
                _history.Add(new ChatMessage("user",
                    "Your last reply was not valid JSON matching the required schema. Reply again with ONLY the JSON object."));
                continue;
            }

            if (action is null)
            {
                continue;
            }

            if (action.Action == "final")
            {
                Console.WriteLine(action.Answer ?? "(no answer given)");
                return;
            }

            var result = ExecuteTool(action);
            Console.WriteLine($"[{action.Action}] {Summarize(action, result)}");
            _history.Add(new ChatMessage("user", $"Tool result for '{action.Action}':\n{result}"));
        }

        Console.WriteLine("[agent stopped: reached max steps without a final answer]");
    }

    private static string Summarize(AgentAction action, string result)
    {
        var oneLine = result.Replace("\n", " ");
        return oneLine.Length > 120 ? oneLine[..120] + "..." : oneLine;
    }

    private string ExecuteTool(AgentAction action)
    {
        try
        {
            return action.Action switch
            {
                "list_files" => ListFiles(action.Pattern ?? "*"),
                "read_file" => ReadFile(action.Path ?? ""),
                "write_file" => WriteFile(action.Path ?? "", action.Content ?? ""),
                _ => $"Unknown action '{action.Action}'. Valid actions: list_files, read_file, write_file, final."
            };
        }
        catch (Exception ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    private string ListFiles(string pattern)
    {
        var files = Directory.GetFiles(_rootFolder, pattern, SearchOption.AllDirectories)
            .Select(f => Path.GetRelativePath(_rootFolder, f))
            .Take(200)
            .ToList();

        return files.Count == 0
            ? "No files matched."
            : string.Join("\n", files);
    }

    private string ReadFile(string relativePath)
    {
        var fullPath = ResolvePath(relativePath);
        if (!File.Exists(fullPath))
        {
            return $"File not found: {relativePath}";
        }

        var content = File.ReadAllText(fullPath);
        const int maxChars = 20000;
        return content.Length > maxChars
            ? content[..maxChars] + $"\n...[truncated, {content.Length - maxChars} more characters]"
            : content;
    }

    private string WriteFile(string relativePath, string content)
    {
        var fullPath = ResolvePath(relativePath);
        var dir = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        File.WriteAllText(fullPath, content);
        return $"Wrote {content.Length} characters to {relativePath}";
    }

    // Prevents the model from reading/writing outside the chosen folder.
    private string ResolvePath(string relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
        {
            throw new ArgumentException("path is required");
        }

        var combined = Path.GetFullPath(Path.Combine(_rootFolder, relativePath));
        var rootWithSep = _rootFolder.EndsWith(Path.DirectorySeparatorChar)
            ? _rootFolder
            : _rootFolder + Path.DirectorySeparatorChar;

        if (!combined.StartsWith(rootWithSep, StringComparison.Ordinal) && combined != _rootFolder)
        {
            throw new UnauthorizedAccessException($"path '{relativePath}' is outside the working folder");
        }

        return combined;
    }

    private static AgentAction? ParseAction(string raw)
    {
        var json = ExtractJson(raw);
        return JsonSerializer.Deserialize<AgentAction>(json, JsonOpts);
    }

    // Local models sometimes wrap JSON in prose or code fences despite instructions;
    // pull out the outermost {...} block before deserializing.
    private static string ExtractJson(string raw)
    {
        var start = raw.IndexOf('{');
        var end = raw.LastIndexOf('}');
        if (start < 0 || end < start)
        {
            throw new FormatException("no JSON object found in model output");
        }

        return raw[start..(end + 1)];
    }

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private static string BuildSystemPrompt(string rootFolder) => $$"""
        You are a coding agent working ONLY inside the folder: {{rootFolder}}
        All paths you use must be relative to that folder.

        You can take one action at a time. Respond with ONLY a single JSON object
        (no prose, no markdown fences) matching one of these shapes:

        List files matching a glob pattern (e.g. "*.cs", "**/*.json", "*"):
        {"action": "list_files", "pattern": "*.cs"}

        Read a file's contents:
        {"action": "read_file", "path": "relative/path.cs"}

        Create or overwrite a file:
        {"action": "write_file", "path": "relative/path.cs", "content": "file contents here"}

        Give your final answer to the user once the task is complete:
        {"action": "final", "answer": "what you did / the answer to the question"}

        Always use list_files or read_file to gather real information before writing
        files or answering. Do not guess at file contents. Keep taking actions until
        the task is actually done, then respond with "final".
        """;
}

record ChatMessage(string Role, string Content);

class AgentAction
{
    [JsonPropertyName("action")]
    public string Action { get; set; } = "";

    [JsonPropertyName("pattern")]
    public string? Pattern { get; set; }

    [JsonPropertyName("path")]
    public string? Path { get; set; }

    [JsonPropertyName("content")]
    public string? Content { get; set; }

    [JsonPropertyName("answer")]
    public string? Answer { get; set; }
}

// ======================================================================
// Console spinner shown while waiting on a slow operation
// ======================================================================

static class Spinner
{
    public static async Task<T> RunAsync<T>(Func<Task<T>> action)
    {
        using var cts = new CancellationTokenSource();
        var spinnerTask = AnimateAsync(cts.Token);
        try
        {
            return await action();
        }
        finally
        {
            cts.Cancel();
            await spinnerTask;
        }
    }

    private static async Task AnimateAsync(CancellationToken token)
    {
        var frames = new[] { '|', '/', '-', '\\' };
        var i = 0;
        try
        {
            while (!token.IsCancellationRequested)
            {
                Console.Write($"\r{frames[i++ % frames.Length]} thinking...");
                await Task.Delay(100, token);
            }
        }
        catch (TaskCanceledException)
        {
        }
        finally
        {
            // Erase the spinner line.
            Console.Write("\r" + new string(' ', 20) + "\r");
        }
    }
}

// ======================================================================
// Minimal Ollama HTTP client (POST /api/chat, non-streaming, JSON-constrained)
// ======================================================================

class OllamaClient
{
    private readonly HttpClient _http = new()
    {
        BaseAddress = new Uri("http://localhost:11434"),
        // Local models can take a while to respond; the HttpClient default of 100s is too short.
        Timeout = TimeSpan.FromMinutes(10)
    };

    public async Task<string> ChatAsync(string model, IReadOnlyList<ChatMessage> history)
    {
        var request = new
        {
            model,
            stream = false,
            format = "json",
            messages = history.Select(m => new { role = m.Role, content = m.Content })
        };

        var body = JsonSerializer.Serialize(request);
        using var content = new StringContent(body, Encoding.UTF8, "application/json");

        using var response = await _http.PostAsync("/api/chat", content);
        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(responseJson);
        return doc.RootElement.GetProperty("message").GetProperty("content").GetString() ?? "";
    }
}
