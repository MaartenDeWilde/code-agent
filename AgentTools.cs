using System.Diagnostics;
using System.Text;
using System.Text.Json;

/// <summary>
/// The tools the model can call, plus the sandboxing and confirmation rules
/// around them. All file access is restricted to the chosen working folder,
/// and destructive actions (overwrites, shell commands) require the user to
/// confirm on the console first.
/// </summary>
class AgentTools
{
    // Never listed: build output and dependency folders are huge and noisy.
    private static readonly string[] IgnoredFolders =
        { ".git", ".vs", ".idea", ".vscode", "bin", "obj", "node_modules", "dist", "target", "packages" };

    private readonly string _rootFolder;

    public AgentTools(string rootFolder) => _rootFolder = rootFolder;

    /// <summary>Tool schemas advertised to the model via Ollama's native tool calling.</summary>
    public static object[] Definitions =>
    [
        Tool("list_files",
            "List files in the working folder (recursively) matching a glob pattern.",
            new
            {
                pattern = Param("string", "Glob pattern, e.g. '*.cs', '*.json' or '*' for everything")
            }, "pattern"),

        Tool("read_file",
            "Read a file's contents.",
            new
            {
                path = Param("string", "File path relative to the working folder")
            }, "path"),

        Tool("write_file",
            "Create a new file or fully overwrite an existing one. Prefer edit_file for small changes.",
            new
            {
                path = Param("string", "File path relative to the working folder"),
                content = Param("string", "The complete new file contents")
            }, "path", "content"),

        Tool("edit_file",
            "Replace an exact text snippet in an existing file. The 'find' text must appear exactly once; include enough surrounding lines to make it unique.",
            new
            {
                path = Param("string", "File path relative to the working folder"),
                find = Param("string", "Exact existing text to replace (copied verbatim from the file)"),
                replace = Param("string", "The replacement text")
            }, "path", "find", "replace"),

        Tool("run_command",
            "Run a shell command inside the working folder and return its exit code and output. Use it to build, test or inspect the project. The user must approve each command.",
            new
            {
                command = Param("string", "The shell command to run, e.g. 'dotnet build'")
            }, "command")
    ];

    /// <summary>Executes a tool call and returns the result text to feed back to the model.</summary>
    public async Task<string> ExecuteAsync(string name, JsonElement args)
    {
        try
        {
            return name switch
            {
                "list_files" => ListFiles(GetString(args, "pattern") ?? "*"),
                "read_file" => ReadFile(GetString(args, "path") ?? ""),
                "write_file" => WriteFile(GetString(args, "path") ?? "", GetString(args, "content") ?? ""),
                "edit_file" => EditFile(GetString(args, "path") ?? "",
                                        GetString(args, "find") ?? "",
                                        GetString(args, "replace") ?? ""),
                "run_command" => await RunCommandAsync(GetString(args, "command") ?? ""),
                _ => $"Unknown tool '{name}'. Valid tools: list_files, read_file, write_file, edit_file, run_command."
            };
        }
        catch (Exception ex)
        {
            return $"Error: {ex.Message}";
        }
    }

    // ------------------------------------------------------------------
    // Individual tools
    // ------------------------------------------------------------------

    private string ListFiles(string pattern)
    {
        var files = Directory.EnumerateFiles(_rootFolder, pattern, SearchOption.AllDirectories)
            .Select(f => Path.GetRelativePath(_rootFolder, f))
            .Where(rel => !rel.Split(Path.DirectorySeparatorChar)
                .Any(segment => IgnoredFolders.Contains(segment, StringComparer.OrdinalIgnoreCase)))
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

        // Overwriting an existing file is destructive: show the size change
        // and let the user veto it.
        if (File.Exists(fullPath))
        {
            var existing = File.ReadAllText(fullPath);
            Console.WriteLine();
            Console.WriteLine($"The agent wants to overwrite '{relativePath}' ({CountLines(existing)} lines -> {CountLines(content)} lines).");
            if (!Confirm("Allow overwrite?"))
            {
                return "User declined the overwrite. Consider edit_file for a smaller change, or ask the user how to proceed.";
            }
        }

        var dir = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        File.WriteAllText(fullPath, content);
        return $"Wrote {content.Length} characters to {relativePath}";
    }

    private string EditFile(string relativePath, string find, string replace)
    {
        if (find.Length == 0)
        {
            return "Error: 'find' must not be empty.";
        }

        var fullPath = ResolvePath(relativePath);
        if (!File.Exists(fullPath))
        {
            return $"File not found: {relativePath}";
        }

        var content = File.ReadAllText(fullPath);

        // Demand a unique match so the model can't accidentally rewrite the
        // wrong occurrence.
        var first = content.IndexOf(find, StringComparison.Ordinal);
        if (first < 0)
        {
            return "Error: 'find' text not found in the file. Read the file again and copy the snippet exactly, including whitespace.";
        }
        if (content.IndexOf(find, first + 1, StringComparison.Ordinal) >= 0)
        {
            return "Error: 'find' text appears more than once. Include more surrounding context to make it unique.";
        }

        File.WriteAllText(fullPath, content[..first] + replace + content[(first + find.Length)..]);
        return $"Replaced {find.Length} characters with {replace.Length} characters in {relativePath}";
    }

    private async Task<string> RunCommandAsync(string command)
    {
        if (string.IsNullOrWhiteSpace(command))
        {
            return "Error: command is required.";
        }

        // Shell access is the most dangerous tool; every command needs an
        // explicit go-ahead from the user.
        Console.WriteLine();
        Console.WriteLine($"The agent wants to run: {command}");
        if (!Confirm("Allow?"))
        {
            return "User declined to run the command.";
        }

        var psi = new ProcessStartInfo
        {
            WorkingDirectory = _rootFolder,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        if (OperatingSystem.IsWindows())
        {
            psi.FileName = "cmd.exe";
            psi.ArgumentList.Add("/c");
            psi.ArgumentList.Add(command);
        }
        else
        {
            psi.FileName = "/bin/sh";
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add(command);
        }

        using var process = Process.Start(psi)!;

        // Read both streams concurrently so a full stderr buffer can't
        // deadlock the process while we wait on stdout.
        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();

        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(2));
        try
        {
            await process.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch { /* already gone */ }
            return "Error: command timed out after 2 minutes.";
        }

        var stdout = await stdoutTask;
        var stderr = await stderrTask;

        var output = new StringBuilder();
        output.AppendLine($"exit code: {process.ExitCode}");
        if (stdout.Trim().Length > 0)
        {
            output.AppendLine("stdout:").AppendLine(stdout.Trim());
        }
        if (stderr.Trim().Length > 0)
        {
            output.AppendLine("stderr:").AppendLine(stderr.Trim());
        }

        return TruncateOutput(output.ToString());
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

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

    // Asks the user for a y/n confirmation before a destructive action.
    private static bool Confirm(string question)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.Write($"{question} [y/N] ");
        Console.ResetColor();
        var answer = Console.ReadLine()?.Trim();
        return string.Equals(answer, "y", StringComparison.OrdinalIgnoreCase);
    }

    // Keeps the *tail* of long command output: build/test errors usually
    // appear at the end.
    private static string TruncateOutput(string text)
    {
        const int maxChars = 8000;
        return text.Length <= maxChars
            ? text
            : $"[truncated, showing last {maxChars} of {text.Length} characters]\n" + text[^maxChars..];
    }

    private static int CountLines(string text) => text.Split('\n').Length;

    private static string? GetString(JsonElement args, string name) =>
        args.ValueKind == JsonValueKind.Object
        && args.TryGetProperty(name, out var value)
        && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    // Shorthand builders for the JSON-schema tool definitions above.
    private static object Tool(string name, string description, object properties, params string[] required) => new
    {
        type = "function",
        function = new
        {
            name,
            description,
            parameters = new { type = "object", properties, required }
        }
    };

    private static object Param(string type, string description) => new { type, description };
}
