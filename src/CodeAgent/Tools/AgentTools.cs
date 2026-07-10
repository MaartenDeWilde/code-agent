using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace CodeAgent.Tools;

/// <summary>
/// The tools the model can call, plus the sandboxing and confirmation rules
/// around them. All file access is restricted to the chosen working folder,
/// and destructive actions (overwrites, shell commands) require the user to
/// confirm on the console first.
/// </summary>
class AgentTools
{
    // Never listed or searched: build output, caches and dependency folders
    // are huge and noisy. Shared with ProjectOrientation.
    public static readonly string[] IgnoredFolders =
    {
        ".git", ".vs", ".idea", ".vscode", "bin", "obj", "node_modules",
        "dist", "target", "packages", ".angular", "coverage", "TestResults", ".nuget"
    };

    // Skipped when searching: binary or generated formats that only waste time.
    private static readonly string[] UnsearchableExtensions =
    {
        ".dll", ".exe", ".pdb", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
        ".zip", ".gz", ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".pdf", ".map", ".min.js"
    };

    private readonly string _rootFolder;

    // Cap on a single tool result (file read, command output), sized by the
    // caller relative to the model's context window.
    private readonly int _maxResultChars;

    public AgentTools(string rootFolder, int maxResultChars)
    {
        _rootFolder = rootFolder;
        _maxResultChars = maxResultChars;
    }

    // All tool schemas, keyed by name, advertised via Ollama's native tool calling.
    private static readonly Dictionary<string, object> AllDefinitions = new()
    {
        ["list_files"] = Tool("list_files",
            "List files in the working folder (recursively) matching a glob pattern.",
            new
            {
                pattern = Param("string", "Glob pattern, e.g. '*.cs', '*.json' or '*' for everything")
            }, "pattern"),

        ["read_file"] = Tool("read_file",
            "Read a file's contents. Each line is prefixed with its line number ('  12: code') for reference. For large files, pass start_line/end_line to read a specific range.",
            new
            {
                path = Param("string", "File path relative to the working folder"),
                start_line = Param("integer", "Optional 1-based first line to read"),
                end_line = Param("integer", "Optional 1-based last line to read (inclusive)")
            }, "path"),

        ["search_files"] = Tool("search_files",
            "Search all file contents with a regular expression (case-insensitive). Returns matching lines as 'path:line: text'. Use this to find symbols, usages or text instead of reading files one by one.",
            new
            {
                pattern = Param("string", "Regular expression to search for, e.g. 'class OrderService' or 'toHaveBeenCalled'"),
                file_pattern = Param("string", "Optional filename glob to limit the search, e.g. '*.cs' or '*.ts'. Defaults to all files.")
            }, "pattern"),

        ["write_file"] = Tool("write_file",
            "Create a new file or fully overwrite an existing one. Prefer edit_file for small changes.",
            new
            {
                path = Param("string", "File path relative to the working folder"),
                content = Param("string", "The complete new file contents")
            }, "path", "content"),

        ["edit_file"] = Tool("edit_file",
            "Replace an exact text snippet in an existing file. The 'find' text must appear exactly once; include enough surrounding lines to make it unique. Copy only the code itself — never include read_file's line-number prefixes.",
            new
            {
                path = Param("string", "File path relative to the working folder"),
                find = Param("string", "Exact existing text to replace (copied verbatim from the file)"),
                replace = Param("string", "The replacement text")
            }, "path", "find", "replace"),

        ["run_command"] = Tool("run_command",
            "Run a shell command inside the working folder and return its exit code and output. Use it to build, test or inspect the project. The user must approve each command.",
            new
            {
                command = Param("string", "The shell command to run, e.g. 'dotnet build'")
            }, "command"),

        ["ask_user"] = Tool("ask_user",
            "Ask the user a clarifying question and wait for their reply. Use this when the task is ambiguous or you must choose between approaches, instead of guessing.",
            new
            {
                question = Param("string", "The question to ask the user")
            }, "question")
    };

    // Plan mode may look around and ask questions, but never change anything.
    private static readonly string[] PlanModeTools = { "list_files", "read_file", "search_files", "ask_user" };

    /// <summary>Every tool, for normal execution mode.</summary>
    public static object[] Definitions => AllDefinitions.Values.ToArray();

    /// <summary>The read-only subset advertised in plan mode.</summary>
    public static object[] PlanDefinitions =>
        PlanModeTools.Select(name => AllDefinitions[name]).ToArray();

    public static bool IsAllowedInPlanMode(string name) => PlanModeTools.Contains(name);

    /// <summary>Executes a tool call and returns the result text to feed back to the model.</summary>
    public async Task<string> ExecuteAsync(string name, JsonElement args)
    {
        try
        {
            return name switch
            {
                "list_files" => ListFiles(GetString(args, "pattern") ?? "*"),
                "read_file" => ReadFile(GetString(args, "path") ?? "",
                                        GetInt(args, "start_line"),
                                        GetInt(args, "end_line")),
                "search_files" => SearchFiles(GetString(args, "pattern") ?? "",
                                              GetString(args, "file_pattern") ?? "*"),
                "write_file" => WriteFile(GetString(args, "path") ?? "", GetString(args, "content") ?? ""),
                "edit_file" => EditFile(GetString(args, "path") ?? "",
                                        GetString(args, "find") ?? "",
                                        GetString(args, "replace") ?? ""),
                "run_command" => await RunCommandAsync(GetString(args, "command") ?? ""),
                "ask_user" => AskUser(GetString(args, "question") ?? ""),
                _ => $"Unknown tool '{name}'. Valid tools: {string.Join(", ", AllDefinitions.Keys)}."
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

    // Relative paths of all files under the root, with clutter filtered out.
    private IEnumerable<string> EnumerateSourceFiles(string pattern)
    {
        // The glob is model-controlled input. It is a filename filter, not a
        // path: refuse separators and '..' so it can never reach outside the
        // working folder, whatever Directory.EnumerateFiles would do with it.
        if (pattern.Contains("..") || pattern.Contains('/') || pattern.Contains('\\'))
        {
            throw new ArgumentException(
                $"pattern '{pattern}' must be a plain filename glob like '*.cs', without path separators or '..'");
        }

        return Directory.EnumerateFiles(_rootFolder, pattern, SearchOption.AllDirectories)
            .Select(f => Path.GetRelativePath(_rootFolder, f))
            .Where(rel => !rel.Split(Path.DirectorySeparatorChar)
                .Any(segment => IgnoredFolders.Contains(segment, StringComparer.OrdinalIgnoreCase)));
    }

    private string ListFiles(string pattern)
    {
        var files = EnumerateSourceFiles(pattern).Take(200).ToList();
        return files.Count == 0
            ? "No files matched."
            : string.Join("\n", files);
    }

    private string ReadFile(string relativePath, int? startLine, int? endLine)
    {
        var fullPath = ResolvePath(relativePath);
        if (!File.Exists(fullPath))
        {
            return $"File not found: {relativePath}";
        }

        var lines = File.ReadAllLines(fullPath);
        var from = Math.Max(startLine ?? 1, 1);
        var to = Math.Min(endLine ?? lines.Length, lines.Length);
        if (from > to)
        {
            return $"Error: line range {from}-{endLine} is out of bounds; the file has {lines.Length} lines.";
        }

        var result = new StringBuilder();
        if (from > 1 || to < lines.Length)
        {
            result.AppendLine($"[lines {from}-{to} of {lines.Length}]");
        }

        for (var lineNo = from; lineNo <= to; lineNo++)
        {
            result.AppendLine($"{lineNo,5}: {lines[lineNo - 1]}");
            if (result.Length > _maxResultChars)
            {
                result.AppendLine($"...[truncated at line {lineNo} of {lines.Length}; call read_file with start_line={lineNo + 1} to continue]");
                break;
            }
        }

        return result.ToString();
    }

    private string SearchFiles(string pattern, string filePattern)
    {
        if (string.IsNullOrWhiteSpace(pattern))
        {
            return "Error: pattern is required.";
        }

        Regex regex;
        try
        {
            // The timeout guards against catastrophic backtracking in
            // model-written patterns.
            regex = new Regex(pattern, RegexOptions.IgnoreCase, TimeSpan.FromSeconds(2));
        }
        catch (ArgumentException ex)
        {
            return $"Error: invalid regular expression: {ex.Message}";
        }

        const int maxMatches = 100;
        var matches = new List<string>();

        foreach (var relativePath in EnumerateSourceFiles(filePattern))
        {
            if (UnsearchableExtensions.Any(ext => relativePath.EndsWith(ext, StringComparison.OrdinalIgnoreCase)))
            {
                continue;
            }

            var fullPath = Path.Combine(_rootFolder, relativePath);
            string[] lines;
            try
            {
                // Skip anything huge; source files aren't megabytes.
                if (new FileInfo(fullPath).Length > 2_000_000)
                {
                    continue;
                }
                lines = File.ReadAllLines(fullPath);
            }
            catch
            {
                continue; // unreadable file; not worth failing the search
            }

            for (var i = 0; i < lines.Length; i++)
            {
                try
                {
                    if (!regex.IsMatch(lines[i]))
                    {
                        continue;
                    }
                }
                catch (RegexMatchTimeoutException)
                {
                    return "Error: the regular expression is too slow; simplify it.";
                }

                var text = lines[i].Trim();
                matches.Add($"{relativePath}:{i + 1}: {(text.Length > 200 ? text[..200] + "..." : text)}");
                if (matches.Count >= maxMatches)
                {
                    return string.Join("\n", matches)
                        + $"\n[stopped after {maxMatches} matches — narrow the pattern or file_pattern]";
                }
            }
        }

        return matches.Count == 0
            ? "No matches found."
            : string.Join("\n", matches);
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
            // Models often paste read_file's "  12: " line-number prefixes
            // into 'find'; strip them from both fields and retry.
            var strippedFind = StripLineNumbers(find);
            if (strippedFind != find && content.Contains(strippedFind, StringComparison.Ordinal))
            {
                find = strippedFind;
                replace = StripLineNumbers(replace);
                first = content.IndexOf(find, StringComparison.Ordinal);
            }
            else
            {
                return "Error: 'find' text not found in the file. Read the file again and copy the snippet exactly, including whitespace and without line-number prefixes.";
            }
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

    // Lets the model pause mid-task and get input from the user.
    private static string AskUser(string question)
    {
        if (string.IsNullOrWhiteSpace(question))
        {
            return "Error: question is required.";
        }

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"The agent asks: {question}");
        Console.Write("your answer> ");
        Console.ResetColor();
        var answer = Console.ReadLine()?.Trim();

        return string.IsNullOrWhiteSpace(answer)
            ? "The user gave no answer. Use your best judgment."
            : $"User answered: {answer}";
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

        // Textual containment: GetFullPath has normalized away '..' segments,
        // and an absolute input to Path.Combine simply replaces the root, so
        // anything that still starts with the root prefix is textually inside.
        if (!combined.StartsWith(rootWithSep, StringComparison.Ordinal) && combined != _rootFolder)
        {
            throw new UnauthorizedAccessException($"path '{relativePath}' is outside the working folder");
        }

        // Physical containment: GetFullPath is purely textual and does NOT
        // resolve symlinks, so 'link/file' where link points elsewhere would
        // pass the check above yet land outside the folder. Refuse any
        // symlink on the way to the target.
        RejectSymlinkComponents(combined, relativePath);

        return combined;
    }

    // Walks each path component below the root and throws if one is a
    // symbolic link. (The root itself may be a symlink — e.g. /tmp on macOS —
    // that's the user's explicit choice and is trusted.)
    private void RejectSymlinkComponents(string fullPath, string originalInput)
    {
        var remainder = Path.GetRelativePath(_rootFolder, fullPath);
        if (remainder == ".")
        {
            return;
        }

        var current = _rootFolder;
        foreach (var component in remainder.Split(Path.DirectorySeparatorChar))
        {
            current = Path.Combine(current, component);

            // LinkTarget is non-null only for symlinks; DirectoryInfo vs
            // FileInfo matters for directories, and FileInfo still reports
            // dangling links whose target no longer exists.
            FileSystemInfo info = Directory.Exists(current)
                ? new DirectoryInfo(current)
                : new FileInfo(current);
            if (info.LinkTarget is not null)
            {
                throw new UnauthorizedAccessException(
                    $"path '{originalInput}' goes through the symbolic link '{component}', which could point outside the working folder");
            }
        }
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
    private string TruncateOutput(string text) =>
        text.Length <= _maxResultChars
            ? text
            : $"[truncated, showing last {_maxResultChars:N0} of {text.Length:N0} characters]\n" + text[^_maxResultChars..];

    private static int CountLines(string text) => text.Split('\n').Length;

    // Removes read_file's "  12: " prefixes from every line of a snippet.
    private static string StripLineNumbers(string text) =>
        Regex.Replace(text, @"^\s*\d+: ", "", RegexOptions.Multiline);

    private static int? GetInt(JsonElement args, string name) =>
        args.ValueKind == JsonValueKind.Object && args.TryGetProperty(name, out var value)
            ? value.ValueKind switch
            {
                JsonValueKind.Number when value.TryGetInt32(out var n) => n,
                // Models sometimes send numbers as strings.
                JsonValueKind.String when int.TryParse(value.GetString(), out var n) => n,
                _ => null
            }
            : null;

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
