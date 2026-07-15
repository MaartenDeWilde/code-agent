using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using CodeAgent.Guardrails;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

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

    // Session-scoped approvals collected via the 'a' confirmation answer:
    // full paths whose overwrites are pre-approved, and command programs
    // (first token, e.g. 'dotnet') that may run without asking again.
    private readonly HashSet<string> _approvedOverwrites = new();
    private readonly HashSet<string> _approvedCommandPrograms = new();

    // How often a whole-file rewrite was rejected per path; after the first
    // rejection the model gets the benefit of the doubt and the user decides.
    private readonly Dictionary<string, int> _rewriteRejections = new();

    private readonly ILogger _logger;

    public AgentTools(string rootFolder, int maxResultChars, ILogger? logger = null)
    {
        _rootFolder = rootFolder;
        _maxResultChars = maxResultChars;
        _logger = logger ?? NullLogger.Instance;
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
            "Create a new file, writing its COMPLETE content in one call. To change an existing file use edit_file instead: rewrites that keep most of the existing content are rejected.",
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
                command = Param("string", "The shell command to run, e.g. 'dotnet build'"),
                timeout_seconds = Param("integer", "Optional timeout in seconds (default 120, max 600). Raise it for slow commands like package installs or full test runs.")
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
                "run_command" => await RunCommandAsync(GetString(args, "command") ?? "",
                                                       GetInt(args, "timeout_seconds")),
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
        const int maxListed = 200;

        // Take one extra so a truncated listing is distinguishable from one
        // that happens to be exactly at the cap.
        var files = EnumerateSourceFiles(pattern).Take(maxListed + 1).ToList();
        if (files.Count == 0)
        {
            return "No files matched.";
        }

        // Without this marker the model may conclude an unlisted file doesn't
        // exist and create a duplicate of it.
        return files.Count <= maxListed
            ? string.Join("\n", files)
            : string.Join("\n", files.Take(maxListed))
              + $"\n[stopped after {maxListed} files — narrow the pattern to see the rest]";
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

            // Models tend to regenerate a whole file to change (or worse,
            // truncate) a small part of it. Redirect those to edit_file
            // instead of executing them.
            if (content == existing)
            {
                return $"No change: {relativePath} already has exactly this content.";
            }
            if (IsMostlyUnchanged(existing, content))
            {
                // Redirect to edit_file once. If the model keeps insisting —
                // a legitimate large refactor (reordering methods, say) also
                // looks 'mostly unchanged' — stop blocking and let the user
                // decide via the confirmation below instead of dead-ending.
                var rejections = _rewriteRejections.GetValueOrDefault(fullPath) + 1;
                _rewriteRejections[fullPath] = rejections;
                if (rejections < 2)
                {
                    return "Error: most of this content is already in the existing file. Do not rewrite " +
                           "the whole file; use edit_file with an exact 'find' snippet and 'replace' text " +
                           "for just the part you want to change.";
                }
            }

            if (!_approvedOverwrites.Contains(fullPath))
            {
                Console.WriteLine();
                Console.WriteLine($"The agent wants to overwrite '{relativePath}' ({CountLines(existing)} lines -> {CountLines(content)} lines).");
                switch (Confirm("Allow overwrite?"))
                {
                    case ConfirmResult.No:
                        return "User declined the overwrite. Consider edit_file for a smaller change, or ask the user how to proceed.";
                    case ConfirmResult.Always:
                        _approvedOverwrites.Add(fullPath);
                        break;
                }
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
                // Last resort: the snippet is often right except for its
                // indentation (tabs vs spaces, a re-wrapped copy). Try a
                // line-by-line match with whitespace trimmed.
                return EditIgnoringIndentation(fullPath, relativePath, content,
                    StripLineNumbers(find), StripLineNumbers(replace));
            }
        }
        if (content.IndexOf(find, first + 1, StringComparison.Ordinal) >= 0)
        {
            return "Error: 'find' text appears more than once. Include more surrounding context to make it unique.";
        }

        var updated = content[..first] + replace + content[(first + find.Length)..];
        File.WriteAllText(fullPath, updated);

        // Echo the edited region back as it now exists on disk. The model's
        // only other picture of the file is an earlier read_file, which this
        // edit just made stale; without fresh text, follow-up edits tend to
        // 'find' snippets that no longer exist.
        var firstChangedLine = CountLines(content[..first]);
        return $"Replaced {find.Length} characters with {replace.Length} characters in {relativePath}.\n"
            + SnippetAround(updated, firstChangedLine, CountLines(replace));
    }

    // Fallback for the most common small-model mistake in edit_file: the
    // 'find' lines are right but their indentation is not. Matches the file
    // line by line with surrounding whitespace ignored; the match must still
    // be unique, and the replacement is re-indented to what the file really
    // uses so the fix doesn't introduce misaligned code.
    private string EditIgnoringIndentation(string fullPath, string relativePath, string content, string find, string replace)
    {
        var fileLines = content.Split('\n');
        var findLines = find.Split('\n');

        // Blank edge lines the model tends to add around a snippet carry no
        // information and would prevent a match at the file's borders.
        var start = Array.FindIndex(findLines, line => line.Trim().Length > 0);
        var end = Array.FindLastIndex(findLines, line => line.Trim().Length > 0);
        if (start < 0)
        {
            return "Error: 'find' must contain at least one non-blank line.";
        }
        findLines = findLines[start..(end + 1)];
        var trimmedFind = findLines.Select(line => line.Trim()).ToArray();

        var matches = new List<int>();
        for (var i = 0; i + trimmedFind.Length <= fileLines.Length; i++)
        {
            var equal = true;
            for (var j = 0; j < trimmedFind.Length && equal; j++)
            {
                equal = fileLines[i + j].Trim() == trimmedFind[j];
            }
            if (equal)
            {
                matches.Add(i);
            }
        }

        if (matches.Count == 0)
        {
            return "Error: 'find' text not found in the file. Read the file again and copy the snippet exactly, including whitespace and without line-number prefixes.";
        }
        if (matches.Count > 1)
        {
            return "Error: 'find' text appears more than once (even ignoring indentation). Include more surrounding context to make it unique.";
        }

        var matchLine = matches[0];
        var fileIndent = LeadingWhitespace(fileLines[matchLine]);
        var findIndent = LeadingWhitespace(findLines[0]);
        var replaceLines = replace.Trim().Length == 0
            ? Array.Empty<string>() // an empty replacement deletes the lines
            : Reindent(replace, fileIndent, findIndent);

        // Splitting a CRLF file on '\n' leaves the '\r' on every kept line;
        // inserted lines need it too, or the file ends up with mixed endings.
        if (content.Contains("\r\n"))
        {
            replaceLines = replaceLines
                .Select(line => line.EndsWith('\r') ? line : line + "\r")
                .ToArray();
        }

        var updated = string.Join('\n',
            fileLines[..matchLine].Concat(replaceLines).Concat(fileLines[(matchLine + trimmedFind.Length)..]));
        File.WriteAllText(fullPath, updated);

        // Tell the model what was corrected, and only what was corrected —
        // a claim of re-indentation when only a stray space differed would
        // teach it the wrong lesson.
        var note = fileIndent == findIndent
            ? "your 'find' did not match the file exactly (stray whitespace?); it was matched with whitespace relaxed"
            : "your 'find' had different indentation than the file; the replacement was re-indented to match";
        return $"Replaced {trimmedFind.Length} lines in {relativePath} ({note}).\n"
            + SnippetAround(updated, matchLine + 1, Math.Max(replaceLines.Length, 1));
    }

    private static string LeadingWhitespace(string line) => line[..(line.Length - line.TrimStart().Length)];

    // Adjusts a replacement written against the model's (wrong) indentation
    // to the file's real one: the leading whitespace the 'find' snippet used
    // is swapped for the whitespace of the line that actually matched. Only
    // lines that share the snippet's base indentation are shifted; anything
    // else is left alone rather than guessed at.
    private static string[] Reindent(string replace, string fileIndent, string findIndent)
    {
        if (fileIndent == findIndent)
        {
            return replace.Split('\n');
        }

        return replace.Split('\n')
            .Select(line => line.Trim().Length > 0 && line.StartsWith(findIndent, StringComparison.Ordinal)
                ? fileIndent + line[findIndent.Length..]
                : line)
            .ToArray();
    }

    // The lines spanned by an edit plus a little context, in read_file's
    // numbered format. Bounded so a large replacement (which the model just
    // wrote itself) isn't echoed back in full.
    private static string SnippetAround(string content, int firstLine, int lineCount)
    {
        const int contextLines = 4;
        const int maxSnippetLines = 40;

        var lines = content.Split('\n');
        var from = Math.Max(firstLine - contextLines, 1);
        var to = Math.Min(firstLine + lineCount - 1 + contextLines, lines.Length);
        var truncated = to - from + 1 > maxSnippetLines;
        if (truncated)
        {
            to = from + maxSnippetLines - 1;
        }

        var snippet = new StringBuilder();
        snippet.AppendLine($"The file now reads (lines {from}-{to} of {lines.Length}):");
        for (var lineNo = from; lineNo <= to; lineNo++)
        {
            snippet.AppendLine($"{lineNo,5}: {lines[lineNo - 1]}");
        }
        if (truncated)
        {
            snippet.AppendLine("...[rest of the edited region not shown; read_file to see more]");
        }

        return snippet.ToString().TrimEnd();
    }

    private async Task<string> RunCommandAsync(string command, int? timeoutSeconds)
    {
        if (string.IsNullOrWhiteSpace(command))
        {
            return "Error: command is required.";
        }

        // Command rail: known-catastrophic commands are refused outright,
        // before the user is even asked. User confirmation protects against
        // an overeager model; this protects against a confirmation clicked
        // through on autopilot.
        var verdict = GuardrailPipeline.CheckCommand(command);
        if (!verdict.Allowed)
        {
            _logger.LogWarning("Guardrail {Rule} blocked command: {Reason}. Command: {Command}",
                verdict.Rule, verdict.Reason, command);
            return $"Blocked by guardrail '{verdict.Rule}': {verdict.Reason}. This command will not be run; take a safer approach.";
        }

        // Shell access is the most dangerous tool; every command needs an
        // explicit go-ahead from the user. 'a' approves the command's program
        // (first token) for the session, so 'dotnet build' also covers
        // 'dotnet test' — exact-string approval made iterative build/fix
        // loops prompt on every variation.
        var program = command.TrimStart().Split(' ', 2)[0];
        if (ContainsShellOperators(command))
        {
            // The command runs under 'sh -c', where ';', '&&', '|', '$()' and
            // redirects can chain an innocent-looking program into anything.
            // Program-level session approval must not cover those, so they
            // are confirmed individually, every time.
            Console.WriteLine();
            Console.WriteLine($"The agent wants to run: {command}");
            if (!ConfirmOnce("Allow? (uses shell operators, so session approvals do not apply)"))
            {
                return "User declined to run the command.";
            }
        }
        else if (!_approvedCommandPrograms.Contains(program))
        {
            Console.WriteLine();
            Console.WriteLine($"The agent wants to run: {command}");
            switch (Confirm("Allow?", $"allow '{program}' commands this session"))
            {
                case ConfirmResult.No:
                    return "User declined to run the command.";
                case ConfirmResult.Always:
                    _approvedCommandPrograms.Add(program);
                    break;
            }
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

        // The model can raise the default for slow commands (installs, full
        // test runs), within a hard cap so a typo can't stall the agent.
        var seconds = Math.Clamp(timeoutSeconds ?? 120, 1, 600);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(seconds));
        try
        {
            await process.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch { /* already gone */ }
            return $"Error: command timed out after {seconds} seconds. Pass a larger timeout_seconds if it legitimately needs longer.";
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

    // Characters that let a command chain, substitute or redirect under
    // 'sh -c'. Plain '$VAR' expansion is allowed; '$(' command substitution
    // is not.
    private static bool ContainsShellOperators(string command) =>
        command.IndexOfAny(new[] { ';', '&', '|', '>', '<', '`', '\n' }) >= 0
        || command.Contains("$(");

    private enum ConfirmResult { No, Yes, Always }

    // A yes/no confirmation without the session-wide 'always' option, for
    // actions where a blanket approval would be unsafe.
    private static bool ConfirmOnce(string question)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.Write($"{question} [y]es / [N]o ");
        Console.ResetColor();
        return Console.ReadLine()?.Trim().ToLowerInvariant() is "y" or "yes";
    }

    // Asks the user to confirm a destructive action. 'a' grants a
    // session-scoped approval; alwaysDescription spells out exactly what
    // would be approved (an overwrite target, a command program).
    private static ConfirmResult Confirm(string question, string alwaysDescription = "this session")
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.Write($"{question} [y]es / [a]lways {alwaysDescription} / [N]o ");
        Console.ResetColor();
        return Console.ReadLine()?.Trim().ToLowerInvariant() switch
        {
            "y" or "yes" => ConfirmResult.Yes,
            "a" or "always" => ConfirmResult.Always,
            _ => ConfirmResult.No
        };
    }

    // True when the bulk of the proposed content already exists in the file,
    // i.e. the model is rewriting a file to change — or truncate — a small
    // part of it. Trivial lines like '}' are ignored so brace-heavy code
    // doesn't look more similar than it is; small files may always be
    // rewritten, since an edit would hardly be cheaper.
    private static bool IsMostlyUnchanged(string existing, string proposed)
    {
        static List<string> SignificantLines(string text) =>
            text.Split('\n').Select(line => line.Trim()).Where(line => line.Length > 3).ToList();

        var oldLines = SignificantLines(existing);
        var newLines = SignificantLines(proposed);
        if (oldLines.Count < 10 || newLines.Count == 0)
        {
            return false;
        }

        // Multiset intersection: how many proposed lines the file already has.
        var available = new Dictionary<string, int>();
        foreach (var line in oldLines)
        {
            available[line] = available.GetValueOrDefault(line) + 1;
        }

        var shared = 0;
        foreach (var line in newLines)
        {
            if (available.TryGetValue(line, out var count) && count > 0)
            {
                available[line] = count - 1;
                shared++;
            }
        }

        return shared >= newLines.Count * 0.8;
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

    // Also used by CodingAgent to describe tool calls on the console.
    internal static int? GetInt(JsonElement args, string name) =>
        args.ValueKind == JsonValueKind.Object && args.TryGetProperty(name, out var value)
            ? value.ValueKind switch
            {
                JsonValueKind.Number when value.TryGetInt32(out var n) => n,
                // Models sometimes send numbers as strings.
                JsonValueKind.String when int.TryParse(value.GetString(), out var n) => n,
                _ => null
            }
            : null;

    internal static string? GetString(JsonElement args, string name) =>
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
