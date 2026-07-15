using System.Text.Json;
using CodeAgent.Tools;

namespace CodeAgent.Tests;

public class AgentToolsTests : IDisposable
{
    private readonly string _root;
    private readonly AgentTools _tools;

    public AgentToolsTests()
    {
        _root = Directory.CreateTempSubdirectory("agenttools-test-").FullName;
        _tools = new AgentTools(_root, maxResultChars: 10_000);
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);

    // Serializes an anonymous object the way Ollama delivers tool arguments.
    private static JsonElement Args(object values) => JsonSerializer.SerializeToElement(values);

    private void CreateFile(string relativePath, string content)
    {
        var full = Path.Combine(_root, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full, content);
    }

    // Simulates the user typing answers to console prompts (confirmations, ask_user).
    private static void UserTypes(string input) => Console.SetIn(new StringReader(input));

    // ---------------- read_file ----------------

    [Fact]
    public async Task ReadFile_PrefixesLineNumbers()
    {
        CreateFile("a.txt", "alpha\nbeta\ngamma");

        var result = await _tools.ExecuteAsync("read_file", Args(new { path = "a.txt" }));

        Assert.Contains("1: alpha", result);
        Assert.Contains("3: gamma", result);
    }

    [Fact]
    public async Task ReadFile_ReturnsRequestedRangeOnly()
    {
        CreateFile("a.txt", "alpha\nbeta\ngamma");

        var result = await _tools.ExecuteAsync("read_file",
            Args(new { path = "a.txt", start_line = 2, end_line = 2 }));

        Assert.Contains("[lines 2-2 of 3]", result);
        Assert.Contains("2: beta", result);
        Assert.DoesNotContain("alpha", result);
        Assert.DoesNotContain("gamma", result);
    }

    [Fact]
    public async Task ReadFile_RangeBeyondEnd_ReturnsError()
    {
        CreateFile("a.txt", "alpha");

        var result = await _tools.ExecuteAsync("read_file",
            Args(new { path = "a.txt", start_line = 10 }));

        Assert.StartsWith("Error:", result);
    }

    [Fact]
    public async Task ReadFile_OutsideWorkingFolder_IsDenied()
    {
        var result = await _tools.ExecuteAsync("read_file", Args(new { path = "../outside.txt" }));

        Assert.StartsWith("Error:", result);
        Assert.Contains("outside the working folder", result);
    }

    // ---------------- write_file ----------------

    [Fact]
    public async Task WriteFile_CreatesFileAndParentFolders()
    {
        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = "sub/dir/new.txt", content = "hello" }));

        Assert.StartsWith("Wrote", result);
        Assert.Equal("hello", File.ReadAllText(Path.Combine(_root, "sub/dir/new.txt")));
    }

    [Fact]
    public async Task WriteFile_OverwriteDeclined_LeavesFileUntouched()
    {
        CreateFile("a.txt", "original");
        UserTypes("n\n");

        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = "a.txt", content = "replaced" }));

        Assert.Contains("declined", result);
        Assert.Equal("original", File.ReadAllText(Path.Combine(_root, "a.txt")));
    }

    [Fact]
    public async Task WriteFile_OverwriteAccepted_ReplacesContent()
    {
        CreateFile("a.txt", "original");
        UserTypes("y\n");

        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = "a.txt", content = "replaced" }));

        Assert.StartsWith("Wrote", result);
        Assert.Equal("replaced", File.ReadAllText(Path.Combine(_root, "a.txt")));
    }

    // A file with enough significant lines to activate the rewrite guard.
    private static string ManyLines(string prefix, int count) =>
        string.Join("\n", Enumerable.Range(1, count).Select(i => $"{prefix} line number {i};"));

    [Fact]
    public async Task WriteFile_IdenticalContent_ReportsNoChange()
    {
        CreateFile("a.txt", "same content");

        // No console input is queued: if this prompted, it would be declined.
        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = "a.txt", content = "same content" }));

        Assert.StartsWith("No change", result);
    }

    [Fact]
    public async Task WriteFile_MinorChange_RedirectsToEditFile()
    {
        var original = ManyLines("alpha", 20);
        CreateFile("a.cs", original);
        var oneLineChanged = original.Replace("alpha line number 7;", "alpha line number seven;");

        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = "a.cs", content = oneLineChanged }));

        Assert.Contains("edit_file", result);
        Assert.Equal(original, File.ReadAllText(Path.Combine(_root, "a.cs"))); // untouched
    }

    [Fact]
    public async Task WriteFile_Truncation_RedirectsToEditFile()
    {
        // The dangerous pattern: rewriting a 96-line file with only its
        // first 15 lines. Every proposed line already exists -> rejected.
        var original = ManyLines("alpha", 96);
        CreateFile("a.cs", original);
        var truncated = ManyLines("alpha", 15);

        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = "a.cs", content = truncated }));

        Assert.Contains("edit_file", result);
        Assert.Equal(original, File.ReadAllText(Path.Combine(_root, "a.cs")));
    }

    [Fact]
    public async Task WriteFile_SubstantiallyDifferentContent_StillAsksAndOverwrites()
    {
        CreateFile("a.cs", ManyLines("alpha", 20));
        UserTypes("y\n");

        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = "a.cs", content = ManyLines("beta", 20) }));

        Assert.StartsWith("Wrote", result);
    }

    [Fact]
    public async Task WriteFile_AlwaysApproval_SkipsLaterPromptsForSamePath()
    {
        CreateFile("a.cs", ManyLines("alpha", 20));
        UserTypes("a\n"); // approve overwrites of a.cs for the session

        var first = await _tools.ExecuteAsync("write_file",
            Args(new { path = "a.cs", content = ManyLines("beta", 20) }));
        // No input queued now: a second prompt would read null and decline.
        var second = await _tools.ExecuteAsync("write_file",
            Args(new { path = "a.cs", content = ManyLines("gamma", 20) }));

        Assert.StartsWith("Wrote", first);
        Assert.StartsWith("Wrote", second);
    }

    // ---------------- edit_file ----------------

    [Fact]
    public async Task EditFile_ReplacesUniqueSnippet()
    {
        CreateFile("a.cs", "var x = 1;\nvar y = 2;\n");

        var result = await _tools.ExecuteAsync("edit_file",
            Args(new { path = "a.cs", find = "var y = 2;", replace = "var y = 3;" }));

        Assert.StartsWith("Replaced", result);
        Assert.Contains("var y = 3;", File.ReadAllText(Path.Combine(_root, "a.cs")));
    }

    [Fact]
    public async Task EditFile_ReturnsEditedRegionWithContext()
    {
        // Ten numbered lines; the edit lands on line 6.
        CreateFile("a.cs", ManyLines("alpha", 10));

        var result = await _tools.ExecuteAsync("edit_file",
            Args(new { path = "a.cs", find = "alpha line number 6;", replace = "alpha line number SIX;" }));

        // The snippet shows the new text in read_file's numbered format plus
        // surrounding context, so the model can follow up without re-reading.
        Assert.Contains("The file now reads (lines 2-10 of 10)", result);
        Assert.Contains("6: alpha line number SIX;", result);
        Assert.Contains("2: alpha line number 2;", result); // context above
        Assert.Contains("10: alpha line number 10;", result); // context below
    }

    [Fact]
    public async Task EditFile_LargeReplacement_TruncatesSnippet()
    {
        CreateFile("a.cs", "start\nMIDDLE\nend");

        var result = await _tools.ExecuteAsync("edit_file",
            Args(new { path = "a.cs", find = "MIDDLE", replace = ManyLines("beta", 100) }));

        // The model wrote those 100 lines itself; echoing them all back would
        // only burn context.
        Assert.StartsWith("Replaced", result);
        Assert.Contains("not shown", result);
        Assert.DoesNotContain("beta line number 60;", result);
    }

    [Fact]
    public async Task EditFile_AmbiguousMatch_ReturnsError()
    {
        CreateFile("a.cs", "int a;\nint a;\n");

        var result = await _tools.ExecuteAsync("edit_file",
            Args(new { path = "a.cs", find = "int a;", replace = "int b;" }));

        Assert.Contains("more than once", result);
    }

    [Fact]
    public async Task EditFile_MissingText_ReturnsError()
    {
        CreateFile("a.cs", "int a;\n");

        var result = await _tools.ExecuteAsync("edit_file",
            Args(new { path = "a.cs", find = "int z;", replace = "int b;" }));

        Assert.Contains("not found", result);
    }

    [Fact]
    public async Task EditFile_StripsPastedLineNumberPrefixes()
    {
        CreateFile("a.cs", "var x = 1;\nvar y = 2;\n");

        // Models often paste read_file's numbered output into 'find'.
        var result = await _tools.ExecuteAsync("edit_file",
            Args(new { path = "a.cs", find = "    2: var y = 2;", replace = "    2: var y = 9;" }));

        Assert.StartsWith("Replaced", result);
        Assert.Contains("var y = 9;", File.ReadAllText(Path.Combine(_root, "a.cs")));
        Assert.DoesNotContain("2:", File.ReadAllText(Path.Combine(_root, "a.cs")));
    }

    // ---------------- search_files ----------------

    [Fact]
    public async Task SearchFiles_FindsMatchesCaseInsensitively()
    {
        CreateFile("src/OrderService.cs", "public class OrderService\n{\n}");
        CreateFile("src/other.txt", "class orderservice mention");

        var result = await _tools.ExecuteAsync("search_files",
            Args(new { pattern = "class orderservice", file_pattern = "*.cs" }));

        Assert.Contains("OrderService.cs:1:", result);
        Assert.DoesNotContain("other.txt", result); // filtered by file_pattern
    }

    [Fact]
    public async Task SearchFiles_IgnoresClutterFolders()
    {
        CreateFile("node_modules/lib/index.js", "class OrderService");
        CreateFile("bin/gen.cs", "class OrderService");
        CreateFile("src/real.cs", "class OrderService");

        var result = await _tools.ExecuteAsync("search_files",
            Args(new { pattern = "OrderService" }));

        Assert.Contains("real.cs", result);
        Assert.DoesNotContain("node_modules", result);
        Assert.DoesNotContain("bin", result);
    }

    [Fact]
    public async Task SearchFiles_InvalidRegex_ReturnsError()
    {
        var result = await _tools.ExecuteAsync("search_files", Args(new { pattern = "(" }));

        Assert.Contains("invalid regular expression", result);
    }

    // ---------------- list_files ----------------

    [Fact]
    public async Task ListFiles_IgnoresClutterFolders()
    {
        CreateFile("node_modules/pkg/x.js", "x");
        CreateFile("obj/tmp.cs", "x");
        CreateFile("src/app.ts", "x");

        var result = await _tools.ExecuteAsync("list_files", Args(new { pattern = "*" }));

        Assert.Contains("app.ts", result);
        Assert.DoesNotContain("node_modules", result);
        Assert.DoesNotContain("obj", result);
    }

    // ---------------- run_command ----------------

    [Fact]
    public async Task RunCommand_Approved_ReturnsExitCodeAndOutput()
    {
        UserTypes("y\n");

        var result = await _tools.ExecuteAsync("run_command", Args(new { command = "echo hello" }));

        Assert.Contains("exit code: 0", result);
        Assert.Contains("hello", result);
    }

    [Fact]
    public async Task RunCommand_Declined_DoesNotRun()
    {
        UserTypes("n\n");

        var result = await _tools.ExecuteAsync("run_command",
            Args(new { command = $"echo leaked > {_root}/leak.txt" }));

        Assert.Contains("declined", result);
        Assert.False(File.Exists(Path.Combine(_root, "leak.txt")));
    }

    [Fact]
    public async Task RunCommand_AlwaysApproval_SkipsLaterPromptsForSameCommand()
    {
        UserTypes("a\n"); // approve this exact command for the session

        var first = await _tools.ExecuteAsync("run_command", Args(new { command = "echo hi" }));
        // No input queued: a second prompt would read null and decline.
        var second = await _tools.ExecuteAsync("run_command", Args(new { command = "echo hi" }));

        Assert.Contains("exit code: 0", first);
        Assert.Contains("exit code: 0", second);
    }

    [Fact]
    public async Task RunCommand_RunsInWorkingFolder()
    {
        UserTypes("y\n");

        var result = await _tools.ExecuteAsync("run_command", Args(new { command = "pwd" }));

        Assert.Contains(Path.GetFileName(_root), result);
    }

    // ---------------- ask_user ----------------

    [Fact]
    public async Task AskUser_ReturnsTypedAnswer()
    {
        UserTypes("use xunit\n");

        var result = await _tools.ExecuteAsync("ask_user",
            Args(new { question = "Which test framework?" }));

        Assert.Equal("User answered: use xunit", result);
    }

    [Fact]
    public async Task AskUser_EmptyAnswer_TellsModelToDecide()
    {
        UserTypes("\n");

        var result = await _tools.ExecuteAsync("ask_user", Args(new { question = "Pick one" }));

        Assert.Contains("best judgment", result);
    }

    // ---------------- sandbox escapes ----------------
    // The path/pattern arguments come straight from model JSON; none of the
    // shapes below may ever touch anything outside the working folder.

    [Fact]
    public async Task WriteFile_PathTraversal_IsDenied()
    {
        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = "../escaped.txt", content = "pwned" }));

        Assert.Contains("outside the working folder", result);
        Assert.False(File.Exists(Path.Combine(Path.GetDirectoryName(_root)!, "escaped.txt")));
    }

    [Fact]
    public async Task WriteFile_AbsolutePath_IsDenied()
    {
        var outside = Path.Combine(Path.GetTempPath(), $"escaped-{Guid.NewGuid():N}.txt");

        var result = await _tools.ExecuteAsync("write_file",
            Args(new { path = outside, content = "pwned" }));

        Assert.Contains("outside the working folder", result);
        Assert.False(File.Exists(outside));
    }

    [Fact]
    public async Task EditFile_PathTraversal_IsDenied()
    {
        var result = await _tools.ExecuteAsync("edit_file",
            Args(new { path = "../../etc/hosts", find = "localhost", replace = "evil" }));

        Assert.Contains("outside the working folder", result);
    }

    [Fact]
    public async Task WriteFile_ThroughSymlink_IsDenied()
    {
        // A symlink inside the folder pointing outside would pass a purely
        // textual path check; it must be refused.
        var outside = Directory.CreateTempSubdirectory("symlink-target-").FullName;
        try
        {
            Directory.CreateSymbolicLink(Path.Combine(_root, "link"), outside);

            var result = await _tools.ExecuteAsync("write_file",
                Args(new { path = "link/escaped.txt", content = "pwned" }));

            Assert.Contains("symbolic link", result);
            Assert.False(File.Exists(Path.Combine(outside, "escaped.txt")));
        }
        finally
        {
            Directory.Delete(outside, recursive: true);
        }
    }

    [Fact]
    public async Task ReadFile_ThroughSymlink_IsDenied()
    {
        var outside = Directory.CreateTempSubdirectory("symlink-target-").FullName;
        try
        {
            File.WriteAllText(Path.Combine(outside, "secret.txt"), "secret data");
            Directory.CreateSymbolicLink(Path.Combine(_root, "link"), outside);

            var result = await _tools.ExecuteAsync("read_file",
                Args(new { path = "link/secret.txt" }));

            Assert.Contains("symbolic link", result);
            Assert.DoesNotContain("secret data", result);
        }
        finally
        {
            Directory.Delete(outside, recursive: true);
        }
    }

    [Fact]
    public async Task ListFiles_PatternWithTraversal_IsRejected()
    {
        // Marker file just outside the sandbox root.
        var marker = Path.Combine(Path.GetDirectoryName(_root)!, $"marker-{Guid.NewGuid():N}.txt");
        File.WriteAllText(marker, "outside");
        try
        {
            var result = await _tools.ExecuteAsync("list_files", Args(new { pattern = "../*" }));

            Assert.StartsWith("Error:", result);
            Assert.DoesNotContain(Path.GetFileName(marker), result);
        }
        finally
        {
            File.Delete(marker);
        }
    }

    [Fact]
    public async Task SearchFiles_PatternWithSeparators_IsRejected()
    {
        var result = await _tools.ExecuteAsync("search_files",
            Args(new { pattern = "secret", file_pattern = "../**/*" }));

        Assert.StartsWith("Error:", result);
    }

    // ---------------- dispatch ----------------

    [Fact]
    public async Task UnknownTool_ListsValidTools()
    {
        var result = await _tools.ExecuteAsync("delete_everything", Args(new { }));

        Assert.StartsWith("Unknown tool", result);
        Assert.Contains("search_files", result);
    }

    [Fact]
    public void PlanMode_OnlyAllowsReadOnlyTools()
    {
        Assert.True(AgentTools.IsAllowedInPlanMode("read_file"));
        Assert.True(AgentTools.IsAllowedInPlanMode("search_files"));
        Assert.False(AgentTools.IsAllowedInPlanMode("write_file"));
        Assert.False(AgentTools.IsAllowedInPlanMode("run_command"));
    }
}
