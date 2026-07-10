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
