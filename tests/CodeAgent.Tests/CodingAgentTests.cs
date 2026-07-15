using System.Text.Json;
using CodeAgent.Agent;

namespace CodeAgent.Tests;

public class CodingAgentTests
{
    private static JsonElement Args(object values) => JsonSerializer.SerializeToElement(values);

    [Fact]
    public void ProgressReminder_RestatesTaskAndProgress()
    {
        var activity = new CodingAgent.TaskActivity();
        activity.Record("read_file", Args(new { path = "src/App.cs" }));
        activity.Record("edit_file", Args(new { path = "src/App.cs" }));
        activity.Record("run_command", Args(new { command = "dotnet build" }));

        var reminder = CodingAgent.BuildProgressReminder("add logging to all services", activity, planMode: false);

        // The reminder must anchor the model back to the original task and
        // list what it already did, so it can work out what remains.
        Assert.Contains("add logging to all services", reminder);
        Assert.Contains("src/App.cs", reminder);
        Assert.Contains("dotnet build", reminder);
        // And it must not read as new instructions from the user.
        Assert.Contains("not a message from the user", reminder);
    }

    [Fact]
    public void ProgressReminder_OmitsEmptyProgressLines()
    {
        var activity = new CodingAgent.TaskActivity();
        activity.Record("read_file", Args(new { path = "a.cs" }));

        var reminder = CodingAgent.BuildProgressReminder("task", activity, planMode: false);

        Assert.Contains("Files read so far: a.cs", reminder);
        Assert.DoesNotContain("Files edited so far", reminder);
        Assert.DoesNotContain("Commands run so far", reminder);
    }

    [Theory]
    [InlineData("I'll read the file now<|channel|>")] // Harmony control token (gpt-oss)
    [InlineData("I'll read the file now<channel|>")] // same, leading pipe eaten by Ollama's partial parse
    [InlineData("some text<|channel>")] // trailing pipe eaten
    [InlineData("<|im_start|>assistant<|im_end|>")] // ChatML leaked through
    [InlineData("<tool_call>{\"name\":\"read_file\"}</tool_call>")] // tool call faked as text
    public void FindLeakedTemplateMarker_FlagsWireFormatInReplyText(string content)
    {
        Assert.NotNull(CodingAgent.FindLeakedTemplateMarker(content));
    }

    [Theory]
    [InlineData("Done: I renamed the method in both files.")]
    [InlineData("Use List<int> here, and a < b || c > d for the check.")] // angle brackets alone are fine
    [InlineData("Wrap it in <start> and <end> tags.")] // pipe-less words in brackets are ordinary text
    public void FindLeakedTemplateMarker_AcceptsNormalReplies(string content)
    {
        Assert.Null(CodingAgent.FindLeakedTemplateMarker(content));
    }

    [Fact]
    public void ProgressReminder_InPlanMode_AsksForAPlanNotASummary()
    {
        var activity = new CodingAgent.TaskActivity();
        activity.Record("read_file", Args(new { path = "a.cs" }));

        var reminder = CodingAgent.BuildProgressReminder("task", activity, planMode: true);

        // Plan mode must end in a presented plan; the normal 'summarize what
        // you did' closing would invite the model to skip the plan.
        Assert.Contains("plan", reminder);
        Assert.DoesNotContain("summary", reminder);
    }
}
