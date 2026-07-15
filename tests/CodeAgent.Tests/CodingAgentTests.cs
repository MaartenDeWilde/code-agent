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

        var reminder = CodingAgent.BuildProgressReminder("add logging to all services", activity);

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

        var reminder = CodingAgent.BuildProgressReminder("task", activity);

        Assert.Contains("Files read so far: a.cs", reminder);
        Assert.DoesNotContain("Files edited so far", reminder);
        Assert.DoesNotContain("Commands run so far", reminder);
    }
}
