using System.Text.Json;
using CodeAgent.Agent;
using CodeAgent.Ollama;

namespace CodeAgent.Tests;

public class ConversationHistoryTests
{
    private static ToolCall Call(string name, object arguments) => new()
    {
        Function = new ToolFunction
        {
            Name = name,
            Arguments = JsonSerializer.SerializeToElement(arguments)
        }
    };

    // Adds enough small chatter to age earlier messages out of the
    // "recent messages are never trimmed" protection window.
    private static void AddRecentChatter(ConversationHistory history, int count = 8)
    {
        for (var i = 0; i < count; i++)
        {
            history.AddUser($"msg {i}");
        }
    }

    [Fact]
    public void StaysIntactWhileUnderBudget()
    {
        var history = new ConversationHistory("system prompt", maxChars: 100_000);
        history.AddUser("task");
        history.AddToolResult("read_file", new string('x', 1000));

        Assert.Contains(history.Messages, m => m.Role == "tool" && m.Content.Length == 1000);
    }

    [Fact]
    public void RemovesOldestToolResultsWhenOverBudget()
    {
        var history = new ConversationHistory("system prompt", maxChars: 500);
        history.AddUser("task");
        history.AddToolResult("read_file", new string('x', 2000));
        AddRecentChatter(history);

        var toolMessage = history.Messages.Single(m => m.Role == "tool");
        Assert.Contains("removed", toolMessage.Content);
        Assert.True(history.TotalChars < 2000);
    }

    [Fact]
    public void RemovedMarkerNamesToolAndSubject()
    {
        var history = new ConversationHistory("system prompt", maxChars: 500);
        history.AddUser("task");
        history.AddToolResult("read_file", new string('x', 2000), subject: "src/OrderService.cs");
        AddRecentChatter(history);

        // The model must be able to tell WHICH call to repeat when it needs
        // the content again, not just that something went missing.
        var toolMessage = history.Messages.Single(m => m.Role == "tool");
        Assert.Contains("read_file", toolMessage.Content);
        Assert.Contains("src/OrderService.cs", toolMessage.Content);
    }

    [Fact]
    public void RemovingTwiceIsANoOp()
    {
        var history = new ConversationHistory("system prompt", maxChars: 500);
        history.AddUser("task");
        history.AddToolResult("read_file", new string('x', 2000), subject: "a.cs");
        AddRecentChatter(history);
        var afterFirstTrim = history.Messages.Single(m => m.Role == "tool").Content;

        // Blow the budget again so trimming re-visits the old messages: the
        // existing marker must survive as-is instead of being wrapped or
        // replaced again.
        history.AddToolResult("read_file", new string('y', 2000), subject: "b.cs");
        AddRecentChatter(history);

        Assert.Equal(afterFirstTrim, history.Messages.First(m => m.Role == "tool").Content);
    }

    [Fact]
    public void NeverTouchesSystemPromptOrRecentMessages()
    {
        var history = new ConversationHistory("system prompt", maxChars: 200);
        history.AddToolResult("read_file", new string('x', 5000)); // way over budget
        AddRecentChatter(history, count: 3); // tool result stays within the recent window

        Assert.Equal("system prompt", history.Messages[0].Content);
        // Recent messages are protected even though the budget is blown.
        Assert.Equal(5000, history.Messages[1].Content.Length);
    }

    [Fact]
    public void ShrinkingTheBudgetTrimsImmediately()
    {
        var history = new ConversationHistory("system prompt", maxChars: 100_000);
        history.AddUser("task");
        history.AddToolResult("read_file", new string('x', 10_000), subject: "a.cs");
        AddRecentChatter(history);

        // Recalibration from real token counts can lower the budget mid-task;
        // the transcript must shrink right away, not on the next Add. (The
        // requested 500 is raised to the 4,000-char floor, still well below
        // the 10,000-char tool result.)
        history.UpdateBudget(500);

        var toolMessage = history.Messages.Single(m => m.Role == "tool");
        Assert.Contains("removed", toolMessage.Content);
    }

    [Fact]
    public void RemoveUserMessages_DeletesOnlyMatchingUserMessages()
    {
        var history = new ConversationHistory("system prompt", maxChars: 100_000);
        history.AddUser("[note] outdated reminder");
        history.AddUser("real task");
        history.AddToolResult("read_file", "[note] looks like a note but is a tool result");

        history.RemoveUserMessages(content => content.StartsWith("[note]"));

        Assert.DoesNotContain(history.Messages, m => m.Role == "user" && m.Content.StartsWith("[note]"));
        Assert.Contains(history.Messages, m => m.Role == "user" && m.Content == "real task");
        Assert.Contains(history.Messages, m => m.Role == "tool"); // non-user messages untouched
    }

    [Fact]
    public void CountsToolCallArgumentsInBudget()
    {
        var history = new ConversationHistory("system", maxChars: 100_000);
        history.AddAssistant("", new List<ToolCall>
        {
            Call("write_file", new { path = "a.txt", content = new string('x', 10_000) })
        });

        // The visible content is empty, but the arguments must count.
        Assert.True(history.TotalChars > 10_000);
    }

    [Fact]
    public void RemovesOldToolCallArgumentsWhenOverBudget()
    {
        var history = new ConversationHistory("system", maxChars: 500);
        history.AddAssistant("", new List<ToolCall>
        {
            Call("write_file", new { path = "a.txt", content = new string('x', 5000) })
        });
        history.AddToolResult("write_file", "Wrote 5000 characters");
        AddRecentChatter(history);

        var assistant = history.Messages.Single(m => m.Role == "assistant");
        Assert.Null(assistant.ToolCalls);
        Assert.Contains("write_file", assistant.Content); // note says what was removed
        Assert.True(history.TotalChars < 5000);
    }
}
