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
    public void ElidesOldestToolResultsWhenOverBudget()
    {
        var history = new ConversationHistory("system prompt", maxChars: 500);
        history.AddUser("task");
        history.AddToolResult("read_file", new string('x', 2000));
        AddRecentChatter(history);

        var toolMessage = history.Messages.Single(m => m.Role == "tool");
        Assert.Contains("elided", toolMessage.Content);
        Assert.True(history.TotalChars < 2000);
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
    public void ElidesOldToolCallArgumentsWhenOverBudget()
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
        Assert.Contains("write_file", assistant.Content); // note says what was elided
        Assert.True(history.TotalChars < 5000);
    }
}
