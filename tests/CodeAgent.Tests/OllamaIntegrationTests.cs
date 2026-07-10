using System.Text;
using CodeAgent.Ollama;
using CodeAgent.Tools;

namespace CodeAgent.Tests;

/// <summary>
/// Integration tests that talk to a live Ollama instance on localhost:11434.
/// They assume Ollama is running and the model below is installed and
/// supports native tool calling. Override the model with the
/// CODEAGENT_TEST_MODEL environment variable.
/// </summary>
public class OllamaIntegrationTests
{
    private static readonly string Model =
        Environment.GetEnvironmentVariable("CODEAGENT_TEST_MODEL") ?? "gemma4:12b-mlx";

    private static ChatMessage User(string content) => new() { Role = "user", Content = content };

    [Fact]
    public async Task ChatAsync_StreamsTheSameContentItReturns()
    {
        var client = new OllamaClient();
        var streamed = new StringBuilder();

        var response = await client.ChatAsync(Model,
            new[] { User("Reply with exactly one word: pong") },
            tools: [],
            contextTokens: 8192,
            onToken: text => streamed.Append(text));

        Assert.False(string.IsNullOrWhiteSpace(response.Content));
        Assert.Equal(response.Content, streamed.ToString());
    }

    [Fact]
    public async Task ChatAsync_ReportsTokenStatistics()
    {
        var client = new OllamaClient();

        var response = await client.ChatAsync(Model,
            new[] { User("Reply with exactly one word: pong") },
            tools: [],
            contextTokens: 8192,
            onToken: _ => { });

        Assert.NotNull(response.Stats);
        Assert.True(response.Stats!.PromptTokens > 0, "prompt token count missing from final chunk");
        Assert.True(response.Stats.ReplyTokens > 0, "reply token count missing from final chunk");
    }

    [Fact]
    public async Task ChatAsync_ModelInvokesAnOfferedTool()
    {
        var client = new OllamaClient();

        // Leave the model no sensible alternative to calling the tool.
        var response = await client.ChatAsync(Model,
            new[] { User("Use the list_files tool with pattern '*' to list the project files. Do not answer in text; call the tool.") },
            tools: AgentTools.PlanDefinitions,
            contextTokens: 8192,
            onToken: _ => { });

        Assert.NotEmpty(response.ToolCalls);
        Assert.Contains(response.ToolCalls, c => c.Function.Name == "list_files");
    }

    [Fact]
    public async Task ChatAsync_UnknownModel_ThrowsWithOllamaMessage()
    {
        var client = new OllamaClient();

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() => client.ChatAsync(
            "definitely-not-installed:1b",
            new[] { User("hi") },
            tools: [],
            contextTokens: 8192,
            onToken: _ => { }));

        // The body of Ollama's error response must surface in the message.
        Assert.Contains("not found", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ChatAsync_CancellationAbortsTheStream()
    {
        var client = new OllamaClient();
        using var cts = new CancellationTokenSource();

        var task = client.ChatAsync(Model,
            new[] { User("Count slowly from 1 to 500, one number per line.") },
            tools: [],
            contextTokens: 8192,
            // Cancel as soon as generation has demonstrably started.
            onToken: _ => cts.Cancel(),
            cts.Token);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
    }
}
