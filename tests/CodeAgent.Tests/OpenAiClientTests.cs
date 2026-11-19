using System.Net;
using System.Text;
using System.Text.Json;
using CodeAgent.Llm;
using CodeAgent.OpenAi;

namespace CodeAgent.Tests;

/// <summary>
/// Offline tests for the SSE parsing and wire mapping in OpenAiClient, using
/// a stubbed HttpClient — no mlx_lm server needed.
/// </summary>
public class OpenAiClientTests
{
    // Serves a canned SSE body for any request and remembers the request body.
    private sealed class StubHandler(string body) : HttpMessageHandler
    {
        public string? RequestBody { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            RequestBody = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(body, Encoding.UTF8, "text/event-stream")
            };
        }
    }

    private static (OpenAiClient Client, StubHandler Handler) ClientServing(params string[] events) =>
        ClientServing(embla: false, events);

    // embla: configured like the Embla backend (string arguments, reasoning echoed).
    private static (OpenAiClient Client, StubHandler Handler) ClientServing(bool embla, params string[] events)
    {
        var handler = new StubHandler(string.Concat(events.Select(e => $"data: {e}\n\n")) + "data: [DONE]\n\n");
        var client = new OpenAiClient(embla ? "embla" : "mlx", new Uri("http://stub.local"),
            new HttpClient(handler) { BaseAddress = new Uri("http://stub.local") },
            argumentsAsObject: !embla, echoReasoning: embla);
        return (client, handler);
    }

    private static string Delta(string deltaJson) => $$"""{"choices":[{"index":0,"delta":{{deltaJson}}}]}""";

    private static readonly ChatMessage[] AnyHistory = [new() { Role = "user", Content = "hi" }];

    [Fact]
    public async Task ChatAsync_AssemblesContentFragmentedToolCallsAndUsage()
    {
        var (client, _) = ClientServing(
            Delta("""{"role":"assistant","content":"hel"}"""),
            Delta("""{"content":"lo"}"""),
            Delta("""{"tool_calls":[{"index":0,"id":"abc","type":"function","function":{"name":"read_file","arguments":"{\"pa"}}]}"""),
            Delta("""{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a.cs\"}"}}]}"""),
            """{"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":10}}""");

        var streamed = new StringBuilder();
        var response = await client.ChatAsync("m", AnyHistory, tools: [], contextTokens: 8192,
            onToken: t => streamed.Append(t));

        Assert.Equal("hello", response.Content);
        Assert.Equal("hello", streamed.ToString());
        var call = Assert.Single(response.ToolCalls);
        Assert.Equal("abc", call.Id);
        Assert.Equal("read_file", call.Function.Name);
        Assert.Equal("a.cs", call.Function.Arguments.GetProperty("path").GetString());
        Assert.Equal(42, response.Stats!.PromptTokens);
        Assert.Equal(10, response.Stats.ReplyTokens);
    }

    [Fact]
    public async Task ChatAsync_ToolCallWithoutId_GetsOne()
    {
        var (client, _) = ClientServing(
            Delta("""{"tool_calls":[{"index":0,"function":{"name":"list_files","arguments":{"pattern":"*"}}}]}"""));

        var response = await client.ChatAsync("m", AnyHistory, [], 8192, onToken: _ => { });

        var call = Assert.Single(response.ToolCalls);
        Assert.False(string.IsNullOrEmpty(call.Id));
        Assert.Equal("*", call.Function.Arguments.GetProperty("pattern").GetString());
    }

    [Fact]
    public async Task ChatAsync_InvalidArguments_ThrowsModelStreamException()
    {
        var (client, _) = ClientServing(
            Delta("""{"tool_calls":[{"index":0,"id":"x","function":{"name":"read_file","arguments":"{\"path\":"}}]}"""));

        await Assert.ThrowsAsync<ModelStreamException>(() =>
            client.ChatAsync("m", AnyHistory, [], 8192, onToken: _ => { }));
    }

    [Fact]
    public async Task ChatAsync_ErrorEvent_ThrowsModelStreamException()
    {
        var (client, _) = ClientServing("""{"error":{"message":"boom"}}""");

        var ex = await Assert.ThrowsAsync<ModelStreamException>(() =>
            client.ChatAsync("m", AnyHistory, [], 8192, onToken: _ => { }));
        Assert.Equal("boom", ex.Message);
    }

    [Fact]
    public async Task ChatAsync_ReasoningField_SignalsThinkingWithoutPollutingContent()
    {
        var (client, _) = ClientServing(
            Delta("""{"reasoning":"let me think"}"""),
            Delta("""{"content":"answer"}"""));

        var thinking = 0;
        var response = await client.ChatAsync("m", AnyHistory, [], 8192, onToken: _ => { }, onThinking: () => thinking++);

        Assert.Equal("answer", response.Content);
        Assert.True(thinking > 0);
    }

    [Fact]
    public async Task ChatAsync_InlineThinkTagsSplitAcrossChunks_AreRemoved()
    {
        var (client, _) = ClientServing(
            Delta("""{"content":"<thi"}"""),
            Delta("""{"content":"nk>secret</th"}"""),
            Delta("""{"content":"ink>\nvisible"}"""));

        var streamed = new StringBuilder();
        var response = await client.ChatAsync("m", AnyHistory, [], 8192, onToken: t => streamed.Append(t));

        Assert.Equal("visible", response.Content);
        Assert.Equal("visible", streamed.ToString());
    }

    [Fact]
    public async Task ChatAsync_BareClosingThinkTag_DropsEarlierReasoningFromContent()
    {
        // Templates that end the prompt with "<think>\n" make the model's
        // output start inside the reasoning, with only a closing tag.
        var (client, _) = ClientServing(
            Delta("""{"content":"reasoning here"}"""),
            Delta("""{"content":"</think>\n\nthe answer"}"""));

        var response = await client.ChatAsync("m", AnyHistory, [], 8192, onToken: _ => { });

        Assert.Equal("the answer", response.Content);
    }

    [Fact]
    public async Task ChatAsync_SendsToolCallsWithIdsAndObjectArguments()
    {
        var (client, handler) = ClientServing(Delta("""{"content":"ok"}"""));
        ChatMessage[] history =
        [
            new() { Role = "user", Content = "read it" },
            new()
            {
                Role = "assistant",
                Content = "",
                ToolCalls = [new ToolCall
                {
                    Id = "c1",
                    Function = new ToolFunction { Name = "read_file", Arguments = JsonDocument.Parse("""{"path":"a.cs"}""").RootElement }
                }]
            },
            new() { Role = "tool", ToolName = "read_file", ToolCallId = "c1", Content = "contents" },
        ];

        await client.ChatAsync("m", history, [], 8192, onToken: _ => { });

        using var sent = JsonDocument.Parse(handler.RequestBody!);
        var messages = sent.RootElement.GetProperty("messages");
        var call = messages[1].GetProperty("tool_calls")[0];
        Assert.Equal("c1", call.GetProperty("id").GetString());
        Assert.Equal("a.cs", call.GetProperty("function").GetProperty("arguments").GetProperty("path").GetString());
        Assert.Equal("c1", messages[2].GetProperty("tool_call_id").GetString());
        Assert.False(sent.RootElement.TryGetProperty("tools", out _)); // empty tool list is omitted
    }

    [Fact]
    public async Task ChatAsync_EchoReasoning_CollectsReasoningContent()
    {
        // Embla's stream: reasoning first, then the answer, then the call whole.
        var (client, _) = ClientServing(embla: true,
            Delta("""{"role":"assistant","content":""}"""),
            Delta("""{"reasoning_content":"look at "}"""),
            Delta("""{"reasoning_content":"the file"}"""),
            Delta("""{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.cs\"}"},"index":0}]}"""));

        var response = await client.ChatAsync("Embla1-4bit", AnyHistory, [], 32768, onToken: _ => { });

        Assert.Equal("look at the file", response.Reasoning);
        Assert.Equal("", response.Content);
        Assert.Equal("a.cs", Assert.Single(response.ToolCalls).Function.Arguments.GetProperty("path").GetString());
    }

    [Fact]
    public async Task ChatAsync_WithoutEchoReasoning_DropsReasoning()
    {
        var (client, _) = ClientServing(Delta("""{"reasoning_content":"hmm"}"""), Delta("""{"content":"ok"}"""));

        var response = await client.ChatAsync("m", AnyHistory, [], 8192, onToken: _ => { });

        Assert.Null(response.Reasoning);
    }

    [Fact]
    public async Task ChatAsync_EchoReasoning_SendsReasoningContentAndStringArguments()
    {
        var (client, handler) = ClientServing(embla: true, Delta("""{"content":"ok"}"""));
        ChatMessage[] history =
        [
            new() { Role = "user", Content = "read it" },
            new()
            {
                Role = "assistant",
                Content = "",
                Reasoning = "look at the file",
                ToolCalls = [new ToolCall
                {
                    Id = "c1",
                    Function = new ToolFunction { Name = "read_file", Arguments = JsonDocument.Parse("""{"path":"a.cs"}""").RootElement }
                }]
            },
            new() { Role = "tool", ToolName = "read_file", ToolCallId = "c1", Content = "contents" },
        ];

        await client.ChatAsync("Embla1-4bit", history, [], 32768, onToken: _ => { });

        using var sent = JsonDocument.Parse(handler.RequestBody!);
        var assistant = sent.RootElement.GetProperty("messages")[1];
        Assert.Equal("look at the file", assistant.GetProperty("reasoning_content").GetString());
        var arguments = assistant.GetProperty("tool_calls")[0].GetProperty("function").GetProperty("arguments");
        Assert.Equal(JsonValueKind.String, arguments.ValueKind);
        Assert.Equal("a.cs", JsonDocument.Parse(arguments.GetString()!).RootElement.GetProperty("path").GetString());
    }
}
