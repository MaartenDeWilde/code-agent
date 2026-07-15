using System.Net;
using System.Text;
using CodeAgent.Ollama;

namespace CodeAgent.Tests;

/// <summary>
/// Offline tests for the stream parsing in OllamaClient, using a stubbed
/// HttpClient — no Ollama needed.
/// </summary>
public class OllamaClientTests
{
    // Serves a canned newline-delimited JSON body for any request.
    private sealed class StubHandler(string body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(body, Encoding.UTF8, "application/x-ndjson")
            });
    }

    private static OllamaClient ClientServing(params string[] chunks) =>
        new(new HttpClient(new StubHandler(string.Join("\n", chunks)))
        {
            BaseAddress = new Uri("http://stub.local")
        });

    private static readonly ChatMessage[] AnyHistory = [new() { Role = "user", Content = "hi" }];

    [Fact]
    public async Task ChatAsync_AssemblesContentToolCallsAndStats()
    {
        var client = ClientServing(
            """{"message":{"role":"assistant","content":"hel"},"done":false}""",
            """{"message":{"role":"assistant","content":"lo"},"done":false}""",
            """{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a.cs"}}}]},"done":false}""",
            """{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":42,"eval_count":10,"eval_duration":2000000000}""");

        var streamed = new StringBuilder();
        var response = await client.ChatAsync("m", AnyHistory, tools: [], contextTokens: 8192,
            onToken: t => streamed.Append(t));

        Assert.Equal("hello", response.Content);
        Assert.Equal("hello", streamed.ToString());
        var call = Assert.Single(response.ToolCalls);
        Assert.Equal("read_file", call.Function.Name);
        Assert.Equal("a.cs", call.Function.Arguments.GetProperty("path").GetString());
        Assert.NotNull(response.Stats);
        Assert.Equal(42, response.Stats!.PromptTokens);
        Assert.Equal(10, response.Stats.ReplyTokens);
        Assert.Equal(5.0, response.Stats.TokensPerSecond); // 10 tokens / 2s
    }

    [Fact]
    public async Task ChatAsync_ThinkingChunks_SignalActivityWithoutPollutingContent()
    {
        // Thinking models (gpt-oss, ...) stream reasoning in a separate
        // field; it must surface as activity, never as reply text.
        var client = ClientServing(
            """{"message":{"role":"assistant","thinking":"let me check"},"done":false}""",
            """{"message":{"role":"assistant","thinking":" the file"},"done":false}""",
            """{"message":{"role":"assistant","content":"answer"},"done":false}""",
            """{"message":{"role":"assistant","content":""},"done":true}""");

        var thinkingSignals = 0;
        var response = await client.ChatAsync("m", AnyHistory, tools: [], contextTokens: 8192,
            onToken: _ => { },
            onThinking: () => thinkingSignals++);

        Assert.Equal(2, thinkingSignals);
        Assert.Equal("answer", response.Content);
        Assert.DoesNotContain("let me check", response.Content);
    }

    [Fact]
    public async Task ChatAsync_ErrorChunk_ThrowsOllamaStreamException()
    {
        // The failure seen live: Ollama's tool-call parser dying mid-stream.
        var client = ClientServing(
            """{"message":{"role":"assistant","content":"partial"},"done":false}""",
            """{"error":"XML syntax error on line 2: element <function> closed by </parameter>"}""");

        var ex = await Assert.ThrowsAsync<OllamaStreamException>(() =>
            client.ChatAsync("m", AnyHistory, tools: [], contextTokens: 8192, onToken: _ => { }));

        Assert.Contains("XML syntax error", ex.Message);
    }

    [Fact]
    public async Task ChatAsync_NonSuccessStatus_SurfacesOllamaBody()
    {
        var response = new HttpResponseMessage(HttpStatusCode.NotFound)
        {
            Content = new StringContent("""{"error":"model 'x' not found"}""")
        };
        var client = new OllamaClient(new HttpClient(new FixedResponseHandler(response))
        {
            BaseAddress = new Uri("http://stub.local")
        });

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() =>
            client.ChatAsync("x", AnyHistory, tools: [], contextTokens: 8192, onToken: _ => { }));

        Assert.Contains("not found", ex.Message);
    }

    private sealed class FixedResponseHandler(HttpResponseMessage response) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(response);
    }
}
