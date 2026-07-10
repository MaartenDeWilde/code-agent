using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

// ======================================================================
// Chat message / tool call models shared between the client and the agent.
// Property names follow Ollama's wire format (snake_case).
// ======================================================================

class ChatMessage
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = "";

    [JsonPropertyName("content")]
    public string Content { get; set; } = "";

    // Present on assistant messages that requested tool calls; must be echoed
    // back to the model so it remembers what it asked for.
    [JsonPropertyName("tool_calls")]
    public List<ToolCall>? ToolCalls { get; set; }

    // Present on 'tool' role messages: which tool produced this result.
    [JsonPropertyName("tool_name")]
    public string? ToolName { get; set; }
}

class ToolCall
{
    [JsonPropertyName("function")]
    public ToolFunction Function { get; set; } = new();
}

class ToolFunction
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    // Ollama delivers arguments as a parsed JSON object, not a string.
    [JsonPropertyName("arguments")]
    public JsonElement Arguments { get; set; }
}

/// <summary>The model's complete reply: streamed text plus any tool calls.</summary>
record ChatResponse(string Content, List<ToolCall> ToolCalls, ChatStats? Stats);

/// <summary>
/// Real token counts reported by Ollama in the final stream chunk — unlike
/// our chars/4 estimate, PromptTokens is exactly what the context cost.
/// </summary>
record ChatStats(int PromptTokens, int ReplyTokens, double TokensPerSecond);

// ======================================================================
// Minimal Ollama HTTP client (POST /api/chat) with streaming and native
// tool calling. Requires a model with tool support (qwen, llama3.1+, ...).
// ======================================================================

class OllamaClient
{
    private readonly HttpClient _http = new()
    {
        BaseAddress = new Uri("http://localhost:11434"),
        // Local models can take a while to respond; the HttpClient default of 100s is too short.
        Timeout = TimeSpan.FromMinutes(10)
    };

    private static readonly JsonSerializerOptions SerializerOpts = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>
    /// Sends the conversation to the model and streams the reply.
    /// Text tokens are passed to <paramref name="onToken"/> as they arrive;
    /// tool calls are collected and returned once the stream completes.
    /// </summary>
    public async Task<ChatResponse> ChatAsync(
        string model,
        IReadOnlyList<ChatMessage> history,
        object[] tools,
        int contextTokens,
        Action<string> onToken,
        CancellationToken cancellationToken = default)
    {
        var request = new
        {
            model,
            stream = true,
            messages = history,
            tools,
            // Ollama applies its own (often small) default context length per
            // request; ask explicitly for the window we plan around.
            options = new { num_ctx = contextTokens }
        };

        var body = JsonSerializer.Serialize(request, SerializerOpts);
        using var content = new StringContent(body, Encoding.UTF8, "application/json");
        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, "/api/chat") { Content = content };

        // ResponseHeadersRead lets us start reading the stream before the
        // model has finished generating.
        using var response = await _http.SendAsync(httpRequest, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            // Ollama puts a useful message in the body, e.g. "model X does not support tools".
            var error = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new HttpRequestException($"Ollama returned {(int)response.StatusCode}: {error.Trim()}");
        }

        // The stream is one JSON object per line.
        var contentBuilder = new StringBuilder();
        var toolCalls = new List<ToolCall>();
        ChatStats? stats = null;

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);
        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            var chunk = JsonSerializer.Deserialize<ChatChunk>(line);
            if (chunk?.Error is { } chunkError)
            {
                throw new HttpRequestException($"Ollama error: {chunkError}");
            }

            // The final chunk carries the generation statistics.
            if (chunk is { Done: true })
            {
                var seconds = chunk.EvalDuration / 1_000_000_000.0;
                stats = new ChatStats(
                    chunk.PromptEvalCount,
                    chunk.EvalCount,
                    seconds > 0 ? chunk.EvalCount / seconds : 0);
            }

            var message = chunk?.Message;
            if (message is null)
            {
                continue;
            }

            if (!string.IsNullOrEmpty(message.Content))
            {
                contentBuilder.Append(message.Content);
                onToken(message.Content);
            }

            if (message.ToolCalls is { Count: > 0 })
            {
                toolCalls.AddRange(message.ToolCalls);
            }
        }

        return new ChatResponse(contentBuilder.ToString(), toolCalls, stats);
    }

    private class ChatChunk
    {
        [JsonPropertyName("message")]
        public ChatMessage? Message { get; set; }

        [JsonPropertyName("done")]
        public bool Done { get; set; }

        [JsonPropertyName("error")]
        public string? Error { get; set; }

        // Statistics, only present on the final (done: true) chunk.
        [JsonPropertyName("prompt_eval_count")]
        public int PromptEvalCount { get; set; }

        [JsonPropertyName("eval_count")]
        public int EvalCount { get; set; }

        // Nanoseconds spent generating the reply.
        [JsonPropertyName("eval_duration")]
        public long EvalDuration { get; set; }
    }
}
