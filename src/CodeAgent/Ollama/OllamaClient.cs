using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace CodeAgent.Ollama;

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

    // Streamed by thinking models (gpt-oss, ...): reasoning tokens that are
    // not part of the reply. Never echoed back; only used to signal that the
    // model is still working while nothing visible is being produced.
    [JsonPropertyName("thinking")]
    public string? Thinking { get; set; }

    // Local bookkeeping, never sent to the model: the main argument of the
    // call that produced this result (file path, search pattern, command).
    // Lets ConversationHistory say WHAT was removed, so the model knows which
    // tool call to repeat instead of working from memory.
    [JsonIgnore]
    public string? ToolSubject { get; set; }
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

/// <summary>
/// An error Ollama reports inside an otherwise successful response stream —
/// e.g. its parser failed on the model's tool-call syntax. These are worth
/// retrying: a resample usually produces parseable output.
/// </summary>
class OllamaStreamException(string message) : Exception(message);

class OllamaClient
{
    private readonly HttpClient _http;
    private readonly ILogger _logger;

    public OllamaClient(HttpClient? http = null, ILogger? logger = null)
    {
        _http = http ?? new HttpClient
        {
            BaseAddress = new Uri("http://localhost:11434"),
            // Local models can take a while to respond; the HttpClient default of 100s is too short.
            Timeout = TimeSpan.FromMinutes(10)
        };
        _logger = logger ?? NullLogger.Instance;
    }

    private static readonly JsonSerializerOptions SerializerOpts = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>Names of the models installed on the Ollama server (GET /api/tags).</summary>
    public async Task<List<string>> ListModelsAsync()
    {
        using var doc = JsonDocument.Parse(await _http.GetStringAsync("/api/tags"));
        return doc.RootElement.GetProperty("models").EnumerateArray()
            .Select(m => m.GetProperty("name").GetString() ?? "")
            .Where(name => name.Length > 0)
            .OrderBy(name => name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <summary>
    /// Sends the conversation to the model and streams the reply.
    /// Text tokens are passed to <paramref name="onToken"/> as they arrive;
    /// <paramref name="onThinking"/> fires for reasoning tokens of thinking
    /// models, whose text is deliberately not exposed — only the fact that
    /// the model is still generating. Tool calls are collected and returned
    /// once the stream completes.
    /// </summary>
    public async Task<ChatResponse> ChatAsync(
        string model,
        IReadOnlyList<ChatMessage> history,
        object[] tools,
        int contextTokens,
        Action<string> onToken,
        Action? onThinking = null,
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
            // The sampling defaults are tuned for chat, not agent work:
            // temperature ~0.8 is where tool-call drift and malformed JSON
            // come from, and repeat_penalty 1.1 corrupts code, which repeats
            // indentation and brackets by nature. Disabling the penalty makes
            // prose loops likelier, so LooksLikeRepetitionLoop guards the
            // stream, and num_predict caps a runaway reply outright (8192
            // tokens still fits a large write_file call).
            options = new { num_ctx = contextTokens, temperature = 0.2, repeat_penalty = 1.0, num_predict = 8192 }
        };

        // One log event per request/response pair, with the numbers that
        // matter for cost and latency. Message CONTENT deliberately stays out
        // at Information level — transcripts routinely contain the user's
        // code and secrets, and logs travel further than transcripts do.
        _logger.LogDebug("LLM request: model {Model}, {MessageCount} messages, {ToolCount} tools, num_ctx {ContextTokens}",
            model, history.Count, tools.Length, contextTokens);
        var startedAt = System.Diagnostics.Stopwatch.GetTimestamp();

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

        // Repetition looping is checked as the reply grows, not at the end:
        // a looping model may never finish on its own.
        var nextLoopCheck = 2_000;

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
                _logger.LogWarning("LLM stream error from model {Model}: {Error}", model, chunkError);
                throw new OllamaStreamException(chunkError);
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

            if (!string.IsNullOrEmpty(message.Thinking))
            {
                onThinking?.Invoke();
            }

            if (!string.IsNullOrEmpty(message.Content))
            {
                contentBuilder.Append(message.Content);
                onToken(message.Content);

                if (contentBuilder.Length >= nextLoopCheck)
                {
                    nextLoopCheck = contentBuilder.Length + 1_000;
                    if (LooksLikeRepetitionLoop(contentBuilder.ToString()))
                    {
                        // Same failure contract as Ollama's own mid-stream
                        // errors: the caller discards the reply and retries.
                        throw new OllamaStreamException("the model is repeating the same lines over and over");
                    }
                }
            }

            if (message.ToolCalls is { Count: > 0 })
            {
                toolCalls.AddRange(message.ToolCalls);
            }
        }

        // Property names follow the OpenTelemetry GenAI semantic conventions
        // (gen_ai.usage.input_tokens, ...) with the dots that message
        // templates don't allow flattened to underscores, so the fields stay
        // recognizable if this ever feeds a real telemetry pipeline.
        _logger.LogInformation(
            "LLM response: model {gen_ai_request_model}, {gen_ai_usage_input_tokens} input + {gen_ai_usage_output_tokens} output tokens, "
            + "{ToolCallCount} tool calls, {ContentChars} content chars, {TokensPerSecond:F1} tok/s, {DurationMs} ms",
            model, stats?.PromptTokens, stats?.ReplyTokens,
            toolCalls.Count, contentBuilder.Length, stats?.TokensPerSecond,
            (long)System.Diagnostics.Stopwatch.GetElapsedTime(startedAt).TotalMilliseconds);

        return new ChatResponse(contentBuilder.ToString(), toolCalls, stats);
    }

    // Degeneration check: with repeat_penalty disabled, a small model can
    // fall into re-emitting the same few sentences forever ("Wait, I'll fix
    // X first. / Actually, I'll do it all at once." ad infinitum). A real,
    // wordy line recurring this often in one reply is a loop, not style.
    // Lines must contain letters (markdown table rules don't) and some
    // length (lone braces and ``` fences repeat legitimately in code talk).
    internal static bool LooksLikeRepetitionLoop(string content)
    {
        const int minLineLength = 10;
        const int loopThreshold = 5;

        var counts = new Dictionary<string, int>();
        foreach (var rawLine in content.Split('\n'))
        {
            var line = rawLine.Trim();
            if (line.Length < minLineLength || !line.Any(char.IsLetter) || line.StartsWith("```"))
            {
                continue;
            }
            if ((counts[line] = counts.GetValueOrDefault(line) + 1) >= loopThreshold)
            {
                return true;
            }
        }

        return false;
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
