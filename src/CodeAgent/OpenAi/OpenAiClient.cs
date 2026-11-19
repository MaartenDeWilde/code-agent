using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using CodeAgent.Llm;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace CodeAgent.OpenAi;

// ======================================================================
// Minimal client for OpenAI-compatible servers (POST /v1/chat/completions)
// with SSE streaming and native tool calling: mlx_lm.server, LM Studio,
// llama.cpp's llama-server, vLLM, ... The differences from Ollama that
// matter here:
//   - the stream is server-sent events ("data: {...}"), ending in [DONE];
//   - tool-call arguments arrive as a JSON *string*, split over chunks;
//   - results are paired to calls by id (tool_call_id), not by name;
//   - there is no per-request context size: the server's own limit applies;
//   - reasoning arrives in delta.reasoning / delta.reasoning_content, or
//     inline as <think>...</think> in the content.
// ======================================================================

class OpenAiClient : IChatClient
{
    private readonly HttpClient _http;
    private readonly ILogger _logger;
    private readonly bool _argumentsAsObject;
    private readonly bool _echoReasoning;

    public string Name { get; }

    /// <param name="argumentsAsObject">Echo tool-call arguments back as a JSON
    /// object instead of the string the OpenAI spec prescribes. mlx_lm hands
    /// messages straight to the model's chat template, and templates iterate
    /// the arguments as a mapping (<c>arguments|items</c>) — a string breaks
    /// them. Strict servers (vLLM, ...) want <c>false</c>.</param>
    /// <param name="echoReasoning">Collect the streamed reasoning and send it
    /// back as <c>reasoning_content</c> on assistant turns. Embla re-renders
    /// the reasoning of the turns after the last user message, so without it
    /// the prompt no longer matches what the model generated: its prefix cache
    /// misses, and the model loses its train of thought between tool calls.</param>
    public OpenAiClient(string name, Uri baseAddress, HttpClient? http = null, ILogger? logger = null,
        bool argumentsAsObject = true, bool echoReasoning = false)
    {
        Name = name;
        _http = http ?? new HttpClient
        {
            BaseAddress = baseAddress,
            // Local models can take a while to respond (and mlx_lm loads the
            // model on the first request); the default of 100s is too short.
            Timeout = TimeSpan.FromMinutes(10)
        };
        _logger = logger ?? NullLogger.Instance;
        _argumentsAsObject = argumentsAsObject;
        _echoReasoning = echoReasoning;
    }

    private static readonly JsonSerializerOptions SerializerOpts = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>Models the server offers (GET /v1/models). For mlx_lm these
    /// are the models in the local Hugging Face cache.</summary>
    public async Task<List<string>> ListModelsAsync(CancellationToken cancellationToken = default)
    {
        using var doc = JsonDocument.Parse(await _http.GetStringAsync("/v1/models", cancellationToken));
        return doc.RootElement.GetProperty("data").EnumerateArray()
            .Select(m => m.GetProperty("id").GetString() ?? "")
            .Where(id => id.Length > 0)
            .OrderBy(id => id, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    public async Task<ChatResponse> ChatAsync(
        string model,
        IReadOnlyList<ChatMessage> history,
        object[] tools,
        int contextTokens,
        Action<string> onToken,
        Action? onThinking = null,
        CancellationToken cancellationToken = default)
    {
        // Same sampling rationale as the Ollama client: low temperature for
        // reliable tool calls, no repetition penalty (it corrupts code; the
        // loop check below guards prose instead), and a cap on reply length.
        // contextTokens can't be sent — the server decides the window — but
        // the agent still trims the transcript to it.
        var request = new
        {
            model,
            stream = true,
            // Ask for token counts in a final chunk; servers that don't know
            // the option ignore it.
            stream_options = new { include_usage = true },
            messages = history.Select(ToWire).ToList(),
            tools = tools.Length > 0 ? tools : null,
            temperature = 0.2,
            max_tokens = 8192
        };

        _logger.LogDebug("LLM request ({Backend}): model {Model}, {MessageCount} messages, {ToolCount} tools",
            Name, model, history.Count, tools.Length);
        var startedAt = System.Diagnostics.Stopwatch.GetTimestamp();

        var body = JsonSerializer.Serialize(request, SerializerOpts);
        using var content = new StringContent(body, Encoding.UTF8, "application/json");
        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions") { Content = content };

        using var response = await _http.SendAsync(httpRequest, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync(cancellationToken);
            throw new HttpRequestException($"{Name} returned {(int)response.StatusCode}: {error.Trim()}");
        }

        var contentBuilder = new StringBuilder();
        var reasoningBuilder = new StringBuilder();
        var thinkFilter = new ThinkTagFilter();
        // Tool calls stream in pieces keyed by index: the first piece has the
        // id and name, later pieces append to the arguments string.
        var pendingCalls = new SortedDictionary<int, PendingCall>();
        int? promptTokens = null, replyTokens = null;
        long? firstTokenAt = null;
        var nextLoopCheck = 2_000;

        void AppendVisible(string text)
        {
            if (text.Length == 0)
            {
                return;
            }
            contentBuilder.Append(text);
            onToken(text);
            if (contentBuilder.Length >= nextLoopCheck)
            {
                nextLoopCheck = contentBuilder.Length + 1_000;
                if (ReplyChecks.LooksLikeRepetitionLoop(contentBuilder.ToString()))
                {
                    throw new ModelStreamException("the model is repeating the same lines over and over");
                }
            }
        }

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);
        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            // SSE: payload lines start with "data:"; blank lines separate
            // events, and ":" lines are keep-alive comments.
            if (!line.StartsWith("data:", StringComparison.Ordinal))
            {
                continue;
            }
            var payload = line[5..].Trim();
            if (payload == "[DONE]")
            {
                break;
            }
            if (payload.Length == 0)
            {
                continue;
            }

            using var doc = JsonDocument.Parse(payload);
            var root = doc.RootElement;

            if (root.TryGetProperty("error", out var error))
            {
                var message = error.ValueKind == JsonValueKind.Object && error.TryGetProperty("message", out var m)
                    ? m.GetString()
                    : error.ToString();
                _logger.LogWarning("LLM stream error from {Backend} model {Model}: {Error}", Name, model, message);
                throw new ModelStreamException(message ?? "unknown error");
            }

            if (root.TryGetProperty("usage", out var usage) && usage.ValueKind == JsonValueKind.Object)
            {
                promptTokens = TryGetInt(usage, "prompt_tokens") ?? promptTokens;
                replyTokens = TryGetInt(usage, "completion_tokens") ?? replyTokens;
            }

            if (!root.TryGetProperty("choices", out var choices) || choices.ValueKind != JsonValueKind.Array
                || choices.GetArrayLength() == 0
                || !choices[0].TryGetProperty("delta", out var delta) || delta.ValueKind != JsonValueKind.Object)
            {
                continue;
            }
            firstTokenAt ??= System.Diagnostics.Stopwatch.GetTimestamp();

            if ((GetString(delta, "reasoning_content") ?? GetString(delta, "reasoning")) is { Length: > 0 } reasoning)
            {
                if (_echoReasoning)
                {
                    reasoningBuilder.Append(reasoning);
                }
                onThinking?.Invoke();
            }

            if (GetString(delta, "content") is { Length: > 0 } text)
            {
                var visible = thinkFilter.Process(text, out var sawThinking, out var reasoningEnded);
                if (reasoningEnded)
                {
                    // A bare </think>: the template opened the thinking block
                    // in the prompt, so everything so far was reasoning. It
                    // was already printed, but it must not become the reply.
                    contentBuilder.Clear();
                }
                if (sawThinking)
                {
                    onThinking?.Invoke();
                }
                AppendVisible(visible);
            }

            if (delta.TryGetProperty("tool_calls", out var callDeltas) && callDeltas.ValueKind == JsonValueKind.Array)
            {
                foreach (var callDelta in callDeltas.EnumerateArray())
                {
                    var index = TryGetInt(callDelta, "index") ?? pendingCalls.Count;
                    if (!pendingCalls.TryGetValue(index, out var pending))
                    {
                        pendingCalls[index] = pending = new PendingCall();
                    }
                    pending.Id ??= GetString(callDelta, "id");
                    if (callDelta.TryGetProperty("function", out var function))
                    {
                        pending.Name += GetString(function, "name") ?? "";
                        if (function.TryGetProperty("arguments", out var args))
                        {
                            // Spec: string fragments. Some servers send the
                            // whole object at once instead.
                            pending.Arguments.Append(args.ValueKind == JsonValueKind.String
                                ? args.GetString()
                                : args.GetRawText());
                        }
                    }
                }
            }
        }

        AppendVisible(thinkFilter.Flush());

        var toolCalls = new List<ToolCall>();
        foreach (var (index, pending) in pendingCalls)
        {
            toolCalls.Add(new ToolCall
            {
                // Results must reference the call by id; invent one if the
                // server didn't send any.
                Id = pending.Id ?? $"call_{index}",
                Function = new ToolFunction { Name = pending.Name, Arguments = ParseArguments(pending) }
            });
        }

        ChatStats? stats = null;
        if (promptTokens is not null || replyTokens is not null)
        {
            var generationSeconds = firstTokenAt is { } t
                ? System.Diagnostics.Stopwatch.GetElapsedTime(t).TotalSeconds
                : 0;
            stats = new ChatStats(promptTokens ?? 0, replyTokens ?? 0,
                generationSeconds > 0 ? (replyTokens ?? 0) / generationSeconds : 0);
        }

        _logger.LogInformation(
            "LLM response ({Backend}): model {gen_ai_request_model}, {gen_ai_usage_input_tokens} input + {gen_ai_usage_output_tokens} output tokens, "
            + "{ToolCallCount} tool calls, {ContentChars} content chars, {TokensPerSecond:F1} tok/s, {DurationMs} ms",
            Name, model, stats?.PromptTokens, stats?.ReplyTokens,
            toolCalls.Count, contentBuilder.Length, stats?.TokensPerSecond,
            (long)System.Diagnostics.Stopwatch.GetElapsedTime(startedAt).TotalMilliseconds);

        return new ChatResponse(contentBuilder.ToString(), toolCalls, stats,
            reasoningBuilder.Length > 0 ? reasoningBuilder.ToString() : null);
    }

    private JsonElement ParseArguments(PendingCall pending)
    {
        var raw = pending.Arguments.ToString().Trim();
        if (raw.Length == 0)
        {
            return JsonDocument.Parse("{}").RootElement.Clone();
        }
        try
        {
            using var doc = JsonDocument.Parse(raw);
            return doc.RootElement.Clone();
        }
        catch (JsonException)
        {
            // Unusable call; the agent resamples, like any malformed reply.
            throw new ModelStreamException($"the model sent invalid JSON arguments for tool '{pending.Name}'");
        }
    }

    // Our transcript → OpenAI chat message.
    private object ToWire(ChatMessage message)
    {
        var wire = new JsonObject
        {
            ["role"] = message.Role,
            ["content"] = message.Content
        };
        if (message.ToolCalls is { Count: > 0 } calls)
        {
            wire["tool_calls"] = new JsonArray(calls.Select((call, i) => (JsonNode)new JsonObject
            {
                ["id"] = call.Id ?? $"call_{i}",
                ["type"] = "function",
                ["function"] = new JsonObject
                {
                    ["name"] = call.Function.Name,
                    ["arguments"] = ArgumentsToWire(call.Function.Arguments)
                }
            }).ToArray());
        }
        if (_echoReasoning && message.Role == "assistant" && !string.IsNullOrEmpty(message.Reasoning))
        {
            wire["reasoning_content"] = message.Reasoning;
        }
        if (message.Role == "tool")
        {
            wire["tool_call_id"] = message.ToolCallId;
            wire["name"] = message.ToolName;
        }
        return wire;
    }

    private JsonNode ArgumentsToWire(JsonElement arguments)
    {
        var raw = arguments.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null ? "{}" : arguments.GetRawText();
        return _argumentsAsObject ? JsonNode.Parse(raw)! : JsonValue.Create(raw);
    }

    private static string? GetString(JsonElement element, string property) =>
        element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    private static int? TryGetInt(JsonElement element, string property) =>
        element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.Number
            && value.TryGetInt32(out var number)
            ? number
            : null;

    private sealed class PendingCall
    {
        public string? Id;
        public string Name = "";
        public readonly StringBuilder Arguments = new();
    }
}

/// <summary>
/// Removes inline <c>&lt;think&gt;...&lt;/think&gt;</c> reasoning from a
/// streamed reply, for servers that don't split it out into its own field.
/// Tags may be split across chunks, so a possible partial tag at the end of
/// a chunk is held back until the next one decides it.
/// </summary>
sealed class ThinkTagFilter
{
    private const string Open = "<think>";
    private const string Close = "</think>";

    private string _pending = "";
    private bool _inThink;

    /// <param name="sawThinking">Reasoning text was swallowed.</param>
    /// <param name="reasoningEnded">A closing tag arrived without an opening
    /// one (the template opened the block in the prompt): everything
    /// returned before was reasoning, not reply.</param>
    public string Process(string chunk, out bool sawThinking, out bool reasoningEnded)
    {
        sawThinking = false;
        reasoningEnded = false;
        var text = _pending + chunk;
        _pending = "";
        var visible = new StringBuilder();

        while (text.Length > 0)
        {
            if (_inThink)
            {
                var close = text.IndexOf(Close, StringComparison.Ordinal);
                if (close < 0)
                {
                    var keep = PartialTagSuffix(text, Close);
                    sawThinking |= text.Length > keep;
                    _pending = text[^keep..];
                    return visible.ToString();
                }
                sawThinking |= close > 0;
                _inThink = false;
                text = text[(close + Close.Length)..].TrimStart('\n');
                continue;
            }

            var open = text.IndexOf(Open, StringComparison.Ordinal);
            var bareClose = text.IndexOf(Close, StringComparison.Ordinal);
            if (bareClose >= 0 && (open < 0 || bareClose < open))
            {
                // Whatever preceded it (this chunk and earlier ones) was
                // reasoning.
                visible.Clear();
                reasoningEnded = true;
                sawThinking = true;
                text = text[(bareClose + Close.Length)..].TrimStart('\n');
                continue;
            }
            if (open >= 0)
            {
                visible.Append(text[..open]);
                _inThink = true;
                text = text[(open + Open.Length)..];
                continue;
            }

            var hold = Math.Max(PartialTagSuffix(text, Open), PartialTagSuffix(text, Close));
            visible.Append(text[..^hold]);
            _pending = text[^hold..];
            break;
        }

        return visible.ToString();
    }

    /// <summary>Text held back at the end of the stream (never a whole tag).</summary>
    public string Flush()
    {
        var rest = _inThink ? "" : _pending;
        _pending = "";
        return rest;
    }

    // Length of the longest suffix of text that is a proper prefix of tag.
    private static int PartialTagSuffix(string text, string tag)
    {
        for (var length = Math.Min(tag.Length - 1, text.Length); length > 0; length--)
        {
            if (text.EndsWith(tag[..length], StringComparison.Ordinal))
            {
                return length;
            }
        }
        return 0;
    }
}
