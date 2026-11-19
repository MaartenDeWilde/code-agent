using System.Text.Json;
using System.Text.Json.Serialization;
using CodeAgent.Ollama;
using CodeAgent.OpenAi;
using Microsoft.Extensions.Logging;

namespace CodeAgent.Llm;

// ======================================================================
// Chat message / tool call models shared between the backends and the
// agent. The JSON names follow Ollama's wire format (snake_case), so the
// Ollama client can send them as-is; other backends map them to their own
// wire format.
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

    // Present on 'tool' role messages: the id of the call this answers.
    // OpenAI-style backends pair results to calls by id; Ollama doesn't use
    // it, so it stays out of Ollama's wire format.
    [JsonIgnore]
    public string? ToolCallId { get; set; }

    // Streamed by thinking models (gpt-oss, ...): reasoning tokens that are
    // not part of the reply. Never echoed back; only used to signal that the
    // model is still working while nothing visible is being produced.
    [JsonPropertyName("thinking")]
    public string? Thinking { get; set; }

    // The reasoning that preceded an assistant turn, kept only for backends
    // that want it echoed back (Embla re-renders the reasoning of the turns
    // still being built, so leaving it out invalidates its prefix cache).
    // Each backend maps it to its own wire format.
    [JsonIgnore]
    public string? Reasoning { get; set; }

    // Local bookkeeping, never sent to the model: the main argument of the
    // call that produced this result (file path, search pattern, command).
    // Lets ConversationHistory say WHAT was removed, so the model knows which
    // tool call to repeat instead of working from memory.
    [JsonIgnore]
    public string? ToolSubject { get; set; }
}

class ToolCall
{
    // Set by OpenAI-style backends, which need it echoed back on the result.
    [JsonIgnore]
    public string? Id { get; set; }

    [JsonPropertyName("function")]
    public ToolFunction Function { get; set; } = new();
}

class ToolFunction
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = "";

    // Always a parsed JSON object here, whatever the backend sends.
    [JsonPropertyName("arguments")]
    public JsonElement Arguments { get; set; }
}

/// <summary>The model's complete reply: streamed text plus any tool calls.
/// Reasoning is only set by backends that need it echoed back.</summary>
record ChatResponse(string Content, List<ToolCall> ToolCalls, ChatStats? Stats, string? Reasoning = null);

/// <summary>
/// Real token counts reported by the backend at the end of the stream —
/// unlike our chars/4 estimate, PromptTokens is what the context cost.
/// </summary>
record ChatStats(int PromptTokens, int ReplyTokens, double TokensPerSecond);

/// <summary>
/// A reply that broke down mid-stream — the backend reported an error inside
/// an otherwise successful response (e.g. its parser failed on the model's
/// tool-call syntax), or the reply degenerated. These are worth retrying: a
/// resample usually produces usable output.
/// </summary>
class ModelStreamException(string message) : Exception(message);

/// <summary>A model server the agent can talk to (Ollama, mlx_lm, ...).</summary>
interface IChatClient
{
    /// <summary>Short name shown in the UI and in error messages.</summary>
    string Name { get; }

    /// <summary>Names of the models the server offers.</summary>
    Task<List<string>> ListModelsAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Sends the conversation to the model and streams the reply.
    /// Text tokens are passed to <paramref name="onToken"/> as they arrive;
    /// <paramref name="onThinking"/> fires for reasoning tokens of thinking
    /// models, whose text is deliberately not exposed — only the fact that
    /// the model is still generating. Tool calls are collected and returned
    /// once the stream completes. Throws <see cref="ModelStreamException"/>
    /// for a reply worth resampling.
    /// </summary>
    Task<ChatResponse> ChatAsync(
        string model,
        IReadOnlyList<ChatMessage> history,
        object[] tools,
        int contextTokens,
        Action<string> onToken,
        Action? onThinking = null,
        CancellationToken cancellationToken = default);
}

static class ReplyChecks
{
    // Degeneration check: with repetition penalties disabled, a small model
    // can fall into re-emitting the same few sentences forever ("Wait, I'll
    // fix X first. / Actually, I'll do it all at once." ad infinitum). A
    // real, wordy line recurring this often in one reply is a loop, not
    // style. Lines must contain letters (markdown table rules don't) and some
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
}

/// <summary>The model servers the user can pick from, with their defaults.</summary>
record Backend(string Name, string Description, string DefaultModel, Func<ILoggerFactory, IChatClient> Create)
{
    public static readonly Backend[] All =
    [
        // embla-metal's server is OpenAI-compatible but ignores the model
        // field: it serves the one model it was started with.
        new("embla", "embla-metal (embla serve) on 127.0.0.1:8080", "Embla1-4bit",
            factory => new OpenAiClient("embla", new Uri("http://127.0.0.1:8080"), logger: factory.CreateLogger<OpenAiClient>(),
                argumentsAsObject: false, echoReasoning: true)),
        new("mlx", "mlx_lm.server on localhost:8080 (OpenAI-compatible API)", "sevren-ai/Embla1-4bit",
            factory => new OpenAiClient("mlx", new Uri("http://localhost:8080"), logger: factory.CreateLogger<OpenAiClient>())),
        new("ollama", "Ollama on localhost:11434", "gemma4:12b-mlx",
            factory => new OllamaClient(logger: factory.CreateLogger<OllamaClient>())),
    ];

    public static Backend? Find(string name) =>
        All.FirstOrDefault(b => b.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
}
