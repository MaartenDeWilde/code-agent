using System.Text.Json;
using CodeAgent.Ollama;

namespace CodeAgent.Agent;

/// <summary>
/// Holds the chat transcript and keeps it within a rough size budget so long
/// sessions don't push the system prompt out of a small model's context
/// window. When the budget is exceeded, the oldest tool results and tool-call
/// arguments are replaced with placeholders; the system prompt and recent
/// messages stay intact.
/// </summary>
class ConversationHistory
{
    // The most recent messages are never removed: the model needs them to
    // continue the current task.
    private const int KeepRecentCount = 6;

    // What replaces a removed tool result. Naming the tool and its subject
    // (file path, pattern, command) matters: the model can then re-issue that
    // exact call when it needs the content again, instead of editing a file
    // it only remembers.
    private static string RemovedMarker(ChatMessage message)
    {
        var subject = message.ToolSubject is { Length: > 0 } s ? $" for '{s}'" : "";
        return $"[older {message.ToolName} result{subject} removed to save space — call the tool again if you still need it]";
    }

    private readonly List<ChatMessage> _messages = new();

    public IReadOnlyList<ChatMessage> Messages => _messages;

    /// <summary>Character budget the transcript is trimmed down to.</summary>
    public int MaxChars { get; }

    /// <summary>Current transcript size in characters, as sent to the model.</summary>
    public int TotalChars => _messages.Sum(SizeOf);

    public ConversationHistory(string systemPrompt, int maxChars)
    {
        MaxChars = maxChars;
        _messages.Add(new ChatMessage { Role = "system", Content = systemPrompt });
    }

    public void AddUser(string content) =>
        Add(new ChatMessage { Role = "user", Content = content });

    public void AddAssistant(string content, List<ToolCall> toolCalls) =>
        Add(new ChatMessage
        {
            Role = "assistant",
            Content = content,
            ToolCalls = toolCalls.Count > 0 ? toolCalls : null
        });

    /// <param name="subject">The call's main argument (file path, pattern,
    /// command), used to describe the result if it is later removed.</param>
    public void AddToolResult(string toolName, string result, string? subject = null) =>
        Add(new ChatMessage { Role = "tool", ToolName = toolName, Content = result, ToolSubject = subject });

    private void Add(ChatMessage message)
    {
        _messages.Add(message);
        TrimToBudget();
    }

    private void TrimToBudget()
    {
        var total = _messages.Sum(SizeOf);
        if (total <= MaxChars)
        {
            return;
        }

        // Old tool traffic is the bulkiest and least important part of the
        // transcript once the model has acted on it, so remove it oldest
        // first. Index 0 is the system prompt and is always skipped.
        for (var i = 1; i < _messages.Count - KeepRecentCount && total > MaxChars; i++)
        {
            var message = _messages[i];

            // Only replace content when the marker is actually smaller: this
            // both guarantees progress and makes a second pass a no-op.
            if (message.Role == "tool" && message.Content.Length > RemovedMarker(message).Length)
            {
                var marker = RemovedMarker(message);
                total -= message.Content.Length - marker.Length;
                message.Content = marker;
            }

            // Tool-call arguments (e.g. a whole file passed to write_file)
            // are echoed back on every request; drop the old ones too.
            if (message.Role == "assistant" && message.ToolCalls is not null)
            {
                var names = string.Join(", ", message.ToolCalls.Select(c => c.Function.Name));
                var note = $"[earlier tool calls removed to save space: {names}]";
                total -= message.ToolCalls.Sum(ArgumentSize) + message.Content.Length;
                message.ToolCalls = null;
                message.Content = string.IsNullOrEmpty(message.Content) ? note : $"{message.Content}\n{note}";
                total += message.Content.Length;
            }
        }
    }

    // Approximate size of a message as the model sees it, including tool-call
    // arguments, which can dwarf the visible content.
    private static int SizeOf(ChatMessage message) =>
        message.Content.Length + (message.ToolCalls?.Sum(ArgumentSize) ?? 0);

    private static int ArgumentSize(ToolCall call) =>
        call.Function.Arguments.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null
            ? 0
            : call.Function.Arguments.GetRawText().Length;
}
