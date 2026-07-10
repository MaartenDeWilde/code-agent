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
    // The most recent messages are never elided: the model needs them to
    // continue the current task.
    private const int KeepRecentCount = 6;

    private const string ElidedMarker = "[older tool result elided to save context]";

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

    public void AddToolResult(string toolName, string result) =>
        Add(new ChatMessage { Role = "tool", ToolName = toolName, Content = result });

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
        // transcript once the model has acted on it, so elide it oldest
        // first. Index 0 is the system prompt and is always skipped.
        for (var i = 1; i < _messages.Count - KeepRecentCount && total > MaxChars; i++)
        {
            var message = _messages[i];

            if (message.Role == "tool" && message.Content.Length > ElidedMarker.Length)
            {
                total -= message.Content.Length - ElidedMarker.Length;
                message.Content = ElidedMarker;
            }

            // Tool-call arguments (e.g. a whole file passed to write_file)
            // are echoed back on every request; drop the old ones too.
            if (message.Role == "assistant" && message.ToolCalls is not null)
            {
                var names = string.Join(", ", message.ToolCalls.Select(c => c.Function.Name));
                var note = $"[earlier tool calls elided: {names}]";
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
