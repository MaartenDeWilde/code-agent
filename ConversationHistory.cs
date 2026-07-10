/// <summary>
/// Holds the chat transcript and keeps it within a rough size budget so long
/// sessions don't push the system prompt out of a small model's context
/// window. When the budget is exceeded, the oldest tool results are replaced
/// with a placeholder; the system prompt and recent messages stay intact.
/// </summary>
class ConversationHistory
{
    // Rough character budget (~4 chars per token, aimed at an 8k-token context).
    private const int MaxChars = 32000;

    // The most recent messages are never elided: the model needs them to
    // continue the current task.
    private const int KeepRecentCount = 6;

    private const string ElidedMarker = "[older tool result elided to save context]";

    private readonly List<ChatMessage> _messages = new();

    public IReadOnlyList<ChatMessage> Messages => _messages;

    public ConversationHistory(string systemPrompt) =>
        _messages.Add(new ChatMessage { Role = "system", Content = systemPrompt });

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
        var total = _messages.Sum(m => m.Content.Length);
        if (total <= MaxChars)
        {
            return;
        }

        // Tool results are the bulkiest and least important messages once the
        // model has acted on them, so elide those first, oldest first.
        // Index 0 is the system prompt and is always skipped.
        for (var i = 1; i < _messages.Count - KeepRecentCount && total > MaxChars; i++)
        {
            var message = _messages[i];
            if (message.Role == "tool" && message.Content.Length > ElidedMarker.Length)
            {
                total -= message.Content.Length - ElidedMarker.Length;
                message.Content = ElidedMarker;
            }
        }
    }
}
