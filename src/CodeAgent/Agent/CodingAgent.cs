using CodeAgent.Ollama;
using CodeAgent.Tools;
using CodeAgent.Ui;

namespace CodeAgent.Agent;

/// <summary>
/// The agent loop: send the conversation to the model, execute the tool calls
/// it makes, feed the results back, and repeat until the model answers with
/// plain text (the final answer) or the step limit is reached.
/// </summary>
class CodingAgent
{
    // Build/fix cycles take many steps; give the agent room to iterate.
    private const int MaxSteps = 30;

    private readonly string _model;
    private readonly int _contextTokens;
    private readonly OllamaClient _ollama = new();
    private readonly AgentTools _tools;
    private readonly ConversationHistory _history;

    // Token counts reported by Ollama for the most recent request; more
    // accurate than any estimate we can make from character counts.
    private ChatStats? _lastStats;

    public CodingAgent(string rootFolder, string model, int contextTokens)
    {
        _model = model;
        _contextTokens = contextTokens;

        // Trim the transcript at ~80% of the window (at ~4 chars per token),
        // keeping the rest free for the model's reply and tool schemas.
        var maxChars = (int)(contextTokens * 4L * 80 / 100);
        _history = new ConversationHistory(BuildSystemPrompt(rootFolder), maxChars);

        // Cap individual tool results relative to the budget so one huge file
        // or command output can't monopolize the window, while still letting
        // big contexts see much more than small ones.
        _tools = new AgentTools(rootFolder, Math.Clamp(maxChars / 16, 8_000, 100_000));
    }

    /// <summary>
    /// Plan mode: the model may only investigate (read-only tools) and must
    /// end with a proposed plan instead of making changes. Approving the plan
    /// is a normal follow-up task, e.g. "The plan is approved. Execute it."
    /// </summary>
    public Task PlanAsync(string task, CancellationToken cancellationToken = default) =>
        RunAsync(task, planMode: true, cancellationToken);

    public async Task RunAsync(string task, bool planMode = false, CancellationToken cancellationToken = default)
    {
        _history.AddUser(planMode ? WrapPlanTask(task) : task);
        var tools = planMode ? AgentTools.PlanDefinitions : AgentTools.Definitions;

        try
        {
            await RunLoopAsync(tools, planMode, cancellationToken);
        }
        finally
        {
            // Always show how full the context is, whichever way the task ended.
            PrintContextSize();
        }
    }

    private async Task RunLoopAsync(object[] tools, bool planMode, CancellationToken cancellationToken)
    {
        for (var step = 0; step < MaxSteps; step++)
        {
            ChatResponse response;
            var spinner = Spinner.Start();
            try
            {
                // Stream tokens straight to the console; the spinner is only
                // visible until the first token arrives.
                response = await _ollama.ChatAsync(_model, _history.Messages, tools, _contextTokens,
                    onToken: text =>
                    {
                        spinner.Stop();
                        Console.Write(text);
                    },
                    cancellationToken);
            }
            // Ctrl+C. The 'when' filter distinguishes a user interrupt from
            // an HttpClient timeout, which also surfaces as cancellation.
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                spinner.Stop();
                Console.WriteLine();
                Console.WriteLine("[task interrupted]");
                // Tell the model why the transcript just stops mid-step.
                _history.AddUser("[Interrupted: the user cancelled this step before it finished.]");
                return;
            }
            catch (Exception ex)
            {
                spinner.Stop();
                Console.WriteLine($"[error talking to Ollama: {ex.Message}]");
                return;
            }
            spinner.Stop();

            _lastStats = response.Stats ?? _lastStats;

            // Finish the streamed line before printing tool output.
            if (response.Content.Length > 0)
            {
                Console.WriteLine();
            }

            _history.AddAssistant(response.Content, response.ToolCalls);

            // No tool calls means the streamed text was the final answer.
            if (response.ToolCalls.Count == 0)
            {
                if (response.Content.Length == 0)
                {
                    Console.WriteLine("(no answer given)");
                }
                if (planMode)
                {
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine("[plan complete — type 'go' to execute it, or refine it with another message]");
                    Console.ResetColor();
                }
                return;
            }

            foreach (var call in response.ToolCalls)
            {
                var name = call.Function.Name;
                // Models sometimes hallucinate tools they weren't offered;
                // in plan mode that must never change anything.
                var result = planMode && !AgentTools.IsAllowedInPlanMode(name)
                    ? $"Error: '{name}' is not available in plan mode. Only investigate and ask questions, then present your plan."
                    : await _tools.ExecuteAsync(name, call.Function.Arguments);
                Console.WriteLine($"[{name}] {Summarize(result)}");
                _history.AddToolResult(name, result);

                // Honor Ctrl+C between tool calls too, not just while the
                // model is generating.
                if (cancellationToken.IsCancellationRequested)
                {
                    Console.WriteLine("[task interrupted]");
                    _history.AddUser("[Interrupted: the user cancelled the task after this tool call.]");
                    return;
                }
            }
        }

        Console.WriteLine("[agent stopped: reached max steps without a final answer]");
    }

    // E.g. "[context: 12,840 chars, 40% of budget | last request: 3,105 prompt
    // + 210 reply tokens, 24.8 tok/s]". The budget is what ConversationHistory
    // trims down to, so >100% can occur briefly when recent messages are too
    // large to elide. Token counts come straight from Ollama when available;
    // otherwise fall back to the ~4 chars/token estimate.
    private void PrintContextSize()
    {
        var chars = _history.TotalChars;
        var line = $"[context: {chars:N0} chars, {100L * chars / _history.MaxChars}% of budget";
        line += _lastStats is { } stats
            ? $" | last request: {stats.PromptTokens:N0} prompt + {stats.ReplyTokens:N0} reply tokens, {stats.TokensPerSecond:F1} tok/s]"
            : $" (~{chars / 4:N0} tokens)]";

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine(line);
        Console.ResetColor();
    }

    private static string Summarize(string result)
    {
        var oneLine = result.Replace("\n", " ");
        return oneLine.Length > 120 ? oneLine[..120] + "..." : oneLine;
    }

    // Instructions prepended to the task in plan mode.
    private static string WrapPlanTask(string task) => $"""
        Plan the following task, but do NOT make any changes yet. Investigate
        the project with the read-only tools and ask the user questions where
        their input matters. Then present a concise numbered plan of the
        changes you would make, and stop so the user can approve or refine it.

        Task: {task}
        """;

    private static string BuildSystemPrompt(string rootFolder) => $"""
        You are a coding agent working ONLY inside the folder: {rootFolder}
        All file paths must be relative to that folder.

        Use the available tools to explore the project before changing it.
        Use search_files to locate symbols and usages instead of reading files
        one by one, and read_file with start_line/end_line for large files.
        read_file output prefixes each line with its number for reference;
        never copy those prefixes into edit_file or write_file content.

        Prefer edit_file for small targeted changes and write_file only for
        new files or full rewrites. After changing code, verify your work with
        run_command (build or test) when the project has a suitable command.

        If the task is ambiguous or an important decision could go multiple
        ways, use ask_user to check with the user instead of guessing.

        Keep taking actions until the task is actually done. When it is, reply
        with a normal text message summarizing what you did, without calling
        any more tools.

        {ProjectOrientation.Describe(rootFolder)}
        """;
}
