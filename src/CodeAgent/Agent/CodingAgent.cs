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
    private const int MaxSteps = 100;

    /// <summary>The Ollama model in use; may be switched between tasks.</summary>
    public string Model { get; set; }
    private readonly int _contextTokens;
    private readonly OllamaClient _ollama = new();
    private readonly AgentTools _tools;
    private readonly ConversationHistory _history;

    // Token counts reported by Ollama for the most recent request; more
    // accurate than any estimate we can make from character counts.
    private ChatStats? _lastStats;

    public CodingAgent(string rootFolder, string model, int contextTokens)
    {
        Model = model;
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
        var activity = new TaskActivity();

        try
        {
            await RunLoopAsync(task, tools, planMode, activity, cancellationToken);
        }
        finally
        {
            // Whichever way the task ended, recap what happened and show how
            // full the context is.
            PrintTaskSummary(activity);
            PrintContextSize();
        }
    }

    // How often a reply Ollama failed to parse (malformed tool-call syntax)
    // is retried before the task is abandoned.
    private const int MaxStreamRetries = 3;

    // Every this many steps the original task and the progress so far are
    // re-injected into the transcript. Long multi-file tasks push the task
    // statement far behind a wall of tool traffic, and small models drift off
    // or forget what remains; a periodic recap anchors them again.
    private const int ProgressReminderInterval = 8;

    private async Task RunLoopAsync(string task, object[] tools, bool planMode, TaskActivity activity, CancellationToken cancellationToken)
    {
        var streamRetries = 0;
        for (var step = 0; step < MaxSteps; step++)
        {
            activity.Turns++;
            if (step > 0 && step % ProgressReminderInterval == 0)
            {
                _history.AddUser(BuildProgressReminder(task, activity));
            }
            if (step >= 30)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"\n[Warning: {step} steps completed out of {MaxSteps}. Do you want to continue? (y/n)]");
                Console.ResetColor();
                var responseLine = Console.ReadLine();
                if (responseLine?.ToLower() != "y")
                {
                    Console.WriteLine("[Task aborted by user]");
                    return;
                }
            }
            ChatResponse response;
            var spinner = Spinner.Start();
            try
            {
                // Stream tokens straight to the console; the spinner is only
                // visible until the first token arrives.
                response = await _ollama.ChatAsync(Model, _history.Messages, tools, _contextTokens,
                    onToken: text =>
                    {
                        spinner.Stop();
                        // The model's own words stand out from tool chatter.
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.Write(text);
                    },
                    cancellationToken);
            }
            // Ollama aborts the stream when it cannot parse the model's
            // tool-call output (e.g. "XML syntax error ..."). The reply never
            // reached the transcript, so resampling — nudged by a corrective
            // note — usually recovers.
            catch (OllamaStreamException ex)
            {
                spinner.Stop();
                Console.ResetColor(); // tokens may have been streaming in green
                Console.WriteLine();

                if (++streamRetries > MaxStreamRetries)
                {
                    Console.WriteLine($"[giving up after {MaxStreamRetries} retries — Ollama keeps rejecting the model's output: {ex.Message}]");
                    return;
                }

                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[Ollama rejected the model's reply ({ex.Message}); retrying {streamRetries}/{MaxStreamRetries}]");
                Console.ResetColor();
                _history.AddUser("[Your previous reply was discarded because its tool-call syntax could not be parsed. Reply again, with correctly formatted tool calls.]");
                continue;
            }
            // Ctrl+C. The 'when' filter distinguishes a user interrupt from
            // an HttpClient timeout, which also surfaces as cancellation.
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                spinner.Stop();
                Console.ResetColor(); // tokens may have been streaming in green
                Console.WriteLine();
                Console.WriteLine("[task interrupted]");
                // Tell the model why the transcript just stops mid-step.
                _history.AddUser("[Interrupted: the user cancelled this step before it finished.]");
                return;
            }
            catch (Exception ex)
            {
                spinner.Stop();
                Console.ResetColor(); // tokens may have been streaming in green
                Console.WriteLine($"[error talking to Ollama: {ex.Message}]");
                return;
            }
            spinner.Stop();
            Console.ResetColor();

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
                activity.Record(name, call.Function.Arguments);
                Console.ForegroundColor = ConsoleColor.DarkCyan;
                Console.WriteLine($"[{name}] {DescribeCall(name, call.Function.Arguments, result)}");
                Console.ResetColor();
                _history.AddToolResult(name, result, SubjectOf(name, call.Function.Arguments));

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
    // large to remove. Token counts come straight from Ollama when available;
    // otherwise fall back to the ~4 chars/token estimate.
    private void PrintContextSize()
    {
        var chars = _history.TotalChars;
        var line = $"[{Model} | context: {chars:N0} chars, {100L * chars / _history.MaxChars}% of budget";
        line += _lastStats is { } stats
            ? $" | last request: {stats.PromptTokens:N0} prompt + {stats.ReplyTokens:N0} reply tokens, {stats.TokensPerSecond:F1} tok/s]"
            : $" (~{chars / 4:N0} tokens)]";

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine(line);
        Console.ResetColor();
    }

    // The argument that identifies what a tool call was about, kept alongside
    // the result so ConversationHistory can name it if the result is later
    // removed from the transcript.
    private static string? SubjectOf(string name, System.Text.Json.JsonElement args) => name switch
    {
        "read_file" or "write_file" or "edit_file" => AgentTools.GetString(args, "path"),
        "search_files" or "list_files" => AgentTools.GetString(args, "pattern"),
        "run_command" => AgentTools.GetString(args, "command"),
        _ => null
    };

    // The recap injected every ProgressReminderInterval steps. Marked as
    // automated so the model doesn't mistake it for new user instructions,
    // and built from TaskActivity so it reflects what actually happened.
    internal static string BuildProgressReminder(string task, TaskActivity activity)
    {
        var recap = new System.Text.StringBuilder();
        recap.AppendLine("[Automated progress check — this is not a message from the user.]");
        recap.AppendLine($"Original task: {task}");
        AppendReminderLine(recap, "Files edited so far", activity.Edited);
        AppendReminderLine(recap, "Files written so far", activity.Wrote);
        AppendReminderLine(recap, "Files read so far", activity.Read);
        AppendReminderLine(recap, "Commands run so far", activity.Commands);
        recap.AppendLine("Check this against your checklist and continue with whatever remains.");
        recap.AppendLine("A file edited after you read it may differ from what you remember; re-read the relevant part before editing it again.");
        recap.Append("When the whole task is done and verified, reply with a plain text summary and no tool calls.");
        return recap.ToString();
    }

    private static void AppendReminderLine(System.Text.StringBuilder recap, string label, IReadOnlyList<string> items)
    {
        if (items.Count > 0)
        {
            recap.AppendLine($"{label}: {string.Join(", ", items)}");
        }
    }

    // The console line for a tool call. For read_file the file name is
    // enough — its content goes to the model, not the terminal. Errors are
    // still shown so the user sees what went wrong.
    private static string DescribeCall(string name, System.Text.Json.JsonElement args, string result)
    {
        if (name == "read_file" && !result.StartsWith("Error") && !result.StartsWith("File not found"))
        {
            var path = AgentTools.GetString(args, "path") ?? "?";
            var from = AgentTools.GetInt(args, "start_line");
            var to = AgentTools.GetInt(args, "end_line");
            return from is null && to is null ? path : $"{path} (lines {from?.ToString() ?? "1"}-{to?.ToString() ?? "end"})";
        }

        return Summarize(result);
    }

    // Recap printed when a task ends, e.g.:
    //   [task summary: 7 turns]
    //     read:   Program.cs, src/App.cs
    //     edited: src/App.cs
    //     ran:    dotnet build
    private static void PrintTaskSummary(TaskActivity activity)
    {
        // A plain question answered without tools needs no recap.
        if (!activity.UsedAnyTool)
        {
            return;
        }

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"[task summary: {activity.Turns} {(activity.Turns == 1 ? "turn" : "turns")}]");
        PrintSummaryLine("searched", activity.Searches);
        PrintSummaryLine("read", activity.Read);
        PrintSummaryLine("edited", activity.Edited);
        PrintSummaryLine("wrote", activity.Wrote);
        PrintSummaryLine("ran", activity.Commands);
        if (activity.QuestionsAsked > 0)
        {
            Console.WriteLine($"  asked:    {activity.QuestionsAsked} {(activity.QuestionsAsked == 1 ? "question" : "questions")}");
        }
        Console.ResetColor();
    }

    private static void PrintSummaryLine(string label, IReadOnlyList<string> items)
    {
        if (items.Count == 0)
        {
            return;
        }

        // Keep the recap scannable when the agent touched many files.
        const int maxShown = 8;
        var shown = string.Join(", ", items.Take(maxShown));
        if (items.Count > maxShown)
        {
            shown += $" (+{items.Count - maxShown} more)";
        }
        Console.WriteLine($"  {label + ":",-9} {shown}");
    }

    /// <summary>
    /// What the agent did during one task, for the end-of-task recap and the
    /// periodic progress reminder.
    /// </summary>
    internal sealed class TaskActivity
    {
        public int Turns;
        public int QuestionsAsked;
        public readonly List<string> Searches = new();
        public readonly List<string> Read = new();
        public readonly List<string> Wrote = new();
        public readonly List<string> Edited = new();
        public readonly List<string> Commands = new();

        public bool UsedAnyTool =>
            QuestionsAsked > 0 || Searches.Count > 0 || Read.Count > 0
            || Wrote.Count > 0 || Edited.Count > 0 || Commands.Count > 0;

        public void Record(string name, System.Text.Json.JsonElement args)
        {
            switch (name)
            {
                case "search_files": AddDistinct(Searches, AgentTools.GetString(args, "pattern") ?? "?"); break;
                case "read_file": AddDistinct(Read, AgentTools.GetString(args, "path") ?? "?"); break;
                case "write_file": AddDistinct(Wrote, AgentTools.GetString(args, "path") ?? "?"); break;
                case "edit_file": AddDistinct(Edited, AgentTools.GetString(args, "path") ?? "?"); break;
                case "run_command": AddDistinct(Commands, AgentTools.GetString(args, "command") ?? "?"); break;
                case "ask_user": QuestionsAsked++; break;
            }
        }

        private static void AddDistinct(List<string> list, string item)
        {
            if (!list.Contains(item))
            {
                list.Add(item);
            }
        }
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

        When you create a file, write its COMPLETE final content in a single
        write_file call — never build a file up through repeated rewrites.
        To change an existing file, always use edit_file with a targeted
        find/replace; write_file on an existing file is rejected unless the
        content is substantially different. After changing code, verify your
        work with run_command (build or test) when the project has a suitable
        command.

        When a task requires changing MORE THAN ONE file, start with a short
        numbered checklist — one line per file with the change it needs — in
        the text of the same message as your first tool call (a text-only
        reply would end the task). Then work through the checklist in order,
        and use it to check what remains before you finish.

        If the task is ambiguous or an important decision could go multiple
        ways, use ask_user to check with the user instead of guessing.

        Keep taking actions until the task is actually done. When it is, reply
        with a normal text message summarizing what you did, without calling
        any more tools.

        {ProjectOrientation.Describe(rootFolder)}
        """;
}
