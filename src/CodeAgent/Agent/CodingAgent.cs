using CodeAgent.Guardrails;
using CodeAgent.Ollama;
using CodeAgent.Tools;
using CodeAgent.Ui;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

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
    private readonly OllamaClient _ollama;
    private readonly AgentTools _tools;
    private readonly ConversationHistory _history;
    private readonly ILogger _logger;

    // Token counts reported by Ollama for the most recent request; more
    // accurate than any estimate we can make from character counts.
    private ChatStats? _lastStats;

    // Lowest chars-per-token ratio observed this session; starts at the ~4
    // the initial budget assumes. Only ever ratcheted DOWN: Ollama's
    // prompt_eval_count excludes tokens served from its session cache, so a
    // warm-cache request makes the ratio read absurdly high — only a
    // cache-cold measurement (like the session's first request) can be
    // trusted, and it can only ever tell us the budget is too generous.
    private double _charsPerToken = 4.0;

    // The command that must succeed after file changes before a final answer
    // is accepted; null when the project's stack isn't recognized.
    private readonly string? _verifyCommand;

    public CodingAgent(string rootFolder, string model, int contextTokens, ILoggerFactory? loggerFactory = null)
    {
        Model = model;
        _contextTokens = contextTokens;
        _verifyCommand = ProjectVerification.DetectCommand(rootFolder);

        // Code against the ILogger ABSTRACTION, not a concrete library:
        // Program.cs decides the provider (Serilog), and tests get a no-op
        // logger for free by leaving the factory out.
        loggerFactory ??= NullLoggerFactory.Instance;
        _logger = loggerFactory.CreateLogger<CodingAgent>();
        _ollama = new OllamaClient(logger: loggerFactory.CreateLogger<OllamaClient>());

        // Trim the transcript at ~80% of the window (at ~4 chars per token),
        // keeping the rest free for the model's reply and tool schemas.
        var maxChars = (int)(contextTokens * 4L * 80 / 100);
        _history = new ConversationHistory(BuildSystemPrompt(rootFolder, _verifyCommand), maxChars);

        // Cap individual tool results relative to the budget so one huge file
        // or command output can't monopolize the window, while still letting
        // big contexts see much more than small ones.
        _tools = new AgentTools(rootFolder, Math.Clamp(maxChars / 16, 8_000, 100_000),
            loggerFactory.CreateLogger<AgentTools>());
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
        // Input rail: screen the request before it ever reaches the model.
        // Blocks are logged at Warning — they are the security-relevant
        // events someone reviewing the log actually looks for.
        var verdict = GuardrailPipeline.CheckUserInput(task);
        if (!verdict.Allowed)
        {
            _logger.LogWarning("Guardrail {Rule} blocked user input: {Reason}. Input: {Task}",
                verdict.Rule, verdict.Reason, task);
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[blocked by guardrail '{verdict.Rule}': {verdict.Reason}]");
            Console.ResetColor();
            return;
        }

        _history.AddUser(planMode ? WrapPlanTask(task) : task);
        var tools = planMode ? AgentTools.PlanDefinitions : AgentTools.Definitions;
        var activity = new TaskActivity();

        // Every log line inside the task carries this id, so one task's
        // events can be filtered out of an interleaved log file.
        using var taskScope = _logger.BeginScope("Task {TaskId}", Guid.NewGuid());
        _logger.LogInformation("Task started: {Task} (model {Model}, plan mode: {PlanMode})", task, Model, planMode);
        try
        {
            await RunLoopAsync(task, tools, planMode, activity, cancellationToken);
        }
        finally
        {
            _logger.LogInformation("Task ended after {Turns} turns", activity.Turns);
            // Whichever way the task ended, recap what happened and show how
            // full the context is.
            PrintTaskSummary(activity);
            PrintContextSize();
        }
    }

    // How often a malformed reply (unparseable tool-call syntax, leaked
    // template markup, empty) is retried before the task is abandoned.
    // Consecutive failures only; any good reply resets the count.
    private const int MaxStreamRetries = 3;

    // The corrective note injected before resampling a malformed reply. One
    // constant for every failure shape, so old copies can be removed by
    // exact match when a fresh one is added.
    private const string MalformedReplyNote =
        "[Your previous reply was discarded because it was malformed (bad tool-call syntax, raw template markup, endless repetition, or empty). "
        + "Reply again, briefly and without repeating yourself: call a tool using the native tool-calling format, or give your final answer as plain text.]";

    // Chat-template control tokens that belong to the wire format, never in
    // reply text: Harmony (gpt-oss) and ChatML (qwen, ...). Ollama's parser
    // can eat part of the token before giving up, so degraded variants like
    // '<channel|>' leak too — accept any bracketed form that still has at
    // least one pipe, while a pipe-less '<end>' stays valid in ordinary text.
    private const string ControlTokenNames = "channel|message|start|end|return|call|im_start|im_end";
    private static readonly System.Text.RegularExpressions.Regex LeakedControlToken = new(
        $@"<\|({ControlTokenNames})\|?>|<({ControlTokenNames})\|>",
        System.Text.RegularExpressions.RegexOptions.Compiled);

    // Tool-call syntax written out as text instead of using native tool
    // calling — another way a reply can look like an answer but mean a call.
    private static readonly string[] FakedToolCallMarkers =
        { "<tool_call>", "</tool_call>", "<function_call>", "[TOOL_CALLS]" };

    // When either kind of markup leaks through, Ollama failed to parse what
    // the model meant (usually a tool call), and treating the reply as a
    // final answer would strand the task mid-way.
    internal static string? FindLeakedTemplateMarker(string content)
    {
        var control = LeakedControlToken.Match(content);
        return control.Success
            ? control.Value
            : FakedToolCallMarkers.FirstOrDefault(marker => content.Contains(marker, StringComparison.Ordinal));
    }

    // Every this many steps the original task and the progress so far are
    // re-injected into the transcript. Long multi-file tasks push the task
    // statement far behind a wall of tool traffic, and small models drift off
    // or forget what remains; a periodic recap anchors them again.
    private const int ProgressReminderInterval = 8;

    // Every this many steps the user is asked whether the task should keep
    // going, so a runaway task doesn't burn all of MaxSteps unattended.
    private const int StepCheckInterval = 30;

    // First line of the injected recap; also used to find and remove the
    // previous recap, so only the latest one occupies context.
    private const string ProgressReminderHeader = "[Automated progress check — this is not a message from the user.]";

    // How often an unverified final answer is sent back before it is
    // accepted anyway (with a console warning). A hard block could ping-pong
    // forever with a model that cannot get the build green or a user who
    // keeps declining the command.
    private const int MaxVerifyNudges = 2;

    private async Task RunLoopAsync(string task, object[] tools, bool planMode, TaskActivity activity, CancellationToken cancellationToken)
    {
        var streamRetries = 0;
        // Set when a file is changed, cleared when the project's verification
        // command succeeds — a final answer is only clean when this is false.
        var unverifiedChanges = false;
        var verifyNudges = 0;
        for (var step = 0; step < MaxSteps; step++)
        {
            activity.Turns++;
            if (step > 0 && step % ProgressReminderInterval == 0)
            {
                // Only the latest recap is worth keeping; drop the previous one.
                _history.RemoveUserMessages(content => content.StartsWith(ProgressReminderHeader, StringComparison.Ordinal));
                _history.AddUser(BuildProgressReminder(task, activity, planMode));
            }
            if (step > 0 && step % StepCheckInterval == 0)
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
            // What the transcript weighs right now, paired below with the
            // prompt token count Ollama reports for this same request to
            // measure the real chars-per-token ratio.
            var charsSent = _history.TotalChars;
            var spinner = Spinner.Start();
            // Whether streamed text left the cursor mid-line, so the spinner
            // (which redraws its own line with '\r') doesn't overwrite it.
            var midLine = false;
            try
            {
                // Stream tokens straight to the console; the spinner is
                // visible until the first token arrives, and again whenever
                // the model goes quiet to think.
                response = await _ollama.ChatAsync(Model, _history.Messages, tools, _contextTokens,
                    onToken: text =>
                    {
                        spinner.Stop();
                        // The model's own words stand out from tool chatter.
                        Console.ForegroundColor = ConsoleColor.Green;
                        Console.Write(text);
                        midLine = !text.EndsWith('\n');
                    },
                    onThinking: () =>
                    {
                        // Thinking models generate reasoning tokens that are
                        // not displayed; without the spinner the agent looks
                        // frozen for however long the model deliberates.
                        if (spinner.Stopped)
                        {
                            if (midLine)
                            {
                                Console.ResetColor();
                                Console.WriteLine();
                                midLine = false;
                            }
                            spinner = Spinner.Start();
                        }
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
                    Console.WriteLine($"[giving up after {MaxStreamRetries} retries — the model keeps producing unusable replies: {ex.Message}]");
                    return;
                }

                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"[reply discarded ({ex.Message}); retrying {streamRetries}/{MaxStreamRetries}]");
                Console.ResetColor();
                // One corrective note is enough; drop copies from earlier retries.
                _history.RemoveUserMessages(content => content == MalformedReplyNote);
                _history.AddUser(MalformedReplyNote);
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

            // Recalibrate the trimming budget from the real token count. The
            // initial ~4 chars/token guess overshoots for code-heavy text,
            // and a transcript that exceeds num_ctx makes Ollama silently
            // drop the oldest part of the prompt — usually the system prompt.
            if (response.Stats is { PromptTokens: > 0 } tokenStats && charsSent > 0)
            {
                var observed = Math.Max((double)charsSent / tokenStats.PromptTokens, 2.0);
                if (observed < _charsPerToken)
                {
                    _charsPerToken = observed;
                    _history.UpdateBudget((int)(_contextTokens * 0.8 * _charsPerToken));
                }
            }

            // Finish the streamed line before printing anything else.
            if (midLine)
            {
                Console.WriteLine();
            }

            // A reply with no tool calls should be the final answer — but not
            // when it leaked template markup (a tool call Ollama failed to
            // parse) or came back empty. Accepting those would strand the
            // task mid-way; discard the reply and resample instead, exactly
            // like a reply Ollama itself rejected.
            if (response.ToolCalls.Count == 0)
            {
                var problem = response.Content.Trim().Length == 0
                    ? "the reply was empty"
                    : FindLeakedTemplateMarker(response.Content) is { } marker
                        ? $"the reply leaked the template marker '{marker}' instead of making a proper tool call"
                        : null;
                if (problem is not null)
                {
                    if (++streamRetries > MaxStreamRetries)
                    {
                        Console.WriteLine($"[giving up after {MaxStreamRetries} retries — the model keeps producing malformed replies]");
                        return;
                    }

                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"[{problem}; retrying {streamRetries}/{MaxStreamRetries}]");
                    Console.ResetColor();
                    _history.RemoveUserMessages(content => content == MalformedReplyNote);
                    _history.AddUser(MalformedReplyNote);
                    continue;
                }
            }

            // Only consecutive malformed replies should abandon the task; a
            // good reply proves the model can still produce valid output.
            streamRetries = 0;

            _history.AddAssistant(response.Content, response.ToolCalls);

            // No tool calls means the streamed text was the final answer.
            if (response.ToolCalls.Count == 0)
            {
                // Changed files whose build/tests haven't passed since are
                // not a finished task; send the answer back with orders to
                // verify first. (Plan mode cannot change files.)
                if (unverifiedChanges && _verifyCommand is not null && verifyNudges < MaxVerifyNudges)
                {
                    verifyNudges++;
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"[final answer rejected: files changed but '{_verifyCommand}' has not succeeded since — asking the model to verify ({verifyNudges}/{MaxVerifyNudges})]");
                    Console.ResetColor();
                    _history.AddUser($"[Automated check — files were changed but '{_verifyCommand}' has not succeeded since. "
                        + "Run it with run_command now, fix any failures, and only then give your final answer.]");
                    continue;
                }
                if (unverifiedChanges && _verifyCommand is not null)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"[warning: task ended without a successful '{_verifyCommand}' after file changes]");
                    Console.ResetColor();
                }
                if (planMode)
                {
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.WriteLine("[plan complete — type 'go' to execute it, or refine it with another message]");
                    Console.ResetColor();
                }
                return;
            }

            var batchFailed = false;
            foreach (var call in response.ToolCalls)
            {
                var name = call.Function.Name;
                string result;
                if (batchFailed)
                {
                    // Calls bundled into one reply usually depend on each
                    // other (edit A, edit B, build); running the rest after a
                    // failure compounds the damage and misleads the model.
                    result = "Skipped: a previous tool call in this batch failed. Fix that first, then retry this call.";
                }
                else if (planMode && !AgentTools.IsAllowedInPlanMode(name))
                {
                    // Models sometimes hallucinate tools they weren't offered;
                    // in plan mode that must never change anything.
                    result = $"Error: '{name}' is not available in plan mode. Only investigate and ask questions, then present your plan.";
                }
                else
                {
                    result = await _tools.ExecuteAsync(name, call.Function.Arguments);
                    // Skipped calls did nothing, so only executed ones count
                    // as progress (the recap must not claim a file was edited
                    // when the edit never ran).
                    activity.Record(name, call.Function.Arguments);
                }
                batchFailed |= IsToolFailure(result);

                // Track whether the working tree is in a proven state: file
                // changes taint it, a passing verification command clears it.
                unverifiedChanges |= ChangesFiles(name, result);
                if (name == "run_command" && IsSuccessfulVerification(
                        _verifyCommand, AgentTools.GetString(call.Function.Arguments, "command") ?? "", result))
                {
                    unverifiedChanges = false;
                }

                Console.ForegroundColor = ConsoleColor.DarkCyan;
                Console.WriteLine($"[{name}] {DescribeCall(name, call.Function.Arguments, result)}");
                Console.ResetColor();
                // Log the call and its outcome, not the full result: tool
                // results can be whole files, and logs should record what
                // happened without duplicating the project's contents.
                _logger.LogInformation("Tool {ToolName} on {ToolSubject}: {ToolOutcome}",
                    name, SubjectOf(name, call.Function.Arguments), Summarize(result));
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

    // A successful write_file or edit_file leaves the code in an unproven
    // state until the verification command passes again.
    internal static bool ChangesFiles(string name, string result) =>
        (name == "write_file" && result.StartsWith("Wrote", StringComparison.Ordinal))
        || (name == "edit_file" && result.StartsWith("Replaced", StringComparison.Ordinal));

    // Whether this run_command call counts as the project's verification
    // passing. 'dotnet test' builds first, so it also satisfies a
    // 'dotnet build' requirement — never the other way around.
    internal static bool IsSuccessfulVerification(string? verifyCommand, string command, string result)
    {
        if (verifyCommand is null || !result.StartsWith("exit code: 0", StringComparison.Ordinal))
        {
            return false;
        }

        command = command.TrimStart();
        return command.StartsWith(verifyCommand, StringComparison.Ordinal)
            || (verifyCommand == "dotnet build" && command.StartsWith("dotnet test", StringComparison.Ordinal));
    }

    // Whether a tool result means the call did not do its job. Failed shell
    // commands (non-zero exit codes) are deliberately not failures here:
    // their output is feedback the model must see and react to.
    private static bool IsToolFailure(string result) =>
        result.StartsWith("Error", StringComparison.Ordinal)
        || result.StartsWith("File not found", StringComparison.Ordinal)
        || result.StartsWith("Unknown tool", StringComparison.Ordinal)
        || result.StartsWith("User declined", StringComparison.Ordinal)
        || result.StartsWith("Blocked by guardrail", StringComparison.Ordinal);

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
    internal static string BuildProgressReminder(string task, TaskActivity activity, bool planMode)
    {
        var recap = new System.Text.StringBuilder();
        recap.AppendLine(ProgressReminderHeader);
        recap.AppendLine($"Original task: {task}");
        AppendReminderLine(recap, "Files edited so far", activity.Edited);
        AppendReminderLine(recap, "Files written so far", activity.Wrote);
        AppendReminderLine(recap, "Files read so far", activity.Read);
        AppendReminderLine(recap, "Commands run so far", activity.Commands);
        recap.AppendLine("Check this against your checklist and continue with whatever remains.");
        recap.AppendLine("A file edited after you read it may differ from what you remember; re-read the relevant part before editing it again.");
        // Plan mode must end in a plan, not in changes-and-summary.
        recap.Append(planMode
            ? "When your investigation is complete, present your numbered plan as a plain text reply with no tool calls."
            : "When the whole task is done and verified, reply with a plain text summary and no tool calls.");
        return recap.ToString();
    }

    private static void AppendReminderLine(System.Text.StringBuilder recap, string label, IReadOnlyList<string> items)
    {
        if (items.Count > 0)
        {
            recap.AppendLine($"{label}: {string.Join(", ", items)}");
        }
    }

    // The console line for a tool call. For read_file and edit_file the file
    // name is enough — the content/snippet goes to the model, not the
    // terminal. Errors are still shown so the user sees what went wrong.
    private static string DescribeCall(string name, System.Text.Json.JsonElement args, string result)
    {
        if (name == "read_file" && !result.StartsWith("Error") && !result.StartsWith("File not found"))
        {
            var path = AgentTools.GetString(args, "path") ?? "?";
            var from = AgentTools.GetInt(args, "start_line");
            var to = AgentTools.GetInt(args, "end_line");
            return from is null && to is null ? path : $"{path} (lines {from?.ToString() ?? "1"}-{to?.ToString() ?? "end"})";
        }
        if (name == "edit_file" && result.StartsWith("Replaced"))
        {
            return AgentTools.GetString(args, "path") ?? "?";
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

    private static string BuildSystemPrompt(string rootFolder, string? verifyCommand) => $"""
        You are a coding agent working ONLY inside the folder: {rootFolder}
        All file paths must be relative to that folder.
        {(verifyCommand is null ? "" : $"""

        This project's verification command is '{verifyCommand}'. After you
        change any file, you MUST run it with run_command and fix any
        failures before giving your final answer. Never end a task with code
        that does not compile.
        """)}

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
