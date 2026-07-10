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
    private readonly OllamaClient _ollama = new();
    private readonly AgentTools _tools;
    private readonly ConversationHistory _history;

    public CodingAgent(string rootFolder, string model)
    {
        _model = model;
        _tools = new AgentTools(rootFolder);
        _history = new ConversationHistory(BuildSystemPrompt(rootFolder));
    }

    public async Task RunAsync(string task)
    {
        _history.AddUser(task);

        for (var step = 0; step < MaxSteps; step++)
        {
            ChatResponse response;
            var spinner = Spinner.Start();
            try
            {
                // Stream tokens straight to the console; the spinner is only
                // visible until the first token arrives.
                response = await _ollama.ChatAsync(_model, _history.Messages, AgentTools.Definitions,
                    onToken: text =>
                    {
                        spinner.Stop();
                        Console.Write(text);
                    });
            }
            catch (Exception ex)
            {
                spinner.Stop();
                Console.WriteLine($"[error talking to Ollama: {ex.Message}]");
                return;
            }
            spinner.Stop();

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
                return;
            }

            foreach (var call in response.ToolCalls)
            {
                var name = call.Function.Name;
                var result = await _tools.ExecuteAsync(name, call.Function.Arguments);
                Console.WriteLine($"[{name}] {Summarize(result)}");
                _history.AddToolResult(name, result);
            }
        }

        Console.WriteLine("[agent stopped: reached max steps without a final answer]");
    }

    private static string Summarize(string result)
    {
        var oneLine = result.Replace("\n", " ");
        return oneLine.Length > 120 ? oneLine[..120] + "..." : oneLine;
    }

    private static string BuildSystemPrompt(string rootFolder) => $"""
        You are a coding agent working ONLY inside the folder: {rootFolder}
        All file paths must be relative to that folder.

        Use the available tools to explore the project before changing it:
        list files, read the relevant ones, and only then write or edit code.
        Prefer edit_file for small targeted changes and write_file only for
        new files or full rewrites. After changing code, verify your work with
        run_command (build or test) when the project has a suitable command.

        Keep taking actions until the task is actually done. When it is, reply
        with a normal text message summarizing what you did, without calling
        any more tools.
        """;
}
