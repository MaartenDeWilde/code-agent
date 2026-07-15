namespace CodeAgent.Ui;

/// <summary>
/// Small console spinner shown while waiting for the model's first token.
/// Start it with <see cref="Start"/>, then call <see cref="Stop"/> (safe to
/// call multiple times) as soon as real output arrives.
/// </summary>
class Spinner
{
    private readonly CancellationTokenSource _cts = new();
    private readonly Task _task;
    private int _stopped;

    private Spinner() => _task = Task.Run(AnimateAsync);

    public static Spinner Start() => new();

    /// <summary>
    /// Whether this spinner has been stopped. A spinner is single-use;
    /// callers that want the animation back start a new one.
    /// </summary>
    public bool Stopped => Volatile.Read(ref _stopped) == 1;

    public void Stop()
    {
        // Interlocked makes Stop idempotent even when called from both the
        // token callback and the finally block.
        if (Interlocked.Exchange(ref _stopped, 1) == 1)
        {
            return;
        }

        _cts.Cancel();
        try
        {
            // Wait until the spinner has erased its line, so streamed tokens
            // don't get mixed into the animation.
            _task.Wait();
        }
        catch
        {
            // Cancellation is the expected way out.
        }
        _cts.Dispose();
    }

    private async Task AnimateAsync()
    {
        var frames = new[] { '|', '/', '-', '\\' };
        var i = 0;
        try
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            while (!_cts.Token.IsCancellationRequested)
            {
                Console.Write($"\r{frames[i++ % frames.Length]} thinking...");
                await Task.Delay(100, _cts.Token);
            }
        }
        catch (OperationCanceledException)
        {
        }
        finally
        {
            Console.ResetColor();
            // Erase the spinner line.
            Console.Write("\r" + new string(' ', 20) + "\r");
        }
    }
}
