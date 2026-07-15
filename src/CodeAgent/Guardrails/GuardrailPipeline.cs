using System.Text.RegularExpressions;

namespace CodeAgent.Guardrails;

/// <summary>
/// The outcome of running text through a guardrail: allowed, or blocked with
/// the rule that fired and a reason the user (and the log) can act on.
/// Modeled as a result object rather than a bool so callers never have to
/// ask a second question ("blocked by what?") through shared state.
/// </summary>
record GuardrailVerdict(bool Allowed, string? Rule = null, string? Reason = null)
{
    public static readonly GuardrailVerdict Allow = new(true);
    public static GuardrailVerdict Block(string rule, string reason) => new(false, rule, reason);
}

/// <summary>
/// One named safety check. Keeping each rule behind this interface means new
/// checks (an LLM-based classifier, an allowlist, rate limiting) can be added
/// to the pipeline without touching the call sites.
/// </summary>
interface IGuardrail
{
    string Name { get; }
    GuardrailVerdict Check(string input);
}

/// <summary>
/// A guardrail driven by a list of regex patterns, each with its own reason.
/// Pattern matching is a first, cheap line of defense: it catches the obvious
/// cases instantly and offline, but a determined user can always rephrase —
/// which is why the dangerous actions themselves (shell commands, overwrites)
/// are ALSO gated by user confirmation at execution time. Guardrails reduce
/// risk in layers; no single layer is the whole story.
/// </summary>
class PatternGuardrail : IGuardrail
{
    public string Name { get; }
    private readonly (Regex Pattern, string Reason)[] _rules;

    public PatternGuardrail(string name, params (string Pattern, string Reason)[] rules)
    {
        Name = name;
        _rules = rules
            .Select(rule => (new Regex(rule.Pattern, RegexOptions.IgnoreCase | RegexOptions.Compiled), rule.Reason))
            .ToArray();
    }

    public GuardrailVerdict Check(string input)
    {
        foreach (var (pattern, reason) in _rules)
        {
            if (pattern.IsMatch(input))
            {
                return GuardrailVerdict.Block(Name, reason);
            }
        }

        return GuardrailVerdict.Allow;
    }
}

/// <summary>
/// The guardrails, grouped by trust boundary. Input rails screen what the
/// user asks for before it reaches the model; command rails screen what the
/// model asks to execute before it reaches the shell. The two lists differ
/// because the threats differ: users try to talk the model out of its rules,
/// while a (confused or manipulated) model tries to run destructive commands.
/// </summary>
static class GuardrailPipeline
{
    private static readonly IGuardrail[] InputRails =
    {
        new PatternGuardrail("prompt-injection",
            // Attempts to displace the system prompt. The agent's rules
            // (stay in the folder, confirm commands) only hold if the task
            // text can't simply talk the model out of them.
            (@"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions", "the request tries to override the agent's instructions"),
            (@"disregard\s+(your|the|all)\s+(system\s+prompt|instructions|rules)", "the request tries to override the agent's instructions"),
            (@"you\s+are\s+no\s+longer\s+(a|an|bound|restricted)", "the request tries to re-role the agent outside its rules"),
            (@"pretend\s+(you\s+have|there\s+are)\s+no\s+(rules|restrictions|limits)", "the request tries to bypass the agent's restrictions"),
            (@"\bjailbreak\b|\bDAN\s+mode\b", "the request asks for a jailbreak")),

        new PatternGuardrail("destructive-intent",
            (@"(delete|remove|wipe)\s+(all|every|the\s+entire)\s+.{0,20}(file|folder|director|project|disk|drive)", "the request asks for mass deletion"),
            (@"rm\s+-[a-z]*r[a-z]*f?\s+([/~]|\$HOME)", "the request asks to recursively delete from the filesystem root or home directory"),
            (@"format\s+(the\s+)?(disk|drive|c:)", "the request asks to format a drive")),

        new PatternGuardrail("secrets-exfiltration",
            // Reading key material is suspicious on its own inside a coding
            // task; combining it with sending it somewhere is never OK.
            (@"(\.ssh/|id_rsa|id_ed25519|\.aws/credentials|\.netrc)", "the request touches credential files outside the project"),
            (@"(send|upload|post|exfiltrate|email)\s+.{0,40}(password|secret|token|api[_\s-]?key|credential)", "the request asks to send secrets somewhere")),
    };

    private static readonly IGuardrail[] CommandRails =
    {
        new PatternGuardrail("destructive-command",
            // Hard blocks that not even user confirmation should offer:
            // these commands are never part of a legitimate coding task.
            (@"rm\s+(-[a-z]+\s+)*(-[a-z]*r[a-z]*\s+)([/~]\s*$|[/~]\s|--no-preserve-root)", "recursive delete of the filesystem root or home directory"),
            (@":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:", "fork bomb"),
            (@"\bmkfs(\.| )", "formats a filesystem"),
            (@"\bdd\b.{0,60}\bof=/dev/", "writes raw bytes over a device"),
            (@">\s*/dev/sd[a-z]", "writes over a raw disk device"),
            (@"chmod\s+(-[a-z]+\s+)*777\s+/\s*$", "makes the filesystem root world-writable")),

        new PatternGuardrail("remote-code-execution",
            // Piping a download straight into a shell executes unreviewed
            // code from the network — the model must fetch, show, then run.
            (@"(curl|wget)\b.+\|\s*(ba|z|da|)sh\b", "pipes a network download directly into a shell")),
    };

    /// <summary>Screens a task the user typed, before it reaches the model.</summary>
    public static GuardrailVerdict CheckUserInput(string task) => Run(InputRails, task);

    /// <summary>Screens a shell command the model wants to run, before the user is even asked.</summary>
    public static GuardrailVerdict CheckCommand(string command) => Run(CommandRails, command);

    private static GuardrailVerdict Run(IGuardrail[] rails, string input)
    {
        foreach (var rail in rails)
        {
            var verdict = rail.Check(input);
            if (!verdict.Allowed)
            {
                return verdict;
            }
        }

        return GuardrailVerdict.Allow;
    }
}
