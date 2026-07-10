using System.Text;

/// <summary>
/// Builds a compact description of the project — a shallow folder tree plus
/// the key configuration files (.sln/.csproj, package.json, angular.json) —
/// that is appended to the system prompt. This saves the model from spending
/// its first steps rediscovering the stack on every task.
/// </summary>
static class ProjectOrientation
{
    // How deep the folder tree goes; enough to reveal the project layout
    // without flooding the prompt on big solutions.
    private const int MaxTreeDepth = 2;
    private const int MaxEntriesPerFolder = 30;

    // Config files worth showing in full (capped) because they identify the
    // stack: solution/projects for .NET, package.json/angular.json for Angular.
    private static readonly string[] KeyFileNames =
        { "angular.json", "package.json", "global.json", "Directory.Build.props" };
    private static readonly string[] KeyFileExtensions = { ".sln", ".csproj" };

    private const int MaxKeyFiles = 6;
    private const int MaxCharsPerKeyFile = 4000;

    public static string Describe(string rootFolder)
    {
        try
        {
            var text = new StringBuilder();

            text.AppendLine("Project layout (folders and files, clutter omitted):");
            AppendTree(text, rootFolder, rootFolder, depth: 0);

            foreach (var file in FindKeyFiles(rootFolder))
            {
                var content = File.ReadAllText(file);
                if (content.Length > MaxCharsPerKeyFile)
                {
                    content = content[..MaxCharsPerKeyFile] + "\n...[truncated]";
                }

                text.AppendLine();
                text.AppendLine($"--- {Path.GetRelativePath(rootFolder, file)} ---");
                text.AppendLine(content.Trim());
            }

            return text.ToString();
        }
        catch (Exception)
        {
            // Orientation is a nicety; a permissions problem or similar must
            // not stop the agent from starting.
            return "";
        }
    }

    private static void AppendTree(StringBuilder text, string folder, string root, int depth)
    {
        var indent = new string(' ', depth * 2);

        var entries = Directory.EnumerateFileSystemEntries(folder)
            .OrderBy(e => File.Exists(e) ? 1 : 0) // folders first
            .ThenBy(e => e, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var shown = 0;
        foreach (var entry in entries)
        {
            var name = Path.GetFileName(entry);
            if (Directory.Exists(entry))
            {
                if (AgentTools.IgnoredFolders.Contains(name, StringComparer.OrdinalIgnoreCase))
                {
                    continue;
                }
                text.AppendLine($"{indent}{name}/");
                if (depth + 1 < MaxTreeDepth)
                {
                    AppendTree(text, entry, root, depth + 1);
                }
            }
            else
            {
                text.AppendLine($"{indent}{name}");
            }

            if (++shown >= MaxEntriesPerFolder)
            {
                text.AppendLine($"{indent}...[{entries.Count - shown} more entries]");
                break;
            }
        }
    }

    private static IEnumerable<string> FindKeyFiles(string rootFolder) =>
        Directory.EnumerateFiles(rootFolder, "*", SearchOption.AllDirectories)
            .Where(f => !Path.GetRelativePath(rootFolder, f).Split(Path.DirectorySeparatorChar)
                .Any(segment => AgentTools.IgnoredFolders.Contains(segment, StringComparer.OrdinalIgnoreCase)))
            .Where(f => KeyFileNames.Contains(Path.GetFileName(f), StringComparer.OrdinalIgnoreCase)
                     || KeyFileExtensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase))
            .OrderBy(f => f.Count(c => c == Path.DirectorySeparatorChar)) // shallowest first
            .Take(MaxKeyFiles);
}
