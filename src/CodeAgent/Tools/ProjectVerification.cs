namespace CodeAgent.Tools;

/// <summary>
/// Detects the command that proves the project still compiles and passes its
/// tests, so the agent loop can insist on running it before a task that
/// changed files is allowed to finish.
/// </summary>
static class ProjectVerification
{
    /// <summary>
    /// The verification command for the project, or null when the stack is
    /// not recognized (no enforcement then). C# today: 'dotnet test' when a
    /// test project exists (testing also builds), otherwise 'dotnet build'.
    /// </summary>
    public static string? DetectCommand(string rootFolder)
    {
        try
        {
            var projects = Directory.EnumerateFiles(rootFolder, "*.csproj", SearchOption.AllDirectories)
                .Where(f => !Path.GetRelativePath(rootFolder, f).Split(Path.DirectorySeparatorChar)
                    .Any(segment => AgentTools.IgnoredFolders.Contains(segment, StringComparer.OrdinalIgnoreCase)))
                .ToList();

            if (projects.Count == 0)
            {
                // A solution without reachable projects still builds.
                return Directory.EnumerateFiles(rootFolder, "*.sln", SearchOption.TopDirectoryOnly).Any()
                    ? "dotnet build"
                    : null;
            }

            return projects.Any(IsTestProject) ? "dotnet test" : "dotnet build";
        }
        catch
        {
            // Detection is best-effort; an unreadable folder just means no
            // enforced verification.
            return null;
        }
    }

    // A csproj counts as a test project when its name says so or it
    // references a known test SDK/framework.
    private static bool IsTestProject(string csprojPath)
    {
        if (Path.GetFileNameWithoutExtension(csprojPath).Contains("test", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        try
        {
            var content = File.ReadAllText(csprojPath);
            return content.Contains("Microsoft.NET.Test.Sdk", StringComparison.OrdinalIgnoreCase)
                || content.Contains("xunit", StringComparison.OrdinalIgnoreCase)
                || content.Contains("NUnit", StringComparison.OrdinalIgnoreCase)
                || content.Contains("MSTest", StringComparison.OrdinalIgnoreCase);
        }
        catch
        {
            return false;
        }
    }
}
