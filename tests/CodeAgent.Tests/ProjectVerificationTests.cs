using CodeAgent.Tools;

namespace CodeAgent.Tests;

public class ProjectVerificationTests : IDisposable
{
    private readonly string _root;

    public ProjectVerificationTests() =>
        _root = Directory.CreateTempSubdirectory("projectverification-test-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private void CreateFile(string relativePath, string content)
    {
        var full = Path.Combine(_root, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full, content);
    }

    [Fact]
    public void CsharpProjectWithoutTests_VerifiesWithBuild()
    {
        CreateFile("src/App/App.csproj", "<Project Sdk=\"Microsoft.NET.Sdk\"></Project>");

        Assert.Equal("dotnet build", ProjectVerification.DetectCommand(_root));
    }

    [Fact]
    public void CsharpProjectWithTestProject_VerifiesWithTest()
    {
        CreateFile("src/App/App.csproj", "<Project Sdk=\"Microsoft.NET.Sdk\"></Project>");
        CreateFile("tests/App.Tests/App.Tests.csproj", "<Project Sdk=\"Microsoft.NET.Sdk\"></Project>");

        Assert.Equal("dotnet test", ProjectVerification.DetectCommand(_root));
    }

    [Fact]
    public void TestProjectRecognizedByReference_NotJustByName()
    {
        CreateFile("src/App/App.csproj", "<Project Sdk=\"Microsoft.NET.Sdk\"></Project>");
        CreateFile("src/Specs/Specs.csproj",
            "<Project Sdk=\"Microsoft.NET.Sdk\"><ItemGroup><PackageReference Include=\"Microsoft.NET.Test.Sdk\" /></ItemGroup></Project>");

        Assert.Equal("dotnet test", ProjectVerification.DetectCommand(_root));
    }

    [Fact]
    public void ProjectsInBuildOutputFolders_AreIgnored()
    {
        // Only clutter contains csproj files; there is no real project.
        CreateFile("obj/Generated/Generated.csproj", "<Project></Project>");
        CreateFile("bin/Copy/Copy.Tests.csproj", "<Project></Project>");

        Assert.Null(ProjectVerification.DetectCommand(_root));
    }

    [Fact]
    public void UnrecognizedStack_HasNoVerificationCommand()
    {
        CreateFile("main.py", "print('hi')");

        Assert.Null(ProjectVerification.DetectCommand(_root));
    }

    [Fact]
    public void SolutionWithoutProjects_StillBuilds()
    {
        CreateFile("All.sln", "Microsoft Visual Studio Solution File");

        Assert.Equal("dotnet build", ProjectVerification.DetectCommand(_root));
    }
}
