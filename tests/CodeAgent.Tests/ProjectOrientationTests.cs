using CodeAgent.Tools;

namespace CodeAgent.Tests;

public class ProjectOrientationTests : IDisposable
{
    private readonly string _root;

    public ProjectOrientationTests()
    {
        _root = Directory.CreateTempSubdirectory("orientation-test-").FullName;
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private void CreateFile(string relativePath, string content)
    {
        var full = Path.Combine(_root, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        File.WriteAllText(full, content);
    }

    [Fact]
    public void IncludesTreeAndKeyConfigFiles()
    {
        CreateFile("package.json", "{ \"name\": \"my-app\", \"scripts\": { \"build\": \"ng build\" } }");
        CreateFile("Api/Api.csproj", "<Project Sdk=\"Microsoft.NET.Sdk.Web\"></Project>");
        CreateFile("src/app/app.component.ts", "export class AppComponent {}");

        var description = ProjectOrientation.Describe(_root);

        // The tree reveals the layout...
        Assert.Contains("src/", description);
        Assert.Contains("app/", description);
        // ...and key config files are inlined so the model knows the stack.
        Assert.Contains("--- package.json ---", description);
        Assert.Contains("ng build", description);
        Assert.Contains("Microsoft.NET.Sdk.Web", description);
    }

    [Fact]
    public void OmitsClutterFolders()
    {
        CreateFile("node_modules/lib/package.json", "{ \"name\": \"dep\" }");
        CreateFile(".angular/cache/data.json", "{}");
        CreateFile("src/main.ts", "boot();");

        var description = ProjectOrientation.Describe(_root);

        Assert.Contains("main.ts", description);
        Assert.DoesNotContain("node_modules", description);
        Assert.DoesNotContain(".angular", description);
    }

    [Fact]
    public void EmptyFolder_ReturnsWithoutThrowing()
    {
        var description = ProjectOrientation.Describe(_root);

        Assert.Contains("Project layout", description);
    }

    [Fact]
    public void MissingFolder_ReturnsEmptyInsteadOfThrowing()
    {
        var description = ProjectOrientation.Describe(Path.Combine(_root, "does-not-exist"));

        Assert.Equal("", description);
    }
}
