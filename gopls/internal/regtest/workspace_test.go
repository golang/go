// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/testenv"
)

const workspaceProxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

func SaySomething() {
	fmt.Println("something")
}
-- random.org@v1.2.3/go.mod --
module random.org

go 1.12
-- random.org@v1.2.3/bye/bye.go --
package bye

func Goodbye() {
	println("Bye")
}
`

// TODO: Add a replace directive.
const workspaceModule = `
-- pkg/go.mod --
module mod.com

go 1.14

require (
	example.com v1.2.3
	random.org v1.2.3
)
-- pkg/go.sum --
example.com v1.2.3 h1:Yryq11hF02fEf2JlOS2eph+ICE2/ceevGV3C9dl5V/c=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
random.org v1.2.3 h1:+JE2Fkp7gS0zsHXGEQJ7hraom3pNTlkxC4b2qPfA+/Q=
random.org v1.2.3/go.mod h1:E9KM6+bBX2g5ykHZ9H27w16sWo3QwgonyjM44Dnej3I=
-- pkg/main.go --
package main

import (
	"example.com/blah"
	"mod.com/inner"
	"random.org/bye"
)

func main() {
	blah.SaySomething()
	inner.Hi()
	bye.Goodbye()
}
-- pkg/main2.go --
package main

import "fmt"

func _() {
	fmt.Print("%s")
}
-- pkg/inner/inner.go --
package inner

import "example.com/blah"

func Hi() {
	blah.SaySomething()
}
-- goodbye/bye/bye.go --
package bye

func Bye() {}
-- goodbye/go.mod --
module random.org

go 1.12
`

// Confirm that find references returns all of the references in the module,
// regardless of what the workspace root is.
func TestReferences(t *testing.T) {
	for _, tt := range []struct {
		name, rootPath string
	}{
		{
			name:     "module root",
			rootPath: "pkg",
		},
		{
			name:     "subdirectory",
			rootPath: "pkg/inner",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			opts := []RunOption{ProxyFiles(workspaceProxy)}
			if tt.rootPath != "" {
				opts = append(opts, WorkspaceFolders(tt.rootPath))
			}
			withOptions(opts...).run(t, workspaceModule, func(t *testing.T, env *Env) {
				f := "pkg/inner/inner.go"
				env.OpenFile(f)
				locations := env.References(f, env.RegexpSearch(f, `SaySomething`))
				want := 3
				if got := len(locations); got != want {
					t.Fatalf("expected %v locations, got %v", want, got)
				}
			})
		})
	}
}

// Make sure that analysis diagnostics are cleared for the whole package when
// the only opened file is closed. This test was inspired by the experience in
// VS Code, where clicking on a reference result triggers a
// textDocument/didOpen without a corresponding textDocument/didClose.
func TestClearAnalysisDiagnostics(t *testing.T) {
	withOptions(
		ProxyFiles(workspaceProxy),
		WorkspaceFolders("pkg/inner"),
	).run(t, workspaceModule, func(t *testing.T, env *Env) {
		env.OpenFile("pkg/main.go")
		env.Await(
			env.DiagnosticAtRegexp("pkg/main2.go", "fmt.Print"),
		)
		env.CloseBuffer("pkg/main.go")
		env.Await(
			EmptyDiagnostics("pkg/main2.go"),
		)
	})
}

// This test checks that gopls updates the set of files it watches when a
// replace target is added to the go.mod.
func TestWatchReplaceTargets(t *testing.T) {
	withOptions(
		ProxyFiles(workspaceProxy),
		WorkspaceFolders("pkg"),
	).run(t, workspaceModule, func(t *testing.T, env *Env) {
		// Add a replace directive and expect the files that gopls is watching
		// to change.
		dir := env.Sandbox.Workdir.URI("goodbye").SpanURI().Filename()
		goModWithReplace := fmt.Sprintf(`%s
replace random.org => %s
`, env.ReadWorkspaceFile("pkg/go.mod"), dir)
		env.WriteWorkspaceFile("pkg/go.mod", goModWithReplace)
		env.Await(
			CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1),
			UnregistrationMatching("didChangeWatchedFiles"),
			RegistrationMatching("didChangeWatchedFiles"),
		)
	})
}

const workspaceModuleProxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

func SaySomething() {
	fmt.Println("something")
}
-- b.com@v1.2.3/go.mod --
module b.com

go 1.12
-- b.com@v1.2.3/b/b.go --
package b

func Hello() {}
`

func TestAutomaticWorkspaceModule_Interdependent(t *testing.T) {
	const multiModule = `
-- moda/a/go.mod --
module a.com

require b.com v1.2.3

-- moda/a/a.go --
package a

import (
	"b.com/b"
)

func main() {
	var x int
	_ = b.Hello()
}
-- modb/go.mod --
module b.com

-- modb/b/b.go --
package b

func Hello() int {
	var x int
}
`
	withOptions(
		ProxyFiles(workspaceModuleProxy),
		Modes(Experimental),
	).run(t, multiModule, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexp("moda/a/a.go", "x"),
			env.DiagnosticAtRegexp("modb/b/b.go", "x"),
			env.NoDiagnosticAtRegexp("moda/a/a.go", `"b.com/b"`),
		)
	})
}

// This change tests that the version of the module used changes after it has
// been deleted from the workspace.
func TestDeleteModule_Interdependent(t *testing.T) {
	const multiModule = `
-- moda/a/go.mod --
module a.com

require b.com v1.2.3

-- moda/a/a.go --
package a

import (
	"b.com/b"
)

func main() {
	var x int
	_ = b.Hello()
}
-- modb/go.mod --
module b.com

-- modb/b/b.go --
package b

func Hello() int {
	var x int
}
`
	withOptions(
		ProxyFiles(workspaceModuleProxy),
		Modes(Experimental),
	).run(t, multiModule, func(t *testing.T, env *Env) {
		env.OpenFile("moda/a/a.go")

		original, _ := env.GoToDefinition("moda/a/a.go", env.RegexpSearch("moda/a/a.go", "Hello"))
		if want := "modb/b/b.go"; !strings.HasSuffix(original, want) {
			t.Errorf("expected %s, got %v", want, original)
		}
		env.CloseBuffer(original)
		env.RemoveWorkspaceFile("modb/b/b.go")
		env.RemoveWorkspaceFile("modb/go.mod")
		env.Await(
			CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 2),
		)
		if testenv.Go1Point() < 14 {
			// On 1.14 and above, the go mod tidy diagnostics accidentally
			// download for us. This is the behavior we actually want.
			d := protocol.PublishDiagnosticsParams{}
			env.Await(
				OnceMet(
					env.DiagnosticAtRegexpWithMessage("moda/a/go.mod", "require b.com v1.2.3", "b.com@v1.2.3"),
					ReadDiagnostics("moda/a/go.mod", &d),
				),
			)
			env.ApplyQuickFixes("moda/a/go.mod", d.Diagnostics)
		}
		got, _ := env.GoToDefinition("moda/a/a.go", env.RegexpSearch("moda/a/a.go", "Hello"))
		if want := "b.com@v1.2.3/b/b.go"; !strings.HasSuffix(got, want) {
			t.Errorf("expected %s, got %v", want, got)
		}
	})
}

// Tests that the version of the module used changes after it has been added
// to the workspace.
func TestCreateModule_Interdependent(t *testing.T) {
	const multiModule = `
-- moda/a/go.mod --
module a.com

require b.com v1.2.3

-- moda/a/a.go --
package a

import (
	"b.com/b"
)

func main() {
	var x int
	_ = b.Hello()
}
`
	withOptions(
		Modes(Experimental),
		ProxyFiles(workspaceModuleProxy),
	).run(t, multiModule, func(t *testing.T, env *Env) {
		env.OpenFile("moda/a/a.go")
		original, _ := env.GoToDefinition("moda/a/a.go", env.RegexpSearch("moda/a/a.go", "Hello"))
		if want := "b.com@v1.2.3/b/b.go"; !strings.HasSuffix(original, want) {
			t.Errorf("expected %s, got %v", want, original)
		}
		env.CloseBuffer(original)
		env.WriteWorkspaceFiles(map[string]string{
			"modb/go.mod": "module b.com",
			"modb/b/b.go": `package b

func Hello() int {
	var x int
}
`,
		})
		env.Await(
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1),
				env.DiagnosticAtRegexp("modb/b/b.go", "x"),
			),
		)
		got, _ := env.GoToDefinition("moda/a/a.go", env.RegexpSearch("moda/a/a.go", "Hello"))
		if want := "modb/b/b.go"; !strings.HasSuffix(got, want) {
			t.Errorf("expected %s, got %v", want, original)
		}
	})
}

// This test confirms that a gopls workspace can recover from initialization
// with one invalid module.
func TestOneBrokenModule(t *testing.T) {
	const multiModule = `
-- moda/a/go.mod --
module a.com

require b.com v1.2.3

-- moda/a/a.go --
package a

import (
	"b.com/b"
)

func main() {
	var x int
	_ = b.Hello()
}
-- modb/go.mod --
modul b.com // typo here

-- modb/b/b.go --
package b

func Hello() int {
	var x int
}
`
	withOptions(
		ProxyFiles(workspaceModuleProxy),
		Modes(Experimental),
	).run(t, multiModule, func(t *testing.T, env *Env) {
		env.OpenFile("modb/go.mod")
		env.Await(
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1),
				DiagnosticAt("modb/go.mod", 0, 0),
			),
		)
		env.RegexpReplace("modb/go.mod", "modul", "module")
		env.SaveBufferWithoutActions("modb/go.mod")
		env.Await(
			env.DiagnosticAtRegexp("modb/b/b.go", "x"),
		)
	})
}

func TestUseGoplsMod(t *testing.T) {
	// This test validates certain functionality related to using a gopls.mod
	// file to specify workspace modules.
	testenv.NeedsGo1Point(t, 14)
	const multiModule = `
-- moda/a/go.mod --
module a.com

require b.com v1.2.3

-- moda/a/a.go --
package a

import (
	"b.com/b"
)

func main() {
	var x int
	_ = b.Hello()
}
-- modb/go.mod --
module b.com

require example.com v1.2.3
-- modb/b/b.go --
package b

func Hello() int {
	var x int
}
-- gopls.mod --
module gopls-workspace

require (
	a.com v0.0.0-goplsworkspace
	b.com v1.2.3
)

replace a.com => $SANDBOX_WORKDIR/moda/a
`
	withOptions(
		ProxyFiles(workspaceModuleProxy),
		Modes(Experimental),
	).run(t, multiModule, func(t *testing.T, env *Env) {
		// Initially, the gopls.mod should cause only the a.com module to be
		// loaded. Validate this by jumping to a definition in b.com and ensuring
		// that we go to the module cache.
		env.OpenFile("moda/a/a.go")
		env.Await(env.DoneWithOpen())

		// To verify which modules are loaded, we'll jump to the definition of
		// b.Hello.
		checkHelloLocation := func(want string) error {
			location, _ := env.GoToDefinition("moda/a/a.go", env.RegexpSearch("moda/a/a.go", "Hello"))
			if !strings.HasSuffix(location, want) {
				return fmt.Errorf("expected %s, got %v", want, location)
			}
			return nil
		}

		// Initially this should be in the module cache, as b.com is not replaced.
		if err := checkHelloLocation("b.com@v1.2.3/b/b.go"); err != nil {
			t.Fatal(err)
		}

		// Now, modify the gopls.mod file on disk to activate the b.com module in
		// the workspace.
		workdir := env.Sandbox.Workdir.RootURI().SpanURI().Filename()
		env.WriteWorkspaceFile("gopls.mod", fmt.Sprintf(`module gopls-workspace

require (
	a.com v0.0.0-goplsworkspace
	b.com v0.0.0-goplsworkspace
)

replace a.com => %s/moda/a
replace b.com => %s/modb
`, workdir, workdir))
		env.Await(env.DoneWithChangeWatchedFiles())
		// Check that go.mod diagnostics picked up the newly active mod file.
		// The local version of modb has an extra dependency we need to download.
		env.OpenFile("modb/go.mod")
		env.Await(env.DoneWithOpen())

		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("modb/go.mod", `require example.com v1.2.3`),
				ReadDiagnostics("modb/go.mod", &d),
			),
		)
		env.ApplyQuickFixes("modb/go.mod", d.Diagnostics)
		env.Await(env.DiagnosticAtRegexp("modb/b/b.go", "x"))
		// Jumping to definition should now go to b.com in the workspace.
		if err := checkHelloLocation("modb/b/b.go"); err != nil {
			t.Fatal(err)
		}

		// Now, let's modify the gopls.mod *overlay* (not on disk), and verify that
		// this change is only picked up once it is saved.
		env.OpenFile("gopls.mod")
		env.Await(env.DoneWithOpen())
		env.SetBufferContent("gopls.mod", fmt.Sprintf(`module gopls-workspace

require (
	a.com v0.0.0-goplsworkspace
)

replace a.com => %s/moda/a
`, workdir))

		// Editing the gopls.mod removes modb from the workspace modules, and so
		// should clear outstanding diagnostics...
		env.Await(OnceMet(
			env.DoneWithChange(),
			EmptyDiagnostics("modb/go.mod"),
		))
		// ...but does not yet cause a workspace reload, so we should still jump to modb.
		if err := checkHelloLocation("modb/b/b.go"); err != nil {
			t.Fatal(err)
		}
		// Saving should reload the workspace.
		env.SaveBufferWithoutActions("gopls.mod")
		if err := checkHelloLocation("b.com@v1.2.3/b/b.go"); err != nil {
			t.Fatal(err)
		}
	})
}

func TestNonWorkspaceFileCreation(t *testing.T) {
	testenv.NeedsGo1Point(t, 13)

	const files = `
-- go.mod --
module mod.com

go 1.12
-- x.go --
package x
`

	const code = `
package foo
import "fmt"
var _ = fmt.Printf
`
	run(t, files, func(t *testing.T, env *Env) {
		env.CreateBuffer("/tmp/foo.go", "")
		env.EditBuffer("/tmp/foo.go", fake.NewEdit(0, 0, 0, 0, code))
		env.GoToDefinition("/tmp/foo.go", env.RegexpSearch("/tmp/foo.go", `Printf`))
	})
}

func TestMultiModuleV2(t *testing.T) {
	const multiModule = `
-- moda/a/go.mod --
module a.com

require b.com/v2 v2.1.9
-- moda/a/a.go --
package a

import (
	"b.com/v2/b"
)

func main() {
	var x int
	_ = b.Hi()
}
-- modb/go.mod --
module b.com

-- modb/b/b.go --
package b

func Hello() int {
	var x int
}
-- modb/v2/go.mod --
module b.com/v2

-- modb/v2/b/b.go --
package b

func Hi() int {
	var x int
}
-- modc/go.mod --
module gopkg.in/yaml.v1 // test gopkg.in versions
-- modc/main.go --
package main

func main() {
	var x int
}
`
	withOptions(
		Modes(Experimental),
	).run(t, multiModule, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexp("moda/a/a.go", "x"),
			env.DiagnosticAtRegexp("modb/b/b.go", "x"),
			env.DiagnosticAtRegexp("modb/v2/b/b.go", "x"),
			env.DiagnosticAtRegexp("modc/main.go", "x"),
		)
	})
}

func TestWorkspaceDirAccess(t *testing.T) {
	const multiModule = `
-- moda/a/go.mod --
module a.com

-- moda/a/a.go --
package main

func main() {
	fmt.Println("Hello")
}
-- modb/go.mod --
module b.com
-- modb/b/b.go --
package main

func main() {
	fmt.Println("World")
}
`
	withOptions(
		Modes(Experimental),
		SendPID(),
	).run(t, multiModule, func(t *testing.T, env *Env) {
		pid := os.Getpid()
		// Don't factor this out of Server.addFolders. vscode-go expects this
		// directory.
		modPath := filepath.Join(os.TempDir(), fmt.Sprintf("gopls-%d.workspace", pid), "go.mod")
		gotb, err := ioutil.ReadFile(modPath)
		if err != nil {
			t.Fatalf("reading expected workspace modfile: %v", err)
		}
		got := string(gotb)
		for _, want := range []string{"a.com v0.0.0-goplsworkspace", "b.com v0.0.0-goplsworkspace"} {
			if !strings.Contains(got, want) {
				// want before got here, since the go.mod is multi-line
				t.Fatalf("workspace go.mod missing %q. got:\n%s", want, got)
			}
		}
		workdir := env.Sandbox.Workdir.RootURI().SpanURI().Filename()
		env.WriteWorkspaceFile("gopls.mod", fmt.Sprintf(`
				module gopls-workspace

				require (
					a.com v0.0.0-goplsworkspace
				)

				replace a.com => %s/moda/a
				`, workdir))
		env.Await(CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1))
		gotb, err = ioutil.ReadFile(modPath)
		if err != nil {
			t.Fatalf("reading expected workspace modfile: %v", err)
		}
		got = string(gotb)
		want := "b.com v0.0.0-goplsworkspace"
		if strings.Contains(got, want) {
			t.Fatalf("workspace go.mod contains unexpected %q. got:\n%s", want, got)
		}
	})
}

func TestDirectoryFiltersLoads(t *testing.T) {
	// exclude, and its error, should be excluded from the workspace.
	const files = `
-- go.mod --
module example.com

go 1.12
-- exclude/exclude.go --
package exclude

const _ = Nonexistant
`
	cfg := EditorConfig{
		DirectoryFilters: []string{"-exclude"},
	}
	withOptions(cfg).run(t, files, func(t *testing.T, env *Env) {
		env.Await(NoDiagnostics("exclude/x.go"))
	})
}

func TestDirectoryFiltersTransitiveDep(t *testing.T) {
	// Even though exclude is excluded from the workspace, it should
	// still be importable as a non-workspace package.
	const files = `
-- go.mod --
module example.com

go 1.12
-- include/include.go --
package include
import "example.com/exclude"

const _ = exclude.X
-- exclude/exclude.go --
package exclude

const _ = Nonexistant // should be ignored, since this is a non-workspace package
const X = 1
`

	cfg := EditorConfig{
		DirectoryFilters: []string{"-exclude"},
	}
	withOptions(cfg).run(t, files, func(t *testing.T, env *Env) {
		env.Await(
			NoDiagnostics("exclude/exclude.go"), // filtered out
			NoDiagnostics("include/include.go"), // successfully builds
		)
	})
}

func TestDirectoryFiltersWorkspaceModules(t *testing.T) {
	// Define a module include.com which should be in the workspace, plus a
	// module exclude.com which should be excluded and therefore come from
	// the proxy.
	const files = `
-- include/go.mod --
module include.com

go 1.12

require exclude.com v1.0.0

-- include/go.sum --
exclude.com v1.0.0 h1:Q5QSfDXY5qyNCBeUiWovUGqcLCRZKoTs9XdBeVz+w1I=
exclude.com v1.0.0/go.mod h1:hFox2uDlNB2s2Jfd9tHlQVfgqUiLVTmh6ZKat4cvnj4=

-- include/include.go --
package include

import "exclude.com"

var _ = exclude.X // satisfied only by the workspace version
-- exclude/go.mod --
module exclude.com

go 1.12
-- exclude/exclude.go --
package exclude

const X = 1
`
	const proxy = `
-- exclude.com@v1.0.0/go.mod --
module exclude.com

go 1.12
-- exclude.com@v1.0.0/exclude.go --
package exclude
`
	cfg := EditorConfig{
		DirectoryFilters: []string{"-exclude"},
	}
	withOptions(cfg, Modes(Experimental), ProxyFiles(proxy)).run(t, files, func(t *testing.T, env *Env) {
		env.Await(env.DiagnosticAtRegexp("include/include.go", `exclude.(X)`))
	})
}
