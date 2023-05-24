// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package workspace

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/testenv"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	Main(m, hooks.Options)
}

const workspaceProxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

import "fmt"

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
example.com v1.2.3 h1:veRD4tUnatQRgsULqULZPjeoBGFr2qBhevSCZllD2Ds=
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
			WithOptions(opts...).Run(t, workspaceModule, func(t *testing.T, env *Env) {
				f := "pkg/inner/inner.go"
				env.OpenFile(f)
				locations := env.References(env.RegexpSearch(f, `SaySomething`))
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
	WithOptions(
		ProxyFiles(workspaceProxy),
		WorkspaceFolders("pkg/inner"),
	).Run(t, workspaceModule, func(t *testing.T, env *Env) {
		env.OpenFile("pkg/main.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("pkg/main2.go", "fmt.Print")),
		)
		env.CloseBuffer("pkg/main.go")
		env.AfterChange(
			NoDiagnostics(ForFile("pkg/main2.go")),
		)
	})
}

// TestReloadOnlyOnce checks that changes to the go.mod file do not result in
// redundant package loads (golang/go#54473).
//
// Note that this test may be fragile, as it depends on specific structure to
// log messages around reinitialization. Nevertheless, it is important for
// guarding against accidentally duplicate reloading.
func TestReloadOnlyOnce(t *testing.T) {
	WithOptions(
		ProxyFiles(workspaceProxy),
		WorkspaceFolders("pkg"),
	).Run(t, workspaceModule, func(t *testing.T, env *Env) {
		dir := env.Sandbox.Workdir.URI("goodbye").SpanURI().Filename()
		goModWithReplace := fmt.Sprintf(`%s
replace random.org => %s
`, env.ReadWorkspaceFile("pkg/go.mod"), dir)
		env.WriteWorkspaceFile("pkg/go.mod", goModWithReplace)
		env.AfterChange(
			LogMatching(protocol.Info, `packages\.Load #\d+\n`, 2, false),
		)
	})
}

const workspaceModuleProxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

import "fmt"

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
	testenv.NeedsGo1Point(t, 18) // uses go.work
	const multiModule = `
-- moda/a/go.mod --
module a.com

require b.com v1.2.3
-- moda/a/go.sum --
b.com v1.2.3 h1:tXrlXP0rnjRpKNmkbLYoWBdq0ikb3C3bKK9//moAWBI=
b.com v1.2.3/go.mod h1:D+J7pfFBZK5vdIdZEFquR586vKKIkqG7Qjw9AxG5BQ8=
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
	WithOptions(
		ProxyFiles(workspaceModuleProxy),
	).Run(t, multiModule, func(t *testing.T, env *Env) {
		env.RunGoCommand("work", "init")
		env.RunGoCommand("work", "use", "-r", ".")
		env.AfterChange(
			Diagnostics(env.AtRegexp("moda/a/a.go", "x")),
			Diagnostics(env.AtRegexp("modb/b/b.go", "x")),
			NoDiagnostics(env.AtRegexp("moda/a/a.go", `"b.com/b"`)),
		)
	})
}

func TestModuleWithExclude(t *testing.T) {
	const proxy = `
-- c.com@v1.2.3/go.mod --
module c.com

go 1.12

require b.com v1.2.3
-- c.com@v1.2.3/blah/blah.go --
package blah

import "fmt"

func SaySomething() {
	fmt.Println("something")
}
-- b.com@v1.2.3/go.mod --
module b.com

go 1.12
-- b.com@v1.2.4/b/b.go --
package b

func Hello() {}
-- b.com@v1.2.4/go.mod --
module b.com

go 1.12
`
	const multiModule = `
-- go.mod --
module a.com

require c.com v1.2.3

exclude b.com v1.2.3
-- go.sum --
c.com v1.2.3 h1:n07Dz9fYmpNqvZMwZi5NEqFcSHbvLa9lacMX+/g25tw=
c.com v1.2.3/go.mod h1:/4TyYgU9Nu5tA4NymP5xyqE8R2VMzGD3TbJCwCOvHAg=
-- main.go --
package a

func main() {
	var x int
}
`
	WithOptions(
		ProxyFiles(proxy),
	).Run(t, multiModule, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			Diagnostics(env.AtRegexp("main.go", "x")),
		)
	})
}

// This change tests that the version of the module used changes after it has
// been deleted from the workspace.
//
// TODO(golang/go#55331): delete this placeholder along with experimental
// workspace module.
func TestDeleteModule_Interdependent(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // uses go.work
	const multiModule = `
-- go.work --
go 1.18

use (
	moda/a
	modb
)
-- moda/a/go.mod --
module a.com

require b.com v1.2.3
-- moda/a/go.sum --
b.com v1.2.3 h1:tXrlXP0rnjRpKNmkbLYoWBdq0ikb3C3bKK9//moAWBI=
b.com v1.2.3/go.mod h1:D+J7pfFBZK5vdIdZEFquR586vKKIkqG7Qjw9AxG5BQ8=
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
	WithOptions(
		ProxyFiles(workspaceModuleProxy),
	).Run(t, multiModule, func(t *testing.T, env *Env) {
		env.OpenFile("moda/a/a.go")
		env.Await(env.DoneWithOpen())

		originalLoc := env.GoToDefinition(env.RegexpSearch("moda/a/a.go", "Hello"))
		original := env.Sandbox.Workdir.URIToPath(originalLoc.URI)
		if want := "modb/b/b.go"; !strings.HasSuffix(original, want) {
			t.Errorf("expected %s, got %v", want, original)
		}
		env.CloseBuffer(original)
		env.AfterChange()

		env.RemoveWorkspaceFile("modb/b/b.go")
		env.RemoveWorkspaceFile("modb/go.mod")
		env.WriteWorkspaceFile("go.work", "go 1.18\nuse moda/a")
		env.AfterChange()

		gotLoc := env.GoToDefinition(env.RegexpSearch("moda/a/a.go", "Hello"))
		got := env.Sandbox.Workdir.URIToPath(gotLoc.URI)
		if want := "b.com@v1.2.3/b/b.go"; !strings.HasSuffix(got, want) {
			t.Errorf("expected %s, got %v", want, got)
		}
	})
}

// Tests that the version of the module used changes after it has been added
// to the workspace.
func TestCreateModule_Interdependent(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // uses go.work
	const multiModule = `
-- go.work --
go 1.18

use (
	moda/a
)
-- moda/a/go.mod --
module a.com

require b.com v1.2.3
-- moda/a/go.sum --
b.com v1.2.3 h1:tXrlXP0rnjRpKNmkbLYoWBdq0ikb3C3bKK9//moAWBI=
b.com v1.2.3/go.mod h1:D+J7pfFBZK5vdIdZEFquR586vKKIkqG7Qjw9AxG5BQ8=
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
	WithOptions(
		ProxyFiles(workspaceModuleProxy),
	).Run(t, multiModule, func(t *testing.T, env *Env) {
		env.OpenFile("moda/a/a.go")
		loc := env.GoToDefinition(env.RegexpSearch("moda/a/a.go", "Hello"))
		original := env.Sandbox.Workdir.URIToPath(loc.URI)
		if want := "b.com@v1.2.3/b/b.go"; !strings.HasSuffix(original, want) {
			t.Errorf("expected %s, got %v", want, original)
		}
		env.CloseBuffer(original)
		env.WriteWorkspaceFiles(map[string]string{
			"go.work": `go 1.18

use (
	moda/a
	modb
)
`,
			"modb/go.mod": "module b.com",
			"modb/b/b.go": `package b

func Hello() int {
	var x int
}
`,
		})
		env.AfterChange(Diagnostics(env.AtRegexp("modb/b/b.go", "x")))
		gotLoc := env.GoToDefinition(env.RegexpSearch("moda/a/a.go", "Hello"))
		got := env.Sandbox.Workdir.URIToPath(gotLoc.URI)
		if want := "modb/b/b.go"; !strings.HasSuffix(got, want) {
			t.Errorf("expected %s, got %v", want, original)
		}
	})
}

// This test confirms that a gopls workspace can recover from initialization
// with one invalid module.
func TestOneBrokenModule(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // uses go.work
	const multiModule = `
-- go.work --
go 1.18

use (
	moda/a
	modb
)
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
	WithOptions(
		ProxyFiles(workspaceModuleProxy),
	).Run(t, multiModule, func(t *testing.T, env *Env) {
		env.OpenFile("modb/go.mod")
		env.AfterChange(
			Diagnostics(AtPosition("modb/go.mod", 0, 0)),
		)
		env.RegexpReplace("modb/go.mod", "modul", "module")
		env.SaveBufferWithoutActions("modb/go.mod")
		env.AfterChange(
			Diagnostics(env.AtRegexp("modb/b/b.go", "x")),
		)
	})
}

// TestBadGoWork exercises the panic from golang/vscode-go#2121.
func TestBadGoWork(t *testing.T) {
	const files = `
-- go.work --
use ./bar
-- bar/go.mod --
module example.com/bar
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("go.work")
	})
}

func TestUseGoWork(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // uses go.work
	// This test validates certain functionality related to using a go.work
	// file to specify workspace modules.
	const multiModule = `
-- moda/a/go.mod --
module a.com

require b.com v1.2.3
-- moda/a/go.sum --
b.com v1.2.3 h1:tXrlXP0rnjRpKNmkbLYoWBdq0ikb3C3bKK9//moAWBI=
b.com v1.2.3/go.mod h1:D+J7pfFBZK5vdIdZEFquR586vKKIkqG7Qjw9AxG5BQ8=
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
-- modb/go.sum --
example.com v1.2.3 h1:Yryq11hF02fEf2JlOS2eph+ICE2/ceevGV3C9dl5V/c=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- modb/b/b.go --
package b

func Hello() int {
	var x int
}
-- go.work --
go 1.17

use (
	./moda/a
)
`
	WithOptions(
		ProxyFiles(workspaceModuleProxy),
		Settings{
			"subdirWatchPatterns": "on",
		},
	).Run(t, multiModule, func(t *testing.T, env *Env) {
		// Initially, the go.work should cause only the a.com module to be loaded,
		// so we shouldn't get any file watches for modb. Further validate this by
		// jumping to a definition in b.com and ensuring that we go to the module
		// cache.
		env.OnceMet(
			InitialWorkspaceLoad,
			NoFileWatchMatching("modb"),
		)
		env.OpenFile("moda/a/a.go")
		env.Await(env.DoneWithOpen())

		// To verify which modules are loaded, we'll jump to the definition of
		// b.Hello.
		checkHelloLocation := func(want string) error {
			loc := env.GoToDefinition(env.RegexpSearch("moda/a/a.go", "Hello"))
			file := env.Sandbox.Workdir.URIToPath(loc.URI)
			if !strings.HasSuffix(file, want) {
				return fmt.Errorf("expected %s, got %v", want, file)
			}
			return nil
		}

		// Initially this should be in the module cache, as b.com is not replaced.
		if err := checkHelloLocation("b.com@v1.2.3/b/b.go"); err != nil {
			t.Fatal(err)
		}

		// Now, modify the go.work file on disk to activate the b.com module in
		// the workspace.
		env.WriteWorkspaceFile("go.work", `
go 1.17

use (
	./moda/a
	./modb
)
`)

		// As of golang/go#54069, writing go.work to the workspace triggers a
		// workspace reload, and new file watches.
		env.AfterChange(
			Diagnostics(env.AtRegexp("modb/b/b.go", "x")),
			// TODO(golang/go#60340): we don't get a file watch yet, because
			// updateWatchedDirectories runs before snapshot.load. Instead, we get it
			// after the next change (the didOpen below).
			// FileWatchMatching("modb"),
		)

		// Jumping to definition should now go to b.com in the workspace.
		if err := checkHelloLocation("modb/b/b.go"); err != nil {
			t.Fatal(err)
		}

		// Now, let's modify the go.work *overlay* (not on disk), and verify that
		// this change is only picked up once it is saved.
		env.OpenFile("go.work")
		env.AfterChange(
			// TODO(golang/go#60340): delete this expectation in favor of
			// the commented-out expectation above, once we fix the evaluation order
			// of file watches. We should not have to wait for a second change to get
			// the correct watches.
			FileWatchMatching("modb"),
		)
		env.SetBufferContent("go.work", `go 1.17

use (
	./moda/a
)`)

		// Simply modifying the go.work file does not cause a reload, so we should
		// still jump within the workspace.
		//
		// TODO: should editing the go.work above cause modb diagnostics to be
		// suppressed?
		env.Await(env.DoneWithChange())
		if err := checkHelloLocation("modb/b/b.go"); err != nil {
			t.Fatal(err)
		}

		// Saving should reload the workspace.
		env.SaveBufferWithoutActions("go.work")
		if err := checkHelloLocation("b.com@v1.2.3/b/b.go"); err != nil {
			t.Fatal(err)
		}

		// This fails if guarded with a OnceMet(DoneWithSave(), ...), because it is
		// delayed (and therefore not synchronous with the change).
		env.Await(NoDiagnostics(ForFile("modb/go.mod")))

		// Test Formatting.
		env.SetBufferContent("go.work", `go 1.18
  use      (



		./moda/a
)
`) // TODO(matloob): For some reason there's a "start position 7:0 is out of bounds" error when the ")" is on the last character/line in the file. Rob probably knows what's going on.
		env.SaveBuffer("go.work")
		env.Await(env.DoneWithSave())
		gotWorkContents := env.ReadWorkspaceFile("go.work")
		wantWorkContents := `go 1.18

use (
	./moda/a
)
`
		if gotWorkContents != wantWorkContents {
			t.Fatalf("formatted contents of workspace: got %q; want %q", gotWorkContents, wantWorkContents)
		}
	})
}

func TestUseGoWorkDiagnosticMissingModule(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // uses go.work

	const files = `
-- go.work --
go 1.18

use ./foo
-- bar/go.mod --
module example.com/bar
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("go.work")
		env.AfterChange(
			Diagnostics(env.AtRegexp("go.work", "use"), WithMessage("directory ./foo does not contain a module")),
		)
		// The following tests is a regression test against an issue where we weren't
		// copying the workFile struct field on workspace when a new one was created in
		// (*workspace).invalidate. Set the buffer content to a working file so that
		// invalidate recognizes the workspace to be change and copies over the workspace
		// struct, and then set the content back to the old contents to make sure
		// the diagnostic still shows up.
		env.SetBufferContent("go.work", "go 1.18 \n\n use ./bar\n")
		env.AfterChange(
			NoDiagnostics(env.AtRegexp("go.work", "use")),
		)
		env.SetBufferContent("go.work", "go 1.18 \n\n use ./foo\n")
		env.AfterChange(
			Diagnostics(env.AtRegexp("go.work", "use"), WithMessage("directory ./foo does not contain a module")),
		)
	})
}

func TestUseGoWorkDiagnosticSyntaxError(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	const files = `
-- go.work --
go 1.18

usa ./foo
replace
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("go.work")
		env.AfterChange(
			Diagnostics(env.AtRegexp("go.work", "usa"), WithMessage("unknown directive: usa")),
			Diagnostics(env.AtRegexp("go.work", "replace"), WithMessage("usage: replace")),
		)
	})
}

func TestUseGoWorkHover(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)

	const files = `
-- go.work --
go 1.18

use ./foo
use (
	./bar
	./bar/baz
)
-- foo/go.mod --
module example.com/foo
-- bar/go.mod --
module example.com/bar
-- bar/baz/go.mod --
module example.com/bar/baz
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("go.work")

		tcs := map[string]string{
			`\./foo`:      "example.com/foo",
			`(?m)\./bar$`: "example.com/bar",
			`\./bar/baz`:  "example.com/bar/baz",
		}

		for hoverRE, want := range tcs {
			got, _ := env.Hover(env.RegexpSearch("go.work", hoverRE))
			if got.Value != want {
				t.Errorf(`hover on %q: got %q, want %q`, hoverRE, got, want)
			}
		}
	})
}

func TestExpandToGoWork(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	const workspace = `
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
-- go.work --
go 1.17

use (
	./moda/a
	./modb
)
`
	WithOptions(
		WorkspaceFolders("moda/a"),
	).Run(t, workspace, func(t *testing.T, env *Env) {
		env.OpenFile("moda/a/a.go")
		env.Await(env.DoneWithOpen())
		loc := env.GoToDefinition(env.RegexpSearch("moda/a/a.go", "Hello"))
		file := env.Sandbox.Workdir.URIToPath(loc.URI)
		want := "modb/b/b.go"
		if !strings.HasSuffix(file, want) {
			t.Errorf("expected %s, got %v", want, file)
		}
	})
}

func TestNonWorkspaceFileCreation(t *testing.T) {
	const files = `
-- work/go.mod --
module mod.com

go 1.12
-- work/x.go --
package x
`

	const code = `
package foo
import "fmt"
var _ = fmt.Printf
`
	WithOptions(
		WorkspaceFolders("work"), // so that outside/... is outside the workspace
	).Run(t, files, func(t *testing.T, env *Env) {
		env.CreateBuffer("outside/foo.go", "")
		env.EditBuffer("outside/foo.go", fake.NewEdit(0, 0, 0, 0, code))
		env.GoToDefinition(env.RegexpSearch("outside/foo.go", `Printf`))
	})
}

func TestGoWork_V2Module(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // uses go.work
	// When using a go.work, we must have proxy content even if it is replaced.
	const proxy = `
-- b.com/v2@v2.1.9/go.mod --
module b.com/v2

go 1.12
-- b.com/v2@v2.1.9/b/b.go --
package b

func Ciao()() int {
	return 0
}
`

	const multiModule = `
-- go.work --
go 1.18

use (
	moda/a
	modb
	modb/v2
	modc
)
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

	WithOptions(
		ProxyFiles(proxy),
	).Run(t, multiModule, func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			// TODO(rfindley): assert on the full set of diagnostics here. We
			// should ensure that we don't have a diagnostic at b.Hi in a.go.
			Diagnostics(env.AtRegexp("moda/a/a.go", "x")),
			Diagnostics(env.AtRegexp("modb/b/b.go", "x")),
			Diagnostics(env.AtRegexp("modb/v2/b/b.go", "x")),
			Diagnostics(env.AtRegexp("modc/main.go", "x")),
		)
	})
}

// Confirm that a fix for a tidy module will correct all modules in the
// workspace.
func TestMultiModule_OneBrokenModule(t *testing.T) {
	// In the earlier 'experimental workspace mode', gopls would aggregate go.sum
	// entries for the workspace module, allowing it to correctly associate
	// missing go.sum with diagnostics. With go.work files, this doesn't work:
	// the go.command will happily write go.work.sum.
	t.Skip("golang/go#57509: go.mod diagnostics do not work in go.work mode")
	testenv.NeedsGo1Point(t, 18) // uses go.work
	const files = `
-- go.work --
go 1.18

use (
	a
	b
)
-- go.work.sum --
-- a/go.mod --
module a.com

go 1.12
-- a/main.go --
package main
-- b/go.mod --
module b.com

go 1.12

require (
	example.com v1.2.3
)
-- b/go.sum --
-- b/main.go --
package b

import "example.com/blah"

func main() {
	blah.Hello()
}
`
	WithOptions(
		ProxyFiles(workspaceProxy),
	).Run(t, files, func(t *testing.T, env *Env) {
		params := &protocol.PublishDiagnosticsParams{}
		env.OpenFile("b/go.mod")
		env.AfterChange(
			Diagnostics(
				env.AtRegexp("go.mod", `example.com v1.2.3`),
				WithMessage("go.sum is out of sync"),
			),
			ReadDiagnostics("b/go.mod", params),
		)
		for _, d := range params.Diagnostics {
			if !strings.Contains(d.Message, "go.sum is out of sync") {
				continue
			}
			actions := env.GetQuickFixes("b/go.mod", []protocol.Diagnostic{d})
			if len(actions) != 2 {
				t.Fatalf("expected 2 code actions, got %v", len(actions))
			}
			env.ApplyQuickFixes("b/go.mod", []protocol.Diagnostic{d})
		}
		env.AfterChange(
			NoDiagnostics(ForFile("b/go.mod")),
		)
	})
}

// Sometimes users may have their module cache within the workspace.
// We shouldn't consider any module in the module cache to be in the workspace.
func TestGOMODCACHEInWorkspace(t *testing.T) {
	const mod = `
-- a/go.mod --
module a.com

go 1.12
-- a/a.go --
package a

func _() {}
-- a/c/c.go --
package c
-- gopath/src/b/b.go --
package b
-- gopath/pkg/mod/example.com/go.mod --
module example.com

go 1.12
-- gopath/pkg/mod/example.com/main.go --
package main
`
	WithOptions(
		EnvVars{"GOPATH": filepath.FromSlash("$SANDBOX_WORKDIR/gopath")},
		Modes(Default),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.Await(
			// Confirm that the build configuration is seen as valid,
			// even though there are technically multiple go.mod files in the
			// worskpace.
			LogMatching(protocol.Info, ".*valid build configuration = true.*", 1, false),
		)
	})
}

func TestAddAndRemoveGoWork(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	// Use a workspace with a module in the root directory to exercise the case
	// where a go.work is added to the existing root directory. This verifies
	// that we're detecting changes to the module source, not just the root
	// directory.
	const nomod = `
-- go.mod --
module a.com

go 1.16
-- main.go --
package main

func main() {}
-- b/go.mod --
module b.com

go 1.16
-- b/main.go --
package main

func main() {}
`
	WithOptions(
		Modes(Default),
	).Run(t, nomod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OpenFile("b/main.go")
		// Since b/main.go is not in the workspace, it should have a warning on its
		// package declaration.
		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
			Diagnostics(env.AtRegexp("b/main.go", "package (main)")),
		)
		env.WriteWorkspaceFile("go.work", `go 1.16

use (
	.
	b
)
`)
		env.AfterChange(NoDiagnostics())
		// Removing the go.work file should put us back where we started.
		env.RemoveWorkspaceFile("go.work")

		// TODO(golang/go#57558, golang/go#57508): file watching is asynchronous,
		// and we must wait for the view to be reconstructed before touching
		// b/main.go, so that the new view "knows" about b/main.go. This is simply
		// a bug, but awaiting the change here avoids it.
		env.Await(env.DoneWithChangeWatchedFiles())

		// TODO(rfindley): fix this bug: reopening b/main.go is necessary here
		// because we no longer "see" the file in any view.
		env.CloseBuffer("b/main.go")
		env.OpenFile("b/main.go")

		env.AfterChange(
			NoDiagnostics(ForFile("main.go")),
			Diagnostics(env.AtRegexp("b/main.go", "package (main)")),
		)
	})
}

// Tests the fix for golang/go#52500.
func TestChangeTestVariant_Issue52500(t *testing.T) {
	const src = `
-- go.mod --
module mod.test

go 1.12
-- main_test.go --
package main_test

type Server struct{}

const mainConst = otherConst
-- other_test.go --
package main_test

const otherConst = 0

func (Server) Foo() {}
`

	Run(t, src, func(t *testing.T, env *Env) {
		env.OpenFile("other_test.go")
		env.RegexpReplace("other_test.go", "main_test", "main")

		// For this test to function, it is necessary to wait on both of the
		// expectations below: the bug is that when switching the package name in
		// other_test.go from main->main_test, metadata for main_test is not marked
		// as invalid. So we need to wait for the metadata of main_test.go to be
		// updated before moving other_test.go back to the main_test package.
		env.Await(
			Diagnostics(env.AtRegexp("other_test.go", "Server")),
			Diagnostics(env.AtRegexp("main_test.go", "otherConst")),
		)
		env.RegexpReplace("other_test.go", "main", "main_test")
		env.AfterChange(
			NoDiagnostics(ForFile("other_test.go")),
			NoDiagnostics(ForFile("main_test.go")),
		)

		// This will cause a test failure if other_test.go is not in any package.
		_ = env.GoToDefinition(env.RegexpSearch("other_test.go", "Server"))
	})
}

// Test for golang/go#48929.
func TestClearNonWorkspaceDiagnostics(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // uses go.work

	const ws = `
-- go.work --
go 1.18

use (
        ./b
)
-- a/go.mod --
module a

go 1.17
-- a/main.go --
package main

func main() {
   var V string
}
-- b/go.mod --
module b

go 1.17
-- b/main.go --
package b

import (
        _ "fmt"
)
`
	Run(t, ws, func(t *testing.T, env *Env) {
		env.OpenFile("b/main.go")
		env.AfterChange(
			NoDiagnostics(ForFile("a/main.go")),
		)
		env.OpenFile("a/main.go")
		env.AfterChange(
			Diagnostics(env.AtRegexp("a/main.go", "V"), WithMessage("not used")),
		)
		env.CloseBuffer("a/main.go")

		// Make an arbitrary edit because gopls explicitly diagnoses a/main.go
		// whenever it is "changed".
		//
		// TODO(rfindley): it should not be necessary to make another edit here.
		// Gopls should be smart enough to avoid diagnosing a.
		env.RegexpReplace("b/main.go", "package b", "package b // a package")
		env.AfterChange(
			NoDiagnostics(ForFile("a/main.go")),
		)
	})
}

// Test that we don't get a version warning when the Go version in PATH is
// supported.
func TestOldGoNotification_SupportedVersion(t *testing.T) {
	v := goVersion(t)
	if v < lsp.OldestSupportedGoVersion() {
		t.Skipf("go version 1.%d is unsupported", v)
	}

	Run(t, "", func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			NoShownMessage("upgrade"),
		)
	})
}

// Test that we do get a version warning when the Go version in PATH is
// unsupported, though this test may never execute if we stop running CI at
// legacy Go versions (see also TestOldGoNotification_Fake)
func TestOldGoNotification_UnsupportedVersion(t *testing.T) {
	v := goVersion(t)
	if v >= lsp.OldestSupportedGoVersion() {
		t.Skipf("go version 1.%d is supported", v)
	}

	Run(t, "", func(t *testing.T, env *Env) {
		env.Await(
			// Note: cannot use OnceMet(InitialWorkspaceLoad, ...) here, as the
			// upgrade message may race with the IWL.
			ShownMessage("Please upgrade"),
		)
	})
}

func TestOldGoNotification_Fake(t *testing.T) {
	// Get the Go version from path, and make sure it's unsupported.
	//
	// In the future we'll stop running CI on legacy Go versions. By mutating the
	// oldest supported Go version here, we can at least ensure that the
	// ShowMessage pop-up works.
	ctx := context.Background()
	goversion, err := gocommand.GoVersion(ctx, gocommand.Invocation{}, &gocommand.Runner{})
	if err != nil {
		t.Fatal(err)
	}
	defer func(t []lsp.GoVersionSupport) {
		lsp.GoVersionTable = t
	}(lsp.GoVersionTable)
	lsp.GoVersionTable = []lsp.GoVersionSupport{
		{GoVersion: goversion, InstallGoplsVersion: "v1.0.0"},
	}

	Run(t, "", func(t *testing.T, env *Env) {
		env.Await(
			// Note: cannot use OnceMet(InitialWorkspaceLoad, ...) here, as the
			// upgrade message may race with the IWL.
			ShownMessage("Please upgrade"),
		)
	})
}

// goVersion returns the version of the Go command in PATH.
func goVersion(t *testing.T) int {
	t.Helper()
	ctx := context.Background()
	goversion, err := gocommand.GoVersion(ctx, gocommand.Invocation{}, &gocommand.Runner{})
	if err != nil {
		t.Fatal(err)
	}
	return goversion
}
