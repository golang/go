// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfile

import (
	"path/filepath"
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/regtest"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	Main(m)
}

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

const proxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

const Name = "Blah"
-- random.org@v1.2.3/go.mod --
module random.org

go 1.12
-- random.org@v1.2.3/blah/blah.go --
package hello

const Name = "Hello"
`

func TestModFileModification(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const untidyModule = `
-- a/go.mod --
module mod.com

-- a/main.go --
package main

import "example.com/blah"

func main() {
	println(blah.Name)
}
`

	runner := RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}

	t.Run("basic", func(t *testing.T) {
		runner.Run(t, untidyModule, func(t *testing.T, env *Env) {
			// Open the file and make sure that the initial workspace load does not
			// modify the go.mod file.
			goModContent := env.ReadWorkspaceFile("a/go.mod")
			env.OpenFile("a/main.go")
			env.Await(
				env.DiagnosticAtRegexp("a/main.go", "\"example.com/blah\""),
			)
			if got := env.ReadWorkspaceFile("a/go.mod"); got != goModContent {
				t.Fatalf("go.mod changed on disk:\n%s", tests.Diff(t, goModContent, got))
			}
			// Save the buffer, which will format and organize imports.
			// Confirm that the go.mod file still does not change.
			env.SaveBuffer("a/main.go")
			env.Await(
				env.DiagnosticAtRegexp("a/main.go", "\"example.com/blah\""),
			)
			if got := env.ReadWorkspaceFile("a/go.mod"); got != goModContent {
				t.Fatalf("go.mod changed on disk:\n%s", tests.Diff(t, goModContent, got))
			}
		})
	})

	// Reproduce golang/go#40269 by deleting and recreating main.go.
	t.Run("delete main.go", func(t *testing.T) {
		t.Skip("This test will be flaky until golang/go#40269 is resolved.")

		runner.Run(t, untidyModule, func(t *testing.T, env *Env) {
			goModContent := env.ReadWorkspaceFile("a/go.mod")
			mainContent := env.ReadWorkspaceFile("a/main.go")
			env.OpenFile("a/main.go")
			env.SaveBuffer("a/main.go")

			env.RemoveWorkspaceFile("a/main.go")
			env.Await(
				env.DoneWithOpen(),
				env.DoneWithSave(),
				env.DoneWithChangeWatchedFiles(),
			)

			env.WriteWorkspaceFile("main.go", mainContent)
			env.Await(
				env.DiagnosticAtRegexp("main.go", "\"example.com/blah\""),
			)
			if got := env.ReadWorkspaceFile("go.mod"); got != goModContent {
				t.Fatalf("go.mod changed on disk:\n%s", tests.Diff(t, goModContent, got))
			}
		})
	})
}

func TestGoGetFix(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)
	const mod = `
-- a/go.mod --
module mod.com

go 1.12

-- a/main.go --
package main

import "example.com/blah"

var _ = blah.Name
`

	const want = `module mod.com

go 1.12

require example.com v1.2.3
`

	RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}.Run(t, mod, func(t *testing.T, env *Env) {
		if strings.Contains(t.Name(), "workspace_module") {
			t.Skip("workspace module mode doesn't set -mod=readonly")
		}
		env.OpenFile("a/main.go")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("a/main.go", `"example.com/blah"`),
				ReadDiagnostics("a/main.go", &d),
			),
		)
		var goGetDiag protocol.Diagnostic
		for _, diag := range d.Diagnostics {
			if strings.Contains(diag.Message, "could not import") {
				goGetDiag = diag
			}
		}
		env.ApplyQuickFixes("a/main.go", []protocol.Diagnostic{goGetDiag})
		if got := env.ReadWorkspaceFile("a/go.mod"); got != want {
			t.Fatalf("unexpected go.mod content:\n%s", tests.Diff(t, want, got))
		}
	})
}

// Tests that multiple missing dependencies gives good single fixes.
func TestMissingDependencyFixes(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)
	const mod = `
-- a/go.mod --
module mod.com

go 1.12

-- a/main.go --
package main

import "example.com/blah"
import "random.org/blah"

var _, _ = blah.Name, hello.Name
`

	const want = `module mod.com

go 1.12

require random.org v1.2.3
`

	RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}.Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("a/main.go")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("a/main.go", `"random.org/blah"`),
				ReadDiagnostics("a/main.go", &d),
			),
		)
		var randomDiag protocol.Diagnostic
		for _, diag := range d.Diagnostics {
			if strings.Contains(diag.Message, "random.org") {
				randomDiag = diag
			}
		}
		env.ApplyQuickFixes("a/main.go", []protocol.Diagnostic{randomDiag})
		if got := env.ReadWorkspaceFile("a/go.mod"); got != want {
			t.Fatalf("unexpected go.mod content:\n%s", tests.Diff(t, want, got))
		}
	})
}

func TestIndirectDependencyFix(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const mod = `
-- a/go.mod --
module mod.com

go 1.12

require example.com v1.2.3 // indirect
-- a/go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- a/main.go --
package main

import "example.com/blah"

func main() {
	fmt.Println(blah.Name)
`
	const want = `module mod.com

go 1.12

require example.com v1.2.3
`

	RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}.Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("a/go.mod")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("a/go.mod", "// indirect"),
				ReadDiagnostics("a/go.mod", &d),
			),
		)
		env.ApplyQuickFixes("a/go.mod", d.Diagnostics)
		if got := env.Editor.BufferText("a/go.mod"); got != want {
			t.Fatalf("unexpected go.mod content:\n%s", tests.Diff(t, want, got))
		}
	})
}

func TestUnusedDiag(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const proxy = `
-- example.com@v1.0.0/x.go --
package pkg
const X = 1
`
	const files = `
-- a/go.mod --
module mod.com
go 1.14
require example.com v1.0.0
-- a/go.sum --
example.com v1.0.0 h1:38O7j5rEBajXk+Q5wzLbRN7KqMkSgEiN9NqcM1O2bBM=
example.com v1.0.0/go.mod h1:vUsPMGpx9ZXXzECCOsOmYCW7npJTwuA16yl89n3Mgls=
-- a/main.go --
package main
func main() {}
`

	const want = `module mod.com

go 1.14
`

	RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}.Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/go.mod")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("a/go.mod", `require example.com`),
				ReadDiagnostics("a/go.mod", &d),
			),
		)
		env.ApplyQuickFixes("a/go.mod", d.Diagnostics)
		if got := env.ReadWorkspaceFile("a/go.mod"); got != want {
			t.Fatalf("unexpected go.mod content:\n%s", tests.Diff(t, want, got))
		}
	})
}

// Test to reproduce golang/go#39041. It adds a new require to a go.mod file
// that already has an unused require.
func TestNewDepWithUnusedDep(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const proxy = `
-- github.com/esimov/caire@v1.2.5/go.mod --
module github.com/esimov/caire

go 1.12
-- github.com/esimov/caire@v1.2.5/caire.go --
package caire

func RemoveTempImage() {}
-- google.golang.org/protobuf@v1.20.0/go.mod --
module google.golang.org/protobuf

go 1.12
-- google.golang.org/protobuf@v1.20.0/hello/hello.go --
package hello
`
	const repro = `
-- a/go.mod --
module mod.com

go 1.14

require google.golang.org/protobuf v1.20.0
-- a/go.sum --
github.com/esimov/caire v1.2.5 h1:OcqDII/BYxcBYj3DuwDKjd+ANhRxRqLa2n69EGje7qw=
github.com/esimov/caire v1.2.5/go.mod h1:mXnjRjg3+WUtuhfSC1rKRmdZU9vJZyS1ZWU0qSvJhK8=
google.golang.org/protobuf v1.20.0 h1:y9T1vAtFKQg0faFNMOxJU7WuEqPWolVkjIkU6aI8qCY=
google.golang.org/protobuf v1.20.0/go.mod h1:FcqsytGClbtLv1ot8NvsJHjBi0h22StKVP+K/j2liKA=
-- a/main.go --
package main

import (
    "github.com/esimov/caire"
)

func _() {
    caire.RemoveTempImage()
}`

	RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}.Run(t, repro, func(t *testing.T, env *Env) {
		env.OpenFile("a/main.go")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("a/main.go", `"github.com/esimov/caire"`),
				ReadDiagnostics("a/main.go", &d),
			),
		)
		env.ApplyQuickFixes("a/main.go", d.Diagnostics)
		want := `module mod.com

go 1.14

require (
	github.com/esimov/caire v1.2.5
	google.golang.org/protobuf v1.20.0
)
`
		if got := env.ReadWorkspaceFile("a/go.mod"); got != want {
			t.Fatalf("TestNewDepWithUnusedDep failed:\n%s", tests.Diff(t, want, got))
		}
	})
}

// TODO: For this test to be effective, the sandbox's file watcher must respect
// the file watching GlobPattern in the capability registration. See
// golang/go#39384.
func TestModuleChangesOnDisk(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const mod = `
-- a/go.mod --
module mod.com

go 1.12

require example.com v1.2.3
-- a/go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- a/main.go --
package main

func main() {
	fmt.Println(blah.Name)
`
	RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}.Run(t, mod, func(t *testing.T, env *Env) {
		env.Await(env.DiagnosticAtRegexp("a/go.mod", "require"))
		env.RunGoCommandInDir("a", "mod", "tidy")
		env.Await(
			EmptyDiagnostics("a/go.mod"),
		)
	})
}

// Tests golang/go#39784: a missing indirect dependency, necessary
// due to blah@v2.0.0's incomplete go.mod file.
func TestBadlyVersionedModule(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const proxy = `
-- example.com/blah/@v/v1.0.0.mod --
module example.com

go 1.12
-- example.com/blah@v1.0.0/blah.go --
package blah

const Name = "Blah"
-- example.com/blah/v2/@v/v2.0.0.mod --
module example.com

go 1.12
-- example.com/blah/v2@v2.0.0/blah.go --
package blah

import "example.com/blah"

var _ = blah.Name
const Name = "Blah"
`
	const files = `
-- a/go.mod --
module mod.com

go 1.12

require example.com/blah/v2 v2.0.0
-- a/go.sum --
example.com/blah v1.0.0 h1:kGPlWJbMsn1P31H9xp/q2mYI32cxLnCvauHN0AVaHnc=
example.com/blah v1.0.0/go.mod h1:PZUQaGFeVjyDmAE8ywmLbmDn3fj4Ws8epg4oLuDzW3M=
example.com/blah/v2 v2.0.0 h1:w5baE9JuuU11s3de3yWx2sU05AhNkgLYdZ4qukv+V0k=
example.com/blah/v2 v2.0.0/go.mod h1:UZiKbTwobERo/hrqFLvIQlJwQZQGxWMVY4xere8mj7w=
-- a/main.go --
package main

import "example.com/blah/v2"

var _ = blah.Name
`
	RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}.Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/main.go")
		env.OpenFile("a/go.mod")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				DiagnosticAt("a/go.mod", 0, 0),
				ReadDiagnostics("a/go.mod", &d),
			),
		)
		env.ApplyQuickFixes("a/main.go", d.Diagnostics)
		const want = `module mod.com

go 1.12

require (
	example.com/blah v1.0.0 // indirect
	example.com/blah/v2 v2.0.0
)
`
		env.Await(EmptyDiagnostics("a/go.mod"))
		if got := env.Editor.BufferText("a/go.mod"); got != want {
			t.Fatalf("suggested fixes failed:\n%s", tests.Diff(t, want, got))
		}
	})
}

// Reproduces golang/go#38232.
func TestUnknownRevision(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const unknown = `
-- a/go.mod --
module mod.com

require (
	example.com v1.2.2
)
-- a/main.go --
package main

import "example.com/blah"

func main() {
	var x = blah.Name
}
`

	runner := RunMultiple{
		{"default", WithOptions(ProxyFiles(proxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(proxy))},
	}
	// Start from a bad state/bad IWL, and confirm that we recover.
	t.Run("bad", func(t *testing.T) {
		runner.Run(t, unknown, func(t *testing.T, env *Env) {
			env.OpenFile("a/go.mod")
			env.Await(
				env.DiagnosticAtRegexp("a/go.mod", "example.com v1.2.2"),
			)
			env.RegexpReplace("a/go.mod", "v1.2.2", "v1.2.3")
			env.Editor.SaveBuffer(env.Ctx, "a/go.mod") // go.mod changes must be on disk

			d := protocol.PublishDiagnosticsParams{}
			env.Await(
				OnceMet(
					env.DiagnosticAtRegexpWithMessage("a/go.mod", "example.com v1.2.3", "example.com@v1.2.3"),
					ReadDiagnostics("a/go.mod", &d),
				),
			)
			env.ApplyQuickFixes("a/go.mod", d.Diagnostics)

			env.Await(
				EmptyDiagnostics("a/go.mod"),
				env.DiagnosticAtRegexp("a/main.go", "x = "),
			)
		})
	})

	const known = `
-- a/go.mod --
module mod.com

require (
	example.com v1.2.3
)
-- a/go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- a/main.go --
package main

import "example.com/blah"

func main() {
	var x = blah.Name
}
`
	// Start from a good state, transform to a bad state, and confirm that we
	// still recover.
	t.Run("good", func(t *testing.T) {
		runner.Run(t, known, func(t *testing.T, env *Env) {
			env.OpenFile("a/go.mod")
			env.Await(
				env.DiagnosticAtRegexp("a/main.go", "x = "),
			)
			env.RegexpReplace("a/go.mod", "v1.2.3", "v1.2.2")
			env.Editor.SaveBuffer(env.Ctx, "a/go.mod") // go.mod changes must be on disk
			env.Await(
				env.DiagnosticAtRegexp("a/go.mod", "example.com v1.2.2"),
			)
			env.RegexpReplace("a/go.mod", "v1.2.2", "v1.2.3")
			env.Editor.SaveBuffer(env.Ctx, "a/go.mod") // go.mod changes must be on disk
			env.Await(
				env.DiagnosticAtRegexp("a/main.go", "x = "),
			)
		})
	})
}

// Confirm that an error in an indirect dependency of a requirement is surfaced
// as a diagnostic in the go.mod file.
func TestErrorInIndirectDependency(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const badProxy = `
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12

require random.org v1.2.3 // indirect
-- example.com@v1.2.3/blah/blah.go --
package blah

const Name = "Blah"
-- random.org@v1.2.3/go.mod --
module bob.org

go 1.12
-- random.org@v1.2.3/blah/blah.go --
package hello

const Name = "Hello"
`
	const module = `
-- a/go.mod --
module mod.com

go 1.14

require example.com v1.2.3
-- a/main.go --
package main

import "example.com/blah"

func main() {
	println(blah.Name)
}
`
	RunMultiple{
		{"default", WithOptions(ProxyFiles(badProxy), WorkspaceFolders("a"))},
		{"nested", WithOptions(ProxyFiles(badProxy))},
	}.Run(t, module, func(t *testing.T, env *Env) {
		env.OpenFile("a/go.mod")
		env.Await(
			env.DiagnosticAtRegexp("a/go.mod", "require example.com v1.2.3"),
		)
	})
}

// A copy of govim's config_set_env_goflags_mod_readonly test.
func TestGovimModReadonly(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.13
-- main.go --
package main

import "example.com/blah"

func main() {
	println(blah.Name)
}
`
	WithOptions(
		EditorConfig{
			Env: map[string]string{
				"GOFLAGS": "-mod=readonly",
			},
		},
		ProxyFiles(proxy),
		Modes(Singleton),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		original := env.ReadWorkspaceFile("go.mod")
		env.Await(
			env.DiagnosticAtRegexp("main.go", `"example.com/blah"`),
		)
		got := env.ReadWorkspaceFile("go.mod")
		if got != original {
			t.Fatalf("go.mod file modified:\n%s", tests.Diff(t, original, got))
		}
		env.RunGoCommand("get", "example.com/blah@v1.2.3")
		env.RunGoCommand("mod", "tidy")
		env.Await(
			EmptyDiagnostics("main.go"),
		)
	})
}

func TestMultiModuleModDiagnostics(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const mod = `
-- a/go.mod --
module moda.com

go 1.14

require (
	example.com v1.2.3
)
-- a/go.sum --
example.com v1.2.3 h1:Yryq11hF02fEf2JlOS2eph+ICE2/ceevGV3C9dl5V/c=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- a/main.go --
package main

func main() {}
-- b/go.mod --
module modb.com

go 1.14
-- b/main.go --
package main

import "example.com/blah"

func main() {
	blah.SaySomething()
}
`
	WithOptions(
		ProxyFiles(workspaceProxy),
		Modes(Experimental),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexpWithMessage("a/go.mod", "example.com v1.2.3", "is not used"),
			env.DiagnosticAtRegexpWithMessage("b/go.mod", "module modb.com", "not in your go.mod file"),
		)
	})
}

func TestModTidyWithBuildTags(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const mod = `
-- go.mod --
module mod.com

go 1.14
-- main.go --
// +build bob

package main

import "example.com/blah"

func main() {
	blah.SaySomething()
}
`
	WithOptions(
		ProxyFiles(workspaceProxy),
		EditorConfig{
			BuildFlags: []string{"-tags", "bob"},
		},
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexp("main.go", `"example.com/blah"`),
		)
	})
}

func TestModTypoDiagnostic(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {}
`
	Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.RegexpReplace("go.mod", "module", "modul")
		env.Await(
			env.DiagnosticAtRegexp("go.mod", "modul"),
		)
	})
}

func TestSumUpdateFixesDiagnostics(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const mod = `
-- go.mod --
module mod.com

go 1.12

require (
	example.com v1.2.3
)
-- go.sum --
-- main.go --
package main

import (
	"example.com/blah"
)

func main() {
	println(blah.Name)
}
`
	WithOptions(
		Modes(Singleton), // workspace modules don't use -mod=readonly (golang/go#43346)
		ProxyFiles(workspaceProxy),
	).Run(t, mod, func(t *testing.T, env *Env) {
		d := &protocol.PublishDiagnosticsParams{}
		env.OpenFile("go.mod")
		env.Await(
			OnceMet(
				DiagnosticAt("go.mod", 0, 0),
				ReadDiagnostics("go.mod", d),
			),
		)
		env.ApplyQuickFixes("go.mod", d.Diagnostics)
		env.Await(
			EmptyDiagnostics("go.mod"),
		)
	})
}

// This test confirms that editing a go.mod file only causes metadata
// to be invalidated when it's saved.
func TestGoModInvalidatesOnSave(t *testing.T) {
	const mod = `
-- go.mod --
module mod.com

go 1.12
-- main.go --
package main

func main() {
	hello()
}
-- hello.go --
package main

func hello() {}
`
	WithOptions(
		// TODO(rFindley) this doesn't work in multi-module workspace mode, because
		// it keeps around the last parsing modfile. Update this test to also
		// exercise the workspace module.
		Modes(Singleton),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.Await(env.DoneWithOpen())
		env.RegexpReplace("go.mod", "module", "modul")
		// Confirm that we still have metadata with only on-disk edits.
		env.OpenFile("main.go")
		file, _ := env.GoToDefinition("main.go", env.RegexpSearch("main.go", "hello"))
		if filepath.Base(file) != "hello.go" {
			t.Fatalf("expected definition in hello.go, got %s", file)
		}
		// Confirm that we no longer have metadata when the file is saved.
		env.SaveBufferWithoutActions("go.mod")
		_, _, err := env.Editor.GoToDefinition(env.Ctx, "main.go", env.RegexpSearch("main.go", "hello"))
		if err == nil {
			t.Fatalf("expected error, got none")
		}
	})
}

func TestRemoveUnusedDependency(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const proxy = `
-- hasdep.com@v1.2.3/go.mod --
module hasdep.com

go 1.12

require example.com v1.2.3
-- hasdep.com@v1.2.3/a/a.go --
package a
-- example.com@v1.2.3/go.mod --
module example.com

go 1.12
-- example.com@v1.2.3/blah/blah.go --
package blah

const Name = "Blah"
-- random.com@v1.2.3/go.mod --
module random.com

go 1.12
-- random.com@v1.2.3/blah/blah.go --
package blah

const Name = "Blah"
`
	t.Run("almost tidied", func(t *testing.T) {
		const mod = `
-- go.mod --
module mod.com

go 1.12

require hasdep.com v1.2.3
-- go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
hasdep.com v1.2.3 h1:00y+N5oD+SpKoqV1zP2VOPawcW65Zb9NebANY3GSzGI=
hasdep.com v1.2.3/go.mod h1:ePVZOlez+KZEOejfLPGL2n4i8qiAjrkhQZ4wcImqAes=
-- main.go --
package main

func main() {}
`
		WithOptions(
			ProxyFiles(proxy),
		).Run(t, mod, func(t *testing.T, env *Env) {
			d := &protocol.PublishDiagnosticsParams{}
			env.Await(
				OnceMet(
					env.DiagnosticAtRegexp("go.mod", "require hasdep.com v1.2.3"),
					ReadDiagnostics("go.mod", d),
				),
			)
			const want = `module mod.com

go 1.12
`
			env.ApplyQuickFixes("go.mod", d.Diagnostics)
			if got := env.ReadWorkspaceFile("go.mod"); got != want {
				t.Fatalf("unexpected content in go.mod:\n%s", tests.Diff(t, want, got))
			}
		})
	})

	t.Run("not tidied", func(t *testing.T) {
		const mod = `
-- go.mod --
module mod.com

go 1.12

require hasdep.com v1.2.3
require random.com v1.2.3
-- go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
hasdep.com v1.2.3 h1:00y+N5oD+SpKoqV1zP2VOPawcW65Zb9NebANY3GSzGI=
hasdep.com v1.2.3/go.mod h1:ePVZOlez+KZEOejfLPGL2n4i8qiAjrkhQZ4wcImqAes=
random.com v1.2.3 h1:PzYTykzqqH6+qU0dIgh9iPFbfb4Mm8zNBjWWreRKtx0=
random.com v1.2.3/go.mod h1:8EGj+8a4Hw1clAp8vbaeHAsKE4sbm536FP7nKyXO+qQ=
-- main.go --
package main

func main() {}
`
		WithOptions(
			ProxyFiles(proxy),
		).Run(t, mod, func(t *testing.T, env *Env) {
			d := &protocol.PublishDiagnosticsParams{}
			env.OpenFile("go.mod")
			pos := env.RegexpSearch("go.mod", "require hasdep.com v1.2.3")
			env.Await(
				OnceMet(
					DiagnosticAt("go.mod", pos.Line, pos.Column),
					ReadDiagnostics("go.mod", d),
				),
			)
			const want = `module mod.com

go 1.12

require random.com v1.2.3
`
			var diagnostics []protocol.Diagnostic
			for _, d := range d.Diagnostics {
				if d.Range.Start.Line != float64(pos.Line) {
					continue
				}
				diagnostics = append(diagnostics, d)
			}
			env.ApplyQuickFixes("go.mod", diagnostics)
			if got := env.Editor.BufferText("go.mod"); got != want {
				t.Fatalf("unexpected content in go.mod:\n%s", tests.Diff(t, want, got))
			}
		})
	})
}

func TestSumUpdateQuickFix(t *testing.T) {
	// Error messages changed in 1.16 that changed the diagnostic positions.
	testenv.NeedsGo1Point(t, 16)

	const mod = `
-- go.mod --
module mod.com

go 1.12

require (
	example.com v1.2.3
)
-- go.sum --
-- main.go --
package main

import (
	"example.com/blah"
)

func main() {
	blah.Hello()
}
`
	WithOptions(
		ProxyFiles(workspaceProxy),
		Modes(Singleton),
	).Run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		pos := env.RegexpSearch("go.mod", "example.com")
		params := &protocol.PublishDiagnosticsParams{}
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("go.mod", "example.com"),
				ReadDiagnostics("go.mod", params),
			),
		)
		var diagnostic protocol.Diagnostic
		for _, d := range params.Diagnostics {
			if d.Range.Start.Line == float64(pos.Line) {
				diagnostic = d
				break
			}
		}
		env.ApplyQuickFixes("go.mod", []protocol.Diagnostic{diagnostic})
		const want = `example.com v1.2.3 h1:Yryq11hF02fEf2JlOS2eph+ICE2/ceevGV3C9dl5V/c=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
`
		if got := env.ReadWorkspaceFile("go.sum"); got != want {
			t.Fatalf("unexpected go.sum contents:\n%s", tests.Diff(t, want, got))
		}
	})
}
