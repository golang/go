// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/testenv"
)

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

func runModfileTest(t *testing.T, files, proxy string, f TestFunc) {
	t.Run("normal", func(t *testing.T) {
		withOptions(WithProxyFiles(proxy)).run(t, files, f)
	})
	t.Run("nested", func(t *testing.T) {
		withOptions(WithProxyFiles(proxy), NestWorkdir(), WithModes(Singleton|Experimental)).run(t, files, f)
	})
}

func TestModFileModification(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const untidyModule = `
-- go.mod --
module mod.com

-- main.go --
package main

import "example.com/blah"

func main() {
	println(blah.Name)
}
`
	t.Run("basic", func(t *testing.T) {
		runModfileTest(t, untidyModule, proxy, func(t *testing.T, env *Env) {
			// Open the file and make sure that the initial workspace load does not
			// modify the go.mod file.
			goModContent := env.ReadWorkspaceFile("go.mod")
			env.OpenFile("main.go")
			env.Await(
				env.DiagnosticAtRegexp("main.go", "\"example.com/blah\""),
			)
			if got := env.ReadWorkspaceFile("go.mod"); got != goModContent {
				t.Fatalf("go.mod changed on disk:\n%s", tests.Diff(goModContent, got))
			}
			// Save the buffer, which will format and organize imports.
			// Confirm that the go.mod file still does not change.
			env.SaveBuffer("main.go")
			env.Await(
				env.DiagnosticAtRegexp("main.go", "\"example.com/blah\""),
			)
			if got := env.ReadWorkspaceFile("go.mod"); got != goModContent {
				t.Fatalf("go.mod changed on disk:\n%s", tests.Diff(goModContent, got))
			}
		})
	})

	// Reproduce golang/go#40269 by deleting and recreating main.go.
	t.Run("delete main.go", func(t *testing.T) {
		t.Skip("This test will be flaky until golang/go#40269 is resolved.")

		runModfileTest(t, untidyModule, proxy, func(t *testing.T, env *Env) {
			goModContent := env.ReadWorkspaceFile("go.mod")
			mainContent := env.ReadWorkspaceFile("main.go")
			env.OpenFile("main.go")
			env.SaveBuffer("main.go")

			env.RemoveWorkspaceFile("main.go")
			env.Await(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1),
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidSave), 1),
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 2),
			)

			env.WriteWorkspaceFile("main.go", mainContent)
			env.Await(
				env.DiagnosticAtRegexp("main.go", "\"example.com/blah\""),
			)
			if got := env.ReadWorkspaceFile("go.mod"); got != goModContent {
				t.Fatalf("go.mod changed on disk:\n%s", tests.Diff(goModContent, got))
			}
		})
	})
}

func TestGoGetFix(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)
	const mod = `
-- go.mod --
module mod.com

go 1.12

-- main.go --
package main

import "example.com/blah"

var _ = blah.Name
`

	const want = `module mod.com

go 1.12

require example.com v1.2.3
`

	runModfileTest(t, mod, proxy, func(t *testing.T, env *Env) {
		if strings.Contains(t.Name(), "workspace_module") {
			t.Skip("workspace module mode doesn't set -mod=readonly")
		}
		env.OpenFile("main.go")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("main.go", `"example.com/blah"`),
				ReadDiagnostics("main.go", &d),
			),
		)
		var goGetDiag protocol.Diagnostic
		for _, diag := range d.Diagnostics {
			if strings.Contains(diag.Message, "could not import") {
				goGetDiag = diag
			}
		}
		env.ApplyQuickFixes("main.go", []protocol.Diagnostic{goGetDiag})
		if got := env.ReadWorkspaceFile("go.mod"); got != want {
			t.Fatalf("unexpected go.mod content:\n%s", tests.Diff(want, got))
		}
	})
}

// Tests that multiple missing dependencies gives good single fixes.
func TestMissingDependencyFixes(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)
	const mod = `
-- go.mod --
module mod.com

go 1.12

-- main.go --
package main

import "example.com/blah"
import "random.org/blah"

var _, _ = blah.Name, hello.Name
`

	const want = `module mod.com

go 1.12

require random.org v1.2.3
`

	runModfileTest(t, mod, proxy, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("main.go", `"random.org/blah"`),
				ReadDiagnostics("main.go", &d),
			),
		)
		var randomDiag protocol.Diagnostic
		for _, diag := range d.Diagnostics {
			if strings.Contains(diag.Message, "random.org") {
				randomDiag = diag
			}
		}
		env.ApplyQuickFixes("main.go", []protocol.Diagnostic{randomDiag})
		if got := env.ReadWorkspaceFile("go.mod"); got != want {
			t.Fatalf("unexpected go.mod content:\n%s", tests.Diff(want, got))
		}
	})
}

func TestIndirectDependencyFix(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const mod = `
-- go.mod --
module mod.com

go 1.12

require example.com v1.2.3 // indirect
-- go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- main.go --
package main

import "example.com/blah"

func main() {
	fmt.Println(blah.Name)
`
	const want = `module mod.com

go 1.12

require example.com v1.2.3
`
	runModfileTest(t, mod, proxy, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("go.mod", "// indirect"),
				ReadDiagnostics("go.mod", &d),
			),
		)
		env.ApplyQuickFixes("go.mod", d.Diagnostics)
		if got := env.Editor.BufferText("go.mod"); got != want {
			t.Fatalf("unexpected go.mod content:\n%s", tests.Diff(want, got))
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
-- go.mod --
module mod.com
go 1.14
require example.com v1.0.0
-- go.sum --
example.com v1.0.0 h1:38O7j5rEBajXk+Q5wzLbRN7KqMkSgEiN9NqcM1O2bBM=
example.com v1.0.0/go.mod h1:vUsPMGpx9ZXXzECCOsOmYCW7npJTwuA16yl89n3Mgls=
-- main.go --
package main
func main() {}
`

	const want = `module mod.com

go 1.14
`

	runModfileTest(t, files, proxy, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("go.mod", `require example.com`),
				ReadDiagnostics("go.mod", &d),
			),
		)
		env.ApplyQuickFixes("go.mod", d.Diagnostics)
		if got := env.ReadWorkspaceFile("go.mod"); got != want {
			t.Fatalf("unexpected go.mod content:\n%s", tests.Diff(want, got))
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
-- go.mod --
module mod.com

go 1.14

require google.golang.org/protobuf v1.20.0
-- go.sum --
github.com/esimov/caire v1.2.5 h1:OcqDII/BYxcBYj3DuwDKjd+ANhRxRqLa2n69EGje7qw=
github.com/esimov/caire v1.2.5/go.mod h1:mXnjRjg3+WUtuhfSC1rKRmdZU9vJZyS1ZWU0qSvJhK8=
google.golang.org/protobuf v1.20.0 h1:y9T1vAtFKQg0faFNMOxJU7WuEqPWolVkjIkU6aI8qCY=
google.golang.org/protobuf v1.20.0/go.mod h1:FcqsytGClbtLv1ot8NvsJHjBi0h22StKVP+K/j2liKA=
-- main.go --
package main

import (
    "github.com/esimov/caire"
)

func _() {
    caire.RemoveTempImage()
}`
	runModfileTest(t, repro, proxy, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				env.DiagnosticAtRegexp("main.go", `"github.com/esimov/caire"`),
				ReadDiagnostics("main.go", &d),
			),
		)
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		want := `module mod.com

go 1.14

require (
	github.com/esimov/caire v1.2.5
	google.golang.org/protobuf v1.20.0
)
`
		if got := env.ReadWorkspaceFile("go.mod"); got != want {
			t.Fatalf("TestNewDepWithUnusedDep failed:\n%s", tests.Diff(want, got))
		}
	})
}

// TODO: For this test to be effective, the sandbox's file watcher must respect
// the file watching GlobPattern in the capability registration. See
// golang/go#39384.
func TestModuleChangesOnDisk(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const mod = `
-- go.mod --
module mod.com

go 1.12

require example.com v1.2.3
-- go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- main.go --
package main

func main() {
	fmt.Println(blah.Name)
`
	runModfileTest(t, mod, proxy, func(t *testing.T, env *Env) {
		env.Await(env.DiagnosticAtRegexp("go.mod", "require"))
		env.RunGoCommand("mod", "tidy")
		env.Await(
			EmptyDiagnostics("go.mod"),
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
-- go.mod --
module mod.com

go 1.12

require example.com/blah/v2 v2.0.0
-- go.sum --
example.com/blah v1.0.0 h1:kGPlWJbMsn1P31H9xp/q2mYI32cxLnCvauHN0AVaHnc=
example.com/blah v1.0.0/go.mod h1:PZUQaGFeVjyDmAE8ywmLbmDn3fj4Ws8epg4oLuDzW3M=
example.com/blah/v2 v2.0.0 h1:w5baE9JuuU11s3de3yWx2sU05AhNkgLYdZ4qukv+V0k=
example.com/blah/v2 v2.0.0/go.mod h1:UZiKbTwobERo/hrqFLvIQlJwQZQGxWMVY4xere8mj7w=
-- main.go --
package main

import "example.com/blah/v2"

var _ = blah.Name
`
	withOptions(WithProxyFiles(proxy)).run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		env.OpenFile("go.mod")
		var d protocol.PublishDiagnosticsParams
		env.Await(
			OnceMet(
				DiagnosticAt("go.mod", 0, 0),
				ReadDiagnostics("go.mod", &d),
			),
		)
		env.ApplyQuickFixes("main.go", d.Diagnostics)
		const want = `module mod.com

go 1.12

require (
	example.com/blah v1.0.0 // indirect
	example.com/blah/v2 v2.0.0
)
`
		env.Await(EmptyDiagnostics("go.mod"))
		if got := env.Editor.BufferText("go.mod"); got != want {
			t.Fatalf("suggested fixes failed:\n%s", tests.Diff(want, got))
		}
	})
}

// Reproduces golang/go#38232.
func TestUnknownRevision(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const unknown = `
-- go.mod --
module mod.com

require (
	example.com v1.2.2
)
-- main.go --
package main

import "example.com/blah"

func main() {
	var x = blah.Name
}
`

	// Start from a bad state/bad IWL, and confirm that we recover.
	t.Run("bad", func(t *testing.T) {
		runModfileTest(t, unknown, proxy, func(t *testing.T, env *Env) {
			env.OpenFile("go.mod")
			env.Await(
				env.DiagnosticAtRegexp("go.mod", "example.com v1.2.2"),
			)
			env.RegexpReplace("go.mod", "v1.2.2", "v1.2.3")
			env.Editor.SaveBuffer(env.Ctx, "go.mod") // go.mod changes must be on disk

			d := protocol.PublishDiagnosticsParams{}
			env.Await(
				OnceMet(
					env.DiagnosticAtRegexpWithMessage("go.mod", "example.com v1.2.3", "example.com@v1.2.3"),
					ReadDiagnostics("go.mod", &d),
				),
			)
			env.ApplyQuickFixes("go.mod", d.Diagnostics)

			env.Await(
				EmptyDiagnostics("go.mod"),
				env.DiagnosticAtRegexp("main.go", "x = "),
			)
		})
	})

	const known = `
-- go.mod --
module mod.com

require (
	example.com v1.2.3
)
-- go.sum --
example.com v1.2.3 h1:ihBTGWGjTU3V4ZJ9OmHITkU9WQ4lGdQkMjgyLFk0FaY=
example.com v1.2.3/go.mod h1:Y2Rc5rVWjWur0h3pd9aEvK5Pof8YKDANh9gHA2Maujo=
-- main.go --
package main

import "example.com/blah"

func main() {
	var x = blah.Name
}
`
	// Start from a good state, transform to a bad state, and confirm that we
	// still recover.
	t.Run("good", func(t *testing.T) {
		runModfileTest(t, known, proxy, func(t *testing.T, env *Env) {
			env.OpenFile("go.mod")
			env.Await(
				env.DiagnosticAtRegexp("main.go", "x = "),
			)
			env.RegexpReplace("go.mod", "v1.2.3", "v1.2.2")
			env.Editor.SaveBuffer(env.Ctx, "go.mod") // go.mod changes must be on disk
			env.Await(
				env.DiagnosticAtRegexp("go.mod", "example.com v1.2.2"),
			)
			env.RegexpReplace("go.mod", "v1.2.2", "v1.2.3")
			env.Editor.SaveBuffer(env.Ctx, "go.mod") // go.mod changes must be on disk
			env.Await(
				env.DiagnosticAtRegexp("main.go", "x = "),
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
-- go.mod --
module mod.com

go 1.14

require example.com v1.2.3
-- main.go --
package main

import "example.com/blah"

func main() {
	println(blah.Name)
}
`
	runModfileTest(t, module, badProxy, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.Await(
			env.DiagnosticAtRegexp("go.mod", "require example.com v1.2.3"),
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
	withOptions(
		EditorConfig{
			Env: map[string]string{
				"GOFLAGS": "-mod=readonly",
			},
		},
		WithProxyFiles(proxy),
		WithModes(Singleton),
	).run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("main.go")
		original := env.ReadWorkspaceFile("go.mod")
		env.Await(
			env.DiagnosticAtRegexp("main.go", `"example.com/blah"`),
		)
		got := env.ReadWorkspaceFile("go.mod")
		if got != original {
			t.Fatalf("go.mod file modified:\n%s", tests.Diff(original, got))
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
module mod.com

go 1.14

require (
	example.com v1.2.3
)
-- a/main.go --
package main

func main() {}
-- b/go.mod --
module mod.com

go 1.14
-- b/main.go --
package main

import "example.com/blah"

func main() {
	blah.SaySomething()
}
`
	withOptions(
		WithProxyFiles(workspaceProxy),
		WithModes(Experimental),
	).run(t, mod, func(t *testing.T, env *Env) {
		env.Await(
			env.DiagnosticAtRegexp("a/go.mod", "example.com v1.2.3"),
			env.DiagnosticAtRegexp("b/go.mod", "module mod.com"),
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
	withOptions(
		WithProxyFiles(workspaceProxy),
		EditorConfig{
			BuildFlags: []string{"-tags", "bob"},
		},
	).run(t, mod, func(t *testing.T, env *Env) {
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
	run(t, mod, func(t *testing.T, env *Env) {
		env.OpenFile("go.mod")
		env.RegexpReplace("go.mod", "module", "modul")
		env.Await(
			env.DiagnosticAtRegexp("go.mod", "modul"),
		)
	})
}
