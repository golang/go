// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"

	"golang.org/x/tools/internal/testenv"
)

// Test that enabling and disabling produces the expected results of showing
// and hiding staticcheck analysis results.
func TestChangeConfiguration(t *testing.T) {
	// Staticcheck only supports Go versions >= 1.17.
	// Note: keep this in sync with TestStaticcheckWarning. Below this version we
	// should get an error when setting staticcheck configuration.
	testenv.NeedsGo1Point(t, 17)

	const files = `
-- go.mod --
module mod.com

go 1.12
-- a/a.go --
package a

import "errors"

// FooErr should be called ErrFoo (ST1012)
var FooErr = errors.New("foo")
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		env.Await(
			env.DoneWithOpen(),
			EmptyDiagnostics("a/a.go"),
		)
		cfg := env.Editor.Config()
		cfg.Settings = map[string]interface{}{
			"staticcheck": true,
		}
		env.ChangeConfiguration(cfg)
		env.Await(
			DiagnosticAt("a/a.go", 5, 4),
		)
	})
}

func TestStaticcheckWarning(t *testing.T) {
	// Note: keep this in sync with TestChangeConfiguration.
	testenv.SkipAfterGo1Point(t, 16)

	const files = `
-- go.mod --
module mod.com

go 1.12
-- a/a.go --
package a

import "errors"

// FooErr should be called ErrFoo (ST1012)
var FooErr = errors.New("foo")
`

	WithOptions(
		Settings{"staticcheck": true},
	).Run(t, files, func(t *testing.T, env *Env) {
		env.Await(
			OnceMet(
				InitialWorkspaceLoad,
				ShownMessage("staticcheck is not supported"),
			),
		)
	})
}

func TestGofumptWarning(t *testing.T) {
	testenv.SkipAfterGo1Point(t, 17)

	WithOptions(
		Settings{"gofumpt": true},
	).Run(t, "", func(t *testing.T, env *Env) {
		env.Await(
			OnceMet(
				InitialWorkspaceLoad,
				ShownMessage("gofumpt is not supported"),
			),
		)
	})
}

func TestDeprecatedSettings(t *testing.T) {
	WithOptions(
		Settings{
			"experimentalUseInvalidMetadata": true,
			"experimentalWatchedFileDelay":   "1s",
			"experimentalWorkspaceModule":    true,
		},
	).Run(t, "", func(t *testing.T, env *Env) {
		env.Await(
			OnceMet(
				InitialWorkspaceLoad,
				ShownMessage("experimentalWorkspaceModule"),
				ShownMessage("experimentalUseInvalidMetadata"),
				ShownMessage("experimentalWatchedFileDelay"),
			),
		)
	})
}
