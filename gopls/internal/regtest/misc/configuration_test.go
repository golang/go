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
	// Staticcheck only supports Go versions >= 1.19.
	// Note: keep this in sync with TestStaticcheckWarning. Below this version we
	// should get an error when setting staticcheck configuration.
	testenv.NeedsGo1Point(t, 19)

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
		env.AfterChange(
			NoDiagnostics(ForFile("a/a.go")),
		)
		cfg := env.Editor.Config()
		cfg.Settings = map[string]interface{}{
			"staticcheck": true,
		}
		env.ChangeConfiguration(cfg)
		env.AfterChange(
			Diagnostics(env.AtRegexp("a/a.go", "var (FooErr)")),
		)
	})
}

// TestMajorOptionsChange is like TestChangeConfiguration, but modifies an
// an open buffer before making a major (but inconsequential) change that
// causes gopls to recreate the view.
//
// Gopls should not get confused about buffer content when recreating the view.
func TestMajorOptionsChange(t *testing.T) {
	testenv.NeedsGo1Point(t, 19) // needs staticcheck

	const files = `
-- go.mod --
module mod.com

go 1.12
-- a/a.go --
package a

import "errors"

var ErrFoo = errors.New("foo")
`
	Run(t, files, func(t *testing.T, env *Env) {
		env.OpenFile("a/a.go")
		// Introduce a staticcheck diagnostic. It should be detected when we enable
		// staticcheck later.
		env.RegexpReplace("a/a.go", "ErrFoo", "FooErr")
		env.AfterChange(
			NoDiagnostics(ForFile("a/a.go")),
		)
		cfg := env.Editor.Config()
		// Any change to environment recreates the view, but this should not cause
		// gopls to get confused about the content of a/a.go: we should get the
		// staticcheck diagnostic below.
		cfg.Env = map[string]string{
			"AN_ARBITRARY_VAR": "FOO",
		}
		cfg.Settings = map[string]interface{}{
			"staticcheck": true,
		}
		env.ChangeConfiguration(cfg)
		env.AfterChange(
			Diagnostics(env.AtRegexp("a/a.go", "var (FooErr)")),
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
		env.OnceMet(
			InitialWorkspaceLoad,
			ShownMessage("staticcheck is not supported"),
		)
	})
}

func TestGofumptWarning(t *testing.T) {
	testenv.SkipAfterGo1Point(t, 17)

	WithOptions(
		Settings{"gofumpt": true},
	).Run(t, "", func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			ShownMessage("gofumpt is not supported"),
		)
	})
}

func TestDeprecatedSettings(t *testing.T) {
	WithOptions(
		Settings{
			"experimentalUseInvalidMetadata": true,
			"experimentalWatchedFileDelay":   "1s",
			"experimentalWorkspaceModule":    true,
			"tempModfile":                    true,
			"expandWorkspaceToModule":        false,
		},
	).Run(t, "", func(t *testing.T, env *Env) {
		env.OnceMet(
			InitialWorkspaceLoad,
			ShownMessage("experimentalWorkspaceModule"),
			ShownMessage("experimentalUseInvalidMetadata"),
			ShownMessage("experimentalWatchedFileDelay"),
			ShownMessage("https://go.dev/issue/63537"), // issue to remove tempModfile
			ShownMessage("https://go.dev/issue/63536"), // issue to remove expandWorkspaceToModule
		)
	})
}
