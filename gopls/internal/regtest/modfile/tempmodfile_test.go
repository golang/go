// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modfile

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// This test replaces an older, problematic test (golang/go#57784). But it has
// been a long time since the go command would mutate go.mod files.
//
// TODO(golang/go#61970): the tempModfile setting should be removed entirely.
func TestTempModfileUnchanged(t *testing.T) {
	// badMod has a go.mod file that is missing a go directive.
	const badMod = `
-- go.mod --
module badmod.test/p
-- p.go --
package p
`

	WithOptions(
		Modes(Default), // no reason to test this with a remote gopls
		ProxyFiles(workspaceProxy),
		Settings{
			"tempModfile": true,
		},
	).Run(t, badMod, func(t *testing.T, env *Env) {
		env.OpenFile("p.go")
		env.AfterChange()
		want := "module badmod.test/p\n"
		got := env.ReadWorkspaceFile("go.mod")
		if got != want {
			t.Errorf("go.mod content:\n%s\nwant:\n%s", got, want)
		}
	})
}
