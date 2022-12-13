// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/lsprpc"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
)

// Test for golang/go#57222.
func TestCacheLeak(t *testing.T) {
	// TODO(rfindley): either fix this test with additional instrumentation, or
	// delete it.
	t.Skip("This test races with cache eviction.")
	const files = `-- a.go --
package a

func _() {
	println("1")
}
`
	c := cache.New(nil, nil)
	env := setupEnv(t, files, c)
	env.Await(InitialWorkspaceLoad)
	env.OpenFile("a.go")

	// Make a couple edits to stabilize cache state.
	//
	// For some reason, after only one edit we're left with two parsed files
	// (perhaps because something had to ParseHeader). If this test proves flaky,
	// we'll need to investigate exactly what is causing various parse modes to
	// be present (or rewrite the test to be more tolerant, for example make ~100
	// modifications and assert that we're within a few of where we're started).
	env.RegexpReplace("a.go", "1", "2")
	env.RegexpReplace("a.go", "2", "3")
	env.AfterChange()

	// Capture cache state, make an arbitrary change, and wait for gopls to do
	// its work. Afterward, we should have the exact same number of parsed
	before := c.MemStats()
	env.RegexpReplace("a.go", "3", "4")
	env.AfterChange()
	after := c.MemStats()

	if diff := cmp.Diff(before, after); diff != "" {
		t.Errorf("store objects differ after change (-before +after)\n%s", diff)
	}
}

// setupEnv creates a new sandbox environment for editing the txtar encoded
// content of files. It uses a new gopls instance backed by the Cache c.
func setupEnv(t *testing.T, files string, c *cache.Cache) *Env {
	ctx := debug.WithInstance(context.Background(), "", "off")
	server := lsprpc.NewStreamServer(c, false, hooks.Options)
	ts := servertest.NewPipeServer(server, jsonrpc2.NewRawStream)
	s, err := fake.NewSandbox(&fake.SandboxConfig{
		Files: fake.UnpackTxt(files),
	})
	if err != nil {
		t.Fatal(err)
	}

	a := NewAwaiter(s.Workdir)
	e, err := fake.NewEditor(s, fake.EditorConfig{}).Connect(ctx, ts, a.Hooks())

	return &Env{
		T:       t,
		Ctx:     ctx,
		Editor:  e,
		Sandbox: s,
		Awaiter: a,
	}
}
