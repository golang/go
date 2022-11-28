// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"context"
	"flag"
	"fmt"
	"testing"
	"time"

	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/lsprpc"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
)

// github.com/pilosa/pilosa is a repository that has historically caused
// significant memory problems for Gopls. We use it for a simple stress test
// that types arbitrarily in a file with lots of dependents.

var pilosaPath = flag.String("pilosa_path", "", "Path to a directory containing "+
	"github.com/pilosa/pilosa, for stress testing. Do not set this unless you "+
	"know what you're doing!")

func TestPilosaStress(t *testing.T) {
	// TODO(rfindley): revisit this test and make it is hermetic: it should check
	// out pilosa into a directory.
	//
	// Note: This stress test has not been run recently, and may no longer
	// function properly.
	if *pilosaPath == "" {
		t.Skip("-pilosa_path not configured")
	}

	sandbox, err := fake.NewSandbox(&fake.SandboxConfig{
		Workdir: *pilosaPath,
		GOPROXY: "https://proxy.golang.org",
	})
	if err != nil {
		t.Fatal(err)
	}

	server := lsprpc.NewStreamServer(cache.New(nil, nil), false, hooks.Options)
	ts := servertest.NewPipeServer(server, jsonrpc2.NewRawStream)
	ctx := context.Background()

	editor, err := fake.NewEditor(sandbox, fake.EditorConfig{}).Connect(ctx, ts, fake.ClientHooks{})
	if err != nil {
		t.Fatal(err)
	}

	files := []string{
		"cmd.go",
		"internal/private.pb.go",
		"roaring/roaring.go",
		"roaring/roaring_internal_test.go",
		"server/handler_test.go",
	}
	for _, file := range files {
		if err := editor.OpenFile(ctx, file); err != nil {
			t.Fatal(err)
		}
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()

	i := 1
	// MagicNumber is an identifier that occurs in roaring.go. Just change it
	// arbitrarily.
	if err := editor.RegexpReplace(ctx, "roaring/roaring.go", "MagicNumber", fmt.Sprintf("MagicNumber%d", 1)); err != nil {
		t.Fatal(err)
	}
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		if err := editor.RegexpReplace(ctx, "roaring/roaring.go", fmt.Sprintf("MagicNumber%d", i), fmt.Sprintf("MagicNumber%d", i+1)); err != nil {
			t.Fatal(err)
		}
		// Simulate (very fast) typing.
		//
		// Typing 80 wpm ~150ms per keystroke.
		time.Sleep(150 * time.Millisecond)
		i++
	}
}
