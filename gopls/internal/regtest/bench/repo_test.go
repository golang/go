// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/fake"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// repos holds shared repositories for use in benchmarks.
var repos = map[string]*repo{
	"tools": {name: "tools", url: "https://go.googlesource.com/tools", commit: "gopls/v0.9.0"},
}

// A repo represents a working directory for a repository checked out at a
// specific commit.
//
// Repos are used for sharing state across benchmarks that operate on the same
// codebase.
type repo struct {
	// static configuration
	name   string // must be unique, used for subdirectory
	url    string // repo url
	commit string // commitish, e.g. tag or short commit hash

	dirOnce sync.Once
	dir     string // directory contaning source code checked out to url@commit

	// shared editor state
	editorOnce sync.Once
	editor     *fake.Editor
	sandbox    *fake.Sandbox
	awaiter    *Awaiter
}

// getDir returns directory containing repo source code, creating it if
// necessary. It is safe for concurrent use.
func (r *repo) getDir() string {
	r.dirOnce.Do(func() {
		r.dir = filepath.Join(getTempDir(), r.name)
		log.Printf("cloning %s@%s into %s", r.url, r.commit, r.dir)
		if err := shallowClone(r.dir, r.url, r.commit); err != nil {
			log.Fatal(err)
		}
	})
	return r.dir
}

// sharedEnv returns a shared benchmark environment. It is safe for concurrent
// use.
//
// Every call to sharedEnv uses the same editor and sandbox, as a means to
// avoid reinitializing the editor for large repos. Calling repo.Close cleans
// up the shared environment.
//
// Repos in the package-local Repos var are closed at the end of the test main
// function.
func (r *repo) sharedEnv(tb testing.TB) *Env {
	r.editorOnce.Do(func() {
		dir := r.getDir()

		ts, err := newGoplsServer(r.name)
		if err != nil {
			log.Fatal(err)
		}
		r.sandbox, r.editor, r.awaiter, err = connectEditor(dir, fake.EditorConfig{}, ts)
		if err != nil {
			log.Fatalf("connecting editor: %v", err)
		}

		if err := r.awaiter.Await(context.Background(), InitialWorkspaceLoad); err != nil {
			log.Fatal(err)
		}
	})

	return &Env{
		T:       tb,
		Ctx:     context.Background(),
		Editor:  r.editor,
		Sandbox: r.sandbox,
		Awaiter: r.awaiter,
	}
}

// newEnv returns a new Env connected to a new gopls process communicating
// over stdin/stdout. It is safe for concurrent use.
//
// It is the caller's responsibility to call Close on the resulting Env when it
// is no longer needed.
func (r *repo) newEnv(tb testing.TB) *Env {
	dir := r.getDir()

	ts, err := newGoplsServer(tb.Name())
	if err != nil {
		tb.Fatal(err)
	}
	sandbox, editor, awaiter, err := connectEditor(dir, fake.EditorConfig{}, ts)
	if err != nil {
		log.Fatalf("connecting editor: %v", err)
	}

	return &Env{
		T:       tb,
		Ctx:     context.Background(),
		Editor:  editor,
		Sandbox: sandbox,
		Awaiter: awaiter,
	}
}

// Close cleans up shared state referenced by the repo.
func (r *repo) Close() error {
	var errBuf bytes.Buffer
	if r.editor != nil {
		if err := r.editor.Close(context.Background()); err != nil {
			fmt.Fprintf(&errBuf, "closing editor: %v", err)
		}
	}
	if r.sandbox != nil {
		if err := r.sandbox.Close(); err != nil {
			fmt.Fprintf(&errBuf, "closing sandbox: %v", err)
		}
	}
	if r.dir != "" {
		if err := os.RemoveAll(r.dir); err != nil {
			fmt.Fprintf(&errBuf, "cleaning dir: %v", err)
		}
	}
	if errBuf.Len() > 0 {
		return errors.New(errBuf.String())
	}
	return nil
}

// cleanup cleans up state that is shared across benchmark functions.
func cleanup() error {
	var errBuf bytes.Buffer
	for _, repo := range repos {
		if err := repo.Close(); err != nil {
			fmt.Fprintf(&errBuf, "closing %q: %v", repo.name, err)
		}
	}
	if tempDir != "" {
		if err := os.RemoveAll(tempDir); err != nil {
			fmt.Fprintf(&errBuf, "cleaning tempDir: %v", err)
		}
	}
	if errBuf.Len() > 0 {
		return errors.New(errBuf.String())
	}
	return nil
}
