// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/fake"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// repos holds shared repositories for use in benchmarks.
//
// These repos were selected to represent a variety of different types of
// codebases.
var repos = map[string]*repo{
	// google-cloud-go has 145 workspace modules (!), and is quite large.
	"google-cloud-go": {
		name:   "google-cloud-go",
		url:    "https://github.com/googleapis/google-cloud-go.git",
		commit: "07da765765218debf83148cc7ed8a36d6e8921d5",
		inDir:  flag.String("cloud_go_dir", "", "if set, reuse this directory as google-cloud-go@07da7657"),
	},

	// Used by x/benchmarks; large.
	"istio": {
		name:   "istio",
		url:    "https://github.com/istio/istio",
		commit: "1.17.0",
		inDir:  flag.String("istio_dir", "", "if set, reuse this directory as istio@v1.17.0"),
	},

	// Kubernetes is a large repo with many dependencies, and in the past has
	// been about as large a repo as gopls could handle.
	"kubernetes": {
		name:   "kubernetes",
		url:    "https://github.com/kubernetes/kubernetes",
		commit: "v1.24.0",
		short:  true,
		inDir:  flag.String("kubernetes_dir", "", "if set, reuse this directory as kubernetes@v1.24.0"),
	},

	// A large, industrial application.
	"kuma": {
		name:   "kuma",
		url:    "https://github.com/kumahq/kuma",
		commit: "2.1.1",
		inDir:  flag.String("kuma_dir", "", "if set, reuse this directory as kuma@v2.1.1"),
	},

	// A repo containing a very large package (./dataintegration).
	"oracle": {
		name:   "oracle",
		url:    "https://github.com/oracle/oci-go-sdk.git",
		commit: "v65.43.0",
		short:  true,
		inDir:  flag.String("oracle_dir", "", "if set, reuse this directory as oracle/oci-go-sdk@v65.43.0"),
	},

	// x/pkgsite is familiar and represents a common use case (a webserver). It
	// also has a number of static non-go files and template files.
	"pkgsite": {
		name:   "pkgsite",
		url:    "https://go.googlesource.com/pkgsite",
		commit: "81f6f8d4175ad0bf6feaa03543cc433f8b04b19b",
		short:  true,
		inDir:  flag.String("pkgsite_dir", "", "if set, reuse this directory as pkgsite@81f6f8d4"),
	},

	// A tiny self-contained project.
	"starlark": {
		name:   "starlark",
		url:    "https://github.com/google/starlark-go",
		commit: "3f75dec8e4039385901a30981e3703470d77e027",
		short:  true,
		inDir:  flag.String("starlark_dir", "", "if set, reuse this directory as starlark@3f75dec8"),
	},

	// The current repository, which is medium-small and has very few dependencies.
	"tools": {
		name:   "tools",
		url:    "https://go.googlesource.com/tools",
		commit: "gopls/v0.9.0",
		short:  true,
		inDir:  flag.String("tools_dir", "", "if set, reuse this directory as x/tools@v0.9.0"),
	},

	// A repo of similar size to kubernetes, but with substantially more
	// complex types that led to a serious performance regression (issue #60621).
	"hashiform": {
		name:   "hashiform",
		url:    "https://github.com/hashicorp/terraform-provider-aws",
		commit: "ac55de2b1950972d93feaa250d7505d9ed829c7c",
		inDir:  flag.String("hashiform_dir", "", "if set, reuse this directory as hashiform@ac55de2"),
	},
}

// getRepo gets the requested repo, and skips the test if -short is set and
// repo is not configured as a short repo.
func getRepo(tb testing.TB, name string) *repo {
	tb.Helper()
	repo := repos[name]
	if repo == nil {
		tb.Fatalf("repo %s does not exist", name)
	}
	if !repo.short && testing.Short() {
		tb.Skipf("large repo %s does not run with -short", repo.name)
	}
	return repo
}

// A repo represents a working directory for a repository checked out at a
// specific commit.
//
// Repos are used for sharing state across benchmarks that operate on the same
// codebase.
type repo struct {
	// static configuration
	name   string  // must be unique, used for subdirectory
	url    string  // repo url
	commit string  // full commit hash or tag
	short  bool    // whether this repo runs with -short
	inDir  *string // if set, use this dir as url@commit, and don't delete

	dirOnce sync.Once
	dir     string // directory contaning source code checked out to url@commit

	// shared editor state
	editorOnce sync.Once
	editor     *fake.Editor
	sandbox    *fake.Sandbox
	awaiter    *Awaiter
}

// reusableDir return a reusable directory for benchmarking, or "".
//
// If the user specifies a directory, the test will create and populate it
// on the first run an re-use it on subsequent runs. Otherwise it will
// create, populate, and delete a temporary directory.
func (r *repo) reusableDir() string {
	if r.inDir == nil {
		return ""
	}
	return *r.inDir
}

// getDir returns directory containing repo source code, creating it if
// necessary. It is safe for concurrent use.
func (r *repo) getDir() string {
	r.dirOnce.Do(func() {
		if r.dir = r.reusableDir(); r.dir == "" {
			r.dir = filepath.Join(getTempDir(), r.name)
		}

		_, err := os.Stat(r.dir)
		switch {
		case os.IsNotExist(err):
			log.Printf("cloning %s@%s into %s", r.url, r.commit, r.dir)
			if err := shallowClone(r.dir, r.url, r.commit); err != nil {
				log.Fatal(err)
			}
		case err != nil:
			log.Fatal(err)
		default:
			log.Printf("reusing %s as %s@%s", r.dir, r.url, r.commit)
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

		start := time.Now()
		log.Printf("starting initial workspace load for %s", r.name)
		ts, err := newGoplsConnector(profileArgs(r.name, false))
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
		log.Printf("initial workspace load (cold) for %s took %v", r.name, time.Since(start))
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
func (r *repo) newEnv(tb testing.TB, config fake.EditorConfig, forOperation string, cpuProfile bool) *Env {
	dir := r.getDir()

	args := profileArgs(qualifiedName(r.name, forOperation), cpuProfile)
	ts, err := newGoplsConnector(args)
	if err != nil {
		tb.Fatal(err)
	}
	sandbox, editor, awaiter, err := connectEditor(dir, config, ts)
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
	if r.dir != "" && r.reusableDir() == "" {
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
