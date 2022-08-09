// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/fakenet"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/lsp/bug"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/lsprpc"

	. "golang.org/x/tools/internal/lsp/regtest"
)

// This package implements benchmarks that share a common editor session.
//
// It is a work-in-progress.
//
// Remaining TODO(rfindley):
//   - add detailed documentation for how to write a benchmark, as a package doc
//   - add benchmarks for more features
//   - eliminate flags, and just run benchmarks on with a predefined set of
//     arguments

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	event.SetExporter(nil) // don't log to stderr
	code := doMain(m)
	os.Exit(code)
}

func doMain(m *testing.M) (code int) {
	defer func() {
		if editor != nil {
			if err := editor.Close(context.Background()); err != nil {
				fmt.Fprintf(os.Stderr, "closing editor: %v", err)
				if code == 0 {
					code = 1
				}
			}
		}
		if tempDir != "" {
			if err := os.RemoveAll(tempDir); err != nil {
				fmt.Fprintf(os.Stderr, "cleaning temp dir: %v", err)
				if code == 0 {
					code = 1
				}
			}
		}
	}()
	return m.Run()
}

var (
	workdir   = flag.String("workdir", "", "if set, working directory to use for benchmarks; overrides -repo and -commit")
	repo      = flag.String("repo", "https://go.googlesource.com/tools", "if set (and -workdir is unset), run benchmarks in this repo")
	file      = flag.String("file", "go/ast/astutil/util.go", "active file, for benchmarks that operate on a file")
	commitish = flag.String("commit", "gopls/v0.9.0", "if set (and -workdir is unset), run benchmarks at this commit")

	goplsPath    = flag.String("gopls_path", "", "if set, use this gopls for testing; incompatible with -gopls_version")
	goplsVersion = flag.String("gopls_version", "", "if set, install and use gopls at this version for testing; incompatible with -gopls_path")

	// If non-empty, tempDir is a temporary working dir that was created by this
	// test suite.
	//
	// The sync.Once variables guard various modifications of the temp directory.
	makeTempDirOnce  sync.Once
	checkoutRepoOnce sync.Once
	installGoplsOnce sync.Once
	tempDir          string

	setupEditorOnce sync.Once
	sandbox         *fake.Sandbox
	editor          *fake.Editor
	awaiter         *Awaiter
)

// getTempDir returns the temporary directory to use for benchmark files,
// creating it if necessary.
func getTempDir() string {
	makeTempDirOnce.Do(func() {
		var err error
		tempDir, err = ioutil.TempDir("", "gopls-bench")
		if err != nil {
			log.Fatal(err)
		}
	})
	return tempDir
}

// benchmarkDir returns the directory to use for benchmarks.
//
// If -workdir is set, just use that directory. Otherwise, check out a shallow
// copy of -repo at the given -commit, and clean up when the test suite exits.
func benchmarkDir() string {
	if *workdir != "" {
		return *workdir
	}
	if *repo == "" {
		log.Fatal("-repo must be provided if -workdir is unset")
	}
	if *commitish == "" {
		log.Fatal("-commit must be provided if -workdir is unset")
	}

	dir := filepath.Join(getTempDir(), "repo")
	checkoutRepoOnce.Do(func() {
		if err := os.Mkdir(dir, 0750); err != nil {
			log.Fatalf("creating repo dir: %v", err)
		}
		log.Printf("checking out %s@%s to %s\n", *repo, *commitish, dir)

		// Set a timeout for git fetch. If this proves flaky, it can be removed.
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
		defer cancel()

		// Use a shallow fetch to download just the releveant commit.
		shInit := fmt.Sprintf("git init && git fetch --depth=1 %q %q && git checkout FETCH_HEAD", *repo, *commitish)
		initCmd := exec.CommandContext(ctx, "/bin/sh", "-c", shInit)
		initCmd.Dir = dir
		if err := initCmd.Run(); err != nil {
			log.Fatalf("checking out %s: %v", *repo, err)
		}
	})
	return dir
}

// benchmarkEnv returns a shared benchmark environment
func benchmarkEnv(tb testing.TB) *Env {
	setupEditorOnce.Do(func() {
		dir := benchmarkDir()

		var err error
		sandbox, editor, awaiter, err = connectEditor(dir, fake.EditorConfig{})
		if err != nil {
			log.Fatalf("connecting editor: %v", err)
		}

		if err := awaiter.Await(context.Background(), InitialWorkspaceLoad); err != nil {
			panic(err)
		}
	})

	return &Env{
		T:       tb,
		Ctx:     context.Background(),
		Editor:  editor,
		Sandbox: sandbox,
		Awaiter: awaiter,
	}
}

// connectEditor connects a fake editor session in the given dir, using the
// given editor config.
func connectEditor(dir string, config fake.EditorConfig) (*fake.Sandbox, *fake.Editor, *Awaiter, error) {
	s, err := fake.NewSandbox(&fake.SandboxConfig{
		Workdir: dir,
		GOPROXY: "https://proxy.golang.org",
	})
	if err != nil {
		return nil, nil, nil, err
	}

	a := NewAwaiter(s.Workdir)
	ts := getServer()
	e, err := fake.NewEditor(s, config).Connect(context.Background(), ts, a.Hooks())
	if err != nil {
		return nil, nil, nil, err
	}
	return s, e, a, nil
}

// getServer returns a server connector that either starts a new in-process
// server, or starts a separate gopls process.
func getServer() servertest.Connector {
	if *goplsPath != "" && *goplsVersion != "" {
		panic("can't set both -gopls_path and -gopls_version")
	}
	if *goplsPath != "" {
		return &SidecarServer{*goplsPath}
	}
	if *goplsVersion != "" {
		path := getInstalledGopls(*goplsVersion)
		return &SidecarServer{path}
	}
	server := lsprpc.NewStreamServer(cache.New(nil, nil, hooks.Options), false)
	return servertest.NewPipeServer(server, jsonrpc2.NewRawStream)
}

func getInstalledGopls(version string) string {
	// Use a temp GOPATH to while installing gopls, to avoid overwriting gopls in
	// the user's PATH.
	gopath := filepath.Join(getTempDir(), "gopath")
	goplsPath := filepath.Join(gopath, "bin", "gopls")
	installGoplsOnce.Do(func() {
		if err := os.Mkdir(gopath, 0755); err != nil {
			log.Fatalf("creating temp GOPATH: %v", err)
		}
		env := append(os.Environ(), "GOPATH="+gopath)

		// Install gopls.
		log.Printf("installing gopls@%s\n", version)
		cmd := exec.Command("go", "install", fmt.Sprintf("golang.org/x/tools/gopls@%s", version))
		cmd.Env = env
		if output, err := cmd.CombinedOutput(); err != nil {
			log.Fatalf("installing gopls: %v:\n%s", err, string(output))
		}

		// Clean the modcache, otherwise we'll get errors when trying to remove the
		// temp directory.
		cleanCmd := exec.Command("go", "clean", "-modcache")
		cleanCmd.Env = env
		if output, err := cleanCmd.CombinedOutput(); err != nil {
			log.Fatalf("cleaning up temp GOPATH: %v\n%s", err, string(output))
		}

		// Confirm that the resulting path now exists.
		if _, err := os.Stat(goplsPath); err != nil {
			log.Fatalf("os.Stat(goplsPath): %v", err)
		}
	})
	return goplsPath
}

// A SidecarServer starts (and connects to) a separate gopls process at the
// given path.
type SidecarServer struct {
	goplsPath string
}

// Connect creates new io.Pipes and binds them to the underlying StreamServer.
func (s *SidecarServer) Connect(ctx context.Context) jsonrpc2.Conn {
	cmd := exec.CommandContext(ctx, s.goplsPath, "serve")

	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}
	cmd.Stderr = os.Stdout
	if err := cmd.Start(); err != nil {
		log.Fatalf("starting gopls: %v", err)
	}

	go cmd.Wait() // to free resources; error is ignored

	clientStream := jsonrpc2.NewHeaderStream(fakenet.NewConn("stdio", stdout, stdin))
	clientConn := jsonrpc2.NewConn(clientStream)
	return clientConn
}
