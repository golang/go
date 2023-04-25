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

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/cmd"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/fakenet"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/tool"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

var (
	goplsPath = flag.String("gopls_path", "", "if set, use this gopls for testing; incompatible with -gopls_commit")

	installGoplsOnce sync.Once // guards installing gopls at -gopls_commit
	goplsCommit      = flag.String("gopls_commit", "", "if set, install and use gopls at this commit for testing; incompatible with -gopls_path")

	cpuProfile   = flag.String("gopls_cpuprofile", "", "if set, the cpu profile file suffix; see \"Profiling\" in the package doc")
	memProfile   = flag.String("gopls_memprofile", "", "if set, the mem profile file suffix; see \"Profiling\" in the package doc")
	allocProfile = flag.String("gopls_allocprofile", "", "if set, the alloc profile file suffix; see \"Profiling\" in the package doc")
	trace        = flag.String("gopls_trace", "", "if set, the trace file suffix; see \"Profiling\" in the package doc")

	// If non-empty, tempDir is a temporary working dir that was created by this
	// test suite.
	makeTempDirOnce sync.Once // guards creation of the temp dir
	tempDir         string
)

// if runAsGopls is "true", run the gopls command instead of the testing.M.
const runAsGopls = "_GOPLS_BENCH_RUN_AS_GOPLS"

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	if os.Getenv(runAsGopls) == "true" {
		tool.Main(context.Background(), cmd.New("gopls", "", nil, hooks.Options), os.Args[1:])
		os.Exit(0)
	}
	event.SetExporter(nil) // don't log to stderr
	code := m.Run()
	if err := cleanup(); err != nil {
		fmt.Fprintf(os.Stderr, "cleaning up after benchmarks: %v\n", err)
		if code == 0 {
			code = 1
		}
	}
	os.Exit(code)
}

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

// shallowClone performs a shallow clone of repo into dir at the given
// 'commitish' ref (any commit reference understood by git).
//
// The directory dir must not already exist.
func shallowClone(dir, repo, commitish string) error {
	if err := os.Mkdir(dir, 0750); err != nil {
		return fmt.Errorf("creating dir for %s: %v", repo, err)
	}

	// Set a timeout for git fetch. If this proves flaky, it can be removed.
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	// Use a shallow fetch to download just the relevant commit.
	shInit := fmt.Sprintf("git init && git fetch --depth=1 %q %q && git checkout FETCH_HEAD", repo, commitish)
	initCmd := exec.CommandContext(ctx, "/bin/sh", "-c", shInit)
	initCmd.Dir = dir
	if output, err := initCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("checking out %s: %v\n%s", repo, err, output)
	}
	return nil
}

// connectEditor connects a fake editor session in the given dir, using the
// given editor config.
func connectEditor(dir string, config fake.EditorConfig, ts servertest.Connector) (*fake.Sandbox, *fake.Editor, *Awaiter, error) {
	s, err := fake.NewSandbox(&fake.SandboxConfig{
		Workdir: dir,
		GOPROXY: "https://proxy.golang.org",
	})
	if err != nil {
		return nil, nil, nil, err
	}

	a := NewAwaiter(s.Workdir)
	const skipApplyEdits = false
	editor, err := fake.NewEditor(s, config).Connect(context.Background(), ts, a.Hooks(), skipApplyEdits)
	if err != nil {
		return nil, nil, nil, err
	}

	return s, editor, a, nil
}

// newGoplsServer returns a connector that connects to a new gopls process.
func newGoplsServer(name string) (servertest.Connector, error) {
	if *goplsPath != "" && *goplsCommit != "" {
		panic("can't set both -gopls_path and -gopls_commit")
	}
	var (
		goplsPath = *goplsPath
		env       []string
	)
	if *goplsCommit != "" {
		goplsPath = getInstalledGopls()
	}
	if goplsPath == "" {
		var err error
		goplsPath, err = os.Executable()
		if err != nil {
			return nil, err
		}
		env = []string{fmt.Sprintf("%s=true", runAsGopls)}
	}
	var args []string
	if *cpuProfile != "" {
		args = append(args, fmt.Sprintf("-profile.cpu=%s", name+"."+*cpuProfile))
	}
	if *memProfile != "" {
		args = append(args, fmt.Sprintf("-profile.mem=%s", name+"."+*memProfile))
	}
	if *allocProfile != "" {
		args = append(args, fmt.Sprintf("-profile.alloc=%s", name+"."+*allocProfile))
	}
	if *trace != "" {
		args = append(args, fmt.Sprintf("-profile.trace=%s", name+"."+*trace))
	}
	return &SidecarServer{
		goplsPath: goplsPath,
		env:       env,
		args:      args,
	}, nil
}

// getInstalledGopls builds gopls at the given -gopls_commit, returning the
// path to the gopls binary.
func getInstalledGopls() string {
	if *goplsCommit == "" {
		panic("must provide -gopls_commit")
	}
	toolsDir := filepath.Join(getTempDir(), "gopls_build")
	goplsPath := filepath.Join(toolsDir, "gopls", "gopls")

	installGoplsOnce.Do(func() {
		log.Printf("installing gopls: checking out x/tools@%s into %s\n", *goplsCommit, toolsDir)
		if err := shallowClone(toolsDir, "https://go.googlesource.com/tools", *goplsCommit); err != nil {
			log.Fatal(err)
		}

		log.Println("installing gopls: building...")
		bld := exec.Command("go", "build", ".")
		bld.Dir = filepath.Join(toolsDir, "gopls")
		if output, err := bld.CombinedOutput(); err != nil {
			log.Fatalf("building gopls: %v\n%s", err, output)
		}

		// Confirm that the resulting path now exists.
		if _, err := os.Stat(goplsPath); err != nil {
			log.Fatalf("os.Stat(%s): %v", goplsPath, err)
		}
	})
	return goplsPath
}

// A SidecarServer starts (and connects to) a separate gopls process at the
// given path.
type SidecarServer struct {
	goplsPath string
	env       []string // additional environment bindings
	args      []string // command-line arguments
}

// Connect creates new io.Pipes and binds them to the underlying StreamServer.
//
// It implements the servertest.Connector interface.
func (s *SidecarServer) Connect(ctx context.Context) jsonrpc2.Conn {
	// Note: don't use CommandContext here, as we want gopls to exit gracefully
	// in order to write out profile data.
	//
	// We close the connection on context cancelation below.
	cmd := exec.Command(s.goplsPath, s.args...)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), s.env...)
	if err := cmd.Start(); err != nil {
		log.Fatalf("starting gopls: %v", err)
	}

	go func() {
		// If we don't log.Fatal here, benchmarks may hang indefinitely if gopls
		// exits abnormally.
		//
		// TODO(rfindley): ideally we would shut down the connection gracefully,
		// but that doesn't currently work.
		if err := cmd.Wait(); err != nil {
			log.Fatalf("gopls invocation failed with error: %v", err)
		}
	}()

	clientStream := jsonrpc2.NewHeaderStream(fakenet.NewConn("stdio", stdout, stdin))
	clientConn := jsonrpc2.NewConn(clientStream)

	go func() {
		select {
		case <-ctx.Done():
			clientConn.Close()
			clientStream.Close()
		case <-clientConn.Done():
		}
	}()

	return clientConn
}
