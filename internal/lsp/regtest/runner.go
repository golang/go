// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/pprof"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/lsprpc"
	"golang.org/x/tools/internal/lsp/protocol"
)

// Mode is a bitmask that defines for which execution modes a test should run.
type Mode int

const (
	// Singleton mode uses a separate in-process gopls instance for each test,
	// and communicates over pipes to mimic the gopls sidecar execution mode,
	// which communicates over stdin/stderr.
	Singleton Mode = 1 << iota

	// Forwarded forwards connections to a shared in-process gopls instance.
	Forwarded
	// SeparateProcess forwards connection to a shared separate gopls process.
	SeparateProcess
	// NormalModes are the global default execution modes, when unmodified by
	// test flags or by individual test options.
	NormalModes = Singleton | Forwarded
)

// A Runner runs tests in gopls execution environments, as specified by its
// modes. For modes that share state (for example, a shared cache or common
// remote), any tests that execute on the same Runner will share the same
// state.
type Runner struct {
	DefaultModes             Mode
	Timeout                  time.Duration
	GoplsPath                string
	AlwaysPrintLogs          bool
	PrintGoroutinesOnFailure bool

	mu        sync.Mutex
	ts        *servertest.TCPServer
	socketDir string
	// closers is a queue of clean-up functions to run at the end of the entire
	// test suite.
	closers []io.Closer
}

type runConfig struct {
	editorConfig fake.EditorConfig
	modes        Mode
	proxyTxt     string
	timeout      time.Duration
	skipCleanup  bool
	gopath       bool
}

func (r *Runner) defaultConfig() *runConfig {
	return &runConfig{
		modes:   r.DefaultModes,
		timeout: r.Timeout,
	}
}

// A RunOption augments the behavior of the test runner.
type RunOption interface {
	set(*runConfig)
}

type optionSetter func(*runConfig)

func (f optionSetter) set(opts *runConfig) {
	f(opts)
}

// WithTimeout configures a custom timeout for this test run.
func WithTimeout(d time.Duration) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.timeout = d
	})
}

// WithProxy configures a file proxy using the given txtar-encoded string.
func WithProxy(txt string) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.proxyTxt = txt
	})
}

// WithModes configures the execution modes that the test should run in.
func WithModes(modes Mode) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.modes = modes
	})
}

// WithEditorConfig configures the editors LSP session.
func WithEditorConfig(config fake.EditorConfig) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.editorConfig = config
	})
}

// InGOPATH configures the workspace working directory to be GOPATH, rather
// than a separate working directory for use with modules.
func InGOPATH() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.gopath = true
	})
}

// SkipCleanup is used only for debugging: is skips cleaning up the tests state
// after completion.
func SkipCleanup() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.skipCleanup = true
	})
}

// Run executes the test function in the default configured gopls execution
// modes. For each a test run, a new workspace is created containing the
// un-txtared files specified by filedata.
func (r *Runner) Run(t *testing.T, filedata string, test func(t *testing.T, e *Env), opts ...RunOption) {
	t.Helper()
	config := r.defaultConfig()
	for _, opt := range opts {
		opt.set(config)
	}

	tests := []struct {
		name      string
		mode      Mode
		getServer func(context.Context, *testing.T) jsonrpc2.StreamServer
	}{
		{"singleton", Singleton, singletonServer},
		{"forwarded", Forwarded, r.forwardedServer},
		{"separate_process", SeparateProcess, r.separateProcessServer},
	}

	for _, tc := range tests {
		tc := tc
		if config.modes&tc.mode == 0 {
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			ctx, cancel := context.WithTimeout(context.Background(), config.timeout)
			defer cancel()
			ctx = debug.WithInstance(ctx, "", "")

			sandbox, err := fake.NewSandbox("regtest", filedata, config.proxyTxt, config.gopath)
			if err != nil {
				t.Fatal(err)
			}
			// Deferring the closure of ws until the end of the entire test suite
			// has, in testing, given the LSP server time to properly shutdown and
			// release any file locks held in workspace, which is a problem on
			// Windows. This may still be flaky however, and in the future we need a
			// better solution to ensure that all Go processes started by gopls have
			// exited before we clean up.
			if config.skipCleanup {
				defer func() {
					t.Logf("Skipping workspace cleanup: running in %s", sandbox.Workdir.RootURI())
				}()
			} else {
				r.AddCloser(sandbox)
			}
			ss := tc.getServer(ctx, t)
			ls := &loggingServer{delegate: ss}
			ts := servertest.NewPipeServer(ctx, ls)
			defer func() {
				ts.Close()
			}()
			env := NewEnv(ctx, t, sandbox, ts, config.editorConfig)
			defer func() {
				if t.Failed() && r.PrintGoroutinesOnFailure {
					pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
				}
				if t.Failed() || r.AlwaysPrintLogs {
					ls.printBuffers(t.Name(), os.Stderr)
				}
				if err := env.Editor.Shutdown(ctx); err != nil {
					panic(err)
				}
			}()
			test(t, env)
		})
	}
}

type loggingServer struct {
	delegate jsonrpc2.StreamServer

	mu      sync.Mutex
	buffers []*bytes.Buffer
}

func (s *loggingServer) ServeStream(ctx context.Context, stream jsonrpc2.Stream) error {
	s.mu.Lock()
	var buf bytes.Buffer
	s.buffers = append(s.buffers, &buf)
	s.mu.Unlock()
	logStream := protocol.LoggingStream(stream, &buf)
	return s.delegate.ServeStream(ctx, logStream)
}

func (s *loggingServer) printBuffers(testname string, w io.Writer) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i, buf := range s.buffers {
		fmt.Fprintf(os.Stderr, "#### Start Gopls Test Logs %d of %d for %q\n", i+1, len(s.buffers), testname)
		io.Copy(w, buf)
		fmt.Fprintf(os.Stderr, "#### End Gopls Test Logs %d of %d for %q\n", i+1, len(s.buffers), testname)
	}
}

func singletonServer(ctx context.Context, t *testing.T) jsonrpc2.StreamServer {
	return lsprpc.NewStreamServer(cache.New(ctx, nil))
}

func (r *Runner) forwardedServer(ctx context.Context, t *testing.T) jsonrpc2.StreamServer {
	ts := r.getTestServer()
	return lsprpc.NewForwarder("tcp", ts.Addr)
}

// getTestServer gets the test server instance to connect to, or creates one if
// it doesn't exist.
func (r *Runner) getTestServer() *servertest.TCPServer {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ts == nil {
		ctx := context.Background()
		ctx = debug.WithInstance(ctx, "", "")
		ss := lsprpc.NewStreamServer(cache.New(ctx, nil))
		r.ts = servertest.NewTCPServer(context.Background(), ss)
	}
	return r.ts
}

func (r *Runner) separateProcessServer(ctx context.Context, t *testing.T) jsonrpc2.StreamServer {
	// TODO(rfindley): can we use the autostart behavior here, instead of
	// pre-starting the remote?
	socket := r.getRemoteSocket(t)
	return lsprpc.NewForwarder("unix", socket)
}

// runTestAsGoplsEnvvar triggers TestMain to run gopls instead of running
// tests. It's a trick to allow tests to find a binary to use to start a gopls
// subprocess.
const runTestAsGoplsEnvvar = "_GOPLS_TEST_BINARY_RUN_AS_GOPLS"

func (r *Runner) getRemoteSocket(t *testing.T) string {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	const daemonFile = "gopls-test-daemon"
	if r.socketDir != "" {
		return filepath.Join(r.socketDir, daemonFile)
	}

	if r.GoplsPath == "" {
		t.Fatal("cannot run tests with a separate process unless a path to a gopls binary is configured")
	}
	var err error
	r.socketDir, err = ioutil.TempDir("", "gopls-regtests")
	if err != nil {
		t.Fatalf("creating tempdir: %v", err)
	}
	socket := filepath.Join(r.socketDir, daemonFile)
	args := []string{"serve", "-listen", "unix;" + socket, "-listen.timeout", "10s"}
	cmd := exec.Command(r.GoplsPath, args...)
	cmd.Env = append(os.Environ(), runTestAsGoplsEnvvar+"=true")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	go func() {
		if err := cmd.Run(); err != nil {
			panic(fmt.Sprintf("error running external gopls: %v\nstderr:\n%s", err, stderr.String()))
		}
	}()
	return socket
}

// AddCloser schedules a closer to be closed at the end of the test run. This
// is useful for Windows in particular, as
func (r *Runner) AddCloser(closer io.Closer) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closers = append(r.closers, closer)
}

// Close cleans up resource that have been allocated to this workspace.
func (r *Runner) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var errmsgs []string
	if r.ts != nil {
		if err := r.ts.Close(); err != nil {
			errmsgs = append(errmsgs, err.Error())
		}
	}
	if r.socketDir != "" {
		if err := os.RemoveAll(r.socketDir); err != nil {
			errmsgs = append(errmsgs, err.Error())
		}
	}
	for _, closer := range r.closers {
		if err := closer.Close(); err != nil {
			errmsgs = append(errmsgs, err.Error())
		}
	}
	if len(errmsgs) > 0 {
		return fmt.Errorf("errors closing the test runner:\n\t%s", strings.Join(errmsgs, "\n\t"))
	}
	return nil
}
