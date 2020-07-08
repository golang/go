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
	"net"
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
	PrintGoroutinesOnFailure bool
	TempDir                  string
	SkipCleanup              bool

	mu        sync.Mutex
	ts        *servertest.TCPServer
	socketDir string
	// closers is a queue of clean-up functions to run at the end of the entire
	// test suite.
	closers []io.Closer
}

type runConfig struct {
	editorConfig            fake.EditorConfig
	modes                   Mode
	proxyTxt                string
	timeout                 time.Duration
	gopath                  bool
	withoutWorkspaceFolders bool
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

// WithEditorConfig configures the editor's LSP session.
func WithEditorConfig(config fake.EditorConfig) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.editorConfig = config
	})
}

// WithoutWorkspaceFolders prevents workspace folders from being sent as part
// of the sandbox's initialization. It is used to simulate opening a single
// file in the editor, without a workspace root. In that case, the client sends
// neither workspace folders nor a root URI.
func WithoutWorkspaceFolders() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.withoutWorkspaceFolders = false
	})
}

// InGOPATH configures the workspace working directory to be GOPATH, rather
// than a separate working directory for use with modules.
func InGOPATH() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.gopath = true
	})
}

type TestFunc func(t *testing.T, env *Env)

// Run executes the test function in the default configured gopls execution
// modes. For each a test run, a new workspace is created containing the
// un-txtared files specified by filedata.
func (r *Runner) Run(t *testing.T, filedata string, test TestFunc, opts ...RunOption) {
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
			ctx, cancel := context.WithTimeout(context.Background(), config.timeout)
			defer cancel()
			ctx = debug.WithInstance(ctx, "", "")

			tempDir := filepath.Join(r.TempDir, filepath.FromSlash(t.Name()))
			if err := os.MkdirAll(tempDir, 0755); err != nil {
				t.Fatal(err)
			}

			sandbox, err := fake.NewSandbox(tempDir, filedata, config.proxyTxt, config.gopath, config.withoutWorkspaceFolders)
			if err != nil {
				t.Fatal(err)
			}
			// Deferring the closure of ws until the end of the entire test suite
			// has, in testing, given the LSP server time to properly shutdown and
			// release any file locks held in workspace, which is a problem on
			// Windows. This may still be flaky however, and in the future we need a
			// better solution to ensure that all Go processes started by gopls have
			// exited before we clean up.
			r.AddCloser(sandbox)
			ss := tc.getServer(ctx, t)
			ls := &loggingFramer{}
			framer := ls.framer(jsonrpc2.NewRawStream)
			ts := servertest.NewPipeServer(ctx, ss, framer)
			env := NewEnv(ctx, t, sandbox, ts, config.editorConfig)
			defer func() {
				if t.Failed() && r.PrintGoroutinesOnFailure {
					pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
				}
				if t.Failed() || testing.Verbose() {
					ls.printBuffers(t.Name(), os.Stderr)
				}
				env.CloseEditor()
			}()
			test(t, env)
		})
	}
}

type loggingFramer struct {
	mu      sync.Mutex
	buffers []*bytes.Buffer
}

func (s *loggingFramer) framer(f jsonrpc2.Framer) jsonrpc2.Framer {
	return func(nc net.Conn) jsonrpc2.Stream {
		s.mu.Lock()
		var buf bytes.Buffer
		s.buffers = append(s.buffers, &buf)
		s.mu.Unlock()
		stream := f(nc)
		return protocol.LoggingStream(stream, &buf)
	}
}

func (s *loggingFramer) printBuffers(testname string, w io.Writer) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i, buf := range s.buffers {
		fmt.Fprintf(os.Stderr, "#### Start Gopls Test Logs %d of %d for %q\n", i+1, len(s.buffers), testname)
		// Re-buffer buf to avoid a data rate (io.Copy mutates src).
		writeBuf := bytes.NewBuffer(buf.Bytes())
		io.Copy(w, writeBuf)
		fmt.Fprintf(os.Stderr, "#### End Gopls Test Logs %d of %d for %q\n", i+1, len(s.buffers), testname)
	}
}

func singletonServer(ctx context.Context, t *testing.T) jsonrpc2.StreamServer {
	return lsprpc.NewStreamServer(cache.New(ctx, nil), false)
}

func (r *Runner) forwardedServer(ctx context.Context, t *testing.T) jsonrpc2.StreamServer {
	ts := r.getTestServer()
	return lsprpc.NewForwarder("tcp", ts.Addr)
}

// getTestServer gets the shared test server instance to connect to, or creates
// one if it doesn't exist.
func (r *Runner) getTestServer() *servertest.TCPServer {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ts == nil {
		ctx := context.Background()
		ctx = debug.WithInstance(ctx, "", "")
		ss := lsprpc.NewStreamServer(cache.New(ctx, nil), false)
		r.ts = servertest.NewTCPServer(ctx, ss, nil)
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
	r.socketDir, err = ioutil.TempDir(r.TempDir, "gopls-regtest-socket")
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
	if !r.SkipCleanup {
		for _, closer := range r.closers {
			if err := closer.Close(); err != nil {
				errmsgs = append(errmsgs, err.Error())
			}
		}
		if err := os.RemoveAll(r.TempDir); err != nil {
			errmsgs = append(errmsgs, err.Error())
		}
	}
	if len(errmsgs) > 0 {
		return fmt.Errorf("errors closing the test runner:\n\t%s", strings.Join(errmsgs, "\n\t"))
	}
	return nil
}
