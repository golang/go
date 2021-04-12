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
	"path/filepath"
	"runtime/pprof"
	"strings"
	"sync"
	"testing"
	"time"

	exec "golang.org/x/sys/execabs"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/lsprpc"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
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
	// Experimental enables all of the experimental configurations that are
	// being developed. Currently, it enables the workspace module.
	Experimental
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
	OptionsHook              func(*source.Options)

	mu        sync.Mutex
	ts        *servertest.TCPServer
	socketDir string
	// closers is a queue of clean-up functions to run at the end of the entire
	// test suite.
	closers []io.Closer
}

type runConfig struct {
	editor      fake.EditorConfig
	sandbox     fake.SandboxConfig
	modes       Mode
	timeout     time.Duration
	debugAddr   string
	skipLogs    bool
	skipHooks   bool
	optionsHook func(*source.Options)
}

func (r *Runner) defaultConfig() *runConfig {
	return &runConfig{
		modes:       r.DefaultModes,
		timeout:     r.Timeout,
		optionsHook: r.OptionsHook,
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

// Timeout configures a custom timeout for this test run.
func Timeout(d time.Duration) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.timeout = d
	})
}

// ProxyFiles configures a file proxy using the given txtar-encoded string.
func ProxyFiles(txt string) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.sandbox.ProxyFiles = txt
	})
}

// Modes configures the execution modes that the test should run in.
func Modes(modes Mode) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.modes = modes
	})
}

// Options configures the various server and user options.
func Options(hook func(*source.Options)) RunOption {
	return optionSetter(func(opts *runConfig) {
		old := opts.optionsHook
		opts.optionsHook = func(o *source.Options) {
			if old != nil {
				old(o)
			}
			hook(o)
		}
	})
}

func SendPID() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.editor.SendPID = true
	})
}

// EditorConfig is a RunOption option that configured the regtest editor.
type EditorConfig fake.EditorConfig

func (c EditorConfig) set(opts *runConfig) {
	opts.editor = fake.EditorConfig(c)
}

// WorkspaceFolders configures the workdir-relative workspace folders to send
// to the LSP server. By default the editor sends a single workspace folder
// corresponding to the workdir root. To explicitly configure no workspace
// folders, use WorkspaceFolders with no arguments.
func WorkspaceFolders(relFolders ...string) RunOption {
	if len(relFolders) == 0 {
		// Use an empty non-nil slice to signal explicitly no folders.
		relFolders = []string{}
	}
	return optionSetter(func(opts *runConfig) {
		opts.editor.WorkspaceFolders = relFolders
	})
}

// InGOPATH configures the workspace working directory to be GOPATH, rather
// than a separate working directory for use with modules.
func InGOPATH() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.sandbox.InGoPath = true
	})
}

// DebugAddress configures a debug server bound to addr. This option is
// currently only supported when executing in Singleton mode. It is intended to
// be used for long-running stress tests.
func DebugAddress(addr string) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.debugAddr = addr
	})
}

// SkipLogs skips the buffering of logs during test execution. It is intended
// for long-running stress tests.
func SkipLogs() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.skipLogs = true
	})
}

// InExistingDir runs the test in a pre-existing directory. If set, no initial
// files may be passed to the runner. It is intended for long-running stress
// tests.
func InExistingDir(dir string) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.sandbox.Workdir = dir
	})
}

// SkipHooks allows for disabling the test runner's client hooks that are used
// for instrumenting expectations (tracking diagnostics, logs, work done,
// etc.). It is intended for performance-sensitive stress tests or benchmarks.
func SkipHooks(skip bool) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.skipHooks = skip
	})
}

// GOPROXY configures the test environment to have an explicit proxy value.
// This is intended for stress tests -- to ensure their isolation, regtests
// should instead use WithProxyFiles.
func GOPROXY(goproxy string) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.sandbox.GOPROXY = goproxy
	})
}

// LimitWorkspaceScope sets the LimitWorkspaceScope configuration.
func LimitWorkspaceScope() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.editor.LimitWorkspaceScope = true
	})
}

type TestFunc func(t *testing.T, env *Env)

// Run executes the test function in the default configured gopls execution
// modes. For each a test run, a new workspace is created containing the
// un-txtared files specified by filedata.
func (r *Runner) Run(t *testing.T, files string, test TestFunc, opts ...RunOption) {
	t.Helper()
	checkBuilder(t)

	tests := []struct {
		name      string
		mode      Mode
		getServer func(context.Context, *testing.T, func(*source.Options)) jsonrpc2.StreamServer
	}{
		{"singleton", Singleton, singletonServer},
		{"forwarded", Forwarded, r.forwardedServer},
		{"separate_process", SeparateProcess, r.separateProcessServer},
		{"experimental_workspace_module", Experimental, experimentalWorkspaceModule},
	}

	for _, tc := range tests {
		tc := tc
		config := r.defaultConfig()
		for _, opt := range opts {
			opt.set(config)
		}
		if config.modes&tc.mode == 0 {
			continue
		}
		if config.debugAddr != "" && tc.mode != Singleton {
			// Debugging is useful for running stress tests, but since the daemon has
			// likely already been started, it would be too late to debug.
			t.Fatalf("debugging regtest servers only works in Singleton mode, "+
				"got debug addr %q and mode %v", config.debugAddr, tc.mode)
		}

		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), config.timeout)
			defer cancel()
			ctx = debug.WithInstance(ctx, "", "off")
			if config.debugAddr != "" {
				di := debug.GetInstance(ctx)
				di.Serve(ctx, config.debugAddr)
				di.MonitorMemory(ctx)
			}

			rootDir := filepath.Join(r.TempDir, filepath.FromSlash(t.Name()))
			if err := os.MkdirAll(rootDir, 0755); err != nil {
				t.Fatal(err)
			}
			config.sandbox.Files = files
			config.sandbox.RootDir = rootDir
			sandbox, err := fake.NewSandbox(&config.sandbox)
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
			ss := tc.getServer(ctx, t, config.optionsHook)
			framer := jsonrpc2.NewRawStream
			ls := &loggingFramer{}
			if !config.skipLogs {
				framer = ls.framer(jsonrpc2.NewRawStream)
			}
			ts := servertest.NewPipeServer(ctx, ss, framer)
			env := NewEnv(ctx, t, sandbox, ts, config.editor, !config.skipHooks)
			defer func() {
				if t.Failed() && r.PrintGoroutinesOnFailure {
					pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
				}
				if t.Failed() || testing.Verbose() {
					ls.printBuffers(t.Name(), os.Stderr)
				}
				env.CloseEditor()
			}()
			// Always await the initial workspace load.
			env.Await(InitialWorkspaceLoad)
			test(t, env)
		})
	}
}

// longBuilders maps builders that are skipped when -short is set to a
// (possibly empty) justification.
var longBuilders = map[string]string{
	"openbsd-amd64-64":        "golang.org/issues/42789",
	"openbsd-386-64":          "golang.org/issues/42789",
	"openbsd-386-68":          "golang.org/issues/42789",
	"openbsd-amd64-68":        "golang.org/issues/42789",
	"darwin-amd64-10_12":      "",
	"freebsd-amd64-race":      "",
	"illumos-amd64":           "",
	"netbsd-arm-bsiegert":     "",
	"solaris-amd64-oraclerel": "",
	"windows-arm-zx2c4":       "",
}

func checkBuilder(t *testing.T) {
	t.Helper()
	builder := os.Getenv("GO_BUILDER_NAME")
	if reason, ok := longBuilders[builder]; ok && testing.Short() {
		if reason != "" {
			t.Skipf("Skipping %s with -short due to %s", builder, reason)
		} else {
			t.Skipf("Skipping %s with -short", builder)
		}
	}
}

type loggingFramer struct {
	mu  sync.Mutex
	buf *safeBuffer
}

// safeBuffer is a threadsafe buffer for logs.
type safeBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *safeBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (s *loggingFramer) framer(f jsonrpc2.Framer) jsonrpc2.Framer {
	return func(nc net.Conn) jsonrpc2.Stream {
		s.mu.Lock()
		framed := false
		if s.buf == nil {
			s.buf = &safeBuffer{buf: bytes.Buffer{}}
			framed = true
		}
		s.mu.Unlock()
		stream := f(nc)
		if framed {
			return protocol.LoggingStream(stream, s.buf)
		}
		return stream
	}
}

func (s *loggingFramer) printBuffers(testname string, w io.Writer) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.buf == nil {
		return
	}
	fmt.Fprintf(os.Stderr, "#### Start Gopls Test Logs for %q\n", testname)
	s.buf.mu.Lock()
	io.Copy(w, &s.buf.buf)
	s.buf.mu.Unlock()
	fmt.Fprintf(os.Stderr, "#### End Gopls Test Logs for %q\n", testname)
}

func singletonServer(ctx context.Context, t *testing.T, optsHook func(*source.Options)) jsonrpc2.StreamServer {
	return lsprpc.NewStreamServer(cache.New(optsHook), false)
}

func experimentalWorkspaceModule(_ context.Context, t *testing.T, optsHook func(*source.Options)) jsonrpc2.StreamServer {
	options := func(o *source.Options) {
		optsHook(o)
		o.ExperimentalWorkspaceModule = true
	}
	return lsprpc.NewStreamServer(cache.New(options), false)
}

func (r *Runner) forwardedServer(ctx context.Context, t *testing.T, optsHook func(*source.Options)) jsonrpc2.StreamServer {
	ts := r.getTestServer(optsHook)
	return lsprpc.NewForwarder("tcp", ts.Addr)
}

// getTestServer gets the shared test server instance to connect to, or creates
// one if it doesn't exist.
func (r *Runner) getTestServer(optsHook func(*source.Options)) *servertest.TCPServer {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.ts == nil {
		ctx := context.Background()
		ctx = debug.WithInstance(ctx, "", "off")
		ss := lsprpc.NewStreamServer(cache.New(optsHook), false)
		r.ts = servertest.NewTCPServer(ctx, ss, nil)
	}
	return r.ts
}

func (r *Runner) separateProcessServer(ctx context.Context, t *testing.T, optsHook func(*source.Options)) jsonrpc2.StreamServer {
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
