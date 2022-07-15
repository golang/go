// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"bytes"
	"context"
	"fmt"
	"go/token"
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
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/internal/xcontext"
)

// Mode is a bitmask that defines for which execution modes a test should run.
//
// Each mode controls several aspects of gopls' configuration:
//   - Which server options to use for gopls sessions
//   - Whether to use a shared cache
//   - Whether to use a shared server
//   - Whether to run the server in-process or in a separate process
//
// The behavior of each mode with respect to these aspects is summarized below.
// TODO(rfindley, cleanup): rather than using arbitrary names for these modes,
// we can compose them explicitly out of the features described here, allowing
// individual tests more freedom in constructing problematic execution modes.
// For example, a test could assert on a certain behavior when running with
// experimental options on a separate process. Moreover, we could unify 'Modes'
// with 'Options', and use RunMultiple rather than a hard-coded loop through
// modes.
//
// Mode            | Options      | Shared Cache? | Shared Server? | In-process?
// ---------------------------------------------------------------------------
// Default         | Default      | Y             | N              | Y
// Forwarded       | Default      | Y             | Y              | Y
// SeparateProcess | Default      | Y             | Y              | N
// Experimental    | Experimental | N             | N              | Y
type Mode int

const (
	// Default mode runs gopls with the default options, communicating over pipes
	// to emulate the lsp sidecar execution mode, which communicates over
	// stdin/stdout.
	//
	// It uses separate servers for each test, but a shared cache, to avoid
	// duplicating work when processing GOROOT.
	Default Mode = 1 << iota

	// Forwarded uses the default options, but forwards connections to a shared
	// in-process gopls server.
	Forwarded

	// SeparateProcess uses the default options, but forwards connection to an
	// external gopls daemon.
	SeparateProcess

	// Experimental enables all of the experimental configurations that are
	// being developed, and runs gopls in sidecar mode.
	//
	// It uses a separate cache for each test, to exercise races that may only
	// appear with cache misses.
	Experimental
)

func (m Mode) String() string {
	switch m {
	case Default:
		return "default"
	case Forwarded:
		return "forwarded"
	case SeparateProcess:
		return "separate process"
	case Experimental:
		return "experimental"
	default:
		return "unknown mode"
	}
}

// A Runner runs tests in gopls execution environments, as specified by its
// modes. For modes that share state (for example, a shared cache or common
// remote), any tests that execute on the same Runner will share the same
// state.
type Runner struct {
	// Configuration
	DefaultModes             Mode                  // modes to run for each test
	Timeout                  time.Duration         // per-test timeout, if set
	PrintGoroutinesOnFailure bool                  // whether to dump goroutines on test failure
	SkipCleanup              bool                  // if set, don't delete test data directories when the test exits
	OptionsHook              func(*source.Options) // if set, use these options when creating gopls sessions

	// Immutable state shared across test invocations
	goplsPath string         // path to the gopls executable (for SeparateProcess mode)
	tempDir   string         // shared parent temp directory
	fset      *token.FileSet // shared FileSet
	store     *memoize.Store // shared store

	// Lazily allocated resources
	mu        sync.Mutex
	ts        *servertest.TCPServer
	socketDir string
}

type runConfig struct {
	editor           fake.EditorConfig
	sandbox          fake.SandboxConfig
	modes            Mode
	noDefaultTimeout bool
	debugAddr        string
	skipLogs         bool
	skipHooks        bool
}

// A RunOption augments the behavior of the test runner.
type RunOption interface {
	set(*runConfig)
}

type optionSetter func(*runConfig)

func (f optionSetter) set(opts *runConfig) {
	f(opts)
}

// NoDefaultTimeout removes the timeout set by the -regtest_timeout flag, for
// individual tests that are expected to run longer than is reasonable for
// ordinary regression tests.
func NoDefaultTimeout() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.noDefaultTimeout = true
	})
}

// ProxyFiles configures a file proxy using the given txtar-encoded string.
func ProxyFiles(txt string) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.sandbox.ProxyFiles = fake.UnpackTxt(txt)
	})
}

// Modes configures the execution modes that the test should run in.
//
// By default, modes are configured by the test runner. If this option is set,
// it overrides the set of default modes and the test runs in exactly these
// modes.
func Modes(modes Mode) RunOption {
	return optionSetter(func(opts *runConfig) {
		if opts.modes != 0 {
			panic("modes set more than once")
		}
		opts.modes = modes
	})
}

// WindowsLineEndings configures the editor to use windows line endings.
func WindowsLineEndings() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.editor.WindowsLineEndings = true
	})
}

// Settings is a RunOption that sets user-provided configuration for the LSP
// server.
//
// As a special case, the env setting must not be provided via Settings: use
// EnvVars instead.
type Settings map[string]interface{}

func (s Settings) set(opts *runConfig) {
	if opts.editor.Settings == nil {
		opts.editor.Settings = make(map[string]interface{})
	}
	for k, v := range s {
		opts.editor.Settings[k] = v
	}
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

// EnvVars sets environment variables for the LSP session. When applying these
// variables to the session, the special string $SANDBOX_WORKDIR is replaced by
// the absolute path to the sandbox working directory.
type EnvVars map[string]string

func (e EnvVars) set(opts *runConfig) {
	if opts.editor.Env == nil {
		opts.editor.Env = make(map[string]string)
	}
	for k, v := range e {
		opts.editor.Env[k] = v
	}
}

// InGOPATH configures the workspace working directory to be GOPATH, rather
// than a separate working directory for use with modules.
func InGOPATH() RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.sandbox.InGoPath = true
	})
}

// DebugAddress configures a debug server bound to addr. This option is
// currently only supported when executing in Default mode. It is intended to
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

type TestFunc func(t *testing.T, env *Env)

// Run executes the test function in the default configured gopls execution
// modes. For each a test run, a new workspace is created containing the
// un-txtared files specified by filedata.
func (r *Runner) Run(t *testing.T, files string, test TestFunc, opts ...RunOption) {
	// TODO(rfindley): this function has gotten overly complicated, and warrants
	// refactoring.
	t.Helper()
	checkBuilder(t)

	tests := []struct {
		name      string
		mode      Mode
		getServer func(*testing.T, func(*source.Options)) jsonrpc2.StreamServer
	}{
		{"default", Default, r.defaultServer},
		{"forwarded", Forwarded, r.forwardedServer},
		{"separate_process", SeparateProcess, r.separateProcessServer},
		{"experimental", Experimental, r.experimentalServer},
	}

	for _, tc := range tests {
		tc := tc
		var config runConfig
		for _, opt := range opts {
			opt.set(&config)
		}
		modes := r.DefaultModes
		if config.modes != 0 {
			modes = config.modes
		}
		if modes&tc.mode == 0 {
			continue
		}

		if config.debugAddr != "" && tc.mode != Default {
			// Debugging is useful for running stress tests, but since the daemon has
			// likely already been started, it would be too late to debug.
			t.Fatalf("debugging regtest servers only works in Default mode, "+
				"got debug addr %q and mode %v", config.debugAddr, tc.mode)
		}

		t.Run(tc.name, func(t *testing.T) {
			// TODO(rfindley): once jsonrpc2 shutdown is fixed, we should not leak
			// goroutines in this test function.
			// stacktest.NoLeak(t)

			ctx := context.Background()
			if r.Timeout != 0 && !config.noDefaultTimeout {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, r.Timeout)
				defer cancel()
			} else if d, ok := testenv.Deadline(t); ok {
				timeout := time.Until(d) * 19 / 20 // Leave an arbitrary 5% for cleanup.
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, timeout)
				defer cancel()
			}

			ctx = debug.WithInstance(ctx, "", "off")
			if config.debugAddr != "" {
				di := debug.GetInstance(ctx)
				di.Serve(ctx, config.debugAddr)
				di.MonitorMemory(ctx)
			}

			rootDir := filepath.Join(r.tempDir, filepath.FromSlash(t.Name()))
			if err := os.MkdirAll(rootDir, 0755); err != nil {
				t.Fatal(err)
			}

			files := fake.UnpackTxt(files)
			if config.editor.WindowsLineEndings {
				for name, data := range files {
					files[name] = bytes.ReplaceAll(data, []byte("\n"), []byte("\r\n"))
				}
			}
			config.sandbox.Files = files
			config.sandbox.RootDir = rootDir
			sandbox, err := fake.NewSandbox(&config.sandbox)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if !r.SkipCleanup {
					if err := sandbox.Close(); err != nil {
						pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
						t.Errorf("closing the sandbox: %v", err)
					}
				}
			}()

			ss := tc.getServer(t, r.OptionsHook)

			framer := jsonrpc2.NewRawStream
			ls := &loggingFramer{}
			if !config.skipLogs {
				framer = ls.framer(jsonrpc2.NewRawStream)
			}
			ts := servertest.NewPipeServer(ss, framer)
			env, cleanup := NewEnv(ctx, t, sandbox, ts, config.editor, !config.skipHooks)
			defer cleanup()
			defer func() {
				if t.Failed() && r.PrintGoroutinesOnFailure {
					pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
				}
				if t.Failed() || *printLogs {
					ls.printBuffers(t.Name(), os.Stderr)
				}
				// For tests that failed due to a timeout, don't fail to shutdown
				// because ctx is done.
				//
				// golang/go#53820: now that we await the completion of ongoing work in
				// shutdown, we must allow a significant amount of time for ongoing go
				// command invocations to exit.
				ctx, cancel := context.WithTimeout(xcontext.Detach(ctx), 30*time.Second)
				defer cancel()
				if err := env.Editor.Close(ctx); err != nil {
					pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
					t.Errorf("closing editor: %v", err)
				}
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

// defaultServer handles the Default execution mode.
func (r *Runner) defaultServer(t *testing.T, optsHook func(*source.Options)) jsonrpc2.StreamServer {
	return lsprpc.NewStreamServer(cache.New(r.fset, r.store, optsHook), false)
}

// experimentalServer handles the Experimental execution mode.
func (r *Runner) experimentalServer(t *testing.T, optsHook func(*source.Options)) jsonrpc2.StreamServer {
	options := func(o *source.Options) {
		optsHook(o)
		o.EnableAllExperiments()
		// ExperimentalWorkspaceModule is not (as of writing) enabled by
		// source.Options.EnableAllExperiments, but we want to test it.
		o.ExperimentalWorkspaceModule = true
	}
	return lsprpc.NewStreamServer(cache.New(nil, nil, options), false)
}

// forwardedServer handles the Forwarded execution mode.
func (r *Runner) forwardedServer(_ *testing.T, optsHook func(*source.Options)) jsonrpc2.StreamServer {
	if r.ts == nil {
		r.mu.Lock()
		ctx := context.Background()
		ctx = debug.WithInstance(ctx, "", "off")
		ss := lsprpc.NewStreamServer(cache.New(nil, nil, optsHook), false)
		r.ts = servertest.NewTCPServer(ctx, ss, nil)
		r.mu.Unlock()
	}
	return newForwarder("tcp", r.ts.Addr)
}

// separateProcessServer handles the SeparateProcess execution mode.
func (r *Runner) separateProcessServer(t *testing.T, optsHook func(*source.Options)) jsonrpc2.StreamServer {
	// TODO(rfindley): can we use the autostart behavior here, instead of
	// pre-starting the remote?
	socket := r.getRemoteSocket(t)
	return newForwarder("unix", socket)
}

func newForwarder(network, address string) *lsprpc.Forwarder {
	server, err := lsprpc.NewForwarder(network+";"+address, nil)
	if err != nil {
		// This should never happen, as we are passing an explicit address.
		panic(fmt.Sprintf("internal error: unable to create forwarder: %v", err))
	}
	return server
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

	if r.goplsPath == "" {
		t.Fatal("cannot run tests with a separate process unless a path to a gopls binary is configured")
	}
	var err error
	r.socketDir, err = ioutil.TempDir(r.tempDir, "gopls-regtest-socket")
	if err != nil {
		t.Fatalf("creating tempdir: %v", err)
	}
	socket := filepath.Join(r.socketDir, daemonFile)
	args := []string{"serve", "-listen", "unix;" + socket, "-listen.timeout", "10s"}
	cmd := exec.Command(r.goplsPath, args...)
	cmd.Env = append(os.Environ(), runTestAsGoplsEnvvar+"=true")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	go func() {
		// TODO(rfindley): this is racy; we're returning before we know that the command is running.
		if err := cmd.Run(); err != nil {
			panic(fmt.Sprintf("error running external gopls: %v\nstderr:\n%s", err, stderr.String()))
		}
	}()
	return socket
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
		if err := os.RemoveAll(r.tempDir); err != nil {
			errmsgs = append(errmsgs, err.Error())
		}
	}
	if len(errmsgs) > 0 {
		return fmt.Errorf("errors closing the test runner:\n\t%s", strings.Join(errmsgs, "\n\t"))
	}
	return nil
}
