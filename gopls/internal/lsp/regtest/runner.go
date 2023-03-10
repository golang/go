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
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"testing"
	"time"

	exec "golang.org/x/sys/execabs"

	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/lsprpc"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
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
	//
	// Only supported on GOOS=linux.
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
	store     *memoize.Store // shared store

	// Lazily allocated resources
	tsOnce sync.Once
	ts     *servertest.TCPServer // shared in-process test server ("forwarded" mode)

	startRemoteOnce sync.Once
	remoteSocket    string // unix domain socket for shared daemon ("separate process" mode)
	remoteErr       error
	cancelRemote    func()
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
	testenv.NeedsGoPackages(t)

	tests := []struct {
		name      string
		mode      Mode
		getServer func(func(*source.Options)) jsonrpc2.StreamServer
	}{
		{"default", Default, r.defaultServer},
		{"forwarded", Forwarded, r.forwardedServer},
		{"separate_process", SeparateProcess, r.separateProcessServer},
		{"experimental", Experimental, r.experimentalServer},
	}

	for _, tc := range tests {
		tc := tc
		config := defaultConfig()
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

		t.Run(tc.name, func(t *testing.T) {
			// TODO(rfindley): once jsonrpc2 shutdown is fixed, we should not leak
			// goroutines in this test function.
			// stacktest.NoLeak(t)

			ctx := context.Background()
			if r.Timeout != 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, r.Timeout)
				defer cancel()
			} else if d, ok := testenv.Deadline(t); ok {
				timeout := time.Until(d) * 19 / 20 // Leave an arbitrary 5% for cleanup.
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, timeout)
				defer cancel()
			}

			// TODO(rfindley): do we need an instance at all? Can it be removed?
			ctx = debug.WithInstance(ctx, "", "off")

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

			ss := tc.getServer(r.OptionsHook)

			framer := jsonrpc2.NewRawStream
			ls := &loggingFramer{}
			framer = ls.framer(jsonrpc2.NewRawStream)
			ts := servertest.NewPipeServer(ss, framer)

			awaiter := NewAwaiter(sandbox.Workdir)
			const skipApplyEdits = false
			editor, err := fake.NewEditor(sandbox, config.editor).Connect(ctx, ts, awaiter.Hooks(), skipApplyEdits)
			if err != nil {
				t.Fatal(err)
			}
			env := &Env{
				T:       t,
				Ctx:     ctx,
				Sandbox: sandbox,
				Editor:  editor,
				Server:  ts,
				Awaiter: awaiter,
			}
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
				// There is little point to setting an arbitrary timeout for closing
				// the editor: in general we want to clean up before proceeding to the
				// next test, and if there is a deadlock preventing closing it will
				// eventually be handled by the `go test` timeout.
				if err := editor.Close(xcontext.Detach(ctx)); err != nil {
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
func (r *Runner) defaultServer(optsHook func(*source.Options)) jsonrpc2.StreamServer {
	return lsprpc.NewStreamServer(cache.New(r.store), false, optsHook)
}

// experimentalServer handles the Experimental execution mode.
func (r *Runner) experimentalServer(optsHook func(*source.Options)) jsonrpc2.StreamServer {
	options := func(o *source.Options) {
		optsHook(o)
		o.EnableAllExperiments()
	}
	return lsprpc.NewStreamServer(cache.New(nil), false, options)
}

// forwardedServer handles the Forwarded execution mode.
func (r *Runner) forwardedServer(optsHook func(*source.Options)) jsonrpc2.StreamServer {
	r.tsOnce.Do(func() {
		ctx := context.Background()
		ctx = debug.WithInstance(ctx, "", "off")
		ss := lsprpc.NewStreamServer(cache.New(nil), false, optsHook)
		r.ts = servertest.NewTCPServer(ctx, ss, nil)
	})
	return newForwarder("tcp", r.ts.Addr)
}

// runTestAsGoplsEnvvar triggers TestMain to run gopls instead of running
// tests. It's a trick to allow tests to find a binary to use to start a gopls
// subprocess.
const runTestAsGoplsEnvvar = "_GOPLS_TEST_BINARY_RUN_AS_GOPLS"

// separateProcessServer handles the SeparateProcess execution mode.
func (r *Runner) separateProcessServer(optsHook func(*source.Options)) jsonrpc2.StreamServer {
	if runtime.GOOS != "linux" {
		panic("separate process execution mode is only supported on linux")
	}

	r.startRemoteOnce.Do(func() {
		socketDir, err := ioutil.TempDir(r.tempDir, "gopls-regtest-socket")
		if err != nil {
			r.remoteErr = err
			return
		}
		r.remoteSocket = filepath.Join(socketDir, "gopls-test-daemon")

		// The server should be killed by when the test runner exits, but to be
		// conservative also set a listen timeout.
		args := []string{"serve", "-listen", "unix;" + r.remoteSocket, "-listen.timeout", "1m"}

		ctx, cancel := context.WithCancel(context.Background())
		cmd := exec.CommandContext(ctx, r.goplsPath, args...)
		cmd.Env = append(os.Environ(), runTestAsGoplsEnvvar+"=true")

		// Start the external gopls process. This is still somewhat racy, as we
		// don't know when gopls binds to the socket, but the gopls forwarder
		// client has built-in retry behavior that should mostly mitigate this
		// problem (and if it doesn't, we probably want to improve the retry
		// behavior).
		if err := cmd.Start(); err != nil {
			cancel()
			r.remoteSocket = ""
			r.remoteErr = err
		} else {
			r.cancelRemote = cancel
			// Spin off a goroutine to wait, so that we free up resources when the
			// server exits.
			go cmd.Wait()
		}
	})

	return newForwarder("unix", r.remoteSocket)
}

func newForwarder(network, address string) *lsprpc.Forwarder {
	server, err := lsprpc.NewForwarder(network+";"+address, nil)
	if err != nil {
		// This should never happen, as we are passing an explicit address.
		panic(fmt.Sprintf("internal error: unable to create forwarder: %v", err))
	}
	return server
}

// Close cleans up resource that have been allocated to this workspace.
func (r *Runner) Close() error {
	var errmsgs []string
	if r.ts != nil {
		if err := r.ts.Close(); err != nil {
			errmsgs = append(errmsgs, err.Error())
		}
	}
	if r.cancelRemote != nil {
		r.cancelRemote()
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
