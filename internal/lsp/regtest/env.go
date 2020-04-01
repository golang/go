// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package regtest provides an environment for writing regression tests.
package regtest

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
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

// EnvMode is a bitmask that defines in which execution environments a test
// should run.
type EnvMode int

const (
	// Singleton mode uses a separate cache for each test.
	Singleton EnvMode = 1 << iota
	// Shared mode uses a Shared cache.
	Shared
	// Forwarded forwards connections to an in-process gopls instance.
	Forwarded
	// SeparateProcess runs a separate gopls process, and forwards connections to
	// it.
	SeparateProcess
	// NormalModes runs tests in all modes.
	NormalModes = Singleton | Forwarded
)

// A Runner runs tests in gopls execution environments, as specified by its
// modes. For modes that share state (for example, a shared cache or common
// remote), any tests that execute on the same Runner will share the same
// state.
type Runner struct {
	defaultModes EnvMode
	timeout      time.Duration
	goplsPath    string

	mu        sync.Mutex
	ts        *servertest.TCPServer
	socketDir string
}

// NewTestRunner creates a Runner with its shared state initialized, ready to
// run tests.
func NewTestRunner(modes EnvMode, testTimeout time.Duration, goplsPath string) *Runner {
	return &Runner{
		defaultModes: modes,
		timeout:      testTimeout,
		goplsPath:    goplsPath,
	}
}

// Modes returns the bitmask of environment modes this runner is configured to
// test.
func (r *Runner) Modes() EnvMode {
	return r.defaultModes
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
	r.socketDir, err = ioutil.TempDir("", "gopls-regtests")
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
	if r.ts != nil {
		r.ts.Close()
	}
	if r.socketDir != "" {
		os.RemoveAll(r.socketDir)
	}
	return nil
}

// Run executes the test function in the default configured gopls execution
// modes. For each a test run, a new workspace is created containing the
// un-txtared files specified by filedata.
func (r *Runner) Run(t *testing.T, filedata string, test func(e *Env)) {
	t.Helper()
	r.RunInMode(r.defaultModes, t, filedata, test)
}

// RunInMode runs the test in the execution modes specified by the modes bitmask.
func (r *Runner) RunInMode(modes EnvMode, t *testing.T, filedata string, test func(e *Env)) {
	t.Helper()
	tests := []struct {
		name         string
		mode         EnvMode
		getConnector func(context.Context, *testing.T) (servertest.Connector, func())
	}{
		{"singleton", Singleton, r.singletonEnv},
		{"shared", Shared, r.sharedEnv},
		{"forwarded", Forwarded, r.forwardedEnv},
		{"separate_process", SeparateProcess, r.separateProcessEnv},
	}

	for _, tc := range tests {
		tc := tc
		if modes&tc.mode == 0 {
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			t.Helper()
			ctx, cancel := context.WithTimeout(context.Background(), r.timeout)
			defer cancel()
			ws, err := fake.NewWorkspace("lsprpc", []byte(filedata))
			if err != nil {
				t.Fatal(err)
			}
			defer ws.Close()
			ts, cleanup := tc.getConnector(ctx, t)
			defer cleanup()
			env := NewEnv(ctx, t, ws, ts)
			defer func() {
				if err := env.E.Shutdown(ctx); err != nil {
					panic(err)
				}
			}()
			test(env)
		})
	}
}

func (r *Runner) singletonEnv(ctx context.Context, t *testing.T) (servertest.Connector, func()) {
	ctx = debug.WithInstance(ctx, "", "")
	ss := lsprpc.NewStreamServer(cache.New(ctx, nil))
	ts := servertest.NewPipeServer(ctx, ss)
	cleanup := func() {
		ts.Close()
	}
	return ts, cleanup
}

func (r *Runner) sharedEnv(ctx context.Context, t *testing.T) (servertest.Connector, func()) {
	return r.getTestServer(), func() {}
}

func (r *Runner) forwardedEnv(ctx context.Context, t *testing.T) (servertest.Connector, func()) {
	ctx = debug.WithInstance(ctx, "", "")
	ts := r.getTestServer()
	forwarder := lsprpc.NewForwarder("tcp", ts.Addr)
	ts2 := servertest.NewPipeServer(ctx, forwarder)
	cleanup := func() {
		ts2.Close()
	}
	return ts2, cleanup
}

func (r *Runner) separateProcessEnv(ctx context.Context, t *testing.T) (servertest.Connector, func()) {
	ctx = debug.WithInstance(ctx, "", "")
	socket := r.getRemoteSocket(t)
	// TODO(rfindley): can we use the autostart behavior here, instead of
	// pre-starting the remote?
	forwarder := lsprpc.NewForwarder("unix", socket)
	ts2 := servertest.NewPipeServer(ctx, forwarder)
	cleanup := func() {
		ts2.Close()
	}
	return ts2, cleanup
}

// Env holds an initialized fake Editor, Workspace, and Server, which may be
// used for writing tests. It also provides adapter methods that call t.Fatal
// on any error, so that tests for the happy path may be written without
// checking errors.
type Env struct {
	T   *testing.T
	Ctx context.Context

	// Most tests should not need to access the workspace, editor, server, or
	// connection, but they are available if needed.
	W      *fake.Workspace
	E      *fake.Editor
	Server servertest.Connector
	Conn   *jsonrpc2.Conn

	// mu guards the fields below, for the purpose of checking conditions on
	// every change to diagnostics.
	mu sync.Mutex
	// For simplicity, each waiter gets a unique ID.
	nextWaiterID    int
	lastDiagnostics map[string]*protocol.PublishDiagnosticsParams
	waiters         map[int]*diagnosticCondition
}

// A diagnosticCondition is satisfied when all expectations are simultaneously
// met. At that point, the 'met' channel is closed. On any failure, err is set
// and the failed channel is closed.
type diagnosticCondition struct {
	expectations []DiagnosticExpectation
	met          chan struct{}
}

// NewEnv creates a new test environment using the given workspace and gopls
// server.
func NewEnv(ctx context.Context, t *testing.T, ws *fake.Workspace, ts servertest.Connector) *Env {
	t.Helper()
	conn := ts.Connect(ctx)
	editor, err := fake.NewConnectedEditor(ctx, ws, conn)
	if err != nil {
		t.Fatal(err)
	}
	env := &Env{
		T:               t,
		Ctx:             ctx,
		W:               ws,
		E:               editor,
		Server:          ts,
		Conn:            conn,
		lastDiagnostics: make(map[string]*protocol.PublishDiagnosticsParams),
		waiters:         make(map[int]*diagnosticCondition),
	}
	env.E.Client().OnDiagnostics(env.onDiagnostics)
	return env
}

func (e *Env) onDiagnostics(_ context.Context, d *protocol.PublishDiagnosticsParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	pth := e.W.URIToPath(d.URI)
	e.lastDiagnostics[pth] = d

	for id, condition := range e.waiters {
		if meetsExpectations(e.lastDiagnostics, condition.expectations) {
			delete(e.waiters, id)
			close(condition.met)
		}
	}
	return nil
}

// ExpectDiagnostics asserts that the current diagnostics in the editor match
// the given expectations. It is intended to be used together with Env.Await to
// allow waiting on simpler diagnostic expectations (for example,
// AnyDiagnosticsACurrenttVersion), followed by more detailed expectations
// tested by ExpectDiagnostics.
//
// For example:
//  env.RegexpReplace("foo.go", "a", "x")
//  env.Await(env.AnyDiagnosticAtCurrentVersion("foo.go"))
//  env.ExpectDiagnostics(env.DiagnosticAtRegexp("foo.go", "x"))
//
// This has the advantage of not timing out if the diagnostic received for
// "foo.go" does not match the expectation: instead it fails early.
func (e *Env) ExpectDiagnostics(expectations ...DiagnosticExpectation) {
	e.T.Helper()
	e.mu.Lock()
	defer e.mu.Unlock()
	if !meetsExpectations(e.lastDiagnostics, expectations) {
		e.T.Fatalf("diagnostic are unmet:\n%s\nlast diagnostics:\n%s", summarizeExpectations(expectations), formatDiagnostics(e.lastDiagnostics))
	}
}

func meetsExpectations(m map[string]*protocol.PublishDiagnosticsParams, expectations []DiagnosticExpectation) bool {
	for _, e := range expectations {
		diags, ok := m[e.Path]
		if !ok {
			return false
		}
		if !e.IsMet(diags) {
			return false
		}
	}
	return true
}

// A DiagnosticExpectation is a condition that must be met by the current set
// of diagnostics.
type DiagnosticExpectation struct {
	// IsMet determines whether the diagnostics for this file version satisfy our
	// expectation.
	IsMet func(*protocol.PublishDiagnosticsParams) bool
	// Description is a human-readable description of the diagnostic expectation.
	Description string
	// Path is the workspace-relative path to the file being asserted on.
	Path string
}

// EmptyDiagnostics asserts that diagnostics are empty for the
// workspace-relative path name.
func EmptyDiagnostics(name string) DiagnosticExpectation {
	isMet := func(diags *protocol.PublishDiagnosticsParams) bool {
		return len(diags.Diagnostics) == 0
	}
	return DiagnosticExpectation{
		IsMet:       isMet,
		Description: "empty diagnostics",
		Path:        name,
	}
}

// AnyDiagnosticAtCurrentVersion asserts that there is a diagnostic report for
// the current edited version of the buffer corresponding to the given
// workspace-relative pathname.
func (e *Env) AnyDiagnosticAtCurrentVersion(name string) DiagnosticExpectation {
	version := e.E.BufferVersion(name)
	isMet := func(diags *protocol.PublishDiagnosticsParams) bool {
		return int(diags.Version) == version
	}
	return DiagnosticExpectation{
		IsMet:       isMet,
		Description: fmt.Sprintf("any diagnostics at version %d", version),
		Path:        name,
	}
}

// DiagnosticAtRegexp expects that there is a diagnostic entry at the start
// position matching the regexp search string re in the buffer specified by
// name. Note that this currently ignores the end position.
func (e *Env) DiagnosticAtRegexp(name, re string) DiagnosticExpectation {
	pos := e.RegexpSearch(name, re)
	expectation := DiagnosticAt(name, pos.Line, pos.Column)
	expectation.Description += fmt.Sprintf(" (location of %q)", re)
	return expectation
}

// DiagnosticAt asserts that there is a diagnostic entry at the position
// specified by line and col, for the workspace-relative path name.
func DiagnosticAt(name string, line, col int) DiagnosticExpectation {
	isMet := func(diags *protocol.PublishDiagnosticsParams) bool {
		for _, d := range diags.Diagnostics {
			if d.Range.Start.Line == float64(line) && d.Range.Start.Character == float64(col) {
				return true
			}
		}
		return false
	}
	return DiagnosticExpectation{
		IsMet:       isMet,
		Description: fmt.Sprintf("diagnostic at {line:%d, column:%d}", line, col),
		Path:        name,
	}
}

// Await waits for all diagnostic expectations to simultaneously be met. It
// should only be called from the main test goroutine.
func (e *Env) Await(expectations ...DiagnosticExpectation) {
	// NOTE: in the future this mechanism extend beyond just diagnostics, for
	// example by modifying IsMet to be a func(*Env) boo.  However, that would
	// require careful checking of conditions around every state change, so for
	// now we just limit the scope to diagnostic conditions.

	e.T.Helper()
	e.mu.Lock()
	// Before adding the waiter, we check if the condition is currently met or
	// failed to avoid a race where the condition was realized before Await was
	// called.
	if meetsExpectations(e.lastDiagnostics, expectations) {
		e.mu.Unlock()
		return
	}
	cond := &diagnosticCondition{
		expectations: expectations,
		met:          make(chan struct{}),
	}
	e.waiters[e.nextWaiterID] = cond
	e.nextWaiterID++
	e.mu.Unlock()

	select {
	case <-e.Ctx.Done():
		// Debugging an unmet expectation can be tricky, so we put some effort into
		// nicely formatting the failure.
		summary := summarizeExpectations(expectations)
		e.mu.Lock()
		diagString := formatDiagnostics(e.lastDiagnostics)
		e.mu.Unlock()
		e.T.Fatalf("waiting on:\n\t%s\nerr: %v\ndiagnostics:\n%s", summary, e.Ctx.Err(), diagString)
	case <-cond.met:
	}
}

func summarizeExpectations(expectations []DiagnosticExpectation) string {
	var descs []string
	for _, e := range expectations {
		descs = append(descs, fmt.Sprintf("%s: %s", e.Path, e.Description))
	}
	return strings.Join(descs, "\n\t")
}

func formatDiagnostics(diags map[string]*protocol.PublishDiagnosticsParams) string {
	var b strings.Builder
	for name, params := range diags {
		b.WriteString(fmt.Sprintf("\t%s (version %d):\n", name, int(params.Version)))
		for _, d := range params.Diagnostics {
			b.WriteString(fmt.Sprintf("\t\t(%d, %d): %s\n", int(d.Range.Start.Line), int(d.Range.Start.Character), d.Message))
		}
	}
	return b.String()
}
