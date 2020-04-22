// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package regtest provides an environment for writing regression tests.
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
	"regexp"
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

// EnvMode is a bitmask that defines in which execution environments a test
// should run.
type EnvMode int

const (
	// Singleton mode uses a separate cache for each test.
	Singleton EnvMode = 1 << iota

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
	DefaultModes             EnvMode
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

// Modes returns the bitmask of environment modes this runner is configured to
// test.
func (r *Runner) Modes() EnvMode {
	return r.DefaultModes
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

type runConfig struct {
	modes    EnvMode
	proxyTxt string
	timeout  time.Duration
	env      []string
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
func WithModes(modes EnvMode) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.modes = modes
	})
}

func WithEnv(env ...string) RunOption {
	return optionSetter(func(opts *runConfig) {
		opts.env = env
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
		mode      EnvMode
		getServer func(context.Context, *testing.T) jsonrpc2.StreamServer
	}{
		{"singleton", Singleton, singletonEnv},
		{"forwarded", Forwarded, r.forwardedEnv},
		{"separate_process", SeparateProcess, r.separateProcessEnv},
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

			ws, err := fake.NewWorkspace("regtest", filedata, config.proxyTxt, config.env...)
			if err != nil {
				t.Fatal(err)
			}
			// Deferring the closure of ws until the end of the entire test suite
			// has, in testing, given the LSP server time to properly shutdown and
			// release any file locks held in workspace, which is a problem on
			// Windows. This may still be flaky however, and in the future we need a
			// better solution to ensure that all Go processes started by gopls have
			// exited before we clean up.
			r.AddCloser(ws)
			ss := tc.getServer(ctx, t)
			ls := &loggingServer{delegate: ss}
			ts := servertest.NewPipeServer(ctx, ls)
			defer func() {
				ts.Close()
			}()
			env := NewEnv(ctx, t, ws, ts)
			defer func() {
				if t.Failed() && r.PrintGoroutinesOnFailure {
					pprof.Lookup("goroutine").WriteTo(os.Stderr, 1)
				}
				if t.Failed() || r.AlwaysPrintLogs {
					ls.printBuffers(t.Name(), os.Stderr)
				}
				if err := env.E.Shutdown(ctx); err != nil {
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

func singletonEnv(ctx context.Context, t *testing.T) jsonrpc2.StreamServer {
	return lsprpc.NewStreamServer(cache.New(ctx, nil))
}

func (r *Runner) forwardedEnv(ctx context.Context, t *testing.T) jsonrpc2.StreamServer {
	ts := r.getTestServer()
	return lsprpc.NewForwarder("tcp", ts.Addr)
}

func (r *Runner) separateProcessEnv(ctx context.Context, t *testing.T) jsonrpc2.StreamServer {
	// TODO(rfindley): can we use the autostart behavior here, instead of
	// pre-starting the remote?
	socket := r.getRemoteSocket(t)
	return lsprpc.NewForwarder("unix", socket)
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
	nextWaiterID int
	state        State
	waiters      map[int]*condition
}

// State encapsulates the server state TODO: explain more
type State struct {
	// diagnostics are a map of relative path->diagnostics params
	diagnostics map[string]*protocol.PublishDiagnosticsParams
	logs        []*protocol.LogMessageParams
	// outstandingWork is a map of token->work summary. All tokens are assumed to
	// be string, though the spec allows for numeric tokens as well.  When work
	// completes, it is deleted from this map.
	outstandingWork map[string]*workProgress
}

type workProgress struct {
	title   string
	percent float64
}

func (s State) String() string {
	var b strings.Builder
	b.WriteString("#### log messages (see RPC logs for full text):\n")
	for _, msg := range s.logs {
		summary := fmt.Sprintf("%v: %q", msg.Type, msg.Message)
		if len(summary) > 60 {
			summary = summary[:57] + "..."
		}
		// Some logs are quite long, and since they should be reproduced in the RPC
		// logs on any failure we include here just a short summary.
		fmt.Fprint(&b, "\t"+summary+"\n")
	}
	b.WriteString("\n")
	b.WriteString("#### diagnostics:\n")
	for name, params := range s.diagnostics {
		fmt.Fprintf(&b, "\t%s (version %d):\n", name, int(params.Version))
		for _, d := range params.Diagnostics {
			fmt.Fprintf(&b, "\t\t(%d, %d): %s\n", int(d.Range.Start.Line), int(d.Range.Start.Character), d.Message)
		}
	}
	b.WriteString("\n")
	b.WriteString("#### outstanding work:\n")
	for token, state := range s.outstandingWork {
		name := state.title
		if name == "" {
			name = fmt.Sprintf("!NO NAME(token: %s)", token)
		}
		fmt.Fprintf(&b, "\t%s: %.2f", name, state.percent)
	}
	return b.String()
}

// A condition is satisfied when all expectations are simultaneously
// met. At that point, the 'met' channel is closed. On any failure, err is set
// and the failed channel is closed.
type condition struct {
	expectations []Expectation
	verdict      chan Verdict
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
		T:      t,
		Ctx:    ctx,
		W:      ws,
		E:      editor,
		Server: ts,
		Conn:   conn,
		state: State{
			diagnostics:     make(map[string]*protocol.PublishDiagnosticsParams),
			outstandingWork: make(map[string]*workProgress),
		},
		waiters: make(map[int]*condition),
	}
	env.E.Client().OnDiagnostics(env.onDiagnostics)
	env.E.Client().OnLogMessage(env.onLogMessage)
	env.E.Client().OnWorkDoneProgressCreate(env.onWorkDoneProgressCreate)
	env.E.Client().OnProgress(env.onProgress)
	return env
}

func (e *Env) onDiagnostics(_ context.Context, d *protocol.PublishDiagnosticsParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	pth := e.W.URIToPath(d.URI)
	e.state.diagnostics[pth] = d
	e.checkConditionsLocked()
	return nil
}

func (e *Env) onLogMessage(_ context.Context, m *protocol.LogMessageParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.state.logs = append(e.state.logs, m)
	e.checkConditionsLocked()
	return nil
}

func (e *Env) onWorkDoneProgressCreate(_ context.Context, m *protocol.WorkDoneProgressCreateParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	// panic if we don't have a string token.
	token := m.Token.(string)
	e.state.outstandingWork[token] = &workProgress{}
	return nil
}

func (e *Env) onProgress(_ context.Context, m *protocol.ProgressParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	token := m.Token.(string)
	work, ok := e.state.outstandingWork[token]
	if !ok {
		panic(fmt.Sprintf("got progress report for unknown report %s", token))
	}
	v := m.Value.(map[string]interface{})
	switch kind := v["kind"]; kind {
	case "begin":
		work.title = v["title"].(string)
	case "report":
		if pct, ok := v["percentage"]; ok {
			work.percent = pct.(float64)
		}
	case "end":
		delete(e.state.outstandingWork, token)
	}
	e.checkConditionsLocked()
	return nil
}

func (e *Env) checkConditionsLocked() {
	for id, condition := range e.waiters {
		if v, _, _ := checkExpectations(e.state, condition.expectations); v != Unmet {
			delete(e.waiters, id)
			condition.verdict <- v
		}
	}
}

// ExpectNow asserts that the current state of the editor matches the given
// expectations.
//
// It can be used together with Env.Await to allow waiting on
// simple expectations, followed by more detailed expectations tested by
// ExpectNow. For example:
//
//  env.RegexpReplace("foo.go", "a", "x")
//  env.Await(env.AnyDiagnosticAtCurrentVersion("foo.go"))
//  env.ExpectNow(env.DiagnosticAtRegexp("foo.go", "x"))
//
// This has the advantage of not timing out if the diagnostic received for
// "foo.go" does not match the expectation: instead it fails early.
func (e *Env) ExpectNow(expectations ...Expectation) {
	e.T.Helper()
	e.mu.Lock()
	defer e.mu.Unlock()
	if verdict, summary, _ := checkExpectations(e.state, expectations); verdict != Met {
		e.T.Fatalf("expectations unmet:\n%s\ncurrent state:\n%v", summary, e.state)
	}
}

// checkExpectations reports whether s meets all expectations.
func checkExpectations(s State, expectations []Expectation) (Verdict, string, []interface{}) {
	finalVerdict := Met
	var metBy []interface{}
	var summary strings.Builder
	for _, e := range expectations {
		v, mb := e.Check(s)
		if v == Met {
			metBy = append(metBy, mb)
		}
		if v > finalVerdict {
			finalVerdict = v
		}
		summary.WriteString(fmt.Sprintf("\t%v: %s\n", v, e.Description()))
	}
	return finalVerdict, summary.String(), metBy
}

// An Expectation asserts that the state of the editor at a point in time
// matches an expected condition. This is used for signaling in tests when
// certain conditions in the editor are met.
type Expectation interface {
	// Check determines whether the state of the editor satisfies the
	// expectation, returning the results that met the condition.
	Check(State) (Verdict, interface{})
	// Description is a human-readable description of the expectation.
	Description() string
}

// A Verdict is the result of checking an expectation against the current
// editor state.
type Verdict int

// Order matters for the following constants: verdicts are sorted in order of
// decisiveness.
const (
	// Met indicates that an expectation is satisfied by the current state.
	Met Verdict = iota
	// Unmet indicates that an expectation is not currently met, but could be met
	// in the future.
	Unmet
	// Unmeetable indicates that an expectation cannot be satisfied in the
	// future.
	Unmeetable
)

func (v Verdict) String() string {
	switch v {
	case Met:
		return "Met"
	case Unmet:
		return "Unmet"
	case Unmeetable:
		return "Unmeetable"
	}
	return fmt.Sprintf("unrecognized verdict %d", v)
}

// SimpleExpectation holds an arbitrary check func, and implements the Expectation interface.
type SimpleExpectation struct {
	check       func(State) Verdict
	description string
}

// Check invokes e.check.
func (e SimpleExpectation) Check(s State) Verdict {
	return e.check(s)
}

// Description returns e.descriptin.
func (e SimpleExpectation) Description() string {
	return e.description
}

// NoOutstandingWork asserts that there is no work initiated using the LSP
// $/progress API that has not completed.
func NoOutstandingWork() SimpleExpectation {
	check := func(s State) Verdict {
		if len(s.outstandingWork) == 0 {
			return Met
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: "no outstanding work",
	}
}

// LogExpectation is an expectation on the log messages received by the editor
// from gopls.
type LogExpectation struct {
	check       func([]*protocol.LogMessageParams) (Verdict, interface{})
	description string
}

// Check implements the Expectation interface.
func (e LogExpectation) Check(s State) (Verdict, interface{}) {
	return e.check(s.logs)
}

// Description implements the Expectation interface.
func (e LogExpectation) Description() string {
	return e.description
}

// NoErrorLogs asserts that the client has not received any log messages of
// error severity.
func NoErrorLogs() LogExpectation {
	check := func(msgs []*protocol.LogMessageParams) (Verdict, interface{}) {
		for _, msg := range msgs {
			if msg.Type == protocol.Error {
				return Unmeetable, nil
			}
		}
		return Met, nil
	}
	return LogExpectation{
		check:       check,
		description: "no errors have been logged",
	}
}

// LogMatching asserts that the client has received a log message
// matching of type typ matching the regexp re.
func LogMatching(typ protocol.MessageType, re string) LogExpectation {
	rec, err := regexp.Compile(re)
	if err != nil {
		panic(err)
	}
	check := func(msgs []*protocol.LogMessageParams) (Verdict, interface{}) {
		for _, msg := range msgs {
			if msg.Type == typ && rec.Match([]byte(msg.Message)) {
				return Met, msg
			}
		}
		return Unmet, nil
	}
	return LogExpectation{
		check:       check,
		description: fmt.Sprintf("log message matching %q", re),
	}
}

// A DiagnosticExpectation is a condition that must be met by the current set
// of diagnostics for a file.
type DiagnosticExpectation struct {
	// IsMet determines whether the diagnostics for this file version satisfy our
	// expectation.
	isMet func(*protocol.PublishDiagnosticsParams) bool
	// Description is a human-readable description of the diagnostic expectation.
	description string
	// Path is the workspace-relative path to the file being asserted on.
	path string
}

// Check implements the Expectation interface.
func (e DiagnosticExpectation) Check(s State) (Verdict, interface{}) {
	if diags, ok := s.diagnostics[e.path]; ok && e.isMet(diags) {
		return Met, diags
	}
	return Unmet, nil
}

// Description implements the Expectation interface.
func (e DiagnosticExpectation) Description() string {
	return fmt.Sprintf("%s: %s", e.path, e.description)
}

// EmptyDiagnostics asserts that diagnostics are empty for the
// workspace-relative path name.
func EmptyDiagnostics(name string) DiagnosticExpectation {
	isMet := func(diags *protocol.PublishDiagnosticsParams) bool {
		return len(diags.Diagnostics) == 0
	}
	return DiagnosticExpectation{
		isMet:       isMet,
		description: "empty diagnostics",
		path:        name,
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
		isMet:       isMet,
		description: fmt.Sprintf("any diagnostics at version %d", version),
		path:        name,
	}
}

// DiagnosticAtRegexp expects that there is a diagnostic entry at the start
// position matching the regexp search string re in the buffer specified by
// name. Note that this currently ignores the end position.
func (e *Env) DiagnosticAtRegexp(name, re string) DiagnosticExpectation {
	pos := e.RegexpSearch(name, re)
	expectation := DiagnosticAt(name, pos.Line, pos.Column)
	expectation.description += fmt.Sprintf(" (location of %q)", re)
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
		isMet:       isMet,
		description: fmt.Sprintf("diagnostic at {line:%d, column:%d}", line, col),
		path:        name,
	}
}

// Await waits for all expectations to simultaneously be met. It should only be
// called from the main test goroutine.
func (e *Env) Await(expectations ...Expectation) []interface{} {
	e.T.Helper()
	e.mu.Lock()
	// Before adding the waiter, we check if the condition is currently met or
	// failed to avoid a race where the condition was realized before Await was
	// called.
	switch verdict, summary, metBy := checkExpectations(e.state, expectations); verdict {
	case Met:
		return metBy
	case Unmeetable:
		e.mu.Unlock()
		e.T.Fatalf("unmeetable expectations:\n%s\nstate:\n%v", summary, e.state)
	}
	cond := &condition{
		expectations: expectations,
		verdict:      make(chan Verdict),
	}
	e.waiters[e.nextWaiterID] = cond
	e.nextWaiterID++
	e.mu.Unlock()

	var err error
	select {
	case <-e.Ctx.Done():
		err = e.Ctx.Err()
	case v := <-cond.verdict:
		if v != Met {
			err = fmt.Errorf("condition has final verdict %v", v)
		}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	_, summary, metBy := checkExpectations(e.state, expectations)

	// Debugging an unmet expectation can be tricky, so we put some effort into
	// nicely formatting the failure.
	if err != nil {
		e.T.Fatalf("waiting on:\n%s\nerr:%v\nstate:\n%v", err, summary, e.state)
	}
	return metBy
}
