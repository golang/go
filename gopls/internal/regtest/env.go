// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

// Env holds an initialized fake Editor, Workspace, and Server, which may be
// used for writing tests. It also provides adapter methods that call t.Fatal
// on any error, so that tests for the happy path may be written without
// checking errors.
type Env struct {
	T   *testing.T
	Ctx context.Context

	// Most tests should not need to access the scratch area, editor, server, or
	// connection, but they are available if needed.
	Sandbox *fake.Sandbox
	Editor  *fake.Editor
	Server  servertest.Connector

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
	diagnostics        map[string]*protocol.PublishDiagnosticsParams
	logs               []*protocol.LogMessageParams
	showMessage        []*protocol.ShowMessageParams
	showMessageRequest []*protocol.ShowMessageRequestParams

	registrations   []*protocol.RegistrationParams
	unregistrations []*protocol.UnregistrationParams

	// outstandingWork is a map of token->work summary. All tokens are assumed to
	// be string, though the spec allows for numeric tokens as well.  When work
	// completes, it is deleted from this map.
	outstandingWork map[protocol.ProgressToken]*workProgress
	completedWork   map[string]int
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
		fmt.Fprintf(&b, "\t%s: %.2f\n", name, state.percent)
	}
	b.WriteString("#### completed work:\n")
	for name, count := range s.completedWork {
		fmt.Fprintf(&b, "\t%s: %d\n", name, count)
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

// NewEnv creates a new test environment using the given scratch environment
// and gopls server.
func NewEnv(ctx context.Context, t *testing.T, sandbox *fake.Sandbox, ts servertest.Connector, editorConfig fake.EditorConfig, withHooks bool) *Env {
	t.Helper()
	conn := ts.Connect(ctx)
	env := &Env{
		T:       t,
		Ctx:     ctx,
		Sandbox: sandbox,
		Server:  ts,
		state: State{
			diagnostics:     make(map[string]*protocol.PublishDiagnosticsParams),
			outstandingWork: make(map[protocol.ProgressToken]*workProgress),
			completedWork:   make(map[string]int),
		},
		waiters: make(map[int]*condition),
	}
	var hooks fake.ClientHooks
	if withHooks {
		hooks = fake.ClientHooks{
			OnDiagnostics:            env.onDiagnostics,
			OnLogMessage:             env.onLogMessage,
			OnWorkDoneProgressCreate: env.onWorkDoneProgressCreate,
			OnProgress:               env.onProgress,
			OnShowMessage:            env.onShowMessage,
			OnShowMessageRequest:     env.onShowMessageRequest,
			OnRegistration:           env.onRegistration,
			OnUnregistration:         env.onUnregistration,
		}
	}
	editor, err := fake.NewEditor(sandbox, editorConfig).Connect(ctx, conn, hooks)
	if err != nil {
		t.Fatal(err)
	}
	env.Editor = editor
	return env
}

func (e *Env) onDiagnostics(_ context.Context, d *protocol.PublishDiagnosticsParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	pth := e.Sandbox.Workdir.URIToPath(d.URI)
	e.state.diagnostics[pth] = d
	e.checkConditionsLocked()
	return nil
}

func (e *Env) onShowMessage(_ context.Context, m *protocol.ShowMessageParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.state.showMessage = append(e.state.showMessage, m)
	e.checkConditionsLocked()
	return nil
}

func (e *Env) onShowMessageRequest(_ context.Context, m *protocol.ShowMessageRequestParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.state.showMessageRequest = append(e.state.showMessageRequest, m)
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

	e.state.outstandingWork[m.Token] = &workProgress{}
	return nil
}

func (e *Env) onProgress(_ context.Context, m *protocol.ProgressParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	work, ok := e.state.outstandingWork[m.Token]
	if !ok {
		panic(fmt.Sprintf("got progress report for unknown report %v: %v", m.Token, m))
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
		title := e.state.outstandingWork[m.Token].title
		e.state.completedWork[title] = e.state.completedWork[title] + 1
		delete(e.state.outstandingWork, m.Token)
	}
	e.checkConditionsLocked()
	return nil
}

func (e *Env) onRegistration(_ context.Context, m *protocol.RegistrationParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.state.registrations = append(e.state.registrations, m)
	e.checkConditionsLocked()
	return nil
}

func (e *Env) onUnregistration(_ context.Context, m *protocol.UnregistrationParams) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.state.unregistrations = append(e.state.unregistrations, m)
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

// OnceMet returns an Expectation that, once the precondition is met, asserts
// that mustMeet is met.
func OnceMet(precondition Expectation, mustMeet Expectation) *SimpleExpectation {
	check := func(s State) (Verdict, interface{}) {
		switch pre, _ := precondition.Check(s); pre {
		case Unmeetable:
			return Unmeetable, nil
		case Met:
			verdict, metBy := mustMeet.Check(s)
			if verdict != Met {
				return Unmeetable, metBy
			}
			return Met, metBy
		default:
			return Unmet, nil
		}
	}
	return &SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("once %q is met, must have %q", precondition.Description(), mustMeet.Description()),
	}
}

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
	check       func(State) (Verdict, interface{})
	description string
}

// Check invokes e.check.
func (e SimpleExpectation) Check(s State) (Verdict, interface{}) {
	return e.check(s)
}

// Description returns e.descriptin.
func (e SimpleExpectation) Description() string {
	return e.description
}

// NoOutstandingWork asserts that there is no work initiated using the LSP
// $/progress API that has not completed.
func NoOutstandingWork() SimpleExpectation {
	check := func(s State) (Verdict, interface{}) {
		if len(s.outstandingWork) == 0 {
			return Met, nil
		}
		return Unmet, nil
	}
	return SimpleExpectation{
		check:       check,
		description: "no outstanding work",
	}
}

// NoShowMessage asserts that the editor has not received a ShowMessage.
func NoShowMessage() SimpleExpectation {
	check := func(s State) (Verdict, interface{}) {
		if len(s.showMessage) == 0 {
			return Met, "no ShowMessage"
		}
		return Unmeetable, nil
	}
	return SimpleExpectation{
		check:       check,
		description: "no ShowMessage received",
	}
}

// SomeShowMessage asserts that the editor has received a ShowMessage with the given title.
func SomeShowMessage(title string) SimpleExpectation {
	check := func(s State) (Verdict, interface{}) {
		for _, m := range s.showMessage {
			if strings.Contains(m.Message, title) {
				return Met, m
			}
		}
		return Unmet, nil
	}
	return SimpleExpectation{
		check:       check,
		description: "received ShowMessage",
	}
}

// ShowMessageRequest asserts that the editor has received a ShowMessageRequest
// with an action item that has the given title.
func ShowMessageRequest(title string) SimpleExpectation {
	check := func(s State) (Verdict, interface{}) {
		if len(s.showMessageRequest) == 0 {
			return Unmet, nil
		}
		// Only check the most recent one.
		m := s.showMessageRequest[len(s.showMessageRequest)-1]
		if len(m.Actions) == 0 || len(m.Actions) > 1 {
			return Unmet, nil
		}
		if m.Actions[0].Title == title {
			return Met, m.Actions[0]
		}
		return Unmet, nil
	}
	return SimpleExpectation{
		check:       check,
		description: "received ShowMessageRequest",
	}
}

// CompletedWork expects a work item to have been completed >= atLeast times.
//
// Since the Progress API doesn't include any hidden metadata, we must use the
// progress notification title to identify the work we expect to be completed.
func CompletedWork(title string, atLeast int) SimpleExpectation {
	check := func(s State) (Verdict, interface{}) {
		if s.completedWork[title] >= atLeast {
			return Met, title
		}
		return Unmet, nil
	}
	return SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("completed work %q at least %d time(s)", title, atLeast),
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
	return NoLogMatching(protocol.Error, "")
}

// LogMatching asserts that the client has received a log message
// of type typ matching the regexp re.
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

// NoLogMatching asserts that the client has not received a log message
// of type typ matching the regexp re. If re is an empty string, any log
// message is considered a match.
func NoLogMatching(typ protocol.MessageType, re string) LogExpectation {
	var r *regexp.Regexp
	if re != "" {
		var err error
		r, err = regexp.Compile(re)
		if err != nil {
			panic(err)
		}
	}
	check := func(msgs []*protocol.LogMessageParams) (Verdict, interface{}) {
		for _, msg := range msgs {
			if msg.Type != typ {
				continue
			}
			if r == nil || r.Match([]byte(msg.Message)) {
				return Unmeetable, nil
			}
		}
		return Met, nil
	}
	return LogExpectation{
		check:       check,
		description: fmt.Sprintf("no log message matching %q", re),
	}
}

// RegistrationExpectation is an expectation on the capability registrations
// received by the editor from gopls.
type RegistrationExpectation struct {
	check       func([]*protocol.RegistrationParams) (Verdict, interface{})
	description string
}

// Check implements the Expectation interface.
func (e RegistrationExpectation) Check(s State) (Verdict, interface{}) {
	return e.check(s.registrations)
}

// Description implements the Expectation interface.
func (e RegistrationExpectation) Description() string {
	return e.description
}

// RegistrationMatching asserts that the client has received a capability
// registration matching the given regexp.
func RegistrationMatching(re string) RegistrationExpectation {
	rec, err := regexp.Compile(re)
	if err != nil {
		panic(err)
	}
	check := func(params []*protocol.RegistrationParams) (Verdict, interface{}) {
		for _, p := range params {
			for _, r := range p.Registrations {
				if rec.Match([]byte(r.Method)) {
					return Met, r
				}
			}
		}
		return Unmet, nil
	}
	return RegistrationExpectation{
		check:       check,
		description: fmt.Sprintf("registration matching %q", re),
	}
}

// UnregistrationExpectation is an expectation on the capability
// unregistrations received by the editor from gopls.
type UnregistrationExpectation struct {
	check       func([]*protocol.UnregistrationParams) (Verdict, interface{})
	description string
}

// Check implements the Expectation interface.
func (e UnregistrationExpectation) Check(s State) (Verdict, interface{}) {
	return e.check(s.unregistrations)
}

// Description implements the Expectation interface.
func (e UnregistrationExpectation) Description() string {
	return e.description
}

// UnregistrationMatching asserts that the client has received an
// unregistration whose ID matches the given regexp.
func UnregistrationMatching(re string) UnregistrationExpectation {
	rec, err := regexp.Compile(re)
	if err != nil {
		panic(err)
	}
	check := func(params []*protocol.UnregistrationParams) (Verdict, interface{}) {
		for _, p := range params {
			for _, r := range p.Unregisterations {
				if rec.Match([]byte(r.Method)) {
					return Met, r
				}
			}
		}
		return Unmet, nil
	}
	return UnregistrationExpectation{
		check:       check,
		description: fmt.Sprintf("unregistration matching %q", re),
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
	// Path is the scratch workdir-relative path to the file being asserted on.
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

// EmptyDiagnostics asserts that empty diagnostics are sent for the
// workspace-relative path name.
func EmptyDiagnostics(name string) Expectation {
	check := func(s State) (Verdict, interface{}) {
		if diags := s.diagnostics[name]; diags != nil && len(diags.Diagnostics) == 0 {
			return Met, nil
		}
		return Unmet, nil
	}
	return SimpleExpectation{
		check:       check,
		description: "empty diagnostics",
	}
}

// NoDiagnostics asserts that no diagnostics are sent for the
// workspace-relative path name. It should be used primarily in conjunction
// with a OnceMet, as it has to check that all outstanding diagnostics have
// already been delivered.
func NoDiagnostics(name string) Expectation {
	check := func(s State) (Verdict, interface{}) {
		if _, ok := s.diagnostics[name]; !ok {
			return Met, nil
		}
		return Unmet, nil
	}
	return SimpleExpectation{
		check:       check,
		description: "no diagnostics",
	}
}

// AnyDiagnosticAtCurrentVersion asserts that there is a diagnostic report for
// the current edited version of the buffer corresponding to the given
// workdir-relative pathname.
func (e *Env) AnyDiagnosticAtCurrentVersion(name string) DiagnosticExpectation {
	version := e.Editor.BufferVersion(name)
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
	e.T.Helper()
	pos := e.RegexpSearch(name, re)
	expectation := DiagnosticAt(name, pos.Line, pos.Column)
	expectation.description += fmt.Sprintf(" (location of %q)", re)
	return expectation
}

// DiagnosticAt asserts that there is a diagnostic entry at the position
// specified by line and col, for the workdir-relative path name.
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

// DiagnosticsFor returns the current diagnostics for the file. It is useful
// after waiting on AnyDiagnosticAtCurrentVersion, when the desired diagnostic
// is not simply described by DiagnosticAt.
func (e *Env) DiagnosticsFor(name string) *protocol.PublishDiagnosticsParams {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.state.diagnostics[name]
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
		e.mu.Unlock()
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
		e.T.Fatalf("waiting on:\n%s\nerr:%v\n\nstate:\n%v", summary, err, e.state)
	}
	return metBy
}
