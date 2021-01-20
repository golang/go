// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"context"
	"fmt"
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
	completedWork   map[string]uint64
}

type workProgress struct {
	title, msg string
	percent    float64
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
			completedWork:   make(map[string]uint64),
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
		if msg, ok := v["message"]; ok {
			work.msg = msg.(string)
		}
	case "report":
		if pct, ok := v["percentage"]; ok {
			work.percent = pct.(float64)
		}
		if msg, ok := v["message"]; ok {
			work.msg = msg.(string)
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
		if v, _ := checkExpectations(e.state, condition.expectations); v != Unmet {
			delete(e.waiters, id)
			condition.verdict <- v
		}
	}
}

// checkExpectations reports whether s meets all expectations.
func checkExpectations(s State, expectations []Expectation) (Verdict, string) {
	finalVerdict := Met
	var summary strings.Builder
	for _, e := range expectations {
		v := e.Check(s)
		if v > finalVerdict {
			finalVerdict = v
		}
		summary.WriteString(fmt.Sprintf("\t%v: %s\n", v, e.Description()))
	}
	return finalVerdict, summary.String()
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
func (e *Env) Await(expectations ...Expectation) {
	e.T.Helper()
	e.mu.Lock()
	// Before adding the waiter, we check if the condition is currently met or
	// failed to avoid a race where the condition was realized before Await was
	// called.
	switch verdict, summary := checkExpectations(e.state, expectations); verdict {
	case Met:
		e.mu.Unlock()
		return
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
	_, summary := checkExpectations(e.state, expectations)

	// Debugging an unmet expectation can be tricky, so we put some effort into
	// nicely formatting the failure.
	if err != nil {
		e.T.Fatalf("waiting on:\n%s\nerr:%v\n\nstate:\n%v", summary, err, e.state)
	}
}
