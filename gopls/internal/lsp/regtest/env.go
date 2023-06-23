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

	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
)

// Env holds the building blocks of an editor testing environment, providing
// wrapper methods that hide the boilerplate of plumbing contexts and checking
// errors.
type Env struct {
	T   testing.TB // TODO(rfindley): rename to TB
	Ctx context.Context

	// Most tests should not need to access the scratch area, editor, server, or
	// connection, but they are available if needed.
	Sandbox *fake.Sandbox
	Server  servertest.Connector

	// Editor is owned by the Env, and shut down
	Editor *fake.Editor

	Awaiter *Awaiter
}

// An Awaiter keeps track of relevant LSP state, so that it may be asserted
// upon with Expectations.
//
// Wire it into a fake.Editor using Awaiter.Hooks().
//
// TODO(rfindley): consider simply merging Awaiter with the fake.Editor. It
// probably is not worth its own abstraction.
type Awaiter struct {
	workdir *fake.Workdir

	mu sync.Mutex
	// For simplicity, each waiter gets a unique ID.
	nextWaiterID int
	state        State
	waiters      map[int]*condition
}

func NewAwaiter(workdir *fake.Workdir) *Awaiter {
	return &Awaiter{
		workdir: workdir,
		state: State{
			diagnostics: make(map[string]*protocol.PublishDiagnosticsParams),
			work:        make(map[protocol.ProgressToken]*workProgress),
		},
		waiters: make(map[int]*condition),
	}
}

// Hooks returns LSP client hooks required for awaiting asynchronous expectations.
func (a *Awaiter) Hooks() fake.ClientHooks {
	return fake.ClientHooks{
		OnDiagnostics:            a.onDiagnostics,
		OnLogMessage:             a.onLogMessage,
		OnWorkDoneProgressCreate: a.onWorkDoneProgressCreate,
		OnProgress:               a.onProgress,
		OnShowDocument:           a.onShowDocument,
		OnShowMessage:            a.onShowMessage,
		OnShowMessageRequest:     a.onShowMessageRequest,
		OnRegisterCapability:     a.onRegisterCapability,
		OnUnregisterCapability:   a.onUnregisterCapability,
		OnApplyEdit:              a.onApplyEdit,
	}
}

// State encapsulates the server state TODO: explain more
type State struct {
	// diagnostics are a map of relative path->diagnostics params
	diagnostics        map[string]*protocol.PublishDiagnosticsParams
	logs               []*protocol.LogMessageParams
	showDocument       []*protocol.ShowDocumentParams
	showMessage        []*protocol.ShowMessageParams
	showMessageRequest []*protocol.ShowMessageRequestParams

	registrations          []*protocol.RegistrationParams
	registeredCapabilities map[string]protocol.Registration
	unregistrations        []*protocol.UnregistrationParams
	documentChanges        []protocol.DocumentChanges // collected from ApplyEdit downcalls

	// outstandingWork is a map of token->work summary. All tokens are assumed to
	// be string, though the spec allows for numeric tokens as well.  When work
	// completes, it is deleted from this map.
	work map[protocol.ProgressToken]*workProgress
}

// outstandingWork counts started but not complete work items by title.
func (s State) outstandingWork() map[string]uint64 {
	outstanding := make(map[string]uint64)
	for _, work := range s.work {
		if !work.complete {
			outstanding[work.title]++
		}
	}
	return outstanding
}

// completedWork counts complete work items by title.
func (s State) completedWork() map[string]uint64 {
	completed := make(map[string]uint64)
	for _, work := range s.work {
		if work.complete {
			completed[work.title]++
		}
	}
	return completed
}

// startedWork counts started (and possibly complete) work items.
func (s State) startedWork() map[string]uint64 {
	started := make(map[string]uint64)
	for _, work := range s.work {
		started[work.title]++
	}
	return started
}

type workProgress struct {
	title, msg, endMsg string
	percent            float64
	complete           bool // seen 'end'.
}

// This method, provided for debugging, accesses mutable fields without a lock,
// so it must not be called concurrent with any State mutation.
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
			fmt.Fprintf(&b, "\t\t(%d, %d) [%s]: %s\n", int(d.Range.Start.Line), int(d.Range.Start.Character), d.Source, d.Message)
		}
	}
	b.WriteString("\n")
	b.WriteString("#### outstanding work:\n")
	for token, state := range s.work {
		if state.complete {
			continue
		}
		name := state.title
		if name == "" {
			name = fmt.Sprintf("!NO NAME(token: %s)", token)
		}
		fmt.Fprintf(&b, "\t%s: %.2f\n", name, state.percent)
	}
	b.WriteString("#### completed work:\n")
	for name, count := range s.completedWork() {
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

func (a *Awaiter) onApplyEdit(_ context.Context, params *protocol.ApplyWorkspaceEditParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.documentChanges = append(a.state.documentChanges, params.Edit.DocumentChanges...)
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) onDiagnostics(_ context.Context, d *protocol.PublishDiagnosticsParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	pth := a.workdir.URIToPath(d.URI)
	a.state.diagnostics[pth] = d
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) onShowDocument(_ context.Context, params *protocol.ShowDocumentParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.showDocument = append(a.state.showDocument, params)
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) onShowMessage(_ context.Context, m *protocol.ShowMessageParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.showMessage = append(a.state.showMessage, m)
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) onShowMessageRequest(_ context.Context, m *protocol.ShowMessageRequestParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.showMessageRequest = append(a.state.showMessageRequest, m)
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) onLogMessage(_ context.Context, m *protocol.LogMessageParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.logs = append(a.state.logs, m)
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) onWorkDoneProgressCreate(_ context.Context, m *protocol.WorkDoneProgressCreateParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.work[m.Token] = &workProgress{}
	return nil
}

func (a *Awaiter) onProgress(_ context.Context, m *protocol.ProgressParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	work, ok := a.state.work[m.Token]
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
		work.complete = true
		if msg, ok := v["message"]; ok {
			work.endMsg = msg.(string)
		}
	}
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) onRegisterCapability(_ context.Context, m *protocol.RegistrationParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.registrations = append(a.state.registrations, m)
	if a.state.registeredCapabilities == nil {
		a.state.registeredCapabilities = make(map[string]protocol.Registration)
	}
	for _, reg := range m.Registrations {
		a.state.registeredCapabilities[reg.Method] = reg
	}
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) onUnregisterCapability(_ context.Context, m *protocol.UnregistrationParams) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.unregistrations = append(a.state.unregistrations, m)
	a.checkConditionsLocked()
	return nil
}

func (a *Awaiter) checkConditionsLocked() {
	for id, condition := range a.waiters {
		if v, _ := checkExpectations(a.state, condition.expectations); v != Unmet {
			delete(a.waiters, id)
			condition.verdict <- v
		}
	}
}

// takeDocumentChanges returns any accumulated document changes (from
// server ApplyEdit RPC downcalls) and resets the list.
func (a *Awaiter) takeDocumentChanges() []protocol.DocumentChanges {
	a.mu.Lock()
	defer a.mu.Unlock()

	res := a.state.documentChanges
	a.state.documentChanges = nil
	return res
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
		fmt.Fprintf(&summary, "%v: %s\n", v, e.Description)
	}
	return finalVerdict, summary.String()
}

// Await blocks until the given expectations are all simultaneously met.
//
// Generally speaking Await should be avoided because it blocks indefinitely if
// gopls ends up in a state where the expectations are never going to be met.
// Use AfterChange or OnceMet instead, so that the runner knows when to stop
// waiting.
func (e *Env) Await(expectations ...Expectation) {
	e.T.Helper()
	if err := e.Awaiter.Await(e.Ctx, expectations...); err != nil {
		e.T.Fatal(err)
	}
}

// OnceMet blocks until the precondition is met by the state or becomes
// unmeetable. If it was met, OnceMet checks that the state meets all
// expectations in mustMeets.
func (e *Env) OnceMet(precondition Expectation, mustMeets ...Expectation) {
	e.T.Helper()
	e.Await(OnceMet(precondition, mustMeets...))
}

// Await waits for all expectations to simultaneously be met. It should only be
// called from the main test goroutine.
func (a *Awaiter) Await(ctx context.Context, expectations ...Expectation) error {
	a.mu.Lock()
	// Before adding the waiter, we check if the condition is currently met or
	// failed to avoid a race where the condition was realized before Await was
	// called.
	switch verdict, summary := checkExpectations(a.state, expectations); verdict {
	case Met:
		a.mu.Unlock()
		return nil
	case Unmeetable:
		err := fmt.Errorf("unmeetable expectations:\n%s\nstate:\n%v", summary, a.state)
		a.mu.Unlock()
		return err
	}
	cond := &condition{
		expectations: expectations,
		verdict:      make(chan Verdict),
	}
	a.waiters[a.nextWaiterID] = cond
	a.nextWaiterID++
	a.mu.Unlock()

	var err error
	select {
	case <-ctx.Done():
		err = ctx.Err()
	case v := <-cond.verdict:
		if v != Met {
			err = fmt.Errorf("condition has final verdict %v", v)
		}
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	_, summary := checkExpectations(a.state, expectations)

	// Debugging an unmet expectation can be tricky, so we put some effort into
	// nicely formatting the failure.
	if err != nil {
		return fmt.Errorf("waiting on:\n%s\nerr:%v\n\nstate:\n%v", summary, err, a.state)
	}
	return nil
}
