// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"regexp"
	"strings"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/testenv"
)

// An Expectation asserts that the state of the editor at a point in time
// matches an expected condition. This is used for signaling in tests when
// certain conditions in the editor are met.
type Expectation interface {
	// Check determines whether the state of the editor satisfies the
	// expectation, returning the results that met the condition.
	Check(State) Verdict
	// Description is a human-readable description of the expectation.
	Description() string
}

var (
	// InitialWorkspaceLoad is an expectation that the workspace initial load has
	// completed. It is verified via workdone reporting.
	InitialWorkspaceLoad = CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad), 1)
)

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

// OnceMet returns an Expectation that, once the precondition is met, asserts
// that mustMeet is met.
func OnceMet(precondition Expectation, mustMeet Expectation) *SimpleExpectation {
	check := func(s State) Verdict {
		switch pre := precondition.Check(s); pre {
		case Unmeetable:
			return Unmeetable
		case Met:
			verdict := mustMeet.Check(s)
			if verdict != Met {
				return Unmeetable
			}
			return Met
		default:
			return Unmet
		}
	}
	return &SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("once %q is met, must have %q", precondition.Description(), mustMeet.Description()),
	}
}

// ReadDiagnostics is an 'expectation' that is used to read diagnostics
// atomically. It is intended to be used with 'OnceMet'.
func ReadDiagnostics(fileName string, into *protocol.PublishDiagnosticsParams) *SimpleExpectation {
	check := func(s State) Verdict {
		diags, ok := s.diagnostics[fileName]
		if !ok {
			return Unmeetable
		}
		*into = *diags
		return Met
	}
	return &SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("read diagnostics for %q", fileName),
	}
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

// NoShowMessage asserts that the editor has not received a ShowMessage.
func NoShowMessage() SimpleExpectation {
	check := func(s State) Verdict {
		if len(s.showMessage) == 0 {
			return Met
		}
		return Unmeetable
	}
	return SimpleExpectation{
		check:       check,
		description: "no ShowMessage received",
	}
}

// ShownMessage asserts that the editor has received a ShownMessage with the
// given title.
func ShownMessage(title string) SimpleExpectation {
	check := func(s State) Verdict {
		for _, m := range s.showMessage {
			if strings.Contains(m.Message, title) {
				return Met
			}
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: "received ShowMessage",
	}
}

// ShowMessageRequest asserts that the editor has received a ShowMessageRequest
// with an action item that has the given title.
func ShowMessageRequest(title string) SimpleExpectation {
	check := func(s State) Verdict {
		if len(s.showMessageRequest) == 0 {
			return Unmet
		}
		// Only check the most recent one.
		m := s.showMessageRequest[len(s.showMessageRequest)-1]
		if len(m.Actions) == 0 || len(m.Actions) > 1 {
			return Unmet
		}
		if m.Actions[0].Title == title {
			return Met
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: "received ShowMessageRequest",
	}
}

// DoneWithOpen expects all didOpen notifications currently sent by the editor
// to be completely processed.
func (e *Env) DoneWithOpen() Expectation {
	opens := e.Editor.Stats().DidOpen
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), opens)
}

// DoneWithChange expects all didChange notifications currently sent by the
// editor to be completely processed.
func (e *Env) DoneWithChange() Expectation {
	changes := e.Editor.Stats().DidChange
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChange), changes)
}

// DoneWithSave expects all didSave notifications currently sent by the editor
// to be completely processed.
func (e *Env) DoneWithSave() Expectation {
	saves := e.Editor.Stats().DidSave
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidSave), saves)
}

// DoneWithChangeWatchedFiles expects all didChangeWatchedFiles notifications
// currently sent by the editor to be completely processed.
func (e *Env) DoneWithChangeWatchedFiles() Expectation {
	changes := e.Editor.Stats().DidChangeWatchedFiles
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), changes)
}

// DoneWithClose expects all didClose notifications currently sent by the
// editor to be completely processed.
func (e *Env) DoneWithClose() Expectation {
	changes := e.Editor.Stats().DidClose
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidClose), changes)
}

// CompletedWork expects a work item to have been completed >= atLeast times.
//
// Since the Progress API doesn't include any hidden metadata, we must use the
// progress notification title to identify the work we expect to be completed.
func CompletedWork(title string, atLeast uint64) SimpleExpectation {
	check := func(s State) Verdict {
		if s.completedWork[title] >= atLeast {
			return Met
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("completed work %q at least %d time(s)", title, atLeast),
	}
}

// OutstandingWork expects a work item to be outstanding. The given title must
// be an exact match, whereas the given msg must only be contained in the work
// item's message.
func OutstandingWork(title, msg string) SimpleExpectation {
	check := func(s State) Verdict {
		for _, work := range s.outstandingWork {
			if work.title == title && strings.Contains(work.msg, msg) {
				return Met
			}
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("outstanding work: %s", title),
	}
}

// LogExpectation is an expectation on the log messages received by the editor
// from gopls.
type LogExpectation struct {
	check       func([]*protocol.LogMessageParams) Verdict
	description string
}

// Check implements the Expectation interface.
func (e LogExpectation) Check(s State) Verdict {
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
func LogMatching(typ protocol.MessageType, re string, count int) LogExpectation {
	rec, err := regexp.Compile(re)
	if err != nil {
		panic(err)
	}
	check := func(msgs []*protocol.LogMessageParams) Verdict {
		var found int
		for _, msg := range msgs {
			if msg.Type == typ && rec.Match([]byte(msg.Message)) {
				found++
			}
		}
		if found == count {
			return Met
		}
		return Unmet
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
	check := func(msgs []*protocol.LogMessageParams) Verdict {
		for _, msg := range msgs {
			if msg.Type != typ {
				continue
			}
			if r == nil || r.Match([]byte(msg.Message)) {
				return Unmeetable
			}
		}
		return Met
	}
	return LogExpectation{
		check:       check,
		description: fmt.Sprintf("no log message matching %q", re),
	}
}

// RegistrationExpectation is an expectation on the capability registrations
// received by the editor from gopls.
type RegistrationExpectation struct {
	check       func([]*protocol.RegistrationParams) Verdict
	description string
}

// Check implements the Expectation interface.
func (e RegistrationExpectation) Check(s State) Verdict {
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
	check := func(params []*protocol.RegistrationParams) Verdict {
		for _, p := range params {
			for _, r := range p.Registrations {
				if rec.Match([]byte(r.Method)) {
					return Met
				}
			}
		}
		return Unmet
	}
	return RegistrationExpectation{
		check:       check,
		description: fmt.Sprintf("registration matching %q", re),
	}
}

// UnregistrationExpectation is an expectation on the capability
// unregistrations received by the editor from gopls.
type UnregistrationExpectation struct {
	check       func([]*protocol.UnregistrationParams) Verdict
	description string
}

// Check implements the Expectation interface.
func (e UnregistrationExpectation) Check(s State) Verdict {
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
	check := func(params []*protocol.UnregistrationParams) Verdict {
		for _, p := range params {
			for _, r := range p.Unregisterations {
				if rec.Match([]byte(r.Method)) {
					return Met
				}
			}
		}
		return Unmet
	}
	return UnregistrationExpectation{
		check:       check,
		description: fmt.Sprintf("unregistration matching %q", re),
	}
}

// A DiagnosticExpectation is a condition that must be met by the current set
// of diagnostics for a file.
type DiagnosticExpectation struct {
	// optionally, the position of the diagnostic and the regex used to calculate it.
	pos *fake.Pos
	re  string

	// optionally, the message that the diagnostic should contain.
	message string

	// whether the expectation is that the diagnostic is present, or absent.
	present bool

	// path is the scratch workdir-relative path to the file being asserted on.
	path string
}

// Check implements the Expectation interface.
func (e DiagnosticExpectation) Check(s State) Verdict {
	diags, ok := s.diagnostics[e.path]
	if !ok {
		if !e.present {
			return Met
		}
		return Unmet
	}

	found := false
	for _, d := range diags.Diagnostics {
		if e.pos != nil {
			if d.Range.Start.Line != uint32(e.pos.Line) || d.Range.Start.Character != uint32(e.pos.Column) {
				continue
			}
		}
		if e.message != "" {
			if !strings.Contains(d.Message, e.message) {
				continue
			}
		}
		found = true
		break
	}

	if found == e.present {
		return Met
	}
	return Unmet
}

// Description implements the Expectation interface.
func (e DiagnosticExpectation) Description() string {
	desc := e.path + ":"
	if !e.present {
		desc += " no"
	}
	desc += " diagnostic"
	if e.pos != nil {
		desc += fmt.Sprintf(" at {line:%d, column:%d}", e.pos.Line, e.pos.Column)
		if e.re != "" {
			desc += fmt.Sprintf(" (location of %q)", e.re)
		}
	}
	if e.message != "" {
		desc += fmt.Sprintf(" with message %q", e.message)
	}
	return desc
}

// EmptyDiagnostics asserts that empty diagnostics are sent for the
// workspace-relative path name.
func EmptyDiagnostics(name string) Expectation {
	check := func(s State) Verdict {
		if diags := s.diagnostics[name]; diags != nil && len(diags.Diagnostics) == 0 {
			return Met
		}
		return Unmet
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
	check := func(s State) Verdict {
		if _, ok := s.diagnostics[name]; !ok {
			return Met
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: "no diagnostics",
	}
}

// AnyDiagnosticAtCurrentVersion asserts that there is a diagnostic report for
// the current edited version of the buffer corresponding to the given
// workdir-relative pathname.
func (e *Env) AnyDiagnosticAtCurrentVersion(name string) Expectation {
	version := e.Editor.BufferVersion(name)
	check := func(s State) Verdict {
		diags, ok := s.diagnostics[name]
		if ok && diags.Version == int32(version) {
			return Met
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("any diagnostics at version %d", version),
	}
}

// DiagnosticAtRegexp expects that there is a diagnostic entry at the start
// position matching the regexp search string re in the buffer specified by
// name. Note that this currently ignores the end position.
func (e *Env) DiagnosticAtRegexp(name, re string) DiagnosticExpectation {
	e.T.Helper()
	pos := e.RegexpSearch(name, re)
	return DiagnosticExpectation{path: name, pos: &pos, re: re, present: true}
}

// DiagnosticAtRegexpWithMessage is like DiagnosticAtRegexp, but it also
// checks for the content of the diagnostic message,
func (e *Env) DiagnosticAtRegexpWithMessage(name, re, msg string) DiagnosticExpectation {
	e.T.Helper()
	pos := e.RegexpSearch(name, re)
	return DiagnosticExpectation{path: name, pos: &pos, re: re, present: true, message: msg}
}

// DiagnosticAt asserts that there is a diagnostic entry at the position
// specified by line and col, for the workdir-relative path name.
func DiagnosticAt(name string, line, col int) DiagnosticExpectation {
	return DiagnosticExpectation{path: name, pos: &fake.Pos{Line: line, Column: col}, present: true}
}

// NoDiagnosticAtRegexp expects that there is no diagnostic entry at the start
// position matching the regexp search string re in the buffer specified by
// name. Note that this currently ignores the end position.
// This should only be used in combination with OnceMet for a given condition,
// otherwise it may always succeed.
func (e *Env) NoDiagnosticAtRegexp(name, re string) DiagnosticExpectation {
	e.T.Helper()
	pos := e.RegexpSearch(name, re)
	return DiagnosticExpectation{path: name, pos: &pos, re: re, present: false}
}

// NoDiagnosticAt asserts that there is no diagnostic entry at the position
// specified by line and col, for the workdir-relative path name.
// This should only be used in combination with OnceMet for a given condition,
// otherwise it may always succeed.
func NoDiagnosticAt(name string, line, col int) DiagnosticExpectation {
	return DiagnosticExpectation{path: name, pos: &fake.Pos{Line: line, Column: col}, present: false}
}

// NoDiagnosticWithMessage asserts that there is no diagnostic entry with the
// given message.
//
// This should only be used in combination with OnceMet for a given condition,
// otherwise it may always succeed.
func NoDiagnosticWithMessage(name, msg string) DiagnosticExpectation {
	return DiagnosticExpectation{path: name, message: msg, present: false}
}

// GoSum asserts that a "go.sum is out of sync" diagnostic for the given module
// (as formatted in a go.mod file, e.g. "example.com v1.0.0") is present.
func (e *Env) GoSumDiagnostic(name, module string) Expectation {
	e.T.Helper()
	// In 1.16, go.sum diagnostics should appear on the relevant module. Earlier
	// errors have no information and appear on the module declaration.
	if testenv.Go1Point() >= 16 {
		return e.DiagnosticAtRegexpWithMessage(name, module, "go.sum is out of sync")
	} else {
		return e.DiagnosticAtRegexpWithMessage(name, `module`, "go.sum is out of sync")
	}
}
