// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"regexp"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
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
	InitialWorkspaceLoad = CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad), 1, false)
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

// Description returns e.description.
func (e SimpleExpectation) Description() string {
	return e.description
}

// OnceMet returns an Expectation that, once the precondition is met, asserts
// that mustMeet is met.
func OnceMet(precondition Expectation, mustMeets ...Expectation) *SimpleExpectation {
	check := func(s State) Verdict {
		switch pre := precondition.Check(s); pre {
		case Unmeetable:
			return Unmeetable
		case Met:
			for _, mustMeet := range mustMeets {
				verdict := mustMeet.Check(s)
				if verdict != Met {
					return Unmeetable
				}
			}
			return Met
		default:
			return Unmet
		}
	}
	description := describeExpectations(mustMeets...)
	return &SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("once %q is met, must have:\n%s", precondition.Description(), description),
	}
}

func describeExpectations(expectations ...Expectation) string {
	var descriptions []string
	for _, e := range expectations {
		descriptions = append(descriptions, e.Description())
	}
	return strings.Join(descriptions, "\n")
}

// AnyOf returns an expectation that is satisfied when any of the given
// expectations is met.
func AnyOf(anyOf ...Expectation) *SimpleExpectation {
	check := func(s State) Verdict {
		for _, e := range anyOf {
			verdict := e.Check(s)
			if verdict == Met {
				return Met
			}
		}
		return Unmet
	}
	description := describeExpectations(anyOf...)
	return &SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("Any of:\n%s", description),
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

// NoShownMessage asserts that the editor has not received a ShowMessage.
func NoShownMessage(subString string) SimpleExpectation {
	check := func(s State) Verdict {
		for _, m := range s.showMessage {
			if strings.Contains(m.Message, subString) {
				return Unmeetable
			}
		}
		return Met
	}
	return SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("no ShowMessage received containing %q", subString),
	}
}

// ShownMessage asserts that the editor has received a ShowMessageRequest
// containing the given substring.
func ShownMessage(containing string) SimpleExpectation {
	check := func(s State) Verdict {
		for _, m := range s.showMessage {
			if strings.Contains(m.Message, containing) {
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
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), opens, true)
}

// StartedChange expects that the server has at least started processing all
// didChange notifications sent from the client.
func (e *Env) StartedChange() Expectation {
	changes := e.Editor.Stats().DidChange
	return StartedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChange), changes)
}

// DoneWithChange expects all didChange notifications currently sent by the
// editor to be completely processed.
func (e *Env) DoneWithChange() Expectation {
	changes := e.Editor.Stats().DidChange
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChange), changes, true)
}

// DoneWithSave expects all didSave notifications currently sent by the editor
// to be completely processed.
func (e *Env) DoneWithSave() Expectation {
	saves := e.Editor.Stats().DidSave
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidSave), saves, true)
}

// DoneWithChangeWatchedFiles expects all didChangeWatchedFiles notifications
// currently sent by the editor to be completely processed.
func (e *Env) DoneWithChangeWatchedFiles() Expectation {
	changes := e.Editor.Stats().DidChangeWatchedFiles
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), changes, true)
}

// DoneWithClose expects all didClose notifications currently sent by the
// editor to be completely processed.
func (e *Env) DoneWithClose() Expectation {
	changes := e.Editor.Stats().DidClose
	return CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidClose), changes, true)
}

// StartedWork expect a work item to have been started >= atLeast times.
//
// See CompletedWork.
func StartedWork(title string, atLeast uint64) SimpleExpectation {
	check := func(s State) Verdict {
		if s.startedWork[title] >= atLeast {
			return Met
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("started work %q at least %d time(s)", title, atLeast),
	}
}

// CompletedWork expects a work item to have been completed >= atLeast times.
//
// Since the Progress API doesn't include any hidden metadata, we must use the
// progress notification title to identify the work we expect to be completed.
func CompletedWork(title string, count uint64, atLeast bool) SimpleExpectation {
	check := func(s State) Verdict {
		if s.completedWork[title] == count || atLeast && s.completedWork[title] > count {
			return Met
		}
		return Unmet
	}
	desc := fmt.Sprintf("completed work %q %v times", title, count)
	if atLeast {
		desc = fmt.Sprintf("completed work %q at least %d time(s)", title, count)
	}
	return SimpleExpectation{
		check:       check,
		description: desc,
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
		description: fmt.Sprintf("outstanding work: %q containing %q", title, msg),
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
// of type typ matching the regexp re a certain number of times.
//
// The count argument specifies the expected number of matching logs. If
// atLeast is set, this is a lower bound, otherwise there must be exactly cound
// matching logs.
func LogMatching(typ protocol.MessageType, re string, count int, atLeast bool) LogExpectation {
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
		// Check for an exact or "at least" match.
		if found == count || (found >= count && atLeast) {
			return Met
		}
		return Unmet
	}
	desc := fmt.Sprintf("log message matching %q expected %v times", re, count)
	if atLeast {
		desc = fmt.Sprintf("log message matching %q expected at least %v times", re, count)
	}
	return LogExpectation{
		check:       check,
		description: desc,
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

// FileWatchMatching expects that a file registration matches re.
func FileWatchMatching(re string) SimpleExpectation {
	return SimpleExpectation{
		check:       checkFileWatch(re, Met, Unmet),
		description: fmt.Sprintf("file watch matching %q", re),
	}
}

// NoFileWatchMatching expects that no file registration matches re.
func NoFileWatchMatching(re string) SimpleExpectation {
	return SimpleExpectation{
		check:       checkFileWatch(re, Unmet, Met),
		description: fmt.Sprintf("no file watch matching %q", re),
	}
}

func checkFileWatch(re string, onMatch, onNoMatch Verdict) func(State) Verdict {
	rec := regexp.MustCompile(re)
	return func(s State) Verdict {
		r := s.registeredCapabilities["workspace/didChangeWatchedFiles"]
		watchers := jsonProperty(r.RegisterOptions, "watchers").([]interface{})
		for _, watcher := range watchers {
			pattern := jsonProperty(watcher, "globPattern").(string)
			if rec.MatchString(pattern) {
				return onMatch
			}
		}
		return onNoMatch
	}
}

// jsonProperty extracts a value from a path of JSON property names, assuming
// the default encoding/json unmarshaling to the empty interface (i.e.: that
// JSON objects are unmarshalled as map[string]interface{})
//
// For example, if obj is unmarshalled from the following json:
//
//	{
//		"foo": { "bar": 3 }
//	}
//
// Then jsonProperty(obj, "foo", "bar") will be 3.
func jsonProperty(obj interface{}, path ...string) interface{} {
	if len(path) == 0 || obj == nil {
		return obj
	}
	m := obj.(map[string]interface{})
	return jsonProperty(m[path[0]], path[1:]...)
}

// RegistrationMatching asserts that the client has received a capability
// registration matching the given regexp.
//
// TODO(rfindley): remove this once TestWatchReplaceTargets has been revisited.
//
// Deprecated: use (No)FileWatchMatching
func RegistrationMatching(re string) SimpleExpectation {
	rec := regexp.MustCompile(re)
	check := func(s State) Verdict {
		for _, p := range s.registrations {
			for _, r := range p.Registrations {
				if rec.Match([]byte(r.Method)) {
					return Met
				}
			}
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("registration matching %q", re),
	}
}

// UnregistrationMatching asserts that the client has received an
// unregistration whose ID matches the given regexp.
func UnregistrationMatching(re string) SimpleExpectation {
	rec := regexp.MustCompile(re)
	check := func(s State) Verdict {
		for _, p := range s.unregistrations {
			for _, r := range p.Unregisterations {
				if rec.Match([]byte(r.Method)) {
					return Met
				}
			}
		}
		return Unmet
	}
	return SimpleExpectation{
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

	// optionally, the diagnostic source
	source string
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
		if e.source != "" && e.source != d.Source {
			continue
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
	if e.source != "" {
		desc += fmt.Sprintf(" from source %q", e.source)
	}
	return desc
}

// NoOutstandingDiagnostics asserts that the workspace has no outstanding
// diagnostic messages.
func NoOutstandingDiagnostics() Expectation {
	check := func(s State) Verdict {
		for _, diags := range s.diagnostics {
			if len(diags.Diagnostics) > 0 {
				return Unmet
			}
		}
		return Met
	}
	return SimpleExpectation{
		check:       check,
		description: "no outstanding diagnostics",
	}
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
		description: fmt.Sprintf("empty diagnostics for %q", name),
	}
}

// EmptyOrNoDiagnostics asserts that either no diagnostics are sent for the
// workspace-relative path name, or empty diagnostics are sent.
// TODO(rFindley): this subtlety shouldn't be necessary. Gopls should always
// send at least one diagnostic set for open files.
func EmptyOrNoDiagnostics(name string) Expectation {
	check := func(s State) Verdict {
		if diags := s.diagnostics[name]; diags == nil || len(diags.Diagnostics) == 0 {
			return Met
		}
		return Unmet
	}
	return SimpleExpectation{
		check:       check,
		description: fmt.Sprintf("empty or no diagnostics for %q", name),
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
		description: fmt.Sprintf("no diagnostics for %q", name),
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

// DiagnosticAtRegexpFromSource expects a diagnostic at the first position
// matching re, from the given source.
func (e *Env) DiagnosticAtRegexpFromSource(name, re, source string) DiagnosticExpectation {
	e.T.Helper()
	pos := e.RegexpSearch(name, re)
	return DiagnosticExpectation{path: name, pos: &pos, re: re, present: true, source: source}
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

// NoDiagnosticWithMessage asserts that there is no diagnostic entry with the
// given message.
//
// This should only be used in combination with OnceMet for a given condition,
// otherwise it may always succeed.
func NoDiagnosticWithMessage(name, msg string) DiagnosticExpectation {
	return DiagnosticExpectation{path: name, message: msg, present: false}
}

// GoSumDiagnostic asserts that a "go.sum is out of sync" diagnostic for the
// given module (as formatted in a go.mod file, e.g. "example.com v1.0.0") is
// present.
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
