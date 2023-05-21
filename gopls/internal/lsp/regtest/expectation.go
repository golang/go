// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

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

// An Expectation is an expected property of the state of the LSP client.
// The Check function reports whether the property is met.
//
// Expectations are combinators. By composing them, tests may express
// complex expectations in terms of simpler ones.
//
// TODO(rfindley): as expectations are combined, it becomes harder to identify
// why they failed. A better signature for Check would be
//
//	func(State) (Verdict, string)
//
// returning a reason for the verdict that can be composed similarly to
// descriptions.
type Expectation struct {
	Check func(State) Verdict

	// Description holds a noun-phrase identifying what the expectation checks.
	//
	// TODO(rfindley): revisit existing descriptions to ensure they compose nicely.
	Description string
}

// OnceMet returns an Expectation that, once the precondition is met, asserts
// that mustMeet is met.
func OnceMet(precondition Expectation, mustMeets ...Expectation) Expectation {
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
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("once %q is met, must have:\n%s", precondition.Description, description),
	}
}

func describeExpectations(expectations ...Expectation) string {
	var descriptions []string
	for _, e := range expectations {
		descriptions = append(descriptions, e.Description)
	}
	return strings.Join(descriptions, "\n")
}

// AnyOf returns an expectation that is satisfied when any of the given
// expectations is met.
func AnyOf(anyOf ...Expectation) Expectation {
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
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("Any of:\n%s", description),
	}
}

// AllOf expects that all given expectations are met.
//
// TODO(rfindley): the problem with these types of combinators (OnceMet, AnyOf
// and AllOf) is that we lose the information of *why* they failed: the Awaiter
// is not smart enough to look inside.
//
// Refactor the API such that the Check function is responsible for explaining
// why an expectation failed. This should allow us to significantly improve
// test output: we won't need to summarize state at all, as the verdict
// explanation itself should describe clearly why the expectation not met.
func AllOf(allOf ...Expectation) Expectation {
	check := func(s State) Verdict {
		verdict := Met
		for _, e := range allOf {
			if v := e.Check(s); v > verdict {
				verdict = v
			}
		}
		return verdict
	}
	description := describeExpectations(allOf...)
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("All of:\n%s", description),
	}
}

// ReadDiagnostics is an Expectation that stores the current diagnostics for
// fileName in into, whenever it is evaluated.
//
// It can be used in combination with OnceMet or AfterChange to capture the
// state of diagnostics when other expectations are satisfied.
func ReadDiagnostics(fileName string, into *protocol.PublishDiagnosticsParams) Expectation {
	check := func(s State) Verdict {
		diags, ok := s.diagnostics[fileName]
		if !ok {
			return Unmeetable
		}
		*into = *diags
		return Met
	}
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("read diagnostics for %q", fileName),
	}
}

// ReadAllDiagnostics is an expectation that stores all published diagnostics
// into the provided map, whenever it is evaluated.
//
// It can be used in combination with OnceMet or AfterChange to capture the
// state of diagnostics when other expectations are satisfied.
func ReadAllDiagnostics(into *map[string]*protocol.PublishDiagnosticsParams) Expectation {
	check := func(s State) Verdict {
		allDiags := make(map[string]*protocol.PublishDiagnosticsParams)
		for name, diags := range s.diagnostics {
			allDiags[name] = diags
		}
		*into = allDiags
		return Met
	}
	return Expectation{
		Check:       check,
		Description: "read all diagnostics",
	}
}

// NoOutstandingWork asserts that there is no work initiated using the LSP
// $/progress API that has not completed.
func NoOutstandingWork() Expectation {
	check := func(s State) Verdict {
		if len(s.outstandingWork()) == 0 {
			return Met
		}
		return Unmet
	}
	return Expectation{
		Check:       check,
		Description: "no outstanding work",
	}
}

// NoShownMessage asserts that the editor has not received a ShowMessage.
func NoShownMessage(subString string) Expectation {
	check := func(s State) Verdict {
		for _, m := range s.showMessage {
			if strings.Contains(m.Message, subString) {
				return Unmeetable
			}
		}
		return Met
	}
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("no ShowMessage received containing %q", subString),
	}
}

// ShownMessage asserts that the editor has received a ShowMessageRequest
// containing the given substring.
func ShownMessage(containing string) Expectation {
	check := func(s State) Verdict {
		for _, m := range s.showMessage {
			if strings.Contains(m.Message, containing) {
				return Met
			}
		}
		return Unmet
	}
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("received window/showMessage containing %q", containing),
	}
}

// ShowMessageRequest asserts that the editor has received a ShowMessageRequest
// with an action item that has the given title.
func ShowMessageRequest(title string) Expectation {
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
	return Expectation{
		Check:       check,
		Description: "received ShowMessageRequest",
	}
}

// DoneDiagnosingChanges expects that diagnostics are complete from common
// change notifications: didOpen, didChange, didSave, didChangeWatchedFiles,
// and didClose.
//
// This can be used when multiple notifications may have been sent, such as
// when a didChange is immediately followed by a didSave. It is insufficient to
// simply await NoOutstandingWork, because the LSP client has no control over
// when the server starts processing a notification. Therefore, we must keep
// track of
func (e *Env) DoneDiagnosingChanges() Expectation {
	stats := e.Editor.Stats()
	statsBySource := map[lsp.ModificationSource]uint64{
		lsp.FromDidOpen:               stats.DidOpen,
		lsp.FromDidChange:             stats.DidChange,
		lsp.FromDidSave:               stats.DidSave,
		lsp.FromDidChangeWatchedFiles: stats.DidChangeWatchedFiles,
		lsp.FromDidClose:              stats.DidClose,
	}

	var expected []lsp.ModificationSource
	for k, v := range statsBySource {
		if v > 0 {
			expected = append(expected, k)
		}
	}

	// Sort for stability.
	sort.Slice(expected, func(i, j int) bool {
		return expected[i] < expected[j]
	})

	var all []Expectation
	for _, source := range expected {
		all = append(all, CompletedWork(lsp.DiagnosticWorkTitle(source), statsBySource[source], true))
	}

	return AllOf(all...)
}

// AfterChange expects that the given expectations will be met after all
// state-changing notifications have been processed by the server.
//
// It awaits the completion of all anticipated work before checking the given
// expectations.
func (e *Env) AfterChange(expectations ...Expectation) {
	e.T.Helper()
	e.OnceMet(
		e.DoneDiagnosingChanges(),
		expectations...,
	)
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

// StartedChangeWatchedFiles expects that the server has at least started
// processing all didChangeWatchedFiles notifications sent from the client.
func (e *Env) StartedChangeWatchedFiles() Expectation {
	changes := e.Editor.Stats().DidChangeWatchedFiles
	return StartedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), changes)
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
func StartedWork(title string, atLeast uint64) Expectation {
	check := func(s State) Verdict {
		if s.startedWork()[title] >= atLeast {
			return Met
		}
		return Unmet
	}
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("started work %q at least %d time(s)", title, atLeast),
	}
}

// CompletedWork expects a work item to have been completed >= atLeast times.
//
// Since the Progress API doesn't include any hidden metadata, we must use the
// progress notification title to identify the work we expect to be completed.
func CompletedWork(title string, count uint64, atLeast bool) Expectation {
	check := func(s State) Verdict {
		completed := s.completedWork()
		if completed[title] == count || atLeast && completed[title] > count {
			return Met
		}
		return Unmet
	}
	desc := fmt.Sprintf("completed work %q %v times", title, count)
	if atLeast {
		desc = fmt.Sprintf("completed work %q at least %d time(s)", title, count)
	}
	return Expectation{
		Check:       check,
		Description: desc,
	}
}

type WorkStatus struct {
	// Last seen message from either `begin` or `report` progress.
	Msg string
	// Message sent with `end` progress message.
	EndMsg string
}

// CompletedProgress expects that workDone progress is complete for the given
// progress token. When non-nil WorkStatus is provided, it will be filled
// when the expectation is met.
//
// If the token is not a progress token that the client has seen, this
// expectation is Unmeetable.
func CompletedProgress(token protocol.ProgressToken, into *WorkStatus) Expectation {
	check := func(s State) Verdict {
		work, ok := s.work[token]
		if !ok {
			return Unmeetable // TODO(rfindley): refactor to allow the verdict to explain this result
		}
		if work.complete {
			if into != nil {
				into.Msg = work.msg
				into.EndMsg = work.endMsg
			}
			return Met
		}
		return Unmet
	}
	desc := fmt.Sprintf("completed work for token %v", token)
	return Expectation{
		Check:       check,
		Description: desc,
	}
}

// OutstandingWork expects a work item to be outstanding. The given title must
// be an exact match, whereas the given msg must only be contained in the work
// item's message.
func OutstandingWork(title, msg string) Expectation {
	check := func(s State) Verdict {
		for _, work := range s.work {
			if work.complete {
				continue
			}
			if work.title == title && strings.Contains(work.msg, msg) {
				return Met
			}
		}
		return Unmet
	}
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("outstanding work: %q containing %q", title, msg),
	}
}

// NoErrorLogs asserts that the client has not received any log messages of
// error severity.
func NoErrorLogs() Expectation {
	return NoLogMatching(protocol.Error, "")
}

// LogMatching asserts that the client has received a log message
// of type typ matching the regexp re a certain number of times.
//
// The count argument specifies the expected number of matching logs. If
// atLeast is set, this is a lower bound, otherwise there must be exactly count
// matching logs.
func LogMatching(typ protocol.MessageType, re string, count int, atLeast bool) Expectation {
	rec, err := regexp.Compile(re)
	if err != nil {
		panic(err)
	}
	check := func(state State) Verdict {
		var found int
		for _, msg := range state.logs {
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
	return Expectation{
		Check:       check,
		Description: desc,
	}
}

// NoLogMatching asserts that the client has not received a log message
// of type typ matching the regexp re. If re is an empty string, any log
// message is considered a match.
func NoLogMatching(typ protocol.MessageType, re string) Expectation {
	var r *regexp.Regexp
	if re != "" {
		var err error
		r, err = regexp.Compile(re)
		if err != nil {
			panic(err)
		}
	}
	check := func(state State) Verdict {
		for _, msg := range state.logs {
			if msg.Type != typ {
				continue
			}
			if r == nil || r.Match([]byte(msg.Message)) {
				return Unmeetable
			}
		}
		return Met
	}
	return Expectation{
		Check:       check,
		Description: fmt.Sprintf("no log message matching %q", re),
	}
}

// FileWatchMatching expects that a file registration matches re.
func FileWatchMatching(re string) Expectation {
	return Expectation{
		Check:       checkFileWatch(re, Met, Unmet),
		Description: fmt.Sprintf("file watch matching %q", re),
	}
}

// NoFileWatchMatching expects that no file registration matches re.
func NoFileWatchMatching(re string) Expectation {
	return Expectation{
		Check:       checkFileWatch(re, Unmet, Met),
		Description: fmt.Sprintf("no file watch matching %q", re),
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

// Diagnostics asserts that there is at least one diagnostic matching the given
// filters.
func Diagnostics(filters ...DiagnosticFilter) Expectation {
	check := func(s State) Verdict {
		diags := flattenDiagnostics(s)
		for _, filter := range filters {
			var filtered []flatDiagnostic
			for _, d := range diags {
				if filter.check(d.name, d.diag) {
					filtered = append(filtered, d)
				}
			}
			if len(filtered) == 0 {
				// TODO(rfindley): if/when expectations describe their own failure, we
				// can provide more useful information here as to which filter caused
				// the failure.
				return Unmet
			}
			diags = filtered
		}
		return Met
	}
	var descs []string
	for _, filter := range filters {
		descs = append(descs, filter.desc)
	}
	return Expectation{
		Check:       check,
		Description: "any diagnostics " + strings.Join(descs, ", "),
	}
}

// NoDiagnostics asserts that there are no diagnostics matching the given
// filters. Notably, if no filters are supplied this assertion checks that
// there are no diagnostics at all, for any file.
func NoDiagnostics(filters ...DiagnosticFilter) Expectation {
	check := func(s State) Verdict {
		diags := flattenDiagnostics(s)
		for _, filter := range filters {
			var filtered []flatDiagnostic
			for _, d := range diags {
				if filter.check(d.name, d.diag) {
					filtered = append(filtered, d)
				}
			}
			diags = filtered
		}
		if len(diags) > 0 {
			return Unmet
		}
		return Met
	}
	var descs []string
	for _, filter := range filters {
		descs = append(descs, filter.desc)
	}
	return Expectation{
		Check:       check,
		Description: "no diagnostics " + strings.Join(descs, ", "),
	}
}

type flatDiagnostic struct {
	name string
	diag protocol.Diagnostic
}

func flattenDiagnostics(state State) []flatDiagnostic {
	var result []flatDiagnostic
	for name, diags := range state.diagnostics {
		for _, diag := range diags.Diagnostics {
			result = append(result, flatDiagnostic{name, diag})
		}
	}
	return result
}

// -- Diagnostic filters --

// A DiagnosticFilter filters the set of diagnostics, for assertion with
// Diagnostics or NoDiagnostics.
type DiagnosticFilter struct {
	desc  string
	check func(name string, _ protocol.Diagnostic) bool
}

// ForFile filters to diagnostics matching the sandbox-relative file name.
func ForFile(name string) DiagnosticFilter {
	return DiagnosticFilter{
		desc: fmt.Sprintf("for file %q", name),
		check: func(diagName string, _ protocol.Diagnostic) bool {
			return diagName == name
		},
	}
}

// FromSource filters to diagnostics matching the given diagnostics source.
func FromSource(source string) DiagnosticFilter {
	return DiagnosticFilter{
		desc: fmt.Sprintf("with source %q", source),
		check: func(_ string, d protocol.Diagnostic) bool {
			return d.Source == source
		},
	}
}

// AtRegexp filters to diagnostics in the file with sandbox-relative path name,
// at the first position matching the given regexp pattern.
//
// TODO(rfindley): pass in the editor to expectations, so that they may depend
// on editor state and AtRegexp can be a function rather than a method.
func (e *Env) AtRegexp(name, pattern string) DiagnosticFilter {
	loc := e.RegexpSearch(name, pattern)
	return DiagnosticFilter{
		desc: fmt.Sprintf("at the first position matching %#q in %q", pattern, name),
		check: func(diagName string, d protocol.Diagnostic) bool {
			return diagName == name && d.Range.Start == loc.Range.Start
		},
	}
}

// AtPosition filters to diagnostics at location name:line:character, for a
// sandbox-relative path name.
//
// Line and character are 0-based, and character measures UTF-16 codes.
//
// Note: prefer the more readable AtRegexp.
func AtPosition(name string, line, character uint32) DiagnosticFilter {
	pos := protocol.Position{Line: line, Character: character}
	return DiagnosticFilter{
		desc: fmt.Sprintf("at %s:%d:%d", name, line, character),
		check: func(diagName string, d protocol.Diagnostic) bool {
			return diagName == name && d.Range.Start == pos
		},
	}
}

// WithMessage filters to diagnostics whose message contains the given
// substring.
func WithMessage(substring string) DiagnosticFilter {
	return DiagnosticFilter{
		desc: fmt.Sprintf("with message containing %q", substring),
		check: func(_ string, d protocol.Diagnostic) bool {
			return strings.Contains(d.Message, substring)
		},
	}
}
