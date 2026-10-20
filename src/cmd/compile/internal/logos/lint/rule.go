// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Rule is one lint rule.
type Rule struct {
	// ID is "group::name", for example "slop::discarded_error". The
	// group part must equal Group.
	ID string
	// Group is the group whose logos.toml severity governs the rule.
	Group Group
	// Default is the severity the rule reports at when logos.toml names no
	// severity for its group, for a rule that SPEC Sec 1.2 sets apart from
	// its group, such as a deny rule in the forbid group correctness. The
	// zero value, Allow, selects the default severity of the group.
	Default Severity
	// Go is true for the language-neutral rules, which logos lint also
	// runs on Go packages. Every other rule runs on loGo/s packages only.
	Go bool
	// Check inspects Pass.Files and reports findings through Pass. A nil
	// Check marks a rule enforced under the same ID by an earlier phase,
	// such as types2: the entry still carries its group and scope, so
	// severity resolution, the allow_forbidden_rule check, and
	// TestRuleScopes cover it.
	Check func(*Pass)
}

// Provenance records where the inferred type of a finding came from. The
// zero value is valid and encodes as empty strings and line 0.
type Provenance struct {
	OriginExpr   string
	OriginLine   int
	InferredType string
}

// Fix is one autofix candidate offered in the JSON output.
type Fix struct {
	Description string
	Replacement string
	// InsertBeforeLine is the 1-based line a candidate that inserts text
	// goes before; 0 means the candidate replaces the diagnostic span.
	InsertBeforeLine int
}

// Diagnostic is one finding of a rule. Pos and End bound the flagged
// source; End may equal Pos. Reportf takes both from the syntax package,
// whose end position of a call stops before its closing parenthesis: a rule
// that flags a call builds its span with Pass.CallEnd.
type Diagnostic struct {
	Pos        syntax.Pos
	End        syntax.Pos
	Message    string
	Provenance Provenance
	Fixes      []Fix
}

// Result is a Diagnostic after severity resolution and directive
// filtering. Severity is never Allow.
type Result struct {
	Rule       string
	Severity   Severity
	Pos        syntax.Pos
	End        syntax.Pos
	Message    string
	Provenance Provenance
	Fixes      []Fix
}

// Input is the checked package handed to Run.
type Input struct {
	Pkg   *types2.Package
	Info  *types2.Info
	Files []*syntax.File
	// Sources holds the original bytes of each file. A file with no entry
	// has no comments to scan, so no directive applies to it.
	Sources map[*syntax.File][]byte
	// Kind selects the rules that run and the Go cap.
	Kind Kind
	// Policy is the resolved logos.toml [lints] table; nil means defaults.
	Policy *Policy
	// Reported holds the findings of the rules that a phase before Run
	// enforces, such as the warnings of types2; Run resolves them with the
	// findings of the rules.
	Reported []Reported
}

// Pass is the view of one package a rule's Check receives.
type Pass struct {
	Pkg    *types2.Package
	Info   *types2.Info
	Files  []*syntax.File
	Kind   Kind
	Policy *Policy
	in     *Input
	diags  []Diagnostic
}

// Report records a finding of the rule being checked.
func (p *Pass) Report(d Diagnostic) {
	if !d.End.IsKnown() {
		d.End = d.Pos
	}
	p.diags = append(p.diags, d)
}

// Reportf records a finding spanning node n.
func (p *Pass) Reportf(n syntax.Node, format string, args ...any) {
	p.Report(Diagnostic{
		Pos:     syntax.StartPos(n),
		End:     syntax.EndPos(n),
		Message: fmt.Sprintf(format, args...),
	})
}

// Source returns the original bytes of f, or nil when none were kept.
func (p *Pass) Source(f *syntax.File) []byte {
	return p.in.Sources[f]
}

// CallEnd returns the position just after the closing parenthesis of call,
// a call in file f. The syntax package's end position of a call stops before
// that parenthesis, so a span built on it is one byte short. Without the
// source of f it returns that approximate end.
func (p *Pass) CallEnd(f *syntax.File, call *syntax.CallExpr) syntax.Pos {
	mode := syntax.Mode(0)
	if p.Kind == KindLgo {
		mode = syntax.LogosMode
	}
	if end, ok := syntax.ParenEnd(p.Source(f), call.Pos(), mode); ok {
		return end
	}
	return syntax.EndPos(call)
}

// IsTestFile reports whether f is a test source: a name ending in
// _test.go, or _test with a loGo/s extension.
func IsTestFile(f *syntax.File) bool {
	name := f.Pos().RelFilename()
	stem, _, found := strings.CutLast(name, ".")
	if !found {
		stem = name
	}
	return strings.HasSuffix(stem, "_test")
}

// TypeOf returns the type recorded on e by the checker, or nil.
func TypeOf(e syntax.Expr) types2.Type {
	if e == nil {
		return nil
	}
	return e.GetTypeInfo().Type
}
