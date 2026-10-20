// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// Reported is a finding of a rule that a phase before Run enforces under the
// same ID: the checker reports suspicious::unapplied_struct_default and
// suspicious::side_effect_before_sort while it checks the package. The rule
// is registered with no Check, and Run resolves the finding like one from a
// Check, so the severity of logos.toml and //logos:allow apply to it. Pos is
// the position of the finding, and Message is the text after the rule tag.
type Reported struct {
	Rule    string
	Pos     syntax.Pos
	Message string
}

// ReportedOf returns the finding for a warn-severity loGo/s diagnostic of
// types2, whose text starts with its rule tag, "[group::name] ". It reports
// false for any other error, which the caller keeps as an error.
func ReportedOf(err error) (Reported, bool) {
	e, ok := err.(types2.Error)
	if !ok || !types2.LogosWarnOnly(e) {
		return Reported{}, false
	}
	tag, message, ok := strings.Cut(e.Msg, "] ")
	rule, found := strings.CutPrefix(tag, "[")
	if !ok || !found {
		return Reported{}, false
	}
	return Reported{Rule: rule, Pos: e.Pos, Message: message}, true
}

// Resolve applies severity resolution and the //logos:allow directives to
// the findings in in.Reported alone and returns the survivors ordered by
// position. It needs only in.Files, in.Sources, in.Kind, and in.Policy, so
// the compiler can report the findings of a package that failed to check,
// and of a compile with no lint pass, the way Run reports them.
func Resolve(in *Input) []Result {
	policy := policyOf(in)
	dirs := collectDirectives(in)
	var out []Result
	for _, r := range registry {
		r.Check = nil
		out = append(out, runRule(in, policy, dirs, r)...)
	}
	slices.SortStableFunc(out, compareResults)
	return out
}

// reportedBy returns the findings of rule id as diagnostics.
func (in *Input) reportedBy(id string) []Diagnostic {
	var out []Diagnostic
	for _, r := range in.Reported {
		if r.Rule == id {
			out = append(out, Diagnostic{Pos: r.Pos, End: r.Pos, Message: r.Message})
		}
	}
	return out
}
