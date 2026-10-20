// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lint is the loGo/s compiler linter (SPEC Sec 1.2, M9).
//
// The compiler calls Run once per package, after the first types2 pass and
// before lowering, so rules see loGo/s syntax and types. A package is
// KindLgo or KindGo: a loGo/s package runs every rule on every compile; a
// Go package runs only the rules with Rule.Go set, and only under logos
// lint (the compile flag -logos.lintonly). Severities come from the
// logos.toml [lints] table, which the go command writes as JSON and passes
// as -logos.cfg (see ParsePolicy). Run applies four resolution steps to a
// finding of rule R:
//
//   - the group severity of R is read from the policy; when the policy
//     names none, the Default of R applies if it has one, else the group
//     default; a Go package lowers forbid to deny (the Go cap);
//   - allow drops the finding, and the rule does not run;
//   - forbid always reports, and a //logos:allow(R) directive naming it in
//     a loGo/s package is itself reported as correctness::allow_forbidden_rule;
//   - deny and warn are silenced by a //logos:allow(R) comment on the line
//     directly above a statement or declaration containing the finding.
//
// # Adding a rule
//
// A rule is one new file plus one registry line:
//
//  1. Create rule_<name>.go with a package-level var of type Rule, with ID
//     "group::name", the matching Group, Go set exactly when the rule is
//     language-neutral (SPEC Sec 1.2 lists the 13), and a Check that walks
//     Pass.Files and calls Pass.Report or Pass.Reportf. Types come from
//     TypeOf and Pass.Info; Check must not mutate the tree.
//  2. Add the var to registry in registry.go.
//  3. Add the attack file test/logos/lint/<rule>.lgo (and <rule>_go.go for a
//     language-neutral rule).
//
// Nothing else changes: severities, directives, JSON output, and the Go
// scope all come from this package. TestRuleScopes already lists the 21
// SPEC rules and checks each registered rule against its scope. A rule that
// an earlier phase already enforces under the same ID registers with a nil
// Check: types2 reports nil_deref_unproven and unsupervised_goroutine as
// errors, and hands suspicious::unapplied_struct_default and
// suspicious::side_effect_before_sort to Run in Input.Reported, which
// resolves them like the findings of a Check (see Resolve for the compile
// that fails before Run).
package lint

import (
	"cmp"
	"slices"
)

// Rules returns the registered rules in registry order.
func Rules() []Rule {
	return slices.Clone(registry)
}

// Lookup returns the registered rule with the given ID.
func Lookup(id string) (Rule, bool) {
	for _, r := range registry {
		if r.ID == id {
			return r, true
		}
	}
	return Rule{}, false
}

// Run applies the registered rules to one checked package and returns the
// findings that survive severity resolution and directives, ordered by
// position. The findings in in.Reported resolve with those of the rules. A
// nil in.Policy means DefaultPolicy.
func Run(in *Input) []Result {
	policy := policyOf(in)
	dirs := collectDirectives(in)
	var out []Result
	for _, r := range registry {
		out = append(out, runRule(in, policy, dirs, r)...)
	}
	out = append(out, dirs.forbidden(policy, in.Kind)...)
	slices.SortStableFunc(out, compareResults)
	return out
}

// runRule runs one rule and resolves the severity of each finding, those of
// its Check and those an earlier phase reported under its ID.
func runRule(in *Input, policy *Policy, dirs *directives, r Rule) []Result {
	level := policy.Level(r.ID, in.Kind)
	if level == Allow || (in.Kind == KindGo && !r.Go) {
		return nil
	}
	diags := in.reportedBy(r.ID)
	if r.Check != nil {
		pass := &Pass{Pkg: in.Pkg, Info: in.Info, Files: in.Files, Kind: in.Kind, Policy: policy, in: in}
		r.Check(pass)
		diags = append(diags, pass.diags...)
	}
	var out []Result
	for _, d := range diags {
		if level != Forbid && dirs.silences(r.ID, d.Pos) {
			continue
		}
		out = append(out, Result{
			Rule:       r.ID,
			Severity:   level,
			Pos:        d.Pos,
			End:        d.End,
			Message:    d.Message,
			Provenance: d.Provenance,
			Fixes:      d.Fixes,
		})
	}
	return out
}

// compareResults orders results by file position, then rule ID.
func compareResults(a, b Result) int {
	return cmp.Or(
		cmp.Compare(a.Pos.RelFilename(), b.Pos.RelFilename()),
		cmp.Compare(a.Pos.Line(), b.Pos.Line()),
		cmp.Compare(a.Pos.Col(), b.Pos.Col()),
		cmp.Compare(a.Rule, b.Rule),
	)
}
