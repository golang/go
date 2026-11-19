// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker_test

import (
	"go/analysis"
	"slices"
	"testing"

	"internal/analysis/gotestchecker"
)

// TestAnalyzers checks the graph of analyzers reachable from the
// variables passed to Main: its canonical order (each analyzer after
// its prerequisites, and each appearing once), and the exclusion of
// invalid variables and invalid members, along with the Requires
// closure of each, without affecting their valid siblings.
func TestAnalyzers(t *testing.T) {
	// Requires: a -> {b, c}, b -> d, c -> d.
	d := newAnalyzer("d")
	b := newAnalyzer("b", d)
	c := newAnalyzer("c", d)
	a := newAnalyzer("a", b, c)

	// Invalid analyzers. Each is invalid only because of its own
	// properties or those of its Requires closure.
	p := newAnalyzer("p") // a valid prerequisite of invalid analyzers
	noRun := newAnalyzer("norun", p)
	noRun.Run = nil
	badName := newAnalyzer("bad-name", p)
	cycle1 := newAnalyzer("cycle1")
	cycle2 := newAnalyzer("cycle2", cycle1)
	cycle1.Requires = []*analysis.Analyzer{cycle2}
	nilReq := newAnalyzer("nilreq", nil)
	valueFact := newAnalyzer("valuefact")
	valueFact.FactTypes = []analysis.Fact{nonPointerFact{}}
	badReq := newAnalyzer("badreq", noRun) // invalid by way of its prerequisite

	for _, test := range []struct {
		name string
		vars []any // values of the variables
		want []string
	}{
		{"single", {d}, {"d"}},
		{"closure", {a}, {"d", "b", "c", "a"}},
		{"shared prerequisite", {b, c}, {"d", "b", "c"}},
		{"var order", {c, a}, {"d", "c", "b", "a"}},
		{"repeated var", {a, a}, {"d", "b", "c", "a"}},
		{"suite", {[]*analysis.Analyzer{c, b}}, {"d", "c", "b"}},
		{"suite with repeated element", {[]*analysis.Analyzer{b, b, d}}, {"d", "b"}},
		{"suite and analyzer", {[]*analysis.Analyzer{b}, a}, {"d", "b", "c", "a"}},
		{"empty suite", {[]*analysis.Analyzer{}}, nil},

		// Invalid members are dropped, with their closures,
		// but not their valid siblings.
		{"nil Run", {noRun, b}, {"d", "b"}},
		{"invalid name", {badName, b}, {"d", "b"}},
		{"cycle", {cycle1, b}, {"d", "b"}},
		{"nil prerequisite", {nilReq, b}, {"d", "b"}},
		{"non-pointer fact type", {valueFact, b}, {"d", "b"}},
		{"invalid prerequisite", {badReq, b}, {"d", "b"}},
		{"invalid suite member", {[]*analysis.Analyzer{noRun, b}}, {"d", "b"}},
		{"valid use of invalid analyzer's prerequisite", {noRun, p}, {"p"}},

		// Invalid variables contribute nothing.
		{"nil analyzer", {(*analysis.Analyzer)(nil), b}, {"d", "b"}},
		{"nil suite element", {[]*analysis.Analyzer{b, nil}, c}, {"d", "c"}},
		{"suite with name clash", {[]*analysis.Analyzer{b, newAnalyzer("b")}, c}, {"d", "c"}},
		{"wrong type", {42, b}, {"d", "b"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			var vars []gotestchecker.Var
			for _, v := range test.vars {
				vars = append(vars, gotestchecker.Var{Symbol: "example.com/p.V", Value: v})
			}
			var got []string
			for _, a := range gotestchecker.Analyzers(vars) {
				got = append(got, a.Name)
			}
			if !slices.Equal(got, test.want) {
				t.Errorf("got %q, want %q", got, test.want)
			}
		})
	}
}

// newAnalyzer returns a new valid analyzer with the specified
// name and prerequisites.
func newAnalyzer(name string, requires ...*analysis.Analyzer) *analysis.Analyzer {
	return &analysis.Analyzer{
		Name:     name,
		Doc:      "doc",
		Requires: requires,
		Run:      func(*analysis.Pass) (any, error) { return nil, nil },
	}
}

// A nonPointerFact is a fact type that is (invalidly) not a pointer.
type nonPointerFact struct{}

func (nonPointerFact) AFact() {}
