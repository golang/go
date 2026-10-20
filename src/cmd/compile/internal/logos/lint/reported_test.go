// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"errors"
	"internal/testenv"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// reportedSrc holds one finding of each rule that types2 reports, and a
// directive above a statement and above a declaration that silences one.
const reportedSrc = `package p

type Cfg struct {
	Port int32 = 8080
}

func bump(x int) int { return 2 * x }

func plain(xs []int) {
	var c Cfg
	_ = c
	_ = xs.map(x => bump(x)).sortBy(x => x)
}

//logos:allow(suspicious::unapplied_struct_default)
func allowedDefault() {
	var c Cfg
	_ = c
}

func allowedSort(xs []int) {
	//logos:allow(suspicious::side_effect_before_sort)
	_ = xs.map(x => bump(x)).sortBy(x => x)
}
`

// reportedPolicy parses a policy that sets the suspicious group.
func reportedPolicy(t *testing.T, level string) *Policy {
	t.Helper()
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"suspicious":"` + level + `"}}`))
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestReportedOf covers the conversion of a types2 diagnostic: only a
// warn-severity loGo/s diagnostic becomes a finding, its tag leaves the
// message, and any other error stays an error.
func TestReportedOf(t *testing.T) {
	pos := syntax.MakePos(syntax.NewFileBase("x.lgo"), 3, 4)
	r, ok := ReportedOf(types2.Error{Pos: pos, Soft: true, Msg: "[suspicious::unapplied_struct_default] zero declaration of T with field defaults"})
	if !ok || r.Rule != "suspicious::unapplied_struct_default" || r.Message != "zero declaration of T with field defaults" || r.Pos != pos {
		t.Errorf("ReportedOf(unapplied default) = %+v, %v", r, ok)
	}
	for name, err := range map[string]error{
		"hard error":     types2.Error{Pos: pos, Msg: "[suspicious::unapplied_struct_default] zero declaration of T with field defaults"},
		"other soft":     types2.Error{Pos: pos, Soft: true, Msg: "[correctness::default_not_constant] default is not constant"},
		"untagged":       types2.Error{Pos: pos, Soft: true, Msg: "zero declaration of T with field defaults"},
		"not types2 err": errors.New("[suspicious::unapplied_struct_default] x"),
	} {
		if r, ok := ReportedOf(err); ok {
			t.Errorf("%s: ReportedOf = %+v, want none", name, r)
		}
	}
}

// TestReportedSeverity checks that a finding of the checker takes the group
// severity of the policy and the directive of its statement or declaration:
// warn and deny are silenced by the directive, allow drops the finding, and
// forbid keeps it, with the directive itself an error.
func TestReportedSeverity(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const (
		dflt = "suspicious::unapplied_struct_default"
		sort = "suspicious::side_effect_before_sort"
	)
	for _, tc := range []struct {
		level string
		want  []string
	}{
		{"warn", []string{"10 " + dflt + " warn", "12 " + sort + " warn"}},
		{"deny", []string{"10 " + dflt + " deny", "12 " + sort + " deny"}},
		{"allow", nil},
		{"forbid", []string{
			"10 " + dflt + " forbid",
			"12 " + sort + " forbid",
			"15 correctness::allow_forbidden_rule forbid",
			"17 " + dflt + " forbid",
			"22 correctness::allow_forbidden_rule forbid",
			"23 " + sort + " forbid",
		}},
	} {
		in := check(t, KindLgo, "x.lgo", reportedSrc)
		in.Policy = reportedPolicy(t, tc.level)
		if got := summarize(Run(in)); !slices.Equal(got, tc.want) {
			t.Errorf("suspicious = %s:\n got %q\nwant %q", tc.level, got, tc.want)
		}
	}
}

// TestReportedDefaultPolicy checks that a nil policy resolves a finding of
// the checker at the default severity of its group, warn.
func TestReportedDefaultPolicy(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	in := check(t, KindLgo, "x.lgo", reportedSrc)
	want := []string{"10 suspicious::unapplied_struct_default warn", "12 suspicious::side_effect_before_sort warn"}
	if got := summarize(Run(in)); !slices.Equal(got, want) {
		t.Errorf("default policy: got %q, want %q", got, want)
	}
}

// TestResolveReported checks Resolve, which handles only the findings of an
// earlier phase and needs neither the package nor its type information.
func TestResolveReported(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, level := range []string{"warn", "deny", "allow", "forbid"} {
		full := check(t, KindLgo, "x.lgo", reportedSrc)
		full.Policy = reportedPolicy(t, level)
		var want []string
		for _, s := range summarize(Run(full)) {
			if strings.Contains(s, " suspicious::") {
				want = append(want, s)
			}
		}
		bare := &Input{
			Files:    full.Files,
			Sources:  full.Sources,
			Kind:     KindLgo,
			Policy:   full.Policy,
			Reported: full.Reported,
		}
		if got := summarize(Resolve(bare)); !slices.Equal(got, want) {
			t.Errorf("Resolve with suspicious = %s:\n got %q\nwant %q", level, got, want)
		}
	}
}

// TestReportedNeverInGoPackage checks that a finding of a rule that does
// not apply to Go packages is dropped for a Go package.
func TestReportedNeverInGoPackage(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	in := check(t, KindGo, "x.go", "package p\n\nfunc f() {}\n")
	pos := in.Files[0].Pos()
	in.Reported = []Reported{{Rule: "suspicious::unapplied_struct_default", Pos: pos, Message: "x"}}
	if got := summarize(Run(in)); len(got) != 0 {
		t.Errorf("Run reported %q for a Go package", got)
	}
	if got := summarize(Resolve(in)); len(got) != 0 {
		t.Errorf("Resolve reported %q for a Go package", got)
	}
}
