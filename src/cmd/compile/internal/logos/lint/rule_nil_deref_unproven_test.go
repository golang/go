// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"internal/testenv"
)

// derefErrors parses and checks a loGo/s source, collecting the type errors
// instead of failing on them, and returns the checker's Info, the file, and
// the errors.
func derefErrors(t *testing.T, src string) (*types2.Info, *syntax.File, []types2.Error) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	file, err := syntax.Parse(syntax.NewFileBase("x.lgo"), strings.NewReader(src), nil, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	info := &types2.Info{
		StoreTypesInSyntax: true,
		Defs:               make(map[*syntax.Name]types2.Object),
		Uses:               make(map[*syntax.Name]types2.Object),
		Selections:         make(map[*syntax.SelectorExpr]*types2.Selection),
		Implicits:          make(map[syntax.Node]types2.Object),
		Scopes:             make(map[syntax.Node]*types2.Scope),
	}
	var errs []types2.Error
	conf := &types2.Config{
		Importer:    &lintImporter{pkgs: make(map[string]*types2.Package)},
		Error:       func(e error) { errs = append(errs, e.(types2.Error)) },
		Logos:       true,
		LogosUnions: true,
	}
	_, checkErr := conf.Check("p", []*syntax.File{file}, info)
	if checkErr != nil && len(errs) == 0 {
		t.Fatalf("conf.Check %v with no collected errors", checkErr)
	}
	return info, file, errs
}

const derefSrc = `package p

type Addr struct{ City string }

type User struct {
	Name string
	Age  int
	Next *User?
	Home Addr
	Self *User
}

func (u User) Greet() string    { return u.Name }
func (u User) Bump()            {}
func (u User) Pair() (int, int) { return 1, 2 }
func (u User) Hold() *User?     { return nil }

type Holder struct{ P *User? }

func field(u *User?) string {
	return u.Name
}

func method(u *User?) string {
	return u.Greet()
}

func statement(u *User?) {
	u.Bump()
}

func operand(u *User?) int {
	return u.Age + 1
}

func optionField(u *User?) *User? {
	return u.Next
}

func optionResult(u *User?) *User? {
	return u.Hold()
}

func methodValue(u *User?) {
	f := u.Greet
	_ = f
}

func pair(u *User?) (int, int) {
	return u.Pair()
}

func structField(u *User?) Addr {
	return u.Home
}

func pointerResult(u *User?) *User {
	return u.Self
}

func nested(h Holder) string {
	return h.P.Name
}

func continued(u *User?) string {
	return u.
		Name
}

func inLoop(us [](*User?)) int {
	n := 0
	for _, u := range us {
		n += u.Age
	}
	return n
}

func inClosure(u *User?) func() string {
	return func() string {
		return u.Name
	}
}

func inLambda(u *User?) {
	var f func(int) string = x => u.Name
	_ = f
}

func twoResults(u *User?) (n int, err error) {
	return u.Age, nil
}
`

// derefWant is one expected finding: the source line it sits on, the source
// it replaces, the receiver text, and the candidates in order.
type derefWant struct {
	line   string
	span   string
	origin string
	fixes  []Fix
}

// guardText returns the text of an early-return guard on u for the given
// results, indented by indent.
func guardText(indent, results string) string {
	return indent + "if u == nil {\n" + indent + "\t" + results + "\n" + indent + "}\n"
}

// TestNilDerefResult covers the span and the autofix candidates of the
// findings the compiler builds from the checker's nil_deref_unproven errors:
// optional chaining with a ?? default, offered when a zero value of the
// result is known, and the early-return guard, offered for a plain name when
// every result of the function has a zero value.
func TestNilDerefResult(t *testing.T) {
	info, file, errs := derefErrors(t, derefSrc)
	var got []Result
	for _, e := range errs {
		r, ok := NilDerefResult(info, file, []byte(derefSrc), e.Pos, e.Msg)
		if !ok {
			t.Fatalf("not a nil_deref_unproven error: %v", e)
		}
		got = append(got, r)
	}
	const chainDefault = "use optional chaining with a ?? default"
	const chain = "use optional chaining"
	want := []derefWant{
		{"\treturn u.Name", "u.Name", "u", []Fix{
			{Description: chainDefault, Replacement: `(u?.Name ?? "")`},
			{Description: "return early when u is nil", Replacement: guardText("\t", `return ""`)},
		}},
		{"\treturn u.Greet()", "u.Greet()", "u", []Fix{
			{Description: chainDefault, Replacement: `(u?.Greet() ?? "")`},
			{Description: "return early when u is nil", Replacement: guardText("\t", `return ""`)},
		}},
		{"\tu.Bump()", "u.Bump()", "u", []Fix{
			{Description: chain, Replacement: "u?.Bump()"},
			{Description: "return early when u is nil", Replacement: guardText("\t", "return")},
		}},
		{"\treturn u.Age + 1", "u.Age", "u", []Fix{
			{Description: chainDefault, Replacement: "(u?.Age ?? 0)"},
			{Description: "return early when u is nil", Replacement: guardText("\t", "return 0")},
		}},
		{"\treturn u.Next", "u.Next", "u", []Fix{
			{Description: chain, Replacement: "u?.Next"},
			{Description: "return early when u is nil", Replacement: guardText("\t", "return nil")},
		}},
		{"\treturn u.Hold()", "u.Hold()", "u", []Fix{
			{Description: chain, Replacement: "u?.Hold()"},
			{Description: "return early when u is nil", Replacement: guardText("\t", "return nil")},
		}},
		{"\tf := u.Greet", "u.Greet", "u", []Fix{
			{Description: chain, Replacement: "u?.Greet"},
			{Description: "return early when u is nil", Replacement: guardText("\t", "return")},
		}},
		{"\treturn u.Pair()", "u.Pair()", "u", []Fix{
			{Description: "return early when u is nil", Replacement: guardText("\t", "return 0, 0")},
		}},
		{"\treturn u.Home", "u.Home", "u", nil},
		{"\treturn u.Self", "u.Self", "u", nil},
		{"\treturn h.P.Name", "h.P.Name", "h.P", []Fix{
			{Description: chainDefault, Replacement: `(h.P?.Name ?? "")`},
		}},
		{"\treturn u.", "u.\n\t\tName", "u", []Fix{
			{Description: chainDefault, Replacement: "(u?.\n\t\tName ?? \"\")"},
			{Description: "return early when u is nil", Replacement: guardText("\t", `return ""`)},
		}},
		{"\t\tn += u.Age", "u.Age", "u", []Fix{
			{Description: chainDefault, Replacement: "(u?.Age ?? 0)"},
			{Description: "return early when u is nil", Replacement: guardText("\t\t", "return 0")},
		}},
		{"\t\treturn u.Name", "u.Name", "u", []Fix{
			{Description: chainDefault, Replacement: `(u?.Name ?? "")`},
			{Description: "return early when u is nil", Replacement: guardText("\t\t", `return ""`)},
		}},
		{"\tvar f func(int) string = x => u.Name", "u.Name", "u", []Fix{
			{Description: chainDefault, Replacement: `(u?.Name ?? "")`},
		}},
		{"\treturn u.Age, nil", "u.Age", "u", []Fix{
			{Description: chainDefault, Replacement: "(u?.Age ?? 0)"},
			{Description: "return early when u is nil", Replacement: guardText("\t", "return 0, nil")},
		}},
	}
	if len(got) != len(want) {
		t.Fatalf("%d findings, want %d", len(got), len(want))
	}
	lines := strings.Split(derefSrc, "\n")
	for i, w := range want {
		r := got[i]
		line := int(r.Pos.Line())
		if !strings.HasPrefix(lines[line-1], w.line) {
			t.Errorf("finding %d on line %d %q, want a line starting %q", i, line, lines[line-1], w.line)
		}
		if text := spanText(derefSrc, r); text != w.span {
			t.Errorf("finding %d span %q, want %q", i, text, w.span)
		}
		if r.Provenance.OriginExpr != w.origin {
			t.Errorf("finding %d origin %q, want %q", i, r.Provenance.OriginExpr, w.origin)
		}
		for j := range w.fixes {
			if w.fixes[j].InsertBeforeLine == 0 && strings.HasPrefix(w.fixes[j].Description, "return early") {
				w.fixes[j].InsertBeforeLine = line
			}
		}
		if !slices.EqualFunc(r.Fixes, w.fixes, func(a, b Fix) bool { return a == b }) {
			t.Errorf("finding %d fixes %+v, want %+v", i, r.Fixes, w.fixes)
		}
		for _, fx := range r.Fixes {
			checkFixClears(t, i, r, fx, len(errs))
		}
	}
}

// checkFixClears applies the candidate fx of the finding r to derefSrc and
// checks the result again: exactly the finding's own error is gone and no
// other error appears, so a wrong span, a wrong precedence, or a zero value
// of the wrong type fails here.
func checkFixClears(t *testing.T, n int, r Result, fx Fix, before int) {
	t.Helper()
	patched := applyFix(t, derefSrc, r, fx)
	_, _, errs := derefErrors(t, patched)
	for _, e := range errs {
		if !strings.HasPrefix(e.Msg, NilDerefTag) {
			t.Errorf("finding %d, candidate %q: new error %v in\n%s", n, fx.Description, e, patched)
		}
	}
	if len(errs) != before-1 {
		t.Errorf("finding %d, candidate %q: %d errors after the fix, want %d", n, fx.Description, len(errs), before-1)
	}
}

// applyFix returns src with the candidate fx of the finding r applied: the
// span replaced, or the lines inserted before InsertBeforeLine.
func applyFix(t *testing.T, src string, r Result, fx Fix) string {
	t.Helper()
	d := derefSite{src: []byte(src)}
	if fx.InsertBeforeLine > 0 {
		at, ok := d.offset(syntax.MakePos(r.Pos.Base(), uint(fx.InsertBeforeLine), 1))
		if !ok {
			t.Fatalf("no line %d", fx.InsertBeforeLine)
		}
		return src[:at] + fx.Replacement + src[at:]
	}
	start, ok1 := d.offset(r.Pos)
	end, ok2 := d.offset(r.End)
	if !ok1 || !ok2 {
		t.Fatalf("no span %v-%v", r.Pos, r.End)
	}
	return src[:start] + fx.Replacement + src[end:]
}

// TestNilDerefResultRejects checks the inputs NilDerefResult leaves to the
// plain error path: another rule's message, and a position that is not the
// member name of a selection.
func TestNilDerefResultRejects(t *testing.T) {
	info, file, errs := derefErrors(t, derefSrc)
	if len(errs) == 0 {
		t.Fatal("no errors")
	}
	e := errs[0]
	if _, ok := NilDerefResult(info, file, []byte(derefSrc), e.Pos, "[correctness::not_an_option] ?. on non-option"); ok {
		t.Error("accepted another rule's message")
	}
	if _, ok := NilDerefResult(info, file, []byte(derefSrc), e.Pos, "no tag"); ok {
		t.Error("accepted an untagged message")
	}
	if _, ok := NilDerefResult(info, file, []byte(derefSrc), syntax.MakePos(e.Pos.Base(), 1, 1), e.Msg); ok {
		t.Error("accepted a position that is not a member name")
	}
}

// TestNilDerefForbid checks the mapping of the types2-enforced rule: a
// loGo/s-only forbid rule with no Check, whose //logos:allow directive is
// reported as correctness::allow_forbidden_rule when the package reaches the
// lint pass.
func TestNilDerefForbid(t *testing.T) {
	r, ok := Lookup("correctness::nil_deref_unproven")
	if !ok || r.Check != nil || r.Go || r.Group != Correctness {
		t.Fatalf("rule %+v, want a registered Correctness rule with no Check and no Go scope", r)
	}
	const src = `package p

type User struct{ Name string }

//logos:allow(correctness::nil_deref_unproven)
func f(u *User) string {
	return u.Name
}
`
	got := summarize(Run(check(t, KindLgo, "x.lgo", src)))
	if want := []string{"5 correctness::allow_forbidden_rule forbid"}; !slices.Equal(got, want) {
		t.Errorf("results %q, want %q", got, want)
	}
}
