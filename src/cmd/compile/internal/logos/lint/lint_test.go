// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/importer"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/goobj"
)

// goRules are the 13 language-neutral rules of SPEC Sec 1.2: the rules
// logos lint applies to Go packages.
var goRules = []string{
	"correctness::arena_pointer_escape",
	"correctness::infinite_recursion",
	"correctness::typed_nil_return",
	"correctness::unclosed_io_closer",
	"correctness::unbounded_channel_op",
	"concurrency::concurrent_multiword_race",
	"concurrency::reentrant_withlock_deadlock",
	"slop::err_chain_reset",
	"slop::discarded_error",
	"slop::todo_placeholder",
	"complexity::cognitive_overflow",
	"perf::large_backing_array_pin",
	"perf::time_after_in_loop",
}

// lgoRules are the 8 rules that never apply to Go packages.
var lgoRules = []string{
	"correctness::nil_deref_unproven",
	"correctness::defer_erases_errdefer_state",
	"concurrency::unsupervised_goroutine",
	"suspicious::unapplied_struct_default",
	"suspicious::side_effect_before_sort",
	"portability::serialized_unsized_int",
	"portability::alias_extension",
	"pedantic::exhaustive_union_switch",
}

// TestRuleScopes asserts that the rules with Go set are exactly the 13
// language-neutral rules, that every registered rule declares a scope
// (appears in exactly one of the two SPEC lists), that each rule's Group
// matches its ID, and that the registry has no duplicate. A SPEC rule that
// is not registered yet fails the test in both full and -short runs, so a
// missing rule can never hide behind the short flag.
func TestRuleScopes(t *testing.T) {
	registered := registeredRules(t)
	var missing []string
	for _, id := range goRules {
		if !checkScope(t, registered, id, true) {
			missing = append(missing, id)
		}
	}
	for _, id := range lgoRules {
		if !checkScope(t, registered, id, false) {
			missing = append(missing, id)
		}
	}
	for id := range registered {
		if !slices.Contains(goRules, id) && !slices.Contains(lgoRules, id) {
			t.Errorf("rule %s declares no scope: add it to goRules or lgoRules", id)
		}
	}
	if len(missing) == 0 {
		return
	}
	report := "SPEC Sec 1.2 rules not registered: " + strings.Join(missing, ", ")
	t.Error(report)
}

// registeredRules indexes the registry by ID, failing the test for a
// duplicate ID or a Group that differs from the ID prefix.
func registeredRules(t *testing.T) map[string]Rule {
	t.Helper()
	registered := make(map[string]Rule)
	for _, r := range Rules() {
		if _, dup := registered[r.ID]; dup {
			t.Errorf("rule %s is registered twice", r.ID)
		}
		registered[r.ID] = r
		if GroupOf(r.ID) != r.Group {
			t.Errorf("rule %s has group %q, want %q", r.ID, r.Group, GroupOf(r.ID))
		}
	}
	return registered
}

// checkScope checks one SPEC rule that is registered: its Go scope must be
// wantGo. It reports whether the rule is registered.
func checkScope(t *testing.T, registered map[string]Rule, id string, wantGo bool) bool {
	t.Helper()
	r, ok := registered[id]
	if ok && r.Go != wantGo {
		t.Errorf("rule %s has Go = %v, want %v", id, r.Go, wantGo)
	}
	return ok
}

// TestTypes2EnforcedRule checks the mapping of concurrency::unsupervised_goroutine,
// which types2 enforces: a loGo/s-only forbid rule with no Check, whose
// //logos:allow directive is an allow_forbidden_rule error and never silences
// the types2 finding.
func TestTypes2EnforcedRule(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	r, ok := Lookup("concurrency::unsupervised_goroutine")
	if !ok || r.Go || r.Check != nil || r.Group != Concurrency {
		t.Fatalf("registry entry = %+v, %v", r, ok)
	}
	if got := DefaultPolicy().Level(r.ID, KindLgo); got != Forbid {
		t.Errorf("default level = %v, want forbid", got)
	}
	const src = `package p

func f() {}

func g() {
	//logos:allow(concurrency::unsupervised_goroutine)
	f()
}
`
	got := summarize(Run(check(t, KindLgo, "x.lgo", src)))
	if want := []string{"6 correctness::allow_forbidden_rule forbid"}; !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
	if got := summarize(Run(check(t, KindGo, "x.go", src))); len(got) != 0 {
		t.Errorf("Go package results = %q, want none", got)
	}
}

// TestRuleScopesSpecCounts pins the 13 and 8 of SPEC Sec 1.2.
func TestRuleScopesSpecCounts(t *testing.T) {
	if len(goRules) != 13 || len(lgoRules) != 8 {
		t.Fatalf("goRules = %d, lgoRules = %d, want 13 and 8", len(goRules), len(lgoRules))
	}
}

// TestPolicyDefaults covers the default severities, the Go cap, and the
// default thresholds.
func TestPolicyDefaults(t *testing.T) {
	d := DefaultPolicy()
	for id, want := range map[string]Severity{
		"correctness::x": Forbid, "concurrency::x": Forbid,
		"slop::x": Deny, "complexity::x": Deny,
		"perf::x": Warn, "style::x": Warn, "suspicious::x": Warn, "portability::x": Warn,
		"pedantic::x": Allow,
	} {
		if got := d.Level(id, KindLgo); got != want {
			t.Errorf("default Level(%s, lgo) = %v, want %v", id, got, want)
		}
	}
	if got := d.Level("correctness::x", KindGo); got != Deny {
		t.Errorf("Go cap: Level(correctness::x, go) = %v, want deny", got)
	}
	if got := d.Thresholds; got != (Thresholds{3, 15, 60, 4, 4096, 100000}) {
		t.Errorf("default thresholds = %+v", got)
	}
}

// TestPolicyParse covers JSON decoding and the Go cap of a configured
// forbid group.
func TestPolicyParse(t *testing.T) {
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"slop":"warn","pedantic":"deny"},"thresholds":{"max_function_lines":80}}`))
	if err != nil {
		t.Fatal(err)
	}
	if p.Level("slop::x", KindLgo) != Warn || p.Level("pedantic::x", KindLgo) != Deny || p.Thresholds.MaxFunctionLines != 80 {
		t.Errorf("parsed policy wrong: %+v", p)
	}
	p, err = ParsePolicy([]byte(`{"version":1,"groups":{"slop":"forbid"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if p.Level("slop::x", KindLgo) != Forbid || p.Level("slop::x", KindGo) != Deny {
		t.Errorf("slop=forbid: lgo %v go %v, want forbid deny", p.Level("slop::x", KindLgo), p.Level("slop::x", KindGo))
	}
}

// TestPolicyRejects covers every invalid policy shape.
func TestPolicyRejects(t *testing.T) {
	for name, text := range map[string]string{
		"lower forbid":      `{"version":1,"groups":{"correctness":"warn"}}`,
		"lower concurrency": `{"version":1,"groups":{"concurrency":"deny"}}`,
		"unknown group":     `{"version":1,"groups":{"speed":"warn"}}`,
		"bad severity":      `{"version":1,"groups":{"slop":"loud"}}`,
		"unknown threshold": `{"version":1,"thresholds":{"max_x":3}}`,
		"zero threshold":    `{"version":1,"thresholds":{"max_function_lines":0}}`,
		"wrong version":     `{"version":2}`,
		"unknown field":     `{"version":1,"extra":1}`,
		"not json":          `slop`,
	} {
		if _, err := ParsePolicy([]byte(text)); err == nil {
			t.Errorf("%s: ParsePolicy(%s) succeeded", name, text)
		}
	}
}

// lintImporter imports standard packages from export data.
type lintImporter struct {
	pkgs map[string]*types2.Package
}

// Import implements types2.Importer.
func (m *lintImporter) Import(path string) (*types2.Package, error) {
	return m.ImportFrom(path, "", 0)
}

// ImportFrom implements types2.ImporterFrom.
func (m *lintImporter) ImportFrom(path, srcDir string, mode types2.ImportMode) (*types2.Package, error) {
	return importer.Import(m.pkgs, path, srcDir, nil)
}

// check parses and type-checks one source file named name and returns the
// Input for Run. Type errors fail the test; a warn-severity diagnostic of
// types2 becomes a Reported finding, as in the compiler.
func check(t testing.TB, kind Kind, name, src string) *Input {
	t.Helper()
	mode := syntax.CheckBranches
	if kind == KindLgo {
		mode |= syntax.LogosMode
	}
	var perrs []error
	file, _ := syntax.Parse(syntax.NewFileBase(name), strings.NewReader(src), func(err error) {
		perrs = append(perrs, err)
	}, nil, mode)
	if len(perrs) > 0 || file == nil {
		t.Fatalf("parse %s: %v", name, perrs)
	}
	info := &types2.Info{
		StoreTypesInSyntax: true,
		Defs:               make(map[*syntax.Name]types2.Object),
		Uses:               make(map[*syntax.Name]types2.Object),
		Selections:         make(map[*syntax.SelectorExpr]*types2.Selection),
		Implicits:          make(map[syntax.Node]types2.Object),
		Scopes:             make(map[syntax.Node]*types2.Scope),
		Instances:          make(map[*syntax.Name]types2.Instance),
	}
	var errs []error
	var reported []Reported
	conf := &types2.Config{
		Importer: &lintImporter{pkgs: make(map[string]*types2.Package)},
		Error: func(err error) {
			if r, ok := ReportedOf(err); ok {
				reported = append(reported, r)
				return
			}
			errs = append(errs, err)
		},
		Logos:       kind == KindLgo,
		LogosUnions: kind == KindLgo,
	}
	pkg, _ := conf.Check("p", []*syntax.File{file}, info)
	if len(errs) > 0 {
		t.Fatalf("check %s: %v", name, errs)
	}
	return &Input{
		Pkg:      pkg,
		Info:     info,
		Files:    []*syntax.File{file},
		Sources:  map[*syntax.File][]byte{file: []byte(src)},
		Kind:     kind,
		Reported: reported,
	}
}

// summarize renders results as "line rule severity" strings.
func summarize(rs []Result) []string {
	var out []string
	for _, r := range rs {
		out = append(out, fmt.Sprintf("%d %s %s", r.Pos.Line(), r.Rule, r.Severity))
	}
	return out
}

// wantMarked returns "line rule severity" for each source line carrying a
// "// want" marker.
func wantMarked(src, rule string, sev Severity) []string {
	var out []string
	for i, line := range strings.Split(src, "\n") {
		if strings.Contains(line, "// want") {
			out = append(out, fmt.Sprintf("%d %s %s", i+1, rule, sev))
		}
	}
	return out
}

// testSource is the name and text of one source file of a test.
type testSource struct {
	name, text string
}

// finding is the rule and severity every marked line must report.
type finding struct {
	rule string
	sev  Severity
}

// expect runs the registered rules over src and compares the results with
// the marked lines.
func expect(t *testing.T, kind Kind, policy *Policy, s testSource, want finding) {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	name, src := s.name, s.text
	in := check(t, kind, name, src)
	in.Policy = policy
	got := summarize(Run(in))
	wantLines := wantMarked(src, want.rule, want.sev)
	if !slices.Equal(got, wantLines) {
		t.Errorf("%s results:\n got %q\nwant %q", name, got, wantLines)
	}
}

const discardedSrc = `package p

import (
	"bytes"
	"fmt"
	"os"
	"strings"
)

func f() error        { return nil }
func g() (int, error) { return 0, nil }

func a() {
	f() // want
	g() // want
	_ = f() // want
	_, _ = g() // want
	n, _ := g() // want
	_ = n
	fmt.Println("x")
	fmt.Fprintf(os.Stderr, "x")
	fmt.Fprintln(os.Stdout, "x")
	var b bytes.Buffer
	b.WriteString("x")
	b.WriteByte('x')
	var sb strings.Builder
	sb.WriteString("x")
	fmt.Fprintf(&b, "x") // want
	fmt.Fprintf(&sb, "x") // want
	err := f()
	_ = err
	defer f()
	h := func() error { return f() }
	h() // want
	m, err := g()
	_, _ = m, err
	_ = (f()) // want
}

func b() error {
	f() // want
	return nil
}
`

// TestDiscardedError covers the detection and the benign exemptions of
// slop::discarded_error in both kinds of package.
func TestDiscardedError(t *testing.T) {
	t.Run("go", func(t *testing.T) {
		expect(t, KindGo, nil, testSource{"x.go", discardedSrc}, finding{"slop::discarded_error", Deny})
	})
	t.Run("lgo", func(t *testing.T) {
		expect(t, KindLgo, nil, testSource{"x.lgo", discardedSrc}, finding{"slop::discarded_error", Deny})
	})
	t.Run("warn", func(t *testing.T) {
		p, err := ParsePolicy([]byte(`{"version":1,"groups":{"slop":"warn"}}`))
		if err != nil {
			t.Fatal(err)
		}
		expect(t, KindGo, p, testSource{"x.go", discardedSrc}, finding{"slop::discarded_error", Warn})
	})
	t.Run("allow", func(t *testing.T) {
		testenv.MustHaveGoBuild(t)
		p, err := ParsePolicy([]byte(`{"version":1,"groups":{"slop":"allow"}}`))
		if err != nil {
			t.Fatal(err)
		}
		in := check(t, KindGo, "x.go", discardedSrc)
		in.Policy = p
		if got := summarize(Run(in)); len(got) != 0 {
			t.Errorf("slop = allow reported %q", got)
		}
	})
}

const spanSrc = `package p

func f(...any) error { return nil }
func g(int) int      { return 0 }

func a() {
	f(g(1))
	f(")")
	f(
		1,
		2,
	)
	f()
	f(/* ) */ 1) // )
	_ = f(g(2))
}
`

// spanText returns the source text a result's span covers.
func spanText(src string, r Result) string {
	lines := strings.SplitAfter(src, "\n")
	offset := func(line, col uint) int {
		n := 0
		for _, l := range lines[:line-1] {
			n += len(l)
		}
		return n + int(col) - 1
	}
	return src[offset(r.Pos.Line(), r.Pos.Col()):offset(r.End.Line(), r.End.Col())]
}

// TestDiscardedErrorSpan checks that the span of a finding covers its whole
// call, through the closing parenthesis, across nested calls, a string
// holding a parenthesis, a call over several lines, a call with no
// argument, and comments: tools splice an autofix by this byte range.
func TestDiscardedErrorSpan(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	want := []string{"f(g(1))", `f(")")`, "f(\n\t\t1,\n\t\t2,\n\t)", "f()", "f(/* ) */ 1)", "f(g(2))"}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		var got []string
		for _, r := range Run(check(t, kind, name, spanSrc)) {
			got = append(got, spanText(spanSrc, r))
		}
		if !slices.Equal(got, want) {
			t.Errorf("%s spans = %q, want %q", name, got, want)
		}
	}
}

// TestDiscardedErrorSpanInterpolation checks the span of a call whose
// argument is a loGo/s interpolated string holding a parenthesis.
func TestDiscardedErrorSpanInterpolation(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

func h(string) error { return nil }

func a() {
	x := "s"
	h($"a){x}(")
}
`
	rs := Run(check(t, KindLgo, "x.lgo", src))
	if len(rs) != 1 {
		t.Fatalf("results = %q", summarize(rs))
	}
	if got, want := spanText(src, rs[0]), `h($"a){x}(")`; got != want {
		t.Errorf("span = %q, want %q", got, want)
	}
}

const fixSrc = `package p

import "os"

func f() error {
	os.Remove("x")
	return nil
}

func g() {
	os.Remove("x")
}
`

// TestAlwaysNilErrorFact checks the exemption of a callee whose package facts
// carry AlwaysNilError. No producer of the fact exists yet, so the test
// injects one for os.Chdir.
func TestAlwaysNilErrorFact(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	exportData := "12345678"
	var fp goobj.FingerprintType
	copy(fp[:], exportData)
	member := facts.Encode(fp, &facts.Facts{Fingerprint: fp, Fns: []facts.FnFact{{Object: "Chdir", AlwaysNilError: true}}})
	if _, err := facts.Import("os", exportData, member); err != nil {
		t.Fatal(err)
	}
	const src = `package p

import "os"

func a() {
	os.Chdir("x")
	os.Setenv("k", "v") // want
}
`
	expect(t, KindGo, nil, testSource{"x.go", src}, finding{"slop::discarded_error", Deny})
}

// TestDiscardedErrorFix checks the JSON provenance and the ? candidate of a
// loGo/s file: offered only in a function that returns error.
func TestDiscardedErrorFix(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	lgo := Run(check(t, KindLgo, "x.lgo", fixSrc))
	if len(lgo) != 2 {
		t.Fatalf("lgo results = %q", summarize(lgo))
	}
	if len(lgo[0].Fixes) != 1 || lgo[0].Fixes[0].Replacement != `os.Remove("x")?` {
		t.Errorf("fix of f = %+v", lgo[0].Fixes)
	}
	if len(lgo[1].Fixes) != 0 {
		t.Errorf("g returns no error, fix = %+v", lgo[1].Fixes)
	}
	if got := lgo[0].Provenance; got.InferredType != "error" || got.OriginExpr != `os.Remove("x")` || got.OriginLine != 6 {
		t.Errorf("provenance = %+v", got)
	}
}

// TestDiscardedErrorNoFixInGo checks that a Go package offers no candidate
// that names loGo/s syntax.
func TestDiscardedErrorNoFixInGo(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, r := range Run(check(t, KindGo, "x.go", fixSrc)) {
		if len(r.Fixes) != 0 {
			t.Errorf("Go package offers fix %+v naming loGo/s syntax", r.Fixes)
		}
	}
}

const recursionSrc = `package p

import (
	"log"
	"os"
)

func r1(n int) int { return r1(n) } // want
func r2()           { r2() } // want
func r3(n int) int {
	if n > 0 {
		return n
	}
	return r3(n)
}
func r4(n int) int { n--; return r4(n) }
func r5(n int) int { return r5(n - 1) }
func r6(a, b int) int { return r6(b, a) }
func r7(n int) int { // want
	if n > 0 {
		return r7(n)
	} else {
		return r7(n)
	}
}
func r8(n int) int { println(n); return r8(n) } // want

type T struct{}

func (t T) m1(n int)  { t.m1(n) } // want
func (t T) m2(n int)  { T{}.m2(n) }
func r9(xs ...int) int { return r9(xs...) } // want
func r10(n int) int    { return 1 + r10(n) } // want
func r11(n int) bool   { return n > 0 && r11(n) }
func r12(n int) int {
	for i := 0; i < 3; i++ {
		return 1
	}
	return r12(n)
}
func r13(_ int) int { return r13(0) }
func r14(p *int) int { *p = 1; return r14(p) }
func r15(n int) {
	f := func() { r15(n) }
	_ = f
}
func work() {}
func r16(n int) int {
	for {
		work()
	}
	return r16(n)
}
func r17(n int) int {
	if n > 0 {
		panic("stop")
	}
	return r17(n)
}
func r18(n int) int { // want
	for i := 0; i < n; i++ {
		work()
	}
	return r18(n)
}
func r19(n int) int {
	{
		n := 5
		return r19(n)
	}
}
func r20(n int) int {
	if n < 0 {
		os.Exit(1)
	}
	return r20(n)
}
func r21(n int) int {
	if n < 0 {
		log.Fatal("x")
	}
	return r21(n)
}
func must(ok bool) {}
func r22(n int) int { // want
	must(n >= 0)
	return r22(n)
}
func r23(n int) int {
	n++
	return r23(n)
}
func r24(n int) int { // want
	if n > 0 {
		r24(n)
		return n
	}
	return r24(n)
}
func r25(n int) int { // want
	if n > 0 {
		r25(n)
	}
	return r25(n)
}
func r26(n int) int { // want
	switch n {
	case 0:
		return r26(n)
	default:
		return r26(n)
	}
}
func r27(n int) int {
	switch n {
	case 0:
		return 0
	default:
		return r27(n)
	}
}
func r28(n int) int {
	switch n {
	case 0:
		return r28(n)
	}
	return 0
}
func r29(n int) int { // want
	switch n {
	case 0:
		return r29(n)
	case 1:
		return r29(n)
	}
	return r29(n)
}
func r30(n int) int { // want
	if n > 0 {
		if n > 1 {
			return r30(n)
		}
	}
	return r30(n)
}
func r31(n int) int {
	if n > 0 {
		if n > 1 {
			return n
		}
	}
	return r31(n)
}
`

// TestInfiniteRecursion covers the always-recurse analysis in both kinds
// of package. Each marker sits on the line of the reported function name.
func TestInfiniteRecursion(t *testing.T) {
	for _, kind := range []Kind{KindGo, KindLgo} {
		name, sev := "x.go", Deny
		if kind == KindLgo {
			name, sev = "x.lgo", Forbid
		}
		expect(t, kind, nil, testSource{name, recursionSrc}, finding{"correctness::infinite_recursion", sev})
	}
}

const recursionLgoSrc = `package p

func g() (int, error) { return 0, nil }

func r1(n int) (int, error) {
	v := g()?
	_ = v
	return r1(n)
}

func r2(n int) (int, error) { // want
	return r2(n)
}
`

// TestInfiniteRecursionTry checks that a loGo/s ? clears the function: it
// may return before the self-call, and the rule cannot be silenced.
func TestInfiniteRecursionTry(t *testing.T) {
	expect(t, KindLgo, nil, testSource{"x.lgo", recursionLgoSrc}, finding{"correctness::infinite_recursion", Forbid})
}

const recursionConstSwitchSrc = `package p

func csSel(n int) int { // want
	const switch int {
	case ~string:
		return n
	case int:
		return csSel(n)
	}
	return n
}

func csDead(n int) int {
	const switch int {
	case ~string:
		return csDead(n)
	case int:
		return n
	}
	return n
}

func csTilde(n int) int { // want
	const switch int {
	case ~int:
		return csTilde(n)
	}
	return n
}

func csIface(n int) int { // want
	const switch int {
	case error:
		return n
	case any:
		return csIface(n)
	}
	return n
}

func csFirst(n int) int { // want
	const switch int {
	case int:
		return csFirst(n)
	case any:
		return n
	}
	return n
}

func csNoMatch(n int) int { // want
	const switch int {
	case ~string:
		return n
	}
	return csNoMatch(n)
}
`

// TestInfiniteRecursionConstSwitch covers const-switch selection: only the
// compile-time selected branch runs, so recursion in the selected branch
// reports while recursion solely in a dead branch does not. Each marked
// line is the reported function name.
func TestInfiniteRecursionConstSwitch(t *testing.T) {
	expect(t, KindLgo, nil, testSource{"x.lgo", recursionConstSwitchSrc}, finding{"correctness::infinite_recursion", Forbid})
}

const recursionGenericSrc = `package p

func genSimple[T any](n T) T { return genSimple[T](n) } // want

func genParen[T any](n T) T { return (genParen[T])(n) } // want

func genMulti[T any, U any](a T, b U) (T, U) { return genMulti[T, U](a, b) } // want

func genSwitchFall[T any](n int, v T) T { // want
	switch n {
	case 0:
		return genSwitchFall[T](n, v)
	case 1:
		return genSwitchFall[T](n, v)
	}
	return genSwitchFall[T](n, v)
}

func genMonoChanged[T any](n int) int { return genMonoChanged[string](n) }

func genChangedVals[T any](a T, b T) T { return genChangedVals[T](b, a) }

func genSwapType[T any, U any](a T, b U) int { return genSwapType[U, T](b, a) }

func genShadow[T any](n T) T {
	{
		n := n
		return genShadow[T](n)
	}
}

type R struct{}

func (r R) MSimple[T any](n T) T { return r.MSimple[T](n) } // want

func (r R) MOtherRecv[T any](n T) T {
	other := R{}
	return other.MOtherRecv[T](n)
}

func (r R) MChangedArg[T any](n int) int { return r.MChangedArg[string](n) }

type GR[T any] struct{ v T }

func (r GR[T]) GM[U any](a T, b U) (T, U) { return r.GM[U](a, b) } // want
`

// TestInfiniteRecursionGenericSelfCall covers declaration-unconditional
// recognition of explicit generic self calls with identical type and value
// parameters: a simple f[T](n) entry recursion, a parenthesized call, a
// multi-type-parameter call with the same positional vector, a
// switch fallthrough with a trailing self call, a method r.M[T](n) on the
// same receiver, and a generic-receiver r.GM[U](a, b). Changed
// type-argument vectors, changed value arguments, swapped multi-type
// vectors, a call on another receiver, and shadowed names must not report.
// Each marked line is the reported function name. RED before the
// production fix: callsFn does not unwrap IndexExpr, so every generic self
// call is missed.
func TestInfiniteRecursionGenericSelfCall(t *testing.T) {
	for _, kind := range []Kind{KindGo, KindLgo} {
		name, sev := "x.go", Deny
		if kind == KindLgo {
			name, sev = "x.lgo", Forbid
		}
		expect(t, kind, nil, testSource{name, recursionGenericSrc}, finding{"correctness::infinite_recursion", sev})
	}
}

const recursionGenericConstSwitchSrc = `package p

func genConstSel[T any](n T) T { // want
	const switch int {
	case ~string:
		return n
	case int:
		return genConstSel[T](n)
	}
	return genConstSel[T](n)
}

func genConstDead[T any](n T) T {
	const switch int {
	case ~string:
		return genConstDead[T](n)
	case int:
		return n
	}
	return n
}

func genTermSwitch[T any](n int) int {
	const switch T {
	case int:
		return genTermSwitch[string](n)
	case string:
		return n
	}
	return n
}
`

// TestInfiniteRecursionGenericConstSwitch covers the generic const-switch
// witness: a concrete tag selects the matching case at compile time, so the
// selected case plus the trailing fallthrough path must both call with the
// identical type and value vector to report. Recursion solely in a dead
// branch, and a terminating f[int]->f[string] dispatch on a type-parameter
// tag, must not report. Each marked line is the reported function name.
// RED before the production fix: the generic selected-case call is missed.
func TestInfiniteRecursionGenericConstSwitch(t *testing.T) {
	expect(t, KindLgo, nil, testSource{"x.lgo", recursionGenericConstSwitchSrc}, finding{"correctness::infinite_recursion", Forbid})
}

// TestDirective covers the //logos:allow scope rules.
func TestDirective(t *testing.T) {
	const src = `package p

func f() error { return nil }

func a() {
	//logos:allow(slop::discarded_error)
	f()
	f() // want
	//logos:allow(slop::discarded_error)

	f() // want
	//logos:allow(slop::discarded_error)
	if true {
		f()
	}
	// doc text
	//logos:allow(slop::discarded_error)
	// more text
	f()
	//logos:allow(perf::time_after_in_loop)
	f() // want
}

//logos:allow(slop::discarded_error)
func b() {
	f()
	f()
}

func c() {
	f() //logos:allow(slop::discarded_error) // want
}

func d() {
	_ = ` + "`" + `
//logos:allow(slop::discarded_error)
` + "`" + `
	f() // want
}
`
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		expect(t, kind, nil, testSource{name, src}, finding{"slop::discarded_error", Deny})
	}
}

// TestDirectiveWarn checks that a directive silences a warn rule too.
func TestDirectiveWarn(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

func f() error { return nil }

func a() {
	//logos:allow(slop::discarded_error)
	f()
	f()
}
`
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"slop":"warn"}}`))
	if err != nil {
		t.Fatal(err)
	}
	in := check(t, KindLgo, "x.lgo", src)
	in.Policy = p
	got := summarize(Run(in))
	if want := []string{"8 slop::discarded_error warn"}; !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
}

const forbidSrc = `package p

//logos:allow(correctness::infinite_recursion)
func loop(n int) int { return loop(n) }
`

// TestForbidDirective covers the forbid rules: ignored directive plus
// correctness::allow_forbidden_rule in loGo/s, silenced at deny in Go.
func TestForbidDirective(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	got := summarize(Run(check(t, KindLgo, "x.lgo", forbidSrc)))
	want := []string{
		"3 correctness::allow_forbidden_rule forbid",
		"4 correctness::infinite_recursion forbid",
	}
	if !slices.Equal(got, want) {
		t.Errorf("lgo results = %q, want %q", got, want)
	}
	if got := summarize(Run(check(t, KindGo, "x.go", forbidSrc))); len(got) != 0 {
		t.Errorf("Go package: directive must silence the capped rule, got %q", got)
	}
}

// TestForbidDirectiveConfig covers a group raised to forbid by logos.toml.
func TestForbidDirectiveConfig(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

func f() error { return nil }

func a() {
	//logos:allow(slop::discarded_error)
	f()
}
`
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"slop":"forbid"}}`))
	if err != nil {
		t.Fatal(err)
	}
	in := check(t, KindLgo, "x.lgo", src)
	in.Policy = p
	got := summarize(Run(in))
	want := []string{
		"6 correctness::allow_forbidden_rule forbid",
		"7 slop::discarded_error forbid",
	}
	if !slices.Equal(got, want) {
		t.Errorf("lgo results = %q, want %q", got, want)
	}
	in = check(t, KindGo, "x.go", src)
	in.Policy = p
	if got := summarize(Run(in)); len(got) != 0 {
		t.Errorf("Go package: cap to deny plus directive must silence, got %q", got)
	}
}

const unionSrc = `package p

type Shape interface {
	Circle | Square
}

type Circle struct{ r int }

type Square struct{ s int }

type Open interface {
	~int | ~string
}

func area(sh Shape) int {
	switch v := sh.(type) {
	case Circle:
		return v.r
	default: // want
		return 0
	}
}

func exact(sh Shape) int {
	switch v := sh.(type) {
	case Circle:
		return v.r
	case Square:
		return v.s
	case nil:
		return 0
	}
	return -1
}

func other(v any) int {
	switch v.(type) {
	case int:
		return 1
	default:
		return 0
	}
}

func value(n int) int {
	switch n {
	case 1:
		return 1
	default:
		return 0
	}
}

func areaExpr(sh Shape) int {
	return switch v := sh.(type) {
	case Circle => v.r
	default => 0 // want
	}
}

func exactExpr(sh Shape) int {
	return switch v := sh.(type) {
	case Circle => v.r
	case Square => v.s
	case nil => 0
	}
}
`

// TestExhaustiveUnionSwitch covers the pedantic rule: silent by default,
// reported as a warning when the group is enabled, never run on Go.
func TestExhaustiveUnionSwitch(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	in := check(t, KindLgo, "x.lgo", unionSrc)
	if got := summarize(Run(in)); len(got) != 0 {
		t.Errorf("default policy reported %q", got)
	}
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"pedantic":"warn"}}`))
	if err != nil {
		t.Fatal(err)
	}
	expect(t, KindLgo, p, testSource{"x.lgo", unionSrc}, finding{"pedantic::exhaustive_union_switch", Warn})
}

// withRule registers r for the duration of the test.
func withRule(t *testing.T, r Rule) {
	t.Helper()
	saved := registry
	registry = append(slices.Clone(registry), r)
	t.Cleanup(func() { registry = saved })
}

// TestGoScope checks that Run calls a rule without Go set on a loGo/s
// package only, and a rule with Go set on both.
func TestGoScope(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	reportPackage := func(pass *Pass) {
		pass.Reportf(pass.Files[0].PkgName, "package")
	}
	withRule(t, Rule{ID: "slop::test_lgo_only", Group: Slop, Check: reportPackage})
	withRule(t, Rule{ID: "slop::test_both", Group: Slop, Go: true, Check: reportPackage})
	const src = "package p\n"
	for kind, want := range map[Kind][]string{
		KindGo:  {"1 slop::test_both deny"},
		KindLgo: {"1 slop::test_both deny", "1 slop::test_lgo_only deny"},
	} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		if got := summarize(Run(check(t, kind, name, src))); !slices.Equal(got, want) {
			t.Errorf("%s results = %q, want %q", name, got, want)
		}
	}
}

// TestRunOrder checks that results come back ordered by position.
func TestRunOrder(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	in := check(t, KindLgo, "x.lgo", discardedSrc)
	rs := Run(in)
	for i := 1; i < len(rs); i++ {
		if rs[i-1].Pos.Line() > rs[i].Pos.Line() {
			t.Fatalf("results out of order: %q", summarize(rs))
		}
	}
	var buf bytes.Buffer
	for _, r := range rs {
		fmt.Fprintln(&buf, r.Message)
	}
	if buf.Len() == 0 {
		t.Fatal("no messages")
	}
}
