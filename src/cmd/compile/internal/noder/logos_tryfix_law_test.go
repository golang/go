// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"encoding/json"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const lawPrelude = `package main

import (
	"context"
	"fmt"
)

var _ = context.Background

type myErr string

func (e myErr) Error() string { return string(e) }

var (
	trace  []string
	failAt string
)

func mark(name string) {
	trace = append(trace, name)
}

func step(name string) error {
	mark(name)
	if failAt == name {
		return myErr(name + " failed")
	}
	return nil
}

func a() (int, error)           { return 1, step("a") }
func b() (int, error)           { return 2, step("b") }
func f(x int) (int, error)      { return x * 3, step("f") }
func two(x, y int) (int, error) { return x + y, step("two") }
func c() int                    { mark("c"); return 10 }
func add(x, y int) int          { mark("add"); return x + y }
func mk() []int                 { mark("mk"); return []int{1, 2, 3} }

func h(x int, g func() (int, error)) int {
	mark("h")
	n, err := g()
	if err != nil {
		mark("g-err")
		return x
	}
	return x + n
}

func h2(g func() (int, error), x int) int {
	mark("h2")
	n, err := g()
	if err != nil {
		mark("g-err")
		return x
	}
	return x + n
}

type S struct{ n int }

func (s S) add(x int) S { mark("S.add"); return S{s.n + x} }
func (s S) done() int   { mark("S.done"); return s.n }

func report(name string, run func() (string, error)) {
	trace = nil
	s, err := run()
	if err != nil {
		fmt.Println(name, "fail", failAt, "|", "|", err, "|", trace)
		return
	}
	fmt.Println(name, "fail", failAt, "|", s, "|", err, "|", trace)
}
`

type lawCase struct {
	name   string
	broken string
	fixed  string
	offer  bool
}

type lawProgram struct {
	text   string
	starts []int
	ends   []int
}

func assembleLaw(names, bodies []string) lawProgram {
	var b strings.Builder
	b.WriteString(lawPrelude)
	var p lawProgram
	for _, body := range bodies {
		b.WriteString("\n")
		p.starts = append(p.starts, b.Len())
		b.WriteString(body)
		p.ends = append(p.ends, b.Len())
	}
	b.WriteString("\nfunc main() {\n\tfor _, fa := range []string{\"\", \"a\", \"b\", \"f\", \"two\"} {\n\t\tfailAt = fa\n")
	for i, name := range names {
		fmt.Fprintf(&b, "\t\treport(%q, run%d)\n", name, i)
	}
	b.WriteString("\t}\n}\n")
	p.text = b.String()
	return p
}

func (p lawProgram) caseAt(off int) int {
	for i := range p.starts {
		if p.starts[i] <= off && off < p.ends[i] {
			return i
		}
	}
	return -1
}

func (p lawProgram) caseAtLine(line int) int {
	for i := range p.starts {
		first := strings.Count(p.text[:p.starts[i]], "\n") + 1
		last := strings.Count(p.text[:p.ends[i]], "\n")
		if first <= line && line <= last {
			return i
		}
	}
	return -1
}

func lawModule(t *testing.T, src string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.lgo"), []byte(src), 0o666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module testprog\n\ngo 1.27\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	return dir
}

func lintLaw(t *testing.T, src string) []diagRecord {
	t.Helper()
	cmd := testenv.Command(t, logosToolPath(t), "lint", "--json-diagnostics", "./...")
	cmd.Dir = lawModule(t, src)
	out, _ := cmd.CombinedOutput()
	var diags []diagRecord
	for line := range strings.SplitSeq(string(out), "\n") {
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var rec diagRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			t.Fatalf("bad JSON diagnostic: %v\nline: %s", err, line)
		}
		if rec.RuleID == misplacedTryRuleID {
			diags = append(diags, rec)
		}
	}
	return diags
}

func runLaw(t *testing.T, src string) (string, error) {
	t.Helper()
	cmd := testenv.Command(t, logosToolPath(t), "run", ".")
	cmd.Dir = lawModule(t, src)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func lawOutputs(out string) map[string]string {
	got := map[string]string{}
	for line := range strings.SplitSeq(out, "\n") {
		if fields := strings.Fields(line); len(fields) > 0 {
			got[fields[0]] += line + "\n"
		}
	}
	return got
}

var lawErrLine = regexp.MustCompile(`main\.lgo:(\d+):`)

var lawCases = []lawCase{
	{
		name:  "closure_try",
		offer: true,
		broken: `func run0() (string, error) {
	v := h(a()?, func() (int, error) {
		n := b()?
		return n, nil
	})
	return fmt.Sprint(v), nil
}
`,
		fixed: `func run0() (string, error) {
	x := a()?
	v := h(x, func() (int, error) {
		n := b()?
		return n, nil
	})
	return fmt.Sprint(v), nil
}
`,
	},
	{
		name:  "lambda_try",
		offer: true,
		broken: `func run1() (string, error) {
	xs := []int{1, 2}
	n := a()? + xs.map(x => f(x)?).sum()
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run1() (string, error) {
	xs := []int{1, 2}
	k := a()?
	n := k + xs.map(x => f(x)?).sum()
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name: "lambda_before",
		broken: `func run2() (string, error) {
	xs := []int{1, 2}
	n := xs.map(x => f(x)?).sum() + a()?
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run2() (string, error) {
	xs := []int{1, 2}
	s := xs.map(x => f(x)?).sum()
	k := a()?
	n := s + k
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name:  "closure_own_misplaced",
		offer: true,
		broken: `func run3() (string, error) {
	v := h(a()?, func() (int, error) {
		return b()? + 1, nil
	})
	return fmt.Sprint(v), nil
}
`,
		fixed: `func run3() (string, error) {
	x := a()?
	v := h(x, func() (int, error) {
		y := b()?
		return y + 1, nil
	})
	return fmt.Sprint(v), nil
}
`,
	},
	{
		name:  "closure_called",
		offer: true,
		broken: `func run4() (string, error) {
	n := func() (int, error) { return b() }()? + a()?
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run4() (string, error) {
	x := func() (int, error) { return b() }()?
	y := a()?
	n := x + y
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name:  "closure_two_inside",
		offer: true,
		broken: `func run5() (string, error) {
	v := h(a()?, func() (int, error) {
		m := b()?
		k := f(m)?
		return m + k, nil
	})
	return fmt.Sprint(v), nil
}
`,
		fixed: `func run5() (string, error) {
	x := a()?
	v := h(x, func() (int, error) {
		m := b()?
		k := f(m)?
		return m + k, nil
	})
	return fmt.Sprint(v), nil
}
`,
	},
	{
		name: "closure_before",
		broken: `func run6() (string, error) {
	v := h2(func() (int, error) {
		n := b()?
		return n, nil
	}, a()?)
	return fmt.Sprint(v), nil
}
`,
		fixed: `func run6() (string, error) {
	x := a()?
	v := h2(func() (int, error) {
		n := b()?
		return n, nil
	}, x)
	return fmt.Sprint(v), nil
}
`,
	},
	{
		name:  "multi_args",
		offer: true,
		broken: `func run7() (string, error) {
	x := add(a()?, b()?)
	return fmt.Sprint(x), nil
}
`,
		fixed: `func run7() (string, error) {
	p := a()?
	q := b()?
	x := add(p, q)
	return fmt.Sprint(x), nil
}
`,
	},
	{
		name:  "chained",
		offer: true,
		broken: `func run8() (string, error) {
	z := two(a()?, 1)? + b()?
	return fmt.Sprint(z), nil
}
`,
		fixed: `func run8() (string, error) {
	p := a()?
	q := two(p, 1)?
	r := b()?
	z := q + r
	return fmt.Sprint(z), nil
}
`,
	},
	{
		name: "order_call_first",
		broken: `func run9() (string, error) {
	n := c() + a()?
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run9() (string, error) {
	x := c()
	y := a()?
	n := x + y
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name: "order_method",
		broken: `func run10() (string, error) {
	s := S{}
	v := s.add(a()?).add(b()?).done()
	return fmt.Sprint(v), nil
}
`,
		fixed: `func run10() (string, error) {
	s := S{}
	x := a()?
	s1 := s.add(x)
	y := b()?
	v := s1.add(y).done()
	return fmt.Sprint(v), nil
}
`,
	},
	{
		name: "if_init_scope",
		broken: `func run11() (string, error) {
	out := "neg"
	if x := 5; f(x)? > 0 {
		out = "pos"
	}
	return out, nil
}
`,
		fixed: `func run11() (string, error) {
	out := "neg"
	{
		x := 5
		y := f(x)?
		if y > 0 {
			out = "pos"
		}
	}
	return out, nil
}
`,
	},
	{
		name:  "if_init_try",
		offer: true,
		broken: `func run12() (string, error) {
	out := "le"
	if v := a()?; b()? > v {
		out = "gt"
	}
	return out, nil
}
`,
		fixed: `func run12() (string, error) {
	out := "le"
	{
		v := a()?
		y := b()?
		if y > v {
			out = "gt"
		}
	}
	return out, nil
}
`,
	},
	{
		name: "if_init_call",
		broken: `func run13() (string, error) {
	out := "neg"
	if x := c(); f(x)? > 0 {
		out = "pos"
	}
	return out, nil
}
`,
		fixed: `func run13() (string, error) {
	out := "neg"
	{
		x := c()
		y := f(x)?
		if y > 0 {
			out = "pos"
		}
	}
	return out, nil
}
`,
	},
	{
		name: "if_init_else_uses",
		broken: `func run14() (string, error) {
	out := ""
	if x := 5; f(x)? > 100 {
		out = "big"
	} else {
		out = fmt.Sprint("small", x)
	}
	return out, nil
}
`,
		fixed: `func run14() (string, error) {
	out := ""
	{
		x := 5
		y := f(x)?
		if y > 100 {
			out = "big"
		} else {
			out = fmt.Sprint("small", x)
		}
	}
	return out, nil
}
`,
	},
	{
		name: "else_if_init",
		broken: `func run15() (string, error) {
	out := ""
	if failAt == "zz" {
		out = "zz"
	} else if x := 5; f(x)? > 0 {
		out = fmt.Sprint("pos", x)
	}
	return out, nil
}
`,
		fixed: `func run15() (string, error) {
	out := ""
	if failAt == "zz" {
		out = "zz"
	} else {
		x := 5
		y := f(x)?
		if y > 0 {
			out = fmt.Sprint("pos", x)
		}
	}
	return out, nil
}
`,
	},
	{
		name:  "if_cond",
		offer: true,
		broken: `func run16() (string, error) {
	out := "neg"
	if f(5)? > 0 {
		out = "pos"
	}
	return out, nil
}
`,
		fixed: `func run16() (string, error) {
	out := "neg"
	y := f(5)?
	if y > 0 {
		out = "pos"
	}
	return out, nil
}
`,
	},
	{
		name:  "if_init_unrelated",
		offer: true,
		broken: `func run17() (string, error) {
	out := "le"
	if y := 1; f(5)? > y {
		out = fmt.Sprint("gt", y)
	}
	return out, nil
}
`,
		fixed: `func run17() (string, error) {
	out := "le"
	z := f(5)?
	if y := 1; z > y {
		out = fmt.Sprint("gt", y)
	}
	return out, nil
}
`,
	},
	{
		name: "if_init_assign",
		broken: `func run18() (string, error) {
	out := "neg"
	x := 2
	if x = 7; f(x)? > 20 {
		out = "pos"
	}
	return out + fmt.Sprint(x), nil
}
`,
		fixed: `func run18() (string, error) {
	out := "neg"
	x := 2
	x = 7
	y := f(x)?
	if y > 20 {
		out = "pos"
	}
	return out + fmt.Sprint(x), nil
}
`,
	},
	{
		name:  "nullary_tail",
		offer: true,
		broken: `func run19() (string, error) {
	n := a()? + c()
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run19() (string, error) {
	x := a()?
	n := x + c()
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name:  "nullary_return",
		offer: true,
		broken: `func run20() (string, error) {
	x, y, err := func() (int, int, error) {
		return a()?, c(), nil
	}()
	if err != nil {
		return "", err
	}
	return fmt.Sprint(x, y), nil
}
`,
		fixed: `func run20() (string, error) {
	x, y, err := func() (int, int, error) {
		k := a()?
		return k, c(), nil
	}()
	if err != nil {
		return "", err
	}
	return fmt.Sprint(x, y), nil
}
`,
	},
	{
		name:  "nullary_args",
		offer: true,
		broken: `func run21() (string, error) {
	fmt.Println("args", a()?, c())
	return "p", nil
}
`,
		fixed: `func run21() (string, error) {
	x := a()?
	fmt.Println("args", x, c())
	return "p", nil
}
`,
	},
	{
		name:  "nullary_chain",
		offer: true,
		broken: `func run22() (string, error) {
	s := S{}
	v := s.add(a()?).done()
	return fmt.Sprint(v), nil
}
`,
		fixed: `func run22() (string, error) {
	s := S{}
	x := a()?
	v := s.add(x).done()
	return fmt.Sprint(v), nil
}
`,
	},
	{
		name:  "nullary_nested",
		offer: true,
		broken: `func run23() (string, error) {
	n := a()? + len(mk())
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run23() (string, error) {
	x := a()?
	n := x + len(mk())
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name:  "nullary_two",
		offer: true,
		broken: `func run24() (string, error) {
	n := a()? + c() + c()
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run24() (string, error) {
	x := a()?
	n := x + c() + c()
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name:  "nullary_lambda_tail",
		offer: true,
		broken: `func run25() (string, error) {
	xs := []int{1, 2}
	n := a()? + xs.map(x => x * 2).sum()
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run25() (string, error) {
	xs := []int{1, 2}
	k := a()?
	n := k + xs.map(x => x * 2).sum()
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name:  "empty_literal_tail",
		offer: true,
		broken: `func run26() (string, error) {
	n := a()? + len([]int{})
	return fmt.Sprint(n), nil
}
`,
		fixed: `func run26() (string, error) {
	x := a()?
	n := x + len([]int{})
	return fmt.Sprint(n), nil
}
`,
	},
	{
		name: "defer_arg",
		broken: `func run27() (string, error) {
	defer fmt.Println("deferred", a()?)
	return "d", nil
}
`,
		fixed: `func run27() (string, error) {
	x := a()?
	defer fmt.Println("deferred", x)
	return "d", nil
}
`,
	},
	{
		name: "group_child",
		broken: `func run28() (string, error) {
	out := 0
	go group (context.Background()) {
		go {
			v := a()? + 1
			out = v
		}
	}
	return fmt.Sprint(out), nil
}
`,
		fixed: `func run28() (string, error) {
	out := 0
	go group (context.Background()) {
		go {
			x := a()?
			v := x + 1
			out = v
		}
	}
	return fmt.Sprint(out), nil
}
`,
	},
	{
		name: "switch_header",
		broken: `func run29() (string, error) {
	out := ""
	switch n := a()? + 1; n {
	case 2:
		out = "two"
	default:
		out = "other"
	}
	return out, nil
}
`,
		fixed: `func run29() (string, error) {
	out := ""
	k := a()?
	switch n := k + 1; n {
	case 2:
		out = "two"
	default:
		out = "other"
	}
	return out, nil
}
`,
	},
	{
		name: "for_header",
		broken: `func run30() (string, error) {
	cnt := 0
	for i := 0; i < a()?+1; i++ {
		cnt += i
	}
	return fmt.Sprint(cnt), nil
}
`,
		fixed: `func run30() (string, error) {
	cnt := 0
	for i := 0; ; i++ {
		lim := a()?
		if !(i < lim+1) {
			break
		}
		cnt += i
	}
	return fmt.Sprint(cnt), nil
}
`,
	},
	{
		name: "case_expr",
		broken: `func run31() (string, error) {
	out := ""
	switch {
	case f(2)? > 0:
		out = "pos"
	default:
		out = "other"
	}
	return out, nil
}
`,
		fixed: `func run31() (string, error) {
	out := ""
	y := f(2)?
	switch {
	case y > 0:
		out = "pos"
	default:
		out = "other"
	}
	return out, nil
}
`,
	},
}

func TestMisplacedTryCandidateLaw(t *testing.T) {
	testenv.MustHaveGoRun(t)
	names := make([]string, len(lawCases))
	broken := make([]string, len(lawCases))
	fixed := make([]string, len(lawCases))
	for i, c := range lawCases {
		names[i], broken[i], fixed[i] = c.name, c.broken, c.fixed
	}

	wantOut, err := runLaw(t, assembleLaw(names, fixed).text)
	if err != nil {
		t.Fatalf("hand-written forms do not run: %v\n%s", err, wantOut)
	}
	want := lawOutputs(wantOut)

	cur := slices.Clone(broken)
	applied := make([]int, len(cur))
	noCandidate := make([]bool, len(cur))
	for round := range 4 {
		prog := assembleLaw(names, cur)
		type pick struct {
			i          int
			start, end int
			repl       string
		}
		var picks []pick
		overlap := false
		hasCandidate := make([]bool, len(cur))
		hasDiag := make([]bool, len(cur))
		diags := lintLaw(t, prog.text)
		slices.SortFunc(diags, func(x, y diagRecord) int { return x.Span.ByteOffset - y.Span.ByteOffset })
		for _, d := range diags {
			i := prog.caseAt(d.Span.ByteOffset)
			if i < 0 {
				t.Fatalf("diagnostic at byte %d is outside every case", d.Span.ByteOffset)
			}
			hasDiag[i] = true
			if len(d.AutofixCandidates) == 0 {
				continue
			}
			hasCandidate[i] = true
			fix := d.AutofixCandidates[0]
			if fix.InsertBeforeLine > 0 {
				t.Fatalf("case %s: unexpected insert-before-line candidate", names[i])
			}
			p := pick{i, d.Span.ByteOffset, d.Span.ByteOffset + d.Span.ByteLen, fix.Replacement}
			if tail := strings.TrimLeft(prog.text[p.end:], " \t"); tail != "" && !strings.ContainsRune("\r\n;}/", rune(tail[0])) {
				t.Errorf("case %s: candidate span ends inside the statement, before %q", names[i], tail[:min(len(tail), 12)])
			}
			if slices.ContainsFunc(picks, func(q pick) bool { return q.i == i && p.start < q.end && q.start < p.end }) {
				overlap = true
				continue
			}
			picks = append(picks, p)
		}
		if round == 0 {
			for i := range cur {
				if !hasDiag[i] {
					t.Errorf("case %s: broken form has no %s diagnostic", names[i], misplacedTryRuleID)
				}
				if hasDiag[i] && !hasCandidate[i] {
					noCandidate[i] = true
					cur[i] = fixed[i]
				}
			}
		}
		slices.SortFunc(picks, func(x, y pick) int { return y.start - x.start })
		for _, p := range picks {
			if noCandidate[p.i] {
				continue
			}
			rel, relEnd := p.start-prog.starts[p.i], p.end-prog.starts[p.i]
			cur[p.i] = cur[p.i][:rel] + p.repl + cur[p.i][relEnd:]
			applied[p.i]++
		}
		if !overlap {
			break
		}
	}

	for i, c := range lawCases {
		switch {
		case noCandidate[i]:
			t.Logf("case %s: no candidate offered", c.name)
			if c.offer {
				t.Errorf("case %s: no candidate offered, want one", c.name)
			}
		case applied[i] == 0 && c.offer:
			t.Errorf("case %s: no candidate applied, want one", c.name)
		}
	}

	appliedSrc := slices.Clone(cur)
	var gotOut string
	for attempt := 0; ; attempt++ {
		prog := assembleLaw(names, cur)
		out, err := runLaw(t, prog.text)
		if err == nil {
			gotOut = out
			break
		}
		var bad []int
		for _, m := range lawErrLine.FindAllStringSubmatch(out, -1) {
			line, _ := strconv.Atoi(m[1])
			if i := prog.caseAtLine(line); i >= 0 && !slices.Contains(bad, i) {
				bad = append(bad, i)
			}
		}
		if len(bad) == 0 || attempt == 3 {
			t.Fatalf("candidate-applied program does not build: %v\n%s\n%s", err, out, prog.text)
		}
		for _, i := range bad {
			t.Errorf("case %s: applied candidate does not build:\n%s\n--- applied source:\n%s", names[i], out, appliedSrc[i])
			cur[i] = fixed[i]
			noCandidate[i] = true
		}
	}
	got := lawOutputs(gotOut)
	for i, name := range names {
		if !noCandidate[i] && got[name] != want[name] {
			t.Errorf("case %s: applied candidate changes the output\n--- applied source:\n%s--- want:\n%s--- got:\n%s", name, appliedSrc[i], want[name], got[name])
		}
	}
}
