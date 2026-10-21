// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// lowerSnippetSrc parses, checks, and lowers one inline .lgo source,
// returning the ejected text. It mirrors lowerSpreadSrc for the M6
// red-team gap probes below.
func lowerSnippetSrc(t *testing.T, name, src string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(src), 0666); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var perrs []error
	f, err := syntax.ParseFile(path, func(err error) { perrs = append(perrs, err) }, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatal(errors.Join(append(perrs, err)...))
	}
	imp := newTestImporter(t)
	info := &types2.Info{
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
	}
	var cerrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { cerrs = append(cerrs, err) },
		Logos:    true,
	}
	files := []*syntax.File{f}
	pkg, _ := conf.Check("main", files, info)
	if err := errors.Join(cerrs...); err != nil {
		t.Fatalf("type check: %v", err)
	}
	changed, edits := Lower(pkg, info, files, map[*syntax.File][]byte{f: data})
	if !changed {
		t.Fatalf("Lower reported no change for %s source", name)
	}
	out, err := ApplyEdits(data, f.Pos().FileBase(), edits[f])
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

// TestScopeDeferOpenCoded proves SPEC #52 and #113 under the revised
// T-104.7 technique: each scope defer binds its callee at the statement
// behind an armed flag, registers one guarding `defer` for the panic
// path, and every block exit runs the cleanup inline. There is no
// function-literal wrapper around the block and no separate unwind
// table: exactly the two guard closures are the only `func()` literals,
// and no identifier mentions unwinding.
func TestScopeDeferOpenCoded(t *testing.T) {
	src := "package main\n\nimport \"fmt\"\n\nfunc note(s string) { fmt.Println(s) }\n\nfunc f(n int) int {\n\t{\n\t\tnote(\"lock\")\n\t\tscope defer note(\"unlock\")\n\t\tnote(\"body\")\n\t\tif n > 0 {\n\t\t\treturn 1\n\t\t}\n\t}\n\treturn 0\n}\n\nfunc g() {\n\tfor range 3 {\n\t\tscope defer note(\"iter\")\n\t\tnote(\"work\")\n\t}\n}\n"
	out := lowerSnippetSrc(t, "scopesnip.lgo", src)
	// One guarding defer per scope-defer site: block exit (f) and loop
	// body (g). The guard is the panic path (#113).
	if n := strings.Count(out, "defer func() {"); n != 2 {
		t.Errorf("got %d guarding defers, want 2 (one per scope defer site)", n)
	}
	// No function-literal wrapper: the guards are every func literal.
	if n := strings.Count(out, "func()"); n != 2 {
		t.Errorf("got %d func() literals, want 2 (the guards only, no block wrapper)", n)
	}
	// Armed-flag inline stanzas: guard-if plus return and fallthrough
	// exits in f, guard-if plus iteration-end in g.
	if n := strings.Count(out, "if _logos_sd0 {"); n != 5 {
		t.Errorf("got %d armed-flag stanzas, want 5", n)
	}
	// Callee bound at the statement, not re-resolved at exit.
	if !strings.Contains(out, "_logos_f0 := note") {
		t.Errorf("callee capture missing:\n%s", out)
	}
	// No separate unwind table under any spelling.
	if strings.Contains(strings.ToLower(out), "unwind") {
		t.Errorf("unwind table reference in open-coded lowering:\n%s", out)
	}
}

// TestChainSumPairwise proves SPEC #67 (M6 half): the sum terminal
// lowers to a fixed pairwise recursion over halves, with no multiply
// operator anywhere, so no implicit FMA contraction can apply. The
// run fixture (crafted 1e16/-1e16/1.0 input) pins the bit-exact result.
// The test reads the scalar lowering, so it pins a build without the
// simd experiment: the vector form of a float slice sum stages the
// elements through array-pointer conversions and keeps this recursion
// (TestSimdSliceSumFloatKeepsTree).
func TestChainSumPairwise(t *testing.T) {
	scalarSliceTarget(t)
	src := "package main\n\nfunc f(xs []float64) float64 {\n\treturn xs.sum()\n}\n"
	out := lowerSnippetSrc(t, "sumsnip.lgo", src)
	// Halving split of a recursive helper.
	if !strings.Contains(out, "/ 2") {
		t.Errorf("pairwise halving split missing:\n%s", out)
	}
	// Fixed order: both halves call the same helper (two recursive
	// calls plus the top-level invocation), left half text before
	// right half text, joined by one plus.
	if n := strings.Count(out, "_sum("); n != 3 {
		t.Errorf("got %d helper calls, want 3 (top-level plus left and right halves)", n)
	}
	left := strings.Index(out, "_s[:")
	right := strings.Index(out, "_m:]")
	if left < 0 || right < 0 || left >= right {
		t.Errorf("left-half slice not before right-half slice:\n%s", out)
	}
	if !strings.Contains(out, ") + ") {
		t.Errorf("halves not joined by plus:\n%s", out)
	}
	// No multiply, hence no FMA contraction surface.
	if strings.Contains(out, "*") {
		t.Errorf("multiply operator in sum lowering (FMA surface):\n%s", out)
	}
	if strings.Contains(out, "FMA") {
		t.Errorf("FMA reference in sum lowering:\n%s", out)
	}
}

// TestChainFuseOneLoop proves SPEC #84: consecutive filter/map/take
// calls fuse into exactly one loop, with the receiver bound once
// before it.
func TestChainFuseOneLoop(t *testing.T) {
	src := "package main\n\nfunc bump(x int) int { return 2 * x }\n\nfunc f(xs []int) []int {\n\treturn xs.filter(x => x > 2).map(x => bump(x)).take(5)\n}\n"
	out := lowerSnippetSrc(t, "fusesnip.lgo", src)
	if n := len(regexp.MustCompile(`\bfor\b`).FindAllString(out, -1)); n != 1 {
		t.Errorf("got %d for loops, want exactly 1 fused loop:\n%s", n, out)
	}
	if n := len(regexp.MustCompile(`_rcv := xs`).FindAllString(out, -1)); n != 1 {
		t.Errorf("got %d receiver bindings, want 1:\n%s", n, out)
	}
	if n := strings.Count(out, "xs"); n != 2 {
		t.Errorf("got %d xs mentions (param + one binding), want 2:\n%s", n, out)
	}
}

// TestChainDelimSpans pins the delimiter-span family: a composite
// literal lambda body and an index-expression take argument keep their
// closing delimiter in the lowered text instead of truncating it (which
// left an unparseable fragment and an internal compiler error).
func TestChainDelimSpans(t *testing.T) {
	bodySrc := "package main\n\ntype Point struct{ X int }\n\nfunc f(xs []int) []Point {\n\treturn xs.map(x => Point{X: x})\n}\n"
	out := lowerSnippetSrc(t, "bodysnip.lgo", bodySrc)
	if !strings.Contains(out, "Point{X: x}") {
		t.Errorf("composite body lost its closing brace:\n%s", out)
	}
	argSrc := "package main\n\nfunc f(xs []int, ys []int) []int {\n\treturn xs.take(ys[0])\n}\n"
	out = lowerSnippetSrc(t, "argsnip.lgo", argSrc)
	if !strings.Contains(out, "ys[0]") {
		t.Errorf("index argument lost its closing bracket:\n%s", out)
	}
}

// TestInterpBindBeforeSizing proves SPEC #126: each `{expr}` binds to a
// local left to right before the buffer is sized, so sizing and
// appending share one evaluation.
func TestInterpBindBeforeSizing(t *testing.T) {
	src := "package main\n\nimport \"fmt\"\n\nfunc f(name string, n int) {\n\tfmt.Println($\"User {name} has {n} items\")\n}\n"
	out := lowerSnippetSrc(t, "interpsnip.lgo", src)
	v0 := strings.Index(out, "_logos_v0 :=")
	v1 := strings.Index(out, "_logos_v1 :=")
	mk := strings.Index(out, "make([]byte, 0,")
	if v0 < 0 || v1 < 0 || mk < 0 {
		t.Fatalf("bind-before-size shape missing:\n%s", out)
	}
	if v0 >= v1 || v1 >= mk {
		t.Errorf("wrong order: v0 at %d, v1 at %d, make at %d (want v0 < v1 < make)", v0, v1, mk)
	}
	if n := strings.Count(out, "_logos_v0 :="); n != 1 {
		t.Errorf("operand 0 bound %d times, want once", n)
	}
	if n := strings.Count(out, "_logos_v1 :="); n != 1 {
		t.Errorf("operand 1 bound %d times, want once", n)
	}
	if !strings.Contains(out, "len(_logos_v0)") {
		t.Errorf("bound does not count the string operand:\n%s", out)
	}
}
