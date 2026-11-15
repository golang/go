// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
)

func chainGrowthSrc(n int) string {
	var sb strings.Builder
	sb.WriteString("package main\n\nfunc bump(x int) int { return 2 * x }\n\nfunc main() {\nxs := []int{3, 1, 4, 1, 5, 9, 2, 6}\n")
	for range n {
		sb.WriteString("_ = xs.filter(x => x > 2).map(x => bump(x)).take(2)\n")
	}
	sb.WriteString("}\n")
	return sb.String()
}

func interpGrowthSrc(n int) string {
	var sb strings.Builder
	sb.WriteString("package main\n\nfunc main() {\nname := \"go\"\nx := 7\n")
	for range n {
		sb.WriteString("_ = $\"hello {name} n={x}\"\n")
	}
	sb.WriteString("}\n")
	return sb.String()
}

func lambdaGrowthSrc(n int) string {
	var sb strings.Builder
	sb.WriteString("package main\n\nfunc main() {\nn := 3\n")
	for i := range n {
		fmt.Fprintf(&sb, "var f%d func(int) int = x => { var g func(int) int = y => x + y + n; return g(1) }\n_ = f%d(2)\n", i, i)
	}
	sb.WriteString("}\n")
	return sb.String()
}

func lowerBytes(t *testing.T, name, src string, lowered func(out string) bool) float64 {
	t.Helper()
	if !lowered(lowerSnippetSrc(t, name, src)) {
		t.Fatalf("%s did not lower", name)
	}
	best := float64(1 << 62)
	for range 5 {
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		lowerSnippetSrc(t, name, src)
		runtime.ReadMemStats(&after)
		if d := float64(after.TotalAlloc - before.TotalAlloc); d < best {
			best = d
		}
	}
	return best
}

func assertGrowthBound(t *testing.T, name string, gen func(n int) string, lowered func(out string) bool) {
	t.Helper()
	small := lowerBytes(t, name, gen(80), lowered)
	large := lowerBytes(t, name, gen(320), lowered)
	t.Logf("lower bytes per run: small=%v large=%v", small, large)
	if small == 0 {
		t.Fatal("no allocations measured at small size")
	}
	if ratio := large / small; ratio > 6 {
		t.Fatalf("%s lowering allocated %.1fx bytes for 4x input, want at most 6x", name, ratio)
	}
}

func TestChainLowerGrowthBound(t *testing.T) {
	nested := lowerSnippetSrc(t, "lambdanest.lgo", "package main\n\nfunc main() {\nvar f func(int) int = x => { var g func(int) int = y => x + y; return g(1) }\n_ = f(2)\n}\n")
	if strings.Contains(nested, "=>") {
		t.Fatal("nested lambda survived lowering")
	}
	assertGrowthBound(t, "growth.lgo", chainGrowthSrc, func(out string) bool { return strings.Contains(out, "for _, ") })
}

func TestInterpLowerGrowthBound(t *testing.T) {
	assertGrowthBound(t, "interpgrowth.lgo", interpGrowthSrc, func(out string) bool { return !strings.Contains(out, "$\"") })
}

func TestLambdaLowerGrowthBound(t *testing.T) {
	assertGrowthBound(t, "lambdagrowth.lgo", lambdaGrowthSrc, func(out string) bool { return !strings.Contains(out, "=>") })
}

func TestChainMultiCallOneLoopShape(t *testing.T) {
	src := "package main\n\nfunc main() {\nprobeItems := []int{3, 1, 4}\nprobeOut := probeItems.filter(probeX => probeX > 1).map(probeX => probeX * 2).take(2)\n_ = probeOut\n}\n"
	out := lowerSnippetSrc(t, "chainshape.lgo", src)
	before, _, ok := strings.CutLast(out, "_rcv := ")
	if !ok {
		t.Fatalf("lowered chain has no receiver binding:\n%s", out)
	}
	bi := len(before)
	_, tail, ok := strings.CutLast(before, "_logos_ch")
	if !ok {
		t.Fatalf("receiver binding lacks a chain prefix:\n%s", out)
	}
	p := "_logos_ch" + tail
	if c := strings.Count(out, p+"_rcv := "); c != 1 {
		t.Fatalf("chain receiver is bound %d times, want exactly one:\n%s", c, out)
	}
	if c := strings.Count(out, "_e := range "); c != 1 {
		t.Fatalf("lowered chain has %d iteration loops, want exactly one:\n%s", c, out)
	}
	loop := p + "_e := range " + p + "_rcv {"
	if c := strings.Count(out, loop); c != 1 {
		t.Fatalf("the single iteration loop must range the bound receiver:\n%s", out)
	}
	if ri := strings.Index(out, loop); ri < bi {
		t.Fatalf("receiver binding must precede the iteration loop:\n%s", out)
	}
	for _, want := range []string{
		p + "_tn2 := ",
		p + "_tc2 := 0",
		"if " + p + "_tc2 >= " + p + "_tn2 { break }",
		"if !" + p + "_f0(",
		p + "_m1 := ",
		p + "_tc2++",
	} {
		if c := strings.Count(out, want); c != 1 {
			t.Fatalf("fused stage %q appears %d times, want exactly once:\n%s", want, c, out)
		}
	}
}

func TestInterpOperandLocalBindingShape(t *testing.T) {
	src := "package main\nimport \"fmt\"\ntype ProbeShout struct{}\nfunc (ProbeShout) String() string { return \"shout\" }\nfunc probeCall(probeX int) int { return probeX + 1 }\nfunc main() {\nvar probeErr error\nvar probeStr fmt.Stringer = ProbeShout{}\nout := $\"x={probeCall(7)} e={probeErr} s={probeStr}\"\n_ = out\n_ = probeErr\n_ = probeStr\n}\n"
	out := lowerSnippetSrc(t, "interpshape.lgo", src)
	if c := strings.Count(out, "probeCall(7)"); c != 1 {
		t.Fatalf("operand probeCall(7) occurs %d times, want one binding site:\n%s", c, out)
	}
	binds := []string{"_logos_v0 := probeCall(7)", "_logos_v1 := probeErr", "_logos_v2 := probeStr"}
	last := -1
	for _, bind := range binds {
		i := strings.Index(out, bind)
		if i < 0 || strings.Count(out, bind) != 1 {
			t.Fatalf("operand binding %q must occur exactly once:\n%s", bind, out)
		}
		if i <= last {
			t.Fatalf("operand bindings are not left-to-right at %q:\n%s", bind, out)
		}
		last = i
	}
	if strings.Contains(out, "_logos_s0") {
		t.Fatalf("plain operand must not get a conversion binding:\n%s", out)
	}
	mi := strings.Index(out, "_logos_b := make([]byte, 0, ")
	if mi < 0 {
		t.Fatalf("lowered interpolation has no sizing statement:\n%s", out)
	}
	convs := []struct{ conv, after string }{
		{"_logos_s1 := _logos_rt.InterpError(_logos_v1, _logos_fmt.Sprint)", "_logos_v1 := probeErr"},
		{"_logos_s2 := _logos_rt.InterpStringer(_logos_v2, _logos_fmt.Sprint)", "_logos_v2 := probeStr"},
	}
	for _, cv := range convs {
		ci := strings.Index(out, cv.conv)
		ai := strings.Index(out, cv.after)
		if ci < 0 || strings.Count(out, cv.conv) != 1 || ai < 0 {
			t.Fatalf("conversion binding %q must occur exactly once after its operand binding:\n%s", cv.conv, out)
		}
		if ci < ai {
			t.Fatalf("conversion binding %q must follow its operand binding:\n%s", cv.conv, out)
		}
		if ci > mi {
			t.Fatalf("conversion binding %q must precede sizing:\n%s", cv.conv, out)
		}
	}
	bound, _, _ := strings.Cut(out[mi:], "\n")
	for _, want := range []string{"len(_logos_s1)", "len(_logos_s2)"} {
		if !strings.Contains(bound, want) {
			t.Fatalf("sizing bound %q must use converted binding %q:\n%s", bound, want, out)
		}
	}
	for _, want := range []string{
		"_logos_b = append(_logos_b, _logos_s1...)",
		"_logos_b = append(_logos_b, _logos_s2...)",
	} {
		if c := strings.Count(out, want); c != 1 {
			t.Fatalf("append must use converted binding %q, got %d occurrences:\n%s", want, c, out)
		}
	}
}
