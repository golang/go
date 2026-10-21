// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"internal/testenv"
)

// logosIcfMod is the go.mod of every ICF fixture module.
const logosIcfMod = "module icftest\n\ngo 1.27\n"

// logosIcfFoldMain holds two byte-identical functions sharing one helper.
const logosIcfFoldMain = `package main

import "fmt"

//go:noinline
func Alpha(n int) int {
	s := 0
	for i := 0; i < n; i++ {
		s += i * (i + 1)
	}
	return s + logosIcfAdd(n)
}

//go:noinline
func Beta(n int) int {
	s := 0
	for i := 0; i < n; i++ {
		s += i * (i + 1)
	}
	return s + logosIcfAdd(n)
}

//go:noinline
func logosIcfAdd(n int) int {
	return 2*n + 1
}

func main() {
	fmt.Println(Alpha(100), Beta(100))
}
`

// logosIcfRelocMain holds two functions differing in one call target.
const logosIcfRelocMain = `package main

import "fmt"

//go:noinline
func one() int { return 1 }

//go:noinline
func two() int { return 2 }

//go:noinline
func Alpha(n int) int {
	s := 0
	for i := 0; i < n; i++ {
		s += i * (i + 1)
	}
	return s + one()
}

//go:noinline
func Beta(n int) int {
	s := 0
	for i := 0; i < n; i++ {
		s += i * (i + 1)
	}
	return s + two()
}

func main() {
	fmt.Println(Alpha(10), Beta(10))
}
`

// logosIcfItabMain converts two pointer types with identical method
// bodies to one interface and calls only M, so each itab keeps a weak
// reference to its unreachable N method next to a foldable M method.
const logosIcfItabMain = `package main

import "fmt"

type I interface {
	M() int
	N() int
}

type A struct{ x int }
type B struct{ x int }

//go:noinline
func (a *A) M() int { return a.x*3 + 7 }

//go:noinline
func (a *A) N() int { return a.x*5 + 9 }

//go:noinline
func (b *B) M() int { return b.x*3 + 7 }

//go:noinline
func (b *B) N() int { return b.x*5 + 9 }

func main() {
	items := []I{&A{1}, &B{2}}
	s := 0
	for _, it := range items {
		s += it.M()
	}
	fmt.Println(s)
}
`

// logosIcfAsmMain declares two assembly functions with identical bodies.
const logosIcfAsmMain = `package main

import "fmt"

func AsmA() int
func AsmB() int

func main() {
	fmt.Println(AsmA(), AsmB())
}
`

// logosIcfAsm holds two byte-identical assembly function bodies.
const logosIcfAsm = `#include "textflag.h"

TEXT ·AsmA(SB), NOSPLIT, $0-8
	MOVQ $7, AX
	MOVQ AX, ret+0(FP)
	RET

TEXT ·AsmB(SB), NOSPLIT, $0-8
	MOVQ $7, AX
	MOVQ AX, ret+0(FP)
	RET
`

// TestLogosIcfFoldsIdentical links a program with two identical
// functions and requires one emitted copy and the same output.
func TestLogosIcfFoldsIdentical(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	logosIcfWriteFiles(t, dir, map[string]string{
		"go.mod":  logosIcfMod,
		"main.go": logosIcfFoldMain,
	})
	plain := filepath.Join(dir, "plain.exe")
	folded := filepath.Join(dir, "folded.exe")
	logosIcfBuild(t, dir, plain)
	logosIcfBuild(t, dir, folded, "-logos.icf")
	outPlain := logosIcfRun(t, plain)
	outFolded := logosIcfRun(t, folded)
	if outPlain != "333501 333501\n" {
		t.Errorf("plain output = %q", outPlain)
	}
	if outFolded != outPlain {
		t.Errorf("folded output = %q, plain = %q", outFolded, outPlain)
	}
	nmPlain := logosIcfNm(t, plain)
	if n := logosIcfTextCount(t, plain, "main.Alpha"); n != 1 {
		t.Errorf("plain nm holds %d main.Alpha text copies, want 1", n)
	}
	if n := logosIcfTextCount(t, plain, "main.Beta"); n != 1 {
		t.Errorf("plain nm holds %d main.Beta text copies, want 1", n)
	}
	nmFolded := logosIcfNm(t, folded)
	if n := logosIcfTextCount(t, folded, "main.Alpha") + logosIcfTextCount(t, folded, "main.Beta"); n != 1 {
		t.Errorf("folded nm keeps %d text copies, want 1", n)
	}
	if logosIcfCount(nmPlain, "runtime.") != logosIcfCount(nmFolded, "runtime.") {
		t.Errorf("runtime symbol count changed: plain=%d folded=%d", logosIcfCount(nmPlain, "runtime."), logosIcfCount(nmFolded, "runtime."))
	}
	if logosIcfCount(nmFolded, "runtime.") < 50 {
		t.Errorf("folded nm holds %d runtime symbols, want at least 50", logosIcfCount(nmFolded, "runtime."))
	}
}

// logosIcfGcMain holds two functions with identical machine code but
// different pointer maps: F threads a pointer across a call while G
// threads a scalar. Deep recursion with live pointers across growth
// and collection smokes the maps at runtime.
const logosIcfGcMain = `package main

import (
	"fmt"
	"runtime"
)

type logosIcfBox struct{ v int }

//go:noinline
func F(p *logosIcfBox) *logosIcfBox {
	runtime.Gosched()
	return p
}

//go:noinline
func G(x uintptr) uintptr {
	runtime.Gosched()
	return x
}

//go:noinline
func grow(n int, p *logosIcfBox) *logosIcfBox {
	if n == 0 {
		return F(p)
	}
	q := F(p)
	if n%32 == 0 {
		runtime.GC()
	}
	return grow(n-1, q)
}

func main() {
	b := &logosIcfBox{v: 41}
	b = grow(2000, b)
	fmt.Println(b.v+1, G(1))
}
`

// TestLogosIcfKeepsPointerMaps links two functions sharing machine
// code but not liveness bitmaps and requires both copies with correct
// output, including under the collector's mark checker.
func TestLogosIcfKeepsPointerMaps(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	logosIcfWriteFiles(t, dir, map[string]string{
		"go.mod":  logosIcfMod,
		"main.go": logosIcfGcMain,
	})
	exe := filepath.Join(dir, "folded.exe")
	logosIcfBuild(t, dir, exe, "-logos.icf")
	if got := logosIcfRun(t, exe); got != "42 1\n" {
		t.Errorf("folded output = %q, want %q", got, "42 1\n")
	}
	if got := logosIcfRunEnv(t, exe, "GODEBUG=gccheckmark=1"); got != "42 1\n" {
		t.Errorf("gccheckmark output = %q, want %q", got, "42 1\n")
	}
	names := logosIcfNm(t, exe)
	if logosIcfCount(names, "main.F") != 1 || logosIcfCount(names, "main.G") != 1 {
		t.Errorf("folded nm lost a pointer-map variant: F=%d G=%d", logosIcfCount(names, "main.F"), logosIcfCount(names, "main.G"))
	}
}

// TestLogosIcfKeepsWeakItabRelocs folds identical methods referenced from
// itabs and requires the link to succeed: rewriting an itab clones it,
// and the clone must keep the weak bit of its references to methods that
// dead-code elimination left unreachable.
func TestLogosIcfKeepsWeakItabRelocs(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	logosIcfWriteFiles(t, dir, map[string]string{
		"go.mod":  logosIcfMod,
		"main.go": logosIcfItabMain,
	})
	exe := filepath.Join(dir, "folded.exe")
	logosIcfBuild(t, dir, exe, "-logos.icf")
	if got := logosIcfRun(t, exe); got != "23\n" {
		t.Errorf("folded output = %q, want %q", got, "23\n")
	}
}

// TestLogosIcfKeepsRelocTargets links two functions differing only in
// a relocation target and requires both copies with correct output.
func TestLogosIcfKeepsRelocTargets(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	logosIcfWriteFiles(t, dir, map[string]string{
		"go.mod":  logosIcfMod,
		"main.go": logosIcfRelocMain,
	})
	exe := filepath.Join(dir, "folded.exe")
	logosIcfBuild(t, dir, exe, "-logos.icf")
	if got := logosIcfRun(t, exe); got != "331 332\n" {
		t.Errorf("folded output = %q, want %q", got, "331 332\n")
	}
	names := logosIcfNm(t, exe)
	if logosIcfCount(names, "main.Alpha") != 1 || logosIcfCount(names, "main.Beta") != 1 {
		t.Errorf("folded nm lost a function: Alpha=%d Beta=%d", logosIcfCount(names, "main.Alpha"), logosIcfCount(names, "main.Beta"))
	}
}

// TestLogosIcfSkipsAssembly links two identical assembly functions and
// requires both copies with correct output.
func TestLogosIcfSkipsAssembly(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	logosIcfWriteFiles(t, dir, map[string]string{
		"go.mod":  logosIcfMod,
		"main.go": logosIcfAsmMain,
		"asm.s":   logosIcfAsm,
	})
	exe := filepath.Join(dir, "folded.exe")
	logosIcfBuild(t, dir, exe, "-logos.icf")
	if got := logosIcfRun(t, exe); got != "7 7\n" {
		t.Errorf("folded output = %q, want %q", got, "7 7\n")
	}
	names := logosIcfNm(t, exe)
	if logosIcfCount(names, "main.AsmA") != 1 || logosIcfCount(names, "main.AsmB") != 1 {
		t.Errorf("folded nm lost an assembly function: AsmA=%d AsmB=%d", logosIcfCount(names, "main.AsmA"), logosIcfCount(names, "main.AsmB"))
	}
}

// logosIcfTransitiveMain holds two callers that differ only in which
// of two identical callees they call. The callees fold in the first
// pass; the callers fold in the second, once rewriting canonicalizes
// their targets.
const logosIcfTransitiveMain = `package main

import "fmt"

//go:noinline
func c1() int { return 3 }

//go:noinline
func c2() int { return 3 }

//go:noinline
func A() int { return c1() + 1 }

//go:noinline
func B() int { return c2() + 1 }

func main() {
	fmt.Println(A(), B())
}
`

// TestLogosIcfFoldsTransitiveCallers links callers that become
// identical only after their callees fold and requires one copy of
// each pair with correct output.
func TestLogosIcfFoldsTransitiveCallers(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	logosIcfWriteFiles(t, dir, map[string]string{
		"go.mod":  logosIcfMod,
		"main.go": logosIcfTransitiveMain,
	})
	exe := filepath.Join(dir, "folded.exe")
	logosIcfBuild(t, dir, exe, "-logos.icf")
	if got := logosIcfRun(t, exe); got != "4 4\n" {
		t.Errorf("folded output = %q, want %q", got, "4 4\n")
	}
	names := logosIcfNm(t, exe)
	if logosIcfCount(names, "main.A")+logosIcfCount(names, "main.B") != 1 {
		t.Errorf("folded nm keeps %d caller copies, want 1", logosIcfCount(names, "main.A")+logosIcfCount(names, "main.B"))
	}
	if logosIcfCount(names, "main.c1")+logosIcfCount(names, "main.c2") != 1 {
		t.Errorf("folded nm keeps %d callee copies, want 1", logosIcfCount(names, "main.c1")+logosIcfCount(names, "main.c2"))
	}
}

// logosIcfCgoMain holds two byte-identical cgo-exported functions.
// C callers would force external linking, where dynamic exports pin
// already; the internal link here covers the gap where static exports
// stay out of dynexp. Exports resolve by symbol, so each keeps its
// own entry.
const logosIcfCgoMain = `package main

import "C"

import "fmt"

//go:noinline
func seriesSum(n int) int {
	s := 0
	for i := range n {
		s += i * (i + 1)
	}
	return s
}

//export GoF
//go:noinline
func GoF() C.int { return C.int(seriesSum(7)) }

//export GoG
//go:noinline
func GoG() C.int { return C.int(seriesSum(7)) }

func main() {
	fmt.Println(int(GoF()), int(GoG()))
}
`

// TestLogosIcfPinsCgoExports links two identical cgo exports with an
// internal link and requires both entries with correct output.
func TestLogosIcfPinsCgoExports(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	dir := t.TempDir()
	logosIcfWriteFiles(t, dir, map[string]string{
		"go.mod":  logosIcfMod,
		"main.go": logosIcfCgoMain,
	})
	exe := filepath.Join(dir, "folded.exe")
	logosIcfBuild(t, dir, exe, "-logos.icf", "-linkmode=internal")
	if got := logosIcfRun(t, exe); got != "112 112\n" {
		t.Errorf("folded output = %q, want %q", got, "112 112\n")
	}
	if n := logosIcfTextCount(t, exe, "main.GoF"); n != 1 {
		t.Errorf("folded nm holds %d GoF copies, want 1", n)
	}
	if n := logosIcfTextCount(t, exe, "main.GoG"); n != 1 {
		t.Errorf("folded nm holds %d GoG copies, want 1", n)
	}
}

// TestLogosIcfRuntimeSkip pins the runtime and assembly fold exclusions.
func TestLogosIcfRuntimeSkip(t *testing.T) {
	for _, tc := range []struct {
		pkg  string
		asm  bool
		want bool
	}{
		{"runtime", false, true},
		{"runtime/cgo", false, true},
		{"main", false, false},
		{"fmt", false, false},
		{"main", true, true},
		{"runtime", true, true},
	} {
		if got := logosIcfSkip(tc.pkg, tc.asm); got != tc.want {
			t.Errorf("logosIcfSkip(%q, %v) = %v, want %v", tc.pkg, tc.asm, got, tc.want)
		}
	}
}

// TestLogosIcfGenericSingleCopy links one generic instantiation used
// from two packages and requires a single copy with correct output.
func TestLogosIcfGenericSingleCopy(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	logosIcfWriteFiles(t, dir, map[string]string{
		"go.mod":   logosIcfMod,
		"main.go":  "package main\n\nimport (\n\t\"fmt\"\n\t\"icftest/p1\"\n\t\"icftest/p2\"\n)\n\nfunc main() {\n\tfmt.Println(p1.Get() + p2.Get())\n}\n",
		"g/g.go":   "package g\n\n//go:noinline\nfunc First[T any](s []T) T {\n\treturn s[0]\n}\n",
		"p1/p1.go": "package p1\n\nimport \"icftest/g\"\n\n//go:noinline\nfunc Get() int {\n\treturn g.First([]int{7})\n}\n",
		"p2/p2.go": "package p2\n\nimport \"icftest/g\"\n\n//go:noinline\nfunc Get() int {\n\treturn g.First([]int{7})\n}\n",
	})
	plain := filepath.Join(dir, "plain.exe")
	folded := filepath.Join(dir, "folded.exe")
	logosIcfBuild(t, dir, plain)
	logosIcfBuild(t, dir, folded, "-logos.icf")
	if got := logosIcfRun(t, plain); got != "14\n" {
		t.Errorf("plain output = %q, want %q", got, "14\n")
	}
	if got := logosIcfRun(t, folded); got != "14\n" {
		t.Errorf("folded output = %q, want %q", got, "14\n")
	}
	if n := logosIcfTextCount(t, plain, "First["); n != 1 {
		t.Errorf("plain nm holds %d First copies, want 1", n)
	}
	if n := logosIcfTextCount(t, folded, "First["); n != 1 {
		t.Errorf("folded nm holds %d First copies, want 1", n)
	}
}

// logosIcfWriteFiles writes one test module into dir.
func logosIcfWriteFiles(t *testing.T, dir string, files map[string]string) {
	t.Helper()
	for name, body := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o777); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o666); err != nil {
			t.Fatal(err)
		}
	}
}

// logosIcfBuild links one test module, appending extra ldflags when given.
func logosIcfBuild(t *testing.T, dir string, out string, ldflags ...string) {
	t.Helper()
	args := []string{"build", "-o", out}
	if len(ldflags) > 0 {
		args = append(args, "-ldflags="+strings.Join(ldflags, " "))
	}
	args = append(args, ".")
	cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOTOOLCHAIN=local")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build %v: %v\n%s", args, err, out)
	}
}

// logosIcfRun runs one linked binary and returns its stdout.
func logosIcfRun(t *testing.T, exe string) string {
	t.Helper()
	return logosIcfRunEnv(t, exe)
}

// logosIcfRunEnv runs one linked binary with extra environment entries
// and returns its stdout.
func logosIcfRunEnv(t *testing.T, exe string, env ...string) string {
	t.Helper()
	cmd := testenv.Command(t, exe)
	cmd.Env = append(os.Environ(), env...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run %s: %v\n%s", exe, err, out)
	}
	return string(out)
}

// logosIcfNmLines returns the raw symbol lines of one linked binary.
func logosIcfNmLines(t *testing.T, exe string) []string {
	t.Helper()
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "nm", exe)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm %s: %v\n%s", exe, err, out)
	}
	var lines []string
	for line := range strings.Lines(string(out)) {
		if strings.TrimSpace(line) == "" {
			continue
		}
		lines = append(lines, line)
	}
	return lines
}

// logosIcfNm returns the symbol names of one linked binary.
func logosIcfNm(t *testing.T, exe string) []string {
	t.Helper()
	var names []string
	for _, line := range logosIcfNmLines(t, exe) {
		f := strings.Fields(line)
		names = append(names, f[len(f)-1])
	}
	return names
}

// logosIcfTextCount counts text symbols holding sub.
func logosIcfTextCount(t *testing.T, exe string, sub string) int {
	t.Helper()
	n := 0
	for _, line := range logosIcfNmLines(t, exe) {
		f := strings.Fields(line)
		if len(f) < 3 || f[len(f)-2] != "T" {
			continue
		}
		if strings.Contains(f[len(f)-1], sub) {
			n++
		}
	}
	return n
}

// logosIcfCount counts names holding sub.
func logosIcfCount(names []string, sub string) int {
	n := 0
	for _, name := range names {
		if strings.Contains(name, sub) {
			n++
		}
	}
	return n
}
