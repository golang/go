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
// functions and requires one copy, a smaller binary, and same output.
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
	if logosIcfSize(t, folded) >= logosIcfSize(t, plain) {
		t.Errorf("folded size %d not smaller than plain size %d", logosIcfSize(t, folded), logosIcfSize(t, plain))
	}
	outPlain := logosIcfRun(t, plain)
	outFolded := logosIcfRun(t, folded)
	if outPlain != "333501 333501\n" {
		t.Errorf("plain output = %q", outPlain)
	}
	if outFolded != outPlain {
		t.Errorf("folded output = %q, plain = %q", outFolded, outPlain)
	}
	nmPlain := logosIcfNm(t, plain)
	if logosIcfCount(nmPlain, "main.Alpha") != 1 || logosIcfCount(nmPlain, "main.Beta") != 1 {
		t.Errorf("plain nm lost a function: Alpha=%d Beta=%d", logosIcfCount(nmPlain, "main.Alpha"), logosIcfCount(nmPlain, "main.Beta"))
	}
	nmFolded := logosIcfNm(t, folded)
	if logosIcfCount(nmFolded, "main.Alpha")+logosIcfCount(nmFolded, "main.Beta") != 1 {
		t.Errorf("folded nm keeps %d copies, want 1", logosIcfCount(nmFolded, "main.Alpha")+logosIcfCount(nmFolded, "main.Beta"))
	}
	if logosIcfCount(nmPlain, "runtime.") != logosIcfCount(nmFolded, "runtime.") {
		t.Errorf("runtime symbol count changed: plain=%d folded=%d", logosIcfCount(nmPlain, "runtime."), logosIcfCount(nmFolded, "runtime."))
	}
	if logosIcfCount(nmFolded, "runtime.") < 50 {
		t.Errorf("folded nm holds %d runtime symbols, want at least 50", logosIcfCount(nmFolded, "runtime."))
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
	cmd := testenv.Command(t, exe)
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

// logosIcfSize returns the file size of one linked binary.
func logosIcfSize(t *testing.T, exe string) int64 {
	t.Helper()
	fi, err := os.Stat(exe)
	if err != nil {
		t.Fatal(err)
	}
	return fi.Size()
}
