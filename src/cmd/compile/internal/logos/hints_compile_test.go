// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"encoding/json"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// hintsRun is the outcome of one compile in hints mode.
type hintsRun struct {
	report *HintsReport
	output string
	exit   int
	object bool
}

// importConfig writes an import config for logos/rt and its dependencies,
// which every loGo/s option lowering imports, and returns its path.
func importConfig(t *testing.T, dir string) string {
	t.Helper()
	cmd := exec.Command(testenv.GoToolPath(t), "list", "-export", "-deps",
		"-f", "{{if .Export}}packagefile {{.ImportPath}}={{.Export}}{{end}}", "logos/rt")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go list -export logos/rt: %v", err)
	}
	path := filepath.Join(dir, "importcfg")
	if err := os.WriteFile(path, out, 0o666); err != nil {
		t.Fatal(err)
	}
	return path
}

// runHints compiles the named fixtures of testdata/hints as package p with
// -logos.mode=hints and reads the report the compiler wrote, when it wrote one.
func runHints(t *testing.T, flags []string, names ...string) hintsRun {
	t.Helper()
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	hints := filepath.Join(dir, "hints.json")
	object := filepath.Join(dir, "out.a")
	args := []string{"tool", "compile", "-p", "p", "-importcfg", importConfig(t, dir), "-logos.hintsout=" + hints, "-o", object}
	args = append(args, flags...)
	for _, name := range names {
		abs, err := filepath.Abs(filepath.Join("testdata", "hints", name))
		if err != nil {
			t.Fatal(err)
		}
		args = append(args, abs)
	}
	out, err := exec.Command(testenv.GoToolPath(t), args...).CombinedOutput()
	run := hintsRun{output: string(out)}
	if ee, ok := err.(*exec.ExitError); ok {
		run.exit = ee.ExitCode()
	} else if err != nil {
		t.Fatal(err)
	}
	_, statErr := os.Stat(object)
	run.object = statErr == nil
	data, readErr := os.ReadFile(hints)
	if readErr != nil {
		return run
	}
	run.report = new(HintsReport)
	if err := json.Unmarshal(data, run.report); err != nil {
		t.Fatalf("hints file is not JSON: %v\n%s", err, data)
	}
	return run
}

// allocAt returns the allocation hint at pos, or false.
func allocAt(allocs []AllocHint, pos HintPos) (AllocHint, bool) {
	for _, a := range allocs {
		if a.Position == pos {
			return a, true
		}
	}
	return AllocHint{}, false
}

// TestCompileHintsGoPackage checks the escape decisions of a plain Go
// package: the same -m decisions as the compiler prints, as tiers at the
// positions -m prints (SPEC Sec 21.2: badges appear for Go packages too).
func TestCompileHintsGoPackage(t *testing.T) {
	run := runHints(t, []string{"-logos.mode=hints"}, "alloc.go")
	if run.exit != 0 || run.report == nil {
		t.Fatalf("hints compile: exit %d, report %v, output %q", run.exit, run.report, run.output)
	}
	src, err := os.ReadFile(filepath.Join("testdata", "hints", "alloc.go"))
	if err != nil {
		t.Fatal(err)
	}
	if run.report.Version != HintsVersion || run.report.Package != "p" {
		t.Errorf("report header %q %q", run.report.Version, run.report.Package)
	}
	allocs := onlyFile(t, run.report).Allocs
	makeParen := positionAt(src, strings.Index(string(src), "make([]byte, n)")+len("make"))
	for _, c := range []struct {
		pos  HintPos
		tier string
		text string
	}{
		{startOf(t, src, "&T{1, 2}"), TierHeap, "escapes to heap"},
		{startOf(t, src, "&T{3, 4}"), TierStack, "does not escape"},
		{makeParen, TierHeap, "escapes to heap"},
		{startOf(t, src, "x := 5"), TierHeap, "moved to heap: x"},
	} {
		a, ok := allocAt(allocs, c.pos)
		if !ok || a.Tier != c.tier || !strings.Contains(a.Message, c.text) {
			t.Errorf("allocation at %+v: got %+v (found %v), want tier %s and message containing %q", c.pos, a, ok, c.tier, c.text)
		}
	}
}

// TestCompileHintsLgoPackage checks one compile of a loGo/s package: the
// checker facts, the lowering sites, the rodata placement, and the escape
// tiers all land in one report keyed by the original file.
func TestCompileHintsLgoPackage(t *testing.T) {
	run := runHints(t, []string{"-logos.mode=hints"}, "sites.lgo")
	if run.exit != 0 || run.report == nil {
		t.Fatalf("hints compile: exit %d, report %v, output %q", run.exit, run.report, run.output)
	}
	f := onlyFile(t, run.report)
	if filepath.Base(f.File) != "sites.lgo" {
		t.Errorf("file %q", f.File)
	}
	src, err := os.ReadFile(filepath.Join("testdata", "hints", "sites.lgo"))
	if err != nil {
		t.Fatal(err)
	}
	if len(f.Narrowed) != 2 || len(f.Unions) != 1 {
		t.Errorf("narrowed %+v, unions %+v", f.Narrowed, f.Unions)
	}
	kinds := map[string]int{}
	for _, s := range f.Sites {
		kinds[s.Kind]++
	}
	for _, k := range []string{SiteTry, SiteInterp, SiteConstSwitch} {
		if kinds[k] != 1 {
			t.Errorf("%d %s sites, want 1", kinds[k], k)
		}
	}
	for _, c := range []struct {
		needle string
		tier   string
	}{
		{"&User{Name: \"ada\"}", TierHeap},
		{"&User{Name: \"kept\"}", TierHeap},
		{"&User{Name: \"tmp\"}", TierStack},
		{"primes", TierRodata},
	} {
		a, ok := allocAt(f.Allocs, startOf(t, src, c.needle))
		if !ok || a.Tier != c.tier {
			t.Errorf("allocation at %s: got %+v (found %v), want tier %s", c.needle, a, ok, c.tier)
		}
	}
}

// TestCompileHintsOneAllocPerPosition checks that the compiler's own report
// holds at most one allocation hint per source position, so an editor shows
// one badge there: lowering stacks an append and a make on one interpolation.
func TestCompileHintsOneAllocPerPosition(t *testing.T) {
	run := runHints(t, []string{"-logos.mode=hints"}, "sites.lgo")
	if run.report == nil {
		t.Fatalf("no report: exit %d, output %q", run.exit, run.output)
	}
	seen := map[HintPos]bool{}
	for _, a := range onlyFile(t, run.report).Allocs {
		if seen[a.Position] {
			t.Errorf("two allocation hints at %+v", a.Position)
		}
		seen[a.Position] = true
	}
}

// TestCompileHintsWritesNoObject checks that hints mode ends after escape
// analysis, like -logos.lintonly: no object file, exit status zero.
func TestCompileHintsWritesNoObject(t *testing.T) {
	run := runHints(t, []string{"-logos.mode=hints"}, "alloc.go")
	if run.exit != 0 || run.object {
		t.Errorf("hints compile: exit %d, object written %v, output %q", run.exit, run.object, run.output)
	}
}

// TestCompileHintsNeedsOutputFile checks that hints mode without a report
// path is a build error and not a silent compile.
func TestCompileHintsNeedsOutputFile(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	abs, err := filepath.Abs(filepath.Join("testdata", "hints", "alloc.go"))
	if err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(testenv.GoToolPath(t), "tool", "compile", "-p", "p", "-logos.mode=hints", "-o", filepath.Join(dir, "o.a"), abs).CombinedOutput()
	if err == nil || !strings.Contains(string(out), "-logos.mode=hints needs -logos.hintsout=FILE") {
		t.Errorf("hints without output: err %v, output %q", err, out)
	}
}

// TestCompileHintsWritesNothingOnError checks that a package with a type
// error produces no report: the diagnostics speak, and half a report lies.
func TestCompileHintsWritesNothingOnError(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()
	src := filepath.Join(dir, "bad.go")
	hints := filepath.Join(dir, "hints.json")
	if err := os.WriteFile(src, []byte("package p\n\nfunc f() { undefined() }\n"), 0o666); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(testenv.GoToolPath(t), "tool", "compile", "-p", "p", "-logos.mode=hints", "-logos.hintsout="+hints, "-o", filepath.Join(dir, "o.a"), src).CombinedOutput()
	if err == nil || !strings.Contains(string(out), "undefined: undefined") {
		t.Errorf("hints on a type error: err %v, output %q", err, out)
	}
	if _, statErr := os.Stat(hints); statErr == nil {
		t.Error("a report was written for a package that does not type-check")
	}
}
