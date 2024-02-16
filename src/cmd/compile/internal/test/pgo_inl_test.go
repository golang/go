// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/profile"
	"internal/testenv"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

const profFile = "inline_hot.pprof"
const preProfFile = "inline_hot.pprof.node_map"

func buildPGOInliningTest(t *testing.T, dir string, gcflag string) []byte {
	const pkg = "example.com/pgo/inline"

	// Add a go.mod so we have a consistent symbol names in this temp dir.
	goMod := fmt.Sprintf(`module %s
go 1.19
`, pkg)
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatalf("error writing go.mod: %v", err)
	}

	exe := filepath.Join(dir, "test.exe")
	args := []string{"test", "-c", "-o", exe, "-gcflags=" + gcflag}
	cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
	cmd.Dir = dir
	cmd = testenv.CleanCmdEnv(cmd)
	t.Log(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build failed: %v, output:\n%s", err, out)
	}
	return out
}

// testPGOIntendedInlining tests that specific functions are inlined.
func testPGOIntendedInlining(t *testing.T, dir string, profFile string) {
	testenv.MustHaveGoRun(t)
	t.Parallel()

	const pkg = "example.com/pgo/inline"

	want := []string{
		"(*BS).NS",
	}

	// The functions which are not expected to be inlined are as follows.
	wantNot := []string{
		// The calling edge main->A is hot and the cost of A is large
		// than inlineHotCalleeMaxBudget.
		"A",
		// The calling edge BenchmarkA" -> benchmarkB is cold and the
		// cost of A is large than inlineMaxBudget.
		"benchmarkB",
	}

	must := map[string]bool{
		"(*BS).NS": true,
	}

	notInlinedReason := make(map[string]string)
	for _, fname := range want {
		fullName := pkg + "." + fname
		if _, ok := notInlinedReason[fullName]; ok {
			t.Errorf("duplicate func: %s", fullName)
		}
		notInlinedReason[fullName] = "unknown reason"
	}

	// If the compiler emit "cannot inline for function A", the entry A
	// in expectedNotInlinedList will be removed.
	expectedNotInlinedList := make(map[string]struct{})
	for _, fname := range wantNot {
		fullName := pkg + "." + fname
		expectedNotInlinedList[fullName] = struct{}{}
	}

	// Build the test with the profile. Use a smaller threshold to test.
	// TODO: maybe adjust the test to work with default threshold.
	gcflag := fmt.Sprintf("-m -m -pgoprofile=%s -d=pgoinlinebudget=160,pgoinlinecdfthreshold=90", profFile)
	out := buildPGOInliningTest(t, dir, gcflag)

	scanner := bufio.NewScanner(bytes.NewReader(out))
	curPkg := ""
	canInline := regexp.MustCompile(`: can inline ([^ ]*)`)
	haveInlined := regexp.MustCompile(`: inlining call to ([^ ]*)`)
	cannotInline := regexp.MustCompile(`: cannot inline ([^ ]*): (.*)`)
	for scanner.Scan() {
		line := scanner.Text()
		t.Logf("child: %s", line)
		if strings.HasPrefix(line, "# ") {
			curPkg = line[2:]
			splits := strings.Split(curPkg, " ")
			curPkg = splits[0]
			continue
		}
		if m := haveInlined.FindStringSubmatch(line); m != nil {
			fname := m[1]
			delete(notInlinedReason, curPkg+"."+fname)
			continue
		}
		if m := canInline.FindStringSubmatch(line); m != nil {
			fname := m[1]
			fullname := curPkg + "." + fname
			// If function must be inlined somewhere, being inlinable is not enough
			if _, ok := must[fullname]; !ok {
				delete(notInlinedReason, fullname)
				continue
			}
		}
		if m := cannotInline.FindStringSubmatch(line); m != nil {
			fname, reason := m[1], m[2]
			fullName := curPkg + "." + fname
			if _, ok := notInlinedReason[fullName]; ok {
				// cmd/compile gave us a reason why
				notInlinedReason[fullName] = reason
			}
			delete(expectedNotInlinedList, fullName)
			continue
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("error reading output: %v", err)
	}
	for fullName, reason := range notInlinedReason {
		t.Errorf("%s was not inlined: %s", fullName, reason)
	}

	// If the list expectedNotInlinedList is not empty, it indicates
	// the functions in the expectedNotInlinedList are marked with caninline.
	for fullName, _ := range expectedNotInlinedList {
		t.Errorf("%s was expected not inlined", fullName)
	}
}

// TestPGOIntendedInlining tests that specific functions are inlined when PGO
// is applied to the exact source that was profiled.
func TestPGOIntendedInlining(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("error getting wd: %v", err)
	}
	srcDir := filepath.Join(wd, "testdata/pgo/inline")

	// Copy the module to a scratch location so we can add a go.mod.
	dir := t.TempDir()

	for _, file := range []string{"inline_hot.go", "inline_hot_test.go", profFile} {
		if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
			t.Fatalf("error copying %s: %v", file, err)
		}
	}

	testPGOIntendedInlining(t, dir, profFile)
}

// TestPGOIntendedInlining tests that specific functions are inlined when PGO
// is applied to the exact source that was profiled.
func TestPGOPreprocessInlining(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("error getting wd: %v", err)
	}
	srcDir := filepath.Join(wd, "testdata/pgo/inline")

	// Copy the module to a scratch location so we can add a go.mod.
	dir := t.TempDir()

	for _, file := range []string{"inline_hot.go", "inline_hot_test.go", preProfFile} {
		if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
			t.Fatalf("error copying %s: %v", file, err)
		}
	}

	testPGOIntendedInlining(t, dir, preProfFile)
}

// TestPGOIntendedInlining tests that specific functions are inlined when PGO
// is applied to the modified source.
func TestPGOIntendedInliningShiftedLines(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("error getting wd: %v", err)
	}
	srcDir := filepath.Join(wd, "testdata/pgo/inline")

	// Copy the module to a scratch location so we can modify the source.
	dir := t.TempDir()

	// Copy most of the files unmodified.
	for _, file := range []string{"inline_hot_test.go", profFile} {
		if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
			t.Fatalf("error copying %s : %v", file, err)
		}
	}

	// Add some comments to the top of inline_hot.go. This adjusts the line
	// numbers of all of the functions without changing the semantics.
	src, err := os.Open(filepath.Join(srcDir, "inline_hot.go"))
	if err != nil {
		t.Fatalf("error opening src inline_hot.go: %v", err)
	}
	defer src.Close()

	dst, err := os.Create(filepath.Join(dir, "inline_hot.go"))
	if err != nil {
		t.Fatalf("error creating dst inline_hot.go: %v", err)
	}
	defer dst.Close()

	if _, err := io.WriteString(dst, `// Autogenerated
// Lines
`); err != nil {
		t.Fatalf("error writing comments to dst: %v", err)
	}

	if _, err := io.Copy(dst, src); err != nil {
		t.Fatalf("error copying inline_hot.go: %v", err)
	}

	dst.Close()

	testPGOIntendedInlining(t, dir, profFile)
}

// TestPGOSingleIndex tests that the sample index can not be 1 and compilation
// will not fail. All it should care about is that the sample type is either
// CPU nanoseconds or samples count, whichever it finds first.
func TestPGOSingleIndex(t *testing.T) {
	for _, tc := range []struct {
		originalIndex int
	}{{
		// The `testdata/pgo/inline/inline_hot.pprof` file is a standard CPU
		// profile as the runtime would generate. The 0 index contains the
		// value-type samples and value-unit count. The 1 index contains the
		// value-type cpu and value-unit nanoseconds. These tests ensure that
		// the compiler can work with profiles that only have a single index,
		// but are either samples count or CPU nanoseconds.
		originalIndex: 0,
	}, {
		originalIndex: 1,
	}} {
		t.Run(fmt.Sprintf("originalIndex=%d", tc.originalIndex), func(t *testing.T) {
			wd, err := os.Getwd()
			if err != nil {
				t.Fatalf("error getting wd: %v", err)
			}
			srcDir := filepath.Join(wd, "testdata/pgo/inline")

			// Copy the module to a scratch location so we can add a go.mod.
			dir := t.TempDir()

			originalPprofFile, err := os.Open(filepath.Join(srcDir, profFile))
			if err != nil {
				t.Fatalf("error opening %v: %v", profFile, err)
			}
			defer originalPprofFile.Close()

			p, err := profile.Parse(originalPprofFile)
			if err != nil {
				t.Fatalf("error parsing %v: %v", profFile, err)
			}

			// Move the samples count value-type to the 0 index.
			p.SampleType = []*profile.ValueType{p.SampleType[tc.originalIndex]}

			// Ensure we only have a single set of sample values.
			for _, s := range p.Sample {
				s.Value = []int64{s.Value[tc.originalIndex]}
			}

			modifiedPprofFile, err := os.Create(filepath.Join(dir, profFile))
			if err != nil {
				t.Fatalf("error creating %v: %v", profFile, err)
			}
			defer modifiedPprofFile.Close()

			if err := p.Write(modifiedPprofFile); err != nil {
				t.Fatalf("error writing %v: %v", profFile, err)
			}

			for _, file := range []string{"inline_hot.go", "inline_hot_test.go"} {
				if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
					t.Fatalf("error copying %s: %v", file, err)
				}
			}

			testPGOIntendedInlining(t, dir, profFile)
		})
	}
}

func copyFile(dst, src string) error {
	s, err := os.Open(src)
	if err != nil {
		return err
	}
	defer s.Close()

	d, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer d.Close()

	_, err = io.Copy(d, s)
	return err
}

// TestPGOHash tests that PGO optimization decisions can be selected by pgohash.
func TestPGOHash(t *testing.T) {
	testenv.MustHaveGoRun(t)
	t.Parallel()

	const pkg = "example.com/pgo/inline"

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("error getting wd: %v", err)
	}
	srcDir := filepath.Join(wd, "testdata/pgo/inline")

	// Copy the module to a scratch location so we can add a go.mod.
	dir := t.TempDir()

	for _, file := range []string{"inline_hot.go", "inline_hot_test.go", profFile} {
		if err := copyFile(filepath.Join(dir, file), filepath.Join(srcDir, file)); err != nil {
			t.Fatalf("error copying %s: %v", file, err)
		}
	}

	pprof := filepath.Join(dir, profFile)
	// build with -trimpath so the source location (thus the hash)
	// does not depend on the temporary directory path.
	gcflag0 := fmt.Sprintf("-pgoprofile=%s -trimpath %s=>%s -d=pgoinlinebudget=160,pgoinlinecdfthreshold=90,pgodebug=1", pprof, dir, pkg)

	// Check that a hash match allows PGO inlining.
	const srcPos = "example.com/pgo/inline/inline_hot.go:81:19"
	const hashMatch = "pgohash triggered " + srcPos + " (inline)"
	pgoDebugRE := regexp.MustCompile(`hot-budget check allows inlining for call .* at ` + strings.ReplaceAll(srcPos, ".", "\\."))
	hash := "v1" // 1 matches srcPos, v for verbose (print source location)
	gcflag := gcflag0 + ",pgohash=" + hash
	out := buildPGOInliningTest(t, dir, gcflag)
	if !bytes.Contains(out, []byte(hashMatch)) || !pgoDebugRE.Match(out) {
		t.Errorf("output does not contain expected source line, out:\n%s", out)
	}

	// Check that a hash mismatch turns off PGO inlining.
	hash = "v0" // 0 should not match srcPos
	gcflag = gcflag0 + ",pgohash=" + hash
	out = buildPGOInliningTest(t, dir, gcflag)
	if bytes.Contains(out, []byte(hashMatch)) || pgoDebugRE.Match(out) {
		t.Errorf("output contains unexpected source line, out:\n%s", out)
	}
}
