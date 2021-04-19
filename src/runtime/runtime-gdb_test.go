// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"fmt"
	"go/build"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func checkGdbEnvironment(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if runtime.GOOS == "darwin" {
		t.Skip("gdb does not work on darwin")
	}
	if runtime.GOOS == "linux" && runtime.GOARCH == "ppc64" {
		t.Skip("skipping gdb tests on linux/ppc64; see golang.org/issue/17366")
	}
	if final := os.Getenv("GOROOT_FINAL"); final != "" && runtime.GOROOT() != final {
		t.Skip("gdb test can fail with GOROOT_FINAL pending")
	}
}

func checkGdbVersion(t *testing.T) {
	// Issue 11214 reports various failures with older versions of gdb.
	out, err := exec.Command("gdb", "--version").CombinedOutput()
	if err != nil {
		t.Skipf("skipping: error executing gdb: %v", err)
	}
	re := regexp.MustCompile(`([0-9]+)\.([0-9]+)`)
	matches := re.FindSubmatch(out)
	if len(matches) < 3 {
		t.Skipf("skipping: can't determine gdb version from\n%s\n", out)
	}
	major, err1 := strconv.Atoi(string(matches[1]))
	minor, err2 := strconv.Atoi(string(matches[2]))
	if err1 != nil || err2 != nil {
		t.Skipf("skipping: can't determine gdb version: %v, %v", err1, err2)
	}
	if major < 7 || (major == 7 && minor < 7) {
		t.Skipf("skipping: gdb version %d.%d too old", major, minor)
	}
	t.Logf("gdb version %d.%d", major, minor)
}

func checkGdbPython(t *testing.T) {
	cmd := exec.Command("gdb", "-nx", "-q", "--batch", "-iex", "python import sys; print('go gdb python support')")
	out, err := cmd.CombinedOutput()

	if err != nil {
		t.Skipf("skipping due to issue running gdb: %v", err)
	}
	if string(out) != "go gdb python support\n" {
		t.Skipf("skipping due to lack of python gdb support: %s", out)
	}
}

const helloSource = `
import "fmt"
var gslice []string
func main() {
	mapvar := make(map[string]string,5)
	mapvar["abc"] = "def"
	mapvar["ghi"] = "jkl"
	strvar := "abc"
	ptrvar := &strvar
	slicevar := make([]string, 0, 16)
	slicevar = append(slicevar, mapvar["abc"])
	fmt.Println("hi") // line 12
	_ = ptrvar
	gslice = slicevar
}
`

func TestGdbPython(t *testing.T) {
	testGdbPython(t, false)
}

func TestGdbPythonCgo(t *testing.T) {
	testGdbPython(t, true)
}

func testGdbPython(t *testing.T, cgo bool) {
	if runtime.GOARCH == "mips64" {
		testenv.SkipFlaky(t, 18173)
	}
	if cgo && !build.Default.CgoEnabled {
		t.Skip("skipping because cgo is not enabled")
	}

	t.Parallel()
	checkGdbEnvironment(t)
	checkGdbVersion(t)
	checkGdbPython(t)

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	var buf bytes.Buffer
	buf.WriteString("package main\n")
	if cgo {
		buf.WriteString(`import "C"` + "\n")
	}
	buf.WriteString(helloSource)

	src := filepath.Join(dir, "main.go")
	err = ioutil.WriteFile(src, buf.Bytes(), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "a.exe")
	cmd.Dir = dir
	out, err := testEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	args := []string{"-nx", "-q", "--batch", "-iex",
		fmt.Sprintf("add-auto-load-safe-path %s/src/runtime", runtime.GOROOT()),
		"-ex", "set startup-with-shell off",
		"-ex", "info auto-load python-scripts",
		"-ex", "set python print-stack full",
		"-ex", "br fmt.Println",
		"-ex", "run",
		"-ex", "echo BEGIN info goroutines\n",
		"-ex", "info goroutines",
		"-ex", "echo END\n",
		"-ex", "up", // up from fmt.Println to main
		"-ex", "echo BEGIN print mapvar\n",
		"-ex", "print mapvar",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN print strvar\n",
		"-ex", "print strvar",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN info locals\n",
		"-ex", "info locals",
		"-ex", "echo END\n",
		"-ex", "down", // back to fmt.Println (goroutine 2 below only works at bottom of stack.  TODO: fix that)
		"-ex", "echo BEGIN goroutine 2 bt\n",
		"-ex", "goroutine 2 bt",
		"-ex", "echo END\n",
		filepath.Join(dir, "a.exe"),
	}
	got, _ := exec.Command("gdb", args...).CombinedOutput()

	firstLine := bytes.SplitN(got, []byte("\n"), 2)[0]
	if string(firstLine) != "Loading Go Runtime support." {
		// This can happen when using all.bash with
		// GOROOT_FINAL set, because the tests are run before
		// the final installation of the files.
		cmd := exec.Command(testenv.GoToolPath(t), "env", "GOROOT")
		cmd.Env = []string{}
		out, err := cmd.CombinedOutput()
		if err != nil && bytes.Contains(out, []byte("cannot find GOROOT")) {
			t.Skipf("skipping because GOROOT=%s does not exist", runtime.GOROOT())
		}

		_, file, _, _ := runtime.Caller(1)

		t.Logf("package testing source file: %s", file)
		t.Fatalf("failed to load Go runtime support: %s\n%s", firstLine, got)
	}

	// Extract named BEGIN...END blocks from output
	partRe := regexp.MustCompile(`(?ms)^BEGIN ([^\n]*)\n(.*?)\nEND`)
	blocks := map[string]string{}
	for _, subs := range partRe.FindAllSubmatch(got, -1) {
		blocks[string(subs[1])] = string(subs[2])
	}

	infoGoroutinesRe := regexp.MustCompile(`\*\s+\d+\s+running\s+`)
	if bl := blocks["info goroutines"]; !infoGoroutinesRe.MatchString(bl) {
		t.Fatalf("info goroutines failed: %s", bl)
	}

	printMapvarRe := regexp.MustCompile(`\Q = map[string]string = {["abc"] = "def", ["ghi"] = "jkl"}\E$`)
	if bl := blocks["print mapvar"]; !printMapvarRe.MatchString(bl) {
		t.Fatalf("print mapvar failed: %s", bl)
	}

	strVarRe := regexp.MustCompile(`\Q = "abc"\E$`)
	if bl := blocks["print strvar"]; !strVarRe.MatchString(bl) {
		t.Fatalf("print strvar failed: %s", bl)
	}

	// Issue 16338: ssa decompose phase can split a structure into
	// a collection of scalar vars holding the fields. In such cases
	// the DWARF variable location expression should be of the
	// form "var.field" and not just "field".
	infoLocalsRe := regexp.MustCompile(`^slicevar.len = `)
	if bl := blocks["info locals"]; !infoLocalsRe.MatchString(bl) {
		t.Fatalf("info locals failed: %s", bl)
	}

	btGoroutineRe := regexp.MustCompile(`^#0\s+runtime.+at`)
	if bl := blocks["goroutine 2 bt"]; !btGoroutineRe.MatchString(bl) {
		t.Fatalf("goroutine 2 bt failed: %s", bl)
	}
}

const backtraceSource = `
package main

//go:noinline
func aaa() bool { return bbb() }

//go:noinline
func bbb() bool { return ccc() }

//go:noinline
func ccc() bool { return ddd() }

//go:noinline
func ddd() bool { return f() }

//go:noinline
func eee() bool { return true }

var f = eee

func main() {
	_ = aaa()
}
`

// TestGdbBacktrace tests that gdb can unwind the stack correctly
// using only the DWARF debug info.
func TestGdbBacktrace(t *testing.T) {
	if runtime.GOOS == "netbsd" {
		testenv.SkipFlaky(t, 15603)
	}
	if runtime.GOARCH == "mips64" {
		testenv.SkipFlaky(t, 18173)
	}

	t.Parallel()
	checkGdbEnvironment(t)
	checkGdbVersion(t)

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err = ioutil.WriteFile(src, []byte(backtraceSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "a.exe")
	cmd.Dir = dir
	out, err := testEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	// Execute gdb commands.
	args := []string{"-nx", "-batch",
		"-ex", "set startup-with-shell off",
		"-ex", "break main.eee",
		"-ex", "run",
		"-ex", "backtrace",
		"-ex", "continue",
		filepath.Join(dir, "a.exe"),
	}
	got, _ := exec.Command("gdb", args...).CombinedOutput()

	// Check that the backtrace matches the source code.
	bt := []string{
		"eee",
		"ddd",
		"ccc",
		"bbb",
		"aaa",
		"main",
	}
	for i, name := range bt {
		s := fmt.Sprintf("#%v.*main\\.%v", i, name)
		re := regexp.MustCompile(s)
		if found := re.Find(got) != nil; !found {
			t.Errorf("could not find '%v' in backtrace", s)
			t.Fatalf("gdb output:\n%v", string(got))
		}
	}
}

const autotmpTypeSource = `
package main

type astruct struct {
	a, b int
}

func main() {
	var iface interface{} = map[string]astruct{}
	var iface2 interface{} = []astruct{}
	println(iface, iface2)
}
`

// TestGdbAutotmpTypes ensures that types of autotmp variables appear in .debug_info
// See bug #17830.
func TestGdbAutotmpTypes(t *testing.T) {
	if runtime.GOARCH == "mips64" {
		testenv.SkipFlaky(t, 18173)
	}

	t.Parallel()
	checkGdbEnvironment(t)
	checkGdbVersion(t)

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err = ioutil.WriteFile(src, []byte(autotmpTypeSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-gcflags=-N -l", "-o", "a.exe")
	cmd.Dir = dir
	out, err := testEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	// Execute gdb commands.
	args := []string{"-nx", "-batch",
		"-ex", "set startup-with-shell off",
		"-ex", "break main.main",
		"-ex", "run",
		"-ex", "step",
		"-ex", "info types astruct",
		filepath.Join(dir, "a.exe"),
	}
	got, _ := exec.Command("gdb", args...).CombinedOutput()

	sgot := string(got)

	// Check that the backtrace matches the source code.
	types := []string{
		"struct []main.astruct;",
		"struct bucket<string,main.astruct>;",
		"struct hash<string,main.astruct>;",
		"struct main.astruct;",
		"typedef struct hash<string,main.astruct> * map[string]main.astruct;",
	}
	for _, name := range types {
		if !strings.Contains(sgot, name) {
			t.Errorf("could not find %s in 'info typrs astruct' output", name)
			t.Fatalf("gdb output:\n%v", sgot)
		}
	}
}
