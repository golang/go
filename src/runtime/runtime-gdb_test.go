// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"fmt"
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

// NOTE: In some configurations, GDB will segfault when sent a SIGWINCH signal.
// Some runtime tests send SIGWINCH to the entire process group, so those tests
// must never run in parallel with GDB tests.
//
// See issue 39021 and https://sourceware.org/bugzilla/show_bug.cgi?id=26056.

func checkGdbEnvironment(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	switch runtime.GOOS {
	case "darwin":
		t.Skip("gdb does not work on darwin")
	case "netbsd":
		t.Skip("gdb does not work with threads on NetBSD; see https://golang.org/issue/22893 and https://gnats.netbsd.org/52548")
	case "windows":
		t.Skip("gdb tests fail on Windows: https://golang.org/issue/22687")
	case "linux":
		if runtime.GOARCH == "ppc64" {
			t.Skip("skipping gdb tests on linux/ppc64; see https://golang.org/issue/17366")
		}
		if runtime.GOARCH == "mips" {
			t.Skip("skipping gdb tests on linux/mips; see https://golang.org/issue/25939")
		}
	case "freebsd":
		t.Skip("skipping gdb tests on FreeBSD; see https://golang.org/issue/29508")
	case "aix":
		if testing.Short() {
			t.Skip("skipping gdb tests on AIX; see https://golang.org/issue/35710")
		}
	case "plan9":
		t.Skip("there is no gdb on Plan 9")
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
	if runtime.GOOS == "solaris" || runtime.GOOS == "illumos" {
		t.Skip("skipping gdb python tests on illumos and solaris; see golang.org/issue/20821")
	}

	cmd := exec.Command("gdb", "-nx", "-q", "--batch", "-iex", "python import sys; print('go gdb python support')")
	out, err := cmd.CombinedOutput()

	if err != nil {
		t.Skipf("skipping due to issue running gdb: %v", err)
	}
	if strings.TrimSpace(string(out)) != "go gdb python support" {
		t.Skipf("skipping due to lack of python gdb support: %s", out)
	}
}

// checkCleanBacktrace checks that the given backtrace is well formed and does
// not contain any error messages from GDB.
func checkCleanBacktrace(t *testing.T, backtrace string) {
	backtrace = strings.TrimSpace(backtrace)
	lines := strings.Split(backtrace, "\n")
	if len(lines) == 0 {
		t.Fatalf("empty backtrace")
	}
	for i, l := range lines {
		if !strings.HasPrefix(l, fmt.Sprintf("#%v  ", i)) {
			t.Fatalf("malformed backtrace at line %v: %v", i, l)
		}
	}
	// TODO(mundaym): check for unknown frames (e.g. "??").
}

const helloSource = `
import "fmt"
import "runtime"
var gslice []string
func main() {
	mapvar := make(map[string]string, 13)
	slicemap := make(map[string][]string,11)
    chanint := make(chan int, 10)
    chanstr := make(chan string, 10)
    chanint <- 99
	chanint <- 11
    chanstr <- "spongepants"
    chanstr <- "squarebob"
	mapvar["abc"] = "def"
	mapvar["ghi"] = "jkl"
	slicemap["a"] = []string{"b","c","d"}
    slicemap["e"] = []string{"f","g","h"}
	strvar := "abc"
	ptrvar := &strvar
	slicevar := make([]string, 0, 16)
	slicevar = append(slicevar, mapvar["abc"])
	fmt.Println("hi")
	runtime.KeepAlive(ptrvar)
	_ = ptrvar // set breakpoint here
	gslice = slicevar
	fmt.Printf("%v, %v, %v\n", slicemap, <-chanint, <-chanstr)
	runtime.KeepAlive(mapvar)
}  // END_OF_PROGRAM
`

func lastLine(src []byte) int {
	eop := []byte("END_OF_PROGRAM")
	for i, l := range bytes.Split(src, []byte("\n")) {
		if bytes.Contains(l, eop) {
			return i
		}
	}
	return 0
}

func TestGdbPython(t *testing.T) {
	testGdbPython(t, false)
}

func TestGdbPythonCgo(t *testing.T) {
	if runtime.GOARCH == "mips" || runtime.GOARCH == "mipsle" || runtime.GOARCH == "mips64" {
		testenv.SkipFlaky(t, 18784)
	}
	testGdbPython(t, true)
}

func testGdbPython(t *testing.T, cgo bool) {
	if cgo {
		testenv.MustHaveCGO(t)
	}

	checkGdbEnvironment(t)
	t.Parallel()
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

	src := buf.Bytes()

	// Locate breakpoint line
	var bp int
	lines := bytes.Split(src, []byte("\n"))
	for i, line := range lines {
		if bytes.Contains(line, []byte("breakpoint")) {
			bp = i
			break
		}
	}

	err = ioutil.WriteFile(filepath.Join(dir, "main.go"), src, 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	nLines := lastLine(src)

	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "a.exe", "main.go")
	cmd.Dir = dir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	args := []string{"-nx", "-q", "--batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(runtime.GOROOT(), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "set print thread-events off",
	}
	if cgo {
		// When we build the cgo version of the program, the system's
		// linker is used. Some external linkers, like GNU gold,
		// compress the .debug_gdb_scripts into .zdebug_gdb_scripts.
		// Until gold and gdb can work together, temporarily load the
		// python script directly.
		args = append(args,
			"-ex", "source "+filepath.Join(runtime.GOROOT(), "src", "runtime", "runtime-gdb.py"),
		)
	} else {
		args = append(args,
			"-ex", "info auto-load python-scripts",
		)
	}
	args = append(args,
		"-ex", "set python print-stack full",
		"-ex", fmt.Sprintf("br main.go:%d", bp),
		"-ex", "run",
		"-ex", "echo BEGIN info goroutines\n",
		"-ex", "info goroutines",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN print mapvar\n",
		"-ex", "print mapvar",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN print slicemap\n",
		"-ex", "print slicemap",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN print strvar\n",
		"-ex", "print strvar",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN print chanint\n",
		"-ex", "print chanint",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN print chanstr\n",
		"-ex", "print chanstr",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN info locals\n",
		"-ex", "info locals",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN goroutine 1 bt\n",
		"-ex", "goroutine 1 bt",
		"-ex", "echo END\n",
		"-ex", "echo BEGIN goroutine all bt\n",
		"-ex", "goroutine all bt",
		"-ex", "echo END\n",
		"-ex", "clear main.go:15", // clear the previous break point
		"-ex", fmt.Sprintf("br main.go:%d", nLines), // new break point at the end of main
		"-ex", "c",
		"-ex", "echo BEGIN goroutine 1 bt at the end\n",
		"-ex", "goroutine 1 bt",
		"-ex", "echo END\n",
		filepath.Join(dir, "a.exe"),
	)
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

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

	printMapvarRe1 := regexp.MustCompile(`^\$[0-9]+ = map\[string\]string = {\[(0x[0-9a-f]+\s+)?"abc"\] = (0x[0-9a-f]+\s+)?"def", \[(0x[0-9a-f]+\s+)?"ghi"\] = (0x[0-9a-f]+\s+)?"jkl"}$`)
	printMapvarRe2 := regexp.MustCompile(`^\$[0-9]+ = map\[string\]string = {\[(0x[0-9a-f]+\s+)?"ghi"\] = (0x[0-9a-f]+\s+)?"jkl", \[(0x[0-9a-f]+\s+)?"abc"\] = (0x[0-9a-f]+\s+)?"def"}$`)
	if bl := blocks["print mapvar"]; !printMapvarRe1.MatchString(bl) &&
		!printMapvarRe2.MatchString(bl) {
		t.Fatalf("print mapvar failed: %s", bl)
	}

	// 2 orders, and possible differences in spacing.
	sliceMapSfx1 := `map[string][]string = {["e"] = []string = {"f", "g", "h"}, ["a"] = []string = {"b", "c", "d"}}`
	sliceMapSfx2 := `map[string][]string = {["a"] = []string = {"b", "c", "d"}, ["e"] = []string = {"f", "g", "h"}}`
	if bl := strings.ReplaceAll(blocks["print slicemap"], "  ", " "); !strings.HasSuffix(bl, sliceMapSfx1) && !strings.HasSuffix(bl, sliceMapSfx2) {
		t.Fatalf("print slicemap failed: %s", bl)
	}

	chanIntSfx := `chan int = {99, 11}`
	if bl := strings.ReplaceAll(blocks["print chanint"], "  ", " "); !strings.HasSuffix(bl, chanIntSfx) {
		t.Fatalf("print chanint failed: %s", bl)
	}

	chanStrSfx := `chan string = {"spongepants", "squarebob"}`
	if bl := strings.ReplaceAll(blocks["print chanstr"], "  ", " "); !strings.HasSuffix(bl, chanStrSfx) {
		t.Fatalf("print chanstr failed: %s", bl)
	}

	strVarRe := regexp.MustCompile(`^\$[0-9]+ = (0x[0-9a-f]+\s+)?"abc"$`)
	if bl := blocks["print strvar"]; !strVarRe.MatchString(bl) {
		t.Fatalf("print strvar failed: %s", bl)
	}

	// The exact format of composite values has changed over time.
	// For issue 16338: ssa decompose phase split a slice into
	// a collection of scalar vars holding its fields. In such cases
	// the DWARF variable location expression should be of the
	// form "var.field" and not just "field".
	// However, the newer dwarf location list code reconstituted
	// aggregates from their fields and reverted their printing
	// back to its original form.
	// Only test that all variables are listed in 'info locals' since
	// different versions of gdb print variables in different
	// order and with differing amount of information and formats.

	if bl := blocks["info locals"]; !strings.Contains(bl, "slicevar") ||
		!strings.Contains(bl, "mapvar") ||
		!strings.Contains(bl, "strvar") {
		t.Fatalf("info locals failed: %s", bl)
	}

	// Check that the backtraces are well formed.
	checkCleanBacktrace(t, blocks["goroutine 1 bt"])
	checkCleanBacktrace(t, blocks["goroutine 1 bt at the end"])

	btGoroutine1Re := regexp.MustCompile(`(?m)^#0\s+(0x[0-9a-f]+\s+in\s+)?main\.main.+at`)
	if bl := blocks["goroutine 1 bt"]; !btGoroutine1Re.MatchString(bl) {
		t.Fatalf("goroutine 1 bt failed: %s", bl)
	}

	if bl := blocks["goroutine all bt"]; !btGoroutine1Re.MatchString(bl) {
		t.Fatalf("goroutine all bt failed: %s", bl)
	}

	btGoroutine1AtTheEndRe := regexp.MustCompile(`(?m)^#0\s+(0x[0-9a-f]+\s+in\s+)?main\.main.+at`)
	if bl := blocks["goroutine 1 bt at the end"]; !btGoroutine1AtTheEndRe.MatchString(bl) {
		t.Fatalf("goroutine 1 bt at the end failed: %s", bl)
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

	checkGdbEnvironment(t)
	t.Parallel()
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
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "a.exe", "main.go")
	cmd.Dir = dir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	// Execute gdb commands.
	args := []string{"-nx", "-batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(runtime.GOROOT(), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "break main.eee",
		"-ex", "run",
		"-ex", "backtrace",
		"-ex", "continue",
		filepath.Join(dir, "a.exe"),
	}
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

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
			t.Fatalf("could not find '%v' in backtrace", s)
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
	checkGdbEnvironment(t)
	t.Parallel()
	checkGdbVersion(t)

	if runtime.GOOS == "aix" && testing.Short() {
		t.Skip("TestGdbAutotmpTypes is too slow on aix/ppc64")
	}

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
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-gcflags=all=-N -l", "-o", "a.exe", "main.go")
	cmd.Dir = dir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	// Execute gdb commands.
	args := []string{"-nx", "-batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(runtime.GOROOT(), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "break main.main",
		"-ex", "run",
		"-ex", "step",
		"-ex", "info types astruct",
		filepath.Join(dir, "a.exe"),
	}
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	sgot := string(got)

	// Check that the backtrace matches the source code.
	types := []string{
		"[]main.astruct;",
		"bucket<string,main.astruct>;",
		"hash<string,main.astruct>;",
		"main.astruct;",
		"hash<string,main.astruct> * map[string]main.astruct;",
	}
	for _, name := range types {
		if !strings.Contains(sgot, name) {
			t.Fatalf("could not find %s in 'info typrs astruct' output", name)
		}
	}
}

const constsSource = `
package main

const aConstant int = 42
const largeConstant uint64 = ^uint64(0)
const minusOne int64 = -1

func main() {
	println("hello world")
}
`

func TestGdbConst(t *testing.T) {
	checkGdbEnvironment(t)
	t.Parallel()
	checkGdbVersion(t)

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err = ioutil.WriteFile(src, []byte(constsSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-gcflags=all=-N -l", "-o", "a.exe", "main.go")
	cmd.Dir = dir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	// Execute gdb commands.
	args := []string{"-nx", "-batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(runtime.GOROOT(), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "break main.main",
		"-ex", "run",
		"-ex", "print main.aConstant",
		"-ex", "print main.largeConstant",
		"-ex", "print main.minusOne",
		"-ex", "print 'runtime.mSpanInUse'",
		"-ex", "print 'runtime._PageSize'",
		filepath.Join(dir, "a.exe"),
	}
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	sgot := strings.ReplaceAll(string(got), "\r\n", "\n")

	if !strings.Contains(sgot, "\n$1 = 42\n$2 = 18446744073709551615\n$3 = -1\n$4 = 1 '\\001'\n$5 = 8192") {
		t.Fatalf("output mismatch")
	}
}

const panicSource = `
package main

import "runtime/debug"

func main() {
	debug.SetTraceback("crash")
	crash()
}

func crash() {
	panic("panic!")
}
`

// TestGdbPanic tests that gdb can unwind the stack correctly
// from SIGABRTs from Go panics.
func TestGdbPanic(t *testing.T) {
	checkGdbEnvironment(t)
	t.Parallel()
	checkGdbVersion(t)

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err = ioutil.WriteFile(src, []byte(panicSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "a.exe", "main.go")
	cmd.Dir = dir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	// Execute gdb commands.
	args := []string{"-nx", "-batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(runtime.GOROOT(), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "run",
		"-ex", "backtrace",
		filepath.Join(dir, "a.exe"),
	}
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	// Check that the backtrace matches the source code.
	bt := []string{
		`crash`,
		`main`,
	}
	for _, name := range bt {
		s := fmt.Sprintf("(#.* .* in )?main\\.%v", name)
		re := regexp.MustCompile(s)
		if found := re.Find(got) != nil; !found {
			t.Fatalf("could not find '%v' in backtrace", s)
		}
	}
}

const InfCallstackSource = `
package main
import "C"
import "time"

func loop() {
        for i := 0; i < 1000; i++ {
                time.Sleep(time.Millisecond*5)
        }
}

func main() {
        go loop()
        time.Sleep(time.Second * 1)
}
`

// TestGdbInfCallstack tests that gdb can unwind the callstack of cgo programs
// on arm64 platforms without endless frames of function 'crossfunc1'.
// https://golang.org/issue/37238
func TestGdbInfCallstack(t *testing.T) {
	checkGdbEnvironment(t)

	testenv.MustHaveCGO(t)
	if runtime.GOARCH != "arm64" {
		t.Skip("skipping infinite callstack test on non-arm64 arches")
	}

	t.Parallel()
	checkGdbVersion(t)

	dir, err := ioutil.TempDir("", "go-build")
	if err != nil {
		t.Fatalf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err = ioutil.WriteFile(src, []byte(InfCallstackSource), 0644)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "a.exe", "main.go")
	cmd.Dir = dir
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("building source %v\n%s", err, out)
	}

	// Execute gdb commands.
	// 'setg_gcc' is the first point where we can reproduce the issue with just one 'run' command.
	args := []string{"-nx", "-batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(runtime.GOROOT(), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "break setg_gcc",
		"-ex", "run",
		"-ex", "backtrace 3",
		"-ex", "disable 1",
		"-ex", "continue",
		filepath.Join(dir, "a.exe"),
	}
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	// Check that the backtrace matches
	// We check the 3 inner most frames only as they are present certainly, according to gcc_<OS>_arm64.c
	bt := []string{
		`setg_gcc`,
		`crosscall1`,
		`threadentry`,
	}
	for i, name := range bt {
		s := fmt.Sprintf("#%v.*%v", i, name)
		re := regexp.MustCompile(s)
		if found := re.Find(got) != nil; !found {
			t.Fatalf("could not find '%v' in backtrace", s)
		}
	}
}
