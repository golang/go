// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"flag"
	"fmt"
	"internal/abi"
	"internal/goexperiment"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
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
	case "linux":
		if runtime.GOARCH == "ppc64" {
			t.Skip("skipping gdb tests on linux/ppc64; see https://golang.org/issue/17366")
		}
		if runtime.GOARCH == "mips" {
			t.Skip("skipping gdb tests on linux/mips; see https://golang.org/issue/25939")
		}
		// Disable GDB tests on alpine until issue #54352 resolved.
		if strings.HasSuffix(testenv.Builder(), "-alpine") {
			t.Skip("skipping gdb tests on alpine; see https://golang.org/issue/54352")
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
	// The Go toolchain now generates DWARF 5 by default, which needs
	// a GDB version of 10 or above.
	if major < 10 {
		t.Skipf("skipping: gdb version %d.%d too old", major, minor)
	}
	t.Logf("gdb version %d.%d", major, minor)
}

func checkGdbPython(t *testing.T) {
	if runtime.GOOS == "solaris" || runtime.GOOS == "illumos" {
		t.Skip("skipping gdb python tests on illumos and solaris; see golang.org/issue/20821")
	}
	args := []string{"-nx", "-q", "--batch", "-iex", "python import sys; print('go gdb python support')"}
	gdbArgsFixup(args)
	cmd := exec.Command("gdb", args...)
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

// checkPtraceScope checks the value of the kernel parameter ptrace_scope,
// skips the test when gdb cannot attach to the target process via ptrace.
// See issue 69932
//
// 0 - Default attach security permissions.
// 1 - Restricted attach. Only child processes plus normal permissions.
// 2 - Admin-only attach. Only executables with CAP_SYS_PTRACE.
// 3 - No attach. No process may call ptrace at all. Irrevocable.
func checkPtraceScope(t *testing.T) {
	if runtime.GOOS != "linux" {
		return
	}

	// If the Linux kernel does not have the YAMA module enabled,
	// there will be no ptrace_scope file, which does not affect the tests.
	path := "/proc/sys/kernel/yama/ptrace_scope"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read file: %v", err)
	}
	value, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		t.Fatalf("failed converting value to int: %v", err)
	}
	switch value {
	case 3:
		t.Skip("skipping ptrace: Operation not permitted")
	case 2:
		if os.Geteuid() != 0 {
			t.Skip("skipping ptrace: Operation not permitted with non-root user")
		}
	}
}

// NOTE: the maps below are allocated larger than abi.MapBucketCount
// to ensure that they are not "optimized out".

var helloSource = `
import "fmt"
import "runtime"
var gslice []string
// TODO(prattmic): Stack allocated maps initialized inline appear "optimized out" in GDB.
var smallmapvar map[string]string
func main() {
	smallmapvar = make(map[string]string)
	mapvar := make(map[string]string, ` + strconv.FormatInt(abi.OldMapBucketCount+9, 10) + `)
	slicemap := make(map[string][]string,` + strconv.FormatInt(abi.OldMapBucketCount+3, 10) + `)
    chanint := make(chan int, 10)
    chanstr := make(chan string, 10)
    chanint <- 99
	chanint <- 11
    chanstr <- "spongepants"
    chanstr <- "squarebob"
	smallmapvar["abc"] = "def"
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
	runtime.KeepAlive(smallmapvar)
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

func gdbArgsFixup(args []string) {
	if runtime.GOOS != "windows" {
		return
	}
	// On Windows, some gdb flavors expect -ex and -iex arguments
	// containing spaces to be double quoted.
	var quote bool
	for i, arg := range args {
		if arg == "-iex" || arg == "-ex" {
			quote = true
		} else if quote {
			if strings.ContainsRune(arg, ' ') {
				args[i] = `"` + arg + `"`
			}
			quote = false
		}
	}
}

func TestGdbPython(t *testing.T) {
	testGdbPython(t, false)
}

func TestGdbPythonCgo(t *testing.T) {
	if strings.HasPrefix(runtime.GOARCH, "mips") {
		testenv.SkipFlaky(t, 37794)
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
	checkPtraceScope(t)

	dir := t.TempDir()

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

	err := os.WriteFile(filepath.Join(dir, "main.go"), src, 0644)
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
		"-iex", "add-auto-load-safe-path " + filepath.Join(testenv.GOROOT(t), "src", "runtime"),
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
			"-ex", "source "+filepath.Join(testenv.GOROOT(t), "src", "runtime", "runtime-gdb.py"),
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
		"-ex", "echo BEGIN print smallmapvar\n",
		"-ex", "print smallmapvar",
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
	gdbArgsFixup(args)
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	got = bytes.ReplaceAll(got, []byte("\r\n"), []byte("\n")) // normalize line endings
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

	printSmallMapvarRe := regexp.MustCompile(`^\$[0-9]+ = map\[string\]string = {\[(0x[0-9a-f]+\s+)?"abc"\] = (0x[0-9a-f]+\s+)?"def"}$`)
	if bl := blocks["print smallmapvar"]; !printSmallMapvarRe.MatchString(bl) {
		t.Fatalf("print smallmapvar failed: %s", bl)
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
	if flag.Lookup("test.parallel").Value.(flag.Getter).Get().(int) < 2 {
		// It is possible that this test will hang for a long time due to an
		// apparent GDB bug reported in https://go.dev/issue/37405.
		// If test parallelism is high enough, that might be ok: the other parallel
		// tests will finish, and then this test will finish right before it would
		// time out. However, if test are running sequentially, a hang in this test
		// would likely cause the remaining tests to run out of time.
		testenv.SkipFlaky(t, 37405)
	}

	checkGdbEnvironment(t)
	t.Parallel()
	checkGdbVersion(t)
	checkPtraceScope(t)

	dir := t.TempDir()

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err := os.WriteFile(src, []byte(backtraceSource), 0644)
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
	start := time.Now()
	args := []string{"-nx", "-batch",
		"-iex", "add-auto-load-safe-path " + filepath.Join(testenv.GOROOT(t), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "break main.eee",
		"-ex", "run",
		"-ex", "backtrace",
		"-ex", "continue",
		filepath.Join(dir, "a.exe"),
	}
	gdbArgsFixup(args)
	cmd = testenv.Command(t, "gdb", args...)

	// Work around the GDB hang reported in https://go.dev/issue/37405.
	// Sometimes (rarely), the GDB process hangs completely when the Go program
	// exits, and we suspect that the bug is on the GDB side.
	//
	// The default Cancel function added by testenv.Command will mark the test as
	// failed if it is in danger of timing out, but we want to instead mark it as
	// skipped. Change the Cancel function to kill the process and merely log
	// instead of failing the test.
	//
	// (This approach does not scale: if the test parallelism is less than or
	// equal to the number of tests that run right up to the deadline, then the
	// remaining parallel tests are likely to time out. But as long as it's just
	// this one flaky test, it's probably fine..?)
	//
	// If there is no deadline set on the test at all, relying on the timeout set
	// by testenv.Command will cause the test to hang indefinitely, but that's
	// what “no deadline” means, after all — and it's probably the right behavior
	// anyway if someone is trying to investigate and fix the GDB bug.
	cmd.Cancel = func() error {
		t.Logf("GDB command timed out after %v: %v", time.Since(start), cmd)
		return cmd.Process.Kill()
	}

	got, err := cmd.CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		switch {
		case bytes.Contains(got, []byte("internal-error: wait returned unexpected status 0x0")):
			// GDB bug: https://sourceware.org/bugzilla/show_bug.cgi?id=28551
			testenv.SkipFlaky(t, 43068)
		case bytes.Contains(got, []byte("Couldn't get registers: No such process.")),
			bytes.Contains(got, []byte("Unable to fetch general registers.: No such process.")),
			bytes.Contains(got, []byte("reading register pc (#64): No such process.")):
			// GDB bug: https://sourceware.org/bugzilla/show_bug.cgi?id=9086
			testenv.SkipFlaky(t, 50838)
		case bytes.Contains(got, []byte("waiting for new child: No child processes.")):
			// GDB bug: Sometimes it fails to wait for a clone child.
			testenv.SkipFlaky(t, 60553)
		case bytes.Contains(got, []byte(" exited normally]\n")):
			// GDB bug: Sometimes the inferior exits fine,
			// but then GDB hangs.
			testenv.SkipFlaky(t, 37405)
		}
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
	checkPtraceScope(t)

	if runtime.GOOS == "aix" && testing.Short() {
		t.Skip("TestGdbAutotmpTypes is too slow on aix/ppc64")
	}

	dir := t.TempDir()

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err := os.WriteFile(src, []byte(autotmpTypeSource), 0644)
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
		"-iex", "add-auto-load-safe-path " + filepath.Join(testenv.GOROOT(t), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		// Some gdb may set scheduling-locking as "step" by default. This prevents background tasks
		// (e.g GC) from completing which may result in a hang when executing the step command.
		// See #49852.
		"-ex", "set scheduler-locking off",
		"-ex", "break main.main",
		"-ex", "run",
		"-ex", "step",
		"-ex", "info types astruct",
		filepath.Join(dir, "a.exe"),
	}
	gdbArgsFixup(args)
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	sgot := string(got)

	// Check that the backtrace matches the source code.
	types := []string{
		"[]main.astruct",
		"main.astruct",
	}
	if goexperiment.SwissMap {
		types = append(types, []string{
			"groupReference<string,main.astruct>",
			"table<string,main.astruct>",
			"map<string,main.astruct>",
			"map<string,main.astruct> * map[string]main.astruct",
		}...)
	} else {
		types = append(types, []string{
			"bucket<string,main.astruct>",
			"hash<string,main.astruct>",
			"hash<string,main.astruct> * map[string]main.astruct",
		}...)
	}
	for _, name := range types {
		if !strings.Contains(sgot, name) {
			t.Fatalf("could not find %q in 'info typrs astruct' output", name)
		}
	}
}

const constsSource = `
package main

const aConstant int = 42
const largeConstant uint64 = ^uint64(0)
const minusOne int64 = -1
const typedS string = "typed string"
const untypedS = "untyped string"
const nulS string = "\x00str"

func main() {
	println("hello world")
}
`

func TestGdbConst(t *testing.T) {
	checkGdbEnvironment(t)
	t.Parallel()
	checkGdbVersion(t)
	checkPtraceScope(t)

	dir := t.TempDir()

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err := os.WriteFile(src, []byte(constsSource), 0644)
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
		"-iex", "add-auto-load-safe-path " + filepath.Join(testenv.GOROOT(t), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "break main.main",
		"-ex", "run",
		"-ex", "print main.aConstant",
		"-ex", "print main.largeConstant",
		"-ex", "print main.minusOne",
		"-ex", "print 'runtime.mSpanInUse'",
		"-ex", "print 'runtime._PageSize'",
		"-ex", "print main.typedS",
		"-ex", "print main.untypedS",
		"-ex", "print main.nulS",
		filepath.Join(dir, "a.exe"),
	}
	gdbArgsFixup(args)
	got, err := exec.Command("gdb", args...).CombinedOutput()
	t.Logf("gdb output:\n%s", got)
	if err != nil {
		t.Fatalf("gdb exited with error: %v", err)
	}

	sgot := strings.ReplaceAll(string(got), "\r\n", "\n")

	if !strings.Contains(sgot, `$1 = 42
$2 = 18446744073709551615
$3 = -1
$4 = 1 '\001'
$5 = 8192
$6 = "typed string"
$7 = "untyped string"
$8 = "\000str"`) {
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
	checkPtraceScope(t)

	if runtime.GOOS == "windows" {
		t.Skip("no signals on windows")
	}

	dir := t.TempDir()

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err := os.WriteFile(src, []byte(panicSource), 0644)
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
		"-iex", "add-auto-load-safe-path " + filepath.Join(testenv.GOROOT(t), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "run",
		"-ex", "backtrace",
		filepath.Join(dir, "a.exe"),
	}
	gdbArgsFixup(args)
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
	checkPtraceScope(t)

	dir := t.TempDir()

	// Build the source code.
	src := filepath.Join(dir, "main.go")
	err := os.WriteFile(src, []byte(InfCallstackSource), 0644)
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
		"-iex", "add-auto-load-safe-path " + filepath.Join(testenv.GOROOT(t), "src", "runtime"),
		"-ex", "set startup-with-shell off",
		"-ex", "break setg_gcc",
		"-ex", "run",
		"-ex", "backtrace 3",
		"-ex", "disable 1",
		"-ex", "continue",
		filepath.Join(dir, "a.exe"),
	}
	gdbArgsFixup(args)
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
