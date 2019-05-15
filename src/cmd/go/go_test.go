// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"context"
	"debug/elf"
	"debug/macho"
	"flag"
	"fmt"
	"go/format"
	"internal/race"
	"internal/testenv"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"cmd/go/internal/cache"
	"cmd/go/internal/cfg"
	"cmd/internal/sys"
)

var (
	canRun  = true  // whether we can run go or ./testgo
	canRace = false // whether we can run the race detector
	canCgo  = false // whether we can use cgo
	canMSan = false // whether we can run the memory sanitizer

	exeSuffix string // ".exe" on Windows

	skipExternal = false // skip external tests
)

func tooSlow(t *testing.T) {
	if testing.Short() {
		// In -short mode; skip test, except run it on the {darwin,linux,windows}/amd64 builders.
		if testenv.Builder() != "" && runtime.GOARCH == "amd64" && (runtime.GOOS == "linux" || runtime.GOOS == "darwin" || runtime.GOOS == "windows") {
			return
		}
		t.Skip("skipping test in -short mode")
	}
}

func init() {
	switch runtime.GOOS {
	case "android", "js", "nacl":
		canRun = false
	case "darwin":
		switch runtime.GOARCH {
		case "arm", "arm64":
			canRun = false
		}
	case "linux":
		switch runtime.GOARCH {
		case "arm":
			// many linux/arm machines are too slow to run
			// the full set of external tests.
			skipExternal = true
		case "mips", "mipsle", "mips64", "mips64le":
			// Also slow.
			skipExternal = true
			if testenv.Builder() != "" {
				// On the builders, skip the cmd/go
				// tests. They're too slow and already
				// covered by other ports. There's
				// nothing os/arch specific in the
				// tests.
				canRun = false
			}
		}
	case "freebsd":
		switch runtime.GOARCH {
		case "arm":
			// many freebsd/arm machines are too slow to run
			// the full set of external tests.
			skipExternal = true
			canRun = false
		}
	case "plan9":
		switch runtime.GOARCH {
		case "arm":
			// many plan9/arm machines are too slow to run
			// the full set of external tests.
			skipExternal = true
		}
	case "windows":
		exeSuffix = ".exe"
	}
}

// testGOROOT is the GOROOT to use when running testgo, a cmd/go binary
// build from this process's current GOROOT, but run from a different
// (temp) directory.
var testGOROOT string

var testCC string
var testGOCACHE string

var testGo string
var testTmpDir string
var testBin string

// testCtx is canceled when the test binary is about to time out.
//
// If https://golang.org/issue/28135 is accepted, uses of this variable in test
// functions should be replaced by t.Context().
var testCtx = context.Background()

// The TestMain function creates a go command for testing purposes and
// deletes it after the tests have been run.
func TestMain(m *testing.M) {
	// $GO_GCFLAGS a compiler debug flag known to cmd/dist, make.bash, etc.
	// It is not a standard go command flag; use os.Getenv, not cfg.Getenv.
	if os.Getenv("GO_GCFLAGS") != "" {
		fmt.Fprintf(os.Stderr, "testing: warning: no tests to run\n") // magic string for cmd/go
		fmt.Printf("cmd/go test is not compatible with $GO_GCFLAGS being set\n")
		fmt.Printf("SKIP\n")
		return
	}
	os.Unsetenv("GOROOT_FINAL")

	flag.Parse()

	timeoutFlag := flag.Lookup("test.timeout")
	if timeoutFlag != nil {
		// TODO(golang.org/issue/28147): The go command does not pass the
		// test.timeout flag unless either -timeout or -test.timeout is explicitly
		// set on the command line.
		if d := timeoutFlag.Value.(flag.Getter).Get().(time.Duration); d != 0 {
			aBitShorter := d * 95 / 100
			var cancel context.CancelFunc
			testCtx, cancel = context.WithTimeout(testCtx, aBitShorter)
			defer cancel()
		}
	}

	if *proxyAddr != "" {
		StartProxy()
		select {}
	}

	// Run with a temporary TMPDIR to check that the tests don't
	// leave anything behind.
	topTmpdir, err := ioutil.TempDir("", "cmd-go-test-")
	if err != nil {
		log.Fatal(err)
	}
	if !*testWork {
		defer removeAll(topTmpdir)
	}
	os.Setenv(tempEnvName(), topTmpdir)

	dir, err := ioutil.TempDir(topTmpdir, "tmpdir")
	if err != nil {
		log.Fatal(err)
	}
	testTmpDir = dir
	if !*testWork {
		defer removeAll(testTmpDir)
	}

	testGOCACHE = cache.DefaultDir()
	if canRun {
		testBin = filepath.Join(testTmpDir, "testbin")
		if err := os.Mkdir(testBin, 0777); err != nil {
			log.Fatal(err)
		}
		testGo = filepath.Join(testBin, "go"+exeSuffix)
		args := []string{"build", "-tags", "testgo", "-o", testGo}
		if race.Enabled {
			args = append(args, "-race")
		}
		gotool, err := testenv.GoTool()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}

		goEnv := func(name string) string {
			out, err := exec.Command(gotool, "env", name).CombinedOutput()
			if err != nil {
				fmt.Fprintf(os.Stderr, "go env %s: %v\n%s", name, err, out)
				os.Exit(2)
			}
			return strings.TrimSpace(string(out))
		}
		testGOROOT = goEnv("GOROOT")

		// The whole GOROOT/pkg tree was installed using the GOHOSTOS/GOHOSTARCH
		// toolchain (installed in GOROOT/pkg/tool/GOHOSTOS_GOHOSTARCH).
		// The testgo.exe we are about to create will be built for GOOS/GOARCH,
		// which means it will use the GOOS/GOARCH toolchain
		// (installed in GOROOT/pkg/tool/GOOS_GOARCH).
		// If these are not the same toolchain, then the entire standard library
		// will look out of date (the compilers in those two different tool directories
		// are built for different architectures and have different buid IDs),
		// which will cause many tests to do unnecessary rebuilds and some
		// tests to attempt to overwrite the installed standard library.
		// Bail out entirely in this case.
		hostGOOS := goEnv("GOHOSTOS")
		hostGOARCH := goEnv("GOHOSTARCH")
		if hostGOOS != runtime.GOOS || hostGOARCH != runtime.GOARCH {
			fmt.Fprintf(os.Stderr, "testing: warning: no tests to run\n") // magic string for cmd/go
			fmt.Printf("cmd/go test is not compatible with GOOS/GOARCH != GOHOSTOS/GOHOSTARCH (%s/%s != %s/%s)\n", runtime.GOOS, runtime.GOARCH, hostGOOS, hostGOARCH)
			fmt.Printf("SKIP\n")
			return
		}

		buildCmd := exec.Command(gotool, args...)
		buildCmd.Env = append(os.Environ(), "GOFLAGS=-mod=vendor")
		out, err := buildCmd.CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "building testgo failed: %v\n%s", err, out)
			os.Exit(2)
		}

		out, err = exec.Command(gotool, "env", "CC").CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "could not find testing CC: %v\n%s", err, out)
			os.Exit(2)
		}
		testCC = strings.TrimSpace(string(out))

		if out, err := exec.Command(testGo, "env", "CGO_ENABLED").Output(); err != nil {
			fmt.Fprintf(os.Stderr, "running testgo failed: %v\n", err)
			canRun = false
		} else {
			canCgo, err = strconv.ParseBool(strings.TrimSpace(string(out)))
			if err != nil {
				fmt.Fprintf(os.Stderr, "can't parse go env CGO_ENABLED output: %v\n", strings.TrimSpace(string(out)))
			}
		}

		out, err = exec.Command(gotool, "env", "GOCACHE").CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "could not find testing GOCACHE: %v\n%s", err, out)
			os.Exit(2)
		}
		testGOCACHE = strings.TrimSpace(string(out))

		canMSan = canCgo && sys.MSanSupported(runtime.GOOS, runtime.GOARCH)
		canRace = canCgo && sys.RaceDetectorSupported(runtime.GOOS, runtime.GOARCH)
		// The race detector doesn't work on Alpine Linux:
		// golang.org/issue/14481
		// gccgo does not support the race detector.
		if isAlpineLinux() || runtime.Compiler == "gccgo" {
			canRace = false
		}
	}
	// Don't let these environment variables confuse the test.
	os.Setenv("GOENV", "off")
	os.Unsetenv("GOBIN")
	os.Unsetenv("GOPATH")
	os.Unsetenv("GIT_ALLOW_PROTOCOL")
	os.Setenv("HOME", "/test-go-home-does-not-exist")
	// On some systems the default C compiler is ccache.
	// Setting HOME to a non-existent directory will break
	// those systems. Disable ccache and use real compiler. Issue 17668.
	os.Setenv("CCACHE_DISABLE", "1")
	if cfg.Getenv("GOCACHE") == "" {
		os.Setenv("GOCACHE", testGOCACHE) // because $HOME is gone
	}

	r := m.Run()
	if !*testWork {
		removeAll(testTmpDir) // os.Exit won't run defer
	}

	if !*testWork {
		// There shouldn't be anything left in topTmpdir.
		dirf, err := os.Open(topTmpdir)
		if err != nil {
			log.Fatal(err)
		}
		names, err := dirf.Readdirnames(0)
		if err != nil {
			log.Fatal(err)
		}
		if len(names) > 0 {
			log.Fatalf("unexpected files left in tmpdir: %v", names)
		}

		removeAll(topTmpdir)
	}

	os.Exit(r)
}

func isAlpineLinux() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	fi, err := os.Lstat("/etc/alpine-release")
	return err == nil && fi.Mode().IsRegular()
}

// The length of an mtime tick on this system. This is an estimate of
// how long we need to sleep to ensure that the mtime of two files is
// different.
// We used to try to be clever but that didn't always work (see golang.org/issue/12205).
var mtimeTick time.Duration = 1 * time.Second

// Manage a single run of the testgo binary.
type testgoData struct {
	t              *testing.T
	temps          []string
	wd             string
	env            []string
	tempdir        string
	ran            bool
	inParallel     bool
	stdout, stderr bytes.Buffer
	execDir        string // dir for tg.run
}

// skipIfGccgo skips the test if using gccgo.
func skipIfGccgo(t *testing.T, msg string) {
	if runtime.Compiler == "gccgo" {
		t.Skipf("skipping test not supported on gccgo: %s", msg)
	}
}

// testgo sets up for a test that runs testgo.
func testgo(t *testing.T) *testgoData {
	t.Helper()
	testenv.MustHaveGoBuild(t)

	if skipExternal {
		t.Skipf("skipping external tests on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	return &testgoData{t: t}
}

// must gives a fatal error if err is not nil.
func (tg *testgoData) must(err error) {
	tg.t.Helper()
	if err != nil {
		tg.t.Fatal(err)
	}
}

// check gives a test non-fatal error if err is not nil.
func (tg *testgoData) check(err error) {
	tg.t.Helper()
	if err != nil {
		tg.t.Error(err)
	}
}

// parallel runs the test in parallel by calling t.Parallel.
func (tg *testgoData) parallel() {
	tg.t.Helper()
	if tg.ran {
		tg.t.Fatal("internal testsuite error: call to parallel after run")
	}
	if tg.wd != "" {
		tg.t.Fatal("internal testsuite error: call to parallel after cd")
	}
	for _, e := range tg.env {
		if strings.HasPrefix(e, "GOROOT=") || strings.HasPrefix(e, "GOPATH=") || strings.HasPrefix(e, "GOBIN=") {
			val := e[strings.Index(e, "=")+1:]
			if strings.HasPrefix(val, "testdata") || strings.HasPrefix(val, "./testdata") {
				tg.t.Fatalf("internal testsuite error: call to parallel with testdata in environment (%s)", e)
			}
		}
	}
	tg.inParallel = true
	tg.t.Parallel()
}

// pwd returns the current directory.
func (tg *testgoData) pwd() string {
	tg.t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		tg.t.Fatalf("could not get working directory: %v", err)
	}
	return wd
}

// cd changes the current directory to the named directory. Note that
// using this means that the test must not be run in parallel with any
// other tests.
func (tg *testgoData) cd(dir string) {
	tg.t.Helper()
	if tg.inParallel {
		tg.t.Fatal("internal testsuite error: changing directory when running in parallel")
	}
	if tg.wd == "" {
		tg.wd = tg.pwd()
	}
	abs, err := filepath.Abs(dir)
	tg.must(os.Chdir(dir))
	if err == nil {
		tg.setenv("PWD", abs)
	}
}

// sleep sleeps for one tick, where a tick is a conservative estimate
// of how long it takes for a file modification to get a different
// mtime.
func (tg *testgoData) sleep() {
	time.Sleep(mtimeTick)
}

// setenv sets an environment variable to use when running the test go
// command.
func (tg *testgoData) setenv(name, val string) {
	tg.t.Helper()
	if tg.inParallel && (name == "GOROOT" || name == "GOPATH" || name == "GOBIN") && (strings.HasPrefix(val, "testdata") || strings.HasPrefix(val, "./testdata")) {
		tg.t.Fatalf("internal testsuite error: call to setenv with testdata (%s=%s) after parallel", name, val)
	}
	tg.unsetenv(name)
	tg.env = append(tg.env, name+"="+val)
}

// unsetenv removes an environment variable.
func (tg *testgoData) unsetenv(name string) {
	if tg.env == nil {
		tg.env = append([]string(nil), os.Environ()...)
		tg.env = append(tg.env, "GO111MODULE=off")
	}
	for i, v := range tg.env {
		if strings.HasPrefix(v, name+"=") {
			tg.env = append(tg.env[:i], tg.env[i+1:]...)
			break
		}
	}
}

func (tg *testgoData) goTool() string {
	return testGo
}

// doRun runs the test go command, recording stdout and stderr and
// returning exit status.
func (tg *testgoData) doRun(args []string) error {
	tg.t.Helper()
	if !canRun {
		panic("testgoData.doRun called but canRun false")
	}
	if tg.inParallel {
		for _, arg := range args {
			if strings.HasPrefix(arg, "testdata") || strings.HasPrefix(arg, "./testdata") {
				tg.t.Fatal("internal testsuite error: parallel run using testdata")
			}
		}
	}

	hasGoroot := false
	for _, v := range tg.env {
		if strings.HasPrefix(v, "GOROOT=") {
			hasGoroot = true
			break
		}
	}
	prog := tg.goTool()
	if !hasGoroot {
		tg.setenv("GOROOT", testGOROOT)
	}

	tg.t.Logf("running testgo %v", args)
	cmd := exec.Command(prog, args...)
	tg.stdout.Reset()
	tg.stderr.Reset()
	cmd.Dir = tg.execDir
	cmd.Stdout = &tg.stdout
	cmd.Stderr = &tg.stderr
	cmd.Env = tg.env
	status := cmd.Run()
	if tg.stdout.Len() > 0 {
		tg.t.Log("standard output:")
		tg.t.Log(tg.stdout.String())
	}
	if tg.stderr.Len() > 0 {
		tg.t.Log("standard error:")
		tg.t.Log(tg.stderr.String())
	}
	tg.ran = true
	return status
}

// run runs the test go command, and expects it to succeed.
func (tg *testgoData) run(args ...string) {
	tg.t.Helper()
	if status := tg.doRun(args); status != nil {
		wd, _ := os.Getwd()
		tg.t.Logf("go %v failed unexpectedly in %s: %v", args, wd, status)
		tg.t.FailNow()
	}
}

// runFail runs the test go command, and expects it to fail.
func (tg *testgoData) runFail(args ...string) {
	tg.t.Helper()
	if status := tg.doRun(args); status == nil {
		tg.t.Fatal("testgo succeeded unexpectedly")
	} else {
		tg.t.Log("testgo failed as expected:", status)
	}
}

// runGit runs a git command, and expects it to succeed.
func (tg *testgoData) runGit(dir string, args ...string) {
	tg.t.Helper()
	cmd := exec.Command("git", args...)
	tg.stdout.Reset()
	tg.stderr.Reset()
	cmd.Stdout = &tg.stdout
	cmd.Stderr = &tg.stderr
	cmd.Dir = dir
	cmd.Env = tg.env
	status := cmd.Run()
	if tg.stdout.Len() > 0 {
		tg.t.Log("git standard output:")
		tg.t.Log(tg.stdout.String())
	}
	if tg.stderr.Len() > 0 {
		tg.t.Log("git standard error:")
		tg.t.Log(tg.stderr.String())
	}
	if status != nil {
		tg.t.Logf("git %v failed unexpectedly: %v", args, status)
		tg.t.FailNow()
	}
}

// getStdout returns standard output of the testgo run as a string.
func (tg *testgoData) getStdout() string {
	tg.t.Helper()
	if !tg.ran {
		tg.t.Fatal("internal testsuite error: stdout called before run")
	}
	return tg.stdout.String()
}

// getStderr returns standard error of the testgo run as a string.
func (tg *testgoData) getStderr() string {
	tg.t.Helper()
	if !tg.ran {
		tg.t.Fatal("internal testsuite error: stdout called before run")
	}
	return tg.stderr.String()
}

// doGrepMatch looks for a regular expression in a buffer, and returns
// whether it is found. The regular expression is matched against
// each line separately, as with the grep command.
func (tg *testgoData) doGrepMatch(match string, b *bytes.Buffer) bool {
	tg.t.Helper()
	if !tg.ran {
		tg.t.Fatal("internal testsuite error: grep called before run")
	}
	re := regexp.MustCompile(match)
	for _, ln := range bytes.Split(b.Bytes(), []byte{'\n'}) {
		if re.Match(ln) {
			return true
		}
	}
	return false
}

// doGrep looks for a regular expression in a buffer and fails if it
// is not found. The name argument is the name of the output we are
// searching, "output" or "error". The msg argument is logged on
// failure.
func (tg *testgoData) doGrep(match string, b *bytes.Buffer, name, msg string) {
	tg.t.Helper()
	if !tg.doGrepMatch(match, b) {
		tg.t.Log(msg)
		tg.t.Logf("pattern %v not found in standard %s", match, name)
		tg.t.FailNow()
	}
}

// grepStdout looks for a regular expression in the test run's
// standard output and fails, logging msg, if it is not found.
func (tg *testgoData) grepStdout(match, msg string) {
	tg.t.Helper()
	tg.doGrep(match, &tg.stdout, "output", msg)
}

// grepStderr looks for a regular expression in the test run's
// standard error and fails, logging msg, if it is not found.
func (tg *testgoData) grepStderr(match, msg string) {
	tg.t.Helper()
	tg.doGrep(match, &tg.stderr, "error", msg)
}

// grepBoth looks for a regular expression in the test run's standard
// output or stand error and fails, logging msg, if it is not found.
func (tg *testgoData) grepBoth(match, msg string) {
	tg.t.Helper()
	if !tg.doGrepMatch(match, &tg.stdout) && !tg.doGrepMatch(match, &tg.stderr) {
		tg.t.Log(msg)
		tg.t.Logf("pattern %v not found in standard output or standard error", match)
		tg.t.FailNow()
	}
}

// doGrepNot looks for a regular expression in a buffer and fails if
// it is found. The name and msg arguments are as for doGrep.
func (tg *testgoData) doGrepNot(match string, b *bytes.Buffer, name, msg string) {
	tg.t.Helper()
	if tg.doGrepMatch(match, b) {
		tg.t.Log(msg)
		tg.t.Logf("pattern %v found unexpectedly in standard %s", match, name)
		tg.t.FailNow()
	}
}

// grepStdoutNot looks for a regular expression in the test run's
// standard output and fails, logging msg, if it is found.
func (tg *testgoData) grepStdoutNot(match, msg string) {
	tg.t.Helper()
	tg.doGrepNot(match, &tg.stdout, "output", msg)
}

// grepStderrNot looks for a regular expression in the test run's
// standard error and fails, logging msg, if it is found.
func (tg *testgoData) grepStderrNot(match, msg string) {
	tg.t.Helper()
	tg.doGrepNot(match, &tg.stderr, "error", msg)
}

// grepBothNot looks for a regular expression in the test run's
// standard output or stand error and fails, logging msg, if it is
// found.
func (tg *testgoData) grepBothNot(match, msg string) {
	tg.t.Helper()
	if tg.doGrepMatch(match, &tg.stdout) || tg.doGrepMatch(match, &tg.stderr) {
		tg.t.Log(msg)
		tg.t.Fatalf("pattern %v found unexpectedly in standard output or standard error", match)
	}
}

// doGrepCount counts the number of times a regexp is seen in a buffer.
func (tg *testgoData) doGrepCount(match string, b *bytes.Buffer) int {
	tg.t.Helper()
	if !tg.ran {
		tg.t.Fatal("internal testsuite error: doGrepCount called before run")
	}
	re := regexp.MustCompile(match)
	c := 0
	for _, ln := range bytes.Split(b.Bytes(), []byte{'\n'}) {
		if re.Match(ln) {
			c++
		}
	}
	return c
}

// grepCountBoth returns the number of times a regexp is seen in both
// standard output and standard error.
func (tg *testgoData) grepCountBoth(match string) int {
	tg.t.Helper()
	return tg.doGrepCount(match, &tg.stdout) + tg.doGrepCount(match, &tg.stderr)
}

// creatingTemp records that the test plans to create a temporary file
// or directory. If the file or directory exists already, it will be
// removed. When the test completes, the file or directory will be
// removed if it exists.
func (tg *testgoData) creatingTemp(path string) {
	tg.t.Helper()
	if filepath.IsAbs(path) && !strings.HasPrefix(path, tg.tempdir) {
		tg.t.Fatalf("internal testsuite error: creatingTemp(%q) with absolute path not in temporary directory", path)
	}
	// If we have changed the working directory, make sure we have
	// an absolute path, because we are going to change directory
	// back before we remove the temporary.
	if tg.wd != "" && !filepath.IsAbs(path) {
		path = filepath.Join(tg.pwd(), path)
	}
	tg.must(os.RemoveAll(path))
	tg.temps = append(tg.temps, path)
}

// makeTempdir makes a temporary directory for a run of testgo. If
// the temporary directory was already created, this does nothing.
func (tg *testgoData) makeTempdir() {
	tg.t.Helper()
	if tg.tempdir == "" {
		var err error
		tg.tempdir, err = ioutil.TempDir("", "gotest")
		tg.must(err)
	}
}

// tempFile adds a temporary file for a run of testgo.
func (tg *testgoData) tempFile(path, contents string) {
	tg.t.Helper()
	tg.makeTempdir()
	tg.must(os.MkdirAll(filepath.Join(tg.tempdir, filepath.Dir(path)), 0755))
	bytes := []byte(contents)
	if strings.HasSuffix(path, ".go") {
		formatted, err := format.Source(bytes)
		if err == nil {
			bytes = formatted
		}
	}
	tg.must(ioutil.WriteFile(filepath.Join(tg.tempdir, path), bytes, 0644))
}

// tempDir adds a temporary directory for a run of testgo.
func (tg *testgoData) tempDir(path string) {
	tg.t.Helper()
	tg.makeTempdir()
	if err := os.MkdirAll(filepath.Join(tg.tempdir, path), 0755); err != nil && !os.IsExist(err) {
		tg.t.Fatal(err)
	}
}

// path returns the absolute pathname to file with the temporary
// directory.
func (tg *testgoData) path(name string) string {
	tg.t.Helper()
	if tg.tempdir == "" {
		tg.t.Fatalf("internal testsuite error: path(%q) with no tempdir", name)
	}
	if name == "." {
		return tg.tempdir
	}
	return filepath.Join(tg.tempdir, name)
}

// mustExist fails if path does not exist.
func (tg *testgoData) mustExist(path string) {
	tg.t.Helper()
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			tg.t.Fatalf("%s does not exist but should", path)
		}
		tg.t.Fatalf("%s stat failed: %v", path, err)
	}
}

// mustNotExist fails if path exists.
func (tg *testgoData) mustNotExist(path string) {
	tg.t.Helper()
	if _, err := os.Stat(path); err == nil || !os.IsNotExist(err) {
		tg.t.Fatalf("%s exists but should not (%v)", path, err)
	}
}

// mustHaveContent succeeds if filePath is a path to a file,
// and that file is readable and not empty.
func (tg *testgoData) mustHaveContent(filePath string) {
	tg.mustExist(filePath)
	f, err := os.Stat(filePath)
	if err != nil {
		tg.t.Fatal(err)
	}
	if f.Size() == 0 {
		tg.t.Fatalf("expected %s to have data, but is empty", filePath)
	}
}

// wantExecutable fails with msg if path is not executable.
func (tg *testgoData) wantExecutable(path, msg string) {
	tg.t.Helper()
	if st, err := os.Stat(path); err != nil {
		if !os.IsNotExist(err) {
			tg.t.Log(err)
		}
		tg.t.Fatal(msg)
	} else {
		if runtime.GOOS != "windows" && st.Mode()&0111 == 0 {
			tg.t.Fatalf("binary %s exists but is not executable", path)
		}
	}
}

// wantArchive fails if path is not an archive.
func (tg *testgoData) wantArchive(path string) {
	tg.t.Helper()
	f, err := os.Open(path)
	if err != nil {
		tg.t.Fatal(err)
	}
	buf := make([]byte, 100)
	io.ReadFull(f, buf)
	f.Close()
	if !bytes.HasPrefix(buf, []byte("!<arch>\n")) {
		tg.t.Fatalf("file %s exists but is not an archive", path)
	}
}

// isStale reports whether pkg is stale, and why
func (tg *testgoData) isStale(pkg string) (bool, string) {
	tg.t.Helper()
	tg.run("list", "-f", "{{.Stale}}:{{.StaleReason}}", pkg)
	v := strings.TrimSpace(tg.getStdout())
	f := strings.SplitN(v, ":", 2)
	if len(f) == 2 {
		switch f[0] {
		case "true":
			return true, f[1]
		case "false":
			return false, f[1]
		}
	}
	tg.t.Fatalf("unexpected output checking staleness of package %v: %v", pkg, v)
	panic("unreachable")
}

// wantStale fails with msg if pkg is not stale.
func (tg *testgoData) wantStale(pkg, reason, msg string) {
	tg.t.Helper()
	stale, why := tg.isStale(pkg)
	if !stale {
		tg.t.Fatal(msg)
	}
	// We always accept the reason as being "not installed but
	// available in build cache", because when that is the case go
	// list doesn't try to sort out the underlying reason why the
	// package is not installed.
	if reason == "" && why != "" || !strings.Contains(why, reason) && !strings.Contains(why, "not installed but available in build cache") {
		tg.t.Errorf("wrong reason for Stale=true: %q, want %q", why, reason)
	}
}

// wantNotStale fails with msg if pkg is stale.
func (tg *testgoData) wantNotStale(pkg, reason, msg string) {
	tg.t.Helper()
	stale, why := tg.isStale(pkg)
	if stale {
		tg.t.Fatal(msg)
	}
	if reason == "" && why != "" || !strings.Contains(why, reason) {
		tg.t.Errorf("wrong reason for Stale=false: %q, want %q", why, reason)
	}
}

// If -testwork is specified, the test prints the name of the temp directory
// and does not remove it when done, so that a programmer can
// poke at the test file tree afterward.
var testWork = flag.Bool("testwork", false, "")

// cleanup cleans up a test that runs testgo.
func (tg *testgoData) cleanup() {
	tg.t.Helper()
	if tg.wd != "" {
		wd, _ := os.Getwd()
		tg.t.Logf("ended in %s", wd)

		if err := os.Chdir(tg.wd); err != nil {
			// We are unlikely to be able to continue.
			fmt.Fprintln(os.Stderr, "could not restore working directory, crashing:", err)
			os.Exit(2)
		}
	}
	if *testWork {
		tg.t.Logf("TESTWORK=%s\n", tg.path("."))
		return
	}
	for _, path := range tg.temps {
		tg.check(removeAll(path))
	}
	if tg.tempdir != "" {
		tg.check(removeAll(tg.tempdir))
	}
}

func removeAll(dir string) error {
	// module cache has 0444 directories;
	// make them writable in order to remove content.
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // ignore errors walking in file system
		}
		if info.IsDir() {
			os.Chmod(path, 0777)
		}
		return nil
	})
	return os.RemoveAll(dir)
}

// failSSH puts an ssh executable in the PATH that always fails.
// This is to stub out uses of ssh by go get.
func (tg *testgoData) failSSH() {
	tg.t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		tg.t.Fatal(err)
	}
	fail := filepath.Join(wd, "testdata/failssh")
	tg.setenv("PATH", fmt.Sprintf("%v%c%v", fail, filepath.ListSeparator, os.Getenv("PATH")))
}

func TestNewReleaseRebuildsStalePackagesInGOPATH(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping lengthy test in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()

	// Copy the runtime packages into a temporary GOROOT
	// so that we can change files.
	for _, copydir := range []string{
		"src/runtime",
		"src/internal/bytealg",
		"src/internal/cpu",
		"src/unsafe",
		filepath.Join("pkg", runtime.GOOS+"_"+runtime.GOARCH),
		filepath.Join("pkg/tool", runtime.GOOS+"_"+runtime.GOARCH),
		"pkg/include",
	} {
		srcdir := filepath.Join(testGOROOT, copydir)
		tg.tempDir(filepath.Join("goroot", copydir))
		err := filepath.Walk(srcdir,
			func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				if info.IsDir() {
					return nil
				}
				srcrel, err := filepath.Rel(srcdir, path)
				if err != nil {
					return err
				}
				dest := filepath.Join("goroot", copydir, srcrel)
				data, err := ioutil.ReadFile(path)
				if err != nil {
					return err
				}
				tg.tempFile(dest, string(data))
				if err := os.Chmod(tg.path(dest), info.Mode()); err != nil {
					return err
				}
				return nil
			})
		if err != nil {
			t.Fatal(err)
		}
	}
	tg.setenv("GOROOT", tg.path("goroot"))

	addVar := func(name string, idx int) (restore func()) {
		data, err := ioutil.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		old := data
		data = append(data, fmt.Sprintf("var DummyUnusedVar%d bool\n", idx)...)
		if err := ioutil.WriteFile(name, append(data, '\n'), 0666); err != nil {
			t.Fatal(err)
		}
		tg.sleep()
		return func() {
			if err := ioutil.WriteFile(name, old, 0666); err != nil {
				t.Fatal(err)
			}
		}
	}

	// Every main package depends on the "runtime".
	tg.tempFile("d1/src/p1/p1.go", `package main; func main(){}`)
	tg.setenv("GOPATH", tg.path("d1"))
	// Pass -i flag to rebuild everything outdated.
	tg.run("install", "-i", "p1")
	tg.wantNotStale("p1", "", "./testgo list claims p1 is stale, incorrectly, before any changes")

	// Changing mtime of runtime/internal/sys/sys.go
	// should have no effect: only the content matters.
	// In fact this should be true even outside a release branch.
	sys := tg.path("goroot/src/runtime/internal/sys/sys.go")
	tg.sleep()
	restore := addVar(sys, 0)
	restore()
	tg.wantNotStale("p1", "", "./testgo list claims p1 is stale, incorrectly, after updating mtime of runtime/internal/sys/sys.go")

	// But changing content of any file should have an effect.
	// Previously zversion.go was the only one that mattered;
	// now they all matter, so keep using sys.go.
	restore = addVar(sys, 1)
	defer restore()
	tg.wantStale("p1", "stale dependency: runtime/internal/sys", "./testgo list claims p1 is NOT stale, incorrectly, after changing sys.go")
	restore()
	tg.wantNotStale("p1", "", "./testgo list claims p1 is stale, incorrectly, after changing back to old release")
	addVar(sys, 2)
	tg.wantStale("p1", "stale dependency: runtime", "./testgo list claims p1 is NOT stale, incorrectly, after changing sys.go again")
	tg.run("install", "-i", "p1")
	tg.wantNotStale("p1", "", "./testgo list claims p1 is stale after building with new release")

	// Restore to "old" release.
	restore()
	tg.wantStale("p1", "stale dependency: runtime/internal/sys", "./testgo list claims p1 is NOT stale, incorrectly, after restoring sys.go")
	tg.run("install", "-i", "p1")
	tg.wantNotStale("p1", "", "./testgo list claims p1 is stale after building with old release")
}

func testLocalRun(tg *testgoData, exepath, local, match string) {
	tg.t.Helper()
	out, err := exec.Command(exepath).Output()
	if err != nil {
		tg.t.Fatalf("error running %v: %v", exepath, err)
	}
	if !regexp.MustCompile(match).Match(out) {
		tg.t.Log(string(out))
		tg.t.Errorf("testdata/%s/easy.go did not generate expected output", local)
	}
}

func testLocalEasy(tg *testgoData, local string) {
	tg.t.Helper()
	exepath := "./easy" + exeSuffix
	tg.creatingTemp(exepath)
	tg.run("build", "-o", exepath, filepath.Join("testdata", local, "easy.go"))
	testLocalRun(tg, exepath, local, `(?m)^easysub\.Hello`)
}

func testLocalEasySub(tg *testgoData, local string) {
	tg.t.Helper()
	exepath := "./easysub" + exeSuffix
	tg.creatingTemp(exepath)
	tg.run("build", "-o", exepath, filepath.Join("testdata", local, "easysub", "main.go"))
	testLocalRun(tg, exepath, local, `(?m)^easysub\.Hello`)
}

func testLocalHard(tg *testgoData, local string) {
	tg.t.Helper()
	exepath := "./hard" + exeSuffix
	tg.creatingTemp(exepath)
	tg.run("build", "-o", exepath, filepath.Join("testdata", local, "hard.go"))
	testLocalRun(tg, exepath, local, `(?m)^sub\.Hello`)
}

func testLocalInstall(tg *testgoData, local string) {
	tg.t.Helper()
	tg.runFail("install", filepath.Join("testdata", local, "easy.go"))
}

func TestLocalImportsEasy(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	testLocalEasy(tg, "local")
}

func TestLocalImportsEasySub(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	testLocalEasySub(tg, "local")
}

func TestLocalImportsHard(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	testLocalHard(tg, "local")
}

func TestLocalImportsGoInstallShouldFail(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	testLocalInstall(tg, "local")
}

const badDirName = `#$%:, &()*;<=>?\^{}`

func copyBad(tg *testgoData) {
	tg.t.Helper()
	if runtime.GOOS == "windows" {
		tg.t.Skipf("skipping test because %q is an invalid directory name", badDirName)
	}

	tg.must(filepath.Walk("testdata/local",
		func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() {
				return nil
			}
			var data []byte
			data, err = ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			newpath := strings.Replace(path, "local", badDirName, 1)
			tg.tempFile(newpath, string(data))
			return nil
		}))
	tg.cd(tg.path("."))
}

func TestBadImportsEasy(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	copyBad(tg)
	testLocalEasy(tg, badDirName)
}

func TestBadImportsEasySub(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	copyBad(tg)
	testLocalEasySub(tg, badDirName)
}

func TestBadImportsHard(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	copyBad(tg)
	testLocalHard(tg, badDirName)
}

func TestBadImportsGoInstallShouldFail(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	copyBad(tg)
	testLocalInstall(tg, badDirName)
}

func TestInternalPackagesInGOROOTAreRespected(t *testing.T) {
	skipIfGccgo(t, "gccgo does not have GOROOT")
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("build", "-v", "./testdata/testinternal")
	tg.grepBoth(`testinternal(\/|\\)p\.go\:3\:8\: use of internal package net/http/internal not allowed`, "wrong error message for testdata/testinternal")
}

func TestInternalPackagesOutsideGOROOTAreRespected(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("build", "-v", "./testdata/testinternal2")
	tg.grepBoth(`testinternal2(\/|\\)p\.go\:3\:8\: use of internal package .*internal/w not allowed`, "wrote error message for testdata/testinternal2")
}

func TestRunInternal(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	dir := filepath.Join(tg.pwd(), "testdata")
	tg.setenv("GOPATH", dir)
	tg.run("run", filepath.Join(dir, "src/run/good.go"))
	tg.runFail("run", filepath.Join(dir, "src/run/bad.go"))
	tg.grepStderr(`testdata(\/|\\)src(\/|\\)run(\/|\\)bad\.go\:3\:8\: use of internal package run/subdir/internal/private not allowed`, "unexpected error for run/bad.go")
}

func TestRunPkg(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	dir := filepath.Join(tg.pwd(), "testdata")
	tg.setenv("GOPATH", dir)
	tg.run("run", "hello")
	tg.grepStderr("hello, world", "did not find hello, world")
	tg.cd(filepath.Join(dir, "src/hello"))
	tg.run("run", ".")
	tg.grepStderr("hello, world", "did not find hello, world")
}

func testMove(t *testing.T, vcs, url, base, config string) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
	tg.must(os.Mkdir(tg.path(".hg"), 0700))
	tg.must(ioutil.WriteFile(filepath.Join(tg.path(".hg"), "hgrc"), nil, 0600))
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-d", url)
	tg.run("get", "-d", "-u", url)
	switch vcs {
	case "svn":
		// SVN doesn't believe in text files so we can't just edit the config.
		// Check out a different repo into the wrong place.
		tg.must(os.RemoveAll(tg.path("src/code.google.com/p/rsc-svn")))
		tg.run("get", "-d", "-u", "code.google.com/p/rsc-svn2/trunk")
		tg.must(os.Rename(tg.path("src/code.google.com/p/rsc-svn2"), tg.path("src/code.google.com/p/rsc-svn")))
	default:
		path := tg.path(filepath.Join("src", config))
		data, err := ioutil.ReadFile(path)
		tg.must(err)
		data = bytes.ReplaceAll(data, []byte(base), []byte(base+"XXX"))
		tg.must(ioutil.WriteFile(path, data, 0644))
	}
	if vcs == "git" {
		// git will ask for a username and password when we
		// run go get -d -f -u. An empty username and
		// password will work. Prevent asking by setting
		// GIT_ASKPASS.
		tg.creatingTemp("sink" + exeSuffix)
		tg.tempFile("src/sink/sink.go", `package main; func main() {}`)
		tg.run("build", "-o", "sink"+exeSuffix, "sink")
		tg.setenv("GIT_ASKPASS", filepath.Join(tg.pwd(), "sink"+exeSuffix))
	}
	tg.runFail("get", "-d", "-u", url)
	tg.grepStderr("is a custom import path for", "go get -d -u "+url+" failed for wrong reason")
	tg.runFail("get", "-d", "-f", "-u", url)
	tg.grepStderr("validating server certificate|[nN]ot [fF]ound", "go get -d -f -u "+url+" failed for wrong reason")
}

func TestInternalPackageErrorsAreHandled(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("list", "./testdata/testinternal3")
}

func TestInternalCache(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/testinternal4"))
	tg.runFail("build", "p")
	tg.grepStderr("internal", "did not fail to build p")
}

func TestMoveGit(t *testing.T) {
	testMove(t, "git", "rsc.io/pdf", "pdf", "rsc.io/pdf/.git/config")
}

func TestMoveHG(t *testing.T) {
	testMove(t, "hg", "vcs-test.golang.org/go/custom-hg-hello", "custom-hg-hello", "vcs-test.golang.org/go/custom-hg-hello/.hg/hgrc")
}

// TODO(rsc): Set up a test case on SourceForge (?) for svn.
// func testMoveSVN(t *testing.T) {
//	testMove(t, "svn", "code.google.com/p/rsc-svn/trunk", "-", "-")
// }

func TestImportCommandMatch(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/importcom"))
	tg.run("build", "./testdata/importcom/works.go")
}

func TestImportCommentMismatch(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/importcom"))
	tg.runFail("build", "./testdata/importcom/wrongplace.go")
	tg.grepStderr(`wrongplace expects import "my/x"`, "go build did not mention incorrect import")
}

func TestImportCommentSyntaxError(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/importcom"))
	tg.runFail("build", "./testdata/importcom/bad.go")
	tg.grepStderr("cannot parse import comment", "go build did not mention syntax error")
}

func TestImportCommentConflict(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/importcom"))
	tg.runFail("build", "./testdata/importcom/conflict.go")
	tg.grepStderr("found import comments", "go build did not mention comment conflict")
}

func TestImportCycle(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/importcycle"))
	tg.runFail("build", "selfimport")

	count := tg.grepCountBoth("import cycle not allowed")
	if count == 0 {
		t.Fatal("go build did not mention cyclical import")
	}
	if count > 1 {
		t.Fatal("go build mentioned import cycle more than once")
	}

	// Don't hang forever.
	tg.run("list", "-e", "-json", "selfimport")
}

// cmd/go: custom import path checking should not apply to Go packages without import comment.
func TestIssue10952(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("skipping because git binary not found")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
	tg.setenv("GOPATH", tg.path("."))
	const importPath = "github.com/zombiezen/go-get-issue-10952"
	tg.run("get", "-d", "-u", importPath)
	repoDir := tg.path("src/" + importPath)
	tg.runGit(repoDir, "remote", "set-url", "origin", "https://"+importPath+".git")
	tg.run("get", "-d", "-u", importPath)
}

func TestIssue16471(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("skipping because git binary not found")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
	tg.setenv("GOPATH", tg.path("."))
	tg.must(os.MkdirAll(tg.path("src/rsc.io/go-get-issue-10952"), 0755))
	tg.runGit(tg.path("src/rsc.io"), "clone", "https://github.com/zombiezen/go-get-issue-10952")
	tg.runFail("get", "-u", "rsc.io/go-get-issue-10952")
	tg.grepStderr("rsc.io/go-get-issue-10952 is a custom import path for https://github.com/rsc/go-get-issue-10952, but .* is checked out from https://github.com/zombiezen/go-get-issue-10952", "did not detect updated import path")
}

// Test git clone URL that uses SCP-like syntax and custom import path checking.
func TestIssue11457(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("skipping because git binary not found")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
	tg.setenv("GOPATH", tg.path("."))
	const importPath = "rsc.io/go-get-issue-11457"
	tg.run("get", "-d", "-u", importPath)
	repoDir := tg.path("src/" + importPath)
	tg.runGit(repoDir, "remote", "set-url", "origin", "git@github.com:rsc/go-get-issue-11457")

	// At this time, custom import path checking compares remotes verbatim (rather than
	// just the host and path, skipping scheme and user), so we expect go get -u to fail.
	// However, the goal of this test is to verify that gitRemoteRepo correctly parsed
	// the SCP-like syntax, and we expect it to appear in the error message.
	tg.runFail("get", "-d", "-u", importPath)
	want := " is checked out from ssh://git@github.com/rsc/go-get-issue-11457"
	if !strings.HasSuffix(strings.TrimSpace(tg.getStderr()), want) {
		t.Error("expected clone URL to appear in stderr")
	}
}

func TestGetGitDefaultBranch(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("skipping because git binary not found")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
	tg.setenv("GOPATH", tg.path("."))

	// This repo has two branches, master and another-branch.
	// The another-branch is the default that you get from 'git clone'.
	// The go get command variants should not override this.
	const importPath = "github.com/rsc/go-get-default-branch"

	tg.run("get", "-d", importPath)
	repoDir := tg.path("src/" + importPath)
	tg.runGit(repoDir, "branch", "--contains", "HEAD")
	tg.grepStdout(`\* another-branch`, "not on correct default branch")

	tg.run("get", "-d", "-u", importPath)
	tg.runGit(repoDir, "branch", "--contains", "HEAD")
	tg.grepStdout(`\* another-branch`, "not on correct default branch")
}

// Security issue. Don't disable. See golang.org/issue/22125.
func TestAccidentalGitCheckout(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("skipping because git binary not found")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")

	tg.setenv("GOPATH", tg.path("."))

	tg.runFail("get", "-u", "vcs-test.golang.org/go/test1-svn-git")
	tg.grepStderr("src[\\\\/]vcs-test.* uses git, but parent .*src[\\\\/]vcs-test.* uses svn", "get did not fail for right reason")

	if _, err := os.Stat(tg.path("SrC")); err == nil {
		// This case only triggers on a case-insensitive file system.
		tg.runFail("get", "-u", "vcs-test.golang.org/go/test2-svn-git/test2main")
		tg.grepStderr("src[\\\\/]vcs-test.* uses git, but parent .*src[\\\\/]vcs-test.* uses svn", "get did not fail for right reason")
	}
}

func TestErrorMessageForSyntaxErrorInTestGoFileSaysFAIL(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("test", "syntaxerror")
	tg.grepStderr("x_test.go:", "did not diagnose error")
	tg.grepStdout("FAIL", "go test did not say FAIL")
}

func TestWildcardsDoNotLookInUselessDirectories(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("list", "...")
	tg.grepBoth("badpkg", "go list ... failure does not mention badpkg")
	tg.run("list", "m...")
}

func TestRelativeImportsGoTest(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "./testdata/testimport")
}

func TestRelativeImportsGoTestDashI(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()

	// don't let test -i overwrite runtime
	tg.wantNotStale("runtime", "", "must be non-stale before test -i")

	tg.run("test", "-i", "./testdata/testimport")
}

func TestRelativeImportsInCommandLinePackage(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	files, err := filepath.Glob("./testdata/testimport/*.go")
	tg.must(err)
	tg.run(append([]string{"test"}, files...)...)
}

func TestNonCanonicalImportPaths(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("build", "canonical/d")
	tg.grepStderr("package canonical/d", "did not report canonical/d")
	tg.grepStderr("imports canonical/b", "did not report canonical/b")
	tg.grepStderr("imports canonical/a/: non-canonical", "did not report canonical/a/")
}

func TestVersionControlErrorMessageIncludesCorrectDirectory(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata/shadow/root1"))
	tg.runFail("get", "-u", "foo")

	// TODO(iant): We should not have to use strconv.Quote here.
	// The code in vcs.go should be changed so that it is not required.
	quoted := strconv.Quote(filepath.Join("testdata", "shadow", "root1", "src", "foo"))
	quoted = quoted[1 : len(quoted)-1]

	tg.grepStderr(regexp.QuoteMeta(quoted), "go get -u error does not mention shadow/root1/src/foo")
}

func TestInstallFailsWithNoBuildableFiles(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("CGO_ENABLED", "0")
	tg.runFail("install", "cgotest")
	tg.grepStderr("build constraints exclude all Go files", "go install cgotest did not report 'build constraints exclude all Go files'")
}

// Issue 21895
func TestMSanAndRaceRequireCgo(t *testing.T) {
	if !canMSan && !canRace {
		t.Skip("skipping because both msan and the race detector are not supported")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("triv.go", `package main; func main() {}`)
	tg.setenv("CGO_ENABLED", "0")
	if canRace {
		tg.runFail("install", "-race", "triv.go")
		tg.grepStderr("-race requires cgo", "did not correctly report that -race requires cgo")
		tg.grepStderrNot("-msan", "reported that -msan instead of -race requires cgo")
	}
	if canMSan {
		tg.runFail("install", "-msan", "triv.go")
		tg.grepStderr("-msan requires cgo", "did not correctly report that -msan requires cgo")
		tg.grepStderrNot("-race", "reported that -race instead of -msan requires cgo")
	}
}

func TestRelativeGOBINFail(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("triv.go", `package main; func main() {}`)
	tg.setenv("GOBIN", ".")
	tg.cd(tg.path("."))
	tg.runFail("install")
	tg.grepStderr("cannot install, GOBIN must be an absolute path", "go install must fail if $GOBIN is a relative path")
}

// Test that without $GOBIN set, binaries get installed
// into the GOPATH bin directory.
func TestInstallIntoGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.creatingTemp("testdata/bin/go-cmd-test" + exeSuffix)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("install", "go-cmd-test")
	tg.wantExecutable("testdata/bin/go-cmd-test"+exeSuffix, "go install go-cmd-test did not write to testdata/bin/go-cmd-test")
}

// Issue 12407
func TestBuildOutputToDevNull(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	fi1, err1 := os.Lstat(os.DevNull)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("build", "-o", os.DevNull, "go-cmd-test")
	fi2, err2 := os.Lstat(os.DevNull)
	if err1 == nil {
		if err2 != nil {
			t.Errorf("second stat of /dev/null failed: %v", err2)
		} else if !os.SameFile(fi1, fi2) {
			t.Errorf("/dev/null changed: now %v was %v", fi1, fi2)
		}
	}
}

// Issue 28549.
func TestTestOutputToDevNull(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	fi1, err1 := os.Lstat(os.DevNull)
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.tempFile("src/p/p.go", "package p\n")
	tg.tempFile("src/p/p_test.go", "package p\nimport \"testing\"\nfunc TestX(t *testing.T) {}\n")
	tg.run("test", "-o", os.DevNull, "-c", "p")
	tg.mustNotExist("p.test")
	fi2, err2 := os.Lstat(os.DevNull)
	if err1 == nil {
		if err2 != nil {
			t.Errorf("second stat of /dev/null failed: %v", err2)
		} else if !os.SameFile(fi1, fi2) {
			t.Errorf("/dev/null changed: now %v was %v", fi1, fi2)
		}
	}
}

func TestPackageMainTestImportsArchiveNotBinary(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	gobin := filepath.Join(tg.pwd(), "testdata", "bin")
	tg.creatingTemp(gobin)
	tg.setenv("GOBIN", gobin)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.must(os.Chtimes("./testdata/src/main_test/m.go", time.Now(), time.Now()))
	tg.sleep()
	tg.run("test", "main_test")
	tg.run("install", "main_test")
	tg.wantNotStale("main_test", "", "after go install, main listed as stale")
	tg.run("test", "main_test")
}

func TestPackageMainTestCompilerFlags(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.tempFile("src/p1/p1.go", "package main\n")
	tg.tempFile("src/p1/p1_test.go", "package main\nimport \"testing\"\nfunc Test(t *testing.T){}\n")
	tg.run("test", "-c", "-n", "p1")
	tg.grepBothNot(`([\\/]compile|gccgo).* (-p main|-fgo-pkgpath=main).*p1\.go`, "should not have run compile -p main p1.go")
	tg.grepStderr(`([\\/]compile|gccgo).* (-p p1|-fgo-pkgpath=p1).*p1\.go`, "should have run compile -p p1 p1.go")
}

// Issue 12690
func TestPackageNotStaleWithTrailingSlash(t *testing.T) {
	skipIfGccgo(t, "gccgo does not have GOROOT")
	tg := testgo(t)
	defer tg.cleanup()

	// Make sure the packages below are not stale.
	tg.wantNotStale("runtime", "", "must be non-stale before test runs")
	tg.wantNotStale("os", "", "must be non-stale before test runs")
	tg.wantNotStale("io", "", "must be non-stale before test runs")

	goroot := runtime.GOROOT()
	tg.setenv("GOROOT", goroot+"/")

	tg.wantNotStale("runtime", "", "with trailing slash in GOROOT, runtime listed as stale")
	tg.wantNotStale("os", "", "with trailing slash in GOROOT, os listed as stale")
	tg.wantNotStale("io", "", "with trailing slash in GOROOT, io listed as stale")
}

// With $GOBIN set, binaries get installed to $GOBIN.
func TestInstallIntoGOBIN(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	gobin := filepath.Join(tg.pwd(), "testdata", "bin1")
	tg.creatingTemp(gobin)
	tg.setenv("GOBIN", gobin)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("install", "go-cmd-test")
	tg.wantExecutable("testdata/bin1/go-cmd-test"+exeSuffix, "go install go-cmd-test did not write to testdata/bin1/go-cmd-test")
}

// Issue 11065
func TestInstallToCurrentDirectoryCreatesExecutable(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	pkg := filepath.Join(tg.pwd(), "testdata", "src", "go-cmd-test")
	tg.creatingTemp(filepath.Join(pkg, "go-cmd-test"+exeSuffix))
	tg.setenv("GOBIN", pkg)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.cd(pkg)
	tg.run("install")
	tg.wantExecutable("go-cmd-test"+exeSuffix, "go install did not write to current directory")
}

// Without $GOBIN set, installing a program outside $GOPATH should fail
// (there is nowhere to install it).
func TestInstallWithoutDestinationFails(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("install", "testdata/src/go-cmd-test/helloworld.go")
	tg.grepStderr("no install location for .go files listed on command line", "wrong error")
}

// With $GOBIN set, should install there.
func TestInstallToGOBINCommandLinePackage(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	gobin := filepath.Join(tg.pwd(), "testdata", "bin1")
	tg.creatingTemp(gobin)
	tg.setenv("GOBIN", gobin)
	tg.run("install", "testdata/src/go-cmd-test/helloworld.go")
	tg.wantExecutable("testdata/bin1/helloworld"+exeSuffix, "go install testdata/src/go-cmd-test/helloworld.go did not write testdata/bin1/helloworld")
}

func TestGoGetNonPkg(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempDir("gobin")
	tg.setenv("GOPATH", tg.path("."))
	tg.setenv("GOBIN", tg.path("gobin"))
	tg.runFail("get", "-d", "golang.org/x/tools")
	tg.grepStderr("golang.org/x/tools: no Go files", "missing error")
	tg.runFail("get", "-d", "-u", "golang.org/x/tools")
	tg.grepStderr("golang.org/x/tools: no Go files", "missing error")
	tg.runFail("get", "-d", "golang.org/x/tools")
	tg.grepStderr("golang.org/x/tools: no Go files", "missing error")
}

func TestGoGetTestOnlyPkg(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempDir("gopath")
	tg.setenv("GOPATH", tg.path("gopath"))
	tg.run("get", "golang.org/x/tour/content...")
	tg.run("get", "-t", "golang.org/x/tour/content...")
}

func TestInstalls(t *testing.T) {
	if testing.Short() {
		t.Skip("don't install into GOROOT in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("gobin")
	tg.setenv("GOPATH", tg.path("."))
	goroot := runtime.GOROOT()
	tg.setenv("GOROOT", goroot)

	// cmd/fix installs into tool
	tg.run("env", "GOOS")
	goos := strings.TrimSpace(tg.getStdout())
	tg.setenv("GOOS", goos)
	tg.run("env", "GOARCH")
	goarch := strings.TrimSpace(tg.getStdout())
	tg.setenv("GOARCH", goarch)
	fixbin := filepath.Join(goroot, "pkg", "tool", goos+"_"+goarch, "fix") + exeSuffix
	tg.must(os.RemoveAll(fixbin))
	tg.run("install", "cmd/fix")
	tg.wantExecutable(fixbin, "did not install cmd/fix to $GOROOT/pkg/tool")
	tg.must(os.Remove(fixbin))
	tg.setenv("GOBIN", tg.path("gobin"))
	tg.run("install", "cmd/fix")
	tg.wantExecutable(fixbin, "did not install cmd/fix to $GOROOT/pkg/tool with $GOBIN set")
	tg.unsetenv("GOBIN")

	// gopath program installs into GOBIN
	tg.tempFile("src/progname/p.go", `package main; func main() {}`)
	tg.setenv("GOBIN", tg.path("gobin"))
	tg.run("install", "progname")
	tg.unsetenv("GOBIN")
	tg.wantExecutable(tg.path("gobin/progname")+exeSuffix, "did not install progname to $GOBIN/progname")

	// gopath program installs into GOPATH/bin
	tg.run("install", "progname")
	tg.wantExecutable(tg.path("bin/progname")+exeSuffix, "did not install progname to $GOPATH/bin/progname")
}

func TestRejectRelativeDotPathInGOPATHCommandLinePackage(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", ".")
	tg.runFail("build", "testdata/src/go-cmd-test/helloworld.go")
	tg.grepStderr("GOPATH entry is relative", "expected an error message rejecting relative GOPATH entries")
}

func TestRejectRelativePathsInGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", sep+filepath.Join(tg.pwd(), "testdata")+sep+".")
	tg.runFail("build", "go-cmd-test")
	tg.grepStderr("GOPATH entry is relative", "expected an error message rejecting relative GOPATH entries")
}

func TestRejectRelativePathsInGOPATHCommandLinePackage(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", "testdata")
	tg.runFail("build", "testdata/src/go-cmd-test/helloworld.go")
	tg.grepStderr("GOPATH entry is relative", "expected an error message rejecting relative GOPATH entries")
}

// Issue 21928.
func TestRejectBlankPathsInGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", " "+sep+filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("build", "go-cmd-test")
	tg.grepStderr("GOPATH entry is relative", "expected an error message rejecting relative GOPATH entries")
}

// Issue 21928.
func TestIgnoreEmptyPathsInGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.creatingTemp("testdata/bin/go-cmd-test" + exeSuffix)
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", ""+sep+filepath.Join(tg.pwd(), "testdata"))
	tg.run("install", "go-cmd-test")
	tg.wantExecutable("testdata/bin/go-cmd-test"+exeSuffix, "go install go-cmd-test did not write to testdata/bin/go-cmd-test")
}

// Issue 4104.
func TestGoTestWithPackageListedMultipleTimes(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.run("test", "errors", "errors", "errors", "errors", "errors")
	if strings.Contains(strings.TrimSpace(tg.getStdout()), "\n") {
		t.Error("go test errors errors errors errors errors tested the same package multiple times")
	}
}

func TestGoListHasAConsistentOrder(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.run("list", "std")
	first := tg.getStdout()
	tg.run("list", "std")
	if first != tg.getStdout() {
		t.Error("go list std ordering is inconsistent")
	}
}

func TestGoListStdDoesNotIncludeCommands(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.run("list", "std")
	tg.grepStdoutNot("cmd/", "go list std shows commands")
}

func TestGoListCmdOnlyShowsCommands(t *testing.T) {
	skipIfGccgo(t, "gccgo does not have GOROOT")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.run("list", "cmd")
	out := strings.TrimSpace(tg.getStdout())
	for _, line := range strings.Split(out, "\n") {
		if !strings.Contains(line, "cmd/") {
			t.Error("go list cmd shows non-commands")
			break
		}
	}
}

func TestGoListDedupsPackages(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("list", "xtestonly", "./testdata/src/xtestonly/...")
	got := strings.TrimSpace(tg.getStdout())
	const want = "xtestonly"
	if got != want {
		t.Errorf("got %q; want %q", got, want)
	}
}

func TestGoListDeps(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src/p1/p2/p3/p4")
	tg.setenv("GOPATH", tg.path("."))
	tg.tempFile("src/p1/p.go", "package p1\nimport _ \"p1/p2\"\n")
	tg.tempFile("src/p1/p2/p.go", "package p2\nimport _ \"p1/p2/p3\"\n")
	tg.tempFile("src/p1/p2/p3/p.go", "package p3\nimport _ \"p1/p2/p3/p4\"\n")
	tg.tempFile("src/p1/p2/p3/p4/p.go", "package p4\n")
	tg.run("list", "-f", "{{.Deps}}", "p1")
	tg.grepStdout("p1/p2/p3/p4", "Deps(p1) does not mention p4")

	tg.run("list", "-deps", "p1")
	tg.grepStdout("p1/p2/p3/p4", "-deps p1 does not mention p4")

	if runtime.Compiler != "gccgo" {
		// Check the list is in dependency order.
		tg.run("list", "-deps", "math")
		want := "internal/cpu\nunsafe\nmath/bits\nmath\n"
		out := tg.stdout.String()
		if !strings.Contains(out, "internal/cpu") {
			// Some systems don't use internal/cpu.
			want = "unsafe\nmath/bits\nmath\n"
		}
		if tg.stdout.String() != want {
			t.Fatalf("list -deps math: wrong order\nhave %q\nwant %q", tg.stdout.String(), want)
		}
	}
}

func TestGoListTest(t *testing.T) {
	skipIfGccgo(t, "gccgo does not have standard packages")
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOCACHE", tg.tempdir)

	tg.run("list", "-test", "-deps", "sort")
	tg.grepStdout(`^sort.test$`, "missing test main")
	tg.grepStdout(`^sort$`, "missing real sort")
	tg.grepStdout(`^sort \[sort.test\]$`, "missing test copy of sort")
	tg.grepStdout(`^testing \[sort.test\]$`, "missing test copy of testing")
	tg.grepStdoutNot(`^testing$`, "unexpected real copy of testing")

	tg.run("list", "-test", "sort")
	tg.grepStdout(`^sort.test$`, "missing test main")
	tg.grepStdout(`^sort$`, "missing real sort")
	tg.grepStdout(`^sort \[sort.test\]$`, "unexpected test copy of sort")
	tg.grepStdoutNot(`^testing \[sort.test\]$`, "unexpected test copy of testing")
	tg.grepStdoutNot(`^testing$`, "unexpected real copy of testing")

	tg.run("list", "-test", "cmd/dist", "cmd/doc")
	tg.grepStdout(`^cmd/dist$`, "missing cmd/dist")
	tg.grepStdout(`^cmd/doc$`, "missing cmd/doc")
	tg.grepStdout(`^cmd/doc\.test$`, "missing cmd/doc test")
	tg.grepStdoutNot(`^cmd/dist\.test$`, "unexpected cmd/dist test")
	tg.grepStdoutNot(`^testing`, "unexpected testing")

	tg.run("list", "-test", "runtime/cgo")
	tg.grepStdout(`^runtime/cgo$`, "missing runtime/cgo")

	tg.run("list", "-deps", "-f", "{{if .DepOnly}}{{.ImportPath}}{{end}}", "sort")
	tg.grepStdout(`^internal/reflectlite$`, "missing internal/reflectlite")
	tg.grepStdoutNot(`^sort`, "unexpected sort")
}

func TestGoListCompiledCgo(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOCACHE", tg.tempdir)

	tg.run("list", "-f", `{{join .CgoFiles "\n"}}`, "net")
	if tg.stdout.String() == "" {
		t.Skip("net does not use cgo")
	}
	if strings.Contains(tg.stdout.String(), tg.tempdir) {
		t.Fatalf(".CgoFiles unexpectedly mentioned cache %s", tg.tempdir)
	}
	tg.run("list", "-compiled", "-f", `{{.Dir}}{{"\n"}}{{join .CompiledGoFiles "\n"}}`, "net")
	if !strings.Contains(tg.stdout.String(), tg.tempdir) {
		t.Fatalf(".CompiledGoFiles with -compiled did not mention cache %s", tg.tempdir)
	}
	dir := ""
	for _, file := range strings.Split(tg.stdout.String(), "\n") {
		if file == "" {
			continue
		}
		if dir == "" {
			dir = file
			continue
		}
		if !strings.Contains(file, "/") && !strings.Contains(file, `\`) {
			file = filepath.Join(dir, file)
		}
		if _, err := os.Stat(file); err != nil {
			t.Fatalf("cannot find .CompiledGoFiles result %s: %v", file, err)
		}
	}
}

func TestGoListExport(t *testing.T) {
	skipIfGccgo(t, "gccgo does not have standard packages")
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOCACHE", tg.tempdir)

	tg.run("list", "-f", "{{.Export}}", "strings")
	if tg.stdout.String() != "" {
		t.Fatalf(".Export without -export unexpectedly set")
	}
	tg.run("list", "-export", "-f", "{{.Export}}", "strings")
	file := strings.TrimSpace(tg.stdout.String())
	if file == "" {
		t.Fatalf(".Export with -export was empty")
	}
	if _, err := os.Stat(file); err != nil {
		t.Fatalf("cannot find .Export result %s: %v", file, err)
	}
}

// Issue 4096. Validate the output of unsuccessful go install foo/quxx.
func TestUnsuccessfulGoInstallShouldMentionMissingPackage(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(`cannot find package "foo/quxx" in any of`) != 1 {
		t.Error(`go install foo/quxx expected error: .*cannot find package "foo/quxx" in any of`)
	}
}

func TestGOROOTSearchFailureReporting(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(regexp.QuoteMeta(filepath.Join("foo", "quxx"))+` \(from \$GOROOT\)$`) != 1 {
		t.Error(`go install foo/quxx expected error: .*foo/quxx (from $GOROOT)`)
	}
}

func TestMultipleGOPATHEntriesReportedSeparately(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata", "a")+sep+filepath.Join(tg.pwd(), "testdata", "b"))
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(`testdata[/\\].[/\\]src[/\\]foo[/\\]quxx`) != 2 {
		t.Error(`go install foo/quxx expected error: .*testdata/a/src/foo/quxx (from $GOPATH)\n.*testdata/b/src/foo/quxx`)
	}
}

// Test (from $GOPATH) annotation is reported for the first GOPATH entry,
func TestMentionGOPATHInFirstGOPATHEntry(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata", "a")+sep+filepath.Join(tg.pwd(), "testdata", "b"))
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(regexp.QuoteMeta(filepath.Join("testdata", "a", "src", "foo", "quxx"))+` \(from \$GOPATH\)$`) != 1 {
		t.Error(`go install foo/quxx expected error: .*testdata/a/src/foo/quxx (from $GOPATH)`)
	}
}

// but not on the second.
func TestMentionGOPATHNotOnSecondEntry(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata", "a")+sep+filepath.Join(tg.pwd(), "testdata", "b"))
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(regexp.QuoteMeta(filepath.Join("testdata", "b", "src", "foo", "quxx"))+`$`) != 1 {
		t.Error(`go install foo/quxx expected error: .*testdata/b/src/foo/quxx`)
	}
}

func homeEnvName() string {
	switch runtime.GOOS {
	case "windows":
		return "USERPROFILE"
	case "plan9":
		return "home"
	default:
		return "HOME"
	}
}

func tempEnvName() string {
	switch runtime.GOOS {
	case "windows":
		return "TMP"
	case "plan9":
		return "TMPDIR" // actually plan 9 doesn't have one at all but this is fine
	default:
		return "TMPDIR"
	}
}

func TestDefaultGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("home/go")
	tg.setenv(homeEnvName(), tg.path("home"))

	tg.run("env", "GOPATH")
	tg.grepStdout(regexp.QuoteMeta(tg.path("home/go")), "want GOPATH=$HOME/go")

	tg.setenv("GOROOT", tg.path("home/go"))
	tg.run("env", "GOPATH")
	tg.grepStdoutNot(".", "want unset GOPATH because GOROOT=$HOME/go")

	tg.setenv("GOROOT", tg.path("home/go")+"/")
	tg.run("env", "GOPATH")
	tg.grepStdoutNot(".", "want unset GOPATH because GOROOT=$HOME/go/")
}

func TestDefaultGOPATHGet(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", "")
	tg.tempDir("home")
	tg.setenv(homeEnvName(), tg.path("home"))

	// warn for creating directory
	tg.run("get", "-v", "github.com/golang/example/hello")
	tg.grepStderr("created GOPATH="+regexp.QuoteMeta(tg.path("home/go"))+"; see 'go help gopath'", "did not create GOPATH")

	// no warning if directory already exists
	tg.must(os.RemoveAll(tg.path("home/go")))
	tg.tempDir("home/go")
	tg.run("get", "github.com/golang/example/hello")
	tg.grepStderrNot(".", "expected no output on standard error")

	// error if $HOME/go is a file
	tg.must(os.RemoveAll(tg.path("home/go")))
	tg.tempFile("home/go", "")
	tg.runFail("get", "github.com/golang/example/hello")
	tg.grepStderr(`mkdir .*[/\\]go: .*(not a directory|cannot find the path)`, "expected error because $HOME/go is a file")
}

func TestDefaultGOPATHPrintedSearchList(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", "")
	tg.tempDir("home")
	tg.setenv(homeEnvName(), tg.path("home"))

	tg.runFail("install", "github.com/golang/example/hello")
	tg.grepStderr(regexp.QuoteMeta(tg.path("home/go/src/github.com/golang/example/hello"))+`.*from \$GOPATH`, "expected default GOPATH")
}

// Issue 4186. go get cannot be used to download packages to $GOROOT.
// Test that without GOPATH set, go get should fail.
func TestGoGetIntoGOROOT(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")

	// Fails because GOROOT=GOPATH
	tg.setenv("GOPATH", tg.path("."))
	tg.setenv("GOROOT", tg.path("."))
	tg.runFail("get", "-d", "github.com/golang/example/hello")
	tg.grepStderr("warning: GOPATH set to GOROOT", "go should detect GOPATH=GOROOT")
	tg.grepStderr(`\$GOPATH must not be set to \$GOROOT`, "go should detect GOPATH=GOROOT")

	// Fails because GOROOT=GOPATH after cleaning.
	tg.setenv("GOPATH", tg.path(".")+"/")
	tg.setenv("GOROOT", tg.path("."))
	tg.runFail("get", "-d", "github.com/golang/example/hello")
	tg.grepStderr("warning: GOPATH set to GOROOT", "go should detect GOPATH=GOROOT")
	tg.grepStderr(`\$GOPATH must not be set to \$GOROOT`, "go should detect GOPATH=GOROOT")

	tg.setenv("GOPATH", tg.path("."))
	tg.setenv("GOROOT", tg.path(".")+"/")
	tg.runFail("get", "-d", "github.com/golang/example/hello")
	tg.grepStderr("warning: GOPATH set to GOROOT", "go should detect GOPATH=GOROOT")
	tg.grepStderr(`\$GOPATH must not be set to \$GOROOT`, "go should detect GOPATH=GOROOT")

	// Fails because GOROOT=$HOME/go so default GOPATH unset.
	tg.tempDir("home/go")
	tg.setenv(homeEnvName(), tg.path("home"))
	tg.setenv("GOPATH", "")
	tg.setenv("GOROOT", tg.path("home/go"))
	tg.runFail("get", "-d", "github.com/golang/example/hello")
	tg.grepStderr(`\$GOPATH not set`, "expected GOPATH not set")

	tg.setenv(homeEnvName(), tg.path("home")+"/")
	tg.setenv("GOPATH", "")
	tg.setenv("GOROOT", tg.path("home/go"))
	tg.runFail("get", "-d", "github.com/golang/example/hello")
	tg.grepStderr(`\$GOPATH not set`, "expected GOPATH not set")

	tg.setenv(homeEnvName(), tg.path("home"))
	tg.setenv("GOPATH", "")
	tg.setenv("GOROOT", tg.path("home/go")+"/")
	tg.runFail("get", "-d", "github.com/golang/example/hello")
	tg.grepStderr(`\$GOPATH not set`, "expected GOPATH not set")
}

func TestLdflagsArgumentsWithSpacesIssue3941(t *testing.T) {
	skipIfGccgo(t, "gccgo does not support -ldflags -X")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("main.go", `package main
		var extern string
		func main() {
			println(extern)
		}`)
	tg.run("run", "-ldflags", `-X "main.extern=hello world"`, tg.path("main.go"))
	tg.grepStderr("^hello world", `ldflags -X "main.extern=hello world"' failed`)
}

func TestGoTestCpuprofileLeavesBinaryBehind(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.run("test", "-cpuprofile", "errors.prof", "errors")
	tg.wantExecutable("errors.test"+exeSuffix, "go test -cpuprofile did not create errors.test")
}

func TestGoTestCpuprofileDashOControlsBinaryLocation(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.run("test", "-cpuprofile", "errors.prof", "-o", "myerrors.test"+exeSuffix, "errors")
	tg.wantExecutable("myerrors.test"+exeSuffix, "go test -cpuprofile -o myerrors.test did not create myerrors.test")
}

func TestGoTestMutexprofileLeavesBinaryBehind(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.run("test", "-mutexprofile", "errors.prof", "errors")
	tg.wantExecutable("errors.test"+exeSuffix, "go test -mutexprofile did not create errors.test")
}

func TestGoTestMutexprofileDashOControlsBinaryLocation(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.run("test", "-mutexprofile", "errors.prof", "-o", "myerrors.test"+exeSuffix, "errors")
	tg.wantExecutable("myerrors.test"+exeSuffix, "go test -mutexprofile -o myerrors.test did not create myerrors.test")
}

func TestGoBuildNonMain(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("build", "-buildmode=exe", "-o", "not_main"+exeSuffix, "not_main")
	tg.grepStderr("-buildmode=exe requires exactly one main package", "go build with -o and -buildmode=exe should on a non-main package should throw an error")
	tg.mustNotExist("not_main" + exeSuffix)
}

func TestGoTestDashCDashOControlsBinaryLocation(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.run("test", "-c", "-o", tg.path("myerrors.test"+exeSuffix), "errors")
	tg.wantExecutable(tg.path("myerrors.test"+exeSuffix), "go test -c -o myerrors.test did not create myerrors.test")
}

func TestGoTestDashOWritesBinary(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.run("test", "-o", tg.path("myerrors.test"+exeSuffix), "errors")
	tg.wantExecutable(tg.path("myerrors.test"+exeSuffix), "go test -o myerrors.test did not create myerrors.test")
}

func TestGoTestDashIDashOWritesBinary(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()

	// don't let test -i overwrite runtime
	tg.wantNotStale("runtime", "", "must be non-stale before test -i")

	tg.run("test", "-v", "-i", "-o", tg.path("myerrors.test"+exeSuffix), "errors")
	tg.grepBothNot("PASS|FAIL", "test should not have run")
	tg.wantExecutable(tg.path("myerrors.test"+exeSuffix), "go test -o myerrors.test did not create myerrors.test")
}

// Issue 4568.
func TestSymlinksList(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.tempDir("src")
	tg.must(os.Symlink(tg.path("."), tg.path("src/dir1")))
	tg.tempFile("src/dir1/p.go", "package p")
	tg.setenv("GOPATH", tg.path("."))
	tg.cd(tg.path("src"))
	tg.run("list", "-f", "{{.Root}}", "dir1")
	if strings.TrimSpace(tg.getStdout()) != tg.path(".") {
		t.Error("confused by symlinks")
	}
}

// Issue 14054.
func TestSymlinksVendor(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.tempDir("gopath/src/dir1/vendor/v")
	tg.tempFile("gopath/src/dir1/p.go", "package main\nimport _ `v`\nfunc main(){}")
	tg.tempFile("gopath/src/dir1/vendor/v/v.go", "package v")
	tg.must(os.Symlink(tg.path("gopath/src/dir1"), tg.path("symdir1")))
	tg.setenv("GOPATH", tg.path("gopath"))
	tg.cd(tg.path("symdir1"))
	tg.run("list", "-f", "{{.Root}}", ".")
	if strings.TrimSpace(tg.getStdout()) != tg.path("gopath") {
		t.Error("list confused by symlinks")
	}

	// All of these should succeed, not die in vendor-handling code.
	tg.run("run", "p.go")
	tg.run("build")
	tg.run("install")
}

// Issue 15201.
func TestSymlinksVendor15201(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tg := testgo(t)
	defer tg.cleanup()

	tg.tempDir("gopath/src/x/y/_vendor/src/x")
	tg.must(os.Symlink("../../..", tg.path("gopath/src/x/y/_vendor/src/x/y")))
	tg.tempFile("gopath/src/x/y/w/w.go", "package w\nimport \"x/y/z\"\n")
	tg.must(os.Symlink("../_vendor/src", tg.path("gopath/src/x/y/w/vendor")))
	tg.tempFile("gopath/src/x/y/z/z.go", "package z\n")

	tg.setenv("GOPATH", tg.path("gopath/src/x/y/_vendor")+string(filepath.ListSeparator)+tg.path("gopath"))
	tg.cd(tg.path("gopath/src"))
	tg.run("list", "./...")
}

func TestSymlinksInternal(t *testing.T) {
	testenv.MustHaveSymlink(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempDir("gopath/src/dir1/internal/v")
	tg.tempFile("gopath/src/dir1/p.go", "package main\nimport _ `dir1/internal/v`\nfunc main(){}")
	tg.tempFile("gopath/src/dir1/internal/v/v.go", "package v")
	tg.must(os.Symlink(tg.path("gopath/src/dir1"), tg.path("symdir1")))
	tg.setenv("GOPATH", tg.path("gopath"))
	tg.cd(tg.path("symdir1"))
	tg.run("list", "-f", "{{.Root}}", ".")
	if strings.TrimSpace(tg.getStdout()) != tg.path("gopath") {
		t.Error("list confused by symlinks")
	}

	// All of these should succeed, not die in internal-handling code.
	tg.run("run", "p.go")
	tg.run("build")
	tg.run("install")
}

// Issue 4515.
func TestInstallWithTags(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("bin")
	tg.tempFile("src/example/a/main.go", `package main
		func main() {}`)
	tg.tempFile("src/example/b/main.go", `// +build mytag

		package main
		func main() {}`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("install", "-tags", "mytag", "example/a", "example/b")
	tg.wantExecutable(tg.path("bin/a"+exeSuffix), "go install example/a example/b did not install binaries")
	tg.wantExecutable(tg.path("bin/b"+exeSuffix), "go install example/a example/b did not install binaries")
	tg.must(os.Remove(tg.path("bin/a" + exeSuffix)))
	tg.must(os.Remove(tg.path("bin/b" + exeSuffix)))
	tg.run("install", "-tags", "mytag", "example/...")
	tg.wantExecutable(tg.path("bin/a"+exeSuffix), "go install example/... did not install binaries")
	tg.wantExecutable(tg.path("bin/b"+exeSuffix), "go install example/... did not install binaries")
	tg.run("list", "-tags", "mytag", "example/b...")
	if strings.TrimSpace(tg.getStdout()) != "example/b" {
		t.Error("go list example/b did not find example/b")
	}
}

// Issue 4773
func TestCaseCollisions(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src/example/a/pkg")
	tg.tempDir("src/example/a/Pkg")
	tg.tempDir("src/example/b")
	tg.setenv("GOPATH", tg.path("."))
	tg.tempFile("src/example/a/a.go", `package p
		import (
			_ "example/a/pkg"
			_ "example/a/Pkg"
		)`)
	tg.tempFile("src/example/a/pkg/pkg.go", `package pkg`)
	tg.tempFile("src/example/a/Pkg/pkg.go", `package pkg`)
	tg.run("list", "-json", "example/a")
	tg.grepStdout("case-insensitive import collision", "go list -json example/a did not report import collision")
	tg.runFail("build", "example/a")
	tg.grepStderr("case-insensitive import collision", "go build example/a did not report import collision")
	tg.tempFile("src/example/b/file.go", `package b`)
	tg.tempFile("src/example/b/FILE.go", `package b`)
	f, err := os.Open(tg.path("src/example/b"))
	tg.must(err)
	names, err := f.Readdirnames(0)
	tg.must(err)
	tg.check(f.Close())
	args := []string{"list"}
	if len(names) == 2 {
		// case-sensitive file system, let directory read find both files
		args = append(args, "example/b")
	} else {
		// case-insensitive file system, list files explicitly on command line
		args = append(args, tg.path("src/example/b/file.go"), tg.path("src/example/b/FILE.go"))
	}
	tg.runFail(args...)
	tg.grepStderr("case-insensitive file name collision", "go list example/b did not report file name collision")

	tg.runFail("list", "example/a/pkg", "example/a/Pkg")
	tg.grepStderr("case-insensitive import collision", "go list example/a/pkg example/a/Pkg did not report import collision")
	tg.run("list", "-json", "-e", "example/a/pkg", "example/a/Pkg")
	tg.grepStdout("case-insensitive import collision", "go list -json -e example/a/pkg example/a/Pkg did not report import collision")
	tg.runFail("build", "example/a/pkg", "example/a/Pkg")
	tg.grepStderr("case-insensitive import collision", "go build example/a/pkg example/a/Pkg did not report import collision")
}

// Issue 17451, 17662.
func TestSymlinkWarning(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))

	tg.tempDir("src/example/xx")
	tg.tempDir("yy/zz")
	tg.tempFile("yy/zz/zz.go", "package zz\n")
	if err := os.Symlink(tg.path("yy"), tg.path("src/example/xx/yy")); err != nil {
		t.Skipf("symlink failed: %v", err)
	}
	tg.run("list", "example/xx/z...")
	tg.grepStdoutNot(".", "list should not have matched anything")
	tg.grepStderr("matched no packages", "list should have reported that pattern matched no packages")
	tg.grepStderrNot("symlink", "list should not have reported symlink")

	tg.run("list", "example/xx/...")
	tg.grepStdoutNot(".", "list should not have matched anything")
	tg.grepStderr("matched no packages", "list should have reported that pattern matched no packages")
	tg.grepStderr("ignoring symlink", "list should have reported symlink")
}

// Issue 8181.
func TestGoGetDashTIssue8181(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-v", "-t", "github.com/rsc/go-get-issue-8181/a", "github.com/rsc/go-get-issue-8181/b")
	tg.run("list", "...")
	tg.grepStdout("x/build/gerrit", "missing expected x/build/gerrit")
}

func TestIssue11307(t *testing.T) {
	// go get -u was not working except in checkout directory
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "github.com/rsc/go-get-issue-11307")
	tg.run("get", "-u", "github.com/rsc/go-get-issue-11307") // was failing
}

func TestShadowingLogic(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tg := testgo(t)
	defer tg.cleanup()
	pwd := tg.pwd()
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", filepath.Join(pwd, "testdata", "shadow", "root1")+sep+filepath.Join(pwd, "testdata", "shadow", "root2"))

	// The math in root1 is not "math" because the standard math is.
	tg.run("list", "-f", "({{.ImportPath}}) ({{.ConflictDir}})", "./testdata/shadow/root1/src/math")
	pwdForwardSlash := strings.ReplaceAll(pwd, string(os.PathSeparator), "/")
	if !strings.HasPrefix(pwdForwardSlash, "/") {
		pwdForwardSlash = "/" + pwdForwardSlash
	}
	// The output will have makeImportValid applies, but we only
	// bother to deal with characters we might reasonably see.
	for _, r := range " :" {
		pwdForwardSlash = strings.ReplaceAll(pwdForwardSlash, string(r), "_")
	}
	want := "(_" + pwdForwardSlash + "/testdata/shadow/root1/src/math) (" + filepath.Join(runtime.GOROOT(), "src", "math") + ")"
	if strings.TrimSpace(tg.getStdout()) != want {
		t.Error("shadowed math is not shadowed; looking for", want)
	}

	// The foo in root1 is "foo".
	tg.run("list", "-f", "({{.ImportPath}}) ({{.ConflictDir}})", "./testdata/shadow/root1/src/foo")
	if strings.TrimSpace(tg.getStdout()) != "(foo) ()" {
		t.Error("unshadowed foo is shadowed")
	}

	// The foo in root2 is not "foo" because the foo in root1 got there first.
	tg.run("list", "-f", "({{.ImportPath}}) ({{.ConflictDir}})", "./testdata/shadow/root2/src/foo")
	want = "(_" + pwdForwardSlash + "/testdata/shadow/root2/src/foo) (" + filepath.Join(pwd, "testdata", "shadow", "root1", "src", "foo") + ")"
	if strings.TrimSpace(tg.getStdout()) != want {
		t.Error("shadowed foo is not shadowed; looking for", want)
	}

	// The error for go install should mention the conflicting directory.
	tg.runFail("install", "./testdata/shadow/root2/src/foo")
	want = "go install: no install location for " + filepath.Join(pwd, "testdata", "shadow", "root2", "src", "foo") + ": hidden by " + filepath.Join(pwd, "testdata", "shadow", "root1", "src", "foo")
	if strings.TrimSpace(tg.getStderr()) != want {
		t.Error("wrong shadowed install error; looking for", want)
	}
}

// Only succeeds if source order is preserved.
func TestSourceFileNameOrderPreserved(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "testdata/example1_test.go", "testdata/example2_test.go")
}

// Check that coverage analysis works at all.
// Don't worry about the exact numbers but require not 0.0%.
func checkCoverage(tg *testgoData, data string) {
	tg.t.Helper()
	if regexp.MustCompile(`[^0-9]0\.0%`).MatchString(data) {
		tg.t.Error("some coverage results are 0.0%")
	}
}

func TestCoverageRuns(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-short", "-coverpkg=strings", "strings", "regexp")
	data := tg.getStdout() + tg.getStderr()
	tg.run("test", "-short", "-cover", "strings", "math", "regexp")
	data += tg.getStdout() + tg.getStderr()
	checkCoverage(tg, data)
}

func TestCoverageDotImport(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("test", "-coverpkg=coverdot1,coverdot2", "coverdot2")
	data := tg.getStdout() + tg.getStderr()
	checkCoverage(tg, data)
}

// Check that coverage analysis uses set mode.
// Also check that coverage profiles merge correctly.
func TestCoverageUsesSetMode(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.creatingTemp("testdata/cover.out")
	tg.run("test", "-short", "-cover", "encoding/binary", "errors", "-coverprofile=testdata/cover.out")
	data := tg.getStdout() + tg.getStderr()
	if out, err := ioutil.ReadFile("testdata/cover.out"); err != nil {
		t.Error(err)
	} else {
		if !bytes.Contains(out, []byte("mode: set")) {
			t.Error("missing mode: set")
		}
		if !bytes.Contains(out, []byte("errors.go")) {
			t.Error("missing errors.go")
		}
		if !bytes.Contains(out, []byte("binary.go")) {
			t.Error("missing binary.go")
		}
		if bytes.Count(out, []byte("mode: set")) != 1 {
			t.Error("too many mode: set")
		}
	}
	checkCoverage(tg, data)
}

func TestCoverageUsesAtomicModeForRace(t *testing.T) {
	tooSlow(t)
	if !canRace {
		t.Skip("skipping because race detector not supported")
	}
	skipIfGccgo(t, "gccgo has no cover tool")

	tg := testgo(t)
	defer tg.cleanup()
	tg.creatingTemp("testdata/cover.out")
	tg.run("test", "-short", "-race", "-cover", "encoding/binary", "-coverprofile=testdata/cover.out")
	data := tg.getStdout() + tg.getStderr()
	if out, err := ioutil.ReadFile("testdata/cover.out"); err != nil {
		t.Error(err)
	} else {
		if !bytes.Contains(out, []byte("mode: atomic")) {
			t.Error("missing mode: atomic")
		}
	}
	checkCoverage(tg, data)
}

func TestCoverageSyncAtomicImport(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("test", "-short", "-cover", "-covermode=atomic", "-coverpkg=coverdep/p1", "coverdep")
}

func TestCoverageDepLoop(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	// coverdep2/p1's xtest imports coverdep2/p2 which imports coverdep2/p1.
	// Make sure that coverage on coverdep2/p2 recompiles coverdep2/p2.
	tg.run("test", "-short", "-cover", "coverdep2/p1")
	tg.grepStdout("coverage: 100.0% of statements", "expected 100.0% coverage")
}

func TestCoverageNoStatements(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-cover", "./testdata/testcover/pkg4")
	tg.grepStdout("[no statements]", "expected [no statements] for pkg4")
}

func TestCoverageImportMainLoop(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("test", "importmain/test")
	tg.grepStderr("not an importable package", "did not detect import main")
	tg.runFail("test", "-cover", "importmain/test")
	tg.grepStderr("not an importable package", "did not detect import main")
}

func TestCoveragePattern(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	// If coverpkg=sleepy... expands by package loading
	// (as opposed to pattern matching on deps)
	// then it will try to load sleepybad, which does not compile,
	// and the test command will fail.
	tg.run("test", "-coverprofile="+tg.path("cover.out"), "-coverpkg=sleepy...", "-run=^$", "sleepy1")
}

func TestCoverageErrorLine(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("GOTMPDIR", tg.tempdir)

	tg.runFail("test", "coverbad")
	tg.grepStderr(`coverbad[\\/]p\.go:4`, "did not find coverbad/p.go:4")
	if canCgo {
		tg.grepStderr(`coverbad[\\/]p1\.go:6`, "did not find coverbad/p1.go:6")
	}
	tg.grepStderrNot(regexp.QuoteMeta(tg.tempdir), "found temporary directory in error")
	stderr := tg.getStderr()

	tg.runFail("test", "-cover", "coverbad")
	stderr2 := tg.getStderr()

	// It's OK that stderr2 drops the character position in the error,
	// because of the //line directive (see golang.org/issue/22662).
	stderr = strings.ReplaceAll(stderr, "p.go:4:2:", "p.go:4:")
	if stderr != stderr2 {
		t.Logf("test -cover changed error messages:\nbefore:\n%s\n\nafter:\n%s", stderr, stderr2)
		t.Skip("golang.org/issue/22660")
		t.FailNow()
	}
}

func TestTestBuildFailureOutput(t *testing.T) {
	tooSlow(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	// Doesn't build, -x output should not claim to run test.
	tg.runFail("test", "-x", "coverbad")
	tg.grepStderrNot(`[\\/]coverbad\.test( |$)`, "claimed to run test")
}

func TestCoverageFunc(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	tg.run("test", "-outputdir="+tg.tempdir, "-coverprofile=cover.out", "coverasm")
	tg.run("tool", "cover", "-func="+tg.path("cover.out"))
	tg.grepStdout(`\tg\t*100.0%`, "did not find g 100% covered")
	tg.grepStdoutNot(`\tf\t*[0-9]`, "reported coverage for assembly function f")
}

// Issue 24588.
func TestCoverageDashC(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("test", "-c", "-o", tg.path("coverdep"), "-coverprofile="+tg.path("no/such/dir/cover.out"), "coverdep")
	tg.wantExecutable(tg.path("coverdep"), "go -test -c -coverprofile did not create executable")
}

func TestPluginNonMain(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	pkg := filepath.Join(wd, "testdata", "testdep", "p2")

	tg := testgo(t)
	defer tg.cleanup()

	tg.runFail("build", "-buildmode=plugin", pkg)
}

func TestTestEmpty(t *testing.T) {
	if !canRace {
		t.Skip("no race detector")
	}

	wd, _ := os.Getwd()
	testdata := filepath.Join(wd, "testdata")
	for _, dir := range []string{"pkg", "test", "xtest", "pkgtest", "pkgxtest", "pkgtestxtest", "testxtest"} {
		t.Run(dir, func(t *testing.T) {
			tg := testgo(t)
			defer tg.cleanup()
			tg.setenv("GOPATH", testdata)
			tg.cd(filepath.Join(testdata, "src/empty/"+dir))
			tg.run("test", "-cover", "-coverpkg=.", "-race")
		})
		if testing.Short() {
			break
		}
	}
}

func TestNoGoError(t *testing.T) {
	wd, _ := os.Getwd()
	testdata := filepath.Join(wd, "testdata")
	for _, dir := range []string{"empty/test", "empty/xtest", "empty/testxtest", "exclude", "exclude/ignore", "exclude/empty"} {
		t.Run(dir, func(t *testing.T) {
			tg := testgo(t)
			defer tg.cleanup()
			tg.setenv("GOPATH", testdata)
			tg.cd(filepath.Join(testdata, "src"))
			tg.runFail("build", "./"+dir)
			var want string
			if strings.Contains(dir, "test") {
				want = "no non-test Go files in "
			} else if dir == "exclude" {
				want = "build constraints exclude all Go files in "
			} else {
				want = "no Go files in "
			}
			tg.grepStderr(want, "wrong reason for failure")
		})
	}
}

func TestTestRaceInstall(t *testing.T) {
	if !canRace {
		t.Skip("no race detector")
	}
	tooSlow(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	tg.tempDir("pkg")
	pkgdir := tg.path("pkg")
	tg.run("install", "-race", "-pkgdir="+pkgdir, "std")
	tg.run("test", "-race", "-pkgdir="+pkgdir, "-i", "-v", "empty/pkg")
	if tg.getStderr() != "" {
		t.Error("go test -i -race: rebuilds cached packages")
	}
}

func TestBuildDryRunWithCgo(t *testing.T) {
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("foo.go", `package main

/*
#include <limits.h>
*/
import "C"

func main() {
        println(C.INT_MAX)
}`)
	tg.run("build", "-n", tg.path("foo.go"))
	tg.grepStderrNot(`os.Stat .* no such file or directory`, "unexpected stat of archive file")
}

func TestCoverageWithCgo(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	for _, dir := range []string{"cgocover", "cgocover2", "cgocover3", "cgocover4"} {
		t.Run(dir, func(t *testing.T) {
			tg := testgo(t)
			tg.parallel()
			defer tg.cleanup()
			tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
			tg.run("test", "-short", "-cover", dir)
			data := tg.getStdout() + tg.getStderr()
			checkCoverage(tg, data)
		})
	}
}

func TestCgoAsmError(t *testing.T) {
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	tg := testgo(t)
	tg.parallel()
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("build", "cgoasm")
	tg.grepBoth("package using cgo has Go assembly file", "did not detect Go assembly file")
}

func TestCgoDependsOnSyscall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test that removes $GOROOT/pkg/*_race in short mode")
	}
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}
	if !canRace {
		t.Skip("skipping because race detector not supported")
	}

	tg := testgo(t)
	defer tg.cleanup()
	files, err := filepath.Glob(filepath.Join(runtime.GOROOT(), "pkg", "*_race"))
	tg.must(err)
	for _, file := range files {
		tg.check(os.RemoveAll(file))
	}
	tg.tempFile("src/foo/foo.go", `
		package foo
		//#include <stdio.h>
		import "C"`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("build", "-race", "foo")
}

func TestCgoShowsFullPathNames(t *testing.T) {
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/x/y/dirname/foo.go", `
		package foo
		import "C"
		func f() {`)
	tg.setenv("GOPATH", tg.path("."))
	tg.runFail("build", "x/y/dirname")
	tg.grepBoth("x/y/dirname", "error did not use full path")
}

func TestCgoHandlesWlORIGIN(t *testing.T) {
	tooSlow(t)
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/origin/origin.go", `package origin
		// #cgo !darwin LDFLAGS: -Wl,-rpath,$ORIGIN
		// void f(void) {}
		import "C"
		func f() { C.f() }`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("build", "origin")
}

func TestCgoPkgConfig(t *testing.T) {
	tooSlow(t)
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	tg.run("env", "PKG_CONFIG")
	pkgConfig := strings.TrimSpace(tg.getStdout())
	if out, err := exec.Command(pkgConfig, "--atleast-pkgconfig-version", "0.24").CombinedOutput(); err != nil {
		t.Skipf("%s --atleast-pkgconfig-version 0.24: %v\n%s", pkgConfig, err, out)
	}

	// OpenBSD's pkg-config is strict about whitespace and only
	// supports backslash-escaped whitespace. It does not support
	// quotes, which the normal freedesktop.org pkg-config does
	// support. See https://man.openbsd.org/pkg-config.1
	tg.tempFile("foo.pc", `
Name: foo
Description: The foo library
Version: 1.0.0
Cflags: -Dhello=10 -Dworld=+32 -DDEFINED_FROM_PKG_CONFIG=hello\ world
`)
	tg.tempFile("foo.go", `package main

/*
#cgo pkg-config: foo
int value() {
	return DEFINED_FROM_PKG_CONFIG;
}
*/
import "C"
import "os"

func main() {
	if C.value() != 42 {
		println("value() =", C.value(), "wanted 42")
		os.Exit(1)
	}
}
`)
	tg.setenv("PKG_CONFIG_PATH", tg.path("."))
	tg.run("run", tg.path("foo.go"))
}

// "go test -c -test.bench=XXX errors" should not hang.
// "go test -c" should also produce reproducible binaries.
// "go test -c" should also appear to write a new binary every time,
// even if it's really just updating the mtime on an existing up-to-date binary.
func TestIssue6480(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.run("test", "-c", "-test.bench=XXX", "errors")
	tg.run("test", "-c", "-o", "errors2.test", "errors")

	data1, err := ioutil.ReadFile("errors.test" + exeSuffix)
	tg.must(err)
	data2, err := ioutil.ReadFile("errors2.test") // no exeSuffix because -o above doesn't have it
	tg.must(err)
	if !bytes.Equal(data1, data2) {
		t.Fatalf("go test -c errors produced different binaries when run twice")
	}

	start := time.Now()
	tg.run("test", "-x", "-c", "-test.bench=XXX", "errors")
	tg.grepStderrNot(`[\\/]link|gccgo`, "incorrectly relinked up-to-date test binary")
	info, err := os.Stat("errors.test" + exeSuffix)
	if err != nil {
		t.Fatal(err)
	}
	start = truncateLike(start, info.ModTime())
	if info.ModTime().Before(start) {
		t.Fatalf("mtime of errors.test predates test -c command (%v < %v)", info.ModTime(), start)
	}

	start = time.Now()
	tg.run("test", "-x", "-c", "-o", "errors2.test", "errors")
	tg.grepStderrNot(`[\\/]link|gccgo`, "incorrectly relinked up-to-date test binary")
	info, err = os.Stat("errors2.test")
	if err != nil {
		t.Fatal(err)
	}
	start = truncateLike(start, info.ModTime())
	if info.ModTime().Before(start) {
		t.Fatalf("mtime of errors2.test predates test -c command (%v < %v)", info.ModTime(), start)
	}
}

// truncateLike returns the result of truncating t to the apparent precision of p.
func truncateLike(t, p time.Time) time.Time {
	nano := p.UnixNano()
	d := 1 * time.Nanosecond
	for nano%int64(d) == 0 && d < 1*time.Second {
		d *= 10
	}
	for nano%int64(d) == 0 && d < 2*time.Second {
		d *= 2
	}
	return t.Truncate(d)
}

// cmd/cgo: undefined reference when linking a C-library using gccgo
func TestIssue7573(t *testing.T) {
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}
	if _, err := exec.LookPath("gccgo"); err != nil {
		t.Skip("skipping because no gccgo compiler found")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/cgoref/cgoref.go", `
package main
// #cgo LDFLAGS: -L alibpath -lalib
// void f(void) {}
import "C"

func main() { C.f() }`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("build", "-n", "-compiler", "gccgo", "cgoref")
	tg.grepStderr(`gccgo.*\-L [^ ]*alibpath \-lalib`, `no Go-inline "#cgo LDFLAGS:" ("-L alibpath -lalib") passed to gccgo linking stage`)
}

func TestListTemplateContextFunction(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		v    string
		want string
	}{
		{"GOARCH", runtime.GOARCH},
		{"GOOS", runtime.GOOS},
		{"GOROOT", filepath.Clean(runtime.GOROOT())},
		{"GOPATH", os.Getenv("GOPATH")},
		{"CgoEnabled", ""},
		{"UseAllFiles", ""},
		{"Compiler", ""},
		{"BuildTags", ""},
		{"ReleaseTags", ""},
		{"InstallSuffix", ""},
	} {
		tt := tt
		t.Run(tt.v, func(t *testing.T) {
			tg := testgo(t)
			tg.parallel()
			defer tg.cleanup()
			tmpl := "{{context." + tt.v + "}}"
			tg.run("list", "-f", tmpl)
			if tt.want == "" {
				return
			}
			if got := strings.TrimSpace(tg.getStdout()); got != tt.want {
				t.Errorf("go list -f %q: got %q; want %q", tmpl, got, tt.want)
			}
		})
	}
}

// cmd/go: "go test" should fail if package does not build
func TestIssue7108(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("test", "notest")
}

// cmd/go: go test -a foo does not rebuild regexp.
func TestIssue6844(t *testing.T) {
	if testing.Short() {
		t.Skip("don't rebuild the standard library in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.creatingTemp("deps.test" + exeSuffix)
	tg.run("test", "-x", "-a", "-c", "testdata/dep_test.go")
	tg.grepStderr("regexp", "go test -x -a -c testdata/dep-test.go did not rebuild regexp")
}

func TestBuildDashIInstallsDependencies(t *testing.T) {
	tooSlow(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/x/y/foo/foo.go", `package foo
		func F() {}`)
	tg.tempFile("src/x/y/bar/bar.go", `package bar
		import "x/y/foo"
		func F() { foo.F() }`)
	tg.setenv("GOPATH", tg.path("."))

	// don't let build -i overwrite runtime
	tg.wantNotStale("runtime", "", "must be non-stale before build -i")

	checkbar := func(desc string) {
		tg.run("build", "-v", "-i", "x/y/bar")
		tg.grepBoth("x/y/foo", "first build -i "+desc+" did not build x/y/foo")
		tg.run("build", "-v", "-i", "x/y/bar")
		tg.grepBothNot("x/y/foo", "second build -i "+desc+" built x/y/foo")
	}
	checkbar("pkg")

	tg.creatingTemp("bar" + exeSuffix)
	tg.sleep()
	tg.tempFile("src/x/y/foo/foo.go", `package foo
		func F() { F() }`)
	tg.tempFile("src/x/y/bar/bar.go", `package main
		import "x/y/foo"
		func main() { foo.F() }`)
	checkbar("cmd")
}

func TestGoBuildTestOnly(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.tempFile("src/testonly/t_test.go", `package testonly`)
	tg.tempFile("src/testonly2/t.go", `package testonly2`)
	tg.cd(tg.path("src"))

	// Named explicitly, test-only packages should be reported as unbuildable/uninstallable,
	// even if there is a wildcard also matching.
	tg.runFail("build", "testonly", "testonly...")
	tg.grepStderr("no non-test Go files in", "go build ./xtestonly produced unexpected error")
	tg.runFail("install", "./testonly")
	tg.grepStderr("no non-test Go files in", "go install ./testonly produced unexpected error")

	// Named through a wildcards, the test-only packages should be silently ignored.
	tg.run("build", "testonly...")
	tg.run("install", "./testonly...")
}

func TestGoTestDetectsTestOnlyImportCycles(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("test", "-c", "testcycle/p3")
	tg.grepStderr("import cycle not allowed in test", "go test testcycle/p3 produced unexpected error")

	tg.runFail("test", "-c", "testcycle/q1")
	tg.grepStderr("import cycle not allowed in test", "go test testcycle/q1 produced unexpected error")
}

func TestGoTestFooTestWorks(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "testdata/standalone_test.go")
}

// Issue 22388
func TestGoTestMainWithWrongSignature(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("test", "testdata/standalone_main_wrong_test.go")
	tg.grepStderr(`wrong signature for TestMain, must be: func TestMain\(m \*testing.M\)`, "detected wrong error message")
}

func TestGoTestMainAsNormalTest(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "testdata/standalone_main_normal_test.go")
	tg.grepBoth(okPattern, "go test did not say ok")
}

func TestGoTestMainTwice(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping in short mode")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOCACHE", tg.tempdir)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("test", "-v", "multimain")
	if strings.Count(tg.getStdout(), "notwithstanding") != 2 {
		t.Fatal("tests did not run twice")
	}
}

func TestGoTestFlagsAfterPackage(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "testdata/flag_test.go", "-v", "-args", "-v=7") // Two distinct -v flags.
	tg.run("test", "-v", "testdata/flag_test.go", "-args", "-v=7") // Two distinct -v flags.
}

func TestGoTestXtestonlyWorks(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("clean", "-i", "xtestonly")
	tg.run("test", "xtestonly")
}

func TestGoTestBuildsAnXtestContainingOnlyNonRunnableExamples(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-v", "./testdata/norunexample")
	tg.grepStdout("File with non-runnable example was built.", "file with non-runnable example was not built")
}

func TestGoGenerateHandlesSimpleCommand(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping because windows has no echo command")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.run("generate", "./testdata/generate/test1.go")
	tg.grepStdout("Success", "go generate ./testdata/generate/test1.go generated wrong output")
}

func TestGoGenerateHandlesCommandAlias(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping because windows has no echo command")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.run("generate", "./testdata/generate/test2.go")
	tg.grepStdout("Now is the time for all good men", "go generate ./testdata/generate/test2.go generated wrong output")
}

func TestGoGenerateVariableSubstitution(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping because windows has no echo command")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.run("generate", "./testdata/generate/test3.go")
	tg.grepStdout(runtime.GOARCH+" test3.go:7 pabc xyzp/test3.go/123", "go generate ./testdata/generate/test3.go generated wrong output")
}

func TestGoGenerateRunFlag(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping because windows has no echo command")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.run("generate", "-run", "y.s", "./testdata/generate/test4.go")
	tg.grepStdout("yes", "go generate -run yes ./testdata/generate/test4.go did not select yes")
	tg.grepStdoutNot("no", "go generate -run yes ./testdata/generate/test4.go selected no")
}

func TestGoGenerateEnv(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping because %s does not have the env command", runtime.GOOS)
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("env.go", "package main\n\n//go:generate env")
	tg.run("generate", tg.path("env.go"))
	for _, v := range []string{"GOARCH", "GOOS", "GOFILE", "GOLINE", "GOPACKAGE", "DOLLAR"} {
		tg.grepStdout("^"+v+"=", "go generate environment missing "+v)
	}
}

func TestGoGenerateXTestPkgName(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping because windows has no echo command")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("env_test.go", "package main_test\n\n//go:generate echo $GOPACKAGE")
	tg.run("generate", tg.path("env_test.go"))
	want := "main_test"
	if got := strings.TrimSpace(tg.getStdout()); got != want {
		t.Errorf("go generate in XTest file got package name %q; want %q", got, want)
	}
}

func TestGoGenerateBadImports(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("skipping because windows has no echo command")
	}

	// This package has an invalid import causing an import cycle,
	// but go generate is supposed to still run.
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("generate", "gencycle")
	tg.grepStdout("hello world", "go generate gencycle did not run generator")
}

func TestGoGetCustomDomainWildcard(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-u", "rsc.io/pdf/...")
	tg.wantExecutable(tg.path("bin/pdfpasswd"+exeSuffix), "did not build rsc/io/pdf/pdfpasswd")
}

func TestGoGetInternalWildcard(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	// used to fail with errors about internal packages
	tg.run("get", "github.com/rsc/go-get-issue-11960/...")
}

func TestGoVetWithExternalTests(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("vet", "vetpkg")
	tg.grepBoth("Printf", "go vet vetpkg did not find missing argument for Printf")
}

func TestGoVetWithTags(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("vet", "-tags", "tagtest", "vetpkg")
	tg.grepBoth(`c\.go.*Printf`, "go vet vetpkg did not run scan tagged file")
}

func TestGoVetWithFlagsOn(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("vet", "-printf", "vetpkg")
	tg.grepBoth("Printf", "go vet -printf vetpkg did not find missing argument for Printf")
}

func TestGoVetWithFlagsOff(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("vet", "-printf=false", "vetpkg")
}

// Issue 23395.
func TestGoVetWithOnlyTestFiles(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/p/p_test.go", "package p; import \"testing\"; func TestMe(*testing.T) {}")
	tg.setenv("GOPATH", tg.path("."))
	tg.run("vet", "p")
}

// Issue 24193.
func TestVetWithOnlyCgoFiles(t *testing.T) {
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}
	tooSlow(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/p/p.go", "package p; import \"C\"; func F() {}")
	tg.setenv("GOPATH", tg.path("."))
	tg.run("vet", "p")
}

// Issue 9767, 19769.
func TestGoGetDotSlashDownload(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempDir("src/rsc.io")
	tg.setenv("GOPATH", tg.path("."))
	tg.cd(tg.path("src/rsc.io"))
	tg.run("get", "./pprof_mac_fix")
}

// Test that you cannot import a main package.
// See golang.org/issue/4210 and golang.org/issue/17475.
func TestImportMain(t *testing.T) {
	tooSlow(t)

	tg := testgo(t)
	tg.parallel()
	defer tg.cleanup()

	// Importing package main from that package main's test should work.
	tg.tempFile("src/x/main.go", `package main
		var X int
		func main() {}`)
	tg.tempFile("src/x/main_test.go", `package main_test
		import xmain "x"
		import "testing"
		var _ = xmain.X
		func TestFoo(t *testing.T) {}
	`)
	tg.setenv("GOPATH", tg.path("."))
	tg.creatingTemp("x" + exeSuffix)
	tg.run("build", "x")
	tg.run("test", "x")

	// Importing package main from another package should fail.
	tg.tempFile("src/p1/p.go", `package p1
		import xmain "x"
		var _ = xmain.X
	`)
	tg.runFail("build", "p1")
	tg.grepStderr("import \"x\" is a program, not an importable package", "did not diagnose package main")

	// ... even in that package's test.
	tg.tempFile("src/p2/p.go", `package p2
	`)
	tg.tempFile("src/p2/p_test.go", `package p2
		import xmain "x"
		import "testing"
		var _ = xmain.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "p2")
	tg.runFail("test", "p2")
	tg.grepStderr("import \"x\" is a program, not an importable package", "did not diagnose package main")

	// ... even if that package's test is an xtest.
	tg.tempFile("src/p3/p.go", `package p
	`)
	tg.tempFile("src/p3/p_test.go", `package p_test
		import xmain "x"
		import "testing"
		var _ = xmain.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "p3")
	tg.runFail("test", "p3")
	tg.grepStderr("import \"x\" is a program, not an importable package", "did not diagnose package main")

	// ... even if that package is a package main
	tg.tempFile("src/p4/p.go", `package main
	func main() {}
	`)
	tg.tempFile("src/p4/p_test.go", `package main
		import xmain "x"
		import "testing"
		var _ = xmain.X
		func TestFoo(t *testing.T) {}
	`)
	tg.creatingTemp("p4" + exeSuffix)
	tg.run("build", "p4")
	tg.runFail("test", "p4")
	tg.grepStderr("import \"x\" is a program, not an importable package", "did not diagnose package main")

	// ... even if that package is a package main using an xtest.
	tg.tempFile("src/p5/p.go", `package main
	func main() {}
	`)
	tg.tempFile("src/p5/p_test.go", `package main_test
		import xmain "x"
		import "testing"
		var _ = xmain.X
		func TestFoo(t *testing.T) {}
	`)
	tg.creatingTemp("p5" + exeSuffix)
	tg.run("build", "p5")
	tg.runFail("test", "p5")
	tg.grepStderr("import \"x\" is a program, not an importable package", "did not diagnose package main")
}

// Test that you cannot use a local import in a package
// accessed by a non-local import (found in a GOPATH/GOROOT).
// See golang.org/issue/17475.
func TestImportLocal(t *testing.T) {
	tooSlow(t)

	tg := testgo(t)
	tg.parallel()
	defer tg.cleanup()

	tg.tempFile("src/dir/x/x.go", `package x
		var X int
	`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("build", "dir/x")

	// Ordinary import should work.
	tg.tempFile("src/dir/p0/p.go", `package p0
		import "dir/x"
		var _ = x.X
	`)
	tg.run("build", "dir/p0")

	// Relative import should not.
	tg.tempFile("src/dir/p1/p.go", `package p1
		import "../x"
		var _ = x.X
	`)
	tg.runFail("build", "dir/p1")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// ... even in a test.
	tg.tempFile("src/dir/p2/p.go", `package p2
	`)
	tg.tempFile("src/dir/p2/p_test.go", `package p2
		import "../x"
		import "testing"
		var _ = x.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "dir/p2")
	tg.runFail("test", "dir/p2")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// ... even in an xtest.
	tg.tempFile("src/dir/p2/p_test.go", `package p2_test
		import "../x"
		import "testing"
		var _ = x.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "dir/p2")
	tg.runFail("test", "dir/p2")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// Relative import starting with ./ should not work either.
	tg.tempFile("src/dir/d.go", `package dir
		import "./x"
		var _ = x.X
	`)
	tg.runFail("build", "dir")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// ... even in a test.
	tg.tempFile("src/dir/d.go", `package dir
	`)
	tg.tempFile("src/dir/d_test.go", `package dir
		import "./x"
		import "testing"
		var _ = x.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "dir")
	tg.runFail("test", "dir")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// ... even in an xtest.
	tg.tempFile("src/dir/d_test.go", `package dir_test
		import "./x"
		import "testing"
		var _ = x.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "dir")
	tg.runFail("test", "dir")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// Relative import plain ".." should not work.
	tg.tempFile("src/dir/x/y/y.go", `package dir
		import ".."
		var _ = x.X
	`)
	tg.runFail("build", "dir/x/y")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// ... even in a test.
	tg.tempFile("src/dir/x/y/y.go", `package y
	`)
	tg.tempFile("src/dir/x/y/y_test.go", `package y
		import ".."
		import "testing"
		var _ = x.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "dir/x/y")
	tg.runFail("test", "dir/x/y")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// ... even in an x test.
	tg.tempFile("src/dir/x/y/y_test.go", `package y_test
		import ".."
		import "testing"
		var _ = x.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "dir/x/y")
	tg.runFail("test", "dir/x/y")
	tg.grepStderr("local import.*in non-local package", "did not diagnose local import")

	// Relative import "." should not work.
	tg.tempFile("src/dir/x/xx.go", `package x
		import "."
		var _ = x.X
	`)
	tg.runFail("build", "dir/x")
	tg.grepStderr("cannot import current directory", "did not diagnose import current directory")

	// ... even in a test.
	tg.tempFile("src/dir/x/xx.go", `package x
	`)
	tg.tempFile("src/dir/x/xx_test.go", `package x
		import "."
		import "testing"
		var _ = x.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "dir/x")
	tg.runFail("test", "dir/x")
	tg.grepStderr("cannot import current directory", "did not diagnose import current directory")

	// ... even in an xtest.
	tg.tempFile("src/dir/x/xx.go", `package x
	`)
	tg.tempFile("src/dir/x/xx_test.go", `package x_test
		import "."
		import "testing"
		var _ = x.X
		func TestFoo(t *testing.T) {}
	`)
	tg.run("build", "dir/x")
	tg.runFail("test", "dir/x")
	tg.grepStderr("cannot import current directory", "did not diagnose import current directory")
}

func TestGoGetInsecure(t *testing.T) {
	test := func(t *testing.T, modules bool) {
		testenv.MustHaveExternalNetwork(t)

		tg := testgo(t)
		defer tg.cleanup()
		tg.makeTempdir()
		tg.failSSH()

		if modules {
			tg.setenv("GOPATH", tg.path("gp"))
			tg.tempFile("go.mod", "module m")
			tg.cd(tg.path("."))
			tg.setenv("GO111MODULE", "on")
			tg.setenv("GOPROXY", "")
		} else {
			tg.setenv("GOPATH", tg.path("."))
			tg.setenv("GO111MODULE", "off")
		}

		const repo = "insecure.go-get-issue-15410.appspot.com/pkg/p"

		// Try go get -d of HTTP-only repo (should fail).
		tg.runFail("get", "-d", repo)

		// Try again with -insecure (should succeed).
		tg.run("get", "-d", "-insecure", repo)

		// Try updating without -insecure (should fail).
		tg.runFail("get", "-d", "-u", "-f", repo)

		if modules {
			tg.run("list", "-m", "...")
			tg.grepStdout("insecure.go-get-issue", "should find insecure module")
		}
	}

	t.Run("gopath", func(t *testing.T) { test(t, false) })
	t.Run("modules", func(t *testing.T) { test(t, true) })
}

func TestGoGetUpdateInsecure(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))

	const repo = "github.com/golang/example"

	// Clone the repo via HTTP manually.
	cmd := exec.Command("git", "clone", "-q", "http://"+repo, tg.path("src/"+repo))
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("cloning %v repo: %v\n%s", repo, err, out)
	}

	// Update without -insecure should fail.
	// Update with -insecure should succeed.
	// We need -f to ignore import comments.
	const pkg = repo + "/hello"
	tg.runFail("get", "-d", "-u", "-f", pkg)
	tg.run("get", "-d", "-u", "-f", "-insecure", pkg)
}

func TestGoGetUpdateUnknownProtocol(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))

	const repo = "github.com/golang/example"

	// Clone the repo via HTTPS manually.
	repoDir := tg.path("src/" + repo)
	cmd := exec.Command("git", "clone", "-q", "https://"+repo, repoDir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("cloning %v repo: %v\n%s", repo, err, out)
	}

	// Configure the repo to use a protocol unknown to cmd/go
	// that still actually works.
	cmd = exec.Command("git", "remote", "set-url", "origin", "xyz://"+repo)
	cmd.Dir = repoDir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git remote set-url: %v\n%s", err, out)
	}
	cmd = exec.Command("git", "config", "--local", "url.https://github.com/.insteadOf", "xyz://github.com/")
	cmd.Dir = repoDir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git config: %v\n%s", err, out)
	}

	// We need -f to ignore import comments.
	tg.run("get", "-d", "-u", "-f", repo+"/hello")
}

func TestGoGetInsecureCustomDomain(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))

	const repo = "insecure.go-get-issue-15410.appspot.com/pkg/p"
	tg.runFail("get", "-d", repo)
	tg.run("get", "-d", "-insecure", repo)
}

func TestGoRunDirs(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.cd("testdata/rundir")
	tg.runFail("run", "x.go", "sub/sub.go")
	tg.grepStderr("named files must all be in one directory; have ./ and sub/", "wrong output")
	tg.runFail("run", "sub/sub.go", "x.go")
	tg.grepStderr("named files must all be in one directory; have sub/ and ./", "wrong output")
}

func TestGoInstallPkgdir(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)

	tg := testgo(t)
	tg.parallel()
	defer tg.cleanup()
	tg.makeTempdir()
	pkg := tg.path(".")
	tg.run("install", "-pkgdir", pkg, "sync")
	tg.mustExist(filepath.Join(pkg, "sync.a"))
	tg.mustNotExist(filepath.Join(pkg, "sync/atomic.a"))
	tg.run("install", "-i", "-pkgdir", pkg, "sync")
	tg.mustExist(filepath.Join(pkg, "sync.a"))
	tg.mustExist(filepath.Join(pkg, "sync/atomic.a"))
}

func TestGoTestRaceInstallCgo(t *testing.T) {
	if !canRace {
		t.Skip("skipping because race detector not supported")
	}

	// golang.org/issue/10500.
	// This used to install a race-enabled cgo.
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("tool", "-n", "cgo")
	cgo := strings.TrimSpace(tg.stdout.String())
	old, err := os.Stat(cgo)
	tg.must(err)
	tg.run("test", "-race", "-i", "runtime/race")
	new, err := os.Stat(cgo)
	tg.must(err)
	if !new.ModTime().Equal(old.ModTime()) {
		t.Fatalf("go test -i runtime/race reinstalled cmd/cgo")
	}
}

func TestGoTestRaceFailures(t *testing.T) {
	tooSlow(t)

	if !canRace {
		t.Skip("skipping because race detector not supported")
	}

	tg := testgo(t)
	tg.parallel()
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	tg.run("test", "testrace")

	tg.runFail("test", "-race", "testrace")
	tg.grepStdout("FAIL: TestRace", "TestRace did not fail")
	tg.grepBothNot("PASS", "something passed")

	tg.runFail("test", "-race", "testrace", "-run", "XXX", "-bench", ".")
	tg.grepStdout("FAIL: BenchmarkRace", "BenchmarkRace did not fail")
	tg.grepBothNot("PASS", "something passed")
}

func TestGoTestImportErrorStack(t *testing.T) {
	const out = `package testdep/p1 (test)
	imports testdep/p2
	imports testdep/p3: build constraints exclude all Go files `

	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("test", "testdep/p1")
	if !strings.Contains(tg.stderr.String(), out) {
		t.Fatalf("did not give full import stack:\n\n%s", tg.stderr.String())
	}
}

func TestGoGetUpdate(t *testing.T) {
	// golang.org/issue/9224.
	// The recursive updating was trying to walk to
	// former dependencies, not current ones.

	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))

	rewind := func() {
		tg.run("get", "github.com/rsc/go-get-issue-9224-cmd")
		cmd := exec.Command("git", "reset", "--hard", "HEAD~")
		cmd.Dir = tg.path("src/github.com/rsc/go-get-issue-9224-lib")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git: %v\n%s", err, out)
		}
	}

	rewind()
	tg.run("get", "-u", "github.com/rsc/go-get-issue-9224-cmd")

	// Again with -d -u.
	rewind()
	tg.run("get", "-d", "-u", "github.com/rsc/go-get-issue-9224-cmd")
}

// Issue #20512.
func TestGoGetRace(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	if !canRace {
		t.Skip("skipping because race detector not supported")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-race", "github.com/rsc/go-get-issue-9224-cmd")
}

func TestGoGetDomainRoot(t *testing.T) {
	// golang.org/issue/9357.
	// go get foo.io (not foo.io/subdir) was not working consistently.

	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))

	// go-get-issue-9357.appspot.com is running
	// the code at github.com/rsc/go-get-issue-9357,
	// a trivial Go on App Engine app that serves a
	// <meta> tag for the domain root.
	tg.run("get", "-d", "go-get-issue-9357.appspot.com")
	tg.run("get", "go-get-issue-9357.appspot.com")
	tg.run("get", "-u", "go-get-issue-9357.appspot.com")

	tg.must(os.RemoveAll(tg.path("src/go-get-issue-9357.appspot.com")))
	tg.run("get", "go-get-issue-9357.appspot.com")

	tg.must(os.RemoveAll(tg.path("src/go-get-issue-9357.appspot.com")))
	tg.run("get", "-u", "go-get-issue-9357.appspot.com")
}

func TestGoInstallShadowedGOPATH(t *testing.T) {
	// golang.org/issue/3652.
	// go get foo.io (not foo.io/subdir) was not working consistently.

	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("gopath1")+string(filepath.ListSeparator)+tg.path("gopath2"))

	tg.tempDir("gopath1/src/test")
	tg.tempDir("gopath2/src/test")
	tg.tempFile("gopath2/src/test/main.go", "package main\nfunc main(){}\n")

	tg.cd(tg.path("gopath2/src/test"))
	tg.runFail("install")
	tg.grepStderr("no install location for.*gopath2.src.test: hidden by .*gopath1.src.test", "missing error")
}

func TestGoBuildGOPATHOrder(t *testing.T) {
	// golang.org/issue/14176#issuecomment-179895769
	// golang.org/issue/14192
	// -I arguments to compiler could end up not in GOPATH order,
	// leading to unexpected import resolution in the compiler.
	// This is still not a complete fix (see golang.org/issue/14271 and next test)
	// but it is clearly OK and enough to fix both of the two reported
	// instances of the underlying problem. It will have to do for now.

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("p1")+string(filepath.ListSeparator)+tg.path("p2"))

	tg.tempFile("p1/src/foo/foo.go", "package foo\n")
	tg.tempFile("p2/src/baz/baz.go", "package baz\n")
	tg.tempFile("p2/pkg/"+runtime.GOOS+"_"+runtime.GOARCH+"/foo.a", "bad\n")
	tg.tempFile("p1/src/bar/bar.go", `
		package bar
		import _ "baz"
		import _ "foo"
	`)

	tg.run("install", "-x", "bar")
}

func TestGoBuildGOPATHOrderBroken(t *testing.T) {
	// This test is known not to work.
	// See golang.org/issue/14271.
	t.Skip("golang.org/issue/14271")

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()

	tg.tempFile("p1/src/foo/foo.go", "package foo\n")
	tg.tempFile("p2/src/baz/baz.go", "package baz\n")
	tg.tempFile("p1/pkg/"+runtime.GOOS+"_"+runtime.GOARCH+"/baz.a", "bad\n")
	tg.tempFile("p2/pkg/"+runtime.GOOS+"_"+runtime.GOARCH+"/foo.a", "bad\n")
	tg.tempFile("p1/src/bar/bar.go", `
		package bar
		import _ "baz"
		import _ "foo"
	`)

	colon := string(filepath.ListSeparator)
	tg.setenv("GOPATH", tg.path("p1")+colon+tg.path("p2"))
	tg.run("install", "-x", "bar")

	tg.setenv("GOPATH", tg.path("p2")+colon+tg.path("p1"))
	tg.run("install", "-x", "bar")
}

func TestIssue11709(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("run.go", `
		package main
		import "os"
		func main() {
			if os.Getenv("TERM") != "" {
				os.Exit(1)
			}
		}`)
	tg.unsetenv("TERM")
	tg.run("run", tg.path("run.go"))
}

func TestIssue12096(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("test_test.go", `
		package main
		import ("os"; "testing")
		func TestEnv(t *testing.T) {
			if os.Getenv("TERM") != "" {
				t.Fatal("TERM is set")
			}
		}`)
	tg.unsetenv("TERM")
	tg.run("test", tg.path("test_test.go"))
}

func TestGoBuildOutput(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()

	tg.makeTempdir()
	tg.cd(tg.path("."))

	nonExeSuffix := ".exe"
	if exeSuffix == ".exe" {
		nonExeSuffix = ""
	}

	tg.tempFile("x.go", "package main\nfunc main(){}\n")
	tg.run("build", "x.go")
	tg.wantExecutable("x"+exeSuffix, "go build x.go did not write x"+exeSuffix)
	tg.must(os.Remove(tg.path("x" + exeSuffix)))
	tg.mustNotExist("x" + nonExeSuffix)

	tg.run("build", "-o", "myprog", "x.go")
	tg.mustNotExist("x")
	tg.mustNotExist("x.exe")
	tg.wantExecutable("myprog", "go build -o myprog x.go did not write myprog")
	tg.mustNotExist("myprog.exe")

	tg.tempFile("p.go", "package p\n")
	tg.run("build", "p.go")
	tg.mustNotExist("p")
	tg.mustNotExist("p.a")
	tg.mustNotExist("p.o")
	tg.mustNotExist("p.exe")

	tg.run("build", "-o", "p.a", "p.go")
	tg.wantArchive("p.a")

	tg.run("build", "cmd/gofmt")
	tg.wantExecutable("gofmt"+exeSuffix, "go build cmd/gofmt did not write gofmt"+exeSuffix)
	tg.must(os.Remove(tg.path("gofmt" + exeSuffix)))
	tg.mustNotExist("gofmt" + nonExeSuffix)

	tg.run("build", "-o", "mygofmt", "cmd/gofmt")
	tg.wantExecutable("mygofmt", "go build -o mygofmt cmd/gofmt did not write mygofmt")
	tg.mustNotExist("mygofmt.exe")
	tg.mustNotExist("gofmt")
	tg.mustNotExist("gofmt.exe")

	tg.run("build", "sync/atomic")
	tg.mustNotExist("atomic")
	tg.mustNotExist("atomic.exe")

	tg.run("build", "-o", "myatomic.a", "sync/atomic")
	tg.wantArchive("myatomic.a")
	tg.mustNotExist("atomic")
	tg.mustNotExist("atomic.a")
	tg.mustNotExist("atomic.exe")

	tg.runFail("build", "-o", "whatever", "cmd/gofmt", "sync/atomic")
	tg.grepStderr("multiple packages", "did not reject -o with multiple packages")
}

func TestGoBuildARM(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping cross-compile in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()

	tg.makeTempdir()
	tg.cd(tg.path("."))

	tg.setenv("GOARCH", "arm")
	tg.setenv("GOOS", "linux")
	tg.setenv("GOARM", "5")
	tg.tempFile("hello.go", `package main
		func main() {}`)
	tg.run("build", "hello.go")
	tg.grepStderrNot("unable to find math.a", "did not build math.a correctly")
}

// For issue 14337.
func TestParallelTest(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	tg.parallel()
	defer tg.cleanup()
	tg.makeTempdir()
	const testSrc = `package package_test
		import (
			"testing"
		)
		func TestTest(t *testing.T) {
		}`
	tg.tempFile("src/p1/p1_test.go", strings.Replace(testSrc, "package_test", "p1_test", 1))
	tg.tempFile("src/p2/p2_test.go", strings.Replace(testSrc, "package_test", "p2_test", 1))
	tg.tempFile("src/p3/p3_test.go", strings.Replace(testSrc, "package_test", "p3_test", 1))
	tg.tempFile("src/p4/p4_test.go", strings.Replace(testSrc, "package_test", "p4_test", 1))
	tg.setenv("GOPATH", tg.path("."))
	tg.run("test", "-p=4", "p1", "p2", "p3", "p4")
}

func TestCgoConsistentResults(t *testing.T) {
	tooSlow(t)
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}
	switch runtime.GOOS {
	case "solaris", "illumos":
		testenv.SkipFlaky(t, 13247)
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	exe1 := tg.path("cgotest1" + exeSuffix)
	exe2 := tg.path("cgotest2" + exeSuffix)
	tg.run("build", "-o", exe1, "cgotest")
	tg.run("build", "-x", "-o", exe2, "cgotest")
	b1, err := ioutil.ReadFile(exe1)
	tg.must(err)
	b2, err := ioutil.ReadFile(exe2)
	tg.must(err)

	if !tg.doGrepMatch(`-fdebug-prefix-map=\$WORK`, &tg.stderr) {
		t.Skip("skipping because C compiler does not support -fdebug-prefix-map")
	}
	if !bytes.Equal(b1, b2) {
		t.Error("building cgotest twice did not produce the same output")
	}
}

// Issue 14444: go get -u .../ duplicate loads errors
func TestGoGetUpdateAllDoesNotTryToLoadDuplicates(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-u", ".../")
	tg.grepStderrNot("duplicate loads of", "did not remove old packages from cache")
}

// Issue 17119 more duplicate load errors
func TestIssue17119(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("build", "dupload")
	tg.grepBothNot("duplicate load|internal error", "internal error")
}

func TestFatalInBenchmarkCauseNonZeroExitStatus(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.runFail("test", "-run", "^$", "-bench", ".", "./testdata/src/benchfatal")
	tg.grepBothNot("^ok", "test passed unexpectedly")
	tg.grepBoth("FAIL.*benchfatal", "test did not run everything")
}

func TestBinaryOnlyPackages(t *testing.T) {
	tooSlow(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))

	tg.tempFile("src/p1/p1.go", `//go:binary-only-package

		package p1
	`)
	tg.wantStale("p1", "binary-only packages are no longer supported", "p1 is binary-only, and this message should always be printed")
	tg.runFail("install", "p1")
	tg.grepStderr("binary-only packages are no longer supported", "did not report attempt to compile binary-only package")

	tg.tempFile("src/p1/p1.go", `
		package p1
		import "fmt"
		func F(b bool) { fmt.Printf("hello from p1\n"); if b { F(false) } }
	`)
	tg.run("install", "p1")
	os.Remove(tg.path("src/p1/p1.go"))
	tg.mustNotExist(tg.path("src/p1/p1.go"))

	tg.tempFile("src/p2/p2.go", `//go:binary-only-packages-are-not-great

		package p2
		import "p1"
		func F() { p1.F(true) }
	`)
	tg.runFail("install", "p2")
	tg.grepStderr("no Go files", "did not complain about missing sources")

	tg.tempFile("src/p1/missing.go", `//go:binary-only-package

		package p1
		import _ "fmt"
		func G()
	`)
	tg.wantStale("p1", "binary-only package", "should NOT want to rebuild p1 (first)")
	tg.runFail("install", "p2")
	tg.grepStderr("p1: binary-only packages are no longer supported", "did not report error for binary-only p1")

	tg.run("list", "-deps", "-f", "{{.ImportPath}}: {{.BinaryOnly}}", "p2")
	tg.grepStdout("p1: true", "p1 not listed as BinaryOnly")
	tg.grepStdout("p2: false", "p2 listed as BinaryOnly")
}

// Issue 16050.
func TestAlwaysLinkSysoFiles(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src/syso")
	tg.tempFile("src/syso/a.syso", ``)
	tg.tempFile("src/syso/b.go", `package syso`)
	tg.setenv("GOPATH", tg.path("."))

	// We should see the .syso file regardless of the setting of
	// CGO_ENABLED.

	tg.setenv("CGO_ENABLED", "1")
	tg.run("list", "-f", "{{.SysoFiles}}", "syso")
	tg.grepStdout("a.syso", "missing syso file with CGO_ENABLED=1")

	tg.setenv("CGO_ENABLED", "0")
	tg.run("list", "-f", "{{.SysoFiles}}", "syso")
	tg.grepStdout("a.syso", "missing syso file with CGO_ENABLED=0")
}

// Issue 16120.
func TestGenerateUsesBuildContext(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("this test won't run under Windows")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src/gen")
	tg.tempFile("src/gen/gen.go", "package gen\n//go:generate echo $GOOS $GOARCH\n")
	tg.setenv("GOPATH", tg.path("."))

	tg.setenv("GOOS", "linux")
	tg.setenv("GOARCH", "amd64")
	tg.run("generate", "gen")
	tg.grepStdout("linux amd64", "unexpected GOOS/GOARCH combination")

	tg.setenv("GOOS", "darwin")
	tg.setenv("GOARCH", "386")
	tg.run("generate", "gen")
	tg.grepStdout("darwin 386", "unexpected GOOS/GOARCH combination")
}

// Issue 14450: go get -u .../ tried to import not downloaded package
func TestGoGetUpdateWithWildcard(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	const aPkgImportPath = "github.com/tmwh/go-get-issue-14450/a"
	tg.run("get", aPkgImportPath)
	tg.runFail("get", "-u", ".../")
	tg.grepStderr("cannot find package.*d-dependency/e", "should have detected e missing")

	// Even though get -u failed, the source for others should be downloaded.
	var expectedPkgPaths = []string{
		"src/github.com/tmwh/go-get-issue-14450/b",
		"src/github.com/tmwh/go-get-issue-14450-b-dependency/c",
		"src/github.com/tmwh/go-get-issue-14450-b-dependency/d",
	}

	for _, importPath := range expectedPkgPaths {
		_, err := os.Stat(tg.path(importPath))
		tg.must(err)
	}
	const notExpectedPkgPath = "src/github.com/tmwh/go-get-issue-14450-c-dependency/e"
	tg.mustNotExist(tg.path(notExpectedPkgPath))
}

func TestGoEnv(t *testing.T) {
	tg := testgo(t)
	tg.parallel()
	defer tg.cleanup()
	tg.setenv("GOOS", "freebsd") // to avoid invalid pair errors
	tg.setenv("GOARCH", "arm")
	tg.run("env", "GOARCH")
	tg.grepStdout("^arm$", "GOARCH not honored")

	tg.run("env", "GCCGO")
	tg.grepStdout(".", "GCCGO unexpectedly empty")

	tg.run("env", "CGO_CFLAGS")
	tg.grepStdout(".", "default CGO_CFLAGS unexpectedly empty")

	tg.setenv("CGO_CFLAGS", "-foobar")
	tg.run("env", "CGO_CFLAGS")
	tg.grepStdout("^-foobar$", "CGO_CFLAGS not honored")

	tg.setenv("CC", "gcc -fmust -fgo -ffaster")
	tg.run("env", "CC")
	tg.grepStdout("gcc", "CC not found")
	tg.run("env", "GOGCCFLAGS")
	tg.grepStdout("-ffaster", "CC arguments not found")
}

const (
	noMatchesPattern = `(?m)^ok.*\[no tests to run\]`
	okPattern        = `(?m)^ok`
)

func TestMatchesNoTests(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.run("test", "-run", "ThisWillNotMatch", "testdata/standalone_test.go")
	tg.grepBoth(noMatchesPattern, "go test did not say [no tests to run]")
}

func TestMatchesNoTestsDoesNotOverrideBuildFailure(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("test", "-run", "ThisWillNotMatch", "syntaxerror")
	tg.grepBothNot(noMatchesPattern, "go test did say [no tests to run]")
	tg.grepBoth("FAIL", "go test did not say FAIL")
}

func TestMatchesNoBenchmarksIsOK(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.run("test", "-run", "^$", "-bench", "ThisWillNotMatch", "testdata/standalone_benchmark_test.go")
	tg.grepBothNot(noMatchesPattern, "go test did say [no tests to run]")
	tg.grepBoth(okPattern, "go test did not say ok")
}

func TestMatchesOnlyExampleIsOK(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.run("test", "-run", "Example", "testdata/example1_test.go")
	tg.grepBothNot(noMatchesPattern, "go test did say [no tests to run]")
	tg.grepBoth(okPattern, "go test did not say ok")
}

func TestMatchesOnlyBenchmarkIsOK(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.run("test", "-run", "^$", "-bench", ".", "testdata/standalone_benchmark_test.go")
	tg.grepBothNot(noMatchesPattern, "go test did say [no tests to run]")
	tg.grepBoth(okPattern, "go test did not say ok")
}

func TestBenchmarkLabels(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("test", "-run", "^$", "-bench", ".", "bench")
	tg.grepStdout(`(?m)^goos: `+runtime.GOOS, "go test did not print goos")
	tg.grepStdout(`(?m)^goarch: `+runtime.GOARCH, "go test did not print goarch")
	tg.grepStdout(`(?m)^pkg: bench`, "go test did not say pkg: bench")
	tg.grepBothNot(`(?s)pkg:.*pkg:`, "go test said pkg multiple times")
}

func TestBenchmarkLabelsOutsideGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.run("test", "-run", "^$", "-bench", ".", "testdata/standalone_benchmark_test.go")
	tg.grepStdout(`(?m)^goos: `+runtime.GOOS, "go test did not print goos")
	tg.grepStdout(`(?m)^goarch: `+runtime.GOARCH, "go test did not print goarch")
	tg.grepBothNot(`(?m)^pkg:`, "go test did say pkg:")
}

func TestMatchesOnlyTestIsOK(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	// TODO: tg.parallel()
	tg.run("test", "-run", "Test", "testdata/standalone_test.go")
	tg.grepBothNot(noMatchesPattern, "go test did say [no tests to run]")
	tg.grepBoth(okPattern, "go test did not say ok")
}

func TestMatchesNoTestsWithSubtests(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-run", "ThisWillNotMatch", "testdata/standalone_sub_test.go")
	tg.grepBoth(noMatchesPattern, "go test did not say [no tests to run]")
}

func TestMatchesNoSubtestsMatch(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-run", "Test/ThisWillNotMatch", "testdata/standalone_sub_test.go")
	tg.grepBoth(noMatchesPattern, "go test did not say [no tests to run]")
}

func TestMatchesNoSubtestsDoesNotOverrideFailure(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("test", "-run", "TestThatFails/ThisWillNotMatch", "testdata/standalone_fail_sub_test.go")
	tg.grepBothNot(noMatchesPattern, "go test did say [no tests to run]")
	tg.grepBoth("FAIL", "go test did not say FAIL")
}

func TestMatchesOnlySubtestIsOK(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-run", "Test/Sub", "testdata/standalone_sub_test.go")
	tg.grepBothNot(noMatchesPattern, "go test did say [no tests to run]")
	tg.grepBoth(okPattern, "go test did not say ok")
}

func TestMatchesNoSubtestsParallel(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-run", "Test/Sub/ThisWillNotMatch", "testdata/standalone_parallel_sub_test.go")
	tg.grepBoth(noMatchesPattern, "go test did not say [no tests to run]")
}

func TestMatchesOnlySubtestParallelIsOK(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-run", "Test/Sub/Nested", "testdata/standalone_parallel_sub_test.go")
	tg.grepBothNot(noMatchesPattern, "go test did say [no tests to run]")
	tg.grepBoth(okPattern, "go test did not say ok")
}

// Issue 18845
func TestBenchTimeout(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-bench", ".", "-timeout", "750ms", "testdata/timeoutbench_test.go")
}

// Issue 19394
func TestWriteProfilesOnTimeout(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempDir("profiling")
	tg.tempFile("profiling/timeouttest_test.go", `package timeouttest_test
import "testing"
import "time"
func TestSleep(t *testing.T) { time.Sleep(time.Second) }`)
	tg.cd(tg.path("profiling"))
	tg.runFail(
		"test",
		"-cpuprofile", tg.path("profiling/cpu.pprof"), "-memprofile", tg.path("profiling/mem.pprof"),
		"-timeout", "1ms")
	tg.mustHaveContent(tg.path("profiling/cpu.pprof"))
	tg.mustHaveContent(tg.path("profiling/mem.pprof"))
}

func TestLinkXImportPathEscape(t *testing.T) {
	// golang.org/issue/16710
	skipIfGccgo(t, "gccgo does not support -ldflags -X")
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	exe := "./linkx" + exeSuffix
	tg.creatingTemp(exe)
	tg.run("build", "-o", exe, "-ldflags", "-X=my.pkg.Text=linkXworked", "my.pkg/main")
	out, err := exec.Command(exe).CombinedOutput()
	if err != nil {
		tg.t.Fatal(err)
	}
	if string(out) != "linkXworked\n" {
		tg.t.Log(string(out))
		tg.t.Fatal(`incorrect output: expected "linkXworked\n"`)
	}
}

// Issue 18044.
func TestLdBindNow(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("LD_BIND_NOW", "1")
	tg.run("help")
}

// Issue 18225.
// This is really a cmd/asm issue but this is a convenient place to test it.
func TestConcurrentAsm(t *testing.T) {
	skipIfGccgo(t, "gccgo does not use cmd/asm")
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	asm := `DATA ·constants<>+0x0(SB)/8,$0
GLOBL ·constants<>(SB),8,$8
`
	tg.tempFile("go/src/p/a.s", asm)
	tg.tempFile("go/src/p/b.s", asm)
	tg.tempFile("go/src/p/p.go", `package p`)
	tg.setenv("GOPATH", tg.path("go"))
	tg.run("build", "p")
}

// Issue 18778.
func TestDotDotDotOutsideGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()

	tg.tempFile("pkgs/a.go", `package x`)
	tg.tempFile("pkgs/a_test.go", `package x_test
import "testing"
func TestX(t *testing.T) {}`)

	tg.tempFile("pkgs/a/a.go", `package a`)
	tg.tempFile("pkgs/a/a_test.go", `package a_test
import "testing"
func TestA(t *testing.T) {}`)

	tg.cd(tg.path("pkgs"))
	tg.run("build", "./...")
	tg.run("test", "./...")
	tg.run("list", "./...")
	tg.grepStdout("pkgs$", "expected package not listed")
	tg.grepStdout("pkgs/a", "expected package not listed")
}

// Issue 18975.
func TestFFLAGS(t *testing.T) {
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	tg.tempFile("p/src/p/main.go", `package main
		// #cgo FFLAGS: -no-such-fortran-flag
		import "C"
		func main() {}
	`)
	tg.tempFile("p/src/p/a.f", `! comment`)
	tg.setenv("GOPATH", tg.path("p"))

	// This should normally fail because we are passing an unknown flag,
	// but issue #19080 points to Fortran compilers that succeed anyhow.
	// To work either way we call doRun directly rather than run or runFail.
	tg.doRun([]string{"build", "-x", "p"})

	tg.grepStderr("no-such-fortran-flag", `missing expected "-no-such-fortran-flag"`)
}

// Issue 19198.
// This is really a cmd/link issue but this is a convenient place to test it.
func TestDuplicateGlobalAsmSymbols(t *testing.T) {
	skipIfGccgo(t, "gccgo does not use cmd/asm")
	tooSlow(t)
	if runtime.GOARCH != "386" && runtime.GOARCH != "amd64" {
		t.Skipf("skipping test on %s", runtime.GOARCH)
	}
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	asm := `
#include "textflag.h"

DATA sym<>+0x0(SB)/8,$0
GLOBL sym<>(SB),(NOPTR+RODATA),$8

TEXT ·Data(SB),NOSPLIT,$0
	MOVB sym<>(SB), AX
	MOVB AX, ret+0(FP)
	RET
`
	tg.tempFile("go/src/a/a.s", asm)
	tg.tempFile("go/src/a/a.go", `package a; func Data() uint8`)
	tg.tempFile("go/src/b/b.s", asm)
	tg.tempFile("go/src/b/b.go", `package b; func Data() uint8`)
	tg.tempFile("go/src/p/p.go", `
package main
import "a"
import "b"
import "C"
func main() {
	_ = a.Data() + b.Data()
}
`)
	tg.setenv("GOPATH", tg.path("go"))
	exe := tg.path("p.exe")
	tg.creatingTemp(exe)
	tg.run("build", "-o", exe, "p")
}

func TestBuildTagsNoComma(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("go"))
	tg.run("build", "-tags", "tag1 tag2", "math")
	tg.runFail("build", "-tags", "tag1,tag2 tag3", "math")
	tg.grepBoth("space-separated list contains comma", "-tags with a comma-separated list didn't error")
}

func copyFile(src, dst string, perm os.FileMode) error {
	sf, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sf.Close()

	df, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}

	_, err = io.Copy(df, sf)
	err2 := df.Close()
	if err != nil {
		return err
	}
	return err2
}

func TestExecutableGOROOT(t *testing.T) {
	skipIfGccgo(t, "gccgo has no GOROOT")
	if runtime.GOOS == "openbsd" {
		t.Skipf("test case does not work on %s, missing os.Executable", runtime.GOOS)
	}

	// Env with no GOROOT.
	var env []string
	for _, e := range os.Environ() {
		if !strings.HasPrefix(e, "GOROOT=") {
			env = append(env, e)
		}
	}

	check := func(t *testing.T, exe, want string) {
		cmd := exec.Command(exe, "env", "GOROOT")
		cmd.Env = env
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s env GOROOT: %v, %s", exe, err, out)
		}
		goroot, err := filepath.EvalSymlinks(strings.TrimSpace(string(out)))
		if err != nil {
			t.Fatal(err)
		}
		want, err = filepath.EvalSymlinks(want)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.EqualFold(goroot, want) {
			t.Errorf("go env GOROOT:\nhave %s\nwant %s", goroot, want)
		} else {
			t.Logf("go env GOROOT: %s", goroot)
		}
	}

	// Note: Must not call tg methods inside subtests: tg is attached to outer t.
	tg := testgo(t)
	defer tg.cleanup()

	tg.makeTempdir()
	tg.tempDir("new/bin")
	newGoTool := tg.path("new/bin/go" + exeSuffix)
	tg.must(copyFile(tg.goTool(), newGoTool, 0775))
	newRoot := tg.path("new")

	t.Run("RelocatedExe", func(t *testing.T) {
		// Should fall back to default location in binary,
		// which is the GOROOT we used when building testgo.exe.
		check(t, newGoTool, testGOROOT)
	})

	// If the binary is sitting in a bin dir next to ../pkg/tool, that counts as a GOROOT,
	// so it should find the new tree.
	tg.tempDir("new/pkg/tool")
	t.Run("RelocatedTree", func(t *testing.T) {
		check(t, newGoTool, newRoot)
	})

	tg.tempDir("other/bin")
	symGoTool := tg.path("other/bin/go" + exeSuffix)

	// Symlink into go tree should still find go tree.
	t.Run("SymlinkedExe", func(t *testing.T) {
		testenv.MustHaveSymlink(t)
		if err := os.Symlink(newGoTool, symGoTool); err != nil {
			t.Fatal(err)
		}
		check(t, symGoTool, newRoot)
	})

	tg.must(os.RemoveAll(tg.path("new/pkg")))

	// Binaries built in the new tree should report the
	// new tree when they call runtime.GOROOT.
	t.Run("RuntimeGoroot", func(t *testing.T) {
		// Build a working GOROOT the easy way, with symlinks.
		testenv.MustHaveSymlink(t)
		if err := os.Symlink(filepath.Join(testGOROOT, "src"), tg.path("new/src")); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join(testGOROOT, "pkg"), tg.path("new/pkg")); err != nil {
			t.Fatal(err)
		}

		cmd := exec.Command(newGoTool, "run", "testdata/print_goroot.go")
		cmd.Env = env
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("%s run testdata/print_goroot.go: %v, %s", newGoTool, err, out)
		}
		goroot, err := filepath.EvalSymlinks(strings.TrimSpace(string(out)))
		if err != nil {
			t.Fatal(err)
		}
		want, err := filepath.EvalSymlinks(tg.path("new"))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.EqualFold(goroot, want) {
			t.Errorf("go run testdata/print_goroot.go:\nhave %s\nwant %s", goroot, want)
		} else {
			t.Logf("go run testdata/print_goroot.go: %s", goroot)
		}
	})
}

func TestNeedVersion(t *testing.T) {
	skipIfGccgo(t, "gccgo does not use cmd/compile")
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("goversion.go", `package main; func main() {}`)
	path := tg.path("goversion.go")
	tg.setenv("TESTGO_VERSION", "go1.testgo")
	tg.runFail("run", path)
	tg.grepStderr("compile", "does not match go tool version")
}

// Test that user can override default code generation flags.
func TestUserOverrideFlags(t *testing.T) {
	skipIfGccgo(t, "gccgo does not use -gcflags")
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}
	if runtime.GOOS != "linux" {
		// We are testing platform-independent code, so it's
		// OK to skip cases that work differently.
		t.Skipf("skipping on %s because test only works if c-archive implies -shared", runtime.GOOS)
	}

	tg := testgo(t)
	defer tg.cleanup()
	// Don't call tg.parallel, as creating override.h and override.a may
	// confuse other tests.
	tg.tempFile("override.go", `package main

import "C"

//export GoFunc
func GoFunc() {}

func main() {}`)
	tg.creatingTemp("override.a")
	tg.creatingTemp("override.h")
	tg.run("build", "-x", "-buildmode=c-archive", "-gcflags=all=-shared=false", tg.path("override.go"))
	tg.grepStderr("compile .*-shared .*-shared=false", "user can not override code generation flag")
}

func TestCgoFlagContainsSpace(t *testing.T) {
	tooSlow(t)
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}
	tg := testgo(t)
	defer tg.cleanup()

	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.tempFile("main.go", `package main
		// #cgo CFLAGS: -I"c flags"
		// #cgo LDFLAGS: -L"ld flags"
		import "C"
		func main() {}
	`)
	tg.run("run", "-x", "main.go")
	tg.grepStderr(`"-I[^"]+c flags"`, "did not find quoted c flags")
	tg.grepStderrNot(`"-I[^"]+c flags".*"-I[^"]+c flags"`, "found too many quoted c flags")
	tg.grepStderr(`"-L[^"]+ld flags"`, "did not find quoted ld flags")
	tg.grepStderrNot(`"-L[^"]+c flags".*"-L[^"]+c flags"`, "found too many quoted ld flags")
}

// Issue #20435.
func TestGoTestRaceCoverModeFailures(t *testing.T) {
	tooSlow(t)
	if !canRace {
		t.Skip("skipping because race detector not supported")
	}

	tg := testgo(t)
	tg.parallel()
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	tg.run("test", "testrace")

	tg.runFail("test", "-race", "-covermode=set", "testrace")
	tg.grepStderr(`-covermode must be "atomic", not "set", when -race is enabled`, "-race -covermode=set was allowed")
	tg.grepBothNot("PASS", "something passed")
}

// Issue 9737: verify that GOARM and GO386 affect the computed build ID.
func TestBuildIDContainsArchModeEnv(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	var tg *testgoData
	testWith := func(before, after func()) func(*testing.T) {
		return func(t *testing.T) {
			tg = testgo(t)
			defer tg.cleanup()
			tg.tempFile("src/mycmd/x.go", `package main
func main() {}`)
			tg.setenv("GOPATH", tg.path("."))

			tg.cd(tg.path("src/mycmd"))
			tg.setenv("GOOS", "linux")
			before()
			tg.run("install", "mycmd")
			after()
			tg.wantStale("mycmd", "stale dependency", "should be stale after environment variable change")
		}
	}

	t.Run("386", testWith(func() {
		tg.setenv("GOARCH", "386")
		tg.setenv("GO386", "387")
	}, func() {
		tg.setenv("GO386", "sse2")
	}))

	t.Run("arm", testWith(func() {
		tg.setenv("GOARCH", "arm")
		tg.setenv("GOARM", "5")
	}, func() {
		tg.setenv("GOARM", "7")
	}))
}

func TestTestRegexps(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("test", "-cpu=1", "-run=X/Y", "-bench=X/Y", "-count=2", "-v", "testregexp")
	var lines []string
	for _, line := range strings.SplitAfter(tg.getStdout(), "\n") {
		if strings.Contains(line, "=== RUN") || strings.Contains(line, "--- BENCH") || strings.Contains(line, "LOG") {
			lines = append(lines, line)
		}
	}

	// Important parts:
	//	TestX is run, twice
	//	TestX/Y is run, twice
	//	TestXX is run, twice
	//	TestZ is not run
	//	BenchmarkX is run but only with N=1, once
	//	BenchmarkXX is run but only with N=1, once
	//	BenchmarkX/Y is run in full, twice
	want := `=== RUN   TestX
=== RUN   TestX/Y
    x_test.go:6: LOG: X running
        x_test.go:8: LOG: Y running
=== RUN   TestXX
    z_test.go:10: LOG: XX running
=== RUN   TestX
=== RUN   TestX/Y
    x_test.go:6: LOG: X running
        x_test.go:8: LOG: Y running
=== RUN   TestXX
    z_test.go:10: LOG: XX running
--- BENCH: BenchmarkX/Y
    x_test.go:15: LOG: Y running N=1
    x_test.go:15: LOG: Y running N=100
    x_test.go:15: LOG: Y running N=10000
    x_test.go:15: LOG: Y running N=1000000
    x_test.go:15: LOG: Y running N=100000000
    x_test.go:15: LOG: Y running N=1000000000
--- BENCH: BenchmarkX/Y
    x_test.go:15: LOG: Y running N=1
    x_test.go:15: LOG: Y running N=100
    x_test.go:15: LOG: Y running N=10000
    x_test.go:15: LOG: Y running N=1000000
    x_test.go:15: LOG: Y running N=100000000
    x_test.go:15: LOG: Y running N=1000000000
--- BENCH: BenchmarkX
    x_test.go:13: LOG: X running N=1
--- BENCH: BenchmarkXX
    z_test.go:18: LOG: XX running N=1
`

	have := strings.Join(lines, "")
	if have != want {
		t.Errorf("reduced output:<<<\n%s>>> want:<<<\n%s>>>", have, want)
	}
}

func TestListTests(t *testing.T) {
	tooSlow(t)
	var tg *testgoData
	testWith := func(listName, expected string) func(*testing.T) {
		return func(t *testing.T) {
			tg = testgo(t)
			defer tg.cleanup()
			tg.run("test", "./testdata/src/testlist/...", fmt.Sprintf("-list=%s", listName))
			tg.grepStdout(expected, fmt.Sprintf("-test.list=%s returned %q, expected %s", listName, tg.getStdout(), expected))
		}
	}

	t.Run("Test", testWith("Test", "TestSimple"))
	t.Run("Bench", testWith("Benchmark", "BenchmarkSimple"))
	t.Run("Example1", testWith("Example", "ExampleSimple"))
	t.Run("Example2", testWith("Example", "ExampleWithEmptyOutput"))
}

func TestBuildmodePIE(t *testing.T) {
	if testing.Short() && testenv.Builder() == "" {
		t.Skipf("skipping in -short mode on non-builder")
	}

	platform := fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH)
	switch platform {
	case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/ppc64le", "linux/s390x",
		"android/amd64", "android/arm", "android/arm64", "android/386",
		"freebsd/amd64":
	case "darwin/amd64":
	default:
		t.Skipf("skipping test because buildmode=pie is not supported on %s", platform)
	}

	tg := testgo(t)
	defer tg.cleanup()

	tg.tempFile("main.go", `package main; func main() { print("hello") }`)
	src := tg.path("main.go")
	obj := tg.path("main")
	tg.run("build", "-buildmode=pie", "-o", obj, src)

	switch runtime.GOOS {
	case "linux", "android", "freebsd":
		f, err := elf.Open(obj)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if f.Type != elf.ET_DYN {
			t.Errorf("PIE type must be ET_DYN, but %s", f.Type)
		}
	case "darwin":
		f, err := macho.Open(obj)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if f.Flags&macho.FlagDyldLink == 0 {
			t.Error("PIE must have DyldLink flag, but not")
		}
		if f.Flags&macho.FlagPIE == 0 {
			t.Error("PIE must have PIE flag, but not")
		}
	default:
		panic("unreachable")
	}

	out, err := exec.Command(obj).CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}

	if string(out) != "hello" {
		t.Errorf("got %q; want %q", out, "hello")
	}
}

func TestExecBuildX(t *testing.T) {
	tooSlow(t)
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	if runtime.GOOS == "plan9" || runtime.GOOS == "windows" {
		t.Skipf("skipping because unix shell is not supported on %s", runtime.GOOS)
	}

	tg := testgo(t)
	defer tg.cleanup()

	tg.tempDir("cache")
	tg.setenv("GOCACHE", tg.path("cache"))

	tg.tempFile("main.go", `package main; import "C"; func main() { print("hello") }`)
	src := tg.path("main.go")
	obj := tg.path("main")
	tg.run("build", "-x", "-o", obj, src)
	sh := tg.path("test.sh")
	cmds := tg.getStderr()
	err := ioutil.WriteFile(sh, []byte("set -e\n"+cmds), 0666)
	if err != nil {
		t.Fatal(err)
	}

	out, err := exec.Command(obj).CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != "hello" {
		t.Fatalf("got %q; want %q", out, "hello")
	}

	err = os.Remove(obj)
	if err != nil {
		t.Fatal(err)
	}

	out, err = exec.Command("/usr/bin/env", "bash", "-x", sh).CombinedOutput()
	if err != nil {
		t.Fatalf("/bin/sh %s: %v\n%s", sh, err, out)
	}
	t.Logf("shell output:\n%s", out)

	out, err = exec.Command(obj).CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != "hello" {
		t.Fatalf("got %q; want %q", out, "hello")
	}

	matches := regexp.MustCompile(`^WORK=(.*)\n`).FindStringSubmatch(cmds)
	if len(matches) == 0 {
		t.Fatal("no WORK directory")
	}
	tg.must(os.RemoveAll(matches[1]))
}

func TestParallelNumber(t *testing.T) {
	tooSlow(t)
	for _, n := range [...]string{"-1", "0"} {
		t.Run(n, func(t *testing.T) {
			tg := testgo(t)
			defer tg.cleanup()
			tg.runFail("test", "-parallel", n, "testdata/standalone_parallel_sub_test.go")
			tg.grepBoth("-parallel can only be given", "go test -parallel with N<1 did not error")
		})
	}
}

func TestWrongGOOSErrorBeforeLoadError(t *testing.T) {
	skipIfGccgo(t, "gccgo assumes cross-compilation is always possible")
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("GOOS", "windwos")
	tg.runFail("build", "exclude")
	tg.grepStderr("unsupported GOOS/GOARCH pair", "GOOS=windwos go build exclude did not report 'unsupported GOOS/GOARCH pair'")
}

func TestUpxCompression(t *testing.T) {
	if runtime.GOOS != "linux" ||
		(runtime.GOARCH != "amd64" && runtime.GOARCH != "386") {
		t.Skipf("skipping upx test on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	out, err := exec.Command("upx", "--version").CombinedOutput()
	if err != nil {
		t.Skip("skipping because upx is not available")
	}

	// upx --version prints `upx <version>` in the first line of output:
	//   upx 3.94
	//   [...]
	re := regexp.MustCompile(`([[:digit:]]+)\.([[:digit:]]+)`)
	upxVersion := re.FindStringSubmatch(string(out))
	if len(upxVersion) != 3 {
		t.Errorf("bad upx version string: %s", upxVersion)
	}

	major, err1 := strconv.Atoi(upxVersion[1])
	minor, err2 := strconv.Atoi(upxVersion[2])
	if err1 != nil || err2 != nil {
		t.Errorf("bad upx version string: %s", upxVersion[0])
	}

	// Anything below 3.94 is known not to work with go binaries
	if (major < 3) || (major == 3 && minor < 94) {
		t.Skipf("skipping because upx version %v.%v is too old", major, minor)
	}

	tg := testgo(t)
	defer tg.cleanup()

	tg.tempFile("main.go", `package main; import "fmt"; func main() { fmt.Print("hello upx") }`)
	src := tg.path("main.go")
	obj := tg.path("main")
	tg.run("build", "-o", obj, src)

	out, err = exec.Command("upx", obj).CombinedOutput()
	if err != nil {
		t.Logf("executing upx\n%s\n", out)
		t.Fatalf("upx failed with %v", err)
	}

	out, err = exec.Command(obj).CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Fatalf("running compressed go binary failed with error %s", err)
	}
	if string(out) != "hello upx" {
		t.Fatalf("bad output from compressed go binary:\ngot %q; want %q", out, "hello upx")
	}
}

// Test that Go binaries can be run under QEMU in user-emulation mode
// (See issue #13024).
func TestQEMUUserMode(t *testing.T) {
	if testing.Short() && testenv.Builder() == "" {
		t.Skipf("skipping in -short mode on non-builder")
	}

	testArchs := []struct {
		g, qemu string
	}{
		{"arm", "arm"},
		{"arm64", "aarch64"},
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("main.go", `package main; import "fmt"; func main() { fmt.Print("hello qemu-user") }`)
	tg.parallel()
	src, obj := tg.path("main.go"), tg.path("main")

	for _, arch := range testArchs {
		out, err := exec.Command("qemu-"+arch.qemu, "--version").CombinedOutput()
		if err != nil {
			t.Logf("Skipping %s test (qemu-%s not available)", arch.g, arch.qemu)
			continue
		}

		tg.setenv("GOARCH", arch.g)
		tg.run("build", "-o", obj, src)

		out, err = exec.Command("qemu-"+arch.qemu, obj).CombinedOutput()
		if err != nil {
			t.Logf("qemu-%s output:\n%s\n", arch.qemu, out)
			t.Errorf("qemu-%s failed with %v", arch.qemu, err)
			continue
		}
		if want := "hello qemu-user"; string(out) != want {
			t.Errorf("bad output from qemu-%s:\ngot %s; want %s", arch.qemu, out, want)
		}
	}

}

func TestCacheListStale(t *testing.T) {
	tooSlow(t)
	if strings.Contains(os.Getenv("GODEBUG"), "gocacheverify") {
		t.Skip("GODEBUG gocacheverify")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOCACHE", tg.path("cache"))
	tg.tempFile("gopath/src/p/p.go", "package p; import _ \"q\"; func F(){}\n")
	tg.tempFile("gopath/src/q/q.go", "package q; func F(){}\n")
	tg.tempFile("gopath/src/m/m.go", "package main; import _ \"q\"; func main(){}\n")

	tg.setenv("GOPATH", tg.path("gopath"))
	tg.run("install", "p", "m")
	tg.run("list", "-f={{.ImportPath}} {{.Stale}}", "m", "q", "p")
	tg.grepStdout("^m false", "m should not be stale")
	tg.grepStdout("^q true", "q should be stale")
	tg.grepStdout("^p false", "p should not be stale")
}

func TestCacheCoverage(t *testing.T) {
	tooSlow(t)

	if strings.Contains(os.Getenv("GODEBUG"), "gocacheverify") {
		t.Skip("GODEBUG gocacheverify")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.makeTempdir()

	tg.setenv("GOCACHE", tg.path("c1"))
	tg.run("test", "-cover", "-short", "strings")
	tg.run("test", "-cover", "-short", "math", "strings")
}

func TestCacheVet(t *testing.T) {
	skipIfGccgo(t, "gccgo has no standard packages")
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	if strings.Contains(os.Getenv("GODEBUG"), "gocacheverify") {
		t.Skip("GODEBUG gocacheverify")
	}
	if testing.Short() {
		// In short mode, reuse cache.
		// Test failures may be masked if the cache has just the right entries already
		// (not a concern during all.bash, which runs in a clean cache).
		if cfg.Getenv("GOCACHE") == "off" {
			tooSlow(t)
		}
	} else {
		tg.makeTempdir()
		tg.setenv("GOCACHE", tg.path("cache"))
	}

	// Check that second vet reuses cgo-derived inputs.
	// The first command could be build instead of vet,
	// except that if the cache is empty and there's a net.a
	// in GOROOT/pkg, the build will not bother to regenerate
	// and cache the cgo outputs, whereas vet always will.
	tg.run("vet", "os/user")
	tg.run("vet", "-x", "os/user")
	tg.grepStderrNot(`^(clang|gcc)`, "should not have run compiler")
	tg.grepStderrNot(`[\\/]cgo `, "should not have run cgo")
}

func TestIssue22588(t *testing.T) {
	// Don't get confused by stderr coming from tools.
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	if _, err := os.Stat("/usr/bin/time"); err != nil {
		t.Skip(err)
	}

	tg.run("list", "-f={{.Stale}}", "runtime")
	tg.run("list", "-toolexec=/usr/bin/time", "-f={{.Stale}}", "runtime")
	tg.grepStdout("false", "incorrectly reported runtime as stale")
}

func TestIssue22531(t *testing.T) {
	tooSlow(t)
	if strings.Contains(os.Getenv("GODEBUG"), "gocacheverify") {
		t.Skip("GODEBUG gocacheverify")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.tempdir)
	tg.setenv("GOCACHE", tg.path("cache"))
	tg.tempFile("src/m/main.go", "package main /* c1 */; func main() {}\n")
	tg.run("install", "-x", "m")
	tg.run("list", "-f", "{{.Stale}}", "m")
	tg.grepStdout("false", "reported m as stale after install")
	tg.run("tool", "buildid", tg.path("bin/m"+exeSuffix))

	// The link action ID did not include the full main build ID,
	// even though the full main build ID is written into the
	// eventual binary. That caused the following install to
	// be a no-op, thinking the gofmt binary was up-to-date,
	// even though .Stale could see it was not.
	tg.tempFile("src/m/main.go", "package main /* c2 */; func main() {}\n")
	tg.run("install", "-x", "m")
	tg.run("list", "-f", "{{.Stale}}", "m")
	tg.grepStdout("false", "reported m as stale after reinstall")
	tg.run("tool", "buildid", tg.path("bin/m"+exeSuffix))
}

func TestIssue22596(t *testing.T) {
	tooSlow(t)
	if strings.Contains(os.Getenv("GODEBUG"), "gocacheverify") {
		t.Skip("GODEBUG gocacheverify")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOCACHE", tg.path("cache"))
	tg.tempFile("gopath1/src/p/p.go", "package p; func F(){}\n")
	tg.tempFile("gopath2/src/p/p.go", "package p; func F(){}\n")

	tg.setenv("GOPATH", tg.path("gopath1"))
	tg.run("list", "-f={{.Target}}", "p")
	target1 := strings.TrimSpace(tg.getStdout())
	tg.run("install", "p")
	tg.wantNotStale("p", "", "p stale after install")

	tg.setenv("GOPATH", tg.path("gopath2"))
	tg.run("list", "-f={{.Target}}", "p")
	target2 := strings.TrimSpace(tg.getStdout())
	tg.must(os.MkdirAll(filepath.Dir(target2), 0777))
	tg.must(copyFile(target1, target2, 0666))
	tg.wantStale("p", "build ID mismatch", "p not stale after copy from gopath1")
	tg.run("install", "p")
	tg.wantNotStale("p", "", "p stale after install2")
}

func TestTestCache(t *testing.T) {
	tooSlow(t)

	if strings.Contains(os.Getenv("GODEBUG"), "gocacheverify") {
		t.Skip("GODEBUG gocacheverify")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.tempdir)
	tg.setenv("GOCACHE", tg.path("cache"))

	if runtime.Compiler != "gccgo" {
		// timeout here should not affect result being cached
		// or being retrieved later.
		tg.run("test", "-x", "-timeout=10s", "errors")
		tg.grepStderr(`[\\/]compile|gccgo`, "did not run compiler")
		tg.grepStderr(`[\\/]link|gccgo`, "did not run linker")
		tg.grepStderr(`errors\.test`, "did not run test")

		tg.run("test", "-x", "errors")
		tg.grepStdout(`ok  \terrors\t\(cached\)`, "did not report cached result")
		tg.grepStderrNot(`[\\/]compile|gccgo`, "incorrectly ran compiler")
		tg.grepStderrNot(`[\\/]link|gccgo`, "incorrectly ran linker")
		tg.grepStderrNot(`errors\.test`, "incorrectly ran test")
		tg.grepStderrNot("DO NOT USE", "poisoned action status leaked")

		// Even very low timeouts do not disqualify cached entries.
		tg.run("test", "-timeout=1ns", "-x", "errors")
		tg.grepStderrNot(`errors\.test`, "incorrectly ran test")

		tg.run("clean", "-testcache")
		tg.run("test", "-x", "errors")
		tg.grepStderr(`errors\.test`, "did not run test")
	}

	// The -p=1 in the commands below just makes the -x output easier to read.

	t.Log("\n\nINITIAL\n\n")

	tg.tempFile("src/p1/p1.go", "package p1\nvar X =  1\n")
	tg.tempFile("src/p2/p2.go", "package p2\nimport _ \"p1\"\nvar X = 1\n")
	tg.tempFile("src/t/t1/t1_test.go", "package t\nimport \"testing\"\nfunc Test1(*testing.T) {}\n")
	tg.tempFile("src/t/t2/t2_test.go", "package t\nimport _ \"p1\"\nimport \"testing\"\nfunc Test2(*testing.T) {}\n")
	tg.tempFile("src/t/t3/t3_test.go", "package t\nimport \"p1\"\nimport \"testing\"\nfunc Test3(t *testing.T) {t.Log(p1.X)}\n")
	tg.tempFile("src/t/t4/t4_test.go", "package t\nimport \"p2\"\nimport \"testing\"\nfunc Test4(t *testing.T) {t.Log(p2.X)}")
	tg.run("test", "-x", "-v", "-short", "t/...")

	t.Log("\n\nREPEAT\n\n")

	tg.run("test", "-x", "-v", "-short", "t/...")
	tg.grepStdout(`ok  \tt/t1\t\(cached\)`, "did not cache t1")
	tg.grepStdout(`ok  \tt/t2\t\(cached\)`, "did not cache t2")
	tg.grepStdout(`ok  \tt/t3\t\(cached\)`, "did not cache t3")
	tg.grepStdout(`ok  \tt/t4\t\(cached\)`, "did not cache t4")
	tg.grepStderrNot(`[\\/](compile|gccgo) `, "incorrectly ran compiler")
	tg.grepStderrNot(`[\\/](link|gccgo) `, "incorrectly ran linker")
	tg.grepStderrNot(`p[0-9]\.test`, "incorrectly ran test")

	t.Log("\n\nCOMMENT\n\n")

	// Changing the program text without affecting the compiled package
	// should result in the package being rebuilt but nothing more.
	tg.tempFile("src/p1/p1.go", "package p1\nvar X = 01\n")
	tg.run("test", "-p=1", "-x", "-v", "-short", "t/...")
	tg.grepStdout(`ok  \tt/t1\t\(cached\)`, "did not cache t1")
	tg.grepStdout(`ok  \tt/t2\t\(cached\)`, "did not cache t2")
	tg.grepStdout(`ok  \tt/t3\t\(cached\)`, "did not cache t3")
	tg.grepStdout(`ok  \tt/t4\t\(cached\)`, "did not cache t4")
	tg.grepStderrNot(`([\\/](compile|gccgo) ).*t[0-9]_test\.go`, "incorrectly ran compiler")
	tg.grepStderrNot(`[\\/](link|gccgo) `, "incorrectly ran linker")
	tg.grepStderrNot(`t[0-9]\.test.*test\.short`, "incorrectly ran test")

	t.Log("\n\nCHANGE\n\n")

	// Changing the actual package should have limited effects.
	tg.tempFile("src/p1/p1.go", "package p1\nvar X = 02\n")
	tg.run("test", "-p=1", "-x", "-v", "-short", "t/...")

	// p2 should have been rebuilt.
	tg.grepStderr(`([\\/]compile|gccgo).*p2.go`, "did not recompile p2")

	// t1 does not import anything, should not have been rebuilt.
	tg.grepStderrNot(`([\\/]compile|gccgo).*t1_test.go`, "incorrectly recompiled t1")
	tg.grepStderrNot(`([\\/]link|gccgo).*t1_test`, "incorrectly relinked t1_test")
	tg.grepStdout(`ok  \tt/t1\t\(cached\)`, "did not cache t/t1")

	// t2 imports p1 and must be rebuilt and relinked,
	// but the change should not have any effect on the test binary,
	// so the test should not have been rerun.
	tg.grepStderr(`([\\/]compile|gccgo).*t2_test.go`, "did not recompile t2")
	tg.grepStderr(`([\\/]link|gccgo).*t2\.test`, "did not relink t2_test")
	// This check does not currently work with gccgo, as garbage
	// collection of unused variables is not turned on by default.
	if runtime.Compiler != "gccgo" {
		tg.grepStdout(`ok  \tt/t2\t\(cached\)`, "did not cache t/t2")
	}

	// t3 imports p1, and changing X changes t3's test binary.
	tg.grepStderr(`([\\/]compile|gccgo).*t3_test.go`, "did not recompile t3")
	tg.grepStderr(`([\\/]link|gccgo).*t3\.test`, "did not relink t3_test")
	tg.grepStderr(`t3\.test.*-test.short`, "did not rerun t3_test")
	tg.grepStdoutNot(`ok  \tt/t3\t\(cached\)`, "reported cached t3_test result")

	// t4 imports p2, but p2 did not change, so t4 should be relinked, not recompiled,
	// and not rerun.
	tg.grepStderrNot(`([\\/]compile|gccgo).*t4_test.go`, "incorrectly recompiled t4")
	tg.grepStderr(`([\\/]link|gccgo).*t4\.test`, "did not relink t4_test")
	// This check does not currently work with gccgo, as garbage
	// collection of unused variables is not turned on by default.
	if runtime.Compiler != "gccgo" {
		tg.grepStdout(`ok  \tt/t4\t\(cached\)`, "did not cache t/t4")
	}
}

func TestTestCacheInputs(t *testing.T) {
	tooSlow(t)

	if strings.Contains(os.Getenv("GODEBUG"), "gocacheverify") {
		t.Skip("GODEBUG gocacheverify")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.setenv("GOCACHE", tg.path("cache"))

	defer os.Remove(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"))
	defer os.Remove(filepath.Join(tg.pwd(), "testdata/src/testcache/script.sh"))
	tg.must(ioutil.WriteFile(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"), []byte("x"), 0644))
	old := time.Now().Add(-1 * time.Minute)
	tg.must(os.Chtimes(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"), old, old))
	info, err := os.Stat(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("file.txt: old=%v, info.ModTime=%v", old, info.ModTime()) // help debug when Chtimes lies about succeeding
	tg.setenv("TESTKEY", "x")

	tg.must(ioutil.WriteFile(filepath.Join(tg.pwd(), "testdata/src/testcache/script.sh"), []byte("#!/bin/sh\nexit 0\n"), 0755))
	tg.must(os.Chtimes(filepath.Join(tg.pwd(), "testdata/src/testcache/script.sh"), old, old))

	tg.run("test", "testcache")
	tg.run("test", "testcache")
	tg.grepStdout(`\(cached\)`, "did not cache")

	tg.setenv("TESTKEY", "y")
	tg.run("test", "testcache")
	tg.grepStdoutNot(`\(cached\)`, "did not notice env var change")
	tg.run("test", "testcache")
	tg.grepStdout(`\(cached\)`, "did not cache")

	tg.run("test", "testcache", "-run=FileSize")
	tg.run("test", "testcache", "-run=FileSize")
	tg.grepStdout(`\(cached\)`, "did not cache")
	tg.must(ioutil.WriteFile(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"), []byte("xxx"), 0644))
	tg.run("test", "testcache", "-run=FileSize")
	tg.grepStdoutNot(`\(cached\)`, "did not notice file size change")
	tg.run("test", "testcache", "-run=FileSize")
	tg.grepStdout(`\(cached\)`, "did not cache")

	tg.run("test", "testcache", "-run=Chdir")
	tg.run("test", "testcache", "-run=Chdir")
	tg.grepStdout(`\(cached\)`, "did not cache")
	tg.must(ioutil.WriteFile(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"), []byte("xxxxx"), 0644))
	tg.run("test", "testcache", "-run=Chdir")
	tg.grepStdoutNot(`\(cached\)`, "did not notice file size change")
	tg.run("test", "testcache", "-run=Chdir")
	tg.grepStdout(`\(cached\)`, "did not cache")

	tg.must(os.Chtimes(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"), old, old))
	tg.run("test", "testcache", "-run=FileContent")
	tg.run("test", "testcache", "-run=FileContent")
	tg.grepStdout(`\(cached\)`, "did not cache")
	tg.must(ioutil.WriteFile(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"), []byte("yyy"), 0644))
	old2 := old.Add(10 * time.Second)
	tg.must(os.Chtimes(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt"), old2, old2))
	tg.run("test", "testcache", "-run=FileContent")
	tg.grepStdoutNot(`\(cached\)`, "did not notice file content change")
	tg.run("test", "testcache", "-run=FileContent")
	tg.grepStdout(`\(cached\)`, "did not cache")

	tg.run("test", "testcache", "-run=DirList")
	tg.run("test", "testcache", "-run=DirList")
	tg.grepStdout(`\(cached\)`, "did not cache")
	tg.must(os.Remove(filepath.Join(tg.pwd(), "testdata/src/testcache/file.txt")))
	tg.run("test", "testcache", "-run=DirList")
	tg.grepStdoutNot(`\(cached\)`, "did not notice directory change")
	tg.run("test", "testcache", "-run=DirList")
	tg.grepStdout(`\(cached\)`, "did not cache")

	tg.tempFile("file.txt", "")
	tg.must(ioutil.WriteFile(filepath.Join(tg.pwd(), "testdata/src/testcache/testcachetmp_test.go"), []byte(`package testcache

		import (
			"os"
			"testing"
		)

		func TestExternalFile(t *testing.T) {
			os.Open(`+fmt.Sprintf("%q", tg.path("file.txt"))+`)
			_, err := os.Stat(`+fmt.Sprintf("%q", tg.path("file.txt"))+`)
			if err != nil {
				t.Fatal(err)
			}
		}
	`), 0666))
	defer os.Remove(filepath.Join(tg.pwd(), "testdata/src/testcache/testcachetmp_test.go"))
	tg.run("test", "testcache", "-run=ExternalFile")
	tg.run("test", "testcache", "-run=ExternalFile")
	tg.grepStdout(`\(cached\)`, "did not cache")
	tg.must(os.Remove(filepath.Join(tg.tempdir, "file.txt")))
	tg.run("test", "testcache", "-run=ExternalFile")
	tg.grepStdout(`\(cached\)`, "did not cache")

	switch runtime.GOOS {
	case "nacl", "plan9", "windows":
		// no shell scripts
	default:
		tg.run("test", "testcache", "-run=Exec")
		tg.run("test", "testcache", "-run=Exec")
		tg.grepStdout(`\(cached\)`, "did not cache")
		tg.must(os.Chtimes(filepath.Join(tg.pwd(), "testdata/src/testcache/script.sh"), old2, old2))
		tg.run("test", "testcache", "-run=Exec")
		tg.grepStdoutNot(`\(cached\)`, "did not notice script change")
		tg.run("test", "testcache", "-run=Exec")
		tg.grepStdout(`\(cached\)`, "did not cache")
	}
}

func TestTestVet(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	tg.tempFile("p1_test.go", `
		package p
		import "testing"
		func Test(t *testing.T) {
			t.Logf("%d") // oops
		}
	`)

	tg.runFail("test", tg.path("p1_test.go"))
	tg.grepStderr(`Logf format %d`, "did not diagnose bad Logf")
	tg.run("test", "-vet=off", tg.path("p1_test.go"))
	tg.grepStdout(`^ok`, "did not print test summary")

	tg.tempFile("p1.go", `
		package p
		import "fmt"
		func F() {
			fmt.Printf("%d") // oops
		}
	`)
	tg.runFail("test", tg.path("p1.go"))
	tg.grepStderr(`Printf format %d`, "did not diagnose bad Printf")
	tg.run("test", "-x", "-vet=shift", tg.path("p1.go"))
	tg.grepStderr(`[\\/]vet.*-shift`, "did not run vet with -shift")
	tg.grepStdout(`\[no test files\]`, "did not print test summary")
	tg.run("test", "-vet=off", tg.path("p1.go"))
	tg.grepStdout(`\[no test files\]`, "did not print test summary")

	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("test", "vetcycle") // must not fail; #22890

	tg.runFail("test", "vetfail/...")
	tg.grepStderr(`Printf format %d`, "did not diagnose bad Printf")
	tg.grepStdout(`ok\s+vetfail/p2`, "did not run vetfail/p2")

	// Use -a so that we need to recompute the vet-specific export data for
	// vetfail/p1.
	tg.run("test", "-a", "vetfail/p2")
	tg.grepStderrNot(`invalid.*constraint`, "did diagnose bad build constraint in vetxonly mode")
}

func TestTestSkipVetAfterFailedBuild(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	tg.tempFile("x_test.go", `package x
		func f() {
			return 1
		}
	`)

	tg.runFail("test", tg.path("x_test.go"))
	tg.grepStderrNot(`vet`, "vet should be skipped after the failed build")
}

func TestTestVetRebuild(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	// golang.org/issue/23701.
	// b_test imports b with augmented method from export_test.go.
	// b_test also imports a, which imports b.
	// Must not accidentally see un-augmented b propagate through a to b_test.
	tg.tempFile("src/a/a.go", `package a
		import "b"
		type Type struct{}
		func (*Type) M() b.T {return 0}
	`)
	tg.tempFile("src/b/b.go", `package b
		type T int
		type I interface {M() T}
	`)
	tg.tempFile("src/b/export_test.go", `package b
		func (*T) Method() *T { return nil }
	`)
	tg.tempFile("src/b/b_test.go", `package b_test
		import (
			"testing"
			"a"
			. "b"
		)
		func TestBroken(t *testing.T) {
			x := new(T)
			x.Method()
			_ = new(a.Type)
		}
	`)

	tg.setenv("GOPATH", tg.path("."))
	tg.run("test", "b")
	tg.run("vet", "b")
}

func TestInstallDeps(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.tempdir)

	tg.tempFile("src/p1/p1.go", "package p1\nvar X =  1\n")
	tg.tempFile("src/p2/p2.go", "package p2\nimport _ \"p1\"\n")
	tg.tempFile("src/main1/main.go", "package main\nimport _ \"p2\"\nfunc main() {}\n")

	tg.run("list", "-f={{.Target}}", "p1")
	p1 := strings.TrimSpace(tg.getStdout())
	tg.run("list", "-f={{.Target}}", "p2")
	p2 := strings.TrimSpace(tg.getStdout())
	tg.run("list", "-f={{.Target}}", "main1")
	main1 := strings.TrimSpace(tg.getStdout())

	tg.run("install", "main1")

	tg.mustExist(main1)
	tg.mustNotExist(p2)
	tg.mustNotExist(p1)

	tg.run("install", "p2")
	tg.mustExist(p2)
	tg.mustNotExist(p1)

	// don't let install -i overwrite runtime
	tg.wantNotStale("runtime", "", "must be non-stale before install -i")

	tg.run("install", "-i", "main1")
	tg.mustExist(p1)
	tg.must(os.Remove(p1))

	tg.run("install", "-i", "p2")
	tg.mustExist(p1)
}

func TestFmtLoadErrors(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("fmt", "does-not-exist")
	tg.run("fmt", "-n", "exclude")
}

func TestGoTestMinusN(t *testing.T) {
	// Intent here is to verify that 'go test -n' works without crashing.
	// This reuses flag_test.go, but really any test would do.
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "testdata/flag_test.go", "-n", "-args", "-v=7")
}

func TestGoTestJSON(t *testing.T) {
	skipIfGccgo(t, "gccgo does not have standard packages")
	tooSlow(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOCACHE", tg.tempdir)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))

	// It would be nice to test that the output is interlaced
	// but it seems to be impossible to do that in a short test
	// that isn't also flaky. Just check that we get JSON output.
	tg.run("test", "-json", "-short", "-v", "errors", "empty/pkg", "skipper")
	tg.grepStdout(`"Package":"errors"`, "did not see JSON output")
	tg.grepStdout(`"Action":"run"`, "did not see JSON output")

	tg.grepStdout(`"Action":"output","Package":"empty/pkg","Output":".*no test files`, "did not see no test files print")
	tg.grepStdout(`"Action":"skip","Package":"empty/pkg"`, "did not see skip")

	tg.grepStdout(`"Action":"output","Package":"skipper","Test":"Test","Output":"--- SKIP:`, "did not see SKIP output")
	tg.grepStdout(`"Action":"skip","Package":"skipper","Test":"Test"`, "did not see skip result for Test")

	tg.run("test", "-json", "-short", "-v", "errors")
	tg.grepStdout(`"Action":"output","Package":"errors","Output":".*\(cached\)`, "did not see no cached output")

	tg.run("test", "-json", "-bench=NONE", "-short", "-v", "errors")
	tg.grepStdout(`"Package":"errors"`, "did not see JSON output")
	tg.grepStdout(`"Action":"run"`, "did not see JSON output")

	tg.run("test", "-o", tg.path("errors.test.exe"), "-c", "errors")
	tg.run("tool", "test2json", "-p", "errors", tg.path("errors.test.exe"), "-test.v", "-test.short")
	tg.grepStdout(`"Package":"errors"`, "did not see JSON output")
	tg.grepStdout(`"Action":"run"`, "did not see JSON output")
	tg.grepStdout(`\{"Action":"pass","Package":"errors"\}`, "did not see final pass")
}

func TestFailFast(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()

	tests := []struct {
		run      string
		failfast bool
		nfail    int
	}{
		{"TestFailingA", true, 1},
		{"TestFailing[AB]", true, 1},
		{"TestFailing[AB]", false, 2},
		// mix with non-failing tests:
		{"TestA|TestFailing[AB]", true, 1},
		{"TestA|TestFailing[AB]", false, 2},
		// mix with parallel tests:
		{"TestFailingB|TestParallelFailingA", true, 2},
		{"TestFailingB|TestParallelFailingA", false, 2},
		{"TestFailingB|TestParallelFailing[AB]", true, 3},
		{"TestFailingB|TestParallelFailing[AB]", false, 3},
		// mix with parallel sub-tests
		{"TestFailingB|TestParallelFailing[AB]|TestParallelFailingSubtestsA", true, 3},
		{"TestFailingB|TestParallelFailing[AB]|TestParallelFailingSubtestsA", false, 5},
		{"TestParallelFailingSubtestsA", true, 1},
		// only parallels:
		{"TestParallelFailing[AB]", false, 2},
		// non-parallel subtests:
		{"TestFailingSubtestsA", true, 1},
		{"TestFailingSubtestsA", false, 2},
		// fatal test
		{"TestFatal[CD]", true, 1},
		{"TestFatal[CD]", false, 2},
	}

	for _, tt := range tests {
		t.Run(tt.run, func(t *testing.T) {
			tg.runFail("test", "./testdata/src/failfast_test.go", "-run="+tt.run, "-failfast="+strconv.FormatBool(tt.failfast))

			nfail := strings.Count(tg.getStdout(), "FAIL - ")

			if nfail != tt.nfail {
				t.Errorf("go test -run=%s -failfast=%t printed %d FAILs, want %d", tt.run, tt.failfast, nfail, tt.nfail)
			}
		})
	}
}

// Issue 22986.
func TestImportPath(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()

	tg.tempFile("src/a/a.go", `
package main

import (
	"log"
	p "a/p-1.0"
)

func main() {
	if !p.V {
		log.Fatal("false")
	}
}`)

	tg.tempFile("src/a/a_test.go", `
package main_test

import (
	p "a/p-1.0"
	"testing"
)

func TestV(t *testing.T) {
	if !p.V {
		t.Fatal("false")
	}
}`)

	tg.tempFile("src/a/p-1.0/p.go", `
package p

var V = true

func init() {}
`)

	tg.setenv("GOPATH", tg.path("."))
	tg.run("build", "-o", tg.path("a.exe"), "a")
	tg.run("test", "a")
}

func TestBadCommandLines(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()

	tg.tempFile("src/x/x.go", "package x\n")
	tg.setenv("GOPATH", tg.path("."))

	tg.run("build", "x")

	tg.tempFile("src/x/@y.go", "package x\n")
	tg.runFail("build", "x")
	tg.grepStderr("invalid input file name \"@y.go\"", "did not reject @y.go")
	tg.must(os.Remove(tg.path("src/x/@y.go")))

	tg.tempFile("src/x/-y.go", "package x\n")
	tg.runFail("build", "x")
	tg.grepStderr("invalid input file name \"-y.go\"", "did not reject -y.go")
	tg.must(os.Remove(tg.path("src/x/-y.go")))

	if runtime.Compiler == "gccgo" {
		tg.runFail("build", "-gccgoflags=all=@x", "x")
	} else {
		tg.runFail("build", "-gcflags=all=@x", "x")
	}
	tg.grepStderr("invalid command-line argument @x in command", "did not reject @x during exec")

	tg.tempFile("src/@x/x.go", "package x\n")
	tg.setenv("GOPATH", tg.path("."))
	tg.runFail("build", "@x")
	tg.grepStderr("invalid input directory name \"@x\"|cannot use path@version syntax", "did not reject @x directory")

	tg.tempFile("src/@x/y/y.go", "package y\n")
	tg.setenv("GOPATH", tg.path("."))
	tg.runFail("build", "@x/y")
	tg.grepStderr("invalid import path \"@x/y\"|cannot use path@version syntax", "did not reject @x/y import path")

	tg.tempFile("src/-x/x.go", "package x\n")
	tg.setenv("GOPATH", tg.path("."))
	tg.runFail("build", "--", "-x")
	tg.grepStderr("invalid input directory name \"-x\"", "did not reject -x directory")

	tg.tempFile("src/-x/y/y.go", "package y\n")
	tg.setenv("GOPATH", tg.path("."))
	tg.runFail("build", "--", "-x/y")
	tg.grepStderr("invalid import path \"-x/y\"", "did not reject -x/y import path")
}

func TestBadCgoDirectives(t *testing.T) {
	if !canCgo {
		t.Skip("no cgo")
	}
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()

	tg.tempFile("src/x/x.go", "package x\n")
	tg.setenv("GOPATH", tg.path("."))

	if runtime.Compiler == "gc" {
		tg.tempFile("src/x/x.go", `package x

			//go:cgo_ldflag "-fplugin=foo.so"

			import "C"
		`)
		tg.runFail("build", "x")
		tg.grepStderr("//go:cgo_ldflag .* only allowed in cgo-generated code", "did not reject //go:cgo_ldflag directive")
	}

	tg.must(os.Remove(tg.path("src/x/x.go")))
	tg.runFail("build", "x")
	tg.grepStderr("no Go files", "did not report missing source code")
	tg.tempFile("src/x/_cgo_yy.go", `package x

		//go:cgo_ldflag "-fplugin=foo.so"

		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("no Go files", "did not report missing source code") // _* files are ignored...

	if runtime.Compiler == "gc" {
		tg.runFail("build", tg.path("src/x/_cgo_yy.go")) // ... but if forced, the comment is rejected
		// Actually, today there is a separate issue that _ files named
		// on the command line are ignored. Once that is fixed,
		// we want to see the cgo_ldflag error.
		tg.grepStderr("//go:cgo_ldflag only allowed in cgo-generated code|no Go files", "did not reject //go:cgo_ldflag directive")
	}

	tg.must(os.Remove(tg.path("src/x/_cgo_yy.go")))

	tg.tempFile("src/x/x.go", "package x\n")
	tg.tempFile("src/x/y.go", `package x
		// #cgo CFLAGS: -fplugin=foo.so
		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("invalid flag in #cgo CFLAGS: -fplugin=foo.so", "did not reject -fplugin")

	tg.tempFile("src/x/y.go", `package x
		// #cgo CFLAGS: -Ibar -fplugin=foo.so
		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("invalid flag in #cgo CFLAGS: -fplugin=foo.so", "did not reject -fplugin")

	tg.tempFile("src/x/y.go", `package x
		// #cgo pkg-config: -foo
		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("invalid pkg-config package name: -foo", "did not reject pkg-config: -foo")

	tg.tempFile("src/x/y.go", `package x
		// #cgo pkg-config: @foo
		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("invalid pkg-config package name: @foo", "did not reject pkg-config: -foo")

	tg.tempFile("src/x/y.go", `package x
		// #cgo CFLAGS: @foo
		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("invalid flag in #cgo CFLAGS: @foo", "did not reject @foo flag")

	tg.tempFile("src/x/y.go", `package x
		// #cgo CFLAGS: -D
		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("invalid flag in #cgo CFLAGS: -D without argument", "did not reject trailing -I flag")

	// Note that -I @foo is allowed because we rewrite it into -I /path/to/src/@foo
	// before the check is applied. There's no such rewrite for -D.

	tg.tempFile("src/x/y.go", `package x
		// #cgo CFLAGS: -D @foo
		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("invalid flag in #cgo CFLAGS: -D @foo", "did not reject -D @foo flag")

	tg.tempFile("src/x/y.go", `package x
		// #cgo CFLAGS: -D@foo
		import "C"
	`)
	tg.runFail("build", "x")
	tg.grepStderr("invalid flag in #cgo CFLAGS: -D@foo", "did not reject -D@foo flag")

	tg.setenv("CGO_CFLAGS", "-D@foo")
	tg.tempFile("src/x/y.go", `package x
		import "C"
	`)
	tg.run("build", "-n", "x")
	tg.grepStderr("-D@foo", "did not find -D@foo in commands")
}

func TestTwoPkgConfigs(t *testing.T) {
	if !canCgo {
		t.Skip("no cgo")
	}
	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		t.Skipf("no shell scripts on %s", runtime.GOOS)
	}
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/x/a.go", `package x
		// #cgo pkg-config: --static a
		import "C"
	`)
	tg.tempFile("src/x/b.go", `package x
		// #cgo pkg-config: --static a
		import "C"
	`)
	tg.tempFile("pkg-config.sh", `#!/bin/sh
echo $* >>`+tg.path("pkg-config.out"))
	tg.must(os.Chmod(tg.path("pkg-config.sh"), 0755))
	tg.setenv("GOPATH", tg.path("."))
	tg.setenv("PKG_CONFIG", tg.path("pkg-config.sh"))
	tg.run("build", "x")
	out, err := ioutil.ReadFile(tg.path("pkg-config.out"))
	tg.must(err)
	out = bytes.TrimSpace(out)
	want := "--cflags --static --static -- a a\n--libs --static --static -- a a"
	if !bytes.Equal(out, []byte(want)) {
		t.Errorf("got %q want %q", out, want)
	}
}

func TestCgoCache(t *testing.T) {
	if !canCgo {
		t.Skip("no cgo")
	}
	tooSlow(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/x/a.go", `package main
		// #ifndef VAL
		// #define VAL 0
		// #endif
		// int val = VAL;
		import "C"
		import "fmt"
		func main() { fmt.Println(C.val) }
	`)
	tg.setenv("GOPATH", tg.path("."))
	exe := tg.path("x.exe")
	tg.run("build", "-o", exe, "x")
	tg.setenv("CGO_LDFLAGS", "-lnosuchlibraryexists")
	tg.runFail("build", "-o", exe, "x")
	tg.grepStderr(`nosuchlibraryexists`, "did not run linker with changed CGO_LDFLAGS")
}

// Issue 23982
func TestFilepathUnderCwdFormat(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-x", "-cover", "log")
	tg.grepStderrNot(`\.log\.cover\.go`, "-x output should contain correctly formatted filepath under cwd")
}

// Issue 24396.
func TestDontReportRemoveOfEmptyDir(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/a/a.go", `package a`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("install", "-x", "a")
	tg.run("install", "-x", "a")
	// The second install should have printed only a WORK= line,
	// nothing else.
	if bytes.Count(tg.stdout.Bytes(), []byte{'\n'})+bytes.Count(tg.stderr.Bytes(), []byte{'\n'}) > 1 {
		t.Error("unnecessary output when installing installed package")
	}
}

// Issue 24704.
func TestLinkerTmpDirIsDeleted(t *testing.T) {
	skipIfGccgo(t, "gccgo does not use cmd/link")
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}
	tooSlow(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("a.go", `package main; import "C"; func main() {}`)
	tg.run("build", "-ldflags", "-v", "-o", os.DevNull, tg.path("a.go"))
	// Find line that has "host link:" in linker output.
	stderr := tg.getStderr()
	var hostLinkLine string
	for _, line := range strings.Split(stderr, "\n") {
		if !strings.Contains(line, "host link:") {
			continue
		}
		hostLinkLine = line
		break
	}
	if hostLinkLine == "" {
		t.Fatal(`fail to find with "host link:" string in linker output`)
	}
	// Find parameter, like "/tmp/go-link-408556474/go.o" inside of
	// "host link:" line, and extract temp directory /tmp/go-link-408556474
	// out of it.
	tmpdir := hostLinkLine
	i := strings.Index(tmpdir, `go.o"`)
	if i == -1 {
		t.Fatalf(`fail to find "go.o" in "host link:" line %q`, hostLinkLine)
	}
	tmpdir = tmpdir[:i-1]
	i = strings.LastIndex(tmpdir, `"`)
	if i == -1 {
		t.Fatalf(`fail to find " in "host link:" line %q`, hostLinkLine)
	}
	tmpdir = tmpdir[i+1:]
	// Verify that temp directory has been removed.
	_, err := os.Stat(tmpdir)
	if err == nil {
		t.Fatalf("temp directory %q has not been removed", tmpdir)
	}
	if !os.IsNotExist(err) {
		t.Fatalf("Stat(%q) returns unexpected error: %v", tmpdir, err)
	}
}

func testCDAndGOPATHAreDifferent(tg *testgoData, cd, gopath string) {
	skipIfGccgo(tg.t, "gccgo does not support -ldflags -X")
	tg.setenv("GOPATH", gopath)

	tg.tempDir("dir")
	exe := tg.path("dir/a.exe")

	tg.cd(cd)

	tg.run("build", "-o", exe, "-ldflags", "-X=my.pkg.Text=linkXworked")
	out, err := exec.Command(exe).CombinedOutput()
	if err != nil {
		tg.t.Fatal(err)
	}
	if string(out) != "linkXworked\n" {
		tg.t.Errorf(`incorrect output with GOPATH=%q and CD=%q: expected "linkXworked\n", but have %q`, gopath, cd, string(out))
	}
}

func TestCDAndGOPATHAreDifferent(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()

	gopath := filepath.Join(tg.pwd(), "testdata")
	cd := filepath.Join(gopath, "src/my.pkg/main")

	testCDAndGOPATHAreDifferent(tg, cd, gopath)
	if runtime.GOOS == "windows" {
		testCDAndGOPATHAreDifferent(tg, cd, strings.ReplaceAll(gopath, `\`, `/`))
		testCDAndGOPATHAreDifferent(tg, cd, strings.ToUpper(gopath))
		testCDAndGOPATHAreDifferent(tg, cd, strings.ToLower(gopath))
	}
}

// Issue 26242.
func TestGoTestWithoutTests(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("test", "testnorun")
	tg.grepStdout(`testnorun\t\[no test files\]`, "do not want test to run")
}

// Issue 25579.
func TestGoBuildDashODevNull(t *testing.T) {
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("build", "-o", os.DevNull, filepath.Join(tg.pwd(), "testdata", "src", "hello", "hello.go"))
	tg.mustNotExist("hello")
	tg.mustNotExist("hello.exe")
}

// Issue 25093.
func TestCoverpkgTestOnly(t *testing.T) {
	skipIfGccgo(t, "gccgo has no cover tool")
	tooSlow(t)
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/a/a.go", `package a
		func F(i int) int {
			return i*i
		}`)
	tg.tempFile("src/atest/a_test.go", `
		package a_test
		import ( "a"; "testing" )
		func TestF(t *testing.T) { a.F(2) }
	`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("test", "-coverpkg=a", "atest")
	tg.grepStderrNot("no packages being tested depend on matches", "bad match message")
	tg.grepStdout("coverage: 100", "no coverage")
}
