// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"flag"
	"fmt"
	"go/build"
	"go/format"
	"internal/race"
	"internal/testenv"
	"io"
	"io/ioutil"
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

var (
	canRun  = true  // whether we can run go or ./testgo
	canRace = false // whether we can run the race detector
	canCgo  = false // whether we can use cgo

	exeSuffix string // ".exe" on Windows

	skipExternal = false // skip external tests
)

func init() {
	switch runtime.GOOS {
	case "android", "nacl":
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
		}
	case "freebsd":
		switch runtime.GOARCH {
		case "arm":
			// many freebsd/arm machines are too slow to run
			// the full set of external tests.
			skipExternal = true
			canRun = false
		}
	case "windows":
		exeSuffix = ".exe"
	}
}

// The TestMain function creates a go command for testing purposes and
// deletes it after the tests have been run.
func TestMain(m *testing.M) {
	flag.Parse()

	if canRun {
		args := []string{"build", "-tags", "testgo", "-o", "testgo" + exeSuffix}
		if race.Enabled {
			args = append(args, "-race")
		}
		out, err := exec.Command("go", args...).CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "building testgo failed: %v\n%s", err, out)
			os.Exit(2)
		}

		if out, err := exec.Command("./testgo"+exeSuffix, "env", "CGO_ENABLED").Output(); err != nil {
			fmt.Fprintf(os.Stderr, "running testgo failed: %v\n", err)
			canRun = false
		} else {
			canCgo, err = strconv.ParseBool(strings.TrimSpace(string(out)))
			if err != nil {
				fmt.Fprintf(os.Stderr, "can't parse go env CGO_ENABLED output: %v\n", strings.TrimSpace(string(out)))
			}
		}

		switch runtime.GOOS {
		case "linux", "darwin", "freebsd", "windows":
			canRace = canCgo && runtime.GOARCH == "amd64"
		}
	}

	// Don't let these environment variables confuse the test.
	os.Unsetenv("GOBIN")
	os.Unsetenv("GOPATH")

	r := m.Run()

	if canRun {
		os.Remove("testgo" + exeSuffix)
	}

	os.Exit(r)
}

// The length of an mtime tick on this system.  This is an estimate of
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
}

// testgo sets up for a test that runs testgo.
func testgo(t *testing.T) *testgoData {
	testenv.MustHaveGoBuild(t)

	if skipExternal {
		t.Skip("skipping external tests on %s/%s", runtime.GOOS, runtime.GOARCH)
	}

	return &testgoData{t: t}
}

// must gives a fatal error if err is not nil.
func (tg *testgoData) must(err error) {
	if err != nil {
		tg.t.Fatal(err)
	}
}

// check gives a test non-fatal error if err is not nil.
func (tg *testgoData) check(err error) {
	if err != nil {
		tg.t.Error(err)
	}
}

// parallel runs the test in parallel by calling t.Parallel.
func (tg *testgoData) parallel() {
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
	wd, err := os.Getwd()
	if err != nil {
		tg.t.Fatalf("could not get working directory: %v", err)
	}
	return wd
}

// cd changes the current directory to the named directory.  Note that
// using this means that the test must not be run in parallel with any
// other tests.
func (tg *testgoData) cd(dir string) {
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
	}
	for i, v := range tg.env {
		if strings.HasPrefix(v, name+"=") {
			tg.env = append(tg.env[:i], tg.env[i+1:]...)
			break
		}
	}
}

// doRun runs the test go command, recording stdout and stderr and
// returning exit status.
func (tg *testgoData) doRun(args []string) error {
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
	tg.t.Logf("running testgo %v", args)
	var prog string
	if tg.wd == "" {
		prog = "./testgo" + exeSuffix
	} else {
		prog = filepath.Join(tg.wd, "testgo"+exeSuffix)
	}
	cmd := exec.Command(prog, args...)
	tg.stdout.Reset()
	tg.stderr.Reset()
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
	if status := tg.doRun(args); status != nil {
		tg.t.Logf("go %v failed unexpectedly: %v", args, status)
		tg.t.FailNow()
	}
}

// runFail runs the test go command, and expects it to fail.
func (tg *testgoData) runFail(args ...string) {
	if status := tg.doRun(args); status == nil {
		tg.t.Fatal("testgo succeeded unexpectedly")
	} else {
		tg.t.Log("testgo failed as expected:", status)
	}
}

// runGit runs a git command, and expects it to succeed.
func (tg *testgoData) runGit(dir string, args ...string) {
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
	if !tg.ran {
		tg.t.Fatal("internal testsuite error: stdout called before run")
	}
	return tg.stdout.String()
}

// getStderr returns standard error of the testgo run as a string.
func (tg *testgoData) getStderr() string {
	if !tg.ran {
		tg.t.Fatal("internal testsuite error: stdout called before run")
	}
	return tg.stderr.String()
}

// doGrepMatch looks for a regular expression in a buffer, and returns
// whether it is found.  The regular expression is matched against
// each line separately, as with the grep command.
func (tg *testgoData) doGrepMatch(match string, b *bytes.Buffer) bool {
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
// is not found.  The name argument is the name of the output we are
// searching, "output" or "error".  The msg argument is logged on
// failure.
func (tg *testgoData) doGrep(match string, b *bytes.Buffer, name, msg string) {
	if !tg.doGrepMatch(match, b) {
		tg.t.Log(msg)
		tg.t.Logf("pattern %v not found in standard %s", match, name)
		tg.t.FailNow()
	}
}

// grepStdout looks for a regular expression in the test run's
// standard output and fails, logging msg, if it is not found.
func (tg *testgoData) grepStdout(match, msg string) {
	tg.doGrep(match, &tg.stdout, "output", msg)
}

// grepStderr looks for a regular expression in the test run's
// standard error and fails, logging msg, if it is not found.
func (tg *testgoData) grepStderr(match, msg string) {
	tg.doGrep(match, &tg.stderr, "error", msg)
}

// grepBoth looks for a regular expression in the test run's standard
// output or stand error and fails, logging msg, if it is not found.
func (tg *testgoData) grepBoth(match, msg string) {
	if !tg.doGrepMatch(match, &tg.stdout) && !tg.doGrepMatch(match, &tg.stderr) {
		tg.t.Log(msg)
		tg.t.Logf("pattern %v not found in standard output or standard error", match)
		tg.t.FailNow()
	}
}

// doGrepNot looks for a regular expression in a buffer and fails if
// it is found.  The name and msg arguments are as for doGrep.
func (tg *testgoData) doGrepNot(match string, b *bytes.Buffer, name, msg string) {
	if tg.doGrepMatch(match, b) {
		tg.t.Log(msg)
		tg.t.Logf("pattern %v found unexpectedly in standard %s", match, name)
		tg.t.FailNow()
	}
}

// grepStdoutNot looks for a regular expression in the test run's
// standard output and fails, logging msg, if it is found.
func (tg *testgoData) grepStdoutNot(match, msg string) {
	tg.doGrepNot(match, &tg.stdout, "output", msg)
}

// grepStderrNot looks for a regular expression in the test run's
// standard error and fails, logging msg, if it is found.
func (tg *testgoData) grepStderrNot(match, msg string) {
	tg.doGrepNot(match, &tg.stderr, "error", msg)
}

// grepBothNot looks for a regular expression in the test run's
// standard output or stand error and fails, logging msg, if it is
// found.
func (tg *testgoData) grepBothNot(match, msg string) {
	if tg.doGrepMatch(match, &tg.stdout) || tg.doGrepMatch(match, &tg.stderr) {
		tg.t.Log(msg)
		tg.t.Fatalf("pattern %v found unexpectedly in standard output or standard error", match)
	}
}

// doGrepCount counts the number of times a regexp is seen in a buffer.
func (tg *testgoData) doGrepCount(match string, b *bytes.Buffer) int {
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

// grepCountStdout returns the number of times a regexp is seen in
// standard output.
func (tg *testgoData) grepCountStdout(match string) int {
	return tg.doGrepCount(match, &tg.stdout)
}

// grepCountStderr returns the number of times a regexp is seen in
// standard error.
func (tg *testgoData) grepCountStderr(match string) int {
	return tg.doGrepCount(match, &tg.stderr)
}

// grepCountBoth returns the number of times a regexp is seen in both
// standard output and standard error.
func (tg *testgoData) grepCountBoth(match string) int {
	return tg.doGrepCount(match, &tg.stdout) + tg.doGrepCount(match, &tg.stderr)
}

// creatingTemp records that the test plans to create a temporary file
// or directory.  If the file or directory exists already, it will be
// removed.  When the test completes, the file or directory will be
// removed if it exists.
func (tg *testgoData) creatingTemp(path string) {
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

// makeTempdir makes a temporary directory for a run of testgo.  If
// the temporary directory was already created, this does nothing.
func (tg *testgoData) makeTempdir() {
	if tg.tempdir == "" {
		var err error
		tg.tempdir, err = ioutil.TempDir("", "gotest")
		tg.must(err)
	}
}

// tempFile adds a temporary file for a run of testgo.
func (tg *testgoData) tempFile(path, contents string) {
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
	tg.makeTempdir()
	if err := os.MkdirAll(filepath.Join(tg.tempdir, path), 0755); err != nil && !os.IsExist(err) {
		tg.t.Fatal(err)
	}
}

// path returns the absolute pathname to file with the temporary
// directory.
func (tg *testgoData) path(name string) string {
	if tg.tempdir == "" {
		tg.t.Fatalf("internal testsuite error: path(%q) with no tempdir", name)
	}
	if name == "." {
		return tg.tempdir
	}
	return filepath.Join(tg.tempdir, name)
}

// mustNotExist fails if path exists.
func (tg *testgoData) mustNotExist(path string) {
	if _, err := os.Stat(path); err == nil || !os.IsNotExist(err) {
		tg.t.Fatalf("%s exists but should not (%v)", path, err)
	}
}

// wantExecutable fails with msg if path is not executable.
func (tg *testgoData) wantExecutable(path, msg string) {
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

// isStale returns whether pkg is stale.
func (tg *testgoData) isStale(pkg string) bool {
	tg.run("list", "-f", "{{.Stale}}", pkg)
	switch v := strings.TrimSpace(tg.getStdout()); v {
	case "true":
		return true
	case "false":
		return false
	default:
		tg.t.Fatalf("unexpected output checking staleness of package %v: %v", pkg, v)
		panic("unreachable")
	}
}

// wantStale fails with msg if pkg is not stale.
func (tg *testgoData) wantStale(pkg, msg string) {
	if !tg.isStale(pkg) {
		tg.t.Fatal(msg)
	}
}

// wantNotStale fails with msg if pkg is stale.
func (tg *testgoData) wantNotStale(pkg, msg string) {
	if tg.isStale(pkg) {
		tg.t.Fatal(msg)
	}
}

// cleanup cleans up a test that runs testgo.
func (tg *testgoData) cleanup() {
	if tg.wd != "" {
		if err := os.Chdir(tg.wd); err != nil {
			// We are unlikely to be able to continue.
			fmt.Fprintln(os.Stderr, "could not restore working directory, crashing:", err)
			os.Exit(2)
		}
	}
	for _, path := range tg.temps {
		tg.check(os.RemoveAll(path))
	}
	if tg.tempdir != "" {
		tg.check(os.RemoveAll(tg.tempdir))
	}
}

// resetReadOnlyFlagAll resets windows read-only flag
// set on path and any children it contains.
// The flag is set by git and has to be removed.
// os.Remove refuses to remove files with read-only flag set.
func (tg *testgoData) resetReadOnlyFlagAll(path string) {
	fi, err := os.Stat(path)
	if err != nil {
		tg.t.Fatalf("resetReadOnlyFlagAll(%q) failed: %v", path, err)
	}
	if !fi.IsDir() {
		err := os.Chmod(path, 0666)
		if err != nil {
			tg.t.Fatalf("resetReadOnlyFlagAll(%q) failed: %v", path, err)
		}
	}
	fd, err := os.Open(path)
	if err != nil {
		tg.t.Fatalf("resetReadOnlyFlagAll(%q) failed: %v", path, err)
	}
	defer fd.Close()
	names, _ := fd.Readdirnames(-1)
	for _, name := range names {
		tg.resetReadOnlyFlagAll(path + string(filepath.Separator) + name)
	}
}

// failSSH puts an ssh executable in the PATH that always fails.
// This is to stub out uses of ssh by go get.
func (tg *testgoData) failSSH() {
	wd, err := os.Getwd()
	if err != nil {
		tg.t.Fatal(err)
	}
	fail := filepath.Join(wd, "testdata/failssh")
	tg.setenv("PATH", fmt.Sprintf("%v%c%v", fail, filepath.ListSeparator, os.Getenv("PATH")))
}

func TestFileLineInErrorMessages(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("err.go", `package main; import "bar"`)
	path := tg.path("err.go")
	tg.runFail("run", path)
	shortPath := path
	if rel, err := filepath.Rel(tg.pwd(), path); err == nil && len(rel) < len(path) {
		shortPath = rel
	}
	tg.grepStderr("^"+regexp.QuoteMeta(shortPath)+":", "missing file:line in error message")
}

func TestProgramNameInCrashMessages(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("triv.go", `package main; func main() {}`)
	tg.runFail("build", "-ldflags", "-crash_for_testing", tg.path("triv.go"))
	tg.grepStderr(`[/\\]tool[/\\].*[/\\]link`, "missing linker name in error message")
}

func TestBrokenTestsWithoutTestFunctionsAllFail(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("test", "./testdata/src/badtest/...")
	tg.grepBothNot("^ok", "test passed unexpectedly")
	tg.grepBoth("FAIL.*badtest/badexec", "test did not run everything")
	tg.grepBoth("FAIL.*badtest/badsyntax", "test did not run everything")
	tg.grepBoth("FAIL.*badtest/badvar", "test did not run everything")
}

func TestGoBuildDashAInDevBranch(t *testing.T) {
	if testing.Short() {
		t.Skip("don't rebuild the standard library in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.run("install", "math") // should be up to date already but just in case
	tg.setenv("TESTGO_IS_GO_RELEASE", "0")
	tg.run("build", "-v", "-a", "math")
	tg.grepStderr("runtime", "testgo build -a math in dev branch DID NOT build runtime, but should have")

	// Everything is out of date. Rebuild to leave things in a better state.
	tg.run("install", "std")
}

func TestGoBuildDashAInReleaseBranch(t *testing.T) {
	if testing.Short() {
		t.Skip("don't rebuild the standard library in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.run("install", "math", "net/http") // should be up to date already but just in case
	tg.setenv("TESTGO_IS_GO_RELEASE", "1")
	tg.run("install", "-v", "-a", "math")
	tg.grepStderr("runtime", "testgo build -a math in release branch DID NOT build runtime, but should have")

	// Now runtime.a is updated (newer mtime), so everything would look stale if not for being a release.
	tg.run("build", "-v", "net/http")
	tg.grepStderrNot("strconv", "testgo build -v net/http in release branch with newer runtime.a DID build strconv but should not have")
	tg.grepStderrNot("golang.org/x/net/http2/hpack", "testgo build -v net/http in release branch with newer runtime.a DID build .../golang.org/x/net/http2/hpack but should not have")
	tg.grepStderrNot("net/http", "testgo build -v net/http in release branch with newer runtime.a DID build net/http but should not have")

	// Everything is out of date. Rebuild to leave things in a better state.
	tg.run("install", "std")
}

func TestNewReleaseRebuildsStalePackagesInGOPATH(t *testing.T) {
	if testing.Short() {
		t.Skip("don't rebuild the standard library in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()

	addNL := func(name string) (restore func()) {
		data, err := ioutil.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		old := data
		data = append(data, '\n')
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

	tg.setenv("TESTGO_IS_GO_RELEASE", "1")

	tg.tempFile("d1/src/p1/p1.go", `package p1`)
	tg.setenv("GOPATH", tg.path("d1"))
	tg.run("install", "-a", "p1")
	tg.wantNotStale("p1", "./testgo list claims p1 is stale, incorrectly")
	tg.sleep()

	// Changing mtime and content of runtime/internal/sys/sys.go
	// should have no effect: we're in a release, which doesn't rebuild
	// for general mtime or content changes.
	sys := runtime.GOROOT() + "/src/runtime/internal/sys/sys.go"
	restore := addNL(sys)
	defer restore()
	tg.wantNotStale("p1", "./testgo list claims p1 is stale, incorrectly, after updating runtime/internal/sys/sys.go")
	restore()
	tg.wantNotStale("p1", "./testgo list claims p1 is stale, incorrectly, after restoring runtime/internal/sys/sys.go")

	// But changing runtime/internal/sys/zversion.go should have an effect:
	// that's how we tell when we flip from one release to another.
	zversion := runtime.GOROOT() + "/src/runtime/internal/sys/zversion.go"
	restore = addNL(zversion)
	defer restore()
	tg.wantStale("p1", "./testgo list claims p1 is NOT stale, incorrectly, after changing to new release")
	restore()
	tg.wantNotStale("p1", "./testgo list claims p1 is stale, incorrectly, after changing back to old release")
	addNL(zversion)
	tg.wantStale("p1", "./testgo list claims p1 is NOT stale, incorrectly, after changing again to new release")
	tg.run("install", "p1")
	tg.wantNotStale("p1", "./testgo list claims p1 is stale after building with new release")

	// Restore to "old" release.
	restore()
	tg.wantStale("p1", "./testgo list claims p1 is NOT stale, incorrectly, after changing to old release after new build")
	tg.run("install", "p1")
	tg.wantNotStale("p1", "./testgo list claims p1 is stale after building with old release")

	// Everything is out of date. Rebuild to leave things in a better state.
	tg.run("install", "std")
}

func TestGoListStandard(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.cd(runtime.GOROOT() + "/src")
	tg.run("list", "-f", "{{if not .Standard}}{{.ImportPath}}{{end}}", "./...")
	stdout := tg.getStdout()
	for _, line := range strings.Split(stdout, "\n") {
		if strings.HasPrefix(line, "_/") && strings.HasSuffix(line, "/src") {
			// $GOROOT/src shows up if there are any .go files there.
			// We don't care.
			continue
		}
		if line == "" {
			continue
		}
		t.Errorf("package in GOROOT not listed as standard: %v", line)
	}

	// Similarly, expanding std should include some of our vendored code.
	tg.run("list", "std", "cmd")
	tg.grepStdout("golang.org/x/net/http2/hpack", "list std cmd did not mention vendored hpack")
	tg.grepStdout("golang.org/x/arch/x86/x86asm", "list std cmd did not mention vendored x86asm")
}

func TestGoInstallCleansUpAfterGoBuild(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("src/mycmd/main.go", `package main; func main(){}`)
	tg.setenv("GOPATH", tg.path("."))
	tg.cd(tg.path("src/mycmd"))

	doesNotExist := func(file, msg string) {
		if _, err := os.Stat(file); err == nil {
			t.Fatal(msg)
		} else if !os.IsNotExist(err) {
			t.Fatal(msg, "error:", err)
		}
	}

	tg.run("build")
	tg.wantExecutable("mycmd"+exeSuffix, "testgo build did not write command binary")
	tg.run("install")
	doesNotExist("mycmd"+exeSuffix, "testgo install did not remove command binary")
	tg.run("build")
	tg.wantExecutable("mycmd"+exeSuffix, "testgo build did not write command binary (second time)")
	// Running install with arguments does not remove the target,
	// even in the same directory.
	tg.run("install", "mycmd")
	tg.wantExecutable("mycmd"+exeSuffix, "testgo install mycmd removed command binary when run in mycmd")
	tg.run("build")
	tg.wantExecutable("mycmd"+exeSuffix, "testgo build did not write command binary (third time)")
	// And especially not outside the directory.
	tg.cd(tg.path("."))
	if data, err := ioutil.ReadFile("src/mycmd/mycmd" + exeSuffix); err != nil {
		t.Fatal("could not read file:", err)
	} else {
		if err := ioutil.WriteFile("mycmd"+exeSuffix, data, 0555); err != nil {
			t.Fatal("could not write file:", err)
		}
	}
	tg.run("install", "mycmd")
	tg.wantExecutable("src/mycmd/mycmd"+exeSuffix, "testgo install mycmd removed command binary from its source dir when run outside mycmd")
	tg.wantExecutable("mycmd"+exeSuffix, "testgo install mycmd removed command binary from current dir when run outside mycmd")
}

func TestGoInstallRebuildsStalePackagesInOtherGOPATH(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("d1/src/p1/p1.go", `package p1
		import "p2"
		func F() { p2.F() }`)
	tg.tempFile("d2/src/p2/p2.go", `package p2
		func F() {}`)
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", tg.path("d1")+sep+tg.path("d2"))
	tg.run("install", "p1")
	tg.wantNotStale("p1", "./testgo list claims p1 is stale, incorrectly")
	tg.wantNotStale("p2", "./testgo list claims p2 is stale, incorrectly")
	tg.sleep()
	if f, err := os.OpenFile(tg.path("d2/src/p2/p2.go"), os.O_WRONLY|os.O_APPEND, 0); err != nil {
		t.Fatal(err)
	} else if _, err = f.WriteString(`func G() {}`); err != nil {
		t.Fatal(err)
	} else {
		tg.must(f.Close())
	}
	tg.wantStale("p2", "./testgo list claims p2 is NOT stale, incorrectly")
	tg.wantStale("p1", "./testgo list claims p1 is NOT stale, incorrectly")

	tg.run("install", "p1")
	tg.wantNotStale("p2", "./testgo list claims p2 is stale after reinstall, incorrectly")
	tg.wantNotStale("p1", "./testgo list claims p1 is stale after reinstall, incorrectly")
}

func TestGoInstallDetectsRemovedFiles(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/mypkg/x.go", `package mypkg`)
	tg.tempFile("src/mypkg/y.go", `package mypkg`)
	tg.tempFile("src/mypkg/z.go", `// +build missingtag

		package mypkg`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("install", "mypkg")
	tg.wantNotStale("mypkg", "./testgo list mypkg claims mypkg is stale, incorrectly")
	// z.go was not part of the build; removing it is okay.
	tg.must(os.Remove(tg.path("src/mypkg/z.go")))
	tg.wantNotStale("mypkg", "./testgo list mypkg claims mypkg is stale after removing z.go; should not be stale")
	// y.go was part of the package; removing it should be detected.
	tg.must(os.Remove(tg.path("src/mypkg/y.go")))
	tg.wantStale("mypkg", "./testgo list mypkg claims mypkg is NOT stale after removing y.go; should be stale")
}

func TestWildcardMatchesSyntaxErrorDirs(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("src/mypkg/x.go", `package mypkg`)
	tg.tempFile("src/mypkg/y.go", `pkg mypackage`)
	tg.setenv("GOPATH", tg.path("."))
	tg.cd(tg.path("src/mypkg"))
	tg.runFail("list", "./...")
	tg.runFail("build", "./...")
	tg.runFail("install", "./...")
}

func TestGoListWithTags(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("src/mypkg/x.go", "// +build thetag\n\npackage mypkg\n")
	tg.setenv("GOPATH", tg.path("."))
	tg.cd(tg.path("./src"))
	tg.run("list", "-tags=thetag", "./my...")
	tg.grepStdout("mypkg", "did not find mypkg")
}

func TestGoInstallErrorOnCrossCompileToBin(t *testing.T) {
	if testing.Short() {
		t.Skip("don't install into GOROOT in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("src/mycmd/x.go", `package main
		func main() {}`)
	tg.setenv("GOPATH", tg.path("."))
	tg.cd(tg.path("src/mycmd"))

	tg.run("build", "mycmd")

	goarch := "386"
	if runtime.GOARCH == "386" {
		goarch = "amd64"
	}
	tg.setenv("GOOS", "linux")
	tg.setenv("GOARCH", goarch)
	tg.run("install", "mycmd")
	tg.setenv("GOBIN", tg.path("."))
	tg.runFail("install", "mycmd")
	tg.run("install", "cmd/pack")
}

func TestGoInstallDetectsRemovedFilesInPackageMain(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/mycmd/x.go", `package main
		func main() {}`)
	tg.tempFile("src/mycmd/y.go", `package main`)
	tg.tempFile("src/mycmd/z.go", `// +build missingtag

		package main`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("install", "mycmd")
	tg.wantNotStale("mycmd", "./testgo list mypkg claims mycmd is stale, incorrectly")
	// z.go was not part of the build; removing it is okay.
	tg.must(os.Remove(tg.path("src/mycmd/z.go")))
	tg.wantNotStale("mycmd", "./testgo list mycmd claims mycmd is stale after removing z.go; should not be stale")
	// y.go was part of the package; removing it should be detected.
	tg.must(os.Remove(tg.path("src/mycmd/y.go")))
	tg.wantStale("mycmd", "./testgo list mycmd claims mycmd is NOT stale after removing y.go; should be stale")
}

func testLocalRun(tg *testgoData, exepath, local, match string) {
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
	exepath := "./easy" + exeSuffix
	tg.creatingTemp(exepath)
	tg.run("build", "-o", exepath, filepath.Join("testdata", local, "easy.go"))
	testLocalRun(tg, exepath, local, `(?m)^easysub\.Hello`)
}

func testLocalEasySub(tg *testgoData, local string) {
	exepath := "./easysub" + exeSuffix
	tg.creatingTemp(exepath)
	tg.run("build", "-o", exepath, filepath.Join("testdata", local, "easysub", "main.go"))
	testLocalRun(tg, exepath, local, `(?m)^easysub\.Hello`)
}

func testLocalHard(tg *testgoData, local string) {
	exepath := "./hard" + exeSuffix
	tg.creatingTemp(exepath)
	tg.run("build", "-o", exepath, filepath.Join("testdata", local, "hard.go"))
	testLocalRun(tg, exepath, local, `(?m)^sub\.Hello`)
}

func testLocalInstall(tg *testgoData, local string) {
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
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("build", "-v", "./testdata/testinternal")
	tg.grepBoth("use of internal package not allowed", "wrong error message for testdata/testinternal")
}

func TestInternalPackagesOutsideGOROOTAreRespected(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("build", "-v", "./testdata/testinternal2")
	tg.grepBoth("use of internal package not allowed", "wrote error message for testdata/testinternal2")
}

func TestRunInternal(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	dir := filepath.Join(tg.pwd(), "testdata")
	tg.setenv("GOPATH", dir)
	tg.run("run", filepath.Join(dir, "src/run/good.go"))
	tg.runFail("run", filepath.Join(dir, "src/run/bad.go"))
	tg.grepStderr("use of internal package not allowed", "unexpected error for run/bad.go")
}

func testMove(t *testing.T, vcs, url, base, config string) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
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
		data = bytes.Replace(data, []byte(base), []byte(base+"XXX"), -1)
		tg.must(ioutil.WriteFile(path, data, 0644))
	}
	if vcs == "git" {
		// git will ask for a username and password when we
		// run go get -d -f -u.  An empty username and
		// password will work.  Prevent asking by setting
		// GIT_ASKPASS.
		tg.creatingTemp("sink" + exeSuffix)
		tg.tempFile("src/sink/sink.go", `package main; func main() {}`)
		tg.run("build", "-o", "sink"+exeSuffix, "sink")
		tg.setenv("GIT_ASKPASS", filepath.Join(tg.pwd(), "sink"+exeSuffix))
	}
	tg.runFail("get", "-d", "-u", url)
	tg.grepStderr("is a custom import path for", "go get -d -u "+url+" failed for wrong reason")
	tg.runFail("get", "-d", "-f", "-u", url)
	tg.grepStderr("validating server certificate|not found", "go get -d -f -u "+url+" failed for wrong reason")
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

// TODO(rsc): Set up a test case on bitbucket for hg.
// func TestMoveHG(t *testing.T) {
// 	testMove(t, "hg", "rsc.io/x86/x86asm", "x86", "rsc.io/x86/.hg/hgrc")
// }

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

// cmd/go: custom import path checking should not apply to github.com/xxx/yyy.
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
	defer tg.resetReadOnlyFlagAll(repoDir)
	tg.runGit(repoDir, "remote", "set-url", "origin", "https://"+importPath+".git")
	tg.run("get", "-d", "-u", importPath)
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
	defer tg.resetReadOnlyFlagAll(repoDir)
	tg.runGit(repoDir, "branch", "--contains", "HEAD")
	tg.grepStdout(`\* another-branch`, "not on correct default branch")

	tg.run("get", "-d", "-u", importPath)
	tg.runGit(repoDir, "branch", "--contains", "HEAD")
	tg.grepStdout(`\* another-branch`, "not on correct default branch")
}

func TestDisallowedCSourceFiles(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("build", "badc")
	tg.grepStderr("C source files not allowed", "go test did not say C source files not allowed")
}

func TestErrorMessageForSyntaxErrorInTestGoFileSaysFAIL(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("test", "syntaxerror")
	tg.grepStderr("FAIL", "go test did not say FAIL")
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
	tg.run("test", "-i", "./testdata/testimport")
}

func TestRelativeImportsInCommandLinePackage(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	files, err := filepath.Glob("./testdata/testimport/*.go")
	tg.must(err)
	tg.run(append([]string{"test"}, files...)...)
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
	tg.grepStderr("no buildable Go source files", "go install cgotest did not report 'no buildable Go Source files'")
}

func TestRelativeGOBINFail(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("triv.go", `package main; func main() {}`)
	tg.setenv("GOBIN", ".")
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
	if runtime.GOOS == "plan9" {
		t.Skip("skipping because /dev/null is a regular file on plan9")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("build", "-o", os.DevNull, "go-cmd-test")
}

func TestPackageMainTestImportsArchiveNotBinary(t *testing.T) {
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
	tg.wantNotStale("main_test", "after go install, main listed as stale")
	tg.run("test", "main_test")
}

// Issue 12690
func TestPackageNotStaleWithTrailingSlash(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	goroot := runtime.GOROOT()
	tg.setenv("GOROOT", goroot+"/")
	tg.wantNotStale("runtime", "with trailing slash in GOROOT, runtime listed as stale")
	tg.wantNotStale("os", "with trailing slash in GOROOT, os listed as stale")
	tg.wantNotStale("io", "with trailing slash in GOROOT, io listed as stale")
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

func TestGodocInstalls(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	// godoc installs into GOBIN
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("gobin")
	tg.setenv("GOPATH", tg.path("."))
	tg.setenv("GOBIN", tg.path("gobin"))
	tg.run("get", "golang.org/x/tools/cmd/godoc")
	tg.wantExecutable(tg.path("gobin/godoc"), "did not install godoc to $GOBIN")
	tg.unsetenv("GOBIN")

	// godoc installs into GOROOT
	goroot := runtime.GOROOT()
	tg.setenv("GOROOT", goroot)
	tg.check(os.RemoveAll(filepath.Join(goroot, "bin", "godoc")))
	tg.run("install", "golang.org/x/tools/cmd/godoc")
	tg.wantExecutable(filepath.Join(goroot, "bin", "godoc"), "did not install godoc to $GOROOT/bin")
}

func TestGoGetNonPkg(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempDir("gobin")
	tg.setenv("GOPATH", tg.path("."))
	tg.setenv("GOBIN", tg.path("gobin"))
	tg.runFail("get", "-d", "golang.org/x/tools")
	tg.grepStderr("golang.org/x/tools: no buildable Go source files", "missing error")
	tg.runFail("get", "-d", "-u", "golang.org/x/tools")
	tg.grepStderr("golang.org/x/tools: no buildable Go source files", "missing error")
	tg.runFail("get", "-d", "golang.org/x/tools")
	tg.grepStderr("golang.org/x/tools: no buildable Go source files", "missing error")
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

// Issue 4104.
func TestGoTestWithPackageListedMultipleTimes(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.run("test", "errors", "errors", "errors", "errors", "errors")
	if strings.Index(strings.TrimSpace(tg.getStdout()), "\n") != -1 {
		t.Error("go test errors errors errors errors errors tested the same package multiple times")
	}
}

func TestGoListHasAConsistentOrder(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("list", "std")
	first := tg.getStdout()
	tg.run("list", "std")
	if first != tg.getStdout() {
		t.Error("go list std ordering is inconsistent")
	}
}

func TestGoListStdDoesNotIncludeCommands(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("list", "std")
	tg.grepStdoutNot("cmd/", "go list std shows commands")
}

func TestGoListCmdOnlyShowsCommands(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("list", "cmd")
	out := strings.TrimSpace(tg.getStdout())
	for _, line := range strings.Split(out, "\n") {
		if strings.Index(line, "cmd/") == -1 {
			t.Error("go list cmd shows non-commands")
			break
		}
	}
}

func TestGoListDedupsPackages(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.run("list", "xtestonly", "./testdata/src/xtestonly/...")
	got := strings.TrimSpace(tg.getStdout())
	const want = "xtestonly"
	if got != want {
		t.Errorf("got %q; want %q", got, want)
	}
}

// Issue 4096. Validate the output of unsuccessful go install foo/quxx.
func TestUnsuccessfulGoInstallShouldMentionMissingPackage(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(`cannot find package "foo/quxx" in any of`) != 1 {
		t.Error(`go install foo/quxx expected error: .*cannot find package "foo/quxx" in any of`)
	}
}

func TestGOROOTSearchFailureReporting(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(regexp.QuoteMeta(filepath.Join("foo", "quxx"))+` \(from \$GOROOT\)$`) != 1 {
		t.Error(`go install foo/quxx expected error: .*foo/quxx (from $GOROOT)`)
	}
}

func TestMultipleGOPATHEntriesReportedSeparately(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
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
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata", "a")+sep+filepath.Join(tg.pwd(), "testdata", "b"))
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(regexp.QuoteMeta(filepath.Join("testdata", "b", "src", "foo", "quxx"))+`$`) != 1 {
		t.Error(`go install foo/quxx expected error: .*testdata/b/src/foo/quxx`)
	}
}

// Test missing GOPATH is reported.
func TestMissingGOPATHIsReported(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GOPATH", "")
	tg.runFail("install", "foo/quxx")
	if tg.grepCountBoth(`\(\$GOPATH not set\)$`) != 1 {
		t.Error(`go install foo/quxx expected error: ($GOPATH not set)`)
	}
}

// Issue 4186.  go get cannot be used to download packages to $GOROOT.
// Test that without GOPATH set, go get should fail.
func TestWithoutGOPATHGoGetFails(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
	tg.setenv("GOPATH", "")
	tg.setenv("GOROOT", tg.path("."))
	tg.runFail("get", "-d", "golang.org/x/codereview/cmd/hgpatch")
}

// Test that with GOPATH=$GOROOT, go get should fail.
func TestWithGOPATHEqualsGOROOTGoGetFails(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
	tg.setenv("GOPATH", tg.path("."))
	tg.setenv("GOROOT", tg.path("."))
	tg.runFail("get", "-d", "golang.org/x/codereview/cmd/hgpatch")
}

func TestLdflagsArgumentsWithSpacesIssue3941(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("main.go", `package main
		var extern string
		func main() {
			println(extern)
		}`)
	tg.run("run", "-ldflags", `-X main.extern "hello world"`, tg.path("main.go"))
	tg.grepStderr("^hello world", `ldflags -X main.extern 'hello world' failed`)
}

func TestGoTestCpuprofileLeavesBinaryBehind(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.run("test", "-cpuprofile", "errors.prof", "errors")
	tg.wantExecutable("errors.test"+exeSuffix, "go test -cpuprofile did not create errors.test")
}

func TestGoTestCpuprofileDashOControlsBinaryLocation(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.run("test", "-cpuprofile", "errors.prof", "-o", "myerrors.test"+exeSuffix, "errors")
	tg.wantExecutable("myerrors.test"+exeSuffix, "go test -cpuprofile -o myerrors.test did not create myerrors.test")
}

func TestGoTestDashCDashOControlsBinaryLocation(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.run("test", "-c", "-o", tg.path("myerrors.test"+exeSuffix), "errors")
	tg.wantExecutable(tg.path("myerrors.test"+exeSuffix), "go test -c -o myerrors.test did not create myerrors.test")
}

func TestGoTestDashOWritesBinary(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.run("test", "-o", tg.path("myerrors.test"+exeSuffix), "errors")
	tg.wantExecutable(tg.path("myerrors.test"+exeSuffix), "go test -o myerrors.test did not create myerrors.test")
}

// Issue 4568.
func TestSymlinksList(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping symlink test on %s", runtime.GOOS)
	}

	tg := testgo(t)
	defer tg.cleanup()
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
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping symlink test on %s", runtime.GOOS)
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.setenv("GO15VENDOREXPERIMENT", "1")
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

func TestSymlinksInternal(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping symlink test on %s", runtime.GOOS)
	}

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
	tg.runFail("list", "example/a")
	tg.grepStderr("case-insensitive import collision", "go list example/a did not report import collision")
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
}

// Issue 8181.
func TestGoGetDashTIssue8181(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping test that uses network in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "-v", "-t", "github.com/rsc/go-get-issue-8181/a", "github.com/rsc/go-get-issue-8181/b")
	tg.run("list", "...")
	tg.grepStdout("x/build/cmd/cl", "missing expected x/build/cmd/cl")
}

func TestIssue11307(t *testing.T) {
	// go get -u was not working except in checkout directory
	if testing.Short() {
		t.Skip("skipping test that uses network in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "github.com/rsc/go-get-issue-11307")
	tg.run("get", "-u", "github.com/rsc/go-get-issue-11307") // was failing
}

func TestShadowingLogic(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	pwd := tg.pwd()
	sep := string(filepath.ListSeparator)
	tg.setenv("GOPATH", filepath.Join(pwd, "testdata", "shadow", "root1")+sep+filepath.Join(pwd, "testdata", "shadow", "root2"))

	// The math in root1 is not "math" because the standard math is.
	tg.run("list", "-f", "({{.ImportPath}}) ({{.ConflictDir}})", "./testdata/shadow/root1/src/math")
	pwdForwardSlash := strings.Replace(pwd, string(os.PathSeparator), "/", -1)
	if !strings.HasPrefix(pwdForwardSlash, "/") {
		pwdForwardSlash = "/" + pwdForwardSlash
	}
	// The output will have makeImportValid applies, but we only
	// bother to deal with characters we might reasonably see.
	pwdForwardSlash = strings.Replace(pwdForwardSlash, ":", "_", -1)
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
	if regexp.MustCompile(`[^0-9]0\.0%`).MatchString(data) {
		tg.t.Error("some coverage results are 0.0%")
	}
	tg.t.Log(data)
}

func TestCoverageRuns(t *testing.T) {
	if testing.Short() {
		t.Skip("don't build libraries for coverage in short mode")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-short", "-coverpkg=strings", "strings", "regexp")
	data := tg.getStdout() + tg.getStderr()
	tg.run("test", "-short", "-cover", "strings", "math", "regexp")
	data += tg.getStdout() + tg.getStderr()
	checkCoverage(tg, data)
}

// Check that coverage analysis uses set mode.
func TestCoverageUsesSetMode(t *testing.T) {
	if testing.Short() {
		t.Skip("don't build libraries for coverage in short mode")
	}
	tg := testgo(t)
	defer tg.cleanup()
	tg.creatingTemp("testdata/cover.out")
	tg.run("test", "-short", "-cover", "encoding/binary", "-coverprofile=testdata/cover.out")
	data := tg.getStdout() + tg.getStderr()
	if out, err := ioutil.ReadFile("testdata/cover.out"); err != nil {
		t.Error(err)
	} else {
		if !bytes.Contains(out, []byte("mode: set")) {
			t.Error("missing mode: set")
		}
	}
	checkCoverage(tg, data)
}

func TestCoverageUsesAtomicModeForRace(t *testing.T) {
	if testing.Short() {
		t.Skip("don't build libraries for coverage in short mode")
	}
	if !canRace {
		t.Skip("skipping because race detector not supported")
	}

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

func TestCoverageUsesActualSettingToOverrideEvenForRace(t *testing.T) {
	if testing.Short() {
		t.Skip("don't build libraries for coverage in short mode")
	}
	if !canRace {
		t.Skip("skipping because race detector not supported")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.creatingTemp("testdata/cover.out")
	tg.run("test", "-short", "-race", "-cover", "encoding/binary", "-covermode=count", "-coverprofile=testdata/cover.out")
	data := tg.getStdout() + tg.getStderr()
	if out, err := ioutil.ReadFile("testdata/cover.out"); err != nil {
		t.Error(err)
	} else {
		if !bytes.Contains(out, []byte("mode: count")) {
			t.Error("missing mode: count")
		}
	}
	checkCoverage(tg, data)
}

func TestCoverageWithCgo(t *testing.T) {
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.run("test", "-short", "-cover", "./testdata/cgocover")
	data := tg.getStdout() + tg.getStderr()
	checkCoverage(tg, data)
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
	if !canCgo {
		t.Skip("skipping because cgo not enabled")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/origin/origin.go", `package origin
		// #cgo !darwin LDFLAGS: -Wl,-rpath -Wl,$ORIGIN
		// void f(void) {}
		import "C"
		func f() { C.f() }`)
	tg.setenv("GOPATH", tg.path("."))
	tg.run("build", "origin")
}

// "go test -c -test.bench=XXX errors" should not hang
func TestIssue6480(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.cd(tg.path("."))
	tg.run("test", "-c", "-test.bench=XXX", "errors")
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
	tg.grepStderr(`gccgo.*\-L alibpath \-lalib`, `no Go-inline "#cgo LDFLAGS:" ("-L alibpath -lalib") passed to gccgo linking stage`)
}

func TestListTemplateCanUseContextFunction(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.run("list", "-f", "GOARCH: {{context.GOARCH}}")
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
		t.Skip("don't rebuild the standard libary in short mode")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.creatingTemp("deps.test" + exeSuffix)
	tg.run("test", "-x", "-a", "-c", "testdata/dep_test.go")
	tg.grepStderr("regexp", "go test -x -a -c testdata/dep-test.go did not rebuild regexp")
}

func TestBuildDashIInstallsDependencies(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempFile("src/x/y/foo/foo.go", `package foo
		func F() {}`)
	tg.tempFile("src/x/y/bar/bar.go", `package bar
		import "x/y/foo"
		func F() { foo.F() }`)
	tg.setenv("GOPATH", tg.path("."))

	checkbar := func(desc string) {
		tg.sleep()
		tg.must(os.Chtimes(tg.path("src/x/y/foo/foo.go"), time.Now(), time.Now()))
		tg.sleep()
		tg.run("build", "-v", "-i", "x/y/bar")
		tg.grepBoth("x/y/foo", "first build -i "+desc+" did not build x/y/foo")
		tg.run("build", "-v", "-i", "x/y/bar")
		tg.grepBothNot("x/y/foo", "second build -i "+desc+" built x/y/foo")
	}
	checkbar("pkg")
	tg.creatingTemp("bar" + exeSuffix)
	tg.tempFile("src/x/y/bar/bar.go", `package main
		import "x/y/foo"
		func main() { foo.F() }`)
	checkbar("cmd")
}

func TestGoBuildInTestOnlyDirectoryFailsWithAGoodError(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.runFail("build", "./testdata/testonly")
	tg.grepStderr("no buildable Go", "go build ./testdata/testonly produced unexpected error")
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

func TestGoTestFlagsAfterPackage(t *testing.T) {
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
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "golang.org/x/tools/cmd/vet")
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("vet", "vetpkg")
	tg.grepBoth("missing argument for Printf", "go vet vetpkg did not find missing argument for Printf")
}

func TestGoVetWithTags(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "golang.org/x/tools/cmd/vet")
	tg.setenv("GOPATH", filepath.Join(tg.pwd(), "testdata"))
	tg.runFail("vet", "-tags", "tagtest", "vetpkg")
	tg.grepBoth(`c\.go.*wrong number of args for format`, "go get vetpkg did not run scan tagged file")
}

// Issue 9767.
func TestGoGetRscIoToolstash(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempDir("src/rsc.io")
	tg.setenv("GOPATH", tg.path("."))
	tg.cd(tg.path("src/rsc.io"))
	tg.run("get", "./toolstash")
}

// Issue 13037: Was not parsing <meta> tags in 404 served over HTTPS
func TestGoGetHTTPS404(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.tempDir("src")
	tg.setenv("GOPATH", tg.path("."))
	tg.run("get", "bazil.org/fuse/fs/fstestutil")
}

// Test that you can not import a main package.
func TestIssue4210(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	tg.tempFile("src/x/main.go", `package main
		var X int
		func main() {}`)
	tg.tempFile("src/y/main.go", `package main
		import "fmt"
		import xmain "x"
		func main() {
			fmt.Println(xmain.X)
		}`)
	tg.setenv("GOPATH", tg.path("."))
	tg.runFail("build", "y")
	tg.grepBoth("is a program", `did not find expected error message ("is a program")`)
}

func TestGoGetInsecure(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))
	tg.failSSH()

	const repo = "wh3rd.net/git.git"

	// Try go get -d of HTTP-only repo (should fail).
	tg.runFail("get", "-d", repo)

	// Try again with -insecure (should succeed).
	tg.run("get", "-d", "-insecure", repo)

	// Try updating without -insecure (should fail).
	tg.runFail("get", "-d", "-u", "-f", repo)
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

func TestGoGetInsecureCustomDomain(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	tg.setenv("GOPATH", tg.path("."))

	const repo = "wh3rd.net/repo"
	tg.runFail("get", "-d", repo)
	tg.run("get", "-d", "-insecure", repo)
}

func TestIssue10193(t *testing.T) {
	t.Skip("depends on code.google.com")
	testenv.MustHaveExternalNetwork(t)
	if _, err := exec.LookPath("hg"); err != nil {
		t.Skip("skipping because hg binary not found")
	}

	tg := testgo(t)
	defer tg.cleanup()
	tg.parallel()
	tg.tempDir("src")
	tg.setenv("GOPATH", tg.path("."))
	tg.runFail("get", "code.google.com/p/rsc/pdf")
	tg.grepStderr("is shutting down", "missed warning about code.google.com")
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
	tg := testgo(t)
	defer tg.cleanup()
	tg.makeTempdir()
	pkg := tg.path(".")
	tg.run("install", "-pkgdir", pkg, "errors")
	_, err := os.Stat(filepath.Join(pkg, "errors.a"))
	tg.must(err)
	_, err = os.Stat(filepath.Join(pkg, "runtime.a"))
	tg.must(err)
}

func TestGoTestRaceInstallCgo(t *testing.T) {
	switch sys := runtime.GOOS + "/" + runtime.GOARCH; sys {
	case "darwin/amd64", "freebsd/amd64", "linux/amd64", "windows/amd64":
		// ok
	default:
		t.Skip("no race detector on %s", sys)
	}

	if !build.Default.CgoEnabled {
		t.Skip("no race detector without cgo")
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
	if new.ModTime() != old.ModTime() {
		t.Fatalf("go test -i runtime/race reinstalled cmd/cgo")
	}
}

func TestGoTestImportErrorStack(t *testing.T) {
	const out = `package testdep/p1 (test)
	imports testdep/p2
	imports testdep/p3: no buildable Go source files`

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

func TestIssue13655(t *testing.T) {
	tg := testgo(t)
	defer tg.cleanup()
	for _, pkg := range []string{"runtime", "runtime/internal/atomic"} {
		tg.run("list", "-f", "{{.Deps}}", pkg)
		tg.grepStdout("runtime/internal/sys", "did not find required dependency of "+pkg+" on runtime/internal/sys")
	}
}

// For issue 14337.
func TestParallelTest(t *testing.T) {
	tg := testgo(t)
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
