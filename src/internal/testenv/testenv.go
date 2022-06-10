// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package testenv provides information about what functionality
// is available in different testing environments run by the Go team.
//
// It is an internal package because these details are specific
// to the Go team's test setup (on build.golang.org) and not
// fundamental to tests in general.
package testenv

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"internal/cfg"
	"internal/testmath"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// Builder reports the name of the builder running this test
// (for example, "linux-amd64" or "windows-386-gce").
// If the test is not running on the build infrastructure,
// Builder returns the empty string.
func Builder() string {
	return os.Getenv("GO_BUILDER_NAME")
}

// HasGoBuild reports whether the current system can build programs with “go build”
// and then run them with os.StartProcess or exec.Command.
func HasGoBuild() bool {
	if os.Getenv("GO_GCFLAGS") != "" {
		// It's too much work to require every caller of the go command
		// to pass along "-gcflags="+os.Getenv("GO_GCFLAGS").
		// For now, if $GO_GCFLAGS is set, report that we simply can't
		// run go build.
		return false
	}
	switch runtime.GOOS {
	case "android", "js", "ios":
		return false
	}
	return true
}

// MustHaveGoBuild checks that the current system can build programs with “go build”
// and then run them with os.StartProcess or exec.Command.
// If not, MustHaveGoBuild calls t.Skip with an explanation.
func MustHaveGoBuild(t testing.TB) {
	if os.Getenv("GO_GCFLAGS") != "" {
		t.Skipf("skipping test: 'go build' not compatible with setting $GO_GCFLAGS")
	}
	if !HasGoBuild() {
		t.Skipf("skipping test: 'go build' not available on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

// HasGoRun reports whether the current system can run programs with “go run.”
func HasGoRun() bool {
	// For now, having go run and having go build are the same.
	return HasGoBuild()
}

// MustHaveGoRun checks that the current system can run programs with “go run.”
// If not, MustHaveGoRun calls t.Skip with an explanation.
func MustHaveGoRun(t testing.TB) {
	if !HasGoRun() {
		t.Skipf("skipping test: 'go run' not available on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

// GoToolPath reports the path to the Go tool.
// It is a convenience wrapper around GoTool.
// If the tool is unavailable GoToolPath calls t.Skip.
// If the tool should be available and isn't, GoToolPath calls t.Fatal.
func GoToolPath(t testing.TB) string {
	MustHaveGoBuild(t)
	path, err := GoTool()
	if err != nil {
		t.Fatal(err)
	}
	// Add all environment variables that affect the Go command to test metadata.
	// Cached test results will be invalidate when these variables change.
	// See golang.org/issue/32285.
	for _, envVar := range strings.Fields(cfg.KnownEnv) {
		os.Getenv(envVar)
	}
	return path
}

var (
	gorootOnce sync.Once
	gorootPath string
	gorootErr  error
)

func findGOROOT() (string, error) {
	gorootOnce.Do(func() {
		gorootPath = runtime.GOROOT()
		if gorootPath != "" {
			// If runtime.GOROOT() is non-empty, assume that it is valid.
			//
			// (It might not be: for example, the user may have explicitly set GOROOT
			// to the wrong directory, or explicitly set GOROOT_FINAL but not GOROOT
			// and hasn't moved the tree to GOROOT_FINAL yet. But those cases are
			// rare, and if that happens the user can fix what they broke.)
			return
		}

		// runtime.GOROOT doesn't know where GOROOT is (perhaps because the test
		// binary was built with -trimpath, or perhaps because GOROOT_FINAL was set
		// without GOROOT and the tree hasn't been moved there yet).
		//
		// Since this is internal/testenv, we can cheat and assume that the caller
		// is a test of some package in a subdirectory of GOROOT/src. ('go test'
		// runs the test in the directory containing the packaged under test.) That
		// means that if we start walking up the tree, we should eventually find
		// GOROOT/src/go.mod, and we can report the parent directory of that.

		cwd, err := os.Getwd()
		if err != nil {
			gorootErr = fmt.Errorf("finding GOROOT: %w", err)
			return
		}

		dir := cwd
		for {
			parent := filepath.Dir(dir)
			if parent == dir {
				// dir is either "." or only a volume name.
				gorootErr = fmt.Errorf("failed to locate GOROOT/src in any parent directory")
				return
			}

			if base := filepath.Base(dir); base != "src" {
				dir = parent
				continue // dir cannot be GOROOT/src if it doesn't end in "src".
			}

			b, err := os.ReadFile(filepath.Join(dir, "go.mod"))
			if err != nil {
				if os.IsNotExist(err) {
					dir = parent
					continue
				}
				gorootErr = fmt.Errorf("finding GOROOT: %w", err)
				return
			}
			goMod := string(b)

			for goMod != "" {
				var line string
				line, goMod, _ = strings.Cut(goMod, "\n")
				fields := strings.Fields(line)
				if len(fields) >= 2 && fields[0] == "module" && fields[1] == "std" {
					// Found "module std", which is the module declaration in GOROOT/src!
					gorootPath = parent
					return
				}
			}
		}
	})

	return gorootPath, gorootErr
}

// GOROOT reports the path to the directory containing the root of the Go
// project source tree. This is normally equivalent to runtime.GOROOT, but
// works even if the test binary was built with -trimpath.
//
// If GOROOT cannot be found, GOROOT skips t if t is non-nil,
// or panics otherwise.
func GOROOT(t testing.TB) string {
	path, err := findGOROOT()
	if err != nil {
		if t == nil {
			panic(err)
		}
		t.Helper()
		t.Skip(err)
	}
	return path
}

// GoTool reports the path to the Go tool.
func GoTool() (string, error) {
	if !HasGoBuild() {
		return "", errors.New("platform cannot run go tool")
	}
	var exeSuffix string
	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}
	goroot, err := findGOROOT()
	if err != nil {
		return "", fmt.Errorf("cannot find go tool: %w", err)
	}
	path := filepath.Join(goroot, "bin", "go"+exeSuffix)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	goBin, err := exec.LookPath("go" + exeSuffix)
	if err != nil {
		return "", errors.New("cannot find go tool: " + err.Error())
	}
	return goBin, nil
}

// HasExec reports whether the current system can start new processes
// using os.StartProcess or (more commonly) exec.Command.
func HasExec() bool {
	switch runtime.GOOS {
	case "js", "ios":
		return false
	}
	return true
}

// HasSrc reports whether the entire source tree is available under GOROOT.
func HasSrc() bool {
	switch runtime.GOOS {
	case "ios":
		return false
	}
	return true
}

// MustHaveExec checks that the current system can start new processes
// using os.StartProcess or (more commonly) exec.Command.
// If not, MustHaveExec calls t.Skip with an explanation.
func MustHaveExec(t testing.TB) {
	if !HasExec() {
		t.Skipf("skipping test: cannot exec subprocess on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

var execPaths sync.Map // path -> error

// MustHaveExecPath checks that the current system can start the named executable
// using os.StartProcess or (more commonly) exec.Command.
// If not, MustHaveExecPath calls t.Skip with an explanation.
func MustHaveExecPath(t testing.TB, path string) {
	MustHaveExec(t)

	err, found := execPaths.Load(path)
	if !found {
		_, err = exec.LookPath(path)
		err, _ = execPaths.LoadOrStore(path, err)
	}
	if err != nil {
		t.Skipf("skipping test: %s: %s", path, err)
	}
}

// HasExternalNetwork reports whether the current system can use
// external (non-localhost) networks.
func HasExternalNetwork() bool {
	return !testing.Short() && runtime.GOOS != "js"
}

// MustHaveExternalNetwork checks that the current system can use
// external (non-localhost) networks.
// If not, MustHaveExternalNetwork calls t.Skip with an explanation.
func MustHaveExternalNetwork(t testing.TB) {
	if runtime.GOOS == "js" {
		t.Skipf("skipping test: no external network on %s", runtime.GOOS)
	}
	if testing.Short() {
		t.Skipf("skipping test: no external network in -short mode")
	}
}

var haveCGO bool

// HasCGO reports whether the current system can use cgo.
func HasCGO() bool {
	return haveCGO
}

// MustHaveCGO calls t.Skip if cgo is not available.
func MustHaveCGO(t testing.TB) {
	if !haveCGO {
		t.Skipf("skipping test: no cgo")
	}
}

// CanInternalLink reports whether the current system can link programs with
// internal linking.
// (This is the opposite of cmd/internal/sys.MustLinkExternal. Keep them in sync.)
func CanInternalLink() bool {
	switch runtime.GOOS {
	case "android":
		if runtime.GOARCH != "arm64" {
			return false
		}
	case "ios":
		if runtime.GOARCH == "arm64" {
			return false
		}
	}
	return true
}

// MustInternalLink checks that the current system can link programs with internal
// linking.
// If not, MustInternalLink calls t.Skip with an explanation.
func MustInternalLink(t testing.TB) {
	if !CanInternalLink() {
		t.Skipf("skipping test: internal linking on %s/%s is not supported", runtime.GOOS, runtime.GOARCH)
	}
}

// HasSymlink reports whether the current system can use os.Symlink.
func HasSymlink() bool {
	ok, _ := hasSymlink()
	return ok
}

// MustHaveSymlink reports whether the current system can use os.Symlink.
// If not, MustHaveSymlink calls t.Skip with an explanation.
func MustHaveSymlink(t testing.TB) {
	ok, reason := hasSymlink()
	if !ok {
		t.Skipf("skipping test: cannot make symlinks on %s/%s%s", runtime.GOOS, runtime.GOARCH, reason)
	}
}

// HasLink reports whether the current system can use os.Link.
func HasLink() bool {
	// From Android release M (Marshmallow), hard linking files is blocked
	// and an attempt to call link() on a file will return EACCES.
	// - https://code.google.com/p/android-developer-preview/issues/detail?id=3150
	return runtime.GOOS != "plan9" && runtime.GOOS != "android"
}

// MustHaveLink reports whether the current system can use os.Link.
// If not, MustHaveLink calls t.Skip with an explanation.
func MustHaveLink(t testing.TB) {
	if !HasLink() {
		t.Skipf("skipping test: hardlinks are not supported on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

var flaky = flag.Bool("flaky", false, "run known-flaky tests too")

func SkipFlaky(t testing.TB, issue int) {
	t.Helper()
	if !*flaky {
		t.Skipf("skipping known flaky test without the -flaky flag; see golang.org/issue/%d", issue)
	}
}

func SkipFlakyNet(t testing.TB) {
	t.Helper()
	if v, _ := strconv.ParseBool(os.Getenv("GO_BUILDER_FLAKY_NET")); v {
		t.Skip("skipping test on builder known to have frequent network failures")
	}
}

// CleanCmdEnv will fill cmd.Env with the environment, excluding certain
// variables that could modify the behavior of the Go tools such as
// GODEBUG and GOTRACEBACK.
func CleanCmdEnv(cmd *exec.Cmd) *exec.Cmd {
	if cmd.Env != nil {
		panic("environment already set")
	}
	for _, env := range os.Environ() {
		// Exclude GODEBUG from the environment to prevent its output
		// from breaking tests that are trying to parse other command output.
		if strings.HasPrefix(env, "GODEBUG=") {
			continue
		}
		// Exclude GOTRACEBACK for the same reason.
		if strings.HasPrefix(env, "GOTRACEBACK=") {
			continue
		}
		cmd.Env = append(cmd.Env, env)
	}
	return cmd
}

// CPUIsSlow reports whether the CPU running the test is suspected to be slow.
func CPUIsSlow() bool {
	switch runtime.GOARCH {
	case "arm", "mips", "mipsle", "mips64", "mips64le":
		return true
	}
	return false
}

// SkipIfShortAndSlow skips t if -short is set and the CPU running the test is
// suspected to be slow.
//
// (This is useful for CPU-intensive tests that otherwise complete quickly.)
func SkipIfShortAndSlow(t testing.TB) {
	if testing.Short() && CPUIsSlow() {
		t.Helper()
		t.Skipf("skipping test in -short mode on %s", runtime.GOARCH)
	}
}

// RunWithTimeout runs cmd and returns its combined output. If the
// subprocess exits with a non-zero status, it will log that status
// and return a non-nil error, but this is not considered fatal.
func RunWithTimeout(t testing.TB, cmd *exec.Cmd) ([]byte, error) {
	args := cmd.Args
	if args == nil {
		args = []string{cmd.Path}
	}

	var b bytes.Buffer
	cmd.Stdout = &b
	cmd.Stderr = &b
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting %s: %v", args, err)
	}

	// If the process doesn't complete within 1 minute,
	// assume it is hanging and kill it to get a stack trace.
	p := cmd.Process
	done := make(chan bool)
	go func() {
		scale := 1
		// This GOARCH/GOOS test is copied from cmd/dist/test.go.
		// TODO(iant): Have cmd/dist update the environment variable.
		if runtime.GOARCH == "arm" || runtime.GOOS == "windows" {
			scale = 2
		}
		if s := os.Getenv("GO_TEST_TIMEOUT_SCALE"); s != "" {
			if sc, err := strconv.Atoi(s); err == nil {
				scale = sc
			}
		}

		select {
		case <-done:
		case <-time.After(time.Duration(scale) * time.Minute):
			p.Signal(Sigquit)
			// If SIGQUIT doesn't do it after a little
			// while, kill the process.
			select {
			case <-done:
			case <-time.After(time.Duration(scale) * 30 * time.Second):
				p.Signal(os.Kill)
			}
		}
	}()

	err := cmd.Wait()
	if err != nil {
		t.Logf("%s exit status: %v", args, err)
	}
	close(done)

	return b.Bytes(), err
}

// CheckLinear checks if the function produced by f scales linearly.
//
// f must accept a scale factor which causes the input to the function it
// produces to scale by that factor.
func CheckLinear(t *testing.T, f func(scale float64) func(*testing.B)) {
	MustHaveExec(t)

	if os.Getenv("GO_PERF_UNIT_TEST") == "" {
		// Invoke the same test as a subprocess with the GO_PERF_UNIT_TEST environment variable set.
		// We create a subprocess for two reasons:
		//
		//   1. There's no other way to set the benchmarking parameters of testing.Benchmark.
		//   2. Since we're effectively running a performance test, running in a subprocess grants
		//      us a little bit more isolation than using the same process.
		//
		// As an alternative, we could fairly easily reimplement the timing code in testing.Benchmark,
		// but a subprocess is just as easy to create.

		selfCmd := CleanCmdEnv(exec.Command(os.Args[0], "-test.v", fmt.Sprintf("-test.run=^%s$", t.Name()), "-test.benchtime=1x"))
		selfCmd.Env = append(selfCmd.Env, "GO_PERF_UNIT_TEST=1")
		output, err := RunWithTimeout(t, selfCmd)
		if err != nil {
			t.Error(err)
			t.Logf("--- subprocess output ---\n%s", string(output))
		}
		if bytes.Contains(output, []byte("insignificant result")) {
			t.Skip("insignificant result")
		}
		return
	}

	// Pick a reasonable sample count.
	const count = 10

	// Collect samples for scale factor 1.
	x1 := make([]testing.BenchmarkResult, 0, count)
	for i := 0; i < count; i++ {
		x1 = append(x1, testing.Benchmark(f(1.0)))
	}

	// Collect samples for scale factor 2.
	x2 := make([]testing.BenchmarkResult, 0, count)
	for i := 0; i < count; i++ {
		x2 = append(x2, testing.Benchmark(f(2.0)))
	}

	// Run a t-test on the results.
	r1 := testmath.BenchmarkResults(x1)
	r2 := testmath.BenchmarkResults(x2)
	result, err := testmath.TwoSampleWelchTTest(r1, r2, testmath.LocationDiffers)
	if err != nil {
		t.Fatalf("failed to run t-test: %v", err)
	}
	if result.P > 0.005 {
		// Insignificant result.
		t.Skip("insignificant result")
	}

	// Let ourselves be within 3x; 2x is too strict.
	if m1, m2 := r1.Mean(), r2.Mean(); 3.0*m1 < m2 {
		t.Fatalf("failure to scale linearly: µ_1=%s µ_2=%s p=%f", time.Duration(m1), time.Duration(m2), result.P)
	}
}
