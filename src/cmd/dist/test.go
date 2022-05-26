// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

func cmdtest() {
	gogcflags = os.Getenv("GO_GCFLAGS")
	setNoOpt()

	var t tester

	var noRebuild bool
	flag.BoolVar(&t.listMode, "list", false, "list available tests")
	flag.BoolVar(&t.rebuild, "rebuild", false, "rebuild everything first")
	flag.BoolVar(&noRebuild, "no-rebuild", false, "overrides -rebuild (historical dreg)")
	flag.BoolVar(&t.keepGoing, "k", false, "keep going even when error occurred")
	flag.BoolVar(&t.race, "race", false, "run in race builder mode (different set of tests)")
	flag.BoolVar(&t.compileOnly, "compile-only", false, "compile tests, but don't run them. This is for some builders. Not all dist tests respect this flag, but most do.")
	flag.StringVar(&t.banner, "banner", "##### ", "banner prefix; blank means no section banners")
	flag.StringVar(&t.runRxStr, "run", os.Getenv("GOTESTONLY"),
		"run only those tests matching the regular expression; empty means to run all. "+
			"Special exception: if the string begins with '!', the match is inverted.")
	flag.BoolVar(&t.msan, "msan", false, "run in memory sanitizer builder mode")
	flag.BoolVar(&t.asan, "asan", false, "run in address sanitizer builder mode")

	xflagparse(-1) // any number of args
	if noRebuild {
		t.rebuild = false
	}

	t.run()
}

// tester executes cmdtest.
type tester struct {
	race        bool
	msan        bool
	asan        bool
	listMode    bool
	rebuild     bool
	failed      bool
	keepGoing   bool
	compileOnly bool // just try to compile all tests, but no need to run
	runRxStr    string
	runRx       *regexp.Regexp
	runRxWant   bool     // want runRx to match (true) or not match (false)
	runNames    []string // tests to run, exclusive with runRx; empty means all
	banner      string   // prefix, or "" for none
	lastHeading string   // last dir heading printed

	cgoEnabled bool
	partial    bool
	haveTime   bool // the 'time' binary is available

	tests        []distTest
	timeoutScale int

	worklist []*work
}

type work struct {
	dt    *distTest
	cmd   *exec.Cmd
	start chan bool
	out   []byte
	err   error
	end   chan bool
}

// A distTest is a test run by dist test.
// Each test has a unique name and belongs to a group (heading)
type distTest struct {
	name    string // unique test name; may be filtered with -run flag
	heading string // group section; this header is printed before the test is run.
	fn      func(*distTest) error
}

func (t *tester) run() {
	timelog("start", "dist test")

	os.Setenv("PATH", fmt.Sprintf("%s%c%s", gorootBin, os.PathListSeparator, os.Getenv("PATH")))

	cmd := exec.Command(gorootBinGo, "env", "CGO_ENABLED")
	cmd.Stderr = new(bytes.Buffer)
	slurp, err := cmd.Output()
	if err != nil {
		fatalf("Error running go env CGO_ENABLED: %v\n%s", err, cmd.Stderr)
	}
	t.cgoEnabled, _ = strconv.ParseBool(strings.TrimSpace(string(slurp)))
	if flag.NArg() > 0 && t.runRxStr != "" {
		fatalf("the -run regular expression flag is mutually exclusive with test name arguments")
	}

	t.runNames = flag.Args()

	if t.hasBash() {
		if _, err := exec.LookPath("time"); err == nil {
			t.haveTime = true
		}
	}

	// Set GOTRACEBACK to system if the user didn't set a level explicitly.
	// Since we're running tests for Go, we want as much detail as possible
	// if something goes wrong.
	//
	// Set it before running any commands just in case something goes wrong.
	if ok := isEnvSet("GOTRACEBACK"); !ok {
		if err := os.Setenv("GOTRACEBACK", "system"); err != nil {
			if t.keepGoing {
				log.Printf("Failed to set GOTRACEBACK: %v", err)
			} else {
				fatalf("Failed to set GOTRACEBACK: %v", err)
			}
		}
	}

	if t.rebuild {
		t.out("Building packages and commands.")
		// Force rebuild the whole toolchain.
		goInstall("go", append([]string{"-a", "-i"}, toolchain...)...)
	}

	if !t.listMode {
		if os.Getenv("GO_BUILDER_NAME") == "" {
			// Complete rebuild bootstrap, even with -no-rebuild.
			// If everything is up-to-date, this is a no-op.
			// If everything is not up-to-date, the first checkNotStale
			// during the test process will kill the tests, so we might
			// as well install the world.
			// Now that for example "go install cmd/compile" does not
			// also install runtime (you need "go install -i cmd/compile"
			// for that), it's easy for previous workflows like
			// "rebuild the compiler and then run run.bash"
			// to break if we don't automatically refresh things here.
			// Rebuilding is a shortened bootstrap.
			// See cmdbootstrap for a description of the overall process.
			goInstall("go", append([]string{"-i"}, toolchain...)...)
			goInstall("go", append([]string{"-i"}, toolchain...)...)
			goInstall("go", "std", "cmd")
		} else {
			// The Go builder infrastructure should always begin running tests from a
			// clean, non-stale state, so there is no need to rebuild the world.
			// Instead, we can just check that it is not stale, which may be less
			// expensive (and is also more likely to catch bugs in the builder
			// implementation).
			willTest := []string{"std"}
			if t.shouldTestCmd() {
				willTest = append(willTest, "cmd")
			}
			checkNotStale("go", willTest...)
		}
	}

	t.timeoutScale = 1
	switch goarch {
	case "arm":
		t.timeoutScale = 2
	case "mips", "mipsle", "mips64", "mips64le":
		t.timeoutScale = 4
	}
	if s := os.Getenv("GO_TEST_TIMEOUT_SCALE"); s != "" {
		t.timeoutScale, err = strconv.Atoi(s)
		if err != nil {
			fatalf("failed to parse $GO_TEST_TIMEOUT_SCALE = %q as integer: %v", s, err)
		}
	}

	if t.runRxStr != "" {
		if t.runRxStr[0] == '!' {
			t.runRxWant = false
			t.runRxStr = t.runRxStr[1:]
		} else {
			t.runRxWant = true
		}
		t.runRx = regexp.MustCompile(t.runRxStr)
	}

	t.registerTests()
	if t.listMode {
		for _, tt := range t.tests {
			fmt.Println(tt.name)
		}
		return
	}

	for _, name := range t.runNames {
		if !t.isRegisteredTestName(name) {
			fatalf("unknown test %q", name)
		}
	}

	// On a few builders, make GOROOT unwritable to catch tests writing to it.
	if strings.HasPrefix(os.Getenv("GO_BUILDER_NAME"), "linux-") {
		if os.Getuid() == 0 {
			// Don't bother making GOROOT unwritable:
			// we're running as root, so permissions would have no effect.
		} else {
			xatexit(t.makeGOROOTUnwritable())
		}
	}

	if err := t.maybeLogMetadata(); err != nil {
		t.failed = true
		if t.keepGoing {
			log.Printf("Failed logging metadata: %v", err)
		} else {
			fatalf("Failed logging metadata: %v", err)
		}
	}

	for _, dt := range t.tests {
		if !t.shouldRunTest(dt.name) {
			t.partial = true
			continue
		}
		dt := dt // dt used in background after this iteration
		if err := dt.fn(&dt); err != nil {
			t.runPending(&dt) // in case that hasn't been done yet
			t.failed = true
			if t.keepGoing {
				log.Printf("Failed: %v", err)
			} else {
				fatalf("Failed: %v", err)
			}
		}
	}
	t.runPending(nil)
	timelog("end", "dist test")

	if t.failed {
		fmt.Println("\nFAILED")
		xexit(1)
	} else if incomplete[goos+"/"+goarch] {
		// The test succeeded, but consider it as failed so we don't
		// forget to remove the port from the incomplete map once the
		// port is complete.
		fmt.Println("\nFAILED (incomplete port)")
		xexit(1)
	} else if t.partial {
		fmt.Println("\nALL TESTS PASSED (some were excluded)")
	} else {
		fmt.Println("\nALL TESTS PASSED")
	}
}

func (t *tester) shouldRunTest(name string) bool {
	if t.runRx != nil {
		return t.runRx.MatchString(name) == t.runRxWant
	}
	if len(t.runNames) == 0 {
		return true
	}
	for _, runName := range t.runNames {
		if runName == name {
			return true
		}
	}
	return false
}

func (t *tester) maybeLogMetadata() error {
	if t.compileOnly {
		// We need to run a subprocess to log metadata. Don't do that
		// on compile-only runs.
		return nil
	}
	t.out("Test execution environment.")
	// Helper binary to print system metadata (CPU model, etc). This is a
	// separate binary from dist so it need not build with the bootstrap
	// toolchain.
	//
	// TODO(prattmic): If we split dist bootstrap and dist test then this
	// could be simplified to directly use internal/sysinfo here.
	return t.dirCmd(filepath.Join(goroot, "src/cmd/internal/metadata"), "go", []string{"run", "main.go"}).Run()
}

// short returns a -short flag value to use with 'go test'
// or a test binary for tests intended to run in short mode.
// It returns "true", unless the environment variable
// GO_TEST_SHORT is set to a non-empty, false-ish string.
//
// This environment variable is meant to be an internal
// detail between the Go build system and cmd/dist for
// the purpose of longtest builders, and is not intended
// for use by users. See golang.org/issue/12508.
func short() string {
	if v := os.Getenv("GO_TEST_SHORT"); v != "" {
		short, err := strconv.ParseBool(v)
		if err != nil {
			fatalf("invalid GO_TEST_SHORT %q: %v", v, err)
		}
		if !short {
			return "false"
		}
	}
	return "true"
}

// goTest returns the beginning of the go test command line.
// Callers should use goTest and then pass flags overriding these
// defaults as later arguments in the command line.
func (t *tester) goTest() []string {
	return []string{
		"go", "test", "-short=" + short(), "-count=1", t.tags(), t.runFlag(""),
	}
}

func (t *tester) tags() string {
	ios := t.iOS()
	switch {
	case ios && noOpt:
		return "-tags=lldb,noopt"
	case ios:
		return "-tags=lldb"
	case noOpt:
		return "-tags=noopt"
	default:
		return "-tags="
	}
}

// timeoutDuration converts the provided number of seconds into a
// time.Duration, scaled by the t.timeoutScale factor.
func (t *tester) timeoutDuration(sec int) time.Duration {
	return time.Duration(sec) * time.Second * time.Duration(t.timeoutScale)
}

// timeout returns the "-timeout=" string argument to "go test" given
// the number of seconds of timeout. It scales it by the
// t.timeoutScale factor.
func (t *tester) timeout(sec int) string {
	return "-timeout=" + t.timeoutDuration(sec).String()
}

// ranGoTest and stdMatches are state closed over by the stdlib
// testing func in registerStdTest below. The tests are run
// sequentially, so there's no need for locks.
//
// ranGoBench and benchMatches are the same, but are only used
// in -race mode.
var (
	ranGoTest  bool
	stdMatches []string

	ranGoBench   bool
	benchMatches []string
)

func (t *tester) registerStdTest(pkg string) {
	heading := "Testing packages."
	testPrefix := "go_test:"
	gcflags := gogcflags

	testName := testPrefix + pkg
	if t.runRx == nil || t.runRx.MatchString(testName) == t.runRxWant {
		stdMatches = append(stdMatches, pkg)
	}

	t.tests = append(t.tests, distTest{
		name:    testName,
		heading: heading,
		fn: func(dt *distTest) error {
			if ranGoTest {
				return nil
			}
			t.runPending(dt)
			timelog("start", dt.name)
			defer timelog("end", dt.name)
			ranGoTest = true

			timeoutSec := 180
			for _, pkg := range stdMatches {
				if pkg == "cmd/go" {
					timeoutSec *= 3
					break
				}
			}
			// Special case for our slow cross-compiled
			// qemu builders:
			if t.shouldUsePrecompiledStdTest() {
				return t.runPrecompiledStdTest(t.timeoutDuration(timeoutSec))
			}
			args := []string{
				"test",
				"-short=" + short(),
				t.tags(),
				t.timeout(timeoutSec),
			}
			if gcflags != "" {
				args = append(args, "-gcflags=all="+gcflags)
			}
			if t.race {
				args = append(args, "-race")
			}
			if t.msan {
				args = append(args, "-msan")
			}
			if t.asan {
				args = append(args, "-asan")
			}
			if t.compileOnly {
				args = append(args, "-run=^$")
			}
			args = append(args, stdMatches...)
			cmd := exec.Command(gorootBinGo, args...)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			return cmd.Run()
		},
	})
}

func (t *tester) registerRaceBenchTest(pkg string) {
	testName := "go_test_bench:" + pkg
	if t.runRx == nil || t.runRx.MatchString(testName) == t.runRxWant {
		benchMatches = append(benchMatches, pkg)
	}
	t.tests = append(t.tests, distTest{
		name:    testName,
		heading: "Running benchmarks briefly.",
		fn: func(dt *distTest) error {
			if ranGoBench {
				return nil
			}
			t.runPending(dt)
			timelog("start", dt.name)
			defer timelog("end", dt.name)
			ranGoBench = true
			args := []string{
				"test",
				"-short=" + short(),
				"-race",
				t.timeout(1200), // longer timeout for race with benchmarks
				"-run=^$",       // nothing. only benchmarks.
				"-benchtime=.1s",
				"-cpu=4",
			}
			if !t.compileOnly {
				args = append(args, "-bench=.*")
			}
			args = append(args, benchMatches...)
			cmd := exec.Command(gorootBinGo, args...)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			return cmd.Run()
		},
	})
}

// stdOutErrAreTerminals is defined in test_linux.go, to report
// whether stdout & stderr are terminals.
var stdOutErrAreTerminals func() bool

func (t *tester) registerTests() {
	// Fast path to avoid the ~1 second of `go list std cmd` when
	// the caller lists specific tests to run. (as the continuous
	// build coordinator does).
	if len(t.runNames) > 0 {
		for _, name := range t.runNames {
			if strings.HasPrefix(name, "go_test:") {
				t.registerStdTest(strings.TrimPrefix(name, "go_test:"))
			}
			if strings.HasPrefix(name, "go_test_bench:") {
				t.registerRaceBenchTest(strings.TrimPrefix(name, "go_test_bench:"))
			}
		}
	} else {
		// Use a format string to only list packages and commands that have tests.
		const format = "{{if (or .TestGoFiles .XTestGoFiles)}}{{.ImportPath}}{{end}}"
		cmd := exec.Command(gorootBinGo, "list", "-f", format)
		if t.race {
			cmd.Args = append(cmd.Args, "-tags=race")
		}
		cmd.Args = append(cmd.Args, "std")
		if t.shouldTestCmd() {
			cmd.Args = append(cmd.Args, "cmd")
		}
		cmd.Stderr = new(bytes.Buffer)
		all, err := cmd.Output()
		if err != nil {
			fatalf("Error running go list std cmd: %v:\n%s", err, cmd.Stderr)
		}
		pkgs := strings.Fields(string(all))
		for _, pkg := range pkgs {
			t.registerStdTest(pkg)
		}
		if t.race {
			for _, pkg := range pkgs {
				if t.packageHasBenchmarks(pkg) {
					t.registerRaceBenchTest(pkg)
				}
			}
		}
	}

	// Test the os/user package in the pure-Go mode too.
	if !t.compileOnly {
		t.tests = append(t.tests, distTest{
			name:    "osusergo",
			heading: "os/user with tag osusergo",
			fn: func(dt *distTest) error {
				t.addCmd(dt, "src", t.goTest(), t.timeout(300), "-tags=osusergo", "os/user")
				return nil
			},
		})
	}

	// Test ios/amd64 for the iOS simulator.
	if goos == "darwin" && goarch == "amd64" && t.cgoEnabled {
		t.tests = append(t.tests, distTest{
			name:    "amd64ios",
			heading: "GOOS=ios on darwin/amd64",
			fn: func(dt *distTest) error {
				cmd := t.addCmd(dt, "src", t.goTest(), t.timeout(300), "-run=SystemRoots", "crypto/x509")
				setEnv(cmd, "GOOS", "ios")
				setEnv(cmd, "CGO_ENABLED", "1")
				return nil
			},
		})
	}

	if t.race {
		return
	}

	// Runtime CPU tests.
	if !t.compileOnly && goos != "js" { // js can't handle -cpu != 1
		testName := "runtime:cpu124"
		t.tests = append(t.tests, distTest{
			name:    testName,
			heading: "GOMAXPROCS=2 runtime -cpu=1,2,4 -quick",
			fn: func(dt *distTest) error {
				cmd := t.addCmd(dt, "src", t.goTest(), "-short=true", t.timeout(300), "runtime", "-cpu=1,2,4", "-quick")
				// We set GOMAXPROCS=2 in addition to -cpu=1,2,4 in order to test runtime bootstrap code,
				// creation of first goroutines and first garbage collections in the parallel setting.
				setEnv(cmd, "GOMAXPROCS", "2")
				return nil
			},
		})
	}

	// morestack tests. We only run these on in long-test mode
	// (with GO_TEST_SHORT=false) because the runtime test is
	// already quite long and mayMoreStackMove makes it about
	// twice as slow.
	if !t.compileOnly && short() == "false" {
		// hooks is the set of maymorestack hooks to test with.
		hooks := []string{"mayMoreStackPreempt", "mayMoreStackMove"}
		// pkgs is the set of test packages to run.
		pkgs := []string{"runtime", "reflect", "sync"}
		// hookPkgs is the set of package patterns to apply
		// the maymorestack hook to.
		hookPkgs := []string{"runtime/...", "reflect", "sync"}
		// unhookPkgs is the set of package patterns to
		// exclude from hookPkgs.
		unhookPkgs := []string{"runtime/testdata/..."}
		for _, hook := range hooks {
			// Construct the build flags to use the
			// maymorestack hook in the compiler and
			// assembler. We pass this via the GOFLAGS
			// environment variable so that it applies to
			// both the test itself and to binaries built
			// by the test.
			goFlagsList := []string{}
			for _, flag := range []string{"-gcflags", "-asmflags"} {
				for _, hookPkg := range hookPkgs {
					goFlagsList = append(goFlagsList, flag+"="+hookPkg+"=-d=maymorestack=runtime."+hook)
				}
				for _, unhookPkg := range unhookPkgs {
					goFlagsList = append(goFlagsList, flag+"="+unhookPkg+"=")
				}
			}
			goFlags := strings.Join(goFlagsList, " ")

			for _, pkg := range pkgs {
				pkg := pkg
				testName := hook + ":" + pkg
				t.tests = append(t.tests, distTest{
					name:    testName,
					heading: "maymorestack=" + hook,
					fn: func(dt *distTest) error {
						cmd := t.addCmd(dt, "src", t.goTest(), t.timeout(600), pkg, "-short")
						setEnv(cmd, "GOFLAGS", goFlags)
						return nil
					},
				})
			}
		}
	}

	// This test needs its stdout/stderr to be terminals, so we don't run it from cmd/go's tests.
	// See issue 18153.
	if goos == "linux" {
		t.tests = append(t.tests, distTest{
			name:    "cmd_go_test_terminal",
			heading: "cmd/go terminal test",
			fn: func(dt *distTest) error {
				t.runPending(dt)
				timelog("start", dt.name)
				defer timelog("end", dt.name)
				if !stdOutErrAreTerminals() {
					fmt.Println("skipping terminal test; stdout/stderr not terminals")
					return nil
				}
				cmd := exec.Command(gorootBinGo, "test")
				setDir(cmd, filepath.Join(os.Getenv("GOROOT"), "src/cmd/go/testdata/testterminal18153"))
				cmd.Stdout = os.Stdout
				cmd.Stderr = os.Stderr
				return cmd.Run()
			},
		})
	}

	// On the builders only, test that a moved GOROOT still works.
	// Fails on iOS because CC_FOR_TARGET refers to clangwrap.sh
	// in the unmoved GOROOT.
	// Fails on Android and js/wasm with an exec format error.
	// Fails on plan9 with "cannot find GOROOT" (issue #21016).
	if os.Getenv("GO_BUILDER_NAME") != "" && goos != "android" && !t.iOS() && goos != "plan9" && goos != "js" {
		t.tests = append(t.tests, distTest{
			name:    "moved_goroot",
			heading: "moved GOROOT",
			fn: func(dt *distTest) error {
				t.runPending(dt)
				timelog("start", dt.name)
				defer timelog("end", dt.name)
				moved := goroot + "-moved"
				if err := os.Rename(goroot, moved); err != nil {
					if goos == "windows" {
						// Fails on Windows (with "Access is denied") if a process
						// or binary is in this directory. For instance, using all.bat
						// when run from c:\workdir\go\src fails here
						// if GO_BUILDER_NAME is set. Our builders invoke tests
						// a different way which happens to work when sharding
						// tests, but we should be tolerant of the non-sharded
						// all.bat case.
						log.Printf("skipping test on Windows")
						return nil
					}
					return err
				}

				// Run `go test fmt` in the moved GOROOT, without explicitly setting
				// GOROOT in the environment. The 'go' command should find itself.
				cmd := exec.Command(filepath.Join(moved, "bin", "go"), "test", "fmt")
				cmd.Stdout = os.Stdout
				cmd.Stderr = os.Stderr
				unsetEnv(cmd, "GOROOT")
				unsetEnv(cmd, "GOCACHE") // TODO(bcmills): ...why‽
				err := cmd.Run()

				if rerr := os.Rename(moved, goroot); rerr != nil {
					fatalf("failed to restore GOROOT: %v", rerr)
				}
				return err
			},
		})
	}

	// Test that internal linking of standard packages does not
	// require libgcc. This ensures that we can install a Go
	// release on a system that does not have a C compiler
	// installed and still build Go programs (that don't use cgo).
	for _, pkg := range cgoPackages {
		if !t.internalLink() {
			break
		}

		// ARM libgcc may be Thumb, which internal linking does not support.
		if goarch == "arm" {
			break
		}

		pkg := pkg
		var run string
		if pkg == "net" {
			run = "TestTCPStress"
		}
		t.tests = append(t.tests, distTest{
			name:    "nolibgcc:" + pkg,
			heading: "Testing without libgcc.",
			fn: func(dt *distTest) error {
				// What matters is that the tests build and start up.
				// Skip expensive tests, especially x509 TestSystemRoots.
				t.addCmd(dt, "src", t.goTest(), "-ldflags=-linkmode=internal -libgcc=none", "-run=^Test[^CS]", pkg, t.runFlag(run))
				return nil
			},
		})
	}

	// Stub out following test on alpine until 54354 resolved.
	builderName := os.Getenv("GO_BUILDER_NAME")
	disablePIE := strings.HasSuffix(builderName, "-alpine")

	// Test internal linking of PIE binaries where it is supported.
	if t.internalLinkPIE() && !disablePIE {
		t.tests = append(t.tests, distTest{
			name:    "pie_internal",
			heading: "internal linking of -buildmode=pie",
			fn: func(dt *distTest) error {
				t.addCmd(dt, "src", t.goTest(), "reflect", "-buildmode=pie", "-ldflags=-linkmode=internal", t.timeout(60))
				return nil
			},
		})
		// Also test a cgo package.
		if t.cgoEnabled && t.internalLink() && !disablePIE {
			t.tests = append(t.tests, distTest{
				name:    "pie_internal_cgo",
				heading: "internal linking of -buildmode=pie",
				fn: func(dt *distTest) error {
					t.addCmd(dt, "src", t.goTest(), "os/user", "-buildmode=pie", "-ldflags=-linkmode=internal", t.timeout(60))
					return nil
				},
			})
		}
	}

	// sync tests
	if goos != "js" { // js doesn't support -cpu=10
		t.tests = append(t.tests, distTest{
			name:    "sync_cpu",
			heading: "sync -cpu=10",
			fn: func(dt *distTest) error {
				t.addCmd(dt, "src", t.goTest(), "sync", t.timeout(120), "-cpu=10", t.runFlag(""))
				return nil
			},
		})
	}

	if t.raceDetectorSupported() {
		t.tests = append(t.tests, distTest{
			name:    "race",
			heading: "Testing race detector",
			fn:      t.raceTest,
		})
	}

	if t.cgoEnabled && !t.iOS() {
		// Disabled on iOS. golang.org/issue/15919
		t.registerHostTest("cgo_stdio", "../misc/cgo/stdio", "misc/cgo/stdio", ".")
		t.registerHostTest("cgo_life", "../misc/cgo/life", "misc/cgo/life", ".")
		fortran := os.Getenv("FC")
		if fortran == "" {
			fortran, _ = exec.LookPath("gfortran")
		}
		if t.hasBash() && goos != "android" && fortran != "" {
			t.tests = append(t.tests, distTest{
				name:    "cgo_fortran",
				heading: "../misc/cgo/fortran",
				fn: func(dt *distTest) error {
					t.addCmd(dt, "misc/cgo/fortran", "./test.bash", fortran)
					return nil
				},
			})
		}
		if t.hasSwig() && goos != "android" {
			t.tests = append(t.tests, distTest{
				name:    "swig_stdio",
				heading: "../misc/swig/stdio",
				fn: func(dt *distTest) error {
					t.addCmd(dt, "misc/swig/stdio", t.goTest(), ".")
					return nil
				},
			})
			if t.hasCxx() {
				t.tests = append(t.tests,
					distTest{
						name:    "swig_callback",
						heading: "../misc/swig/callback",
						fn: func(dt *distTest) error {
							t.addCmd(dt, "misc/swig/callback", t.goTest(), ".")
							return nil
						},
					},
					distTest{
						name:    "swig_callback_lto",
						heading: "../misc/swig/callback",
						fn: func(dt *distTest) error {
							cmd := t.addCmd(dt, "misc/swig/callback", t.goTest(), ".")
							setEnv(cmd, "CGO_CFLAGS", "-flto -Wno-lto-type-mismatch -Wno-unknown-warning-option")
							setEnv(cmd, "CGO_CXXFLAGS", "-flto -Wno-lto-type-mismatch -Wno-unknown-warning-option")
							setEnv(cmd, "CGO_LDFLAGS", "-flto -Wno-lto-type-mismatch -Wno-unknown-warning-option")
							return nil
						},
					},
				)
			}
		}
	}
	if t.cgoEnabled {
		t.tests = append(t.tests, distTest{
			name:    "cgo_test",
			heading: "../misc/cgo/test",
			fn:      t.cgoTest,
		})
	}

	// Don't run these tests with $GO_GCFLAGS because most of them
	// assume that they can run "go install" with no -gcflags and not
	// recompile the entire standard library. If make.bash ran with
	// special -gcflags, that's not true.
	if t.cgoEnabled && gogcflags == "" {
		t.registerHostTest("testgodefs", "../misc/cgo/testgodefs", "misc/cgo/testgodefs", ".")

		t.registerTest("testso", "../misc/cgo/testso", t.goTest(), t.timeout(600), ".")
		t.registerTest("testsovar", "../misc/cgo/testsovar", t.goTest(), t.timeout(600), ".")
		if t.supportedBuildmode("c-archive") {
			t.registerHostTest("testcarchive", "../misc/cgo/testcarchive", "misc/cgo/testcarchive", ".")
		}
		if t.supportedBuildmode("c-shared") {
			t.registerHostTest("testcshared", "../misc/cgo/testcshared", "misc/cgo/testcshared", ".")
		}
		if t.supportedBuildmode("shared") {
			t.registerTest("testshared", "../misc/cgo/testshared", t.goTest(), t.timeout(600), ".")
		}
		if t.supportedBuildmode("plugin") {
			t.registerTest("testplugin", "../misc/cgo/testplugin", t.goTest(), t.timeout(600), ".")
		}
		if gohostos == "linux" && (goarch == "amd64" || goarch == "ppc64le") {
			t.registerTest("testasan", "../misc/cgo/testasan", "go", "run", ".")
		}
		if goos == "linux" {
			// because syscall.SysProcAttr struct used in misc/cgo/testsanitizers is only built on linux.
			t.registerHostTest("testsanitizers", "../misc/cgo/testsanitizers", "misc/cgo/testsanitizers", ".")
		}
		if t.hasBash() && goos != "android" && !t.iOS() && gohostos != "windows" {
			t.registerHostTest("cgo_errors", "../misc/cgo/errors", "misc/cgo/errors", ".")
		}
		if gohostos == "linux" && t.extLink() {
			t.registerTest("testsigfwd", "../misc/cgo/testsigfwd", "go", "run", ".")
		}
	}

	if goos != "android" && !t.iOS() {
		// There are no tests in this directory, only benchmarks.
		// Check that the test binary builds but don't bother running it.
		// (It has init-time work to set up for the benchmarks that is not worth doing unnecessarily.)
		t.registerTest("bench_go1", "../test/bench/go1", t.goTest(), "-c", "-o="+os.DevNull)
	}
	if goos != "android" && !t.iOS() {
		// Only start multiple test dir shards on builders,
		// where they get distributed to multiple machines.
		// See issues 20141 and 31834.
		nShards := 1
		if os.Getenv("GO_BUILDER_NAME") != "" {
			nShards = 10
		}
		if n, err := strconv.Atoi(os.Getenv("GO_TEST_SHARDS")); err == nil {
			nShards = n
		}
		for shard := 0; shard < nShards; shard++ {
			shard := shard
			t.tests = append(t.tests, distTest{
				name:    fmt.Sprintf("test:%d_%d", shard, nShards),
				heading: "../test",
				fn:      func(dt *distTest) error { return t.testDirTest(dt, shard, nShards) },
			})
		}
	}
	// Only run the API check on fast development platforms.
	// Every platform checks the API on every GOOS/GOARCH/CGO_ENABLED combination anyway,
	// so we really only need to run this check once anywhere to get adequate coverage.
	// To help developers avoid trybot-only failures, we try to run on typical developer machines
	// which is darwin/linux/windows and amd64/arm64.
	if (goos == "darwin" || goos == "linux" || goos == "windows") && (goarch == "amd64" || goarch == "arm64") {
		t.tests = append(t.tests, distTest{
			name:    "api",
			heading: "API check",
			fn: func(dt *distTest) error {
				if t.compileOnly {
					t.addCmd(dt, "src", "go", "build", "-o", os.DevNull, filepath.Join(goroot, "src/cmd/api/run.go"))
					return nil
				}
				t.addCmd(dt, "src", "go", "run", filepath.Join(goroot, "src/cmd/api/run.go"))
				return nil
			},
		})
	}

	// Ensure that the toolchain can bootstrap itself.
	// This test adds another ~45s to all.bash if run sequentially, so run it only on the builders.
	if os.Getenv("GO_BUILDER_NAME") != "" && goos != "android" && !t.iOS() {
		t.registerHostTest("reboot", "../misc/reboot", "misc/reboot", ".")
	}
}

// isRegisteredTestName reports whether a test named testName has already
// been registered.
func (t *tester) isRegisteredTestName(testName string) bool {
	for _, tt := range t.tests {
		if tt.name == testName {
			return true
		}
	}
	return false
}

func (t *tester) registerTest1(seq bool, name, dirBanner string, cmdline ...interface{}) {
	bin, args := flattenCmdline(cmdline)
	if bin == "time" && !t.haveTime {
		bin, args = args[0], args[1:]
	}
	if t.isRegisteredTestName(name) {
		panic("duplicate registered test name " + name)
	}
	t.tests = append(t.tests, distTest{
		name:    name,
		heading: dirBanner,
		fn: func(dt *distTest) error {
			if seq {
				t.runPending(dt)
				timelog("start", name)
				defer timelog("end", name)
				return t.dirCmd(filepath.Join(goroot, "src", dirBanner), bin, args).Run()
			}
			t.addCmd(dt, filepath.Join(goroot, "src", dirBanner), bin, args)
			return nil
		},
	})
}

func (t *tester) registerTest(name, dirBanner string, cmdline ...interface{}) {
	t.registerTest1(false, name, dirBanner, cmdline...)
}

func (t *tester) registerSeqTest(name, dirBanner string, cmdline ...interface{}) {
	t.registerTest1(true, name, dirBanner, cmdline...)
}

func (t *tester) bgDirCmd(dir, bin string, args ...string) *exec.Cmd {
	cmd := exec.Command(bin, args...)
	if filepath.IsAbs(dir) {
		setDir(cmd, dir)
	} else {
		setDir(cmd, filepath.Join(goroot, dir))
	}
	return cmd
}

func (t *tester) dirCmd(dir string, cmdline ...interface{}) *exec.Cmd {
	bin, args := flattenCmdline(cmdline)
	cmd := t.bgDirCmd(dir, bin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if vflag > 1 {
		errprintf("%s\n", strings.Join(cmd.Args, " "))
	}
	return cmd
}

// flattenCmdline flattens a mixture of string and []string as single list
// and then interprets it as a command line: first element is binary, then args.
func flattenCmdline(cmdline []interface{}) (bin string, args []string) {
	var list []string
	for _, x := range cmdline {
		switch x := x.(type) {
		case string:
			list = append(list, x)
		case []string:
			list = append(list, x...)
		default:
			panic("invalid addCmd argument type: " + reflect.TypeOf(x).String())
		}
	}

	// The go command is too picky about duplicated flags.
	// Drop all but the last of the allowed duplicated flags.
	drop := make([]bool, len(list))
	have := map[string]int{}
	for i := 1; i < len(list); i++ {
		j := strings.Index(list[i], "=")
		if j < 0 {
			continue
		}
		flag := list[i][:j]
		switch flag {
		case "-run", "-tags":
			if have[flag] != 0 {
				drop[have[flag]] = true
			}
			have[flag] = i
		}
	}
	out := list[:0]
	for i, x := range list {
		if !drop[i] {
			out = append(out, x)
		}
	}
	list = out

	bin = list[0]
	if bin == "go" {
		bin = gorootBinGo
	}
	return bin, list[1:]
}

func (t *tester) addCmd(dt *distTest, dir string, cmdline ...interface{}) *exec.Cmd {
	bin, args := flattenCmdline(cmdline)
	w := &work{
		dt:  dt,
		cmd: t.bgDirCmd(dir, bin, args...),
	}
	t.worklist = append(t.worklist, w)
	return w.cmd
}

func (t *tester) iOS() bool {
	return goos == "ios"
}

func (t *tester) out(v string) {
	if t.banner == "" {
		return
	}
	fmt.Println("\n" + t.banner + v)
}

func (t *tester) extLink() bool {
	pair := gohostos + "-" + goarch
	switch pair {
	case "aix-ppc64",
		"android-arm", "android-arm64",
		"darwin-amd64", "darwin-arm64",
		"dragonfly-amd64",
		"freebsd-386", "freebsd-amd64", "freebsd-arm",
		"linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-loong64", "linux-ppc64le", "linux-mips64", "linux-mips64le", "linux-mips", "linux-mipsle", "linux-riscv64", "linux-s390x",
		"netbsd-386", "netbsd-amd64",
		"openbsd-386", "openbsd-amd64",
		"windows-386", "windows-amd64":
		return true
	}
	return false
}

func (t *tester) internalLink() bool {
	if gohostos == "dragonfly" {
		// linkmode=internal fails on dragonfly since errno is a TLS relocation.
		return false
	}
	if goos == "android" {
		return false
	}
	if goos == "ios" {
		return false
	}
	if goos == "windows" && goarch == "arm64" {
		return false
	}
	// Internally linking cgo is incomplete on some architectures.
	// https://golang.org/issue/10373
	// https://golang.org/issue/14449
	if goarch == "loong64" || goarch == "mips64" || goarch == "mips64le" || goarch == "mips" || goarch == "mipsle" || goarch == "riscv64" {
		return false
	}
	if goos == "aix" {
		// linkmode=internal isn't supported.
		return false
	}
	return true
}

func (t *tester) internalLinkPIE() bool {
	switch goos + "-" + goarch {
	case "darwin-amd64", "darwin-arm64",
		"linux-amd64", "linux-arm64", "linux-ppc64le",
		"android-arm64",
		"windows-amd64", "windows-386", "windows-arm":
		return true
	}
	return false
}

func (t *tester) supportedBuildmode(mode string) bool {
	pair := goos + "-" + goarch
	switch mode {
	case "c-archive":
		if !t.extLink() {
			return false
		}
		switch pair {
		case "aix-ppc64",
			"darwin-amd64", "darwin-arm64", "ios-arm64",
			"linux-amd64", "linux-386", "linux-ppc64le", "linux-riscv64", "linux-s390x",
			"freebsd-amd64",
			"windows-amd64", "windows-386":
			return true
		}
		return false
	case "c-shared":
		switch pair {
		case "linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-ppc64le", "linux-riscv64", "linux-s390x",
			"darwin-amd64", "darwin-arm64",
			"freebsd-amd64",
			"android-arm", "android-arm64", "android-386",
			"windows-amd64", "windows-386", "windows-arm64":
			return true
		}
		return false
	case "shared":
		switch pair {
		case "linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-ppc64le", "linux-s390x":
			return true
		}
		return false
	case "plugin":
		switch pair {
		case "linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-s390x", "linux-ppc64le":
			return true
		case "darwin-amd64", "darwin-arm64":
			return true
		case "freebsd-amd64":
			return true
		}
		return false
	case "pie":
		switch pair {
		case "aix/ppc64",
			"linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-ppc64le", "linux-riscv64", "linux-s390x",
			"android-amd64", "android-arm", "android-arm64", "android-386":
			return true
		case "darwin-amd64", "darwin-arm64":
			return true
		case "windows-amd64", "windows-386", "windows-arm":
			return true
		}
		return false

	default:
		fatalf("internal error: unknown buildmode %s", mode)
		return false
	}
}

func (t *tester) registerHostTest(name, heading, dir, pkg string) {
	t.tests = append(t.tests, distTest{
		name:    name,
		heading: heading,
		fn: func(dt *distTest) error {
			t.runPending(dt)
			timelog("start", name)
			defer timelog("end", name)
			return t.runHostTest(dir, pkg)
		},
	})
}

func (t *tester) runHostTest(dir, pkg string) error {
	out, err := exec.Command(gorootBinGo, "env", "GOEXE", "GOTMPDIR").Output()
	if err != nil {
		return err
	}

	parts := strings.Split(string(out), "\n")
	if len(parts) < 2 {
		return fmt.Errorf("'go env GOEXE GOTMPDIR' output contains <2 lines")
	}
	GOEXE := strings.TrimSpace(parts[0])
	GOTMPDIR := strings.TrimSpace(parts[1])

	f, err := ioutil.TempFile(GOTMPDIR, "test.test-*"+GOEXE)
	if err != nil {
		return err
	}
	f.Close()
	defer os.Remove(f.Name())

	cmd := t.dirCmd(dir, t.goTest(), "-c", "-o", f.Name(), pkg)
	setEnv(cmd, "GOARCH", gohostarch)
	setEnv(cmd, "GOOS", gohostos)
	if err := cmd.Run(); err != nil {
		return err
	}
	return t.dirCmd(dir, f.Name(), "-test.short="+short(), "-test.timeout="+t.timeoutDuration(300).String()).Run()
}

func (t *tester) cgoTest(dt *distTest) error {
	cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), ".")
	setEnv(cmd, "GOFLAGS", "-ldflags=-linkmode=auto")

	// Stub out various buildmode=pie tests  on alpine until 54354 resolved.
	builderName := os.Getenv("GO_BUILDER_NAME")
	disablePIE := strings.HasSuffix(builderName, "-alpine")

	if t.internalLink() {
		cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), "-tags=internal", ".")
		setEnv(cmd, "GOFLAGS", "-ldflags=-linkmode=internal")
	}

	pair := gohostos + "-" + goarch
	switch pair {
	case "darwin-amd64", "darwin-arm64",
		"windows-386", "windows-amd64":
		// test linkmode=external, but __thread not supported, so skip testtls.
		if !t.extLink() {
			break
		}
		cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), ".")
		setEnv(cmd, "GOFLAGS", "-ldflags=-linkmode=external")

		t.addCmd(dt, "misc/cgo/test", t.goTest(), "-ldflags", "-linkmode=external -s", ".")

		if t.supportedBuildmode("pie") && !disablePIE {

			t.addCmd(dt, "misc/cgo/test", t.goTest(), "-buildmode=pie", ".")
			if t.internalLink() && t.internalLinkPIE() {
				t.addCmd(dt, "misc/cgo/test", t.goTest(), "-buildmode=pie", "-ldflags=-linkmode=internal", "-tags=internal,internal_pie", ".")
			}
		}

	case "aix-ppc64",
		"android-arm", "android-arm64",
		"dragonfly-amd64",
		"freebsd-386", "freebsd-amd64", "freebsd-arm",
		"linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-ppc64le", "linux-riscv64", "linux-s390x",
		"netbsd-386", "netbsd-amd64",
		"openbsd-386", "openbsd-amd64", "openbsd-arm", "openbsd-arm64", "openbsd-mips64":

		cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), ".")
		setEnv(cmd, "GOFLAGS", "-ldflags=-linkmode=external")
		// cgo should be able to cope with both -g arguments and colored
		// diagnostics.
		setEnv(cmd, "CGO_CFLAGS", "-g0 -fdiagnostics-color")

		t.addCmd(dt, "misc/cgo/testtls", t.goTest(), "-ldflags", "-linkmode=auto", ".")
		t.addCmd(dt, "misc/cgo/testtls", t.goTest(), "-ldflags", "-linkmode=external", ".")

		switch pair {
		case "aix-ppc64", "netbsd-386", "netbsd-amd64":
			// no static linking
		case "freebsd-arm":
			// -fPIC compiled tls code will use __tls_get_addr instead
			// of __aeabi_read_tp, however, on FreeBSD/ARM, __tls_get_addr
			// is implemented in rtld-elf, so -fPIC isn't compatible with
			// static linking on FreeBSD/ARM with clang. (cgo depends on
			// -fPIC fundamentally.)
		default:
			cmd := t.dirCmd("misc/cgo/test",
				compilerEnvLookup(defaultcc, goos, goarch), "-xc", "-o", "/dev/null", "-static", "-")
			cmd.Stdin = strings.NewReader("int main() {}")
			if err := cmd.Run(); err != nil {
				fmt.Println("No support for static linking found (lacks libc.a?), skip cgo static linking test.")
			} else {
				if goos != "android" {
					t.addCmd(dt, "misc/cgo/testtls", t.goTest(), "-ldflags", `-linkmode=external -extldflags "-static -pthread"`, ".")
				}
				t.addCmd(dt, "misc/cgo/nocgo", t.goTest(), ".")
				t.addCmd(dt, "misc/cgo/nocgo", t.goTest(), "-ldflags", `-linkmode=external`, ".")
				if goos != "android" {
					t.addCmd(dt, "misc/cgo/nocgo", t.goTest(), "-ldflags", `-linkmode=external -extldflags "-static -pthread"`, ".")
					t.addCmd(dt, "misc/cgo/test", t.goTest(), "-tags=static", "-ldflags", `-linkmode=external -extldflags "-static -pthread"`, ".")
					// -static in CGO_LDFLAGS triggers a different code path
					// than -static in -extldflags, so test both.
					// See issue #16651.
					cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), "-tags=static", ".")
					setEnv(cmd, "CGO_LDFLAGS", "-static -pthread")
				}
			}

			if t.supportedBuildmode("pie") && !disablePIE {
				t.addCmd(dt, "misc/cgo/test", t.goTest(), "-buildmode=pie", ".")
				if t.internalLink() && t.internalLinkPIE() {
					t.addCmd(dt, "misc/cgo/test", t.goTest(), "-buildmode=pie", "-ldflags=-linkmode=internal", "-tags=internal,internal_pie", ".")
				}
				t.addCmd(dt, "misc/cgo/testtls", t.goTest(), "-buildmode=pie", ".")
				t.addCmd(dt, "misc/cgo/nocgo", t.goTest(), "-buildmode=pie", ".")
			}
		}
	}

	return nil
}

// run pending test commands, in parallel, emitting headers as appropriate.
// When finished, emit header for nextTest, which is going to run after the
// pending commands are done (and runPending returns).
// A test should call runPending if it wants to make sure that it is not
// running in parallel with earlier tests, or if it has some other reason
// for needing the earlier tests to be done.
func (t *tester) runPending(nextTest *distTest) {
	checkNotStale("go", "std")
	worklist := t.worklist
	t.worklist = nil
	for _, w := range worklist {
		w.start = make(chan bool)
		w.end = make(chan bool)
		go func(w *work) {
			if !<-w.start {
				timelog("skip", w.dt.name)
				w.out = []byte(fmt.Sprintf("skipped due to earlier error\n"))
			} else {
				timelog("start", w.dt.name)
				w.out, w.err = w.cmd.CombinedOutput()
				if w.err != nil {
					if isUnsupportedVMASize(w) {
						timelog("skip", w.dt.name)
						w.out = []byte(fmt.Sprintf("skipped due to unsupported VMA\n"))
						w.err = nil
					}
				}
			}
			timelog("end", w.dt.name)
			w.end <- true
		}(w)
	}

	started := 0
	ended := 0
	var last *distTest
	for ended < len(worklist) {
		for started < len(worklist) && started-ended < maxbg {
			w := worklist[started]
			started++
			w.start <- !t.failed || t.keepGoing
		}
		w := worklist[ended]
		dt := w.dt
		if dt.heading != "" && t.lastHeading != dt.heading {
			t.lastHeading = dt.heading
			t.out(dt.heading)
		}
		if dt != last {
			// Assumes all the entries for a single dt are in one worklist.
			last = w.dt
			if vflag > 0 {
				fmt.Printf("# go tool dist test -run=^%s$\n", dt.name)
			}
		}
		if vflag > 1 {
			errprintf("%s\n", strings.Join(w.cmd.Args, " "))
		}
		ended++
		<-w.end
		os.Stdout.Write(w.out)
		if w.err != nil {
			log.Printf("Failed: %v", w.err)
			t.failed = true
		}
		checkNotStale("go", "std")
	}
	if t.failed && !t.keepGoing {
		fatalf("FAILED")
	}

	if dt := nextTest; dt != nil {
		if dt.heading != "" && t.lastHeading != dt.heading {
			t.lastHeading = dt.heading
			t.out(dt.heading)
		}
		if vflag > 0 {
			fmt.Printf("# go tool dist test -run=^%s$\n", dt.name)
		}
	}
}

func (t *tester) hasBash() bool {
	switch gohostos {
	case "windows", "plan9":
		return false
	}
	return true
}

func (t *tester) hasCxx() bool {
	cxx, _ := exec.LookPath(compilerEnvLookup(defaultcxx, goos, goarch))
	return cxx != ""
}

func (t *tester) hasSwig() bool {
	swig, err := exec.LookPath("swig")
	if err != nil {
		return false
	}

	// Check that swig was installed with Go support by checking
	// that a go directory exists inside the swiglib directory.
	// See https://golang.org/issue/23469.
	output, err := exec.Command(swig, "-go", "-swiglib").Output()
	if err != nil {
		return false
	}
	swigDir := strings.TrimSpace(string(output))

	_, err = os.Stat(filepath.Join(swigDir, "go"))
	if err != nil {
		return false
	}

	// Check that swig has a new enough version.
	// See https://golang.org/issue/22858.
	out, err := exec.Command(swig, "-version").CombinedOutput()
	if err != nil {
		return false
	}

	re := regexp.MustCompile(`[vV]ersion +([\d]+)([.][\d]+)?([.][\d]+)?`)
	matches := re.FindSubmatch(out)
	if matches == nil {
		// Can't find version number; hope for the best.
		return true
	}

	major, err := strconv.Atoi(string(matches[1]))
	if err != nil {
		// Can't find version number; hope for the best.
		return true
	}
	if major < 3 {
		return false
	}
	if major > 3 {
		// 4.0 or later
		return true
	}

	// We have SWIG version 3.x.
	if len(matches[2]) > 0 {
		minor, err := strconv.Atoi(string(matches[2][1:]))
		if err != nil {
			return true
		}
		if minor > 0 {
			// 3.1 or later
			return true
		}
	}

	// We have SWIG version 3.0.x.
	if len(matches[3]) > 0 {
		patch, err := strconv.Atoi(string(matches[3][1:]))
		if err != nil {
			return true
		}
		if patch < 6 {
			// Before 3.0.6.
			return false
		}
	}

	return true
}

func (t *tester) raceDetectorSupported() bool {
	if gohostos != goos {
		return false
	}
	if !t.cgoEnabled {
		return false
	}
	if !raceDetectorSupported(goos, goarch) {
		return false
	}
	// The race detector doesn't work on Alpine Linux:
	// golang.org/issue/14481
	if isAlpineLinux() {
		return false
	}
	// NetBSD support is unfinished.
	// golang.org/issue/26403
	if goos == "netbsd" {
		return false
	}
	return true
}

func isAlpineLinux() bool {
	if runtime.GOOS != "linux" {
		return false
	}
	fi, err := os.Lstat("/etc/alpine-release")
	return err == nil && fi.Mode().IsRegular()
}

func (t *tester) runFlag(rx string) string {
	if t.compileOnly {
		return "-run=^$"
	}
	return "-run=" + rx
}

func (t *tester) raceTest(dt *distTest) error {
	t.addCmd(dt, "src", t.goTest(), "-race", t.runFlag("Output"), "runtime/race")
	t.addCmd(dt, "src", t.goTest(), "-race", t.runFlag("TestParse|TestEcho|TestStdinCloseRace|TestClosedPipeRace|TestTypeRace|TestFdRace|TestFdReadRace|TestFileCloseRace"), "flag", "net", "os", "os/exec", "encoding/gob")
	// We don't want the following line, because it
	// slows down all.bash (by 10 seconds on my laptop).
	// The race builder should catch any error here, but doesn't.
	// TODO(iant): Figure out how to catch this.
	// t.addCmd(dt, "src", t.goTest(),  "-race", "-run=TestParallelTest", "cmd/go")
	if t.cgoEnabled {
		// Building misc/cgo/test takes a long time.
		// There are already cgo-enabled packages being tested with the race detector.
		// We shouldn't need to redo all of misc/cgo/test too.
		// The race buildler will take care of this.
		// cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), "-race")
		// setEnv(cmd, "GOTRACEBACK", "2")
	}
	if t.extLink() {
		// Test with external linking; see issue 9133.
		t.addCmd(dt, "src", t.goTest(), "-race", "-ldflags=-linkmode=external", t.runFlag("TestParse|TestEcho|TestStdinCloseRace"), "flag", "os/exec")
	}
	return nil
}

var runtest struct {
	sync.Once
	exe string
	err error
}

func (t *tester) testDirTest(dt *distTest, shard, shards int) error {
	runtest.Do(func() {
		f, err := ioutil.TempFile("", "runtest-*.exe") // named exe for Windows, but harmless elsewhere
		if err != nil {
			runtest.err = err
			return
		}
		f.Close()

		runtest.exe = f.Name()
		xatexit(func() {
			os.Remove(runtest.exe)
		})

		cmd := t.dirCmd("test", "go", "build", "-o", runtest.exe, "run.go")
		setEnv(cmd, "GOOS", gohostos)
		setEnv(cmd, "GOARCH", gohostarch)
		runtest.err = cmd.Run()
	})
	if runtest.err != nil {
		return runtest.err
	}
	if t.compileOnly {
		return nil
	}
	t.addCmd(dt, "test", runtest.exe,
		fmt.Sprintf("--shard=%d", shard),
		fmt.Sprintf("--shards=%d", shards),
	)
	return nil
}

// cgoPackages is the standard packages that use cgo.
var cgoPackages = []string{
	"net",
	"os/user",
}

var funcBenchmark = []byte("\nfunc Benchmark")

// packageHasBenchmarks reports whether pkg has benchmarks.
// On any error, it conservatively returns true.
//
// This exists just to eliminate work on the builders, since compiling
// a test in race mode just to discover it has no benchmarks costs a
// second or two per package, and this function returns false for
// about 100 packages.
func (t *tester) packageHasBenchmarks(pkg string) bool {
	pkgDir := filepath.Join(goroot, "src", pkg)
	d, err := os.Open(pkgDir)
	if err != nil {
		return true // conservatively
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		return true // conservatively
	}
	for _, name := range names {
		if !strings.HasSuffix(name, "_test.go") {
			continue
		}
		slurp, err := ioutil.ReadFile(filepath.Join(pkgDir, name))
		if err != nil {
			return true // conservatively
		}
		if bytes.Contains(slurp, funcBenchmark) {
			return true
		}
	}
	return false
}

// makeGOROOTUnwritable makes all $GOROOT files & directories non-writable to
// check that no tests accidentally write to $GOROOT.
func (t *tester) makeGOROOTUnwritable() (undo func()) {
	dir := os.Getenv("GOROOT")
	if dir == "" {
		panic("GOROOT not set")
	}

	type pathMode struct {
		path string
		mode os.FileMode
	}
	var dirs []pathMode // in lexical order

	undo = func() {
		for i := range dirs {
			os.Chmod(dirs[i].path, dirs[i].mode) // best effort
		}
	}

	gocache := os.Getenv("GOCACHE")
	if gocache == "" {
		panic("GOCACHE not set")
	}
	gocacheSubdir, _ := filepath.Rel(dir, gocache)

	// Note: Can't use WalkDir here, because this has to compile with Go 1.4.
	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if suffix := strings.TrimPrefix(path, dir+string(filepath.Separator)); suffix != "" {
			if suffix == gocacheSubdir {
				// Leave GOCACHE writable: we may need to write test binaries into it.
				return filepath.SkipDir
			}
			if suffix == ".git" {
				// Leave Git metadata in whatever state it was in. It may contain a lot
				// of files, and it is highly unlikely that a test will try to modify
				// anything within that directory.
				return filepath.SkipDir
			}
		}
		if err == nil {
			mode := info.Mode()
			if mode&0222 != 0 && (mode.IsDir() || mode.IsRegular()) {
				dirs = append(dirs, pathMode{path, mode})
			}
		}
		return nil
	})

	// Run over list backward to chmod children before parents.
	for i := len(dirs) - 1; i >= 0; i-- {
		err := os.Chmod(dirs[i].path, dirs[i].mode&^0222)
		if err != nil {
			dirs = dirs[i:] // Only undo what we did so far.
			undo()
			fatalf("failed to make GOROOT read-only: %v", err)
		}
	}

	return undo
}

// shouldUsePrecompiledStdTest reports whether "dist test" should use
// a pre-compiled go test binary on disk rather than running "go test"
// and compiling it again. This is used by our slow qemu-based builder
// that do full processor emulation where we cross-compile the
// make.bash step as well as pre-compile each std test binary.
//
// This only reports true if dist is run with an single go_test:foo
// argument (as the build coordinator does with our slow qemu-based
// builders), we're in a builder environment ("GO_BUILDER_NAME" is set),
// and the pre-built test binary exists.
func (t *tester) shouldUsePrecompiledStdTest() bool {
	bin := t.prebuiltGoPackageTestBinary()
	if bin == "" {
		return false
	}
	_, err := os.Stat(bin)
	return err == nil
}

func (t *tester) shouldTestCmd() bool {
	if goos == "js" && goarch == "wasm" {
		// Issues 25911, 35220
		return false
	}
	return true
}

// prebuiltGoPackageTestBinary returns the path where we'd expect
// the pre-built go test binary to be on disk when dist test is run with
// a single argument.
// It returns an empty string if a pre-built binary should not be used.
func (t *tester) prebuiltGoPackageTestBinary() string {
	if len(stdMatches) != 1 || t.race || t.compileOnly || os.Getenv("GO_BUILDER_NAME") == "" {
		return ""
	}
	pkg := stdMatches[0]
	return filepath.Join(os.Getenv("GOROOT"), "src", pkg, path.Base(pkg)+".test")
}

// runPrecompiledStdTest runs the pre-compiled standard library package test binary.
// See shouldUsePrecompiledStdTest above; it must return true for this to be called.
func (t *tester) runPrecompiledStdTest(timeout time.Duration) error {
	bin := t.prebuiltGoPackageTestBinary()
	fmt.Fprintf(os.Stderr, "# %s: using pre-built %s...\n", stdMatches[0], bin)
	cmd := exec.Command(bin, "-test.short="+short(), "-test.timeout="+timeout.String())
	setDir(cmd, filepath.Dir(bin))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	// And start a timer to kill the process if it doesn't kill
	// itself in the prescribed timeout.
	const backupKillFactor = 1.05 // add 5%
	timer := time.AfterFunc(time.Duration(float64(timeout)*backupKillFactor), func() {
		fmt.Fprintf(os.Stderr, "# %s: timeout running %s; killing...\n", stdMatches[0], bin)
		cmd.Process.Kill()
	})
	defer timer.Stop()
	return cmd.Wait()
}

// raceDetectorSupported is a copy of the function
// cmd/internal/sys.RaceDetectorSupported, which can't be used here
// because cmd/dist has to be buildable by Go 1.4.
// The race detector only supports 48-bit VMA on arm64. But we don't have
// a good solution to check VMA size(See https://golang.org/issue/29948)
// raceDetectorSupported will always return true for arm64. But race
// detector tests may abort on non 48-bit VMA configuration, the tests
// will be marked as "skipped" in this case.
func raceDetectorSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "amd64" || goarch == "ppc64le" || goarch == "arm64" || goarch == "s390x"
	case "darwin":
		return goarch == "amd64" || goarch == "arm64"
	case "freebsd", "netbsd", "openbsd", "windows":
		return goarch == "amd64"
	default:
		return false
	}
}

// isUnsupportedVMASize reports whether the failure is caused by an unsupported
// VMA for the race detector (for example, running the race detector on an
// arm64 machine configured with 39-bit VMA)
func isUnsupportedVMASize(w *work) bool {
	unsupportedVMA := []byte("unsupported VMA range")
	return w.dt.name == "race" && bytes.Contains(w.out, unsupportedVMA)
}

// isEnvSet reports whether the environment variable evar is
// set in the environment.
func isEnvSet(evar string) bool {
	evarEq := evar + "="
	for _, e := range os.Environ() {
		if strings.HasPrefix(e, evarEq) {
			return true
		}
	}
	return false
}
