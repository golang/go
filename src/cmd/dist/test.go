// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
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
	xflagparse(-1) // any number of args
	if noRebuild {
		t.rebuild = false
	}
	t.run()
}

// tester executes cmdtest.
type tester struct {
	race        bool
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

	var exeSuffix string
	if goos == "windows" {
		exeSuffix = ".exe"
	}
	if _, err := os.Stat(filepath.Join(gobin, "go"+exeSuffix)); err == nil {
		os.Setenv("PATH", fmt.Sprintf("%s%c%s", gobin, os.PathListSeparator, os.Getenv("PATH")))
	}

	slurp, err := exec.Command("go", "env", "CGO_ENABLED").Output()
	if err != nil {
		log.Fatalf("Error running go env CGO_ENABLED: %v", err)
	}
	t.cgoEnabled, _ = strconv.ParseBool(strings.TrimSpace(string(slurp)))
	if flag.NArg() > 0 && t.runRxStr != "" {
		log.Fatalf("the -run regular expression flag is mutually exclusive with test name arguments")
	}

	t.runNames = flag.Args()

	if t.hasBash() {
		if _, err := exec.LookPath("time"); err == nil {
			t.haveTime = true
		}
	}

	if t.rebuild {
		t.out("Building packages and commands.")
		// Force rebuild the whole toolchain.
		goInstall("go", append([]string{"-a", "-i"}, toolchain...)...)
	}

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
	//
	// But don't do this if we're running in the Go build system,
	// where cmd/dist is invoked many times. This just slows that
	// down (Issue 24300).
	if !t.listMode && os.Getenv("GO_BUILDER_NAME") == "" {
		goInstall("go", append([]string{"-i"}, toolchain...)...)
		goInstall("go", append([]string{"-i"}, toolchain...)...)
		goInstall("go", "std", "cmd")
		checkNotStale("go", "std", "cmd")
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
			log.Fatalf("failed to parse $GO_TEST_TIMEOUT_SCALE = %q as integer: %v", s, err)
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

	// We must unset GOROOT_FINAL before tests, because runtime/debug requires
	// correct access to source code, so if we have GOROOT_FINAL in effect,
	// at least runtime/debug test will fail.
	// If GOROOT_FINAL was set before, then now all the commands will appear stale.
	// Nothing we can do about that other than not checking them below.
	// (We call checkNotStale but only with "std" not "cmd".)
	os.Setenv("GOROOT_FINAL_OLD", os.Getenv("GOROOT_FINAL")) // for cmd/link test
	os.Unsetenv("GOROOT_FINAL")

	for _, name := range t.runNames {
		if !t.isRegisteredTestName(name) {
			log.Fatalf("unknown test %q", name)
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
				log.Fatalf("Failed: %v", err)
			}
		}
	}
	t.runPending(nil)
	timelog("end", "dist test")
	if t.failed {
		fmt.Println("\nFAILED")
		os.Exit(1)
	} else if incomplete[goos+"/"+goarch] {
		fmt.Println("\nFAILED (incomplete port)")
		os.Exit(1)
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

// short returns a -short flag to pass to 'go test'.
// It returns "-short", unless the environment variable
// GO_TEST_SHORT is set to a non-empty, false-ish string.
//
// This environment variable is meant to be an internal
// detail between the Go build system and cmd/dist
// and is not intended for use by users.
func short() string {
	if v := os.Getenv("GO_TEST_SHORT"); v != "" {
		short, err := strconv.ParseBool(v)
		if err != nil {
			log.Fatalf("invalid GO_TEST_SHORT %q: %v", v, err)
		}
		if !short {
			return "-short=false"
		}
	}
	return "-short"
}

// goTest returns the beginning of the go test command line.
// Callers should use goTest and then pass flags overriding these
// defaults as later arguments in the command line.
func (t *tester) goTest() []string {
	return []string{
		"go", "test", short(), "-count=1", t.tags(), t.runFlag(""),
	}
}

func (t *tester) tags() string {
	if t.iOS() {
		return "-tags=lldb"
	}
	return "-tags="
}

func (t *tester) timeout(sec int) string {
	return "-timeout=" + fmt.Sprint(time.Duration(sec)*time.Second*time.Duration(t.timeoutScale))
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
	testName := "go_test:" + pkg
	if t.runRx == nil || t.runRx.MatchString(testName) == t.runRxWant {
		stdMatches = append(stdMatches, pkg)
	}
	t.tests = append(t.tests, distTest{
		name:    testName,
		heading: "Testing packages.",
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
			args := []string{
				"test",
				short(),
				t.tags(),
				t.timeout(timeoutSec),
				"-gcflags=all=" + gogcflags,
			}
			if t.race {
				args = append(args, "-race")
			}
			if t.compileOnly {
				args = append(args, "-run=^$")
			} else if goos == "js" && goarch == "wasm" {
				args = append(args, "-run=^Test") // exclude examples; Issue 25913
			}
			args = append(args, stdMatches...)
			cmd := exec.Command("go", args...)
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
				short(),
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
			cmd := exec.Command("go", args...)
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
	if strings.HasSuffix(os.Getenv("GO_BUILDER_NAME"), "-vetall") {
		// Run vet over std and cmd and call it quits.
		for k := range cgoEnabled {
			osarch := k
			t.tests = append(t.tests, distTest{
				name:    "vet/" + osarch,
				heading: "cmd/vet/all",
				fn: func(dt *distTest) error {
					t.addCmd(dt, "src/cmd/vet/all", "go", "run", "main.go", "-p="+osarch)
					return nil
				},
			})
		}
		return
	}

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
		cmd := exec.Command("go", "list", "-f", format)
		if t.race {
			cmd.Args = append(cmd.Args, "-tags=race")
		}
		cmd.Args = append(cmd.Args, "std")
		if !t.race {
			cmd.Args = append(cmd.Args, "cmd")
		}
		all, err := cmd.Output()
		if err != nil {
			log.Fatalf("Error running go list std cmd: %v, %s", err, all)
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
				cmd := t.addCmd(dt, "src", t.goTest(), t.timeout(300), "runtime", "-cpu=1,2,4", "-quick")
				// We set GOMAXPROCS=2 in addition to -cpu=1,2,4 in order to test runtime bootstrap code,
				// creation of first goroutines and first garbage collections in the parallel setting.
				cmd.Env = append(os.Environ(), "GOMAXPROCS=2")
				return nil
			},
		})
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
				cmd := exec.Command("go", "test")
				cmd.Dir = filepath.Join(os.Getenv("GOROOT"), "src/cmd/go/testdata/testterminal18153")
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

				// Run `go test fmt` in the moved GOROOT.
				cmd := exec.Command(filepath.Join(moved, "bin", "go"), "test", "fmt")
				cmd.Stdout = os.Stdout
				cmd.Stderr = os.Stderr
				// Don't set GOROOT in the environment.
				for _, e := range os.Environ() {
					if !strings.HasPrefix(e, "GOROOT=") && !strings.HasPrefix(e, "GOCACHE=") {
						cmd.Env = append(cmd.Env, e)
					}
				}
				err := cmd.Run()

				if rerr := os.Rename(moved, goroot); rerr != nil {
					log.Fatalf("failed to restore GOROOT: %v", rerr)
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
				t.addCmd(dt, "src", t.goTest(), "-ldflags=-linkmode=internal -libgcc=none", pkg, t.runFlag(run))
				return nil
			},
		})
	}

	// Test internal linking of PIE binaries where it is supported.
	if goos == "linux" && goarch == "amd64" && !isAlpineLinux() {
		// Issue 18243: We don't have a way to set the default
		// dynamic linker used in internal linking mode. So
		// this test is skipped on Alpine.
		t.tests = append(t.tests, distTest{
			name:    "pie_internal",
			heading: "internal linking of -buildmode=pie",
			fn: func(dt *distTest) error {
				t.addCmd(dt, "src", t.goTest(), "reflect", "-buildmode=pie", "-ldflags=-linkmode=internal", t.timeout(60))
				return nil
			},
		})
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
		t.tests = append(t.tests, distTest{
			name:    "cgo_stdio",
			heading: "../misc/cgo/stdio",
			fn: func(dt *distTest) error {
				t.addCmd(dt, "misc/cgo/stdio", "go", "run", filepath.Join(os.Getenv("GOROOT"), "test/run.go"), "-", ".")
				return nil
			},
		})
		t.tests = append(t.tests, distTest{
			name:    "cgo_life",
			heading: "../misc/cgo/life",
			fn: func(dt *distTest) error {
				t.addCmd(dt, "misc/cgo/life", "go", "run", filepath.Join(os.Getenv("GOROOT"), "test/run.go"), "-", ".")
				return nil
			},
		})
		fortran := os.Getenv("FC")
		if fortran == "" {
			fortran, _ = exec.LookPath("gfortran")
		}
		if t.hasBash() && fortran != "" {
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
					t.addCmd(dt, "misc/swig/stdio", t.goTest())
					return nil
				},
			})
			if t.hasCxx() {
				t.tests = append(t.tests, distTest{
					name:    "swig_callback",
					heading: "../misc/swig/callback",
					fn: func(dt *distTest) error {
						t.addCmd(dt, "misc/swig/callback", t.goTest())
						return nil
					},
				})
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

	if t.hasBash() && t.cgoEnabled && goos != "android" && goos != "darwin" {
		t.registerTest("testgodefs", "../misc/cgo/testgodefs", "./test.bash")
	}

	// Don't run these tests with $GO_GCFLAGS because most of them
	// assume that they can run "go install" with no -gcflags and not
	// recompile the entire standard library. If make.bash ran with
	// special -gcflags, that's not true.
	if t.cgoEnabled && gogcflags == "" {
		if t.cgoTestSOSupported() {
			t.tests = append(t.tests, distTest{
				name:    "testso",
				heading: "../misc/cgo/testso",
				fn: func(dt *distTest) error {
					return t.cgoTestSO(dt, "misc/cgo/testso")
				},
			})
			t.tests = append(t.tests, distTest{
				name:    "testsovar",
				heading: "../misc/cgo/testsovar",
				fn: func(dt *distTest) error {
					return t.cgoTestSO(dt, "misc/cgo/testsovar")
				},
			})
		}
		if t.supportedBuildmode("c-archive") {
			t.registerHostTest("testcarchive", "../misc/cgo/testcarchive", "misc/cgo/testcarchive", "carchive_test.go")
		}
		if t.supportedBuildmode("c-shared") {
			t.registerHostTest("testcshared", "../misc/cgo/testcshared", "misc/cgo/testcshared", "cshared_test.go")
		}
		if t.supportedBuildmode("shared") {
			t.registerTest("testshared", "../misc/cgo/testshared", t.goTest(), t.timeout(600))
		}
		if t.supportedBuildmode("plugin") {
			t.registerTest("testplugin", "../misc/cgo/testplugin", "./test.bash")
		}
		if gohostos == "linux" && goarch == "amd64" {
			t.registerTest("testasan", "../misc/cgo/testasan", "go", "run", "main.go")
		}
		if mSanSupported(goos, goarch) {
			t.registerHostTest("testsanitizers/msan", "../misc/cgo/testsanitizers", "misc/cgo/testsanitizers", ".")
		}
		if t.hasBash() && goos != "android" && !t.iOS() && gohostos != "windows" {
			t.registerHostTest("cgo_errors", "../misc/cgo/errors", "misc/cgo/errors", ".")
		}
		if gohostos == "linux" && t.extLink() {
			t.registerTest("testsigfwd", "../misc/cgo/testsigfwd", "go", "run", "main.go")
		}
	}

	// Doc tests only run on builders.
	// They find problems approximately never.
	if t.hasBash() && goos != "nacl" && goos != "js" && goos != "android" && !t.iOS() && os.Getenv("GO_BUILDER_NAME") != "" {
		t.registerTest("doc_progs", "../doc/progs", "time", "go", "run", "run.go")
		t.registerTest("wiki", "../doc/articles/wiki", "./test.bash")
		t.registerTest("codewalk", "../doc/codewalk", "time", "./run")
	}

	if goos != "android" && !t.iOS() {
		t.registerTest("bench_go1", "../test/bench/go1", t.goTest(), t.timeout(600))
	}
	if goos != "android" && !t.iOS() {
		// Only start multiple test dir shards on builders,
		// where they get distributed to multiple machines.
		// See issue 20141.
		nShards := 1
		if os.Getenv("GO_BUILDER_NAME") != "" {
			nShards = 10
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
	if goos != "nacl" && goos != "android" && !t.iOS() && goos != "js" {
		t.tests = append(t.tests, distTest{
			name:    "api",
			heading: "API check",
			fn: func(dt *distTest) error {
				if t.compileOnly {
					t.addCmd(dt, "src", "go", "build", filepath.Join(goroot, "src/cmd/api/run.go"))
					return nil
				}
				t.addCmd(dt, "src", "go", "run", filepath.Join(goroot, "src/cmd/api/run.go"))
				return nil
			},
		})
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
		cmd.Dir = dir
	} else {
		cmd.Dir = filepath.Join(goroot, dir)
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

	return list[0], list[1:]
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
	return goos == "darwin" && (goarch == "arm" || goarch == "arm64")
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
	case "android-arm",
		"darwin-386", "darwin-amd64", "darwin-arm", "darwin-arm64",
		"dragonfly-amd64",
		"freebsd-386", "freebsd-amd64", "freebsd-arm",
		"linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-ppc64le", "linux-mips64", "linux-mips64le", "linux-mips", "linux-mipsle", "linux-s390x",
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
	if gohostarch == "ppc64le" {
		// linkmode=internal fails on ppc64le because cmd/link doesn't
		// handle the TOC correctly (issue 15409).
		return false
	}
	if goos == "android" {
		return false
	}
	if goos == "darwin" && (goarch == "arm" || goarch == "arm64") {
		return false
	}
	// Internally linking cgo is incomplete on some architectures.
	// https://golang.org/issue/10373
	// https://golang.org/issue/14449
	if goarch == "arm64" || goarch == "mips64" || goarch == "mips64le" || goarch == "mips" || goarch == "mipsle" {
		return false
	}
	if isAlpineLinux() {
		// Issue 18243.
		return false
	}
	return true
}

func (t *tester) supportedBuildmode(mode string) bool {
	pair := goos + "-" + goarch
	switch mode {
	case "c-archive":
		if !t.extLink() {
			return false
		}
		switch pair {
		case "darwin-386", "darwin-amd64", "darwin-arm", "darwin-arm64",
			"linux-amd64", "linux-386", "linux-ppc64le", "linux-s390x",
			"freebsd-amd64",
			"windows-amd64", "windows-386":
			return true
		}
		return false
	case "c-shared":
		switch pair {
		case "linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-ppc64le", "linux-s390x",
			"darwin-amd64", "darwin-386",
			"freebsd-amd64",
			"android-arm", "android-arm64", "android-386",
			"windows-amd64", "windows-386":
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
		// linux-arm64 is missing because it causes the external linker
		// to crash, see https://golang.org/issue/17138
		switch pair {
		case "linux-386", "linux-amd64", "linux-arm", "linux-s390x", "linux-ppc64le":
			return true
		case "darwin-amd64":
			return true
		}
		return false
	case "pie":
		switch pair {
		case "linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-ppc64le", "linux-s390x",
			"android-amd64", "android-arm", "android-arm64", "android-386":
			return true
		case "darwin-amd64":
			return true
		}
		return false

	default:
		log.Fatalf("internal error: unknown buildmode %s", mode)
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
	defer os.Remove(filepath.Join(goroot, dir, "test.test"))
	cmd := t.dirCmd(dir, t.goTest(), "-c", "-o", "test.test", pkg)
	cmd.Env = append(os.Environ(), "GOARCH="+gohostarch, "GOOS="+gohostos)
	if err := cmd.Run(); err != nil {
		return err
	}
	return t.dirCmd(dir, "./test.test").Run()
}

func (t *tester) cgoTest(dt *distTest) error {
	t.addCmd(dt, "misc/cgo/test", t.goTest(), "-ldflags", "-linkmode=auto")

	if t.internalLink() {
		t.addCmd(dt, "misc/cgo/test", t.goTest(), "-tags=internal", "-ldflags", "-linkmode=internal")
	}

	pair := gohostos + "-" + goarch
	switch pair {
	case "darwin-386", "darwin-amd64",
		"openbsd-386", "openbsd-amd64",
		"windows-386", "windows-amd64":
		// test linkmode=external, but __thread not supported, so skip testtls.
		if !t.extLink() {
			break
		}
		t.addCmd(dt, "misc/cgo/test", t.goTest(), "-ldflags", "-linkmode=external")
		t.addCmd(dt, "misc/cgo/test", t.goTest(), "-ldflags", "-linkmode=external -s")
	case "android-arm",
		"dragonfly-amd64",
		"freebsd-386", "freebsd-amd64", "freebsd-arm",
		"linux-386", "linux-amd64", "linux-arm", "linux-ppc64le", "linux-s390x",
		"netbsd-386", "netbsd-amd64":

		cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), "-ldflags", "-linkmode=external")
		// A -g argument in CGO_CFLAGS should not affect how the test runs.
		cmd.Env = append(os.Environ(), "CGO_CFLAGS=-g0")

		t.addCmd(dt, "misc/cgo/testtls", t.goTest(), "-ldflags", "-linkmode=auto")
		t.addCmd(dt, "misc/cgo/testtls", t.goTest(), "-ldflags", "-linkmode=external")

		switch pair {
		case "netbsd-386", "netbsd-amd64":
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
					t.addCmd(dt, "misc/cgo/testtls", t.goTest(), "-ldflags", `-linkmode=external -extldflags "-static -pthread"`)
				}
				t.addCmd(dt, "misc/cgo/nocgo", t.goTest())
				t.addCmd(dt, "misc/cgo/nocgo", t.goTest(), "-ldflags", `-linkmode=external`)
				if goos != "android" {
					t.addCmd(dt, "misc/cgo/nocgo", t.goTest(), "-ldflags", `-linkmode=external -extldflags "-static -pthread"`)
					t.addCmd(dt, "misc/cgo/test", t.goTest(), "-tags=static", "-ldflags", `-linkmode=external -extldflags "-static -pthread"`)
					// -static in CGO_LDFLAGS triggers a different code path
					// than -static in -extldflags, so test both.
					// See issue #16651.
					cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), "-tags=static")
					cmd.Env = append(os.Environ(), "CGO_LDFLAGS=-static -pthread")
				}
			}

			if t.supportedBuildmode("pie") {
				t.addCmd(dt, "misc/cgo/test", t.goTest(), "-buildmode=pie")
				t.addCmd(dt, "misc/cgo/testtls", t.goTest(), "-buildmode=pie")
				t.addCmd(dt, "misc/cgo/nocgo", t.goTest(), "-buildmode=pie")
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
			//println("start", started)
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
		//println("wait", ended)
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
		log.Fatal("FAILED")
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

func (t *tester) cgoTestSOSupported() bool {
	if goos == "android" || t.iOS() {
		// No exec facility on Android or iOS.
		return false
	}
	if goarch == "ppc64" {
		// External linking not implemented on ppc64 (issue #8912).
		return false
	}
	if goarch == "mips64le" || goarch == "mips64" {
		// External linking not implemented on mips64.
		return false
	}
	return true
}

func (t *tester) cgoTestSO(dt *distTest, testpath string) error {
	t.runPending(dt)

	timelog("start", dt.name)
	defer timelog("end", dt.name)

	dir := filepath.Join(goroot, testpath)

	// build shared object
	output, err := exec.Command("go", "env", "CC").Output()
	if err != nil {
		return fmt.Errorf("Error running go env CC: %v", err)
	}
	cc := strings.TrimSuffix(string(output), "\n")
	if cc == "" {
		return errors.New("CC environment variable (go env CC) cannot be empty")
	}
	output, err = exec.Command("go", "env", "GOGCCFLAGS").Output()
	if err != nil {
		return fmt.Errorf("Error running go env GOGCCFLAGS: %v", err)
	}
	gogccflags := strings.Split(strings.TrimSuffix(string(output), "\n"), " ")

	ext := "so"
	args := append(gogccflags, "-shared")
	switch goos {
	case "darwin":
		ext = "dylib"
		args = append(args, "-undefined", "suppress", "-flat_namespace")
	case "windows":
		ext = "dll"
		args = append(args, "-DEXPORT_DLL")
	}
	sofname := "libcgosotest." + ext
	args = append(args, "-o", sofname, "cgoso_c.c")

	if err := t.dirCmd(dir, cc, args).Run(); err != nil {
		return err
	}
	defer os.Remove(filepath.Join(dir, sofname))

	if err := t.dirCmd(dir, "go", "build", "-o", "main.exe", "main.go").Run(); err != nil {
		return err
	}
	defer os.Remove(filepath.Join(dir, "main.exe"))

	cmd := t.dirCmd(dir, "./main.exe")
	if goos != "windows" {
		s := "LD_LIBRARY_PATH"
		if goos == "darwin" {
			s = "DYLD_LIBRARY_PATH"
		}
		cmd.Env = append(os.Environ(), s+"=.")

		// On FreeBSD 64-bit architectures, the 32-bit linker looks for
		// different environment variables.
		if goos == "freebsd" && gohostarch == "386" {
			cmd.Env = append(cmd.Env, "LD_32_LIBRARY_PATH=.")
		}
	}
	return cmd.Run()
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
	if rx == "" && goos == "js" && goarch == "wasm" {
		return "-run=^Test" // exclude examples; Issue 25913
	}
	return "-run=" + rx
}

func (t *tester) raceTest(dt *distTest) error {
	t.addCmd(dt, "src", t.goTest(), "-race", "-i", "runtime/race", "flag", "os", "os/exec")
	t.addCmd(dt, "src", t.goTest(), "-race", t.runFlag("Output"), "runtime/race")
	t.addCmd(dt, "src", t.goTest(), "-race", t.runFlag("TestParse|TestEcho|TestStdinCloseRace|TestClosedPipeRace|TestTypeRace|TestFdRace|TestFdReadRace|TestFileCloseRace"), "flag", "net", "os", "os/exec", "encoding/gob")
	// We don't want the following line, because it
	// slows down all.bash (by 10 seconds on my laptop).
	// The race builder should catch any error here, but doesn't.
	// TODO(iant): Figure out how to catch this.
	// t.addCmd(dt, "src", t.goTest(),  "-race", "-run=TestParallelTest", "cmd/go")
	if t.cgoEnabled {
		cmd := t.addCmd(dt, "misc/cgo/test", t.goTest(), "-race")
		cmd.Env = append(os.Environ(), "GOTRACEBACK=2")
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
		const exe = "runtest.exe" // named exe for Windows, but harmless elsewhere
		cmd := t.dirCmd("test", "go", "build", "-o", exe, "run.go")
		cmd.Env = append(os.Environ(), "GOOS="+gohostos, "GOARCH="+gohostarch)
		runtest.exe = filepath.Join(cmd.Dir, exe)
		if err := cmd.Run(); err != nil {
			runtest.err = err
			return
		}
		xatexit(func() {
			os.Remove(runtest.exe)
		})
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
	"crypto/x509",
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

// raceDetectorSupported is a copy of the function
// cmd/internal/sys.RaceDetectorSupported, which can't be used here
// because cmd/dist has to be buildable by Go 1.4.
func raceDetectorSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "amd64" || goarch == "ppc64le" || goarch == "arm64"
	case "darwin", "freebsd", "netbsd", "windows":
		return goarch == "amd64"
	default:
		return false
	}
}

// mSanSupported is a copy of the function cmd/internal/sys.MSanSupported,
// which can't be used here because cmd/dist has to be buildable by Go 1.4.
func mSanSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "amd64" || goarch == "arm64"
	default:
		return false
	}
}
