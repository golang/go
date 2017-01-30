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
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"
)

func cmdtest() {
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

	goroot     string
	goarch     string
	gohostarch string
	goos       string
	gohostos   string
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

func mustEnv(k string) string {
	v := os.Getenv(k)
	if v == "" {
		log.Fatalf("Unset environment variable %v", k)
	}
	return v
}

func (t *tester) run() {
	t.goroot = mustEnv("GOROOT")
	t.goos = mustEnv("GOOS")
	t.gohostos = mustEnv("GOHOSTOS")
	t.goarch = mustEnv("GOARCH")
	t.gohostarch = mustEnv("GOHOSTARCH")
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
		cmd := exec.Command("go", "install", "-a", "-v", "std", "cmd")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatalf("building packages and commands: %v", err)
		}
	}

	if t.iOS() {
		// Install the Mach exception handler used to intercept
		// EXC_BAD_ACCESS and convert it into a Go panic. This is
		// necessary for a Go program running under lldb (the way
		// we run tests). It is disabled by default because iOS
		// apps are not allowed to access the exc_server symbol.
		cmd := exec.Command("go", "install", "-a", "-tags", "lldb", "runtime/cgo")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatalf("building mach exception handler: %v", err)
		}

		defer func() {
			cmd := exec.Command("go", "install", "-a", "runtime/cgo")
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			if err := cmd.Run(); err != nil {
				log.Fatalf("reverting mach exception handler: %v", err)
			}
		}()
	}

	t.timeoutScale = 1
	switch t.goarch {
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

	// we must unset GOROOT_FINAL before tests, because runtime/debug requires
	// correct access to source code, so if we have GOROOT_FINAL in effect,
	// at least runtime/debug test will fail.
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
	if t.failed {
		fmt.Println("\nFAILED")
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
	if t.runRx == nil || t.runRx.MatchString(testName) {
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
			ranGoTest = true
			args := []string{
				"test",
				"-short",
				t.tags(),
				t.timeout(180),
				"-gcflags=" + os.Getenv("GO_GCFLAGS"),
			}
			if t.race {
				args = append(args, "-race")
			}
			if t.compileOnly {
				args = append(args, "-run=^$")
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
	if t.runRx == nil || t.runRx.MatchString(testName) {
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
			ranGoBench = true
			args := []string{
				"test",
				"-short",
				"-race",
				"-run=^$", // nothing. only benchmarks.
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
		t.tests = append(t.tests, distTest{
			name:    "vet/all",
			heading: "go vet std cmd",
			fn: func(dt *distTest) error {
				// This runs vet/all for the current platform.
				// TODO: on a fast builder or builders, run over all platforms.
				t.addCmd(dt, "src/cmd/vet/all", "go", "run", "main.go", "-all")
				return nil
			},
		})
		return
	}

	// This test needs its stdout/stderr to be terminals, so we don't run it from cmd/go's tests.
	// See issue 18153.
	if t.goos == "linux" {
		t.tests = append(t.tests, distTest{
			name:    "cmd_go_test_terminal",
			heading: "cmd/go terminal test",
			fn: func(dt *distTest) error {
				t.runPending(dt)
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
			cmd.Args = append(cmd.Args, "-tags", "race")
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

	if t.race {
		return
	}

	// Runtime CPU tests.
	if !t.compileOnly {
		testName := "runtime:cpu124"
		t.tests = append(t.tests, distTest{
			name:    testName,
			heading: "GOMAXPROCS=2 runtime -cpu=1,2,4",
			fn: func(dt *distTest) error {
				cmd := t.addCmd(dt, "src", "go", "test", "-short", t.timeout(300), t.tags(), "runtime", "-cpu=1,2,4")
				// We set GOMAXPROCS=2 in addition to -cpu=1,2,4 in order to test runtime bootstrap code,
				// creation of first goroutines and first garbage collections in the parallel setting.
				cmd.Env = mergeEnvLists([]string{"GOMAXPROCS=2"}, os.Environ())
				return nil
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
		if t.goarch == "arm" {
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
				t.addCmd(dt, "src", "go", "test", "-short", "-ldflags=-linkmode=internal -libgcc=none", t.tags(), pkg, t.runFlag(run))
				return nil
			},
		})
	}

	// Test internal linking of PIE binaries where it is supported.
	if t.goos == "linux" && t.goarch == "amd64" {
		t.tests = append(t.tests, distTest{
			name:    "pie_internal",
			heading: "internal linking of -buildmode=pie",
			fn: func(dt *distTest) error {
				t.addCmd(dt, "src", "go", "test", "reflect", "-short", "-buildmode=pie", "-ldflags=-linkmode=internal", t.timeout(60), t.tags(), t.runFlag(""))
				return nil
			},
		})
	}

	// sync tests
	t.tests = append(t.tests, distTest{
		name:    "sync_cpu",
		heading: "sync -cpu=10",
		fn: func(dt *distTest) error {
			t.addCmd(dt, "src", "go", "test", "sync", "-short", t.timeout(120), t.tags(), "-cpu=10", t.runFlag(""))
			return nil
		},
	})

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
	}
	if t.cgoEnabled {
		t.tests = append(t.tests, distTest{
			name:    "cgo_test",
			heading: "../misc/cgo/test",
			fn:      t.cgoTest,
		})
	}

	if t.raceDetectorSupported() {
		t.tests = append(t.tests, distTest{
			name:    "race",
			heading: "Testing race detector",
			fn:      t.raceTest,
		})
	}

	if t.hasBash() && t.cgoEnabled && t.goos != "android" && t.goos != "darwin" {
		t.registerTest("testgodefs", "../misc/cgo/testgodefs", "./test.bash")
	}
	if t.cgoEnabled {
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
			t.registerTest("testcshared", "../misc/cgo/testcshared", "./test.bash")
		}
		if t.supportedBuildmode("shared") {
			t.registerTest("testshared", "../misc/cgo/testshared", "go", "test")
		}
		if t.supportedBuildmode("plugin") {
			t.registerTest("testplugin", "../misc/cgo/testplugin", "./test.bash")
		}
		if t.gohostos == "linux" && t.goarch == "amd64" {
			t.registerTest("testasan", "../misc/cgo/testasan", "go", "run", "main.go")
		}
		if t.goos == "linux" && t.goarch == "amd64" {
			t.registerTest("testsanitizers", "../misc/cgo/testsanitizers", "./test.bash")
		}
		if t.hasBash() && t.goos != "android" && !t.iOS() && t.gohostos != "windows" {
			t.registerTest("cgo_errors", "../misc/cgo/errors", "./test.bash")
		}
		if t.gohostos == "linux" && t.extLink() {
			t.registerTest("testsigfwd", "../misc/cgo/testsigfwd", "go", "run", "main.go")
		}
	}

	// Doc tests only run on builders.
	// They find problems approximately never.
	if t.hasBash() && t.goos != "nacl" && t.goos != "android" && !t.iOS() && os.Getenv("GO_BUILDER_NAME") != "" {
		t.registerTest("doc_progs", "../doc/progs", "time", "go", "run", "run.go")
		t.registerTest("wiki", "../doc/articles/wiki", "./test.bash")
		t.registerTest("codewalk", "../doc/codewalk", "time", "./run")
	}

	if t.goos != "android" && !t.iOS() {
		t.registerTest("bench_go1", "../test/bench/go1", "go", "test", t.timeout(600), t.runFlag(""))
	}
	if t.goos != "android" && !t.iOS() {
		const nShards = 5
		for shard := 0; shard < nShards; shard++ {
			shard := shard
			t.tests = append(t.tests, distTest{
				name:    fmt.Sprintf("test:%d_%d", shard, nShards),
				heading: "../test",
				fn:      func(dt *distTest) error { return t.testDirTest(dt, shard, nShards) },
			})
		}
	}
	if t.goos != "nacl" && t.goos != "android" && !t.iOS() {
		t.tests = append(t.tests, distTest{
			name:    "api",
			heading: "API check",
			fn: func(dt *distTest) error {
				if t.compileOnly {
					t.addCmd(dt, "src", "go", "build", filepath.Join(t.goroot, "src/cmd/api/run.go"))
					return nil
				}
				t.addCmd(dt, "src", "go", "run", filepath.Join(t.goroot, "src/cmd/api/run.go"))
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

func (t *tester) registerTest1(seq bool, name, dirBanner, bin string, args ...string) {
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
				return t.dirCmd(filepath.Join(t.goroot, "src", dirBanner), bin, args...).Run()
			}
			t.addCmd(dt, filepath.Join(t.goroot, "src", dirBanner), bin, args...)
			return nil
		},
	})
}

func (t *tester) registerTest(name, dirBanner, bin string, args ...string) {
	t.registerTest1(false, name, dirBanner, bin, args...)
}

func (t *tester) registerSeqTest(name, dirBanner, bin string, args ...string) {
	t.registerTest1(true, name, dirBanner, bin, args...)
}

func (t *tester) bgDirCmd(dir, bin string, args ...string) *exec.Cmd {
	cmd := exec.Command(bin, args...)
	if filepath.IsAbs(dir) {
		cmd.Dir = dir
	} else {
		cmd.Dir = filepath.Join(t.goroot, dir)
	}
	return cmd
}

func (t *tester) dirCmd(dir, bin string, args ...string) *exec.Cmd {
	cmd := t.bgDirCmd(dir, bin, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if vflag > 1 {
		errprintf("%s\n", strings.Join(cmd.Args, " "))
	}
	return cmd
}

func (t *tester) addCmd(dt *distTest, dir, bin string, args ...string) *exec.Cmd {
	w := &work{
		dt:  dt,
		cmd: t.bgDirCmd(dir, bin, args...),
	}
	t.worklist = append(t.worklist, w)
	return w.cmd
}

func (t *tester) iOS() bool {
	return t.goos == "darwin" && (t.goarch == "arm" || t.goarch == "arm64")
}

func (t *tester) out(v string) {
	if t.banner == "" {
		return
	}
	fmt.Println("\n" + t.banner + v)
}

func (t *tester) extLink() bool {
	pair := t.gohostos + "-" + t.goarch
	switch pair {
	case "android-arm",
		"darwin-arm", "darwin-arm64",
		"dragonfly-386", "dragonfly-amd64",
		"freebsd-386", "freebsd-amd64", "freebsd-arm",
		"linux-386", "linux-amd64", "linux-arm", "linux-arm64", "linux-ppc64le", "linux-mips64", "linux-mips64le", "linux-mips", "linux-mipsle", "linux-s390x",
		"netbsd-386", "netbsd-amd64",
		"openbsd-386", "openbsd-amd64",
		"windows-386", "windows-amd64":
		return true
	case "darwin-386", "darwin-amd64":
		// linkmode=external fails on OS X 10.6 and earlier == Darwin
		// 10.8 and earlier.
		unameR, err := exec.Command("uname", "-r").Output()
		if err != nil {
			log.Fatalf("uname -r: %v", err)
		}
		major, _ := strconv.Atoi(string(unameR[:bytes.IndexByte(unameR, '.')]))
		return major > 10
	}
	return false
}

func (t *tester) internalLink() bool {
	if t.gohostos == "dragonfly" {
		// linkmode=internal fails on dragonfly since errno is a TLS relocation.
		return false
	}
	if t.gohostarch == "ppc64le" {
		// linkmode=internal fails on ppc64le because cmd/link doesn't
		// handle the TOC correctly (issue 15409).
		return false
	}
	if t.goos == "android" {
		return false
	}
	if t.goos == "darwin" && (t.goarch == "arm" || t.goarch == "arm64") {
		return false
	}
	// Internally linking cgo is incomplete on some architectures.
	// https://golang.org/issue/10373
	// https://golang.org/issue/14449
	if t.goarch == "arm64" || t.goarch == "mips64" || t.goarch == "mips64le" || t.goarch == "mips" || t.goarch == "mipsle" {
		return false
	}
	return true
}

func (t *tester) supportedBuildmode(mode string) bool {
	pair := t.goos + "-" + t.goarch
	switch mode {
	case "c-archive":
		if !t.extLink() {
			return false
		}
		switch pair {
		case "darwin-386", "darwin-amd64", "darwin-arm", "darwin-arm64",
			"linux-amd64", "linux-386", "windows-amd64", "windows-386":
			return true
		}
		return false
	case "c-shared":
		switch pair {
		case "linux-386", "linux-amd64", "linux-arm", "linux-arm64",
			"darwin-amd64", "darwin-386",
			"android-arm", "android-arm64", "android-386":
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
		if os.Getenv("GO_BUILDER_NAME") == "linux-amd64-noopt" {
			// Skip the plugin tests on noopt. They're
			// causing build failures potentially
			// obscuring other issues. This is hopefully a
			// temporary workaround. See golang.org/issue/17937.
			return false
		}

		// linux-arm64 is missing because it causes the external linker
		// to crash, see https://golang.org/issue/17138
		switch pair {
		case "linux-386", "linux-amd64", "linux-arm":
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
			return t.runHostTest(dir, pkg)
		},
	})
}

func (t *tester) runHostTest(dir, pkg string) error {
	env := mergeEnvLists([]string{"GOARCH=" + t.gohostarch, "GOOS=" + t.gohostos}, os.Environ())
	defer os.Remove(filepath.Join(t.goroot, dir, "test.test"))
	cmd := t.dirCmd(dir, "go", "test", t.tags(), "-c", "-o", "test.test", pkg)
	cmd.Env = env
	if err := cmd.Run(); err != nil {
		return err
	}
	return t.dirCmd(dir, "./test.test").Run()
}

func (t *tester) cgoTest(dt *distTest) error {
	env := mergeEnvLists([]string{"GOTRACEBACK=2"}, os.Environ())

	cmd := t.addCmd(dt, "misc/cgo/test", "go", "test", t.tags(), "-ldflags", "-linkmode=auto", t.runFlag(""))
	cmd.Env = env

	if t.internalLink() {
		cmd := t.addCmd(dt, "misc/cgo/test", "go", "test", "-ldflags", "-linkmode=internal", t.runFlag(""))
		cmd.Env = env
	}

	pair := t.gohostos + "-" + t.goarch
	switch pair {
	case "darwin-386", "darwin-amd64",
		"openbsd-386", "openbsd-amd64",
		"windows-386", "windows-amd64":
		// test linkmode=external, but __thread not supported, so skip testtls.
		if !t.extLink() {
			break
		}
		cmd := t.addCmd(dt, "misc/cgo/test", "go", "test", "-ldflags", "-linkmode=external")
		cmd.Env = env
		cmd = t.addCmd(dt, "misc/cgo/test", "go", "test", "-ldflags", "-linkmode=external -s")
		cmd.Env = env
	case "android-arm",
		"dragonfly-386", "dragonfly-amd64",
		"freebsd-386", "freebsd-amd64", "freebsd-arm",
		"linux-386", "linux-amd64", "linux-arm", "linux-ppc64le", "linux-s390x",
		"netbsd-386", "netbsd-amd64":

		cmd := t.addCmd(dt, "misc/cgo/test", "go", "test", "-ldflags", "-linkmode=external")
		cmd.Env = env

		cmd = t.addCmd(dt, "misc/cgo/testtls", "go", "test", "-ldflags", "-linkmode=auto")
		cmd.Env = env

		cmd = t.addCmd(dt, "misc/cgo/testtls", "go", "test", "-ldflags", "-linkmode=external")
		cmd.Env = env

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
			cc := mustEnv("CC")
			cmd := t.dirCmd("misc/cgo/test",
				cc, "-xc", "-o", "/dev/null", "-static", "-")
			cmd.Env = env
			cmd.Stdin = strings.NewReader("int main() {}")
			if err := cmd.Run(); err != nil {
				fmt.Println("No support for static linking found (lacks libc.a?), skip cgo static linking test.")
			} else {
				if t.goos != "android" {
					cmd = t.addCmd(dt, "misc/cgo/testtls", "go", "test", "-ldflags", `-linkmode=external -extldflags "-static -pthread"`)
					cmd.Env = env
				}

				cmd = t.addCmd(dt, "misc/cgo/nocgo", "go", "test")
				cmd.Env = env

				cmd = t.addCmd(dt, "misc/cgo/nocgo", "go", "test", "-ldflags", `-linkmode=external`)
				cmd.Env = env

				if t.goos != "android" {
					cmd = t.addCmd(dt, "misc/cgo/nocgo", "go", "test", "-ldflags", `-linkmode=external -extldflags "-static -pthread"`)
					cmd.Env = env
				}
			}

			if pair != "freebsd-amd64" { // clang -pie fails to link misc/cgo/test
				cmd := t.dirCmd("misc/cgo/test",
					cc, "-xc", "-o", "/dev/null", "-pie", "-")
				cmd.Env = env
				cmd.Stdin = strings.NewReader("int main() {}")
				if err := cmd.Run(); err != nil {
					fmt.Println("No support for -pie found, skip cgo PIE test.")
				} else {
					cmd = t.addCmd(dt, "misc/cgo/test", "go", "test", "-ldflags", `-linkmode=external -extldflags "-pie"`)
					cmd.Env = env

					cmd = t.addCmd(dt, "misc/cgo/testtls", "go", "test", "-ldflags", `-linkmode=external -extldflags "-pie"`)
					cmd.Env = env

					cmd = t.addCmd(dt, "misc/cgo/nocgo", "go", "test", "-ldflags", `-linkmode=external -extldflags "-pie"`)
					cmd.Env = env

				}
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
	worklist := t.worklist
	t.worklist = nil
	for _, w := range worklist {
		w.start = make(chan bool)
		w.end = make(chan bool)
		go func(w *work) {
			if !<-w.start {
				w.out = []byte(fmt.Sprintf("skipped due to earlier error\n"))
			} else {
				w.out, w.err = w.cmd.CombinedOutput()
			}
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
	if t.goos == "android" || t.iOS() {
		// No exec facility on Android or iOS.
		return false
	}
	if t.goarch == "ppc64" {
		// External linking not implemented on ppc64 (issue #8912).
		return false
	}
	if t.goarch == "mips64le" || t.goarch == "mips64" {
		// External linking not implemented on mips64.
		return false
	}
	return true
}

func (t *tester) cgoTestSO(dt *distTest, testpath string) error {
	t.runPending(dt)

	dir := filepath.Join(t.goroot, testpath)

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
	switch t.goos {
	case "darwin":
		ext = "dylib"
		args = append(args, "-undefined", "suppress", "-flat_namespace")
	case "windows":
		ext = "dll"
		args = append(args, "-DEXPORT_DLL")
	}
	sofname := "libcgosotest." + ext
	args = append(args, "-o", sofname, "cgoso_c.c")

	if err := t.dirCmd(dir, cc, args...).Run(); err != nil {
		return err
	}
	defer os.Remove(filepath.Join(dir, sofname))

	if err := t.dirCmd(dir, "go", "build", "-o", "main.exe", "main.go").Run(); err != nil {
		return err
	}
	defer os.Remove(filepath.Join(dir, "main.exe"))

	cmd := t.dirCmd(dir, "./main.exe")
	if t.goos != "windows" {
		s := "LD_LIBRARY_PATH"
		if t.goos == "darwin" {
			s = "DYLD_LIBRARY_PATH"
		}
		cmd.Env = mergeEnvLists([]string{s + "=."}, os.Environ())

		// On FreeBSD 64-bit architectures, the 32-bit linker looks for
		// different environment variables.
		if t.goos == "freebsd" && t.gohostarch == "386" {
			cmd.Env = mergeEnvLists([]string{"LD_32_LIBRARY_PATH=."}, cmd.Env)
		}
	}
	return cmd.Run()
}

func (t *tester) hasBash() bool {
	switch t.gohostos {
	case "windows", "plan9":
		return false
	}
	return true
}

func (t *tester) raceDetectorSupported() bool {
	switch t.gohostos {
	case "linux", "darwin", "freebsd", "windows":
		return t.cgoEnabled && t.goarch == "amd64" && t.gohostos == t.goos
	}
	return false
}

func (t *tester) runFlag(rx string) string {
	if t.compileOnly {
		return "-run=^$"
	}
	return "-run=" + rx
}

func (t *tester) raceTest(dt *distTest) error {
	t.addCmd(dt, "src", "go", "test", "-race", "-i", "runtime/race", "flag", "os/exec")
	t.addCmd(dt, "src", "go", "test", "-race", t.runFlag("Output"), "runtime/race")
	t.addCmd(dt, "src", "go", "test", "-race", "-short", t.runFlag("TestParse|TestEcho|TestStdinCloseRace"), "flag", "os/exec")
	// We don't want the following line, because it
	// slows down all.bash (by 10 seconds on my laptop).
	// The race builder should catch any error here, but doesn't.
	// TODO(iant): Figure out how to catch this.
	// t.addCmd(dt, "src", "go", "test", "-race", "-run=TestParallelTest", "cmd/go")
	if t.cgoEnabled {
		env := mergeEnvLists([]string{"GOTRACEBACK=2"}, os.Environ())
		cmd := t.addCmd(dt, "misc/cgo/test", "go", "test", "-race", "-short", t.runFlag(""))
		cmd.Env = env
	}
	if t.extLink() {
		// Test with external linking; see issue 9133.
		t.addCmd(dt, "src", "go", "test", "-race", "-short", "-ldflags=-linkmode=external", t.runFlag("TestParse|TestEcho|TestStdinCloseRace"), "flag", "os/exec")
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
		cmd.Env = mergeEnvLists([]string{"GOOS=" + t.gohostos, "GOARCH=" + t.gohostarch, "GOMAXPROCS="}, os.Environ())
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

// mergeEnvLists merges the two environment lists such that
// variables with the same name in "in" replace those in "out".
// out may be mutated.
func mergeEnvLists(in, out []string) []string {
NextVar:
	for _, inkv := range in {
		k := strings.SplitAfterN(inkv, "=", 2)[0]
		for i, outkv := range out {
			if strings.HasPrefix(outkv, k) {
				out[i] = inkv
				continue NextVar
			}
		}
		out = append(out, inkv)
	}
	return out
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
	pkgDir := filepath.Join(t.goroot, "src", pkg)
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
