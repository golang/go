// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

func cmdtest() {
	var t tester
	flag.BoolVar(&t.listMode, "list", false, "list available tests")
	flag.BoolVar(&t.noRebuild, "no-rebuild", false, "don't rebuild std and cmd packages")
	flag.BoolVar(&t.keepGoing, "k", false, "keep going even when error occurred")
	flag.BoolVar(&t.race, "race", false, "run in race builder mode (different set of tests)")
	flag.StringVar(&t.banner, "banner", "##### ", "banner prefix; blank means no section banners")
	flag.StringVar(&t.runRxStr, "run", os.Getenv("GOTESTONLY"),
		"run only those tests matching the regular expression; empty means to run all. "+
			"Special exception: if the string begins with '!', the match is inverted.")
	xflagparse(-1) // any number of args
	t.run()
}

// tester executes cmdtest.
type tester struct {
	race      bool
	listMode  bool
	noRebuild bool
	keepGoing bool
	runRxStr  string
	runRx     *regexp.Regexp
	runRxWant bool     // want runRx to match (true) or not match (false)
	runNames  []string // tests to run, exclusive with runRx; empty means all
	banner    string   // prefix, or "" for none

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
}

// A distTest is a test run by dist test.
// Each test has a unique name and belongs to a group (heading)
type distTest struct {
	name    string // unique test name; may be filtered with -run flag
	heading string // group section; this header is printed before the test is run.
	fn      func() error
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

	if !t.noRebuild {
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
	if t.goarch == "arm" || t.goos == "windows" {
		t.timeoutScale = 2
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

	var lastHeading string
	ok := true
	for _, dt := range t.tests {
		if !t.shouldRunTest(dt.name) {
			t.partial = true
			continue
		}
		if dt.heading != "" && lastHeading != dt.heading {
			lastHeading = dt.heading
			t.out(dt.heading)
		}
		if vflag > 0 {
			fmt.Printf("# go tool dist test -run=^%s$\n", dt.name)
		}
		if err := dt.fn(); err != nil {
			ok = false
			if t.keepGoing {
				log.Printf("Failed: %v", err)
			} else {
				log.Fatalf("Failed: %v", err)
			}
		}
	}
	if !ok {
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
		fn: func() error {
			if ranGoTest {
				return nil
			}
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
		fn: func() error {
			if ranGoBench {
				return nil
			}
			ranGoBench = true
			args := []string{
				"test",
				"-short",
				"-race",
				"-run=^$", // nothing. only benchmarks.
				"-bench=.*",
				"-benchtime=.1s",
				"-cpu=4",
			}
			args = append(args, benchMatches...)
			cmd := exec.Command("go", args...)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			return cmd.Run()
		},
	})
}

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
		cmd := exec.Command("go", "list", "-f", format, "std")
		if !t.race {
			cmd.Args = append(cmd.Args, "cmd")
		}
		all, err := cmd.CombinedOutput()
		if err != nil {
			log.Fatalf("Error running go list std cmd: %v, %s", err, all)
		}
		pkgs := strings.Fields(string(all))
		for _, pkg := range pkgs {
			t.registerStdTest(pkg)
		}
		if t.race {
			for _, pkg := range pkgs {
				t.registerRaceBenchTest(pkg)
			}
		}
	}

	if t.race {
		return
	}

	// Runtime CPU tests.
	testName := "runtime:cpu124"
	t.tests = append(t.tests, distTest{
		name:    testName,
		heading: "GOMAXPROCS=2 runtime -cpu=1,2,4",
		fn: func() error {
			cmd := t.dirCmd("src", "go", "test", "-short", t.timeout(300), t.tags(), "runtime", "-cpu=1,2,4")
			// We set GOMAXPROCS=2 in addition to -cpu=1,2,4 in order to test runtime bootstrap code,
			// creation of first goroutines and first garbage collections in the parallel setting.
			cmd.Env = mergeEnvLists([]string{"GOMAXPROCS=2"}, os.Environ())
			return cmd.Run()
		},
	})

	// sync tests
	t.tests = append(t.tests, distTest{
		name:    "sync_cpu",
		heading: "sync -cpu=10",
		fn: func() error {
			return t.dirCmd("src", "go", "test", "sync", "-short", t.timeout(120), t.tags(), "-cpu=10").Run()
		},
	})

	if t.cgoEnabled && t.goos != "android" && !t.iOS() {
		// Disabled on android and iOS. golang.org/issue/8345
		t.tests = append(t.tests, distTest{
			name:    "cgo_stdio",
			heading: "../misc/cgo/stdio",
			fn: func() error {
				return t.dirCmd("misc/cgo/stdio",
					"go", "run", filepath.Join(os.Getenv("GOROOT"), "test/run.go"), "-", ".").Run()
			},
		})
		t.tests = append(t.tests, distTest{
			name:    "cgo_life",
			heading: "../misc/cgo/life",
			fn: func() error {
				return t.dirCmd("misc/cgo/life",
					"go", "run", filepath.Join(os.Getenv("GOROOT"), "test/run.go"), "-", ".").Run()
			},
		})
	}
	if t.cgoEnabled && t.goos != "android" && !t.iOS() {
		// TODO(crawshaw): reenable on android and iOS
		// golang.org/issue/8345
		//
		// These tests are not designed to run off the host.
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
				fn: func() error {
					return t.cgoTestSO("misc/cgo/testso")
				},
			})
			t.tests = append(t.tests, distTest{
				name:    "testsovar",
				heading: "../misc/cgo/testsovar",
				fn: func() error {
					return t.cgoTestSO("misc/cgo/testsovar")
				},
			})
		}
		if t.supportedBuildmode("c-archive") {
			t.registerTest("testcarchive", "../misc/cgo/testcarchive", "./test.bash")
		}
		if t.supportedBuildmode("c-shared") {
			t.registerTest("testcshared", "../misc/cgo/testcshared", "./test.bash")
		}
		if t.supportedBuildmode("shared") {
			t.registerTest("testshared", "../misc/cgo/testshared", "go", "test")
		}
		if t.gohostos == "linux" && t.goarch == "amd64" {
			t.registerTest("testasan", "../misc/cgo/testasan", "go", "run", "main.go")
		}
		if t.hasBash() && t.goos != "android" && !t.iOS() && t.gohostos != "windows" {
			t.registerTest("cgo_errors", "../misc/cgo/errors", "./test.bash")
		}
		if t.gohostos == "linux" && t.extLink() {
			t.registerTest("testsigfwd", "../misc/cgo/testsigfwd", "go", "run", "main.go")
		}
	}
	if t.hasBash() && t.goos != "nacl" && t.goos != "android" && !t.iOS() {
		t.registerTest("doc_progs", "../doc/progs", "time", "go", "run", "run.go")
		t.registerTest("wiki", "../doc/articles/wiki", "./test.bash")
		t.registerTest("codewalk", "../doc/codewalk", "time", "./run")
		t.registerTest("shootout", "../test/bench/shootout", "time", "./timing.sh", "-test")
	}
	if t.goos != "android" && !t.iOS() {
		t.registerTest("bench_go1", "../test/bench/go1", "go", "test")
	}
	if t.goos != "android" && !t.iOS() {
		const nShards = 5
		for shard := 0; shard < nShards; shard++ {
			shard := shard
			t.tests = append(t.tests, distTest{
				name:    fmt.Sprintf("test:%d_%d", shard, nShards),
				heading: "../test",
				fn:      func() error { return t.testDirTest(shard, nShards) },
			})
		}
	}
	if t.goos != "nacl" && t.goos != "android" && !t.iOS() {
		t.tests = append(t.tests, distTest{
			name:    "api",
			heading: "API check",
			fn: func() error {
				return t.dirCmd("src", "go", "run", filepath.Join(t.goroot, "src/cmd/api/run.go")).Run()
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

func (t *tester) registerTest(name, dirBanner, bin string, args ...string) {
	if bin == "time" && !t.haveTime {
		bin, args = args[0], args[1:]
	}
	if t.isRegisteredTestName(name) {
		panic("duplicate registered test name " + name)
	}
	t.tests = append(t.tests, distTest{
		name:    name,
		heading: dirBanner,
		fn: func() error {
			return t.dirCmd(filepath.Join(t.goroot, "src", dirBanner), bin, args...).Run()
		},
	})
}

func (t *tester) dirCmd(dir string, bin string, args ...string) *exec.Cmd {
	cmd := exec.Command(bin, args...)
	if filepath.IsAbs(dir) {
		cmd.Dir = dir
	} else {
		cmd.Dir = filepath.Join(t.goroot, dir)
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if vflag > 1 {
		errprintf("%s\n", strings.Join(cmd.Args, " "))
	}
	return cmd
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
		"linux-386", "linux-amd64", "linux-arm", "linux-arm64",
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

func (t *tester) supportedBuildmode(mode string) bool {
	pair := t.goos + "-" + t.goarch
	switch mode {
	case "c-archive":
		if !t.extLink() {
			return false
		}
		switch pair {
		case "darwin-amd64", "darwin-arm", "darwin-arm64",
			"linux-amd64", "linux-386":
			return true
		}
		return false
	case "c-shared":
		// TODO(hyangah): add linux-386.
		switch pair {
		case "linux-amd64", "darwin-amd64", "android-arm":
			return true
		}
		return false
	case "shared":
		switch pair {
		case "linux-amd64":
			return true
		}
		return false
	default:
		log.Fatal("internal error: unknown buildmode %s", mode)
		return false
	}
}

func (t *tester) cgoTest() error {
	env := mergeEnvLists([]string{"GOTRACEBACK=2"}, os.Environ())

	if t.goos == "android" || t.iOS() {
		cmd := t.dirCmd("misc/cgo/test", "go", "test", t.tags())
		cmd.Env = env
		return cmd.Run()
	}

	cmd := t.dirCmd("misc/cgo/test", "go", "test", t.tags(), "-ldflags", "-linkmode=auto")
	cmd.Env = env
	if err := cmd.Run(); err != nil {
		return err
	}

	if t.gohostos != "dragonfly" {
		// linkmode=internal fails on dragonfly since errno is a TLS relocation.
		cmd := t.dirCmd("misc/cgo/test", "go", "test", "-ldflags", "-linkmode=internal")
		cmd.Env = env
		if err := cmd.Run(); err != nil {
			return err
		}
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
		cmd := t.dirCmd("misc/cgo/test", "go", "test", "-ldflags", "-linkmode=external")
		cmd.Env = env
		if err := cmd.Run(); err != nil {
			return err
		}
		cmd = t.dirCmd("misc/cgo/test", "go", "test", "-ldflags", "-linkmode=external -s")
		cmd.Env = env
		if err := cmd.Run(); err != nil {
			return err
		}
	case "android-arm",
		"dragonfly-386", "dragonfly-amd64",
		"freebsd-386", "freebsd-amd64", "freebsd-arm",
		"linux-386", "linux-amd64", "linux-arm",
		"netbsd-386", "netbsd-amd64":

		cmd := t.dirCmd("misc/cgo/test", "go", "test", "-ldflags", "-linkmode=external")
		cmd.Env = env
		if err := cmd.Run(); err != nil {
			return err
		}
		cmd = t.dirCmd("misc/cgo/testtls", "go", "test", "-ldflags", "-linkmode=auto")
		cmd.Env = env
		if err := cmd.Run(); err != nil {
			return err
		}
		cmd = t.dirCmd("misc/cgo/testtls", "go", "test", "-ldflags", "-linkmode=external")
		cmd.Env = env
		if err := cmd.Run(); err != nil {
			return err
		}

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
				cmd = t.dirCmd("misc/cgo/testtls", "go", "test", "-ldflags", `-linkmode=external -extldflags "-static -pthread"`)
				cmd.Env = env
				if err := cmd.Run(); err != nil {
					return err
				}

				cmd = t.dirCmd("misc/cgo/nocgo", "go", "test")
				cmd.Env = env
				if err := cmd.Run(); err != nil {
					return err
				}

				cmd = t.dirCmd("misc/cgo/nocgo", "go", "test", "-ldflags", `-linkmode=external`)
				cmd.Env = env
				if err := cmd.Run(); err != nil {
					return err
				}

				cmd = t.dirCmd("misc/cgo/nocgo", "go", "test", "-ldflags", `-linkmode=external -extldflags "-static -pthread"`)
				cmd.Env = env
				if err := cmd.Run(); err != nil {
					return err
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
					cmd = t.dirCmd("misc/cgo/test", "go", "test", "-ldflags", `-linkmode=external -extldflags "-pie"`)
					cmd.Env = env
					if err := cmd.Run(); err != nil {
						return fmt.Errorf("pie cgo/test: %v", err)
					}
					cmd = t.dirCmd("misc/cgo/testtls", "go", "test", "-ldflags", `-linkmode=external -extldflags "-pie"`)
					cmd.Env = env
					if err := cmd.Run(); err != nil {
						return fmt.Errorf("pie cgo/testtls: %v", err)
					}
					cmd = t.dirCmd("misc/cgo/nocgo", "go", "test", "-ldflags", `-linkmode=external -extldflags "-pie"`)
					cmd.Env = env
					if err := cmd.Run(); err != nil {
						return fmt.Errorf("pie cgo/nocgo: %v", err)
					}
				}
			}
		}
	}

	return nil
}

func (t *tester) cgoTestSOSupported() bool {
	if t.goos == "android" || t.iOS() {
		// No exec facility on Android or iOS.
		return false
	}
	if t.goarch == "ppc64le" || t.goarch == "ppc64" {
		// External linking not implemented on ppc64 (issue #8912).
		return false
	}
	return true
}

func (t *tester) cgoTestSO(testpath string) error {
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

func (t *tester) raceTest() error {
	if err := t.dirCmd("src", "go", "test", "-race", "-i", "runtime/race", "flag", "os/exec").Run(); err != nil {
		return err
	}
	if err := t.dirCmd("src", "go", "test", "-race", "-run=Output", "runtime/race").Run(); err != nil {
		return err
	}
	if err := t.dirCmd("src", "go", "test", "-race", "-short", "flag", "os/exec").Run(); err != nil {
		return err
	}
	if t.cgoEnabled {
		env := mergeEnvLists([]string{"GOTRACEBACK=2"}, os.Environ())
		cmd := t.dirCmd("misc/cgo/test", "go", "test", "-race", "-short")
		cmd.Env = env
		if err := cmd.Run(); err != nil {
			return err
		}
	}
	if t.extLink() {
		// Test with external linking; see issue 9133.
		if err := t.dirCmd("src", "go", "test", "-race", "-short", "-ldflags=-linkmode=external", "flag", "os/exec").Run(); err != nil {
			return err
		}
	}
	return nil
}

func (t *tester) testDirTest(shard, shards int) error {
	const runExe = "runtest.exe" // named exe for Windows, but harmless elsewhere
	cmd := t.dirCmd("test", "go", "build", "-o", runExe, "run.go")
	cmd.Env = mergeEnvLists([]string{"GOOS=" + t.gohostos, "GOARCH=" + t.gohostarch, "GOMAXPROCS="}, os.Environ())
	if err := cmd.Run(); err != nil {
		return err
	}
	absExe := filepath.Join(cmd.Dir, runExe)
	defer os.Remove(absExe)
	return t.dirCmd("test", absExe,
		fmt.Sprintf("--shard=%d", shard),
		fmt.Sprintf("--shards=%d", shards),
	).Run()
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
