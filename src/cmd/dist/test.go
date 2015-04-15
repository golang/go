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
	flag.StringVar(&t.banner, "banner", "##### ", "banner prefix; blank means no section banners")
	flag.StringVar(&t.runRxStr, "run", "", "run only those tests matching the regular expression; empty means to run all")
	xflagparse(0)
	t.run()
}

// tester executes cmdtest.
type tester struct {
	listMode  bool
	noRebuild bool
	runRxStr  string
	runRx     *regexp.Regexp
	banner    string // prefix, or "" for none

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

	t.timeoutScale = 1
	if t.goarch == "arm" || t.goos == "windows" {
		t.timeoutScale = 2
	}

	if t.runRxStr != "" {
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

	var lastHeading string
	for _, dt := range t.tests {
		if t.runRx != nil && !t.runRx.MatchString(dt.name) {
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
			log.Fatalf("Failed: %v", err)
		}
	}
	if t.partial {
		fmt.Println("\nALL TESTS PASSED (some were excluded)")
	} else {
		fmt.Println("\nALL TESTS PASSED")
	}
}

func (t *tester) timeout(sec int) string {
	return "-timeout=" + fmt.Sprint(time.Duration(sec)*time.Second*time.Duration(t.timeoutScale))
}

func (t *tester) registerTests() {
	// Register a separate logical test for each package in the standard library
	// but actually group them together at execution time to share the cost of
	// building packages shared between them.
	all, err := exec.Command("go", "list", "std", "cmd").Output()
	if err != nil {
		log.Fatalf("Error running go list std cmd: %v", err)
	}
	// ranGoTest and stdMatches are state closed over by the
	// stdlib testing func below. The tests are run sequentially,
	// so there's no need for locks.
	var (
		ranGoTest  bool
		stdMatches []string
	)
	for _, pkg := range strings.Fields(string(all)) {
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
				cmd := exec.Command("go", append([]string{
					"test",
					"-short",
					t.timeout(120),
					"-gcflags=" + os.Getenv("GO_GCFLAGS"),
				}, stdMatches...)...)
				cmd.Stdout = os.Stdout
				cmd.Stderr = os.Stderr
				return cmd.Run()
			},
		})
	}

	// Old hack for when Plan 9 on GCE was too slow.
	// We're keeping this until test sharding (Issue 10029) is finished, though.
	if os.Getenv("GOTESTONLY") == "std" {
		t.partial = true
		return
	}

	// Runtime CPU tests.
	for _, cpu := range []string{"1", "2", "4"} {
		cpu := cpu
		testName := "runtime:cpu" + cpu
		t.tests = append(t.tests, distTest{
			name:    testName,
			heading: "GOMAXPROCS=2 runtime -cpu=1,2,4",
			fn: func() error {
				cmd := t.dirCmd(".", "go", "test", "-short", t.timeout(300), "runtime", "-cpu="+cpu)
				// We set GOMAXPROCS=2 in addition to -cpu=1,2,4 in order to test runtime bootstrap code,
				// creation of first goroutines and first garbage collections in the parallel setting.
				cmd.Env = mergeEnvLists([]string{"GOMAXPROCS=2"}, os.Environ())
				return cmd.Run()
			},
		})
	}

	// sync tests
	t.tests = append(t.tests, distTest{
		name:    "sync_cpu",
		heading: "sync -cpu=10",
		fn: func() error {
			return t.dirCmd(".", "go", "test", "sync", "-short", t.timeout(120), "-cpu=10").Run()
		},
	})

	iOS := t.goos == "darwin" && (t.goarch == "arm" || t.goarch == "arm64")

	if t.cgoEnabled && t.goos != "android" && !iOS {
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
	if t.cgoEnabled && t.goos != "android" && !iOS {
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
		if t.gohostos == "windows" {
			t.tests = append(t.tests, distTest{
				name:    "testso",
				heading: "../misc/cgo/testso",
				fn:      t.cgoTestSOWindows,
			})
		} else if t.hasBash() && t.goos != "android" && !iOS {
			t.registerTest("testso", "../misc/cgo/testso", "./test.bash")
		}
		if t.extLink() && t.goos == "darwin" && t.goarch == "amd64" {
			// TODO(crawshaw): add darwin/arm{,64}
			t.registerTest("testcarchive", "../misc/cgo/testcarchive", "./test.bash")
		}
		if t.gohostos == "linux" && t.goarch == "amd64" {
			t.registerTest("testasan", "../misc/cgo/testasan", "go", "run", "main.go")
		}
		if t.hasBash() && t.goos != "android" && !iOS && t.gohostos != "windows" {
			t.registerTest("cgo_errors", "../misc/cgo/errors", "./test.bash")
		}
	}
	if t.hasBash() && t.goos != "nacl" && t.goos != "android" && !iOS {
		t.registerTest("doc_progs", "../doc/progs", "time", "go", "run", "run.go")
		t.registerTest("wiki", "../doc/articles/wiki", "./test.bash")
		t.registerTest("codewalk", "../doc/codewalk", "time", "./run")
		t.registerTest("shootout", "../test/bench/shootout", "time", "./timing.sh", "-test")
	}
	if t.goos != "android" && !iOS {
		t.registerTest("bench_go1", "../test/bench/go1", "go", "test")
	}
	if t.goos != "android" && !iOS {
		// TODO(bradfitz): shard down into these tests, as
		// this is one of the slowest (and most shardable)
		// tests.
		t.tests = append(t.tests, distTest{
			name:    "test",
			heading: "../test",
			fn:      t.testDirTest,
		})
	}
	if t.goos != "nacl" && t.goos != "android" && !iOS {
		t.tests = append(t.tests, distTest{
			name:    "api",
			heading: "API check",
			fn: func() error {
				return t.dirCmd(".", "go", "run", filepath.Join(t.goroot, "src/cmd/api/run.go")).Run()
			},
		})
	}

}

func (t *tester) registerTest(name, dirBanner, bin string, args ...string) {
	if bin == "time" && !t.haveTime {
		bin, args = args[0], args[1:]
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
	return cmd
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
		"dragonfly-386", "dragonfly-amd64",
		"freebsd-386", "freebsd-amd64", "freebsd-arm",
		"linux-386", "linux-amd64", "linux-arm",
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

func (t *tester) cgoTest() error {
	env := mergeEnvLists([]string{"GOTRACEBACK=2"}, os.Environ())

	iOS := t.goos == "darwin" && (t.goarch == "arm" || t.goarch == "arm64")
	if t.goos == "android" || iOS {
		cmd := t.dirCmd("misc/cgo/test", "go", "test")
		cmd.Env = env
		return cmd.Run()
	}

	cmd := t.dirCmd("misc/cgo/test", "go", "test", "-ldflags", "-linkmode=auto")
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
	case "openbsd-386", "openbsd-amd64":
		// test linkmode=external, but __thread not supported, so skip testtls.
		cmd := t.dirCmd("misc/cgo/test", "go", "test", "-ldflags", "-linkmode=external")
		cmd.Env = env
		if err := cmd.Run(); err != nil {
			return err
		}
	case "darwin-386", "darwin-amd64",
		"windows-386", "windows-amd64":
		if t.extLink() {
			cmd := t.dirCmd("misc/cgo/test", "go", "test", "-ldflags", "-linkmode=external")
			cmd.Env = env
			if err := cmd.Run(); err != nil {
				return err
			}
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

func (t *tester) cgoTestSOWindows() error {
	cmd := t.dirCmd("misc/cgo/testso", `.\test`)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	s := buf.String()
	fmt.Println(s)
	if err != nil {
		return err
	}
	if strings.Contains(s, "FAIL") {
		return errors.New("test failed")
	}
	return nil
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
	if err := t.dirCmd(".", "go", "test", "-race", "-i", "runtime/race", "flag", "os/exec").Run(); err != nil {
		return err
	}
	if err := t.dirCmd(".", "go", "test", "-race", "-run=Output", "runtime/race").Run(); err != nil {
		return err
	}
	if err := t.dirCmd(".", "go", "test", "-race", "-short", "flag", "os/exec").Run(); err != nil {
		return err
	}
	if t.extLink() {
		// Test with external linking; see issue 9133.
		if err := t.dirCmd(".", "go", "test", "-race", "-short", "-ldflags=-linkmode=external", "flag", "os/exec").Run(); err != nil {
			return err
		}
	}
	return nil
}

func (t *tester) testDirTest() error {
	const runExe = "runtest.exe" // named exe for Windows, but harmless elsewhere
	cmd := t.dirCmd("test", "go", "build", "-o", runExe, "run.go")
	cmd.Env = mergeEnvLists([]string{"GOOS=" + t.gohostos, "GOARCH=" + t.gohostarch, "GOMAXPROCS="}, os.Environ())
	if err := cmd.Run(); err != nil {
		return err
	}
	absExe := filepath.Join(cmd.Dir, runExe)
	defer os.Remove(absExe)
	if t.haveTime {
		return t.dirCmd("test", "time", absExe).Run()
	}
	return t.dirCmd("test", absExe).Run()
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
