// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
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
	flag.BoolVar(&t.compileOnly, "compile-only", false, "compile tests, but don't run them")
	flag.StringVar(&t.banner, "banner", "##### ", "banner prefix; blank means no section banners")
	flag.StringVar(&t.runRxStr, "run", "",
		"run only those tests matching the regular expression; empty means to run all. "+
			"Special exception: if the string begins with '!', the match is inverted.")
	flag.BoolVar(&t.msan, "msan", false, "run in memory sanitizer builder mode")
	flag.BoolVar(&t.asan, "asan", false, "run in address sanitizer builder mode")
	flag.BoolVar(&t.json, "json", false, "report test results in JSON")

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

	short      bool
	cgoEnabled bool
	json       bool

	tests        []distTest // use addTest to extend
	testNames    map[string]bool
	timeoutScale int

	worklist []*work
}

// work tracks command execution for a test.
type work struct {
	dt    *distTest     // unique test name, etc.
	cmd   *exec.Cmd     // must write stdout/stderr to out
	flush func()        // if non-nil, called after cmd.Run
	start chan bool     // a true means to start, a false means to skip
	out   bytes.Buffer  // combined stdout/stderr from cmd
	err   error         // work result
	end   chan struct{} // a value means cmd ended (or was skipped)
}

// printSkip prints a skip message for all of work.
func (w *work) printSkip(t *tester, msg string) {
	if t.json {
		synthesizeSkipEvent(json.NewEncoder(&w.out), w.dt.name, msg)
		return
	}
	fmt.Fprintln(&w.out, msg)
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

	t.short = true
	if v := os.Getenv("GO_TEST_SHORT"); v != "" {
		short, err := strconv.ParseBool(v)
		if err != nil {
			fatalf("invalid GO_TEST_SHORT %q: %v", v, err)
		}
		t.short = short
	}

	cmd := exec.Command(gorootBinGo, "env", "CGO_ENABLED")
	cmd.Stderr = new(bytes.Buffer)
	slurp, err := cmd.Output()
	if err != nil {
		fatalf("Error running %s: %v\n%s", cmd, err, cmd.Stderr)
	}
	parts := strings.Split(string(slurp), "\n")
	if nlines := len(parts) - 1; nlines < 1 {
		fatalf("Error running %s: output contains <1 lines\n%s", cmd, cmd.Stderr)
	}
	t.cgoEnabled, _ = strconv.ParseBool(parts[0])

	if flag.NArg() > 0 && t.runRxStr != "" {
		fatalf("the -run regular expression flag is mutually exclusive with test name arguments")
	}

	t.runNames = flag.Args()

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
		goInstall(toolenv(), gorootBinGo, append([]string{"-a"}, toolchain...)...)
	}

	if !t.listMode {
		if builder := os.Getenv("GO_BUILDER_NAME"); builder == "" {
			// Ensure that installed commands are up to date, even with -no-rebuild,
			// so that tests that run commands end up testing what's actually on disk.
			// If everything is up-to-date, this is a no-op.
			// We first build the toolchain twice to allow it to converge,
			// as when we first bootstrap.
			// See cmdbootstrap for a description of the overall process.
			//
			// On the builders, we skip this step: we assume that 'dist test' is
			// already using the result of a clean build, and because of test sharding
			// and virtualization we usually start with a clean GOCACHE, so we would
			// end up rebuilding large parts of the standard library that aren't
			// otherwise relevant to the actual set of packages under test.
			goInstall(toolenv(), gorootBinGo, toolchain...)
			goInstall(toolenv(), gorootBinGo, toolchain...)
			goInstall(toolenv(), gorootBinGo, toolsToInstall...)
		}
	}

	t.timeoutScale = 1
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
		if !t.testNames[name] {
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

	if !t.json {
		if err := t.maybeLogMetadata(); err != nil {
			t.failed = true
			if t.keepGoing {
				log.Printf("Failed logging metadata: %v", err)
			} else {
				fatalf("Failed logging metadata: %v", err)
			}
		}
	}

	var anyIncluded, someExcluded bool
	for _, dt := range t.tests {
		if !t.shouldRunTest(dt.name) {
			someExcluded = true
			continue
		}
		anyIncluded = true
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

	if !t.json {
		if t.failed {
			fmt.Println("\nFAILED")
		} else if !anyIncluded {
			fmt.Println()
			errprintf("go tool dist: warning: %q matched no tests; use the -list flag to list available tests\n", t.runRxStr)
			fmt.Println("NO TESTS TO RUN")
		} else if someExcluded {
			fmt.Println("\nALL TESTS PASSED (some were excluded)")
		} else {
			fmt.Println("\nALL TESTS PASSED")
		}
	}
	if t.failed {
		xexit(1)
	}
}

func (t *tester) shouldRunTest(name string) bool {
	if t.runRx != nil {
		return t.runRx.MatchString(name) == t.runRxWant
	}
	if len(t.runNames) == 0 {
		return true
	}
	return slices.Contains(t.runNames, name)
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
	return t.dirCmd(filepath.Join(goroot, "src/cmd/internal/metadata"), gorootBinGo, []string{"run", "main.go"}).Run()
}

// testName returns the dist test name for a given package and variant.
func testName(pkg, variant string) string {
	name := pkg
	if variant != "" {
		name += ":" + variant
	}
	return name
}

// goTest represents all options to a "go test" command. The final command will
// combine configuration from goTest and tester flags.
type goTest struct {
	timeout  time.Duration // If non-zero, override timeout
	short    bool          // If true, force -short
	tags     []string      // Build tags
	race     bool          // Force -race
	bench    bool          // Run benchmarks (briefly), not tests.
	runTests string        // Regexp of tests to run
	cpu      string        // If non-empty, -cpu flag
	skip     string        // If non-empty, -skip flag

	gcflags   string // If non-empty, build with -gcflags=all=X
	ldflags   string // If non-empty, build with -ldflags=X
	buildmode string // If non-empty, -buildmode flag

	env []string // Environment variables to add, as KEY=VAL. KEY= unsets a variable

	runOnHost bool // When cross-compiling, run this test on the host instead of guest

	// variant, if non-empty, is a name used to distinguish different
	// configurations of the same test package(s). If set and omitVariant is false,
	// the Package field in test2json output is rewritten to pkg:variant.
	variant string
	// omitVariant indicates that variant is used solely for the dist test name and
	// that the set of test names run by each variant (including empty) of a package
	// is non-overlapping.
	omitVariant bool

	// We have both pkg and pkgs as a convenience. Both may be set, in which
	// case they will be combined. At least one must be set.
	pkgs []string // Multiple packages to test
	pkg  string   // A single package to test

	testFlags []string // Additional flags accepted by this test
}

// compileOnly reports whether this test is only for compiling,
// indicated by runTests being set to '^$' and bench being false.
func (opts *goTest) compileOnly() bool {
	return opts.runTests == "^$" && !opts.bench
}

// bgCommand returns a go test Cmd and a post-Run flush function. The result
// will write its output to stdout and stderr. If stdout==stderr, bgCommand
// ensures Writes are serialized. The caller should call flush() after Cmd exits.
func (opts *goTest) bgCommand(t *tester, stdout, stderr io.Writer) (cmd *exec.Cmd, flush func()) {
	build, run, pkgs, testFlags, setupCmd := opts.buildArgs(t)

	// Combine the flags.
	args := append([]string{"test"}, build...)
	if t.compileOnly || opts.compileOnly() {
		args = append(args, "-c", "-o", os.DevNull)
	} else {
		args = append(args, run...)
	}
	args = append(args, pkgs...)
	if !t.compileOnly && !opts.compileOnly() {
		args = append(args, testFlags...)
	}

	cmd = exec.Command(gorootBinGo, args...)
	setupCmd(cmd)
	if t.json && opts.variant != "" && !opts.omitVariant {
		// Rewrite Package in the JSON output to be pkg:variant. When omitVariant
		// is true, pkg.TestName is already unambiguous, so we don't need to
		// rewrite the Package field.
		//
		// We only want to process JSON on the child's stdout. Ideally if
		// stdout==stderr, we would also use the same testJSONFilter for
		// cmd.Stdout and cmd.Stderr in order to keep the underlying
		// interleaving of writes, but then it would see even partial writes
		// interleaved, which would corrupt the JSON. So, we only process
		// cmd.Stdout. This has another consequence though: if stdout==stderr,
		// we have to serialize Writes in case the Writer is not concurrent
		// safe. If we were just passing stdout/stderr through to exec, it would
		// do this for us, but since we're wrapping stdout, we have to do it
		// ourselves.
		if stdout == stderr {
			stdout = &lockedWriter{w: stdout}
			stderr = stdout
		}
		f := &testJSONFilter{w: stdout, variant: opts.variant}
		cmd.Stdout = f
		flush = f.Flush
	} else {
		cmd.Stdout = stdout
		flush = func() {}
	}
	cmd.Stderr = stderr

	return cmd, flush
}

// run runs a go test and returns an error if it does not succeed.
func (opts *goTest) run(t *tester) error {
	cmd, flush := opts.bgCommand(t, os.Stdout, os.Stderr)
	err := cmd.Run()
	flush()
	return err
}

// buildArgs is in internal helper for goTest that constructs the elements of
// the "go test" command line. build is the flags for building the test. run is
// the flags for running the test. pkgs is the list of packages to build and
// run. testFlags is the list of flags to pass to the test package.
//
// The caller must call setupCmd on the resulting exec.Cmd to set its directory
// and environment.
func (opts *goTest) buildArgs(t *tester) (build, run, pkgs, testFlags []string, setupCmd func(*exec.Cmd)) {
	run = append(run, "-count=1") // Disallow caching
	if opts.timeout != 0 {
		d := opts.timeout * time.Duration(t.timeoutScale)
		run = append(run, "-timeout="+d.String())
	} else if t.timeoutScale != 1 {
		const goTestDefaultTimeout = 10 * time.Minute // Default value of go test -timeout flag.
		run = append(run, "-timeout="+(goTestDefaultTimeout*time.Duration(t.timeoutScale)).String())
	}
	if opts.short || t.short {
		run = append(run, "-short")
	}
	var tags []string
	if t.iOS() {
		tags = append(tags, "lldb")
	}
	if noOpt {
		tags = append(tags, "noopt")
	}
	tags = append(tags, opts.tags...)
	if len(tags) > 0 {
		build = append(build, "-tags="+strings.Join(tags, ","))
	}
	if t.race || opts.race {
		build = append(build, "-race")
	}
	if t.msan {
		build = append(build, "-msan")
	}
	if t.asan {
		build = append(build, "-asan")
	}
	if opts.bench {
		// Run no tests.
		run = append(run, "-run=^$")
		// Run benchmarks briefly as a smoke test.
		run = append(run, "-bench=.*", "-benchtime=.1s")
	} else if opts.runTests != "" {
		run = append(run, "-run="+opts.runTests)
	}
	if opts.cpu != "" {
		run = append(run, "-cpu="+opts.cpu)
	}
	if opts.skip != "" {
		run = append(run, "-skip="+opts.skip)
	}
	if t.json {
		run = append(run, "-json")
	}

	if opts.gcflags != "" {
		build = append(build, "-gcflags=all="+opts.gcflags)
	}
	if opts.ldflags != "" {
		build = append(build, "-ldflags="+opts.ldflags)
	}
	if opts.buildmode != "" {
		build = append(build, "-buildmode="+opts.buildmode)
	}

	pkgs = opts.packages()

	runOnHost := opts.runOnHost && (goarch != gohostarch || goos != gohostos)
	needTestFlags := len(opts.testFlags) > 0 || runOnHost
	if needTestFlags {
		testFlags = append([]string{"-args"}, opts.testFlags...)
	}
	if runOnHost {
		// -target is a special flag understood by tests that can run on the host
		testFlags = append(testFlags, "-target="+goos+"/"+goarch)
	}

	setupCmd = func(cmd *exec.Cmd) {
		setDir(cmd, filepath.Join(goroot, "src"))
		if len(opts.env) != 0 {
			for _, kv := range opts.env {
				if i := strings.Index(kv, "="); i < 0 {
					unsetEnv(cmd, kv[:len(kv)-1])
				} else {
					setEnv(cmd, kv[:i], kv[i+1:])
				}
			}
		}
		if runOnHost {
			setEnv(cmd, "GOARCH", gohostarch)
			setEnv(cmd, "GOOS", gohostos)
		}
	}

	return
}

// packages returns the full list of packages to be run by this goTest. This
// will always include at least one package.
func (opts *goTest) packages() []string {
	pkgs := opts.pkgs
	if opts.pkg != "" {
		pkgs = append(pkgs[:len(pkgs):len(pkgs)], opts.pkg)
	}
	if len(pkgs) == 0 {
		panic("no packages")
	}
	return pkgs
}

// printSkip prints a skip message for all of goTest.
func (opts *goTest) printSkip(t *tester, msg string) {
	if t.json {
		enc := json.NewEncoder(os.Stdout)
		for _, pkg := range opts.packages() {
			synthesizeSkipEvent(enc, pkg, msg)
		}
		return
	}
	fmt.Println(msg)
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
	const stdTestHeading = "Testing packages." // known to addTest for a safety check
	gcflags := gogcflags
	name := testName(pkg, "")
	if t.runRx == nil || t.runRx.MatchString(name) == t.runRxWant {
		stdMatches = append(stdMatches, pkg)
	}
	t.addTest(name, stdTestHeading, func(dt *distTest) error {
		if ranGoTest {
			return nil
		}
		t.runPending(dt)
		timelog("start", dt.name)
		defer timelog("end", dt.name)
		ranGoTest = true

		timeoutSec := 180 * time.Second
		for _, pkg := range stdMatches {
			if pkg == "cmd/go" {
				timeoutSec *= 3
				break
			}
		}
		return (&goTest{
			timeout: timeoutSec,
			gcflags: gcflags,
			pkgs:    stdMatches,
		}).run(t)
	})
}

func (t *tester) registerRaceBenchTest(pkg string) {
	const raceBenchHeading = "Running benchmarks briefly." // known to addTest for a safety check
	name := testName(pkg, "racebench")
	if t.runRx == nil || t.runRx.MatchString(name) == t.runRxWant {
		benchMatches = append(benchMatches, pkg)
	}
	t.addTest(name, raceBenchHeading, func(dt *distTest) error {
		if ranGoBench {
			return nil
		}
		t.runPending(dt)
		timelog("start", dt.name)
		defer timelog("end", dt.name)
		ranGoBench = true
		return (&goTest{
			variant:     "racebench",
			omitVariant: true,               // The only execution of benchmarks in dist; benchmark names are guaranteed not to overlap with test names.
			timeout:     1200 * time.Second, // longer timeout for race with benchmarks
			race:        true,
			bench:       true,
			cpu:         "4",
			pkgs:        benchMatches,
		}).run(t)
	})
}

func (t *tester) registerTests() {
	// registerStdTestSpecially tracks import paths in the standard library
	// whose test registration happens in a special way.
	//
	// These tests *must* be able to run normally as part of "go test std cmd",
	// even if they are also registered separately by dist, because users often
	// run go test directly. Use skips or build tags in preference to expanding
	// this list.
	registerStdTestSpecially := map[string]bool{
		// testdir can run normally as part of "go test std cmd", but because
		// it's a very large test, we register is specially as several shards to
		// enable better load balancing on sharded builders. Ideally the build
		// system would know how to shard any large test package.
		"cmd/internal/testdir": true,
	}

	// Fast path to avoid the ~1 second of `go list std cmd` when
	// the caller lists specific tests to run. (as the continuous
	// build coordinator does).
	if len(t.runNames) > 0 {
		for _, name := range t.runNames {
			if !strings.Contains(name, ":") {
				t.registerStdTest(name)
			} else if strings.HasSuffix(name, ":racebench") {
				t.registerRaceBenchTest(strings.TrimSuffix(name, ":racebench"))
			}
		}
	} else {
		// Use 'go list std cmd' to get a list of all Go packages
		// that running 'go test std cmd' could find problems in.
		// (In race test mode, also set -tags=race.)
		// This includes vendored packages and other
		// packages without tests so that 'dist test' finds if any of
		// them don't build, have a problem reported by high-confidence
		// vet checks that come with 'go test', and anything else it
		// may check in the future. See go.dev/issue/60463.
		// Most packages have tests, so there is not much saved
		// by skipping non-test packages.
		// For the packages without any test files,
		// 'go test' knows not to actually build a test binary,
		// so the only cost is the vet, and we still want to run vet.
		cmd := exec.Command(gorootBinGo, "list")
		if t.race {
			cmd.Args = append(cmd.Args, "-tags=race")
		}
		cmd.Args = append(cmd.Args, "std", "cmd")
		cmd.Stderr = new(bytes.Buffer)
		all, err := cmd.Output()
		if err != nil {
			fatalf("Error running go list std cmd: %v:\n%s", err, cmd.Stderr)
		}
		pkgs := strings.Fields(string(all))
		for _, pkg := range pkgs {
			if registerStdTestSpecially[pkg] {
				continue
			}
			if t.short && (strings.HasPrefix(pkg, "vendor/") || strings.HasPrefix(pkg, "cmd/vendor/")) {
				// Vendored code has no tests, and we don't care too much about vet errors
				// since we can't modify the code, so skip the tests in short mode.
				// We still let the longtest builders vet them.
				continue
			}
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

	// Test the os/user package in the pure-Go mode too.
	if !t.compileOnly {
		t.registerTest("os/user with tag osusergo",
			&goTest{
				variant: "osusergo",
				timeout: 300 * time.Second,
				tags:    []string{"osusergo"},
				pkg:     "os/user",
			})
		t.registerTest("hash/maphash purego implementation",
			&goTest{
				variant: "purego",
				timeout: 300 * time.Second,
				tags:    []string{"purego"},
				pkg:     "hash/maphash",
			})
	}

	// Check that all crypto packages compile with the purego build tag.
	t.registerTest("crypto with tag purego (build and vet only)", &goTest{
		variant:  "purego",
		tags:     []string{"purego"},
		pkg:      "crypto/...",
		runTests: "^$", // only ensure they compile
	})

	// Check that all crypto packages compile (and test correctly, in longmode) with fips.
	if t.fipsSupported() {
		// Test standard crypto packages with fips140=on.
		t.registerTest("GOFIPS140=latest go test crypto/...", &goTest{
			variant: "gofips140",
			env:     []string{"GOFIPS140=latest"},
			pkg:     "crypto/...",
		})

		// Test that earlier FIPS snapshots build.
		// In long mode, test that they work too.
		for _, version := range fipsVersions(t.short) {
			suffix := " # (build and vet only)"
			run := "^$" // only ensure they compile
			if !t.short {
				suffix = ""
				run = ""
			}
			t.registerTest("GOFIPS140="+version+" go test crypto/..."+suffix, &goTest{
				variant:  "gofips140-" + version,
				pkg:      "crypto/...",
				runTests: run,
				env:      []string{"GOFIPS140=" + version, "GOMODCACHE=" + filepath.Join(workdir, "fips-"+version)},
			})
		}
	}

	// Test ios/amd64 for the iOS simulator.
	if goos == "darwin" && goarch == "amd64" && t.cgoEnabled {
		t.registerTest("GOOS=ios on darwin/amd64",
			&goTest{
				variant:  "amd64ios",
				timeout:  300 * time.Second,
				runTests: "SystemRoots",
				env:      []string{"GOOS=ios", "CGO_ENABLED=1"},
				pkg:      "crypto/x509",
			})
	}

	// GC debug mode tests. We only run these in long-test mode
	// (with GO_TEST_SHORT=0) because this is just testing a
	// non-critical debug setting.
	if !t.compileOnly && !t.short {
		t.registerTest("GODEBUG=gcstoptheworld=2 archive/zip",
			&goTest{
				variant: "runtime:gcstoptheworld2",
				timeout: 300 * time.Second,
				short:   true,
				env:     []string{"GODEBUG=gcstoptheworld=2"},
				pkg:     "archive/zip",
			})
		t.registerTest("GODEBUG=gccheckmark=1 runtime",
			&goTest{
				variant: "runtime:gccheckmark",
				timeout: 300 * time.Second,
				short:   true,
				env:     []string{"GODEBUG=gccheckmark=1"},
				pkg:     "runtime",
			})
	}

	// morestack tests. We only run these in long-test mode
	// (with GO_TEST_SHORT=0) because the runtime test is
	// already quite long and mayMoreStackMove makes it about
	// twice as slow.
	if !t.compileOnly && !t.short {
		// hooks is the set of maymorestack hooks to test with.
		hooks := []string{"mayMoreStackPreempt", "mayMoreStackMove"}
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

			t.registerTest("maymorestack="+hook,
				&goTest{
					variant: hook,
					timeout: 600 * time.Second,
					short:   true,
					env:     []string{"GOFLAGS=" + goFlags},
					pkgs:    []string{"runtime", "reflect", "sync"},
				})
		}
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

		// What matters is that the tests build and start up.
		// Skip expensive tests, especially x509 TestSystemRoots.
		run := "^Test[^CS]"
		if pkg == "net" {
			run = "TestTCPStress"
		}
		t.registerTest("Testing without libgcc.",
			&goTest{
				variant:  "nolibgcc",
				ldflags:  "-linkmode=internal -libgcc=none",
				runTests: run,
				pkg:      pkg,
			})
	}

	// Stub out following test on alpine until 54354 resolved.
	builderName := os.Getenv("GO_BUILDER_NAME")
	disablePIE := strings.HasSuffix(builderName, "-alpine")

	// Test internal linking of PIE binaries where it is supported.
	if t.internalLinkPIE() && !disablePIE {
		t.registerTest("internal linking, -buildmode=pie",
			&goTest{
				variant:   "pie_internal",
				timeout:   60 * time.Second,
				buildmode: "pie",
				ldflags:   "-linkmode=internal",
				env:       []string{"CGO_ENABLED=0"},
				pkg:       "reflect",
			})
		t.registerTest("internal linking, -buildmode=pie",
			&goTest{
				variant:   "pie_internal",
				timeout:   60 * time.Second,
				buildmode: "pie",
				ldflags:   "-linkmode=internal",
				env:       []string{"CGO_ENABLED=0"},
				pkg:       "crypto/internal/fips140test",
				runTests:  "TestFIPSCheck",
			})
		// Also test a cgo package.
		if t.cgoEnabled && t.internalLink() && !disablePIE {
			t.registerTest("internal linking, -buildmode=pie",
				&goTest{
					variant:   "pie_internal",
					timeout:   60 * time.Second,
					buildmode: "pie",
					ldflags:   "-linkmode=internal",
					pkg:       "os/user",
				})
		}
	}

	if t.extLink() && !t.compileOnly {
		if goos != "android" { // Android does not support non-PIE linking
			t.registerTest("external linking, -buildmode=exe",
				&goTest{
					variant:   "exe_external",
					timeout:   60 * time.Second,
					buildmode: "exe",
					ldflags:   "-linkmode=external",
					env:       []string{"CGO_ENABLED=1"},
					pkg:       "crypto/internal/fips140test",
					runTests:  "TestFIPSCheck",
				})
		}
		if t.externalLinkPIE() && !disablePIE {
			t.registerTest("external linking, -buildmode=pie",
				&goTest{
					variant:   "pie_external",
					timeout:   60 * time.Second,
					buildmode: "pie",
					ldflags:   "-linkmode=external",
					env:       []string{"CGO_ENABLED=1"},
					pkg:       "crypto/internal/fips140test",
					runTests:  "TestFIPSCheck",
				})
		}
	}

	// sync tests
	if t.hasParallelism() {
		t.registerTest("sync -cpu=10",
			&goTest{
				variant: "cpu10",
				timeout: 120 * time.Second,
				cpu:     "10",
				pkg:     "sync",
			})
	}

	const cgoHeading = "Testing cgo"
	if t.cgoEnabled {
		t.registerCgoTests(cgoHeading)
	}

	if goos == "wasip1" {
		t.registerTest("wasip1 host tests",
			&goTest{
				variant:   "host",
				pkg:       "internal/runtime/wasitest",
				timeout:   1 * time.Minute,
				runOnHost: true,
			})
	}

	// Only run the API check on fast development platforms.
	// Every platform checks the API on every GOOS/GOARCH/CGO_ENABLED combination anyway,
	// so we really only need to run this check once anywhere to get adequate coverage.
	// To help developers avoid trybot-only failures, we try to run on typical developer machines
	// which is darwin,linux,windows/amd64 and darwin/arm64.
	//
	// The same logic applies to the release notes that correspond to each api/next file.
	if goos == "darwin" || ((goos == "linux" || goos == "windows") && goarch == "amd64") {
		t.registerTest("API release note check", &goTest{variant: "check", pkg: "cmd/relnote", testFlags: []string{"-check"}})
		t.registerTest("API check", &goTest{variant: "check", pkg: "cmd/api", timeout: 5 * time.Minute, testFlags: []string{"-check"}})
	}

	// Runtime CPU tests.
	if !t.compileOnly && t.hasParallelism() {
		for i := 1; i <= 4; i *= 2 {
			t.registerTest(fmt.Sprintf("GOMAXPROCS=2 runtime -cpu=%d -quick", i),
				&goTest{
					variant:   "cpu" + strconv.Itoa(i),
					timeout:   300 * time.Second,
					cpu:       strconv.Itoa(i),
					gcflags:   gogcflags,
					short:     true,
					testFlags: []string{"-quick"},
					// We set GOMAXPROCS=2 in addition to -cpu=1,2,4 in order to test runtime bootstrap code,
					// creation of first goroutines and first garbage collections in the parallel setting.
					env: []string{"GOMAXPROCS=2"},
					pkg: "runtime",
				})
		}
	}

	if t.raceDetectorSupported() && !t.msan && !t.asan {
		// N.B. -race is incompatible with -msan and -asan.
		t.registerRaceTests()
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
			id := fmt.Sprintf("%d_%d", shard, nShards)
			t.registerTest("../test",
				&goTest{
					variant:     id,
					omitVariant: true, // Shards of the same Go package; tests are guaranteed not to overlap.
					pkg:         "cmd/internal/testdir",
					testFlags:   []string{fmt.Sprintf("-shard=%d", shard), fmt.Sprintf("-shards=%d", nShards)},
					runOnHost:   true,
				},
			)
		}
	}
}

// addTest adds an arbitrary test callback to the test list.
//
// name must uniquely identify the test and heading must be non-empty.
func (t *tester) addTest(name, heading string, fn func(*distTest) error) {
	if t.testNames[name] {
		panic("duplicate registered test name " + name)
	}
	if heading == "" {
		panic("empty heading")
	}
	// Two simple checks for cases that would conflict with the fast path in registerTests.
	if !strings.Contains(name, ":") && heading != "Testing packages." {
		panic("empty variant is reserved exclusively for registerStdTest")
	} else if strings.HasSuffix(name, ":racebench") && heading != "Running benchmarks briefly." {
		panic("racebench variant is reserved exclusively for registerRaceBenchTest")
	}
	if t.testNames == nil {
		t.testNames = make(map[string]bool)
	}
	t.testNames[name] = true
	t.tests = append(t.tests, distTest{
		name:    name,
		heading: heading,
		fn:      fn,
	})
}

type registerTestOpt interface {
	isRegisterTestOpt()
}

// rtSkipFunc is a registerTest option that runs a skip check function before
// running the test.
type rtSkipFunc struct {
	skip func(*distTest) (string, bool) // Return message, true to skip the test
}

func (rtSkipFunc) isRegisterTestOpt() {}

// registerTest registers a test that runs the given goTest.
//
// Each Go package in goTest will have a corresponding test
// "<pkg>:<variant>", which must uniquely identify the test.
//
// heading and test.variant must be non-empty.
func (t *tester) registerTest(heading string, test *goTest, opts ...registerTestOpt) {
	var skipFunc func(*distTest) (string, bool)
	for _, opt := range opts {
		switch opt := opt.(type) {
		case rtSkipFunc:
			skipFunc = opt.skip
		}
	}
	// Register each test package as a separate test.
	register1 := func(test *goTest) {
		if test.variant == "" {
			panic("empty variant")
		}
		name := testName(test.pkg, test.variant)
		t.addTest(name, heading, func(dt *distTest) error {
			if skipFunc != nil {
				msg, skip := skipFunc(dt)
				if skip {
					test.printSkip(t, msg)
					return nil
				}
			}
			w := &work{dt: dt}
			w.cmd, w.flush = test.bgCommand(t, &w.out, &w.out)
			t.worklist = append(t.worklist, w)
			return nil
		})
	}
	if test.pkg != "" && len(test.pkgs) == 0 {
		// Common case. Avoid copying.
		register1(test)
		return
	}
	// TODO(dmitshur,austin): It might be better to unify the execution of 'go test pkg'
	// invocations for the same variant to be done with a single 'go test pkg1 pkg2 pkg3'
	// command, just like it's already done in registerStdTest and registerRaceBenchTest.
	// Those methods accumulate matched packages in stdMatches and benchMatches slices,
	// and we can extend that mechanism to work for all other equal variant registrations.
	// Do the simple thing to start with.
	for _, pkg := range test.packages() {
		test1 := *test
		test1.pkg, test1.pkgs = pkg, nil
		register1(&test1)
	}
}

// dirCmd constructs a Cmd intended to be run in the foreground.
// The command will be run in dir, and Stdout and Stderr will go to os.Stdout
// and os.Stderr.
func (t *tester) dirCmd(dir string, cmdline ...interface{}) *exec.Cmd {
	bin, args := flattenCmdline(cmdline)
	cmd := exec.Command(bin, args...)
	if filepath.IsAbs(dir) {
		setDir(cmd, dir)
	} else {
		setDir(cmd, filepath.Join(goroot, dir))
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if vflag > 1 {
		errprintf("%#q\n", cmd)
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
			panic("invalid dirCmd argument type: " + reflect.TypeOf(x).String())
		}
	}

	bin = list[0]
	if !filepath.IsAbs(bin) {
		panic("command is not absolute: " + bin)
	}
	return bin, list[1:]
}

func (t *tester) iOS() bool {
	return goos == "ios"
}

func (t *tester) out(v string) {
	if t.json {
		return
	}
	if t.banner == "" {
		return
	}
	fmt.Println("\n" + t.banner + v)
}

// extLink reports whether the current goos/goarch supports
// external linking.
func (t *tester) extLink() bool {
	if !cgoEnabled[goos+"/"+goarch] {
		return false
	}
	if goarch == "ppc64" && goos != "aix" {
		return false
	}
	return true
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
	if goarch == "mips64" || goarch == "mips64le" || goarch == "mips" || goarch == "mipsle" || goarch == "riscv64" {
		return false
	}
	if goos == "aix" {
		// linkmode=internal isn't supported.
		return false
	}
	if t.msan || t.asan {
		// linkmode=internal isn't supported by msan or asan.
		return false
	}
	return true
}

func (t *tester) internalLinkPIE() bool {
	if t.msan || t.asan {
		// linkmode=internal isn't supported by msan or asan.
		return false
	}
	switch goos + "-" + goarch {
	case "darwin-amd64", "darwin-arm64",
		"linux-amd64", "linux-arm64", "linux-loong64", "linux-ppc64le",
		"android-arm64",
		"windows-amd64", "windows-386", "windows-arm":
		return true
	}
	return false
}

func (t *tester) externalLinkPIE() bool {
	// General rule is if -buildmode=pie and -linkmode=external both work, then they work together.
	// Handle exceptions and then fall back to the general rule.
	switch goos + "-" + goarch {
	case "linux-s390x":
		return true
	}
	return t.internalLinkPIE() && t.extLink()
}

// supportedBuildMode reports whether the given build mode is supported.
func (t *tester) supportedBuildmode(mode string) bool {
	switch mode {
	case "c-archive", "c-shared", "shared", "plugin", "pie":
	default:
		fatalf("internal error: unknown buildmode %s", mode)
		return false
	}

	return buildModeSupported("gc", mode, goos, goarch)
}

func (t *tester) registerCgoTests(heading string) {
	cgoTest := func(variant string, subdir, linkmode, buildmode string, opts ...registerTestOpt) *goTest {
		gt := &goTest{
			variant:   variant,
			pkg:       "cmd/cgo/internal/" + subdir,
			buildmode: buildmode,
		}
		var ldflags []string
		if linkmode != "auto" {
			// "auto" is the default, so avoid cluttering the command line for "auto"
			ldflags = append(ldflags, "-linkmode="+linkmode)
		}

		if linkmode == "internal" {
			gt.tags = append(gt.tags, "internal")
			if buildmode == "pie" {
				gt.tags = append(gt.tags, "internal_pie")
			}
		}
		if buildmode == "static" {
			// This isn't actually a Go buildmode, just a convenient way to tell
			// cgoTest we want static linking.
			gt.buildmode = ""
			if linkmode == "external" {
				ldflags = append(ldflags, `-extldflags "-static -pthread"`)
			} else if linkmode == "auto" {
				gt.env = append(gt.env, "CGO_LDFLAGS=-static -pthread")
			} else {
				panic("unknown linkmode with static build: " + linkmode)
			}
			gt.tags = append(gt.tags, "static")
		}
		gt.ldflags = strings.Join(ldflags, " ")

		t.registerTest(heading, gt, opts...)
		return gt
	}

	// test, testtls, and testnocgo are run with linkmode="auto", buildmode=""
	// as part of go test cmd. Here we only have to register the non-default
	// build modes of these tests.

	// Stub out various buildmode=pie tests  on alpine until 54354 resolved.
	builderName := os.Getenv("GO_BUILDER_NAME")
	disablePIE := strings.HasSuffix(builderName, "-alpine")

	if t.internalLink() {
		cgoTest("internal", "test", "internal", "")
	}

	os := gohostos
	p := gohostos + "/" + goarch
	switch {
	case os == "darwin", os == "windows":
		if !t.extLink() {
			break
		}
		// test linkmode=external, but __thread not supported, so skip testtls.
		cgoTest("external", "test", "external", "")

		gt := cgoTest("external-s", "test", "external", "")
		gt.ldflags += " -s"

		if t.supportedBuildmode("pie") && !disablePIE {
			cgoTest("auto-pie", "test", "auto", "pie")
			if t.internalLink() && t.internalLinkPIE() {
				cgoTest("internal-pie", "test", "internal", "pie")
			}
		}

	case os == "aix", os == "android", os == "dragonfly", os == "freebsd", os == "linux", os == "netbsd", os == "openbsd":
		gt := cgoTest("external-g0", "test", "external", "")
		gt.env = append(gt.env, "CGO_CFLAGS=-g0 -fdiagnostics-color")

		cgoTest("external", "testtls", "external", "")
		switch {
		case os == "aix":
			// no static linking
		case p == "freebsd/arm":
			// -fPIC compiled tls code will use __tls_get_addr instead
			// of __aeabi_read_tp, however, on FreeBSD/ARM, __tls_get_addr
			// is implemented in rtld-elf, so -fPIC isn't compatible with
			// static linking on FreeBSD/ARM with clang. (cgo depends on
			// -fPIC fundamentally.)
		default:
			// Check for static linking support
			var staticCheck rtSkipFunc
			ccName := compilerEnvLookup("CC", defaultcc, goos, goarch)
			cc, err := exec.LookPath(ccName)
			if err != nil {
				staticCheck.skip = func(*distTest) (string, bool) {
					return fmt.Sprintf("$CC (%q) not found, skip cgo static linking test.", ccName), true
				}
			} else {
				cmd := t.dirCmd("src/cmd/cgo/internal/test", cc, "-xc", "-o", "/dev/null", "-static", "-")
				cmd.Stdin = strings.NewReader("int main() {}")
				cmd.Stdout, cmd.Stderr = nil, nil // Discard output
				if err := cmd.Run(); err != nil {
					// Skip these tests
					staticCheck.skip = func(*distTest) (string, bool) {
						return "No support for static linking found (lacks libc.a?), skip cgo static linking test.", true
					}
				}
			}

			// Doing a static link with boringcrypto gets
			// a C linker warning on Linux.
			// in function `bio_ip_and_port_to_socket_and_addr':
			// warning: Using 'getaddrinfo' in statically linked applications requires at runtime the shared libraries from the glibc version used for linking
			if staticCheck.skip == nil && goos == "linux" && strings.Contains(goexperiment, "boringcrypto") {
				staticCheck.skip = func(*distTest) (string, bool) {
					return "skipping static linking check on Linux when using boringcrypto to avoid C linker warning about getaddrinfo", true
				}
			}

			// Static linking tests
			if goos != "android" && p != "netbsd/arm" && !t.msan && !t.asan {
				// TODO(#56629): Why does this fail on netbsd-arm?
				// TODO(#70080): Why does this fail with msan?
				// asan doesn't support static linking (this is an explicit build error on the C side).
				cgoTest("static", "testtls", "external", "static", staticCheck)
			}
			cgoTest("external", "testnocgo", "external", "", staticCheck)
			if goos != "android" && !t.msan && !t.asan {
				// TODO(#70080): Why does this fail with msan?
				// asan doesn't support static linking (this is an explicit build error on the C side).
				cgoTest("static", "testnocgo", "external", "static", staticCheck)
				cgoTest("static", "test", "external", "static", staticCheck)
				// -static in CGO_LDFLAGS triggers a different code path
				// than -static in -extldflags, so test both.
				// See issue #16651.
				if goarch != "loong64" && !t.msan && !t.asan {
					// TODO(#56623): Why does this fail on loong64?
					cgoTest("auto-static", "test", "auto", "static", staticCheck)
				}
			}

			// PIE linking tests
			if t.supportedBuildmode("pie") && !disablePIE {
				cgoTest("auto-pie", "test", "auto", "pie")
				if t.internalLink() && t.internalLinkPIE() {
					cgoTest("internal-pie", "test", "internal", "pie")
				}
				cgoTest("auto-pie", "testtls", "auto", "pie")
				cgoTest("auto-pie", "testnocgo", "auto", "pie")
			}
		}
	}
}

// runPending runs pending test commands, in parallel, emitting headers as appropriate.
// When finished, it emits header for nextTest, which is going to run after the
// pending commands are done (and runPending returns).
// A test should call runPending if it wants to make sure that it is not
// running in parallel with earlier tests, or if it has some other reason
// for needing the earlier tests to be done.
func (t *tester) runPending(nextTest *distTest) {
	worklist := t.worklist
	t.worklist = nil
	for _, w := range worklist {
		w.start = make(chan bool)
		w.end = make(chan struct{})
		// w.cmd must be set up to write to w.out. We can't check that, but we
		// can check for easy mistakes.
		if w.cmd.Stdout == nil || w.cmd.Stdout == os.Stdout || w.cmd.Stderr == nil || w.cmd.Stderr == os.Stderr {
			panic("work.cmd.Stdout/Stderr must be redirected")
		}
		go func(w *work) {
			if !<-w.start {
				timelog("skip", w.dt.name)
				w.printSkip(t, "skipped due to earlier error")
			} else {
				timelog("start", w.dt.name)
				w.err = w.cmd.Run()
				if w.flush != nil {
					w.flush()
				}
				if w.err != nil {
					if isUnsupportedVMASize(w) {
						timelog("skip", w.dt.name)
						w.out.Reset()
						w.printSkip(t, "skipped due to unsupported VMA")
						w.err = nil
					}
				}
			}
			timelog("end", w.dt.name)
			w.end <- struct{}{}
		}(w)
	}

	maxbg := maxbg
	// for runtime.NumCPU() < 4 ||  runtime.GOMAXPROCS(0) == 1, do not change maxbg.
	// Because there is not enough CPU to parallel the testing of multiple packages.
	if runtime.NumCPU() > 4 && runtime.GOMAXPROCS(0) != 1 {
		for _, w := range worklist {
			// See go.dev/issue/65164
			// because GOMAXPROCS=2 runtime CPU usage is low,
			// so increase maxbg to avoid slowing down execution with low CPU usage.
			// This makes testing a single package slower,
			// but testing multiple packages together faster.
			if strings.Contains(w.dt.heading, "GOMAXPROCS=2 runtime") {
				maxbg = runtime.NumCPU()
				break
			}
		}
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
		if t.lastHeading != dt.heading {
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
			errprintf("%#q\n", w.cmd)
		}
		ended++
		<-w.end
		os.Stdout.Write(w.out.Bytes())
		// We no longer need the output, so drop the buffer.
		w.out = bytes.Buffer{}
		if w.err != nil {
			log.Printf("Failed: %v", w.err)
			t.failed = true
		}
	}
	if t.failed && !t.keepGoing {
		fatalf("FAILED")
	}

	if dt := nextTest; dt != nil {
		if t.lastHeading != dt.heading {
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

// hasParallelism is a copy of the function
// internal/testenv.HasParallelism, which can't be used here
// because cmd/dist can not import internal packages during bootstrap.
func (t *tester) hasParallelism() bool {
	switch goos {
	case "js", "wasip1":
		return false
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

func (t *tester) registerRaceTests() {
	hdr := "Testing race detector"
	t.registerTest(hdr,
		&goTest{
			variant:  "race",
			race:     true,
			runTests: "Output",
			pkg:      "runtime/race",
		})
	t.registerTest(hdr,
		&goTest{
			variant:  "race",
			race:     true,
			runTests: "TestParse|TestEcho|TestStdinCloseRace|TestClosedPipeRace|TestTypeRace|TestFdRace|TestFdReadRace|TestFileCloseRace",
			pkgs:     []string{"flag", "net", "os", "os/exec", "encoding/gob"},
		})
	// We don't want the following line, because it
	// slows down all.bash (by 10 seconds on my laptop).
	// The race builder should catch any error here, but doesn't.
	// TODO(iant): Figure out how to catch this.
	// t.registerTest(hdr, &goTest{variant: "race", race: true, runTests: "TestParallelTest", pkg: "cmd/go"})
	if t.cgoEnabled {
		// Building cmd/cgo/internal/test takes a long time.
		// There are already cgo-enabled packages being tested with the race detector.
		// We shouldn't need to redo all of cmd/cgo/internal/test too.
		// The race builder will take care of this.
		// t.registerTest(hdr, &goTest{variant: "race", race: true, env: []string{"GOTRACEBACK=2"}, pkg: "cmd/cgo/internal/test"})
	}
	if t.extLink() {
		// Test with external linking; see issue 9133.
		t.registerTest(hdr,
			&goTest{
				variant:  "race-external",
				race:     true,
				ldflags:  "-linkmode=external",
				runTests: "TestParse|TestEcho|TestStdinCloseRace",
				pkgs:     []string{"flag", "os/exec"},
			})
	}
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
		slurp, err := os.ReadFile(filepath.Join(pkgDir, name))
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

	filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if suffix := strings.TrimPrefix(path, dir+string(filepath.Separator)); suffix != "" {
			if suffix == ".git" {
				// Leave Git metadata in whatever state it was in. It may contain a lot
				// of files, and it is highly unlikely that a test will try to modify
				// anything within that directory.
				return filepath.SkipDir
			}
		}
		if err != nil {
			return nil
		}

		info, err := d.Info()
		if err != nil {
			return nil
		}

		mode := info.Mode()
		if mode&0222 != 0 && (mode.IsDir() || mode.IsRegular()) {
			dirs = append(dirs, pathMode{path, mode})
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

// raceDetectorSupported is a copy of the function
// internal/platform.RaceDetectorSupported, which can't be used here
// because cmd/dist can not import internal packages during bootstrap.
// The race detector only supports 48-bit VMA on arm64. But we don't have
// a good solution to check VMA size (see https://go.dev/issue/29948).
// raceDetectorSupported will always return true for arm64. But race
// detector tests may abort on non 48-bit VMA configuration, the tests
// will be marked as "skipped" in this case.
func raceDetectorSupported(goos, goarch string) bool {
	switch goos {
	case "linux":
		return goarch == "amd64" || goarch == "ppc64le" || goarch == "arm64" || goarch == "s390x" || goarch == "loong64"
	case "darwin":
		return goarch == "amd64" || goarch == "arm64"
	case "freebsd", "netbsd", "windows":
		return goarch == "amd64"
	default:
		return false
	}
}

// buildModeSupported is a copy of the function
// internal/platform.BuildModeSupported, which can't be used here
// because cmd/dist can not import internal packages during bootstrap.
func buildModeSupported(compiler, buildmode, goos, goarch string) bool {
	if compiler == "gccgo" {
		return true
	}

	platform := goos + "/" + goarch

	switch buildmode {
	case "archive":
		return true

	case "c-archive":
		switch goos {
		case "aix", "darwin", "ios", "windows":
			return true
		case "linux":
			switch goarch {
			case "386", "amd64", "arm", "armbe", "arm64", "arm64be", "loong64", "ppc64le", "riscv64", "s390x":
				// linux/ppc64 not supported because it does
				// not support external linking mode yet.
				return true
			default:
				// Other targets do not support -shared,
				// per ParseFlags in
				// cmd/compile/internal/base/flag.go.
				// For c-archive the Go tool passes -shared,
				// so that the result is suitable for inclusion
				// in a PIE or shared library.
				return false
			}
		case "freebsd":
			return goarch == "amd64"
		}
		return false

	case "c-shared":
		switch platform {
		case "linux/amd64", "linux/arm", "linux/arm64", "linux/loong64", "linux/386", "linux/ppc64le", "linux/riscv64", "linux/s390x",
			"android/amd64", "android/arm", "android/arm64", "android/386",
			"freebsd/amd64",
			"darwin/amd64", "darwin/arm64",
			"windows/amd64", "windows/386", "windows/arm64",
			"wasip1/wasm":
			return true
		}
		return false

	case "default":
		return true

	case "exe":
		return true

	case "pie":
		switch platform {
		case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/loong64", "linux/ppc64le", "linux/riscv64", "linux/s390x",
			"android/amd64", "android/arm", "android/arm64", "android/386",
			"freebsd/amd64",
			"darwin/amd64", "darwin/arm64",
			"ios/amd64", "ios/arm64",
			"aix/ppc64",
			"openbsd/arm64",
			"windows/386", "windows/amd64", "windows/arm", "windows/arm64":
			return true
		}
		return false

	case "shared":
		switch platform {
		case "linux/386", "linux/amd64", "linux/arm", "linux/arm64", "linux/ppc64le", "linux/s390x":
			return true
		}
		return false

	case "plugin":
		switch platform {
		case "linux/amd64", "linux/arm", "linux/arm64", "linux/386", "linux/loong64", "linux/riscv64", "linux/s390x", "linux/ppc64le",
			"android/amd64", "android/386",
			"darwin/amd64", "darwin/arm64",
			"freebsd/amd64":
			return true
		}
		return false

	default:
		return false
	}
}

// isUnsupportedVMASize reports whether the failure is caused by an unsupported
// VMA for the race detector (for example, running the race detector on an
// arm64 machine configured with 39-bit VMA).
func isUnsupportedVMASize(w *work) bool {
	unsupportedVMA := []byte("unsupported VMA range")
	return strings.Contains(w.dt.name, ":race") && bytes.Contains(w.out.Bytes(), unsupportedVMA)
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

func (t *tester) fipsSupported() bool {
	// Keep this in sync with [crypto/internal/fips140.Supported].

	// Use GOFIPS140 or GOEXPERIMENT=boringcrypto, but not both.
	if strings.Contains(goexperiment, "boringcrypto") {
		return false
	}

	// If this goos/goarch does not support FIPS at all, return no versions.
	// The logic here matches crypto/internal/fips140/check.Supported for now.
	// In the future, if some snapshots add support for these, we will have
	// to make a decision on a per-version basis.
	switch {
	case goarch == "wasm",
		goos == "windows" && goarch == "386",
		goos == "windows" && goarch == "arm",
		goos == "openbsd",
		goos == "aix":
		return false
	}

	// For now, FIPS+ASAN doesn't need to work.
	// If this is made to work, also re-enable the test in check_test.go.
	if t.asan {
		return false
	}

	return true
}

// fipsVersions returns the list of versions available in lib/fips140.
func fipsVersions(short bool) []string {
	var versions []string
	zips, err := filepath.Glob(filepath.Join(goroot, "lib/fips140/*.zip"))
	if err != nil {
		fatalf("%v", err)
	}
	for _, zip := range zips {
		versions = append(versions, strings.TrimSuffix(filepath.Base(zip), ".zip"))
	}
	txts, err := filepath.Glob(filepath.Join(goroot, "lib/fips140/*.txt"))
	if err != nil {
		fatalf("%v", err)
	}
	for _, txt := range txts {
		versions = append(versions, strings.TrimSuffix(filepath.Base(txt), ".txt"))
	}
	return versions
}
