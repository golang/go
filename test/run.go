// skip

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run runs tests in the test directory.
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"go/build/constraint"
	"hash/fnv"
	"io"
	"io/fs"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

var (
	verbose        = flag.Bool("v", false, "verbose. if set, parallelism is set to 1.")
	keep           = flag.Bool("k", false, "keep. keep temporary directory.")
	numParallel    = flag.Int("n", runtime.NumCPU(), "number of parallel tests to run")
	summary        = flag.Bool("summary", false, "show summary of results")
	allCodegen     = flag.Bool("all_codegen", defaultAllCodeGen(), "run all goos/goarch for codegen")
	showSkips      = flag.Bool("show_skips", false, "show skipped tests")
	runSkips       = flag.Bool("run_skips", false, "run skipped tests (ignore skip and build tags)")
	linkshared     = flag.Bool("linkshared", false, "")
	updateErrors   = flag.Bool("update_errors", false, "update error messages in test file based on compiler output")
	runoutputLimit = flag.Int("l", defaultRunOutputLimit(), "number of parallel runoutput tests to run")
	force          = flag.Bool("f", false, "ignore expected-failure test lists")

	shard  = flag.Int("shard", 0, "shard index to run. Only applicable if -shards is non-zero.")
	shards = flag.Int("shards", 0, "number of shards. If 0, all tests are run. This is used by the continuous build.")
)

type envVars struct {
	GOOS         string
	GOARCH       string
	GOEXPERIMENT string
	CGO_ENABLED  string
}

var env = func() (res envVars) {
	cmd := exec.Command(goTool(), "env", "-json")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatal("StdoutPipe:", err)
	}
	if err := cmd.Start(); err != nil {
		log.Fatal("Start:", err)
	}
	if err := json.NewDecoder(stdout).Decode(&res); err != nil {
		log.Fatal("Decode:", err)
	}
	if err := cmd.Wait(); err != nil {
		log.Fatal("Wait:", err)
	}
	return
}()

var unifiedEnabled = func() bool {
	for _, tag := range build.Default.ToolTags {
		if tag == "goexperiment.unified" {
			return true
		}
	}
	return false
}()

// defaultAllCodeGen returns the default value of the -all_codegen
// flag. By default, we prefer to be fast (returning false), except on
// the linux-amd64 builder that's already very fast, so we get more
// test coverage on trybots. See https://golang.org/issue/34297.
func defaultAllCodeGen() bool {
	return os.Getenv("GO_BUILDER_NAME") == "linux-amd64"
}

func optimizationOff() bool {
	return strings.HasSuffix(os.Getenv("GO_BUILDER_NAME"), "-noopt")
}

var (
	goos          = env.GOOS
	goarch        = env.GOARCH
	cgoEnabled, _ = strconv.ParseBool(env.CGO_ENABLED)

	// dirs are the directories to look for *.go files in.
	// TODO(bradfitz): just use all directories?
	dirs = []string{".", "ken", "chan", "interface", "syntax", "dwarf", "fixedbugs", "codegen", "runtime", "abi", "typeparam", "typeparam/mdempsky"}

	// ratec controls the max number of tests running at a time.
	ratec chan bool

	// toRun is the channel of tests to run.
	// It is nil until the first test is started.
	toRun chan *test

	// rungatec controls the max number of runoutput tests
	// executed in parallel as they can each consume a lot of memory.
	rungatec chan bool
)

// maxTests is an upper bound on the total number of tests.
// It is used as a channel buffer size to make sure sends don't block.
const maxTests = 5000

func main() {
	flag.Parse()

	findExecCmd()

	// Disable parallelism if printing or if using a simulator.
	if *verbose || len(findExecCmd()) > 0 {
		*numParallel = 1
		*runoutputLimit = 1
	}

	ratec = make(chan bool, *numParallel)
	rungatec = make(chan bool, *runoutputLimit)

	var tests []*test
	if flag.NArg() > 0 {
		for _, arg := range flag.Args() {
			if arg == "-" || arg == "--" {
				// Permit running:
				// $ go run run.go - env.go
				// $ go run run.go -- env.go
				// $ go run run.go - ./fixedbugs
				// $ go run run.go -- ./fixedbugs
				continue
			}
			if fi, err := os.Stat(arg); err == nil && fi.IsDir() {
				for _, baseGoFile := range goFiles(arg) {
					tests = append(tests, startTest(arg, baseGoFile))
				}
			} else if strings.HasSuffix(arg, ".go") {
				dir, file := filepath.Split(arg)
				tests = append(tests, startTest(dir, file))
			} else {
				log.Fatalf("can't yet deal with non-directory and non-go file %q", arg)
			}
		}
	} else {
		for _, dir := range dirs {
			for _, baseGoFile := range goFiles(dir) {
				tests = append(tests, startTest(dir, baseGoFile))
			}
		}
	}

	failed := false
	resCount := map[string]int{}
	for _, test := range tests {
		<-test.donec
		status := "ok  "
		errStr := ""
		if e, isSkip := test.err.(skipError); isSkip {
			test.err = nil
			errStr = "unexpected skip for " + path.Join(test.dir, test.gofile) + ": " + string(e)
			status = "FAIL"
		}
		if test.err != nil {
			errStr = test.err.Error()
			if test.expectFail {
				errStr += " (expected)"
			} else {
				status = "FAIL"
			}
		} else if test.expectFail {
			status = "FAIL"
			errStr = "unexpected success"
		}
		if status == "FAIL" {
			failed = true
		}
		resCount[status]++
		dt := fmt.Sprintf("%.3fs", test.dt.Seconds())
		if status == "FAIL" {
			fmt.Printf("# go run run.go -- %s\n%s\nFAIL\t%s\t%s\n",
				path.Join(test.dir, test.gofile),
				errStr, test.goFileName(), dt)
			continue
		}
		if !*verbose {
			continue
		}
		fmt.Printf("%s\t%s\t%s\n", status, test.goFileName(), dt)
	}

	if *summary {
		for k, v := range resCount {
			fmt.Printf("%5d %s\n", v, k)
		}
	}

	if failed {
		os.Exit(1)
	}
}

// goTool reports the path of the go tool to use to run the tests.
// If possible, use the same Go used to run run.go, otherwise
// fallback to the go version found in the PATH.
func goTool() string {
	var exeSuffix string
	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}
	path := filepath.Join(runtime.GOROOT(), "bin", "go"+exeSuffix)
	if _, err := os.Stat(path); err == nil {
		return path
	}
	// Just run "go" from PATH
	return "go"
}

func shardMatch(name string) bool {
	if *shards == 0 {
		return true
	}
	h := fnv.New32()
	io.WriteString(h, name)
	return int(h.Sum32()%uint32(*shards)) == *shard
}

func goFiles(dir string) []string {
	f, err := os.Open(dir)
	if err != nil {
		log.Fatal(err)
	}
	dirnames, err := f.Readdirnames(-1)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	names := []string{}
	for _, name := range dirnames {
		if !strings.HasPrefix(name, ".") && strings.HasSuffix(name, ".go") && shardMatch(name) {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

type runCmd func(...string) ([]byte, error)

func compileFile(runcmd runCmd, longname string, flags []string) (out []byte, err error) {
	cmd := []string{goTool(), "tool", "compile", "-e", "-p=p"}
	cmd = append(cmd, flags...)
	if *linkshared {
		cmd = append(cmd, "-dynlink", "-installsuffix=dynlink")
	}
	cmd = append(cmd, longname)
	return runcmd(cmd...)
}

func compileInDir(runcmd runCmd, dir string, flags []string, pkgname string, names ...string) (out []byte, err error) {
	cmd := []string{goTool(), "tool", "compile", "-e", "-D", "test", "-I", "."}
	if pkgname == "main" {
		cmd = append(cmd, "-p=main")
	} else {
		pkgname = path.Join("test", strings.TrimSuffix(names[0], ".go"))
		cmd = append(cmd, "-o", pkgname+".a", "-p", pkgname)
	}
	cmd = append(cmd, flags...)
	if *linkshared {
		cmd = append(cmd, "-dynlink", "-installsuffix=dynlink")
	}
	for _, name := range names {
		cmd = append(cmd, filepath.Join(dir, name))
	}
	return runcmd(cmd...)
}

func linkFile(runcmd runCmd, goname string, ldflags []string) (err error) {
	pfile := strings.Replace(goname, ".go", ".o", -1)
	cmd := []string{goTool(), "tool", "link", "-w", "-o", "a.exe", "-L", "."}
	if *linkshared {
		cmd = append(cmd, "-linkshared", "-installsuffix=dynlink")
	}
	if ldflags != nil {
		cmd = append(cmd, ldflags...)
	}
	cmd = append(cmd, pfile)
	_, err = runcmd(cmd...)
	return
}

// skipError describes why a test was skipped.
type skipError string

func (s skipError) Error() string { return string(s) }

// test holds the state of a test.
type test struct {
	dir, gofile string
	donec       chan bool // closed when done
	dt          time.Duration

	src string

	tempDir string
	err     error

	// expectFail indicates whether the (overall) test recipe is
	// expected to fail under the current test configuration (e.g.,
	// GOEXPERIMENT=unified).
	expectFail bool
}

// initExpectFail initializes t.expectFail based on the build+test
// configuration.
func (t *test) initExpectFail() {
	if *force {
		return
	}

	failureSets := []map[string]bool{types2Failures}

	// Note: gccgo supports more 32-bit architectures than this, but
	// hopefully the 32-bit failures are fixed before this matters.
	switch goarch {
	case "386", "arm", "mips", "mipsle":
		failureSets = append(failureSets, types2Failures32Bit)
	}

	if unifiedEnabled {
		failureSets = append(failureSets, unifiedFailures)
	} else {
		failureSets = append(failureSets, go118Failures)
	}

	filename := strings.Replace(t.goFileName(), "\\", "/", -1) // goFileName() uses \ on Windows

	for _, set := range failureSets {
		if set[filename] {
			t.expectFail = true
			return
		}
	}
}

func startTest(dir, gofile string) *test {
	t := &test{
		dir:    dir,
		gofile: gofile,
		donec:  make(chan bool, 1),
	}
	if toRun == nil {
		toRun = make(chan *test, maxTests)
		go runTests()
	}
	select {
	case toRun <- t:
	default:
		panic("toRun buffer size (maxTests) is too small")
	}
	return t
}

// runTests runs tests in parallel, but respecting the order they
// were enqueued on the toRun channel.
func runTests() {
	for {
		ratec <- true
		t := <-toRun
		go func() {
			t.run()
			<-ratec
		}()
	}
}

var cwd, _ = os.Getwd()

func (t *test) goFileName() string {
	return filepath.Join(t.dir, t.gofile)
}

func (t *test) goDirName() string {
	return filepath.Join(t.dir, strings.Replace(t.gofile, ".go", ".dir", -1))
}

func goDirFiles(longdir string) (filter []os.FileInfo, err error) {
	files, dirErr := ioutil.ReadDir(longdir)
	if dirErr != nil {
		return nil, dirErr
	}
	for _, gofile := range files {
		if filepath.Ext(gofile.Name()) == ".go" {
			filter = append(filter, gofile)
		}
	}
	return
}

var packageRE = regexp.MustCompile(`(?m)^package ([\p{Lu}\p{Ll}\w]+)`)

func getPackageNameFromSource(fn string) (string, error) {
	data, err := ioutil.ReadFile(fn)
	if err != nil {
		return "", err
	}
	pkgname := packageRE.FindStringSubmatch(string(data))
	if pkgname == nil {
		return "", fmt.Errorf("cannot find package name in %s", fn)
	}
	return pkgname[1], nil
}

type goDirPkg struct {
	name  string
	files []string
}

// If singlefilepkgs is set, each file is considered a separate package
// even if the package names are the same.
func goDirPackages(longdir string, singlefilepkgs bool) ([]*goDirPkg, error) {
	files, err := goDirFiles(longdir)
	if err != nil {
		return nil, err
	}
	var pkgs []*goDirPkg
	m := make(map[string]*goDirPkg)
	for _, file := range files {
		name := file.Name()
		pkgname, err := getPackageNameFromSource(filepath.Join(longdir, name))
		if err != nil {
			log.Fatal(err)
		}
		p, ok := m[pkgname]
		if singlefilepkgs || !ok {
			p = &goDirPkg{name: pkgname}
			pkgs = append(pkgs, p)
			m[pkgname] = p
		}
		p.files = append(p.files, name)
	}
	return pkgs, nil
}

type context struct {
	GOOS       string
	GOARCH     string
	cgoEnabled bool
	noOptEnv   bool
}

// shouldTest looks for build tags in a source file and returns
// whether the file should be used according to the tags.
func shouldTest(src string, goos, goarch string) (ok bool, whyNot string) {
	if *runSkips {
		return true, ""
	}
	for _, line := range strings.Split(src, "\n") {
		if strings.HasPrefix(line, "package ") {
			break
		}

		if expr, err := constraint.Parse(line); err == nil {
			gcFlags := os.Getenv("GO_GCFLAGS")
			ctxt := &context{
				GOOS:       goos,
				GOARCH:     goarch,
				cgoEnabled: cgoEnabled,
				noOptEnv:   strings.Contains(gcFlags, "-N") || strings.Contains(gcFlags, "-l"),
			}

			if !expr.Eval(ctxt.match) {
				return false, line
			}
		}
	}
	return true, ""
}

func (ctxt *context) match(name string) bool {
	if name == "" {
		return false
	}

	// Tags must be letters, digits, underscores or dots.
	// Unlike in Go identifiers, all digits are fine (e.g., "386").
	for _, c := range name {
		if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
			return false
		}
	}

	if strings.HasPrefix(name, "goexperiment.") {
		for _, tag := range build.Default.ToolTags {
			if tag == name {
				return true
			}
		}
		return false
	}

	if name == "cgo" && ctxt.cgoEnabled {
		return true
	}

	if name == ctxt.GOOS || name == ctxt.GOARCH || name == "gc" {
		return true
	}

	if ctxt.noOptEnv && name == "gcflags_noopt" {
		return true
	}

	if name == "test_run" {
		return true
	}

	return false
}

func init() {
	checkShouldTest()
}

// goGcflags returns the -gcflags argument to use with go build / go run.
// This must match the flags used for building the standard library,
// or else the commands will rebuild any needed packages (like runtime)
// over and over.
func (t *test) goGcflags() string {
	return "-gcflags=all=" + os.Getenv("GO_GCFLAGS")
}

func (t *test) goGcflagsIsEmpty() bool {
	return "" == os.Getenv("GO_GCFLAGS")
}

var errTimeout = errors.New("command exceeded time limit")

// run runs a test.
func (t *test) run() {
	start := time.Now()
	defer func() {
		t.dt = time.Since(start)
		close(t.donec)
	}()

	srcBytes, err := ioutil.ReadFile(t.goFileName())
	if err != nil {
		t.err = err
		return
	}
	t.src = string(srcBytes)
	if t.src[0] == '\n' {
		t.err = skipError("starts with newline")
		return
	}

	// Execution recipe stops at first blank line.
	action, _, ok := strings.Cut(t.src, "\n\n")
	if !ok {
		t.err = fmt.Errorf("double newline ending execution recipe not found in %s", t.goFileName())
		return
	}
	if firstLine, rest, ok := strings.Cut(action, "\n"); ok && strings.Contains(firstLine, "+build") {
		// skip first line
		action = rest
	}
	action = strings.TrimPrefix(action, "//")

	// Check for build constraints only up to the actual code.
	header, _, ok := strings.Cut(t.src, "\npackage")
	if !ok {
		header = action // some files are intentionally malformed
	}
	if ok, why := shouldTest(header, goos, goarch); !ok {
		if *showSkips {
			fmt.Printf("%-20s %-20s: %s\n", "skip", t.goFileName(), why)
		}
		return
	}

	var args, flags, runenv []string
	var tim int
	wantError := false
	wantAuto := false
	singlefilepkgs := false
	f, err := splitQuoted(action)
	if err != nil {
		t.err = fmt.Errorf("invalid test recipe: %v", err)
		return
	}
	if len(f) > 0 {
		action = f[0]
		args = f[1:]
	}

	// TODO: Clean up/simplify this switch statement.
	switch action {
	case "compile", "compiledir", "build", "builddir", "buildrundir", "run", "buildrun", "runoutput", "rundir", "runindir", "asmcheck":
		// nothing to do
	case "errorcheckandrundir":
		wantError = false // should be no error if also will run
	case "errorcheckwithauto":
		action = "errorcheck"
		wantAuto = true
		wantError = true
	case "errorcheck", "errorcheckdir", "errorcheckoutput":
		wantError = true
	case "skip":
		if *runSkips {
			break
		}
		return
	default:
		t.err = skipError("skipped; unknown pattern: " + action)
		return
	}

	goexp := env.GOEXPERIMENT

	// collect flags
	for len(args) > 0 && strings.HasPrefix(args[0], "-") {
		switch args[0] {
		case "-1":
			wantError = true
		case "-0":
			wantError = false
		case "-s":
			singlefilepkgs = true
		case "-t": // timeout in seconds
			args = args[1:]
			var err error
			tim, err = strconv.Atoi(args[0])
			if err != nil {
				t.err = fmt.Errorf("need number of seconds for -t timeout, got %s instead", args[0])
			}
			if s := os.Getenv("GO_TEST_TIMEOUT_SCALE"); s != "" {
				timeoutScale, err := strconv.Atoi(s)
				if err != nil {
					log.Fatalf("failed to parse $GO_TEST_TIMEOUT_SCALE = %q as integer: %v", s, err)
				}
				tim *= timeoutScale
			}
		case "-goexperiment": // set GOEXPERIMENT environment
			args = args[1:]
			if goexp != "" {
				goexp += ","
			}
			goexp += args[0]
			runenv = append(runenv, "GOEXPERIMENT="+goexp)

		default:
			flags = append(flags, args[0])
		}
		args = args[1:]
	}
	if action == "errorcheck" {
		found := false
		for i, f := range flags {
			if strings.HasPrefix(f, "-d=") {
				flags[i] = f + ",ssa/check/on"
				found = true
				break
			}
		}
		if !found {
			flags = append(flags, "-d=ssa/check/on")
		}
	}

	t.initExpectFail()
	t.makeTempDir()
	if !*keep {
		defer os.RemoveAll(t.tempDir)
	}

	err = ioutil.WriteFile(filepath.Join(t.tempDir, t.gofile), srcBytes, 0644)
	if err != nil {
		log.Fatal(err)
	}

	// A few tests (of things like the environment) require these to be set.
	if os.Getenv("GOOS") == "" {
		os.Setenv("GOOS", runtime.GOOS)
	}
	if os.Getenv("GOARCH") == "" {
		os.Setenv("GOARCH", runtime.GOARCH)
	}

	var (
		runInDir        = t.tempDir
		tempDirIsGOPATH = false
	)
	runcmd := func(args ...string) ([]byte, error) {
		cmd := exec.Command(args[0], args[1:]...)
		var buf bytes.Buffer
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		cmd.Env = append(os.Environ(), "GOENV=off", "GOFLAGS=")
		if runInDir != "" {
			cmd.Dir = runInDir
			// Set PWD to match Dir to speed up os.Getwd in the child process.
			cmd.Env = append(cmd.Env, "PWD="+cmd.Dir)
		}
		if tempDirIsGOPATH {
			cmd.Env = append(cmd.Env, "GOPATH="+t.tempDir)
		}
		// Put the bin directory of the GOROOT that built this program
		// first in the path. This ensures that tests that use the "go"
		// tool use the same one that built this program. This ensures
		// that if you do "../bin/go run run.go" in this directory, all
		// the tests that start subprocesses that "go tool compile" or
		// whatever, use ../bin/go as their go tool, not whatever happens
		// to be first in the user's path.
		path := os.Getenv("PATH")
		newdir := filepath.Join(runtime.GOROOT(), "bin")
		if path != "" {
			path = newdir + string(filepath.ListSeparator) + path
		} else {
			path = newdir
		}
		cmd.Env = append(cmd.Env, "PATH="+path)

		cmd.Env = append(cmd.Env, runenv...)

		var err error

		if tim != 0 {
			err = cmd.Start()
			// This command-timeout code adapted from cmd/go/test.go
			// Note: the Go command uses a more sophisticated timeout
			// strategy, first sending SIGQUIT (if appropriate for the
			// OS in question) to try to trigger a stack trace, then
			// finally much later SIGKILL. If timeouts prove to be a
			// common problem here, it would be worth porting over
			// that code as well. See https://do.dev/issue/50973
			// for more discussion.
			if err == nil {
				tick := time.NewTimer(time.Duration(tim) * time.Second)
				done := make(chan error)
				go func() {
					done <- cmd.Wait()
				}()
				select {
				case err = <-done:
					// ok
				case <-tick.C:
					cmd.Process.Signal(os.Interrupt)
					time.Sleep(1 * time.Second)
					cmd.Process.Kill()
					<-done
					err = errTimeout
				}
				tick.Stop()
			}
		} else {
			err = cmd.Run()
		}
		if err != nil && err != errTimeout {
			err = fmt.Errorf("%s\n%s", err, buf.Bytes())
		}
		return buf.Bytes(), err
	}

	long := filepath.Join(cwd, t.goFileName())
	switch action {
	default:
		t.err = fmt.Errorf("unimplemented action %q", action)

	case "asmcheck":
		// Compile Go file and match the generated assembly
		// against a set of regexps in comments.
		ops := t.wantedAsmOpcodes(long)
		self := runtime.GOOS + "/" + runtime.GOARCH
		for _, env := range ops.Envs() {
			// Only run checks relevant to the current GOOS/GOARCH,
			// to avoid triggering a cross-compile of the runtime.
			if string(env) != self && !strings.HasPrefix(string(env), self+"/") && !*allCodegen {
				continue
			}
			// -S=2 forces outermost line numbers when disassembling inlined code.
			cmdline := []string{"build", "-gcflags", "-S=2"}

			// Append flags, but don't override -gcflags=-S=2; add to it instead.
			for i := 0; i < len(flags); i++ {
				flag := flags[i]
				switch {
				case strings.HasPrefix(flag, "-gcflags="):
					cmdline[2] += " " + strings.TrimPrefix(flag, "-gcflags=")
				case strings.HasPrefix(flag, "--gcflags="):
					cmdline[2] += " " + strings.TrimPrefix(flag, "--gcflags=")
				case flag == "-gcflags", flag == "--gcflags":
					i++
					if i < len(flags) {
						cmdline[2] += " " + flags[i]
					}
				default:
					cmdline = append(cmdline, flag)
				}
			}

			cmdline = append(cmdline, long)
			cmd := exec.Command(goTool(), cmdline...)
			cmd.Env = append(os.Environ(), env.Environ()...)
			if len(flags) > 0 && flags[0] == "-race" {
				cmd.Env = append(cmd.Env, "CGO_ENABLED=1")
			}

			var buf bytes.Buffer
			cmd.Stdout, cmd.Stderr = &buf, &buf
			if err := cmd.Run(); err != nil {
				fmt.Println(env, "\n", cmd.Stderr)
				t.err = err
				return
			}

			t.err = t.asmCheck(buf.String(), long, env, ops[env])
			if t.err != nil {
				return
			}
		}
		return

	case "errorcheck":
		// Compile Go file.
		// Fail if wantError is true and compilation was successful and vice versa.
		// Match errors produced by gc against errors in comments.
		// TODO(gri) remove need for -C (disable printing of columns in error messages)
		cmdline := []string{goTool(), "tool", "compile", "-p=p", "-d=panic", "-C", "-e", "-o", "a.o"}
		// No need to add -dynlink even if linkshared if we're just checking for errors...
		cmdline = append(cmdline, flags...)
		cmdline = append(cmdline, long)
		out, err := runcmd(cmdline...)
		if wantError {
			if err == nil {
				t.err = fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
				return
			}
			if err == errTimeout {
				t.err = fmt.Errorf("compilation timed out")
				return
			}
		} else {
			if err != nil {
				t.err = err
				return
			}
		}
		if *updateErrors {
			t.updateErrors(string(out), long)
		}
		t.err = t.errorCheck(string(out), wantAuto, long, t.gofile)

	case "compile":
		// Compile Go file.
		_, t.err = compileFile(runcmd, long, flags)

	case "compiledir":
		// Compile all files in the directory as packages in lexicographic order.
		longdir := filepath.Join(cwd, t.goDirName())
		pkgs, err := goDirPackages(longdir, singlefilepkgs)
		if err != nil {
			t.err = err
			return
		}
		for _, pkg := range pkgs {
			_, t.err = compileInDir(runcmd, longdir, flags, pkg.name, pkg.files...)
			if t.err != nil {
				return
			}
		}

	case "errorcheckdir", "errorcheckandrundir":
		flags = append(flags, "-d=panic")
		// Compile and errorCheck all files in the directory as packages in lexicographic order.
		// If errorcheckdir and wantError, compilation of the last package must fail.
		// If errorcheckandrundir and wantError, compilation of the package prior the last must fail.
		longdir := filepath.Join(cwd, t.goDirName())
		pkgs, err := goDirPackages(longdir, singlefilepkgs)
		if err != nil {
			t.err = err
			return
		}
		errPkg := len(pkgs) - 1
		if wantError && action == "errorcheckandrundir" {
			// The last pkg should compiled successfully and will be run in next case.
			// Preceding pkg must return an error from compileInDir.
			errPkg--
		}
		for i, pkg := range pkgs {
			out, err := compileInDir(runcmd, longdir, flags, pkg.name, pkg.files...)
			if i == errPkg {
				if wantError && err == nil {
					t.err = fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
					return
				} else if !wantError && err != nil {
					t.err = err
					return
				}
			} else if err != nil {
				t.err = err
				return
			}
			var fullshort []string
			for _, name := range pkg.files {
				fullshort = append(fullshort, filepath.Join(longdir, name), name)
			}
			t.err = t.errorCheck(string(out), wantAuto, fullshort...)
			if t.err != nil {
				break
			}
		}
		if action == "errorcheckdir" {
			return
		}
		fallthrough

	case "rundir":
		// Compile all files in the directory as packages in lexicographic order.
		// In case of errorcheckandrundir, ignore failed compilation of the package before the last.
		// Link as if the last file is the main package, run it.
		// Verify the expected output.
		longdir := filepath.Join(cwd, t.goDirName())
		pkgs, err := goDirPackages(longdir, singlefilepkgs)
		if err != nil {
			t.err = err
			return
		}
		// Split flags into gcflags and ldflags
		ldflags := []string{}
		for i, fl := range flags {
			if fl == "-ldflags" {
				ldflags = flags[i+1:]
				flags = flags[0:i]
				break
			}
		}

		for i, pkg := range pkgs {
			_, err := compileInDir(runcmd, longdir, flags, pkg.name, pkg.files...)
			// Allow this package compilation fail based on conditions below;
			// its errors were checked in previous case.
			if err != nil && !(wantError && action == "errorcheckandrundir" && i == len(pkgs)-2) {
				t.err = err
				return
			}
			if i == len(pkgs)-1 {
				err = linkFile(runcmd, pkg.files[0], ldflags)
				if err != nil {
					t.err = err
					return
				}
				var cmd []string
				cmd = append(cmd, findExecCmd()...)
				cmd = append(cmd, filepath.Join(t.tempDir, "a.exe"))
				cmd = append(cmd, args...)
				out, err := runcmd(cmd...)
				if err != nil {
					t.err = err
					return
				}
				t.checkExpectedOutput(out)
			}
		}

	case "runindir":
		// Make a shallow copy of t.goDirName() in its own module and GOPATH, and
		// run "go run ." in it. The module path (and hence import path prefix) of
		// the copy is equal to the basename of the source directory.
		//
		// It's used when test a requires a full 'go build' in order to compile
		// the sources, such as when importing multiple packages (issue29612.dir)
		// or compiling a package containing assembly files (see issue15609.dir),
		// but still needs to be run to verify the expected output.
		tempDirIsGOPATH = true
		srcDir := t.goDirName()
		modName := filepath.Base(srcDir)
		gopathSrcDir := filepath.Join(t.tempDir, "src", modName)
		runInDir = gopathSrcDir

		if err := overlayDir(gopathSrcDir, srcDir); err != nil {
			t.err = err
			return
		}

		modFile := fmt.Sprintf("module %s\ngo 1.14\n", modName)
		if err := ioutil.WriteFile(filepath.Join(gopathSrcDir, "go.mod"), []byte(modFile), 0666); err != nil {
			t.err = err
			return
		}

		cmd := []string{goTool(), "run", t.goGcflags()}
		if *linkshared {
			cmd = append(cmd, "-linkshared")
		}
		cmd = append(cmd, flags...)
		cmd = append(cmd, ".")
		out, err := runcmd(cmd...)
		if err != nil {
			t.err = err
			return
		}
		t.checkExpectedOutput(out)

	case "build":
		// Build Go file.
		cmd := []string{goTool(), "build", t.goGcflags()}
		cmd = append(cmd, flags...)
		cmd = append(cmd, "-o", "a.exe", long)
		_, err := runcmd(cmd...)
		if err != nil {
			t.err = err
		}

	case "builddir", "buildrundir":
		// Build an executable from all the .go and .s files in a subdirectory.
		// Run it and verify its output in the buildrundir case.
		longdir := filepath.Join(cwd, t.goDirName())
		files, dirErr := ioutil.ReadDir(longdir)
		if dirErr != nil {
			t.err = dirErr
			break
		}
		var gos []string
		var asms []string
		for _, file := range files {
			switch filepath.Ext(file.Name()) {
			case ".go":
				gos = append(gos, filepath.Join(longdir, file.Name()))
			case ".s":
				asms = append(asms, filepath.Join(longdir, file.Name()))
			}

		}
		if len(asms) > 0 {
			emptyHdrFile := filepath.Join(t.tempDir, "go_asm.h")
			if err := ioutil.WriteFile(emptyHdrFile, nil, 0666); err != nil {
				t.err = fmt.Errorf("write empty go_asm.h: %s", err)
				return
			}
			cmd := []string{goTool(), "tool", "asm", "-p=main", "-gensymabis", "-o", "symabis"}
			cmd = append(cmd, asms...)
			_, err = runcmd(cmd...)
			if err != nil {
				t.err = err
				break
			}
		}
		var objs []string
		cmd := []string{goTool(), "tool", "compile", "-p=main", "-e", "-D", ".", "-I", ".", "-o", "go.o"}
		if len(asms) > 0 {
			cmd = append(cmd, "-asmhdr", "go_asm.h", "-symabis", "symabis")
		}
		cmd = append(cmd, gos...)
		_, err := runcmd(cmd...)
		if err != nil {
			t.err = err
			break
		}
		objs = append(objs, "go.o")
		if len(asms) > 0 {
			cmd = []string{goTool(), "tool", "asm", "-p=main", "-e", "-I", ".", "-o", "asm.o"}
			cmd = append(cmd, asms...)
			_, err = runcmd(cmd...)
			if err != nil {
				t.err = err
				break
			}
			objs = append(objs, "asm.o")
		}
		cmd = []string{goTool(), "tool", "pack", "c", "all.a"}
		cmd = append(cmd, objs...)
		_, err = runcmd(cmd...)
		if err != nil {
			t.err = err
			break
		}
		cmd = []string{goTool(), "tool", "link", "-o", "a.exe", "all.a"}
		_, err = runcmd(cmd...)
		if err != nil {
			t.err = err
			break
		}
		if action == "buildrundir" {
			cmd = append(findExecCmd(), filepath.Join(t.tempDir, "a.exe"))
			out, err := runcmd(cmd...)
			if err != nil {
				t.err = err
				break
			}
			t.checkExpectedOutput(out)
		}

	case "buildrun":
		// Build an executable from Go file, then run it, verify its output.
		// Useful for timeout tests where failure mode is infinite loop.
		// TODO: not supported on NaCl
		cmd := []string{goTool(), "build", t.goGcflags(), "-o", "a.exe"}
		if *linkshared {
			cmd = append(cmd, "-linkshared")
		}
		longdirgofile := filepath.Join(filepath.Join(cwd, t.dir), t.gofile)
		cmd = append(cmd, flags...)
		cmd = append(cmd, longdirgofile)
		_, err := runcmd(cmd...)
		if err != nil {
			t.err = err
			return
		}
		cmd = []string{"./a.exe"}
		out, err := runcmd(append(cmd, args...)...)
		if err != nil {
			t.err = err
			return
		}

		t.checkExpectedOutput(out)

	case "run":
		// Run Go file if no special go command flags are provided;
		// otherwise build an executable and run it.
		// Verify the output.
		runInDir = ""
		var out []byte
		var err error
		if len(flags)+len(args) == 0 && t.goGcflagsIsEmpty() && !*linkshared && goarch == runtime.GOARCH && goos == runtime.GOOS && goexp == env.GOEXPERIMENT {
			// If we're not using special go command flags,
			// skip all the go command machinery.
			// This avoids any time the go command would
			// spend checking whether, for example, the installed
			// package runtime is up to date.
			// Because we run lots of trivial test programs,
			// the time adds up.
			pkg := filepath.Join(t.tempDir, "pkg.a")
			if _, err := runcmd(goTool(), "tool", "compile", "-p=main", "-o", pkg, t.goFileName()); err != nil {
				t.err = err
				return
			}
			exe := filepath.Join(t.tempDir, "test.exe")
			cmd := []string{goTool(), "tool", "link", "-s", "-w"}
			cmd = append(cmd, "-o", exe, pkg)
			if _, err := runcmd(cmd...); err != nil {
				t.err = err
				return
			}
			out, err = runcmd(append([]string{exe}, args...)...)
		} else {
			cmd := []string{goTool(), "run", t.goGcflags()}
			if *linkshared {
				cmd = append(cmd, "-linkshared")
			}
			cmd = append(cmd, flags...)
			cmd = append(cmd, t.goFileName())
			out, err = runcmd(append(cmd, args...)...)
		}
		if err != nil {
			t.err = err
			return
		}
		t.checkExpectedOutput(out)

	case "runoutput":
		// Run Go file and write its output into temporary Go file.
		// Run generated Go file and verify its output.
		rungatec <- true
		defer func() {
			<-rungatec
		}()
		runInDir = ""
		cmd := []string{goTool(), "run", t.goGcflags()}
		if *linkshared {
			cmd = append(cmd, "-linkshared")
		}
		cmd = append(cmd, t.goFileName())
		out, err := runcmd(append(cmd, args...)...)
		if err != nil {
			t.err = err
			return
		}
		tfile := filepath.Join(t.tempDir, "tmp__.go")
		if err := ioutil.WriteFile(tfile, out, 0666); err != nil {
			t.err = fmt.Errorf("write tempfile:%s", err)
			return
		}
		cmd = []string{goTool(), "run", t.goGcflags()}
		if *linkshared {
			cmd = append(cmd, "-linkshared")
		}
		cmd = append(cmd, tfile)
		out, err = runcmd(cmd...)
		if err != nil {
			t.err = err
			return
		}
		t.checkExpectedOutput(out)

	case "errorcheckoutput":
		// Run Go file and write its output into temporary Go file.
		// Compile and errorCheck generated Go file.
		runInDir = ""
		cmd := []string{goTool(), "run", t.goGcflags()}
		if *linkshared {
			cmd = append(cmd, "-linkshared")
		}
		cmd = append(cmd, t.goFileName())
		out, err := runcmd(append(cmd, args...)...)
		if err != nil {
			t.err = err
			return
		}
		tfile := filepath.Join(t.tempDir, "tmp__.go")
		err = ioutil.WriteFile(tfile, out, 0666)
		if err != nil {
			t.err = fmt.Errorf("write tempfile:%s", err)
			return
		}
		cmdline := []string{goTool(), "tool", "compile", "-p=p", "-d=panic", "-e", "-o", "a.o"}
		cmdline = append(cmdline, flags...)
		cmdline = append(cmdline, tfile)
		out, err = runcmd(cmdline...)
		if wantError {
			if err == nil {
				t.err = fmt.Errorf("compilation succeeded unexpectedly\n%s", out)
				return
			}
		} else {
			if err != nil {
				t.err = err
				return
			}
		}
		t.err = t.errorCheck(string(out), false, tfile, "tmp__.go")
		return
	}
}

var execCmd []string

func findExecCmd() []string {
	if execCmd != nil {
		return execCmd
	}
	execCmd = []string{} // avoid work the second time
	if goos == runtime.GOOS && goarch == runtime.GOARCH {
		return execCmd
	}
	path, err := exec.LookPath(fmt.Sprintf("go_%s_%s_exec", goos, goarch))
	if err == nil {
		execCmd = []string{path}
	}
	return execCmd
}

func (t *test) String() string {
	return filepath.Join(t.dir, t.gofile)
}

func (t *test) makeTempDir() {
	var err error
	t.tempDir, err = ioutil.TempDir("", "")
	if err != nil {
		log.Fatal(err)
	}
	if *keep {
		log.Printf("Temporary directory is %s", t.tempDir)
	}
	err = os.Mkdir(filepath.Join(t.tempDir, "test"), 0o755)
	if err != nil {
		log.Fatal(err)
	}
}

// checkExpectedOutput compares the output from compiling and/or running with the contents
// of the corresponding reference output file, if any (replace ".go" with ".out").
// If they don't match, fail with an informative message.
func (t *test) checkExpectedOutput(gotBytes []byte) {
	got := string(gotBytes)
	filename := filepath.Join(t.dir, t.gofile)
	filename = filename[:len(filename)-len(".go")]
	filename += ".out"
	b, err := ioutil.ReadFile(filename)
	// File is allowed to be missing (err != nil) in which case output should be empty.
	got = strings.Replace(got, "\r\n", "\n", -1)
	if got != string(b) {
		if err == nil {
			t.err = fmt.Errorf("output does not match expected in %s. Instead saw\n%s", filename, got)
		} else {
			t.err = fmt.Errorf("output should be empty when (optional) expected-output file %s is not present. Instead saw\n%s", filename, got)
		}
	}
}

func splitOutput(out string, wantAuto bool) []string {
	// gc error messages continue onto additional lines with leading tabs.
	// Split the output at the beginning of each line that doesn't begin with a tab.
	// <autogenerated> lines are impossible to match so those are filtered out.
	var res []string
	for _, line := range strings.Split(out, "\n") {
		if strings.HasSuffix(line, "\r") { // remove '\r', output by compiler on windows
			line = line[:len(line)-1]
		}
		if strings.HasPrefix(line, "\t") {
			res[len(res)-1] += "\n" + line
		} else if strings.HasPrefix(line, "go tool") || strings.HasPrefix(line, "#") || !wantAuto && strings.HasPrefix(line, "<autogenerated>") {
			continue
		} else if strings.TrimSpace(line) != "" {
			res = append(res, line)
		}
	}
	return res
}

// errorCheck matches errors in outStr against comments in source files.
// For each line of the source files which should generate an error,
// there should be a comment of the form // ERROR "regexp".
// If outStr has an error for a line which has no such comment,
// this function will report an error.
// Likewise if outStr does not have an error for a line which has a comment,
// or if the error message does not match the <regexp>.
// The <regexp> syntax is Perl but it's best to stick to egrep.
//
// Sources files are supplied as fullshort slice.
// It consists of pairs: full path to source file and its base name.
func (t *test) errorCheck(outStr string, wantAuto bool, fullshort ...string) (err error) {
	defer func() {
		if *verbose && err != nil {
			log.Printf("%s gc output:\n%s", t, outStr)
		}
	}()
	var errs []error
	out := splitOutput(outStr, wantAuto)

	// Cut directory name.
	for i := range out {
		for j := 0; j < len(fullshort); j += 2 {
			full, short := fullshort[j], fullshort[j+1]
			out[i] = strings.Replace(out[i], full, short, -1)
		}
	}

	var want []wantedError
	for j := 0; j < len(fullshort); j += 2 {
		full, short := fullshort[j], fullshort[j+1]
		want = append(want, t.wantedErrors(full, short)...)
	}

	for _, we := range want {
		var errmsgs []string
		if we.auto {
			errmsgs, out = partitionStrings("<autogenerated>", out)
		} else {
			errmsgs, out = partitionStrings(we.prefix, out)
		}
		if len(errmsgs) == 0 {
			errs = append(errs, fmt.Errorf("%s:%d: missing error %q", we.file, we.lineNum, we.reStr))
			continue
		}
		matched := false
		n := len(out)
		for _, errmsg := range errmsgs {
			// Assume errmsg says "file:line: foo".
			// Cut leading "file:line: " to avoid accidental matching of file name instead of message.
			text := errmsg
			if _, suffix, ok := strings.Cut(text, " "); ok {
				text = suffix
			}
			if we.re.MatchString(text) {
				matched = true
			} else {
				out = append(out, errmsg)
			}
		}
		if !matched {
			errs = append(errs, fmt.Errorf("%s:%d: no match for %#q in:\n\t%s", we.file, we.lineNum, we.reStr, strings.Join(out[n:], "\n\t")))
			continue
		}
	}

	if len(out) > 0 {
		errs = append(errs, fmt.Errorf("Unmatched Errors:"))
		for _, errLine := range out {
			errs = append(errs, fmt.Errorf("%s", errLine))
		}
	}

	if len(errs) == 0 {
		return nil
	}
	if len(errs) == 1 {
		return errs[0]
	}
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "\n")
	for _, err := range errs {
		fmt.Fprintf(&buf, "%s\n", err.Error())
	}
	return errors.New(buf.String())
}

func (t *test) updateErrors(out, file string) {
	base := path.Base(file)
	// Read in source file.
	src, err := ioutil.ReadFile(file)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
	lines := strings.Split(string(src), "\n")
	// Remove old errors.
	for i := range lines {
		lines[i], _, _ = strings.Cut(lines[i], " // ERROR ")
	}
	// Parse new errors.
	errors := make(map[int]map[string]bool)
	tmpRe := regexp.MustCompile(`autotmp_[0-9]+`)
	for _, errStr := range splitOutput(out, false) {
		errFile, rest, ok := strings.Cut(errStr, ":")
		if !ok || errFile != file {
			continue
		}
		lineStr, msg, ok := strings.Cut(rest, ":")
		if !ok {
			continue
		}
		line, err := strconv.Atoi(lineStr)
		line--
		if err != nil || line < 0 || line >= len(lines) {
			continue
		}
		msg = strings.Replace(msg, file, base, -1) // normalize file mentions in error itself
		msg = strings.TrimLeft(msg, " \t")
		for _, r := range []string{`\`, `*`, `+`, `?`, `[`, `]`, `(`, `)`} {
			msg = strings.Replace(msg, r, `\`+r, -1)
		}
		msg = strings.Replace(msg, `"`, `.`, -1)
		msg = tmpRe.ReplaceAllLiteralString(msg, `autotmp_[0-9]+`)
		if errors[line] == nil {
			errors[line] = make(map[string]bool)
		}
		errors[line][msg] = true
	}
	// Add new errors.
	for line, errs := range errors {
		var sorted []string
		for e := range errs {
			sorted = append(sorted, e)
		}
		sort.Strings(sorted)
		lines[line] += " // ERROR"
		for _, e := range sorted {
			lines[line] += fmt.Sprintf(` "%s$"`, e)
		}
	}
	// Write new file.
	err = ioutil.WriteFile(file, []byte(strings.Join(lines, "\n")), 0640)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
	// Polish.
	exec.Command(goTool(), "fmt", file).CombinedOutput()
}

// matchPrefix reports whether s is of the form ^(.*/)?prefix(:|[),
// That is, it needs the file name prefix followed by a : or a [,
// and possibly preceded by a directory name.
func matchPrefix(s, prefix string) bool {
	i := strings.Index(s, ":")
	if i < 0 {
		return false
	}
	j := strings.LastIndex(s[:i], "/")
	s = s[j+1:]
	if len(s) <= len(prefix) || s[:len(prefix)] != prefix {
		return false
	}
	switch s[len(prefix)] {
	case '[', ':':
		return true
	}
	return false
}

func partitionStrings(prefix string, strs []string) (matched, unmatched []string) {
	for _, s := range strs {
		if matchPrefix(s, prefix) {
			matched = append(matched, s)
		} else {
			unmatched = append(unmatched, s)
		}
	}
	return
}

type wantedError struct {
	reStr   string
	re      *regexp.Regexp
	lineNum int
	auto    bool // match <autogenerated> line
	file    string
	prefix  string
}

var (
	errRx       = regexp.MustCompile(`// (?:GC_)?ERROR (.*)`)
	errAutoRx   = regexp.MustCompile(`// (?:GC_)?ERRORAUTO (.*)`)
	errQuotesRx = regexp.MustCompile(`"([^"]*)"`)
	lineRx      = regexp.MustCompile(`LINE(([+-])([0-9]+))?`)
)

func (t *test) wantedErrors(file, short string) (errs []wantedError) {
	cache := make(map[string]*regexp.Regexp)

	src, _ := ioutil.ReadFile(file)
	for i, line := range strings.Split(string(src), "\n") {
		lineNum := i + 1
		if strings.Contains(line, "////") {
			// double comment disables ERROR
			continue
		}
		var auto bool
		m := errAutoRx.FindStringSubmatch(line)
		if m != nil {
			auto = true
		} else {
			m = errRx.FindStringSubmatch(line)
		}
		if m == nil {
			continue
		}
		all := m[1]
		mm := errQuotesRx.FindAllStringSubmatch(all, -1)
		if mm == nil {
			log.Fatalf("%s:%d: invalid errchk line: %s", t.goFileName(), lineNum, line)
		}
		for _, m := range mm {
			rx := lineRx.ReplaceAllStringFunc(m[1], func(m string) string {
				n := lineNum
				if strings.HasPrefix(m, "LINE+") {
					delta, _ := strconv.Atoi(m[5:])
					n += delta
				} else if strings.HasPrefix(m, "LINE-") {
					delta, _ := strconv.Atoi(m[5:])
					n -= delta
				}
				return fmt.Sprintf("%s:%d", short, n)
			})
			re := cache[rx]
			if re == nil {
				var err error
				re, err = regexp.Compile(rx)
				if err != nil {
					log.Fatalf("%s:%d: invalid regexp \"%s\" in ERROR line: %v", t.goFileName(), lineNum, rx, err)
				}
				cache[rx] = re
			}
			prefix := fmt.Sprintf("%s:%d", short, lineNum)
			errs = append(errs, wantedError{
				reStr:   rx,
				re:      re,
				prefix:  prefix,
				auto:    auto,
				lineNum: lineNum,
				file:    short,
			})
		}
	}

	return
}

const (
	// Regexp to match a single opcode check: optionally begin with "-" (to indicate
	// a negative check), followed by a string literal enclosed in "" or ``. For "",
	// backslashes must be handled.
	reMatchCheck = `-?(?:\x60[^\x60]*\x60|"(?:[^"\\]|\\.)*")`
)

var (
	// Regexp to split a line in code and comment, trimming spaces
	rxAsmComment = regexp.MustCompile(`^\s*(.*?)\s*(?://\s*(.+)\s*)?$`)

	// Regexp to extract an architecture check: architecture name (or triplet),
	// followed by semi-colon, followed by a comma-separated list of opcode checks.
	// Extraneous spaces are ignored.
	rxAsmPlatform = regexp.MustCompile(`(\w+)(/\w+)?(/\w*)?\s*:\s*(` + reMatchCheck + `(?:\s*,\s*` + reMatchCheck + `)*)`)

	// Regexp to extract a single opcoded check
	rxAsmCheck = regexp.MustCompile(reMatchCheck)

	// List of all architecture variants. Key is the GOARCH architecture,
	// value[0] is the variant-changing environment variable, and values[1:]
	// are the supported variants.
	archVariants = map[string][]string{
		"386":     {"GO386", "sse2", "softfloat"},
		"amd64":   {"GOAMD64", "v1", "v2", "v3", "v4"},
		"arm":     {"GOARM", "5", "6", "7"},
		"arm64":   {},
		"loong64": {},
		"mips":    {"GOMIPS", "hardfloat", "softfloat"},
		"mips64":  {"GOMIPS64", "hardfloat", "softfloat"},
		"ppc64":   {"GOPPC64", "power8", "power9"},
		"ppc64le": {"GOPPC64", "power8", "power9"},
		"s390x":   {},
		"wasm":    {},
		"riscv64": {},
	}
)

// wantedAsmOpcode is a single asmcheck check
type wantedAsmOpcode struct {
	fileline string         // original source file/line (eg: "/path/foo.go:45")
	line     int            // original source line
	opcode   *regexp.Regexp // opcode check to be performed on assembly output
	negative bool           // true if the check is supposed to fail rather than pass
	found    bool           // true if the opcode check matched at least one in the output
}

// A build environment triplet separated by slashes (eg: linux/386/sse2).
// The third field can be empty if the arch does not support variants (eg: "plan9/amd64/")
type buildEnv string

// Environ returns the environment it represents in cmd.Environ() "key=val" format
// For instance, "linux/386/sse2".Environ() returns {"GOOS=linux", "GOARCH=386", "GO386=sse2"}
func (b buildEnv) Environ() []string {
	fields := strings.Split(string(b), "/")
	if len(fields) != 3 {
		panic("invalid buildEnv string: " + string(b))
	}
	env := []string{"GOOS=" + fields[0], "GOARCH=" + fields[1]}
	if fields[2] != "" {
		env = append(env, archVariants[fields[1]][0]+"="+fields[2])
	}
	return env
}

// asmChecks represents all the asmcheck checks present in a test file
// The outer map key is the build triplet in which the checks must be performed.
// The inner map key represent the source file line ("filename.go:1234") at which the
// checks must be performed.
type asmChecks map[buildEnv]map[string][]wantedAsmOpcode

// Envs returns all the buildEnv in which at least one check is present
func (a asmChecks) Envs() []buildEnv {
	var envs []buildEnv
	for e := range a {
		envs = append(envs, e)
	}
	sort.Slice(envs, func(i, j int) bool {
		return string(envs[i]) < string(envs[j])
	})
	return envs
}

func (t *test) wantedAsmOpcodes(fn string) asmChecks {
	ops := make(asmChecks)

	comment := ""
	src, _ := ioutil.ReadFile(fn)
	for i, line := range strings.Split(string(src), "\n") {
		matches := rxAsmComment.FindStringSubmatch(line)
		code, cmt := matches[1], matches[2]

		// Keep comments pending in the comment variable until
		// we find a line that contains some code.
		comment += " " + cmt
		if code == "" {
			continue
		}

		// Parse and extract any architecture check from comments,
		// made by one architecture name and multiple checks.
		lnum := fn + ":" + strconv.Itoa(i+1)
		for _, ac := range rxAsmPlatform.FindAllStringSubmatch(comment, -1) {
			archspec, allchecks := ac[1:4], ac[4]

			var arch, subarch, os string
			switch {
			case archspec[2] != "": // 3 components: "linux/386/sse2"
				os, arch, subarch = archspec[0], archspec[1][1:], archspec[2][1:]
			case archspec[1] != "": // 2 components: "386/sse2"
				os, arch, subarch = "linux", archspec[0], archspec[1][1:]
			default: // 1 component: "386"
				os, arch, subarch = "linux", archspec[0], ""
				if arch == "wasm" {
					os = "js"
				}
			}

			if _, ok := archVariants[arch]; !ok {
				log.Fatalf("%s:%d: unsupported architecture: %v", t.goFileName(), i+1, arch)
			}

			// Create the build environments corresponding the above specifiers
			envs := make([]buildEnv, 0, 4)
			if subarch != "" {
				envs = append(envs, buildEnv(os+"/"+arch+"/"+subarch))
			} else {
				subarchs := archVariants[arch]
				if len(subarchs) == 0 {
					envs = append(envs, buildEnv(os+"/"+arch+"/"))
				} else {
					for _, sa := range archVariants[arch][1:] {
						envs = append(envs, buildEnv(os+"/"+arch+"/"+sa))
					}
				}
			}

			for _, m := range rxAsmCheck.FindAllString(allchecks, -1) {
				negative := false
				if m[0] == '-' {
					negative = true
					m = m[1:]
				}

				rxsrc, err := strconv.Unquote(m)
				if err != nil {
					log.Fatalf("%s:%d: error unquoting string: %v", t.goFileName(), i+1, err)
				}

				// Compile the checks as regular expressions. Notice that we
				// consider checks as matching from the beginning of the actual
				// assembler source (that is, what is left on each line of the
				// compile -S output after we strip file/line info) to avoid
				// trivial bugs such as "ADD" matching "FADD". This
				// doesn't remove genericity: it's still possible to write
				// something like "F?ADD", but we make common cases simpler
				// to get right.
				oprx, err := regexp.Compile("^" + rxsrc)
				if err != nil {
					log.Fatalf("%s:%d: %v", t.goFileName(), i+1, err)
				}

				for _, env := range envs {
					if ops[env] == nil {
						ops[env] = make(map[string][]wantedAsmOpcode)
					}
					ops[env][lnum] = append(ops[env][lnum], wantedAsmOpcode{
						negative: negative,
						fileline: lnum,
						line:     i + 1,
						opcode:   oprx,
					})
				}
			}
		}
		comment = ""
	}

	return ops
}

func (t *test) asmCheck(outStr string, fn string, env buildEnv, fullops map[string][]wantedAsmOpcode) (err error) {
	// The assembly output contains the concatenated dump of multiple functions.
	// the first line of each function begins at column 0, while the rest is
	// indented by a tabulation. These data structures help us index the
	// output by function.
	functionMarkers := make([]int, 1)
	lineFuncMap := make(map[string]int)

	lines := strings.Split(outStr, "\n")
	rxLine := regexp.MustCompile(fmt.Sprintf(`\((%s:\d+)\)\s+(.*)`, regexp.QuoteMeta(fn)))

	for nl, line := range lines {
		// Check if this line begins a function
		if len(line) > 0 && line[0] != '\t' {
			functionMarkers = append(functionMarkers, nl)
		}

		// Search if this line contains a assembly opcode (which is prefixed by the
		// original source file/line in parenthesis)
		matches := rxLine.FindStringSubmatch(line)
		if len(matches) == 0 {
			continue
		}
		srcFileLine, asm := matches[1], matches[2]

		// Associate the original file/line information to the current
		// function in the output; it will be useful to dump it in case
		// of error.
		lineFuncMap[srcFileLine] = len(functionMarkers) - 1

		// If there are opcode checks associated to this source file/line,
		// run the checks.
		if ops, found := fullops[srcFileLine]; found {
			for i := range ops {
				if !ops[i].found && ops[i].opcode.FindString(asm) != "" {
					ops[i].found = true
				}
			}
		}
	}
	functionMarkers = append(functionMarkers, len(lines))

	var failed []wantedAsmOpcode
	for _, ops := range fullops {
		for _, o := range ops {
			// There's a failure if a negative match was found,
			// or a positive match was not found.
			if o.negative == o.found {
				failed = append(failed, o)
			}
		}
	}
	if len(failed) == 0 {
		return
	}

	// At least one asmcheck failed; report them
	sort.Slice(failed, func(i, j int) bool {
		return failed[i].line < failed[j].line
	})

	lastFunction := -1
	var errbuf bytes.Buffer
	fmt.Fprintln(&errbuf)
	for _, o := range failed {
		// Dump the function in which this opcode check was supposed to
		// pass but failed.
		funcIdx := lineFuncMap[o.fileline]
		if funcIdx != 0 && funcIdx != lastFunction {
			funcLines := lines[functionMarkers[funcIdx]:functionMarkers[funcIdx+1]]
			log.Println(strings.Join(funcLines, "\n"))
			lastFunction = funcIdx // avoid printing same function twice
		}

		if o.negative {
			fmt.Fprintf(&errbuf, "%s:%d: %s: wrong opcode found: %q\n", t.goFileName(), o.line, env, o.opcode.String())
		} else {
			fmt.Fprintf(&errbuf, "%s:%d: %s: opcode not found: %q\n", t.goFileName(), o.line, env, o.opcode.String())
		}
	}
	err = errors.New(errbuf.String())
	return
}

// defaultRunOutputLimit returns the number of runoutput tests that
// can be executed in parallel.
func defaultRunOutputLimit() int {
	const maxArmCPU = 2

	cpu := runtime.NumCPU()
	if runtime.GOARCH == "arm" && cpu > maxArmCPU {
		cpu = maxArmCPU
	}
	return cpu
}

// checkShouldTest runs sanity checks on the shouldTest function.
func checkShouldTest() {
	assert := func(ok bool, _ string) {
		if !ok {
			panic("fail")
		}
	}
	assertNot := func(ok bool, _ string) { assert(!ok, "") }

	// Simple tests.
	assert(shouldTest("// +build linux", "linux", "arm"))
	assert(shouldTest("// +build !windows", "linux", "arm"))
	assertNot(shouldTest("// +build !windows", "windows", "amd64"))

	// A file with no build tags will always be tested.
	assert(shouldTest("// This is a test.", "os", "arch"))

	// Build tags separated by a space are OR-ed together.
	assertNot(shouldTest("// +build arm 386", "linux", "amd64"))

	// Build tags separated by a comma are AND-ed together.
	assertNot(shouldTest("// +build !windows,!plan9", "windows", "amd64"))
	assertNot(shouldTest("// +build !windows,!plan9", "plan9", "386"))

	// Build tags on multiple lines are AND-ed together.
	assert(shouldTest("// +build !windows\n// +build amd64", "linux", "amd64"))
	assertNot(shouldTest("// +build !windows\n// +build amd64", "windows", "amd64"))

	// Test that (!a OR !b) matches anything.
	assert(shouldTest("// +build !windows !plan9", "windows", "amd64"))
}

// overlayDir makes a minimal-overhead copy of srcRoot in which new files may be added.
func overlayDir(dstRoot, srcRoot string) error {
	dstRoot = filepath.Clean(dstRoot)
	if err := os.MkdirAll(dstRoot, 0777); err != nil {
		return err
	}

	srcRoot, err := filepath.Abs(srcRoot)
	if err != nil {
		return err
	}

	return filepath.WalkDir(srcRoot, func(srcPath string, d fs.DirEntry, err error) error {
		if err != nil || srcPath == srcRoot {
			return err
		}

		suffix := strings.TrimPrefix(srcPath, srcRoot)
		for len(suffix) > 0 && suffix[0] == filepath.Separator {
			suffix = suffix[1:]
		}
		dstPath := filepath.Join(dstRoot, suffix)

		var info fs.FileInfo
		if d.Type()&os.ModeSymlink != 0 {
			info, err = os.Stat(srcPath)
		} else {
			info, err = d.Info()
		}
		if err != nil {
			return err
		}
		perm := info.Mode() & os.ModePerm

		// Always copy directories (don't symlink them).
		// If we add a file in the overlay, we don't want to add it in the original.
		if info.IsDir() {
			return os.MkdirAll(dstPath, perm|0200)
		}

		// If the OS supports symlinks, use them instead of copying bytes.
		if err := os.Symlink(srcPath, dstPath); err == nil {
			return nil
		}

		// Otherwise, copy the bytes.
		src, err := os.Open(srcPath)
		if err != nil {
			return err
		}
		defer src.Close()

		dst, err := os.OpenFile(dstPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, perm)
		if err != nil {
			return err
		}

		_, err = io.Copy(dst, src)
		if closeErr := dst.Close(); err == nil {
			err = closeErr
		}
		return err
	})
}

// The following sets of files are excluded from testing depending on configuration.
// The types2Failures(32Bit) files pass with the 1.17 compiler but don't pass with
// the 1.18 compiler using the new types2 type checker, or pass with sub-optimal
// error(s).

// List of files that the compiler cannot errorcheck with the new typechecker (types2).
var types2Failures = setOf(
	"notinheap.go",            // types2 doesn't report errors about conversions that are invalid due to //go:notinheap
	"shift1.go",               // types2 reports two new errors which are probably not right
	"fixedbugs/issue10700.go", // types2 should give hint about ptr to interface
	"fixedbugs/issue18331.go", // missing error about misuse of //go:noescape (irgen needs code from noder)
	"fixedbugs/issue18419.go", // types2 reports no field or method member, but should say unexported
	"fixedbugs/issue20233.go", // types2 reports two instead of one error (preference: 1.17 compiler)
	"fixedbugs/issue20245.go", // types2 reports two instead of one error (preference: 1.17 compiler)
	"fixedbugs/issue31053.go", // types2 reports "unknown field" instead of "cannot refer to unexported field"
)

var types2Failures32Bit = setOf(
	"printbig.go",             // large untyped int passed to print (32-bit)
	"fixedbugs/bug114.go",     // large untyped int passed to println (32-bit)
	"fixedbugs/issue23305.go", // large untyped int passed to println (32-bit)
)

var go118Failures = setOf(
	"fixedbugs/issue54343.go",  // 1.18 compiler assigns receiver parameter to global variable
	"typeparam/nested.go",      // 1.18 compiler doesn't support function-local types with generics
	"typeparam/issue51521.go",  // 1.18 compiler produces bad panic message and link error
	"typeparam/issue54456.go",  // 1.18 compiler fails to distinguish local generic types
	"typeparam/issue54497.go",  // 1.18 compiler is more conservative about inlining due to repeated issues
	"typeparam/mdempsky/16.go", // 1.18 compiler uses interface shape type in failed type assertions
	"typeparam/mdempsky/17.go", // 1.18 compiler mishandles implicit conversions from range loops
	"typeparam/mdempsky/18.go", // 1.18 compiler mishandles implicit conversions in select statements
	"typeparam/mdempsky/20.go", // 1.18 compiler crashes on method expressions promoted to derived types
)

// In all of these cases, the 1.17 compiler reports reasonable errors, but either the
// 1.17 or 1.18 compiler report extra errors, so we can't match correctly on both. We
// now set the patterns to match correctly on all the 1.18 errors.
// This list remains here just as a reference and for comparison - these files all pass.
var _ = setOf(
	"import1.go",      // types2 reports extra errors
	"initializerr.go", // types2 reports extra error
	"typecheck.go",    // types2 reports extra error at function call

	"fixedbugs/bug176.go", // types2 reports all errors (pref: types2)
	"fixedbugs/bug195.go", // types2 reports slight different errors, and an extra error
	"fixedbugs/bug412.go", // types2 produces a follow-on error

	"fixedbugs/issue11614.go", // types2 reports an extra error
	"fixedbugs/issue17038.go", // types2 doesn't report a follow-on error (pref: types2)
	"fixedbugs/issue23732.go", // types2 reports different (but ok) line numbers
	"fixedbugs/issue4510.go",  // types2 reports different (but ok) line numbers
	"fixedbugs/issue7525b.go", // types2 reports init cycle error on different line - ok otherwise
	"fixedbugs/issue7525c.go", // types2 reports init cycle error on different line - ok otherwise
	"fixedbugs/issue7525d.go", // types2 reports init cycle error on different line - ok otherwise
	"fixedbugs/issue7525e.go", // types2 reports init cycle error on different line - ok otherwise
	"fixedbugs/issue7525.go",  // types2 reports init cycle error on different line - ok otherwise
)

var unifiedFailures = setOf(
	"closure3.go", // unified IR numbers closures differently than -d=inlfuncswithclosures
	"escape4.go",  // unified IR can inline f5 and f6; test doesn't expect this

	"typeparam/issue47631.go", // unified IR can handle local type declarations
)

func setOf(keys ...string) map[string]bool {
	m := make(map[string]bool, len(keys))
	for _, key := range keys {
		m[key] = true
	}
	return m
}

// splitQuoted splits the string s around each instance of one or more consecutive
// white space characters while taking into account quotes and escaping, and
// returns an array of substrings of s or an empty list if s contains only white space.
// Single quotes and double quotes are recognized to prevent splitting within the
// quoted region, and are removed from the resulting substrings. If a quote in s
// isn't closed err will be set and r will have the unclosed argument as the
// last element. The backslash is used for escaping.
//
// For example, the following string:
//
//	a b:"c d" 'e''f'  "g\""
//
// Would be parsed as:
//
//	[]string{"a", "b:c d", "ef", `g"`}
//
// [copied from src/go/build/build.go]
func splitQuoted(s string) (r []string, err error) {
	var args []string
	arg := make([]rune, len(s))
	escaped := false
	quoted := false
	quote := '\x00'
	i := 0
	for _, rune := range s {
		switch {
		case escaped:
			escaped = false
		case rune == '\\':
			escaped = true
			continue
		case quote != '\x00':
			if rune == quote {
				quote = '\x00'
				continue
			}
		case rune == '"' || rune == '\'':
			quoted = true
			quote = rune
			continue
		case unicode.IsSpace(rune):
			if quoted || i > 0 {
				quoted = false
				args = append(args, string(arg[:i]))
				i = 0
			}
			continue
		}
		arg[i] = rune
		i++
	}
	if quoted || i > 0 {
		args = append(args, string(arg[:i]))
	}
	if quote != 0 {
		err = errors.New("unclosed quote")
	} else if escaped {
		err = errors.New("unfinished escaping")
	}
	return args, err
}
