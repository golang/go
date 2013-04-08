// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"text/template"
	"time"
	"unicode"
	"unicode/utf8"
)

// Break init loop.
func init() {
	cmdTest.Run = runTest
}

var cmdTest = &Command{
	CustomFlags: true,
	UsageLine:   "test [-c] [-i] [build flags] [packages] [flags for test binary]",
	Short:       "test packages",
	Long: `
'Go test' automates testing the packages named by the import paths.
It prints a summary of the test results in the format:

	ok   archive/tar   0.011s
	FAIL archive/zip   0.022s
	ok   compress/gzip 0.033s
	...

followed by detailed output for each failed package.

'Go test' recompiles each package along with any files with names matching
the file pattern "*_test.go".  These additional files can contain test functions,
benchmark functions, and example functions.  See 'go help testfunc' for more.
Each listed package causes the execution of a separate test binary.

Test files that declare a package with the suffix "_test" will be compiled as a
separate package, and then linked and run with the main test binary.

By default, go test needs no arguments.  It compiles and tests the package
with source in the current directory, including tests, and runs the tests.

The package is built in a temporary directory so it does not interfere with the
non-test installation.

In addition to the build flags, the flags handled by 'go test' itself are:

	-c  Compile the test binary to pkg.test but do not run it.
	    (Where pkg is the last element of the package's import path.)

	-i
	    Install packages that are dependencies of the test.
	    Do not run the test.

The test binary also accepts flags that control execution of the test; these
flags are also accessible by 'go test'.  See 'go help testflag' for details.

For more about build flags, see 'go help build'.
For more about specifying packages, see 'go help packages'.

See also: go build, go vet.
`,
}

var helpTestflag = &Command{
	UsageLine: "testflag",
	Short:     "description of testing flags",
	Long: `
The 'go test' command takes both flags that apply to 'go test' itself
and flags that apply to the resulting test binary.

Several of the flags control profiling and write an execution profile
suitable for "go tool pprof"; run "go tool pprof help" for more
information.  The --alloc_space, --alloc_objects, and --show_bytes
options of pprof control how the information is presented.

The following flags are recognized by the 'go test' command and
control the execution of any test:

	-bench regexp
	    Run benchmarks matching the regular expression.
	    By default, no benchmarks run. To run all benchmarks,
	    use '-bench .' or '-bench=.'.

	-benchmem
	    Print memory allocation statistics for benchmarks.

	-benchtime t
	    Run enough iterations of each benchmark to take t, specified
	    as a time.Duration (for example, -benchtime 1h30s).
	    The default is 1 second (1s).

	-blockprofile block.out
	    Write a goroutine blocking profile to the specified file
	    when all tests are complete.

	-blockprofilerate n
	    Control the detail provided in goroutine blocking profiles by
	    calling runtime.SetBlockProfileRate with n.
	    See 'godoc runtime SetBlockProfileRate'.
	    The profiler aims to sample, on average, one blocking event every
	    n nanoseconds the program spends blocked.  By default,
	    if -test.blockprofile is set without this flag, all blocking events
	    are recorded, equivalent to -test.blockprofilerate=1.

	-cpu 1,2,4
	    Specify a list of GOMAXPROCS values for which the tests or
	    benchmarks should be executed.  The default is the current value
	    of GOMAXPROCS.

	-cpuprofile cpu.out
	    Write a CPU profile to the specified file before exiting.

	-memprofile mem.out
	    Write a memory profile to the specified file after all tests
	    have passed.

	-memprofilerate n
	    Enable more precise (and expensive) memory profiles by setting
	    runtime.MemProfileRate.  See 'godoc runtime MemProfileRate'.
	    To profile all memory allocations, use -test.memprofilerate=1
	    and set the environment variable GOGC=off to disable the
	    garbage collector, provided the test can run in the available
	    memory without garbage collection.

	-parallel n
	    Allow parallel execution of test functions that call t.Parallel.
	    The value of this flag is the maximum number of tests to run
	    simultaneously; by default, it is set to the value of GOMAXPROCS.

	-run regexp
	    Run only those tests and examples matching the regular
	    expression.

	-short
	    Tell long-running tests to shorten their run time.
	    It is off by default but set during all.bash so that installing
	    the Go tree can run a sanity check but not spend time running
	    exhaustive tests.

	-timeout t
	    If a test runs longer than t, panic.

	-v
	    Verbose output: log all tests as they are run. Also print all
	    text from Log and Logf calls even if the test succeeds.

The test binary, called pkg.test where pkg is the name of the
directory containing the package sources, can be invoked directly
after building it with 'go test -c'. When invoking the test binary
directly, each of the standard flag names must be prefixed with 'test.',
as in -test.run=TestMyFunc or -test.v.

When running 'go test', flags not listed above are passed through
unaltered. For instance, the command

	go test -x -v -cpuprofile=prof.out -dir=testdata -update

will compile the test binary and then run it as

	pkg.test -test.v -test.cpuprofile=prof.out -dir=testdata -update

The test flags that generate profiles also leave the test binary in pkg.test
for use when analyzing the profiles.

Flags not recognized by 'go test' must be placed after any specified packages.
`,
}

var helpTestfunc = &Command{
	UsageLine: "testfunc",
	Short:     "description of testing functions",
	Long: `
The 'go test' command expects to find test, benchmark, and example functions
in the "*_test.go" files corresponding to the package under test.

A test function is one named TestXXX (where XXX is any alphanumeric string
not starting with a lower case letter) and should have the signature,

	func TestXXX(t *testing.T) { ... }

A benchmark function is one named BenchmarkXXX and should have the signature,

	func BenchmarkXXX(b *testing.B) { ... }

An example function is similar to a test function but, instead of using
*testing.T to report success or failure, prints output to os.Stdout.
That output is compared against the function's "Output:" comment, which
must be the last comment in the function body (see example below). An
example with no such comment, or with no text after "Output:" is compiled
but not executed.

Godoc displays the body of ExampleXXX to demonstrate the use
of the function, constant, or variable XXX.  An example of a method M with
receiver type T or *T is named ExampleT_M.  There may be multiple examples
for a given function, constant, or variable, distinguished by a trailing _xxx,
where xxx is a suffix not beginning with an upper case letter.

Here is an example of an example:

	func ExamplePrintln() {
		Println("The output of\nthis example.")
		// Output: The output of
		// this example.
	}

The entire test file is presented as the example when it contains a single
example function, at least one other function, type, variable, or constant
declaration, and no test or benchmark functions.

See the documentation of the testing package for more information.
`,
}

var (
	testC            bool     // -c flag
	testProfile      bool     // some profiling flag
	testI            bool     // -i flag
	testV            bool     // -v flag
	testFiles        []string // -file flag(s)  TODO: not respected
	testTimeout      string   // -timeout flag
	testArgs         []string
	testBench        bool
	testStreamOutput bool // show output as it is generated
	testShowPass     bool // show passing output

	testKillTimeout = 10 * time.Minute
)

func runTest(cmd *Command, args []string) {
	var pkgArgs []string
	pkgArgs, testArgs = testFlags(args)

	raceInit()
	pkgs := packagesForBuild(pkgArgs)
	if len(pkgs) == 0 {
		fatalf("no packages to test")
	}

	if testC && len(pkgs) != 1 {
		fatalf("cannot use -c flag with multiple packages")
	}
	if testProfile && len(pkgs) != 1 {
		fatalf("cannot use test profile flag with multiple packages")
	}

	// If a test timeout was given and is parseable, set our kill timeout
	// to that timeout plus one minute.  This is a backup alarm in case
	// the test wedges with a goroutine spinning and its background
	// timer does not get a chance to fire.
	if dt, err := time.ParseDuration(testTimeout); err == nil && dt > 0 {
		testKillTimeout = dt + 1*time.Minute
	}

	// show passing test output (after buffering) with -v flag.
	// must buffer because tests are running in parallel, and
	// otherwise the output will get mixed.
	testShowPass = testV

	// stream test output (no buffering) when no package has
	// been given on the command line (implicit current directory)
	// or when benchmarking.
	// Also stream if we're showing output anyway with a
	// single package under test.  In that case, streaming the
	// output produces the same result as not streaming,
	// just more immediately.
	testStreamOutput = len(pkgArgs) == 0 || testBench ||
		(len(pkgs) <= 1 && testShowPass)

	var b builder
	b.init()

	if testI {
		buildV = testV

		deps := map[string]bool{
			// Dependencies for testmain.
			"testing": true,
			"regexp":  true,
		}
		for _, p := range pkgs {
			// Dependencies for each test.
			for _, path := range p.Imports {
				deps[path] = true
			}
			for _, path := range p.TestImports {
				deps[path] = true
			}
			for _, path := range p.XTestImports {
				deps[path] = true
			}
		}

		// translate C to runtime/cgo
		if deps["C"] {
			delete(deps, "C")
			deps["runtime/cgo"] = true
			if buildContext.GOOS == runtime.GOOS && buildContext.GOARCH == runtime.GOARCH {
				deps["cmd/cgo"] = true
			}
		}
		// Ignore pseudo-packages.
		delete(deps, "unsafe")

		all := []string{}
		for path := range deps {
			if !build.IsLocalImport(path) {
				all = append(all, path)
			}
		}
		sort.Strings(all)

		a := &action{}
		for _, p := range packagesForBuild(all) {
			a.deps = append(a.deps, b.action(modeInstall, modeInstall, p))
		}
		b.do(a)
		if !testC {
			return
		}
		b.init()
	}

	var builds, runs, prints []*action

	// Prepare build + run + print actions for all packages being tested.
	for _, p := range pkgs {
		buildTest, runTest, printTest, err := b.test(p)
		if err != nil {
			str := err.Error()
			if strings.HasPrefix(str, "\n") {
				str = str[1:]
			}
			if p.ImportPath != "" {
				errorf("# %s\n%s", p.ImportPath, str)
			} else {
				errorf("%s", str)
			}
			continue
		}
		builds = append(builds, buildTest)
		runs = append(runs, runTest)
		prints = append(prints, printTest)
	}

	// Ultimately the goal is to print the output.
	root := &action{deps: prints}

	// Force the printing of results to happen in order,
	// one at a time.
	for i, a := range prints {
		if i > 0 {
			a.deps = append(a.deps, prints[i-1])
		}
	}

	// If we are benchmarking, force everything to
	// happen in serial.  Could instead allow all the
	// builds to run before any benchmarks start,
	// but try this for now.
	if testBench {
		for i, a := range builds {
			if i > 0 {
				// Make build of test i depend on
				// completing the run of test i-1.
				a.deps = append(a.deps, runs[i-1])
			}
		}
	}

	// If we are building any out-of-date packages other
	// than those under test, warn.
	okBuild := map[*Package]bool{}
	for _, p := range pkgs {
		okBuild[p] = true
	}

	warned := false
	for _, a := range actionList(root) {
		if a.p != nil && a.f != nil && !okBuild[a.p] && !a.p.fake && !a.p.local {
			okBuild[a.p] = true // don't warn again
			if !warned {
				fmt.Fprintf(os.Stderr, "warning: building out-of-date packages:\n")
				warned = true
			}
			fmt.Fprintf(os.Stderr, "\t%s\n", a.p.ImportPath)
		}
	}
	if warned {
		args := strings.Join(pkgArgs, " ")
		if args != "" {
			args = " " + args
		}
		extraOpts := ""
		if buildRace {
			extraOpts = "-race "
		}
		fmt.Fprintf(os.Stderr, "installing these packages with 'go test %s-i%s' will speed future tests.\n\n", extraOpts, args)
	}

	b.do(root)
}

func (b *builder) test(p *Package) (buildAction, runAction, printAction *action, err error) {
	if len(p.TestGoFiles)+len(p.XTestGoFiles) == 0 {
		build := &action{p: p}
		run := &action{p: p}
		print := &action{f: (*builder).notest, p: p, deps: []*action{build}}
		return build, run, print, nil
	}

	// Build Package structs describing:
	//	ptest - package + test files
	//	pxtest - package of external test files
	//	pmain - pkg.test binary
	var ptest, pxtest, pmain *Package

	var imports, ximports []*Package
	var stk importStack
	stk.push(p.ImportPath + "_test")
	for _, path := range p.TestImports {
		p1 := loadImport(path, p.Dir, &stk, p.build.TestImportPos[path])
		if p1.Error != nil {
			return nil, nil, nil, p1.Error
		}
		imports = append(imports, p1)
	}
	for _, path := range p.XTestImports {
		if path == p.ImportPath {
			continue
		}
		p1 := loadImport(path, p.Dir, &stk, p.build.XTestImportPos[path])
		if p1.Error != nil {
			return nil, nil, nil, p1.Error
		}
		ximports = append(ximports, p1)
	}
	stk.pop()

	// Use last element of import path, not package name.
	// They differ when package name is "main".
	// But if the import path is "command-line-arguments",
	// like it is during 'go run', use the package name.
	var elem string
	if p.ImportPath == "command-line-arguments" {
		elem = p.Name
	} else {
		_, elem = path.Split(p.ImportPath)
	}
	testBinary := elem + ".test"

	// The ptest package needs to be importable under the
	// same import path that p has, but we cannot put it in
	// the usual place in the temporary tree, because then
	// other tests will see it as the real package.
	// Instead we make a _test directory under the import path
	// and then repeat the import path there.  We tell the
	// compiler and linker to look in that _test directory first.
	//
	// That is, if the package under test is unicode/utf8,
	// then the normal place to write the package archive is
	// $WORK/unicode/utf8.a, but we write the test package archive to
	// $WORK/unicode/utf8/_test/unicode/utf8.a.
	// We write the external test package archive to
	// $WORK/unicode/utf8/_test/unicode/utf8_test.a.
	testDir := filepath.Join(b.work, filepath.FromSlash(p.ImportPath+"/_test"))
	ptestObj := buildToolchain.pkgpath(testDir, p)

	// Create the directory for the .a files.
	ptestDir, _ := filepath.Split(ptestObj)
	if err := b.mkdir(ptestDir); err != nil {
		return nil, nil, nil, err
	}
	if err := writeTestmain(filepath.Join(testDir, "_testmain.go"), p); err != nil {
		return nil, nil, nil, err
	}

	// Test package.
	if len(p.TestGoFiles) > 0 {
		ptest = new(Package)
		*ptest = *p
		ptest.GoFiles = nil
		ptest.GoFiles = append(ptest.GoFiles, p.GoFiles...)
		ptest.GoFiles = append(ptest.GoFiles, p.TestGoFiles...)
		ptest.target = ""
		ptest.Imports = stringList(p.Imports, p.TestImports)
		ptest.imports = append(append([]*Package{}, p.imports...), imports...)
		ptest.pkgdir = testDir
		ptest.fake = true
		ptest.forceLibrary = true
		ptest.Stale = true
		ptest.build = new(build.Package)
		*ptest.build = *p.build
		m := map[string][]token.Position{}
		for k, v := range p.build.ImportPos {
			m[k] = append(m[k], v...)
		}
		for k, v := range p.build.TestImportPos {
			m[k] = append(m[k], v...)
		}
		ptest.build.ImportPos = m
	} else {
		ptest = p
	}

	// External test package.
	if len(p.XTestGoFiles) > 0 {
		pxtest = &Package{
			Name:        p.Name + "_test",
			ImportPath:  p.ImportPath + "_test",
			localPrefix: p.localPrefix,
			Root:        p.Root,
			Dir:         p.Dir,
			GoFiles:     p.XTestGoFiles,
			Imports:     p.XTestImports,
			build: &build.Package{
				ImportPos: p.build.XTestImportPos,
			},
			imports: append(ximports, ptest),
			pkgdir:  testDir,
			fake:    true,
			Stale:   true,
		}
	}

	// Action for building pkg.test.
	pmain = &Package{
		Name:       "main",
		Dir:        testDir,
		GoFiles:    []string{"_testmain.go"},
		ImportPath: "testmain",
		Root:       p.Root,
		imports:    []*Package{ptest},
		build:      &build.Package{Name: "main"},
		fake:       true,
		Stale:      true,
	}
	if pxtest != nil {
		pmain.imports = append(pmain.imports, pxtest)
	}

	// The generated main also imports testing and regexp.
	stk.push("testmain")
	ptesting := loadImport("testing", "", &stk, nil)
	if ptesting.Error != nil {
		return nil, nil, nil, ptesting.Error
	}
	pregexp := loadImport("regexp", "", &stk, nil)
	if pregexp.Error != nil {
		return nil, nil, nil, pregexp.Error
	}
	pmain.imports = append(pmain.imports, ptesting, pregexp)
	computeStale(pmain)

	if ptest != p {
		a := b.action(modeBuild, modeBuild, ptest)
		a.objdir = testDir + string(filepath.Separator)
		a.objpkg = ptestObj
		a.target = ptestObj
		a.link = false
	}

	if pxtest != nil {
		a := b.action(modeBuild, modeBuild, pxtest)
		a.objdir = testDir + string(filepath.Separator)
		a.objpkg = buildToolchain.pkgpath(testDir, pxtest)
		a.target = a.objpkg
	}

	a := b.action(modeBuild, modeBuild, pmain)
	a.objdir = testDir + string(filepath.Separator)
	a.objpkg = filepath.Join(testDir, "main.a")
	a.target = filepath.Join(testDir, testBinary) + exeSuffix
	pmainAction := a

	if testC || testProfile {
		// -c or profiling flag: create action to copy binary to ./test.out.
		runAction = &action{
			f:      (*builder).install,
			deps:   []*action{pmainAction},
			p:      pmain,
			target: filepath.Join(cwd, testBinary+exeSuffix),
		}
		pmainAction = runAction // in case we are running the test
	}
	if testC {
		printAction = &action{p: p, deps: []*action{runAction}} // nop
	} else {
		// run test
		runAction = &action{
			f:          (*builder).runTest,
			deps:       []*action{pmainAction},
			p:          p,
			ignoreFail: true,
		}
		cleanAction := &action{
			f:    (*builder).cleanTest,
			deps: []*action{runAction},
			p:    p,
		}
		printAction = &action{
			f:    (*builder).printTest,
			deps: []*action{cleanAction},
			p:    p,
		}
	}

	return pmainAction, runAction, printAction, nil
}

// runTest is the action for running a test binary.
func (b *builder) runTest(a *action) error {
	args := stringList(a.deps[0].target, testArgs)
	a.testOutput = new(bytes.Buffer)

	if buildN || buildX {
		b.showcmd("", "%s", strings.Join(args, " "))
		if buildN {
			return nil
		}
	}

	if a.failed {
		// We were unable to build the binary.
		a.failed = false
		fmt.Fprintf(a.testOutput, "FAIL\t%s [build failed]\n", a.p.ImportPath)
		setExitStatus(1)
		return nil
	}

	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = a.p.Dir
	cmd.Env = envForDir(cmd.Dir)
	var buf bytes.Buffer
	if testStreamOutput {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	} else {
		cmd.Stdout = &buf
		cmd.Stderr = &buf
	}

	// If there are any local SWIG dependencies, we want to load
	// the shared library from the build directory.
	if a.p.usesSwig() {
		env := cmd.Env
		found := false
		prefix := "LD_LIBRARY_PATH="
		for i, v := range env {
			if strings.HasPrefix(v, prefix) {
				env[i] = v + ":."
				found = true
				break
			}
		}
		if !found {
			env = append(env, "LD_LIBRARY_PATH=.")
		}
		cmd.Env = env
	}

	t0 := time.Now()
	err := cmd.Start()

	// This is a last-ditch deadline to detect and
	// stop wedged test binaries, to keep the builders
	// running.
	if err == nil {
		tick := time.NewTimer(testKillTimeout)
		startSigHandlers()
		done := make(chan error)
		go func() {
			done <- cmd.Wait()
		}()
		select {
		case err = <-done:
			// ok
		case <-tick.C:
			cmd.Process.Kill()
			err = <-done
			fmt.Fprintf(&buf, "*** Test killed: ran too long (%v).\n", testKillTimeout)
		}
		tick.Stop()
	}
	out := buf.Bytes()
	t := fmt.Sprintf("%.3fs", time.Since(t0).Seconds())
	if err == nil {
		if testShowPass {
			a.testOutput.Write(out)
		}
		fmt.Fprintf(a.testOutput, "ok  \t%s\t%s\n", a.p.ImportPath, t)
		return nil
	}

	setExitStatus(1)
	if len(out) > 0 {
		a.testOutput.Write(out)
		// assume printing the test binary's exit status is superfluous
	} else {
		fmt.Fprintf(a.testOutput, "%s\n", err)
	}
	fmt.Fprintf(a.testOutput, "FAIL\t%s\t%s\n", a.p.ImportPath, t)

	return nil
}

// cleanTest is the action for cleaning up after a test.
func (b *builder) cleanTest(a *action) error {
	if buildWork {
		return nil
	}
	run := a.deps[0]
	testDir := filepath.Join(b.work, filepath.FromSlash(run.p.ImportPath+"/_test"))
	os.RemoveAll(testDir)
	return nil
}

// printTest is the action for printing a test result.
func (b *builder) printTest(a *action) error {
	clean := a.deps[0]
	run := clean.deps[0]
	os.Stdout.Write(run.testOutput.Bytes())
	run.testOutput = nil
	return nil
}

// notest is the action for testing a package with no test files.
func (b *builder) notest(a *action) error {
	fmt.Printf("?   \t%s\t[no test files]\n", a.p.ImportPath)
	return nil
}

// isTest tells whether name looks like a test (or benchmark, according to prefix).
// It is a Test (say) if there is a character after Test that is not a lower-case letter.
// We don't want TesticularCancer.
func isTest(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	if len(name) == len(prefix) { // "Test" is ok
		return true
	}
	rune, _ := utf8.DecodeRuneInString(name[len(prefix):])
	return !unicode.IsLower(rune)
}

// writeTestmain writes the _testmain.go file for package p to
// the file named out.
func writeTestmain(out string, p *Package) error {
	t := &testFuncs{
		Package: p,
	}
	for _, file := range p.TestGoFiles {
		if err := t.load(filepath.Join(p.Dir, file), "_test", &t.NeedTest); err != nil {
			return err
		}
	}
	for _, file := range p.XTestGoFiles {
		if err := t.load(filepath.Join(p.Dir, file), "_xtest", &t.NeedXtest); err != nil {
			return err
		}
	}

	f, err := os.Create(out)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := testmainTmpl.Execute(f, t); err != nil {
		return err
	}

	return nil
}

type testFuncs struct {
	Tests      []testFunc
	Benchmarks []testFunc
	Examples   []testFunc
	Package    *Package
	NeedTest   bool
	NeedXtest  bool
}

type testFunc struct {
	Package string // imported package name (_test or _xtest)
	Name    string // function name
	Output  string // output, for examples
}

var testFileSet = token.NewFileSet()

func (t *testFuncs) load(filename, pkg string, seen *bool) error {
	f, err := parser.ParseFile(testFileSet, filename, nil, parser.ParseComments)
	if err != nil {
		return expandScanner(err)
	}
	for _, d := range f.Decls {
		n, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if n.Recv != nil {
			continue
		}
		name := n.Name.String()
		switch {
		case isTest(name, "Test"):
			t.Tests = append(t.Tests, testFunc{pkg, name, ""})
			*seen = true
		case isTest(name, "Benchmark"):
			t.Benchmarks = append(t.Benchmarks, testFunc{pkg, name, ""})
			*seen = true
		}
	}
	ex := doc.Examples(f)
	sort.Sort(byOrder(ex))
	for _, e := range ex {
		if e.Output == "" && !e.EmptyOutput {
			// Don't run examples with no output.
			continue
		}
		t.Examples = append(t.Examples, testFunc{pkg, "Example" + e.Name, e.Output})
		*seen = true
	}
	return nil
}

type byOrder []*doc.Example

func (x byOrder) Len() int           { return len(x) }
func (x byOrder) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x byOrder) Less(i, j int) bool { return x[i].Order < x[j].Order }

var testmainTmpl = template.Must(template.New("main").Parse(`
package main

import (
	"regexp"
	"testing"

{{if .NeedTest}}
	_test {{.Package.ImportPath | printf "%q"}}
{{end}}
{{if .NeedXtest}}
	_xtest {{.Package.ImportPath | printf "%s_test" | printf "%q"}}
{{end}}
)

var tests = []testing.InternalTest{
{{range .Tests}}
	{"{{.Name}}", {{.Package}}.{{.Name}}},
{{end}}
}

var benchmarks = []testing.InternalBenchmark{
{{range .Benchmarks}}
	{"{{.Name}}", {{.Package}}.{{.Name}}},
{{end}}
}

var examples = []testing.InternalExample{
{{range .Examples}}
	{"{{.Name}}", {{.Package}}.{{.Name}}, {{.Output | printf "%q"}}},
{{end}}
}

var matchPat string
var matchRe *regexp.Regexp

func matchString(pat, str string) (result bool, err error) {
	if matchRe == nil || matchPat != pat {
		matchPat = pat
		matchRe, err = regexp.Compile(matchPat)
		if err != nil {
			return
		}
	}
	return matchRe.MatchString(str), nil
}

func main() {
	testing.Main(matchString, tests, benchmarks, examples)
}

`))
