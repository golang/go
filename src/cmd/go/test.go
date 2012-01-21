// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
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
	UsageLine:   "test [-c] [-x] [-file a.go -file b.go ...] [-p n] [importpath...] [flags for test binary]",
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

By default, go test needs no arguments.  It compiles and tests the package
with source in the current directory, including tests, and runs the tests.
If file names are given (with flag -file=test.go, one per extra test source file),
only those test files are added to the package.  (The non-test files are always
compiled.)

The package is built in a temporary directory so it does not interfere with the
non-test installation.

See 'go help testflag' for details about flags handled by 'go test'
and the test binary.

See 'go help importpath' for more about import paths.

See also: go build, go vet.
	`,
}

var helpTestflag = &Command{
	UsageLine: "testflag",
	Short:     "description of testing flags",
	Long: `
The 'go test' command takes both flags that apply to 'go test' itself
and flags that apply to the resulting test binary.

The flags handled by 'go test' are:

	-c  Compile the test binary to test.out but do not run it.

	-file a.go
	    Use only the tests in the source file a.go.
	    Multiple -file flags may be provided.

	-p n
	    Compile and test up to n packages in parallel.
	    The default value is the number of CPUs available.

	-x  Print each subcommand go test executes.

The resulting test binary, called test.out, has its own flags:

	-test.v
	    Verbose output: log all tests as they are run.

	-test.run pattern
	    Run only those tests matching the regular expression.

	-test.bench pattern
	    Run benchmarks matching the regular expression.
	    By default, no benchmarks run.

	-test.cpuprofile cpu.out
	    Write a CPU profile to the specified file before exiting.

	-test.memprofile mem.out
	    Write a memory profile to the specified file when all tests
	    are complete.

	-test.memprofilerate n
	    Enable more precise (and expensive) memory profiles by setting
	    runtime.MemProfileRate.  See 'godoc runtime MemProfileRate'.
	    To profile all memory allocations, use -test.memprofilerate=1
	    and set the environment variable GOGC=off to disable the
	    garbage collector, provided the test can run in the available
	    memory without garbage collection.

	-test.parallel n
	    Allow parallel execution of test functions that call t.Parallel.
	    The value of this flag is the maximum number of tests to run
	    simultaneously; by default, it is set to the value of GOMAXPROCS.

	-test.short
	    Tell long-running tests to shorten their run time.
	    It is off by default but set during all.bash so that installing
	    the Go tree can run a sanity check but not spend time running
	    exhaustive tests.

	-test.timeout t
		If a test runs longer than t, panic.

	-test.benchtime n
		Run enough iterations of each benchmark to take n seconds.
		The default is 1 second.

	-test.cpu 1,2,4
	    Specify a list of GOMAXPROCS values for which the tests or 
	    benchmarks should be executed.  The default is the current value
	    of GOMAXPROCS.

For convenience, each of these -test.X flags of the test binary is
also available as the flag -X in 'go test' itself.  Flags not listed
here are passed through unaltered.  For instance, the command

	go test -x -v -cpuprofile=prof.out -dir=testdata -update -file x_test.go

will compile the test binary using x_test.go and then run it as

	test.out -test.v -test.cpuprofile=prof.out -dir=testdata -update
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

An example function is similar to a test function but, instead of using *testing.T
to report success or failure, prints output to os.Stdout and os.Stderr.
That output is compared against the function's doc comment.
An example without a doc comment is compiled but not executed.

Godoc displays the body of ExampleXXX to demonstrate the use
of the function, constant, or variable XXX.  An example of a method M with
receiver type T or *T is named ExampleT_M.  There may be multiple examples
for a given function, constant, or variable, distinguished by a trailing _xxx,
where xxx is a suffix not beginning with an upper case letter.

Here is an example of an example:

	// The output of this example function.
	func ExamplePrintln() {
		Println("The output of this example function.")
	}

See the documentation of the testing package for more information.
		`,
}

// TODO(rsc): Rethink the flag handling.
// It might be better to do
//	go test [go-test-flags] [importpaths] [flags for test binary]
// If there are no import paths then the two flag sections
// run together, but we can deal with that.  Right now, 
//	go test -x  (ok)
//	go test -x math (NOT OK)
//	go test math -x (ok)
// which is weird, because the -x really does apply to go test, not to math.
// It is also possible that -file can go away.
// For now just use the gotest code.

var (
	testC        bool     // -c flag
	testP        int      // -p flag
	testX        bool     // -x flag
	testV        bool     // -v flag
	testFiles    []string // -file flag(s)  TODO: not respected
	testArgs     []string
	testShowPass bool // whether to display passing output
	testBench    bool
)

func runTest(cmd *Command, args []string) {
	var pkgArgs []string
	pkgArgs, testArgs = testFlags(args)

	// show test PASS output when no packages
	// are listed (implicitly current directory: "go test")
	// or when the -v flag has been given.
	testShowPass = len(pkgArgs) == 0 || testV

	pkgs := packages(pkgArgs)
	if len(pkgs) == 0 {
		fatalf("no packages to test")
	}

	if testC && len(pkgs) != 1 {
		fatalf("cannot use -c flag with multiple packages")
	}

	buildX = testX
	if testP > 0 {
		buildP = testP
	}

	var b builder
	b.init()

	var builds, runs, prints []*action

	// Prepare build + run + print actions for all packages being tested.
	for _, p := range pkgs {
		buildTest, runTest, printTest, err := b.test(p)
		if err != nil {
			errorf("%s", err)
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
		if a.p != nil && a.f != nil && !okBuild[a.p] && !a.p.fake {
			okBuild[a.p] = true // don't warn again
			if !warned {
				fmt.Fprintf(os.Stderr, "warning: building out-of-date packages:\n")
				warned = true
			}
			fmt.Fprintf(os.Stderr, "\t%s\n", a.p.ImportPath)
		}
	}
	if warned {
		fmt.Fprintf(os.Stderr, "installing these packages with 'go install' will speed future tests.\n\n")
	}

	b.do(root)
}

func (b *builder) test(p *Package) (buildAction, runAction, printAction *action, err error) {
	if len(p.info.TestGoFiles)+len(p.info.XTestGoFiles) == 0 {
		build := &action{p: p}
		run := &action{p: p}
		print := &action{f: (*builder).notest, p: p, deps: []*action{build}}
		return build, run, print, nil
	}

	// Build Package structs describing:
	//	ptest - package + test files
	//	pxtest - package of external test files
	//	pmain - test.out binary
	var ptest, pxtest, pmain *Package

	// go/build does not distinguish the dependencies used
	// by the TestGoFiles from the dependencies used by the
	// XTestGoFiles, so we build one list and use it for both
	// ptest and pxtest.  No harm done.
	var imports []*Package
	for _, path := range p.info.TestImports {
		p1, err := loadPackage(path)
		if err != nil {
			return nil, nil, nil, err
		}
		imports = append(imports, p1)
	}

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
	ptestObj := filepath.Join(testDir, filepath.FromSlash(p.ImportPath+".a"))
	pxtestObj := filepath.Join(testDir, filepath.FromSlash(p.ImportPath+"_test.a"))

	// Create the directory for the .a files.
	ptestDir, _ := filepath.Split(ptestObj)
	if err := b.mkdir(ptestDir); err != nil {
		return nil, nil, nil, err
	}
	if err := writeTestmain(filepath.Join(testDir, "_testmain.go"), p); err != nil {
		return nil, nil, nil, err
	}

	// Test package.
	if len(p.info.TestGoFiles) > 0 {
		ptest = new(Package)
		*ptest = *p
		ptest.GoFiles = nil
		ptest.GoFiles = append(ptest.GoFiles, p.GoFiles...)
		ptest.GoFiles = append(ptest.GoFiles, p.info.TestGoFiles...)
		ptest.target = ""
		ptest.Imports = stringList(p.info.Imports, p.info.TestImports)
		ptest.imports = append(append([]*Package{}, p.imports...), imports...)
		ptest.pkgdir = testDir
		ptest.fake = true
		a := b.action(modeBuild, modeBuild, ptest)
		a.objdir = testDir + string(filepath.Separator)
		a.objpkg = ptestObj
		a.target = ptestObj
		a.link = false
	} else {
		ptest = p
	}

	// External test package.
	if len(p.info.XTestGoFiles) > 0 {
		pxtest = &Package{
			Name:       p.Name + "_test",
			ImportPath: p.ImportPath + "_test",
			Dir:        p.Dir,
			GoFiles:    p.info.XTestGoFiles,
			Imports:    p.info.TestImports,
			t:          p.t,
			info:       &build.DirInfo{},
			imports:    imports,
			pkgdir:     testDir,
			fake:       true,
		}
		pxtest.imports = append(pxtest.imports, ptest)
		a := b.action(modeBuild, modeBuild, pxtest)
		a.objdir = testDir + string(filepath.Separator)
		a.objpkg = pxtestObj
		a.target = pxtestObj
	}

	// Action for building test.out.
	pmain = &Package{
		Name:    "main",
		Dir:     testDir,
		GoFiles: []string{"_testmain.go"},
		t:       p.t,
		info:    &build.DirInfo{},
		imports: []*Package{ptest},
		fake:    true,
	}
	if pxtest != nil {
		pmain.imports = append(pmain.imports, pxtest)
	}

	// The generated main also imports testing and regexp.
	ptesting, err := loadPackage("testing")
	if err != nil {
		return nil, nil, nil, err
	}
	pregexp, err := loadPackage("regexp")
	if err != nil {
		return nil, nil, nil, err
	}
	pmain.imports = append(pmain.imports, ptesting, pregexp)

	a := b.action(modeBuild, modeBuild, pmain)
	a.objdir = testDir + string(filepath.Separator)
	a.objpkg = filepath.Join(testDir, "main.a")
	a.target = filepath.Join(testDir, "test.out") + b.exe
	pmainAction := a

	if testC {
		// -c flag: create action to copy binary to ./test.out.
		runAction = &action{
			f:      (*builder).install,
			deps:   []*action{pmainAction},
			p:      pmain,
			target: "test.out" + b.exe,
		}
		printAction = &action{p: p, deps: []*action{runAction}} // nop
	} else {
		// run test
		runAction = &action{
			f:          (*builder).runTest,
			deps:       []*action{pmainAction},
			p:          p,
			ignoreFail: true,
		}
		printAction = &action{
			f:    (*builder).printTest,
			deps: []*action{runAction},
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
		exitStatus = 1
		return nil
	}

	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = a.p.Dir
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	t0 := time.Now()
	err := cmd.Start()

	// This is a last-ditch deadline to detect and
	// stop wedged test binaries, to keep the builders
	// running.
	const deadline = 10 * time.Minute

	tick := time.NewTimer(deadline)
	if err == nil {
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
			fmt.Fprintf(&buf, "*** Test killed: ran too long.\n")
		}
		tick.Stop()
	}
	out := buf.Bytes()
	t1 := time.Now()
	t := fmt.Sprintf("%.3fs", t1.Sub(t0).Seconds())
	if err == nil {
		fmt.Fprintf(a.testOutput, "ok  \t%s\t%s\n", a.p.ImportPath, t)
		if testShowPass {
			a.testOutput.Write(out)
		}
		return nil
	}

	fmt.Fprintf(a.testOutput, "FAIL\t%s\t%s\n", a.p.ImportPath, t)
	exitStatus = 1
	if len(out) > 0 {
		a.testOutput.Write(out)
		// assume printing the test binary's exit status is superfluous
	} else {
		fmt.Fprintf(a.testOutput, "%s\n", err)
	}
	return nil
}

// printTest is the action for printing a test result.
func (b *builder) printTest(a *action) error {
	run := a.deps[0]
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
		Info:    p.info,
	}
	for _, file := range p.info.TestGoFiles {
		if err := t.load(filepath.Join(p.Dir, file), "_test", &t.NeedTest); err != nil {
			return err
		}
	}
	for _, file := range p.info.XTestGoFiles {
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
	Info       *build.DirInfo
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
		return err
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
		case isTest(name, "Example"):
			output := n.Doc.Text()
			if output == "" {
				// Don't run examples with no output.
				continue
			}
			t.Examples = append(t.Examples, testFunc{pkg, name, output})
			*seen = true
		}
	}

	return nil
}

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
