// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/parser"
	"go/token"
	"io/ioutil"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// Environment for commands.
var (
	XGC       []string // 6g -I _test -o _xtest_.6
	GC        []string // 6g -I _test _testmain.go
	GL        []string // 6l -L _test _testmain.6
	GOARCH    string
	GOROOT    string
	GORUN     string
	O         string
	args      []string // arguments passed to gotest; also passed to the binary
	fileNames []string
	env       = os.Environ()
)

// These strings are created by getTestNames.
var (
	insideFileNames  []string // list of *.go files inside the package.
	outsideFileNames []string // list of *.go files outside the package (in package foo_test).
)

var (
	files      []*File
	importPath string
)

// Flags for our own purposes. We do our own flag processing.
var (
	cFlag bool
	xFlag bool
)

// elapsed returns the number of seconds since gotest started.
func elapsed() float64 {
	return time.Now().Sub(start).Seconds()
}

var start = time.Now()

// File represents a file that contains tests.
type File struct {
	name       string
	pkg        string
	file       *os.File
	astFile    *ast.File
	tests      []string // The names of the TestXXXs.
	benchmarks []string // The names of the BenchmarkXXXs.
	examples   []example
}

type example struct {
	name   string // The name of the example function (ExampleXXX).
	output string // The expected output (stdout/stderr) of the function.
}

func main() {
	flags()
	needMakefile()
	setEnvironment()
	getTestFileNames()
	parseFiles()
	getTestNames()
	run("gomake", "testpackage-clean")
	run("gomake", "testpackage", fmt.Sprintf("GOTESTFILES=%s", strings.Join(insideFileNames, " ")))
	if len(outsideFileNames) > 0 {
		run(append(XGC, outsideFileNames...)...)
	}
	importPath = runWithStdout("gomake", "-s", "importpath")
	writeTestmainGo()
	run(GC...)
	run(GL...)
	if !cFlag {
		runTestWithArgs("./" + O + ".out")
	}
	if xFlag {
		fmt.Printf("gotest %.2fs: done\n", elapsed())
	}
}

// needMakefile tests that we have a Makefile in this directory.
func needMakefile() {
	if _, err := os.Stat("Makefile"); err != nil {
		Fatalf("please create a Makefile for gotest; see http://golang.org/doc/code.html for details")
	}
}

// Fatalf formats its arguments, prints the message with a final newline, and exits.
func Fatalf(s string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "gotest: "+s+"\n", args...)
	os.Exit(2)
}

// addEnv adds a name=value pair to the environment passed to subcommands.
// If the item is already in the environment, addEnv replaces the value.
func addEnv(name, value string) {
	for i := 0; i < len(env); i++ {
		if strings.HasPrefix(env[i], name+"=") {
			env[i] = name + "=" + value
			return
		}
	}
	env = append(env, name+"="+value)
}

// setEnvironment assembles the configuration for gotest and its subcommands.
func setEnvironment() {
	// Basic environment.
	GOROOT = runtime.GOROOT()
	addEnv("GOROOT", GOROOT)
	GOARCH = build.DefaultContext.GOARCH
	addEnv("GOARCH", GOARCH)
	var err error
	O, err = build.ArchChar(GOARCH)
	if err != nil {
		Fatalf("unknown architecture: %s", err)
	}

	// Commands and their flags.
	gc := os.Getenv("GC")
	if gc == "" {
		gc = O + "g"
	}
	var gcflags []string
	if gf := strings.TrimSpace(os.Getenv("GCFLAGS")); gf != "" {
		gcflags = strings.Fields(gf)
	}
	XGC = append([]string{gc, "-I", "_test", "-o", "_xtest_." + O}, gcflags...)
	GC = append(append([]string{gc, "-I", "_test"}, gcflags...), "_testmain.go")
	gl := os.Getenv("GL")
	if gl == "" {
		gl = O + "l"
	}
	GL = []string{gl, "-L", "_test", "_testmain." + O}

	// Silence make on Linux
	addEnv("MAKEFLAGS", "")
	addEnv("MAKELEVEL", "")
}

// getTestFileNames gets the set of files we're looking at.
// If gotest has no arguments, it scans the current directory
// for test files.
func getTestFileNames() {
	names := fileNames
	if len(names) == 0 {
		info, err := build.ScanDir(".")
		if err != nil {
			Fatalf("scanning directory: %v", err)
		}
		names = append(info.TestGoFiles, info.XTestGoFiles...)
		sort.Strings(names)
		if len(names) == 0 {
			Fatalf("no test files found in current directory")
		}
	}
	for _, n := range names {
		fd, err := os.Open(n)
		if err != nil {
			Fatalf("%s: %s", n, err)
		}
		f := &File{name: n, file: fd}
		files = append(files, f)
	}
}

// parseFiles parses the files and remembers the packages we find.
func parseFiles() {
	fileSet := token.NewFileSet()
	for _, f := range files {
		// Report declaration errors so we can abort if the files are incorrect Go.
		file, err := parser.ParseFile(fileSet, f.name, nil, parser.DeclarationErrors|parser.ParseComments)
		if err != nil {
			Fatalf("parse error: %s", err)
		}
		f.astFile = file
		f.pkg = file.Name.String()
		if f.pkg == "" {
			Fatalf("cannot happen: no package name in %s", f.name)
		}
	}
}

// getTestNames extracts the names of tests and benchmarks.  They are all
// top-level functions that are not methods.
func getTestNames() {
	for _, f := range files {
		for _, d := range f.astFile.Decls {
			n, ok := d.(*ast.FuncDecl)
			if !ok {
				continue
			}
			if n.Recv != nil { // a method, not a function.
				continue
			}
			name := n.Name.String()
			if isTest(name, "Test") {
				f.tests = append(f.tests, name)
			} else if isTest(name, "Benchmark") {
				f.benchmarks = append(f.benchmarks, name)
			} else if isTest(name, "Example") {
				output := doc.CommentText(n.Doc)
				if output == "" {
					// Don't run examples with no output.
					continue
				}
				f.examples = append(f.examples, example{
					name:   name,
					output: output,
				})
			}
			// TODO: worth checking the signature? Probably not.
		}
		if isOutsideTest(f.pkg) {
			outsideFileNames = append(outsideFileNames, f.name)
		} else {
			insideFileNames = append(insideFileNames, f.name)
		}
	}
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

func run(args ...string) {
	doRun(args, false)
}

// runWithStdout is like run, but returns the text of standard output with the last newline dropped.
func runWithStdout(argv ...string) string {
	s := doRun(argv, true)
	if strings.HasSuffix(s, "\r\n") {
		s = s[:len(s)-2]
	} else if strings.HasSuffix(s, "\n") {
		s = s[:len(s)-1]
	}
	if len(s) == 0 {
		Fatalf("no output from command %s", strings.Join(argv, " "))
	}
	return s
}

// runTestWithArgs appends gotest's runs the provided binary with the args passed on the command line.
func runTestWithArgs(binary string) {
	doRun(append([]string{binary}, args...), false)
}

// doRun is the general command runner.  The flag says whether we want to
// retrieve standard output.
func doRun(argv []string, returnStdout bool) string {
	if xFlag {
		fmt.Printf("gotest %.2fs: %s\n", elapsed(), strings.Join(argv, " "))
		start := time.Now()
		defer func() {
			t := time.Now().Sub(start)
			fmt.Printf(" [+%.2fs]\n", t.Seconds())
		}()
	}
	command := argv[0]
	if runtime.GOOS == "windows" && command == "gomake" {
		// gomake is a shell script and it cannot be executed directly on Windows.
		cmd := ""
		for i, v := range argv {
			if i > 0 {
				cmd += " "
			}
			cmd += `"` + v + `"`
		}
		command = "bash"
		argv = []string{"bash", "-c", cmd}
	}
	var err error
	argv[0], err = exec.LookPath(argv[0])
	if err != nil {
		Fatalf("can't find %s: %s", command, err)
	}
	procAttr := &os.ProcAttr{
		Env: env,
		Files: []*os.File{
			os.Stdin,
			os.Stdout,
			os.Stderr,
		},
	}
	var r, w *os.File
	if returnStdout {
		r, w, err = os.Pipe()
		if err != nil {
			Fatalf("can't create pipe: %s", err)
		}
		procAttr.Files[1] = w
	}
	proc, err := os.StartProcess(argv[0], argv, procAttr)
	if err != nil {
		Fatalf("%s failed to start: %s", command, err)
	}
	if returnStdout {
		defer r.Close()
		w.Close()
	}
	waitMsg, err := proc.Wait(0)
	if err != nil || waitMsg == nil {
		Fatalf("%s failed: %s", command, err)
	}
	if !waitMsg.Exited() || waitMsg.ExitStatus() != 0 {
		Fatalf("%q failed: %s", strings.Join(argv, " "), waitMsg)
	}
	if returnStdout {
		b, err := ioutil.ReadAll(r)
		if err != nil {
			Fatalf("can't read output from command: %s", err)
		}
		return string(b)
	}
	return ""
}

// writeTestmainGo generates the test program to be compiled, "./_testmain.go".
func writeTestmainGo() {
	f, err := os.Create("_testmain.go")
	if err != nil {
		Fatalf("can't create _testmain.go: %s", err)
	}
	defer f.Close()
	b := bufio.NewWriter(f)
	defer b.Flush()

	// Package and imports.
	fmt.Fprint(b, "package main\n\n")
	// Are there tests from a package other than the one we're testing?
	// We can't just use file names because some of the things we compiled
	// contain no tests.
	outsideTests := false
	insideTests := false
	for _, f := range files {
		//println(f.name, f.pkg)
		if len(f.tests) == 0 && len(f.benchmarks) == 0 && len(f.examples) == 0 {
			continue
		}
		if isOutsideTest(f.pkg) {
			outsideTests = true
		} else {
			insideTests = true
		}
	}
	// Rename the imports for the system under test,
	// in case the tested package has the same name
	// as any of the other imports, variables or methods.
	if insideTests {
		switch importPath {
		case "testing":
		case "main":
			// Import path main is reserved, so import with
			// explicit reference to ./_test/main instead.
			fmt.Fprintf(b, "import target %q\n", "./_test/main")
		default:
			fmt.Fprintf(b, "import target %q\n", importPath)
		}
	}
	if outsideTests {
		// It is possible to have both inside and outside tests
		// at the same time, so a different import name is needed.
		fmt.Fprintf(b, "import target_test %q\n", "./_xtest_")
	}
	fmt.Fprintf(b, "import %q\n", "testing")
	fmt.Fprintf(b, "import %q\n", "regexp")
	fmt.Fprintln(b) // for gofmt

	// Tests.
	fmt.Fprintln(b, "var tests = []testing.InternalTest{")
	for _, f := range files {
		for _, t := range f.tests {
			fmt.Fprintf(b, "\t{\"%s.%s\", %s.%s},\n", f.pkg, t, renamedPackage(f.pkg), t)
		}
	}
	fmt.Fprintln(b, "}")
	fmt.Fprintln(b)

	// Benchmarks.
	fmt.Fprintf(b, "var benchmarks = []testing.InternalBenchmark{")
	for _, f := range files {
		for _, bm := range f.benchmarks {
			fmt.Fprintf(b, "\t{\"%s.%s\", %s.%s},\n", f.pkg, bm, renamedPackage(f.pkg), bm)
		}
	}
	fmt.Fprintln(b, "}")

	// Examples.
	fmt.Fprintf(b, "var examples = []testing.InternalExample{")
	for _, f := range files {
		for _, eg := range f.examples {
			fmt.Fprintf(b, "\t{%q, %s.%s, %q},\n", eg.name, renamedPackage(f.pkg), eg.name, eg.output)
		}
	}
	fmt.Fprintln(b, "}")

	// Body.
	fmt.Fprintln(b, testBody)
}

// renamedPackage returns the name under which the test package was imported.
func renamedPackage(pkg string) string {
	if isOutsideTest(pkg) {
		return "target_test"
	}
	return "target"
}

func isOutsideTest(pkg string) bool {
	return strings.HasSuffix(pkg, "_test")
}

// testBody is just copied to the output. It's the code that runs the tests.
var testBody = `
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
}`
