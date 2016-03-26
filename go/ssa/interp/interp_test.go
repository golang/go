// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

// +build !android,!windows,!plan9

package interp_test

import (
	"bytes"
	"fmt"
	"go/build"
	"go/types"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/interp"
	"golang.org/x/tools/go/ssa/ssautil"
)

// Each line contains a space-separated list of $GOROOT/test/
// filenames comprising the main package of a program.
// They are ordered quickest-first, roughly.
//
// TODO(adonovan): integrate into the $GOROOT/test driver scripts,
// golden file checking, etc.
var gorootTestTests = []string{
	"235.go",
	"alias1.go",
	"chancap.go",
	"func5.go",
	"func6.go",
	"func7.go",
	"func8.go",
	"helloworld.go",
	"varinit.go",
	"escape3.go",
	"initcomma.go",
	"cmp.go",
	"compos.go",
	"turing.go",
	"indirect.go",
	"complit.go",
	"for.go",
	"struct0.go",
	"intcvt.go",
	"printbig.go",
	"deferprint.go",
	"escape.go",
	"range.go",
	"const4.go",
	"float_lit.go",
	"bigalg.go",
	"decl.go",
	"if.go",
	"named.go",
	"bigmap.go",
	"func.go",
	"reorder2.go",
	"closure.go",
	"gc.go",
	"simassign.go",
	"iota.go",
	"nilptr2.go",
	"goprint.go", // doesn't actually assert anything (cmpout)
	"utf.go",
	"method.go",
	"char_lit.go",
	"env.go",
	"int_lit.go",
	"string_lit.go",
	"defer.go",
	"typeswitch.go",
	"stringrange.go",
	"reorder.go",
	"method3.go",
	"literal.go",
	"nul1.go", // doesn't actually assert anything (errorcheckoutput)
	"zerodivide.go",
	"convert.go",
	"convT2X.go",
	"switch.go",
	"initialize.go",
	"ddd.go",
	"blank.go", // partly disabled
	"map.go",
	"closedchan.go",
	"divide.go",
	"rename.go",
	"const3.go",
	"nil.go",
	"recover.go", // reflection parts disabled
	"recover1.go",
	"recover2.go",
	"recover3.go",
	"typeswitch1.go",
	"floatcmp.go",
	"crlf.go", // doesn't actually assert anything (runoutput)
	// Slow tests follow.
	"bom.go", // ~1.7s
	"gc1.go", // ~1.7s
	"cmplxdivide.go cmplxdivide1.go", // ~2.4s

	// Working, but not worth enabling:
	// "append.go",    // works, but slow (15s).
	// "gc2.go",       // works, but slow, and cheats on the memory check.
	// "sigchld.go",   // works, but only on POSIX.
	// "peano.go",     // works only up to n=9, and slow even then.
	// "stack.go",     // works, but too slow (~30s) by default.
	// "solitaire.go", // works, but too slow (~30s).
	// "const.go",     // works but for but one bug: constant folder doesn't consider representations.
	// "init1.go",     // too slow (80s) and not that interesting. Cheats on ReadMemStats check too.
	// "rotate.go rotate0.go", // emits source for a test
	// "rotate.go rotate1.go", // emits source for a test
	// "rotate.go rotate2.go", // emits source for a test
	// "rotate.go rotate3.go", // emits source for a test
	// "64bit.go",             // emits source for a test
	// "run.go",               // test driver, not a test.

	// Broken.  TODO(adonovan): fix.
	// copy.go         // very slow; but with N=4 quickly crashes, slice index out of range.
	// nilptr.go       // interp: V > uintptr not implemented. Slow test, lots of mem
	// args.go         // works, but requires specific os.Args from the driver.
	// index.go        // a template, not a real test.
	// mallocfin.go    // SetFinalizer not implemented.

	// TODO(adonovan): add tests from $GOROOT/test/* subtrees:
	// bench chan bugs fixedbugs interface ken.
}

// These are files in go.tools/go/ssa/interp/testdata/.
var testdataTests = []string{
	"boundmeth.go",
	"complit.go",
	"coverage.go",
	"defer.go",
	"fieldprom.go",
	"ifaceconv.go",
	"ifaceprom.go",
	"initorder.go",
	"methprom.go",
	"mrvchain.go",
	"range.go",
	"recover.go",
	"reflect.go",
	"static.go",
	"callstack.go",
}

// These are files and packages in $GOROOT/src/.
var gorootSrcTests = []string{
	"encoding/ascii85",
	"encoding/hex",
	// "encoding/pem", // TODO(adonovan): implement (reflect.Value).SetString
	// "testing",      // TODO(adonovan): implement runtime.Goexit correctly
	// "hash/crc32",   // TODO(adonovan): implement hash/crc32.haveCLMUL
	// "log",          // TODO(adonovan): implement runtime.Callers correctly

	// Too slow:
	// "container/ring",
	// "hash/adler32",

	"unicode/utf8",
	"path",
	"flag",
	"encoding/csv",
	"text/scanner",
	"unicode",
}

type successPredicate func(exitcode int, output string) error

func run(t *testing.T, dir, input string, success successPredicate) bool {
	fmt.Printf("Input: %s\n", input)

	start := time.Now()

	var inputs []string
	for _, i := range strings.Split(input, " ") {
		if strings.HasSuffix(i, ".go") {
			i = dir + i
		}
		inputs = append(inputs, i)
	}

	var conf loader.Config
	if _, err := conf.FromArgs(inputs, true); err != nil {
		t.Errorf("FromArgs(%s) failed: %s", inputs, err)
		return false
	}

	conf.Import("runtime")

	// Print a helpful hint if we don't make it to the end.
	var hint string
	defer func() {
		if hint != "" {
			fmt.Println("FAIL")
			fmt.Println(hint)
		} else {
			fmt.Println("PASS")
		}

		interp.CapturedOutput = nil
	}()

	hint = fmt.Sprintf("To dump SSA representation, run:\n%% go build golang.org/x/tools/cmd/ssadump && ./ssadump -test -build=CFP %s\n", input)

	iprog, err := conf.Load()
	if err != nil {
		t.Errorf("conf.Load(%s) failed: %s", inputs, err)
		return false
	}

	prog := ssautil.CreateProgram(iprog, ssa.SanityCheckFunctions)
	prog.Build()

	var mainPkg *ssa.Package
	var initialPkgs []*ssa.Package
	for _, info := range iprog.InitialPackages() {
		if info.Pkg.Path() == "runtime" {
			continue // not an initial package
		}
		p := prog.Package(info.Pkg)
		initialPkgs = append(initialPkgs, p)
		if mainPkg == nil && p.Func("main") != nil {
			mainPkg = p
		}
	}
	if mainPkg == nil {
		testmainPkg := prog.CreateTestMainPackage(initialPkgs...)
		if testmainPkg == nil {
			t.Errorf("CreateTestMainPackage(%s) returned nil", mainPkg)
			return false
		}
		if testmainPkg.Func("main") == nil {
			t.Errorf("synthetic testmain package has no main")
			return false
		}
		mainPkg = testmainPkg
	}

	var out bytes.Buffer
	interp.CapturedOutput = &out

	hint = fmt.Sprintf("To trace execution, run:\n%% go build golang.org/x/tools/cmd/ssadump && ./ssadump -build=C -run --interp=T %s\n", input)
	exitCode := interp.Interpret(mainPkg, 0, &types.StdSizes{WordSize: 8, MaxAlign: 8}, inputs[0], []string{})

	// The definition of success varies with each file.
	if err := success(exitCode, out.String()); err != nil {
		t.Errorf("interp.Interpret(%s) failed: %s", inputs, err)
		return false
	}

	hint = "" // call off the hounds

	if false {
		fmt.Println(input, time.Since(start)) // test profiling
	}

	return true
}

const slash = string(os.PathSeparator)

func printFailures(failures []string) {
	if failures != nil {
		fmt.Println("The following tests failed:")
		for _, f := range failures {
			fmt.Printf("\t%s\n", f)
		}
	}
}

func success(exitcode int, output string) error {
	if exitcode != 0 {
		return fmt.Errorf("exit code was %d", exitcode)
	}
	if strings.Contains(output, "BUG") {
		return fmt.Errorf("exited zero but output contained 'BUG'")
	}
	return nil
}

// TestTestdataFiles runs the interpreter on testdata/*.go.
func TestTestdataFiles(t *testing.T) {
	var failures []string
	start := time.Now()
	for _, input := range testdataTests {
		if testing.Short() && time.Since(start) > 30*time.Second {
			printFailures(failures)
			t.Skipf("timeout - aborting test")
		}
		if !run(t, "testdata"+slash, input, success) {
			failures = append(failures, input)
		}
	}
	printFailures(failures)
}

// TestGorootTest runs the interpreter on $GOROOT/test/*.go.
func TestGorootTest(t *testing.T) {
	if testing.Short() {
		t.Skip() // too slow (~30s)
	}

	var failures []string

	for _, input := range gorootTestTests {
		if !run(t, filepath.Join(build.Default.GOROOT, "test")+slash, input, success) {
			failures = append(failures, input)
		}
	}
	for _, input := range gorootSrcTests {
		if !run(t, filepath.Join(build.Default.GOROOT, "src")+slash, input, success) {
			failures = append(failures, input)
		}
	}
	printFailures(failures)
}

// TestTestmainPackage runs the interpreter on a synthetic "testmain" package.
func TestTestmainPackage(t *testing.T) {
	if testing.Short() {
		t.Skip() // too slow on some platforms
	}

	success := func(exitcode int, output string) error {
		if exitcode == 0 {
			return fmt.Errorf("unexpected success")
		}
		if !strings.Contains(output, "FAIL: TestFoo") {
			return fmt.Errorf("missing failure log for TestFoo")
		}
		if !strings.Contains(output, "FAIL: TestBar") {
			return fmt.Errorf("missing failure log for TestBar")
		}
		// TODO(adonovan): test benchmarks too
		return nil
	}
	run(t, "testdata"+slash, "a_test.go", success)
}

// CreateTestMainPackage should return nil if there were no tests.
func TestNullTestmainPackage(t *testing.T) {
	var conf loader.Config
	conf.CreateFromFilenames("", "testdata/b_test.go")
	iprog, err := conf.Load()
	if err != nil {
		t.Fatalf("CreatePackages failed: %s", err)
	}
	prog := ssautil.CreateProgram(iprog, ssa.SanityCheckFunctions)
	mainPkg := prog.Package(iprog.Created[0].Pkg)
	if mainPkg.Func("main") != nil {
		t.Fatalf("unexpected main function")
	}
	if prog.CreateTestMainPackage(mainPkg) != nil {
		t.Fatalf("CreateTestMainPackage returned non-nil")
	}
}
