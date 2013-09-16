// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows,!plan9

package interp_test

import (
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
	"code.google.com/p/go.tools/ssa/interp"
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
	"goprint.go", // doesn't actually assert anything
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
	"nul1.go",
	"zerodivide.go",
	"convert.go",
	"convT2X.go",
	"initialize.go",
	"ddd.go",
	"blank.go", // partly disabled; TODO(adonovan): skip blank fields in struct{_} equivalence.
	"map.go",
	"closedchan.go",
	"divide.go",
	"rename.go",
	"const3.go",
	"nil.go",
	"recover.go", // partly disabled; TODO(adonovan): fix.
	"typeswitch1.go",
	"floatcmp.go",
	"crlf.go", // doesn't actually assert anything
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

	// Typechecker failures:
	// "switch.go",            // https://code.google.com/p/go/issues/detail?id=5505

	// Broken.  TODO(adonovan): fix.
	// copy.go         // very slow; but with N=4 quickly crashes, slice index out of range.
	// nilptr.go       // interp: V > uintptr not implemented. Slow test, lots of mem
	// recover1.go     // error: "spurious recover"
	// recover2.go     // panic: interface conversion: string is not error: missing method Error
	// recover3.go     // logic errors: panicked with wrong Error.
	// args.go         // works, but requires specific os.Args from the driver.
	// index.go        // a template, not a real test.
	// mallocfin.go    // SetFinalizer not implemented.

	// TODO(adonovan): add tests from $GOROOT/test/* subtrees:
	// bench chan bugs fixedbugs interface ken.
}

// These are files in go.tools/ssa/interp/testdata/.
var testdataTests = []string{
	"boundmeth.go",
	"coverage.go",
	"fieldprom.go",
	"ifaceconv.go",
	"ifaceprom.go",
	"initorder.go",
	"methprom.go",
	"mrvchain.go",
}

// These are files in $GOROOT/src/pkg/.
// These tests exercise the "testing" package.
var gorootSrcPkgTests = []string{
	"unicode/script_test.go",
	"unicode/digit_test.go",
	"hash/crc32/crc32.go hash/crc32/crc32_generic.go hash/crc32/crc32_test.go",
	"path/path.go path/path_test.go",
	// TODO(adonovan): figure out the package loading error here:
	// "strings.go strings/search.go strings/search_test.go",
}

func run(t *testing.T, dir, input string) bool {
	fmt.Printf("Input: %s\n", input)

	start := time.Now()

	var inputs []string
	for _, i := range strings.Split(input, " ") {
		inputs = append(inputs, dir+i)
	}

	imp := importer.New(&importer.Config{Build: &build.Default})
	// TODO(adonovan): use LoadInitialPackages, then un-export ParseFiles.
	files, err := importer.ParseFiles(imp.Fset, ".", inputs...)
	if err != nil {
		t.Errorf("ssa.ParseFiles(%s) failed: %s", inputs, err.Error())
		return false
	}

	// Print a helpful hint if we don't make it to the end.
	var hint string
	defer func() {
		if hint != "" {
			fmt.Println("FAIL")
			fmt.Println(hint)
		} else {
			fmt.Println("PASS")
		}
	}()

	hint = fmt.Sprintf("To dump SSA representation, run:\n%% go build code.google.com/p/go.tools/cmd/ssadump && ./ssadump -build=CFP %s\n", input)
	mainInfo := imp.LoadMainPackage(files...)

	prog := ssa.NewProgram(imp.Fset, ssa.SanityCheckFunctions)
	if err := prog.CreatePackages(imp); err != nil {
		t.Errorf("CreatePackages failed: %s", err)
		return false
	}
	prog.BuildAll()

	mainPkg := prog.Package(mainInfo.Pkg)
	mainPkg.CreateTestMainFunction() // (no-op if main already exists)

	hint = fmt.Sprintf("To trace execution, run:\n%% go build code.google.com/p/go.tools/cmd/ssadump && ./ssadump -build=C -run --interp=T %s\n", input)
	if exitCode := interp.Interpret(mainPkg, 0, inputs[0], []string{}); exitCode != 0 {
		t.Errorf("interp.Interpret(%s) exited with code %d, want zero", inputs, exitCode)
		return false
	}

	hint = "" // call off the hounds

	if false {
		fmt.Println(input, time.Since(start)) // test profiling
	}

	return true
}

const slash = string(os.PathSeparator)

// TestInterp runs the interpreter on a selection of small Go programs.
func TestInterp(t *testing.T) {
	var failures []string

	for _, input := range testdataTests {
		if !run(t, "testdata"+slash, input) {
			failures = append(failures, input)
		}
	}

	if !testing.Short() {
		for _, input := range gorootTestTests {
			if !run(t, filepath.Join(build.Default.GOROOT, "test")+slash, input) {
				failures = append(failures, input)
			}
		}

		for _, input := range gorootSrcPkgTests {
			if !run(t, filepath.Join(build.Default.GOROOT, "src/pkg")+slash, input) {
				failures = append(failures, input)
			}
		}
	}

	if failures != nil {
		fmt.Println("The following tests failed:")
		for _, f := range failures {
			fmt.Printf("\t%s\n", f)
		}
	}
}
