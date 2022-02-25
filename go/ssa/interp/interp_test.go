// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package interp_test

// This test runs the SSA interpreter over sample Go programs.
// Because the interpreter requires intrinsics for assembly
// functions and many low-level runtime routines, it is inherently
// not robust to evolutionary change in the standard library.
// Therefore the test cases are restricted to programs that
// use a fake standard library in testdata/src containing a tiny
// subset of simple functions useful for writing assertions.
//
// We no longer attempt to interpret any real standard packages such as
// fmt or testing, as it proved too fragile.

import (
	"bytes"
	"fmt"
	"go/build"
	"go/types"
	"log"
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
// If a test in this list fails spuriously, remove it.
var gorootTestTests = []string{
	"235.go",
	"alias1.go",
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
	"gc.go",
	"simassign.go",
	"iota.go",
	"nilptr2.go",
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
	"ddd.go",
	"blank.go", // partly disabled
	"closedchan.go",
	"divide.go",
	"rename.go",
	"nil.go",
	"recover1.go",
	"recover2.go",
	"recover3.go",
	"typeswitch1.go",
	"floatcmp.go",
	"crlf.go", // doesn't actually assert anything (runoutput)
}

// These are files in go.tools/go/ssa/interp/testdata/.
var testdataTests = []string{
	"boundmeth.go",
	"complit.go",
	"convert.go",
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
}

func run(t *testing.T, input string) bool {
	// The recover2 test case is broken on Go 1.14+. See golang/go#34089.
	// TODO(matloob): Fix this.
	if filepath.Base(input) == "recover2.go" {
		t.Skip("The recover2.go test is broken in go1.14+. See golang.org/issue/34089.")
	}

	t.Logf("Input: %s\n", input)

	start := time.Now()

	ctx := build.Default    // copy
	ctx.GOROOT = "testdata" // fake goroot
	ctx.GOOS = "linux"
	ctx.GOARCH = "amd64"

	conf := loader.Config{Build: &ctx}
	if _, err := conf.FromArgs([]string{input}, true); err != nil {
		t.Errorf("FromArgs(%s) failed: %s", input, err)
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
		t.Errorf("conf.Load(%s) failed: %s", input, err)
		return false
	}

	prog := ssautil.CreateProgram(iprog, ssa.SanityCheckFunctions)
	prog.Build()

	mainPkg := prog.Package(iprog.Created[0].Pkg)
	if mainPkg == nil {
		t.Fatalf("not a main package: %s", input)
	}

	interp.CapturedOutput = new(bytes.Buffer)

	hint = fmt.Sprintf("To trace execution, run:\n%% go build golang.org/x/tools/cmd/ssadump && ./ssadump -build=C -test -run --interp=T %s\n", input)
	exitCode := interp.Interpret(mainPkg, 0, &types.StdSizes{WordSize: 8, MaxAlign: 8}, input, []string{})
	if exitCode != 0 {
		t.Fatalf("interpreting %s: exit code was %d", input, exitCode)
	}
	// $GOROOT/test tests use this convention:
	if strings.Contains(interp.CapturedOutput.String(), "BUG") {
		t.Fatalf("interpreting %s: exited zero but output contained 'BUG'", input)
	}

	hint = "" // call off the hounds

	if false {
		t.Log(input, time.Since(start)) // test profiling
	}

	return true
}

func printFailures(failures []string) {
	if failures != nil {
		fmt.Println("The following tests failed:")
		for _, f := range failures {
			fmt.Printf("\t%s\n", f)
		}
	}
}

// TestTestdataFiles runs the interpreter on testdata/*.go.
func TestTestdataFiles(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}

	var failures []string
	for _, input := range testdataTests {
		if !run(t, filepath.Join(cwd, "testdata", input)) {
			failures = append(failures, input)
		}
	}
	printFailures(failures)
}

// TestGorootTest runs the interpreter on $GOROOT/test/*.go.
func TestGorootTest(t *testing.T) {
	var failures []string

	for _, input := range gorootTestTests {
		if !run(t, filepath.Join(build.Default.GOROOT, "test", input)) {
			failures = append(failures, input)
		}
	}
	printFailures(failures)
}
