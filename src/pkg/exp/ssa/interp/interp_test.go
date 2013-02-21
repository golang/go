package interp_test

import (
	"exp/ssa"
	"exp/ssa/interp"
	"flag"
	"fmt"
	"go/build"
	"strings"
	"testing"
)

// ANSI terminal sequences.
const (
	ansiRed   = "\x1b[1;31m"
	ansiGreen = "\x1b[1;32m"
	ansiReset = "\x1b[0m"
)

var color = flag.Bool("color", false, "Emit color codes for an ANSI terminal.")

func red(s string) string {
	if *color {
		return ansiRed + s + ansiReset
	}
	return s
}

func green(s string) string {
	if *color {
		return ansiGreen + s + ansiReset
	}
	return s
}

// Each line contains a space-separated list of $GOROOT/test/
// filenames comprising the main package of a program.
// They are ordered quickest-first, roughly.
//
// TODO(adonovan): integrate into the $GOROOT/test driver scripts,
// golden file checking, etc.
var gorootTests = []string{
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
	// The following tests are disabled until the typechecker supports shifts correctly.
	// They can be enabled if you patch workaround https://codereview.appspot.com/7312068.
	// "closure.go",
	// "gc.go",
	// "goprint.go",  // doesn't actually assert anything
	// "utf.go",
	"method.go",
	// "char_lit.go",
	//"env.go",
	// "int_lit.go",
	// "string_lit.go",
	// "defer.go",
	// "typeswitch.go",
	// "stringrange.go",
	// "reorder.go",
	"literal.go",
	// "nul1.go",
	// "zerodivide.go",
	// "convert.go",
	"convT2X.go",
	// "switch.go",
	// "initialize.go",
	// "blank.go", // partly disabled; TODO(adonovan): skip blank fields in struct{_} equivalence.
	// "map.go",
	// "bom.go",
	// "closedchan.go",
	// "divide.go",
	// "rename.go",
	// "const3.go",
	// "nil.go",
	// "recover.go", // partly disabled; TODO(adonovan): fix.
	// Slow tests follow.
	// "cmplxdivide.go cmplxdivide1.go",
	// "append.go",
	// "crlf.go", // doesn't actually assert anything
	//"typeswitch1.go",
	// "floatcmp.go",
	"gc1.go",

	// Working, but not worth enabling:
	// "gc2.go",       // works, but slow, and cheats on the memory check.
	// "sigchld.go",   // works, but only on POSIX.
	// "peano.go",     // works only up to n=9, and slow even then.
	// "stack.go",     // works, but too slow (~30s) by default.
	// "solitaire.go", // works, but too slow (~30s).
	// "const.go",     // works but for but one bug: constant folder doesn't consider representations.
	// "init1.go",     // too slow (80s) and not that interesting. Cheats on ReadMemStats check too.

	// Broken.  TODO(adonovan): fix.
	// ddd.go          // builder: variadic methods
	// copy.go         // very slow; but with N=4 quickly crashes, slice index out of range.
	// nilptr.go       // interp: V > uintptr not implemented. Slow test, lots of mem
	// iota.go         // typechecker: crash
	// rotate.go       // typechecker: shifts
	// rune.go         // typechecker: shifts
	// 64bit.go        // typechecker: shifts
	// cmp.go          // typechecker: comparison
	// recover1.go     // error: "spurious recover"
	// recover2.go     // panic: interface conversion: string is not error: missing method Error
	// recover3.go     // logic errors: panicked with wrong Error.
	// simassign.go    // requires support for f(f(x,y)).
	// method3.go      // Fails dynamically; (*T).f vs (T).f are distinct methods.
	// ddd2.go         // fails
	// run.go          // rtype.NumOut not yet implemented.  Not really a test though.
	// args.go         // works, but requires specific os.Args from the driver.
	// index.go        // a template, not a real test.
	// mallocfin.go    // SetFinalizer not implemented.

	// TODO(adonovan): add tests from $GOROOT/test/* subtrees:
	// bench chan bugs fixedbugs interface ken.
}

// These are files in exp/ssa/interp/testdata/.
var testdataTests = []string{
// "coverage.go",  // shifts
}

func run(t *testing.T, dir, input string) bool {
	fmt.Printf("Input: %s\n", input)

	var inputs []string
	for _, i := range strings.Split(input, " ") {
		inputs = append(inputs, dir+i)
	}

	b := ssa.NewBuilder(ssa.SanityCheckFunctions, ssa.GorootLoader, nil)
	files, err := ssa.ParseFiles(b.Prog.Files, ".", inputs...)
	if err != nil {
		t.Errorf("ssa.ParseFiles(%s) failed: %s", inputs, err.Error())
		return false
	}

	// Print a helpful hint if we don't make it to the end.
	var hint string
	defer func() {
		if hint != "" {
			fmt.Println(red("FAIL"))
			fmt.Println(hint)
		} else {
			fmt.Println(green("PASS"))
		}
	}()

	hint = fmt.Sprintf("To dump SSA representation, run:\n%% go run exp/ssa/ssadump.go -build=CFP %s\n", input)
	mainpkg, err := b.CreatePackage("main", files)
	if err != nil {
		t.Errorf("ssa.Builder.CreatePackage(%s) failed: %s", inputs, err.Error())

		return false
	}

	b.BuildPackage(mainpkg)
	b = nil // discard Builder

	hint = fmt.Sprintf("To trace execution, run:\n%% go run exp/ssa/ssadump.go -build=C -run --interp=T %s\n", input)
	if exitCode := interp.Interpret(mainpkg, 0, inputs[0], []string{}); exitCode != 0 {
		t.Errorf("interp.Interpret(%s) exited with code %d, want zero", inputs, exitCode)
		return false
	}

	hint = "" // call off the hounds
	return true
}

// TestInterp runs the interpreter on a selection of small Go programs.
func TestInterp(t *testing.T) {
	var failures []string

	for _, input := range testdataTests {
		if !run(t, build.Default.GOROOT+"/src/pkg/exp/ssa/interp/testdata/", input) {
			failures = append(failures, input)
		}
	}

	if !testing.Short() {
		for _, input := range gorootTests {
			if !run(t, build.Default.GOROOT+"/test/", input) {
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
