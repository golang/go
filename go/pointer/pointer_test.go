// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

//go:build !android
// +build !android

package pointer_test

// This test uses 'expectation' comments embedded within testdata/*.go
// files to specify the expected pointer analysis behaviour.
// See below for grammar.

import (
	"bytes"
	"errors"
	"fmt"
	"go/token"
	"go/types"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"unsafe"

	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
	"golang.org/x/tools/go/types/typeutil"
)

var inputs = []string{
	"testdata/a_test.go",
	"testdata/another.go",
	"testdata/arrayreflect.go",
	"testdata/arrays.go",
	"testdata/channels.go",
	"testdata/chanreflect.go",
	"testdata/context.go",
	"testdata/conv.go",
	"testdata/extended.go",
	"testdata/finalizer.go",
	"testdata/flow.go",
	"testdata/fmtexcerpt.go",
	"testdata/func.go",
	"testdata/funcreflect.go",
	"testdata/hello.go", // NB: causes spurious failure of HVN cross-check
	"testdata/interfaces.go",
	"testdata/issue9002.go",
	"testdata/mapreflect.go",
	"testdata/maps.go",
	"testdata/panic.go",
	"testdata/recur.go",
	"testdata/reflect.go",
	"testdata/rtti.go",
	"testdata/structreflect.go",
	"testdata/structs.go",
	// "testdata/timer.go", // TODO(adonovan): fix broken assumptions about runtime timers
}

// Expectation grammar:
//
// @calls f -> g
//
//	A 'calls' expectation asserts that edge (f, g) appears in the
//	callgraph.  f and g are notated as per Function.String(), which
//	may contain spaces (e.g. promoted method in anon struct).
//
// @pointsto a | b | c
//
//	A 'pointsto' expectation asserts that the points-to set of its
//	operand contains exactly the set of labels {a,b,c} notated as per
//	labelString.
//
//	A 'pointsto' expectation must appear on the same line as a
//	print(x) statement; the expectation's operand is x.
//
//	If one of the strings is "...", the expectation asserts that the
//	points-to set at least the other labels.
//
//	We use '|' because label names may contain spaces, e.g.  methods
//	of anonymous structs.
//
//	From a theoretical perspective, concrete types in interfaces are
//	labels too, but they are represented differently and so have a
//	different expectation, @types, below.
//
// @types t | u | v
//
//	A 'types' expectation asserts that the set of possible dynamic
//	types of its interface operand is exactly {t,u,v}, notated per
//	go/types.Type.String(). In other words, it asserts that the type
//	component of the interface may point to that set of concrete type
//	literals.  It also works for reflect.Value, though the types
//	needn't be concrete in that case.
//
//	A 'types' expectation must appear on the same line as a
//	print(x) statement; the expectation's operand is x.
//
//	If one of the strings is "...", the expectation asserts that the
//	interface's type may point to at least the other types.
//
//	We use '|' because type names may contain spaces.
//
// @warning "regexp"
//
//	A 'warning' expectation asserts that the analysis issues a
//	warning that matches the regular expression within the string
//	literal.
//
// @line id
//
//	A line directive associates the name "id" with the current
//	file:line.  The string form of labels will use this id instead of
//	a file:line, making @pointsto expectations more robust against
//	perturbations in the source file.
//	(NB, anon functions still include line numbers.)
type expectation struct {
	kind     string // "pointsto" | "pointstoquery" | "types" | "calls" | "warning"
	filepath string
	linenum  int // source line number, 1-based
	args     []string
	query    string           // extended query
	extended *pointer.Pointer // extended query pointer
	types    []types.Type     // for types
}

func (e *expectation) String() string {
	return fmt.Sprintf("@%s[%s]", e.kind, strings.Join(e.args, " | "))
}

func (e *expectation) errorf(format string, args ...interface{}) {
	fmt.Printf("%s:%d: ", e.filepath, e.linenum)
	fmt.Printf(format, args...)
	fmt.Println()
}

func (e *expectation) needsProbe() bool {
	return e.kind == "pointsto" || e.kind == "pointstoquery" || e.kind == "types"
}

// Find probe (call to print(x)) of same source file/line as expectation.
func findProbe(prog *ssa.Program, probes map[*ssa.CallCommon]bool, queries map[ssa.Value]pointer.Pointer, e *expectation) (site *ssa.CallCommon, pts pointer.PointsToSet) {
	for call := range probes {
		pos := prog.Fset.Position(call.Pos())
		if pos.Line == e.linenum && pos.Filename == e.filepath {
			// TODO(adonovan): send this to test log (display only on failure).
			// fmt.Printf("%s:%d: info: found probe for %s: %s\n",
			// 	e.filepath, e.linenum, e, p.arg0) // debugging
			return call, queries[call.Args[0]].PointsTo()
		}
	}
	return // e.g. analysis didn't reach this call
}

func doOneInput(t *testing.T, input, fpath string) bool {
	cfg := &packages.Config{
		Mode:  packages.LoadAllSyntax,
		Tests: true,
	}
	pkgs, err := packages.Load(cfg, fpath)
	if err != nil {
		fmt.Println(err)
		return false
	}
	if packages.PrintErrors(pkgs) > 0 {
		fmt.Println("loaded packages have errors")
		return false
	}

	// SSA creation + building.
	prog, ssaPkgs := ssautil.AllPackages(pkgs, ssa.SanityCheckFunctions)
	prog.Build()

	// main underlying packages.Package.
	mainPpkg := pkgs[0]
	mainpkg := ssaPkgs[0]
	ptrmain := mainpkg // main package for the pointer analysis
	if mainpkg.Func("main") == nil {
		// For test programs without main, such as testdata/a_test.go,
		// the package with the original code is "main [main.test]" and
		// the package with the main is "main.test".
		for i, pkg := range pkgs {
			if pkg.ID == mainPpkg.ID+".test" {
				ptrmain = ssaPkgs[i]
			} else if pkg.ID == fmt.Sprintf("%s [%s.test]", mainPpkg.ID, mainPpkg.ID) {
				mainpkg = ssaPkgs[i]
			}
		}
	}

	// Find all calls to the built-in print(x).  Analytically,
	// print is a no-op, but it's a convenient hook for testing
	// the PTS of an expression, so our tests use it.
	probes := make(map[*ssa.CallCommon]bool)
	for fn := range ssautil.AllFunctions(prog) {
		if fn.Pkg == mainpkg {
			for _, b := range fn.Blocks {
				for _, instr := range b.Instrs {
					if instr, ok := instr.(ssa.CallInstruction); ok {
						call := instr.Common()
						if b, ok := call.Value.(*ssa.Builtin); ok && b.Name() == "print" && len(call.Args) == 1 {
							probes[instr.Common()] = true
						}
					}
				}
			}
		}
	}

	ok := true

	lineMapping := make(map[string]string) // maps "file:line" to @line tag

	// Parse expectations in this input.
	var exps []*expectation
	re := regexp.MustCompile("// *@([a-z]*) *(.*)$")
	lines := strings.Split(input, "\n")
	for linenum, line := range lines {
		linenum++ // make it 1-based
		if matches := re.FindAllStringSubmatch(line, -1); matches != nil {
			match := matches[0]
			kind, rest := match[1], match[2]
			e := &expectation{kind: kind, filepath: fpath, linenum: linenum}

			if kind == "line" {
				if rest == "" {
					ok = false
					e.errorf("@%s expectation requires identifier", kind)
				} else {
					lineMapping[fmt.Sprintf("%s:%d", fpath, linenum)] = rest
				}
				continue
			}

			if e.needsProbe() && !strings.Contains(line, "print(") {
				ok = false
				e.errorf("@%s expectation must follow call to print(x)", kind)
				continue
			}

			switch kind {
			case "pointsto":
				e.args = split(rest, "|")

			case "pointstoquery":
				args := strings.SplitN(rest, " ", 2)
				e.query = args[0]
				e.args = split(args[1], "|")
			case "types":
				for _, typstr := range split(rest, "|") {
					var t types.Type = types.Typ[types.Invalid] // means "..."
					if typstr != "..." {
						tv, err := types.Eval(prog.Fset, mainpkg.Pkg, mainPpkg.Syntax[0].Pos(), typstr)
						if err != nil {
							ok = false
							// Don't print err since its location is bad.
							e.errorf("'%s' is not a valid type: %s", typstr, err)
							continue
						}
						t = tv.Type
					}
					e.types = append(e.types, t)
				}

			case "calls":
				e.args = split(rest, "->")
				// TODO(adonovan): eagerly reject the
				// expectation if fn doesn't denote
				// existing function, rather than fail
				// the expectation after analysis.
				if len(e.args) != 2 {
					ok = false
					e.errorf("@calls expectation wants 'caller -> callee' arguments")
					continue
				}

			case "warning":
				lit, err := strconv.Unquote(strings.TrimSpace(rest))
				if err != nil {
					ok = false
					e.errorf("couldn't parse @warning operand: %s", err.Error())
					continue
				}
				e.args = append(e.args, lit)

			default:
				ok = false
				e.errorf("unknown expectation kind: %s", e)
				continue
			}
			exps = append(exps, e)
		}
	}

	var log bytes.Buffer
	fmt.Fprintf(&log, "Input: %s\n", fpath)

	// Run the analysis.
	config := &pointer.Config{
		Reflection:     true,
		BuildCallGraph: true,
		Mains:          []*ssa.Package{ptrmain},
		Log:            &log,
	}
probeLoop:
	for probe := range probes {
		v := probe.Args[0]
		pos := prog.Fset.Position(probe.Pos())
		for _, e := range exps {
			if e.linenum == pos.Line && e.filepath == pos.Filename && e.kind == "pointstoquery" {
				var err error
				e.extended, err = config.AddExtendedQuery(v, e.query)
				if err != nil {
					panic(err)
				}
				continue probeLoop
			}
		}
		if pointer.CanPoint(v.Type()) {
			config.AddQuery(v)
		}
	}

	// Print the log is there was an error or a panic.
	complete := false
	defer func() {
		if !complete || !ok {
			log.WriteTo(os.Stderr)
		}
	}()

	result, err := pointer.Analyze(config)
	if err != nil {
		panic(err) // internal error in pointer analysis
	}

	// Check the expectations.
	for _, e := range exps {
		var call *ssa.CallCommon
		var pts pointer.PointsToSet
		var tProbe types.Type
		if e.needsProbe() {
			if call, pts = findProbe(prog, probes, result.Queries, e); call == nil {
				ok = false
				e.errorf("unreachable print() statement has expectation %s", e)
				continue
			}
			if e.extended != nil {
				pts = e.extended.PointsTo()
			}
			tProbe = call.Args[0].Type()
			if !pointer.CanPoint(tProbe) {
				ok = false
				e.errorf("expectation on non-pointerlike operand: %s", tProbe)
				continue
			}
		}

		switch e.kind {
		case "pointsto", "pointstoquery":
			if !checkPointsToExpectation(e, pts, lineMapping, prog) {
				ok = false
			}

		case "types":
			if !checkTypesExpectation(e, pts, tProbe) {
				ok = false
			}

		case "calls":
			if !checkCallsExpectation(prog, e, result.CallGraph) {
				ok = false
			}

		case "warning":
			if !checkWarningExpectation(prog, e, result.Warnings) {
				ok = false
			}
		}
	}

	complete = true

	// ok = false // debugging: uncomment to always see log

	return ok
}

func labelString(l *pointer.Label, lineMapping map[string]string, prog *ssa.Program) string {
	// Functions and Globals need no pos suffix,
	// nor do allocations in intrinsic operations
	// (for which we'll print the function name).
	switch l.Value().(type) {
	case nil, *ssa.Function, *ssa.Global:
		return l.String()
	}

	str := l.String()
	if pos := l.Pos(); pos != token.NoPos {
		// Append the position, using a @line tag instead of a line number, if defined.
		posn := prog.Fset.Position(pos)
		s := fmt.Sprintf("%s:%d", posn.Filename, posn.Line)
		if tag, ok := lineMapping[s]; ok {
			return fmt.Sprintf("%s@%s:%d", str, tag, posn.Column)
		}
		str = fmt.Sprintf("%s@%s", str, posn)
	}
	return str
}

func checkPointsToExpectation(e *expectation, pts pointer.PointsToSet, lineMapping map[string]string, prog *ssa.Program) bool {
	expected := make(map[string]int)
	surplus := make(map[string]int)
	exact := true
	for _, g := range e.args {
		if g == "..." {
			exact = false
			continue
		}
		expected[g]++
	}
	// Find the set of labels that the probe's
	// argument (x in print(x)) may point to.
	for _, label := range pts.Labels() {
		name := labelString(label, lineMapping, prog)
		if expected[name] > 0 {
			expected[name]--
		} else if exact {
			surplus[name]++
		}
	}
	// Report multiset difference:
	ok := true
	for _, count := range expected {
		if count > 0 {
			ok = false
			e.errorf("value does not alias these expected labels: %s", join(expected))
			break
		}
	}
	for _, count := range surplus {
		if count > 0 {
			ok = false
			e.errorf("value may additionally alias these labels: %s", join(surplus))
			break
		}
	}
	return ok
}

func checkTypesExpectation(e *expectation, pts pointer.PointsToSet, typ types.Type) bool {
	var expected typeutil.Map
	var surplus typeutil.Map
	exact := true
	for _, g := range e.types {
		if g == types.Typ[types.Invalid] {
			exact = false
			continue
		}
		expected.Set(g, struct{}{})
	}

	if !pointer.CanHaveDynamicTypes(typ) {
		e.errorf("@types expectation requires an interface- or reflect.Value-typed operand, got %s", typ)
		return false
	}

	// Find the set of types that the probe's
	// argument (x in print(x)) may contain.
	for _, T := range pts.DynamicTypes().Keys() {
		if expected.At(T) != nil {
			expected.Delete(T)
		} else if exact {
			surplus.Set(T, struct{}{})
		}
	}
	// Report set difference:
	ok := true
	if expected.Len() > 0 {
		ok = false
		e.errorf("interface cannot contain these types: %s", expected.KeysString())
	}
	if surplus.Len() > 0 {
		ok = false
		e.errorf("interface may additionally contain these types: %s", surplus.KeysString())
	}
	return ok
}

var errOK = errors.New("OK")

func checkCallsExpectation(prog *ssa.Program, e *expectation, cg *callgraph.Graph) bool {
	found := make(map[string]int)
	err := callgraph.GraphVisitEdges(cg, func(edge *callgraph.Edge) error {
		// Name-based matching is inefficient but it allows us to
		// match functions whose names that would not appear in an
		// index ("<root>") or which are not unique ("func@1.2").
		if edge.Caller.Func.String() == e.args[0] {
			calleeStr := edge.Callee.Func.String()
			if calleeStr == e.args[1] {
				return errOK // expectation satisfied; stop the search
			}
			found[calleeStr]++
		}
		return nil
	})
	if err == errOK {
		return true
	}
	if len(found) == 0 {
		e.errorf("didn't find any calls from %s", e.args[0])
	}
	e.errorf("found no call from %s to %s, but only to %s",
		e.args[0], e.args[1], join(found))
	return false
}

func checkWarningExpectation(prog *ssa.Program, e *expectation, warnings []pointer.Warning) bool {
	// TODO(adonovan): check the position part of the warning too?
	re, err := regexp.Compile(e.args[0])
	if err != nil {
		e.errorf("invalid regular expression in @warning expectation: %s", err.Error())
		return false
	}

	if len(warnings) == 0 {
		e.errorf("@warning %q expectation, but no warnings", e.args[0])
		return false
	}

	for _, w := range warnings {
		if re.MatchString(w.Message) {
			return true
		}
	}

	e.errorf("@warning %q expectation not satisfied; found these warnings though:", e.args[0])
	for _, w := range warnings {
		fmt.Printf("%s: warning: %s\n", prog.Fset.Position(w.Pos), w.Message)
	}
	return false
}

func TestInput(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode; this test requires tons of memory; https://golang.org/issue/14113")
	}
	if unsafe.Sizeof(unsafe.Pointer(nil)) <= 4 {
		t.Skip("skipping memory-intensive test on platform with small address space; https://golang.org/issue/14113")
	}
	ok := true

	wd, err := os.Getwd()
	if err != nil {
		t.Errorf("os.Getwd: %s", err)
		return
	}

	// 'go test' does a chdir so that relative paths in
	// diagnostics no longer make sense relative to the invoking
	// shell's cwd.  We print a special marker so that Emacs can
	// make sense of them.
	fmt.Fprintf(os.Stderr, "Entering directory `%s'\n", wd)

	for _, filename := range inputs {
		content, err := ioutil.ReadFile(filename)
		if err != nil {
			t.Errorf("couldn't read file '%s': %s", filename, err)
			continue
		}

		fpath, err := filepath.Abs(filename)
		if err != nil {
			t.Errorf("couldn't get absolute path for '%s': %s", filename, err)
		}

		if !doOneInput(t, string(content), fpath) {
			ok = false
		}
	}
	if !ok {
		t.Fail()
	}
}

// join joins the elements of multiset with " | "s.
func join(set map[string]int) string {
	var buf bytes.Buffer
	sep := ""
	for name, count := range set {
		for i := 0; i < count; i++ {
			buf.WriteString(sep)
			sep = " | "
			buf.WriteString(name)
		}
	}
	return buf.String()
}

// split returns the list of sep-delimited non-empty strings in s.
func split(s, sep string) (r []string) {
	for _, elem := range strings.Split(s, sep) {
		elem = strings.TrimSpace(elem)
		if elem != "" {
			r = append(r, elem)
		}
	}
	return
}
