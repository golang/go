// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"encoding/json"
	"fmt"
	"internal/buildcfg"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	debugTraceFuncs = 1 << iota
	debugTraceFuncFlags
	debugTraceResults
	debugTraceParams
	debugTraceExprClassify
	debugTraceCalls
	debugTraceScoring
)

// propAnalyzer interface is used for defining one or more analyzer
// helper objects, each tasked with computing some specific subset of
// the properties we're interested in. The assumption is that
// properties are independent, so each new analyzer that implements
// this interface can operate entirely on its own. For a given analyzer
// there will be a sequence of calls to nodeVisitPre and nodeVisitPost
// as the nodes within a function are visited, then a followup call to
// setResults so that the analyzer can transfer its results into the
// final properties object.
type propAnalyzer interface {
	nodeVisitPre(n ir.Node)
	nodeVisitPost(n ir.Node)
	setResults(funcProps *FuncProps)
}

// fnInlHeur contains inline heuristics state information about a
// specific Go function being analyzed/considered by the inliner. Note
// that in addition to constructing a fnInlHeur object by analyzing a
// specific *ir.Func, there is also code in the test harness
// (funcprops_test.go) that builds up fnInlHeur's by reading in and
// parsing a dump. This is the reason why we have file/fname/line
// fields below instead of just an *ir.Func field.
type fnInlHeur struct {
	props *FuncProps
	cstab CallSiteTab
	fname string
	file  string
	line  uint
}

var fpmap = map[*ir.Func]fnInlHeur{}

// AnalyzeFunc computes function properties for fn and its contained
// closures, updating the global 'fpmap' table. It is assumed that
// "CanInline" has been run on fn and on the closures that feed
// directly into calls; other closures not directly called will also
// be checked inlinability for inlinability here in case they are
// returned as a result.
func AnalyzeFunc(fn *ir.Func, canInline func(*ir.Func), budgetForFunc func(*ir.Func) int32, inlineMaxBudget int) {
	if fpmap == nil {
		// If fpmap is nil this indicates that the main inliner pass is
		// complete and we're doing inlining of wrappers (no heuristics
		// used here).
		return
	}
	if fn.OClosure != nil {
		// closures will be processed along with their outer enclosing func.
		return
	}
	enableDebugTraceIfEnv()
	if debugTrace&debugTraceFuncs != 0 {
		fmt.Fprintf(os.Stderr, "=-= AnalyzeFunc(%v)\n", fn)
	}
	// Build up a list containing 'fn' and any closures it contains. Along
	// the way, test to see whether each closure is inlinable in case
	// we might be returning it.
	funcs := []*ir.Func{fn}
	ir.VisitFuncAndClosures(fn, func(n ir.Node) {
		if clo, ok := n.(*ir.ClosureExpr); ok {
			funcs = append(funcs, clo.Func)
		}
	})

	// Analyze the list of functions. We want to visit a given func
	// only after the closures it contains have been processed, so
	// iterate through the list in reverse order. Once a function has
	// been analyzed, revisit the question of whether it should be
	// inlinable; if it is over the default hairyness limit and it
	// doesn't have any interesting properties, then we don't want
	// the overhead of writing out its inline body.
	nameFinder := newNameFinder(fn)
	for i := len(funcs) - 1; i >= 0; i-- {
		f := funcs[i]
		if f.OClosure != nil && !f.InlinabilityChecked() {
			canInline(f)
		}
		funcProps := analyzeFunc(f, inlineMaxBudget, nameFinder)
		revisitInlinability(f, funcProps, budgetForFunc)
		if f.Inl != nil {
			f.Inl.Properties = funcProps.SerializeToString()
		}
	}
	disableDebugTrace()
}

// TearDown is invoked at the end of the main inlining pass; doing
// function analysis and call site scoring is unlikely to help a lot
// after this point, so nil out fpmap and other globals to reclaim
// storage.
func TearDown() {
	fpmap = nil
	scoreCallsCache.tab = nil
	scoreCallsCache.csl = nil
}

func analyzeFunc(fn *ir.Func, inlineMaxBudget int, nf *nameFinder) *FuncProps {
	if funcInlHeur, ok := fpmap[fn]; ok {
		return funcInlHeur.props
	}
	funcProps, fcstab := computeFuncProps(fn, inlineMaxBudget, nf)
	file, line := fnFileLine(fn)
	entry := fnInlHeur{
		fname: fn.Sym().Name,
		file:  file,
		line:  line,
		props: funcProps,
		cstab: fcstab,
	}
	fn.SetNeverReturns(entry.props.Flags&FuncPropNeverReturns != 0)
	fpmap[fn] = entry
	if fn.Inl != nil && fn.Inl.Properties == "" {
		fn.Inl.Properties = entry.props.SerializeToString()
	}
	return funcProps
}

// revisitInlinability revisits the question of whether to continue to
// treat function 'fn' as an inline candidate based on the set of
// properties we've computed for it. If (for example) it has an
// initial size score of 150 and no interesting properties to speak
// of, then there isn't really any point to moving ahead with it as an
// inline candidate.
func revisitInlinability(fn *ir.Func, funcProps *FuncProps, budgetForFunc func(*ir.Func) int32) {
	if fn.Inl == nil {
		return
	}
	maxAdj := int32(LargestNegativeScoreAdjustment(fn, funcProps))
	budget := budgetForFunc(fn)
	if fn.Inl.Cost+maxAdj > budget {
		fn.Inl = nil
	}
}

// computeFuncProps examines the Go function 'fn' and computes for it
// a function "properties" object, to be used to drive inlining
// heuristics. See comments on the FuncProps type for more info.
func computeFuncProps(fn *ir.Func, inlineMaxBudget int, nf *nameFinder) (*FuncProps, CallSiteTab) {
	if debugTrace&debugTraceFuncs != 0 {
		fmt.Fprintf(os.Stderr, "=-= starting analysis of func %v:\n%+v\n",
			fn, fn)
	}
	funcProps := new(FuncProps)
	ffa := makeFuncFlagsAnalyzer(fn)
	analyzers := []propAnalyzer{ffa}
	analyzers = addResultsAnalyzer(fn, analyzers, funcProps, inlineMaxBudget, nf)
	analyzers = addParamsAnalyzer(fn, analyzers, funcProps, nf)
	runAnalyzersOnFunction(fn, analyzers)
	for _, a := range analyzers {
		a.setResults(funcProps)
	}
	cstab := computeCallSiteTable(fn, fn.Body, nil, ffa.panicPathTable(), 0, nf)
	return funcProps, cstab
}

func runAnalyzersOnFunction(fn *ir.Func, analyzers []propAnalyzer) {
	var doNode func(ir.Node) bool
	doNode = func(n ir.Node) bool {
		for _, a := range analyzers {
			a.nodeVisitPre(n)
		}
		ir.DoChildren(n, doNode)
		for _, a := range analyzers {
			a.nodeVisitPost(n)
		}
		return false
	}
	doNode(fn)
}

func propsForFunc(fn *ir.Func) *FuncProps {
	if funcInlHeur, ok := fpmap[fn]; ok {
		return funcInlHeur.props
	} else if fn.Inl != nil && fn.Inl.Properties != "" {
		// FIXME: considering adding some sort of cache or table
		// for deserialized properties of imported functions.
		return DeserializeFromString(fn.Inl.Properties)
	}
	return nil
}

func fnFileLine(fn *ir.Func) (string, uint) {
	p := base.Ctxt.InnermostPos(fn.Pos())
	return filepath.Base(p.Filename()), p.Line()
}

func Enabled() bool {
	return buildcfg.Experiment.NewInliner || UnitTesting()
}

func UnitTesting() bool {
	return base.Debug.DumpInlFuncProps != "" ||
		base.Debug.DumpInlCallSiteScores != 0
}

// DumpFuncProps computes and caches function properties for the func
// 'fn', writing out a description of the previously computed set of
// properties to the file given in 'dumpfile'. Used for the
// "-d=dumpinlfuncprops=..." command line flag, intended for use
// primarily in unit testing.
func DumpFuncProps(fn *ir.Func, dumpfile string) {
	if fn != nil {
		if fn.OClosure != nil {
			// closures will be processed along with their outer enclosing func.
			return
		}
		captureFuncDumpEntry(fn)
		ir.VisitFuncAndClosures(fn, func(n ir.Node) {
			if clo, ok := n.(*ir.ClosureExpr); ok {
				captureFuncDumpEntry(clo.Func)
			}
		})
	} else {
		emitDumpToFile(dumpfile)
	}
}

// emitDumpToFile writes out the buffer function property dump entries
// to a file, for unit testing. Dump entries need to be sorted by
// definition line, and due to generics we need to account for the
// possibility that several ir.Func's will have the same def line.
func emitDumpToFile(dumpfile string) {
	mode := os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	if dumpfile[0] == '+' {
		dumpfile = dumpfile[1:]
		mode = os.O_WRONLY | os.O_APPEND | os.O_CREATE
	}
	if dumpfile[0] == '%' {
		dumpfile = dumpfile[1:]
		d, b := filepath.Dir(dumpfile), filepath.Base(dumpfile)
		ptag := strings.ReplaceAll(types.LocalPkg.Path, "/", ":")
		dumpfile = d + "/" + ptag + "." + b
	}
	outf, err := os.OpenFile(dumpfile, mode, 0644)
	if err != nil {
		base.Fatalf("opening function props dump file %q: %v\n", dumpfile, err)
	}
	defer outf.Close()
	dumpFilePreamble(outf)

	atline := map[uint]uint{}
	sl := make([]fnInlHeur, 0, len(dumpBuffer))
	for _, e := range dumpBuffer {
		sl = append(sl, e)
		atline[e.line] = atline[e.line] + 1
	}
	sl = sortFnInlHeurSlice(sl)

	prevline := uint(0)
	for _, entry := range sl {
		idx := uint(0)
		if prevline == entry.line {
			idx++
		}
		prevline = entry.line
		atl := atline[entry.line]
		if err := dumpFnPreamble(outf, &entry, nil, idx, atl); err != nil {
			base.Fatalf("function props dump: %v\n", err)
		}
	}
	dumpBuffer = nil
}

// captureFuncDumpEntry grabs the function properties object for 'fn'
// and enqueues it for later dumping. Used for the
// "-d=dumpinlfuncprops=..." command line flag, intended for use
// primarily in unit testing.
func captureFuncDumpEntry(fn *ir.Func) {
	// avoid capturing compiler-generated equality funcs.
	if strings.HasPrefix(fn.Sym().Name, ".eq.") {
		return
	}
	funcInlHeur, ok := fpmap[fn]
	if !ok {
		// Missing entry is expected for functions that are too large
		// to inline. We still want to write out call site scores in
		// this case however.
		funcInlHeur = fnInlHeur{cstab: callSiteTab}
	}
	if dumpBuffer == nil {
		dumpBuffer = make(map[*ir.Func]fnInlHeur)
	}
	if _, ok := dumpBuffer[fn]; ok {
		return
	}
	if debugTrace&debugTraceFuncs != 0 {
		fmt.Fprintf(os.Stderr, "=-= capturing dump for %v:\n", fn)
	}
	dumpBuffer[fn] = funcInlHeur
}

// dumpFilePreamble writes out a file-level preamble for a given
// Go function as part of a function properties dump.
func dumpFilePreamble(w io.Writer) {
	fmt.Fprintf(w, "// DO NOT EDIT (use 'go test -v -update-expected' instead.)\n")
	fmt.Fprintf(w, "// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt\n")
	fmt.Fprintf(w, "// for more information on the format of this file.\n")
	fmt.Fprintf(w, "// %s\n", preambleDelimiter)
}

// dumpFnPreamble writes out a function-level preamble for a given
// Go function as part of a function properties dump. See the
// README.txt file in testdata/props for more on the format of
// this preamble.
func dumpFnPreamble(w io.Writer, funcInlHeur *fnInlHeur, ecst encodedCallSiteTab, idx, atl uint) error {
	fmt.Fprintf(w, "// %s %s %d %d %d\n",
		funcInlHeur.file, funcInlHeur.fname, funcInlHeur.line, idx, atl)
	// emit props as comments, followed by delimiter
	fmt.Fprintf(w, "%s// %s\n", funcInlHeur.props.ToString("// "), comDelimiter)
	data, err := json.Marshal(funcInlHeur.props)
	if err != nil {
		return fmt.Errorf("marshall error %v\n", err)
	}
	fmt.Fprintf(w, "// %s\n", string(data))
	dumpCallSiteComments(w, funcInlHeur.cstab, ecst)
	fmt.Fprintf(w, "// %s\n", fnDelimiter)
	return nil
}

// sortFnInlHeurSlice sorts a slice of fnInlHeur based on
// the starting line of the function definition, then by name.
func sortFnInlHeurSlice(sl []fnInlHeur) []fnInlHeur {
	sort.SliceStable(sl, func(i, j int) bool {
		if sl[i].line != sl[j].line {
			return sl[i].line < sl[j].line
		}
		return sl[i].fname < sl[j].fname
	})
	return sl
}

// delimiters written to various preambles to make parsing of
// dumps easier.
const preambleDelimiter = "<endfilepreamble>"
const fnDelimiter = "<endfuncpreamble>"
const comDelimiter = "<endpropsdump>"
const csDelimiter = "<endcallsites>"

// dumpBuffer stores up function properties dumps when
// "-d=dumpinlfuncprops=..." is in effect.
var dumpBuffer map[*ir.Func]fnInlHeur
