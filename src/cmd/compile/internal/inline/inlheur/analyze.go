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
	"internal/goexperiment"
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
	setResults(fp *FuncProps)
}

// fnInlHeur contains inline heuristics state information about a
// specific Go function being analyzed/considered by the inliner. Note
// that in addition to constructing a fnInlHeur object by analyzing a
// specific *ir.Func, there is also code in the test harness
// (funcprops_test.go) that builds up fnInlHeur's by reading in and
// parsing a dump. This is the reason why we have file/fname/line
// fields below instead of just an *ir.Func field.
type fnInlHeur struct {
	fname string
	file  string
	line  uint
	props *FuncProps
	cstab CallSiteTab
}

var fpmap = map[*ir.Func]fnInlHeur{}

func AnalyzeFunc(fn *ir.Func, canInline func(*ir.Func)) *FuncProps {
	if fih, ok := fpmap[fn]; ok {
		return fih.props
	}
	fp, fcstab := computeFuncProps(fn, canInline)
	file, line := fnFileLine(fn)
	entry := fnInlHeur{
		fname: fn.Sym().Name,
		file:  file,
		line:  line,
		props: fp,
		cstab: fcstab,
	}
	// Merge this functions call sites into the package level table.
	if err := cstab.merge(fcstab); err != nil {
		base.FatalfAt(fn.Pos(), "%v", err)
	}
	fn.SetNeverReturns(entry.props.Flags&FuncPropNeverReturns != 0)
	fpmap[fn] = entry
	if fn.Inl != nil && fn.Inl.Properties == "" {
		fn.Inl.Properties = entry.props.SerializeToString()
	}
	return fp
}

// computeFuncProps examines the Go function 'fn' and computes for it
// a function "properties" object, to be used to drive inlining
// heuristics. See comments on the FuncProps type for more info.
func computeFuncProps(fn *ir.Func, canInline func(*ir.Func)) (*FuncProps, CallSiteTab) {
	enableDebugTraceIfEnv()
	if debugTrace&debugTraceFuncs != 0 {
		fmt.Fprintf(os.Stderr, "=-= starting analysis of func %v:\n%+v\n",
			fn.Sym().Name, fn)
	}
	ra := makeResultsAnalyzer(fn, canInline)
	pa := makeParamsAnalyzer(fn)
	ffa := makeFuncFlagsAnalyzer(fn)
	analyzers := []propAnalyzer{ffa, ra, pa}
	fp := new(FuncProps)
	runAnalyzersOnFunction(fn, analyzers)
	for _, a := range analyzers {
		a.setResults(fp)
	}
	// Now build up a partial table of callsites for this func.
	cstab := computeCallSiteTable(fn, ffa.panicPathTable())
	disableDebugTrace()
	return fp, cstab
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
	if fih, ok := fpmap[fn]; ok {
		return fih.props
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

func UnitTesting() bool {
	return base.Debug.DumpInlFuncProps != ""
}

// DumpFuncProps computes and caches function properties for the func
// 'fn' and any closures it contains, or if fn is nil, it writes out the
// cached set of properties to the file given in 'dumpfile'. Used for
// the "-d=dumpinlfuncprops=..." command line flag, intended for use
// primarily in unit testing.
func DumpFuncProps(fn *ir.Func, dumpfile string, canInline func(*ir.Func)) {
	if fn != nil {
		enableDebugTraceIfEnv()
		dmp := func(fn *ir.Func) {
			if !goexperiment.NewInliner {
				ScoreCalls(fn)
			}
			captureFuncDumpEntry(fn, canInline)
		}
		captureFuncDumpEntry(fn, canInline)
		dmp(fn)
		ir.Visit(fn, func(n ir.Node) {
			if clo, ok := n.(*ir.ClosureExpr); ok {
				dmp(clo.Func)
			}
		})
		disableDebugTrace()
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
func captureFuncDumpEntry(fn *ir.Func, canInline func(*ir.Func)) {
	// avoid capturing compiler-generated equality funcs.
	if strings.HasPrefix(fn.Sym().Name, ".eq.") {
		return
	}
	fih, ok := fpmap[fn]
	// Props object should already be present, unless this is a
	// directly recursive routine.
	if !ok {
		AnalyzeFunc(fn, canInline)
		fih = fpmap[fn]
		if fn.Inl != nil && fn.Inl.Properties == "" {
			fn.Inl.Properties = fih.props.SerializeToString()
		}
	}
	if dumpBuffer == nil {
		dumpBuffer = make(map[*ir.Func]fnInlHeur)
	}
	if _, ok := dumpBuffer[fn]; ok {
		// we can wind up seeing closures multiple times here,
		// so don't add them more than once.
		return
	}
	if debugTrace&debugTraceFuncs != 0 {
		fmt.Fprintf(os.Stderr, "=-= capturing dump for %v:\n", fn)
	}
	dumpBuffer[fn] = fih
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
func dumpFnPreamble(w io.Writer, fih *fnInlHeur, ecst encodedCallSiteTab, idx, atl uint) error {
	fmt.Fprintf(w, "// %s %s %d %d %d\n",
		fih.file, fih.fname, fih.line, idx, atl)
	// emit props as comments, followed by delimiter
	fmt.Fprintf(w, "%s// %s\n", fih.props.ToString("// "), comDelimiter)
	data, err := json.Marshal(fih.props)
	if err != nil {
		return fmt.Errorf("marshall error %v\n", err)
	}
	fmt.Fprintf(w, "// %s\n", string(data))
	dumpCallSiteComments(w, fih.cstab, ecst)
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
