// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	debugTraceFuncs = 1 << iota
)

// fnInlHeur contains inline heuristics state information about
// a specific Go function being analyzed/considered by the inliner.
type fnInlHeur struct {
	fname string
	file  string
	line  uint
	props *FuncProps
}

// computeFuncProps examines the Go function 'fn' and computes for it
// a function "properties" object, to be used to drive inlining
// heuristics. See comments on the FuncProps type for more info.
func computeFuncProps(fn *ir.Func) *FuncProps {
	if debugTrace&debugTraceFuncs != 0 {
		fmt.Fprintf(os.Stderr, "=-= starting analysis of func %v:\n%+v\n",
			fn.Sym().Name, fn)
	}
	// implementation stubbed out for now
	return &FuncProps{}
}

func fnFileLine(fn *ir.Func) (string, uint) {
	p := base.Ctxt.InnermostPos(fn.Pos())
	return filepath.Base(p.Filename()), p.Line()
}

// DumpFuncProps computes and caches function properties for the func
// 'fn', or if fn is nil, writes out the cached set of properties to
// the file given in 'dumpfile'. Used for the "-d=dumpinlfuncprops=..."
// command line flag, intended for use primarily in unit testing.
func DumpFuncProps(fn *ir.Func, dumpfile string) {
	if fn != nil {
		captureFuncDumpEntry(fn)
	} else {
		emitDumpToFile(dumpfile)
	}
}

// emitDumpToFile writes out the buffer function property dump entries
// to a file, for unit testing. Dump entries need to be sorted by
// definition line, and due to generics we need to account for the
// possibility that several ir.Func's will have the same def line.
func emitDumpToFile(dumpfile string) {
	outf, err := os.OpenFile(dumpfile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
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
		if err := dumpFnPreamble(outf, &entry, idx, atl); err != nil {
			base.Fatalf("function props dump: %v\n", err)
		}
	}
	dumpBuffer = nil
}

// captureFuncDumpEntry analyzes function 'fn' and adds a entry
// for it to 'dumpBuffer'. Used for unit testing.
func captureFuncDumpEntry(fn *ir.Func) {
	// avoid capturing compiler-generated equality funcs.
	if strings.HasPrefix(fn.Sym().Name, ".eq.") {
		return
	}
	if dumpBuffer == nil {
		dumpBuffer = make(map[*ir.Func]fnInlHeur)
	}
	if _, ok := dumpBuffer[fn]; ok {
		// we can wind up seeing closures multiple times here,
		// so don't add them more than once.
		return
	}
	fp := computeFuncProps(fn)
	file, line := fnFileLine(fn)
	entry := fnInlHeur{
		fname: fn.Sym().Name,
		file:  file,
		line:  line,
		props: fp,
	}
	dumpBuffer[fn] = entry
}

// dumpFilePreamble writes out a file-level preamble for a given
// Go function as part of a function properties dump.
func dumpFilePreamble(w io.Writer) {
	fmt.Fprintf(w, "// DO NOT EDIT (use 'go test -v -update-expected' instead.)\n")
	fmt.Fprintf(w, "// See cmd/compile/internal/inline/inlheur/testdata/props/README.txt\n")
	fmt.Fprintf(w, "// for more information on the format of this file.\n")
	fmt.Fprintf(w, "// %s\n", preambleDelimiter)
}

// dumpFilePreamble writes out a function-level preamble for a given
// Go function as part of a function properties dump. See the
// README.txt file in testdata/props for more on the format of
// this preamble.
func dumpFnPreamble(w io.Writer, fih *fnInlHeur, idx, atl uint) error {
	fmt.Fprintf(w, "// %s %s %d %d %d\n",
		fih.file, fih.fname, fih.line, idx, atl)
	// emit props as comments, followed by delimiter
	fmt.Fprintf(w, "%s// %s\n", fih.props.ToString("// "), comDelimiter)
	data, err := json.Marshal(fih.props)
	if err != nil {
		return fmt.Errorf("marshall error %v\n", err)
	}
	fmt.Fprintf(w, "// %s\n// %s\n", string(data), fnDelimiter)
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

// dumpBuffer stores up function properties dumps when
// "-d=dumpinlfuncprops=..." is in effect.
var dumpBuffer map[*ir.Func]fnInlHeur
