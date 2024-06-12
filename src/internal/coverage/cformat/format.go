// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cformat

// This package provides apis for producing human-readable summaries
// of coverage data (e.g. a coverage percentage for a given package or
// set of packages) and for writing data in the legacy test format
// emitted by "go test -coverprofile=<outfile>".
//
// The model for using these apis is to create a Formatter object,
// then make a series of calls to SetPackage and AddUnit passing in
// data read from coverage meta-data and counter-data files. E.g.
//
//		myformatter := cformat.NewFormatter()
//		...
//		for each package P in meta-data file: {
//			myformatter.SetPackage(P)
//			for each function F in P: {
//				for each coverable unit U in F: {
//					myformatter.AddUnit(U)
//				}
//			}
//		}
//		myformatter.EmitPercent(os.Stdout, nil, "", true, true)
//		myformatter.EmitTextual(somefile)
//
// These apis are linked into tests that are built with "-cover", and
// called at the end of test execution to produce text output or
// emit coverage percentages.

import (
	"cmp"
	"fmt"
	"internal/coverage"
	"internal/coverage/cmerge"
	"io"
	"slices"
	"strings"
	"text/tabwriter"
)

type Formatter struct {
	// Maps import path to package state.
	pm map[string]*pstate
	// Records current package being visited.
	pkg string
	// Pointer to current package state.
	p *pstate
	// Counter mode.
	cm coverage.CounterMode
}

// pstate records package-level coverage data state:
// - a table of functions (file/fname/literal)
// - a map recording the index/ID of each func encountered so far
// - a table storing execution count for the coverable units in each func
type pstate struct {
	// slice of unique functions
	funcs []fnfile
	// maps function to index in slice above (index acts as function ID)
	funcTable map[fnfile]uint32

	// A table storing coverage counts for each coverable unit.
	unitTable map[extcu]uint32
}

// extcu encapsulates a coverable unit within some function.
type extcu struct {
	fnfid uint32 // index into p.funcs slice
	coverage.CoverableUnit
}

// fnfile is a function-name/file-name tuple.
type fnfile struct {
	file  string
	fname string
	lit   bool
}

func NewFormatter(cm coverage.CounterMode) *Formatter {
	return &Formatter{
		pm: make(map[string]*pstate),
		cm: cm,
	}
}

// SetPackage tells the formatter that we're about to visit the
// coverage data for the package with the specified import path.
// Note that it's OK to call SetPackage more than once with the
// same import path; counter data values will be accumulated.
func (fm *Formatter) SetPackage(importpath string) {
	if importpath == fm.pkg {
		return
	}
	fm.pkg = importpath
	ps, ok := fm.pm[importpath]
	if !ok {
		ps = new(pstate)
		fm.pm[importpath] = ps
		ps.unitTable = make(map[extcu]uint32)
		ps.funcTable = make(map[fnfile]uint32)
	}
	fm.p = ps
}

// AddUnit passes info on a single coverable unit (file, funcname,
// literal flag, range of lines, and counter value) to the formatter.
// Counter values will be accumulated where appropriate.
func (fm *Formatter) AddUnit(file string, fname string, isfnlit bool, unit coverage.CoverableUnit, count uint32) {
	if fm.p == nil {
		panic("AddUnit invoked before SetPackage")
	}
	fkey := fnfile{file: file, fname: fname, lit: isfnlit}
	idx, ok := fm.p.funcTable[fkey]
	if !ok {
		idx = uint32(len(fm.p.funcs))
		fm.p.funcs = append(fm.p.funcs, fkey)
		fm.p.funcTable[fkey] = idx
	}
	ukey := extcu{fnfid: idx, CoverableUnit: unit}
	pcount := fm.p.unitTable[ukey]
	var result uint32
	if fm.cm == coverage.CtrModeSet {
		if count != 0 || pcount != 0 {
			result = 1
		}
	} else {
		// Use saturating arithmetic.
		result, _ = cmerge.SaturatingAdd(pcount, count)
	}
	fm.p.unitTable[ukey] = result
}

// sortUnits sorts a slice of extcu objects in a package according to
// source position information (e.g. file and line). Note that we don't
// include function name as part of the sorting criteria, the thinking
// being that is better to provide things in the original source order.
func (p *pstate) sortUnits(units []extcu) {
	slices.SortFunc(units, func(ui, uj extcu) int {
		ifile := p.funcs[ui.fnfid].file
		jfile := p.funcs[uj.fnfid].file
		if r := strings.Compare(ifile, jfile); r != 0 {
			return r
		}
		// NB: not taking function literal flag into account here (no
		// need, since other fields are guaranteed to be distinct).
		if r := cmp.Compare(ui.StLine, uj.StLine); r != 0 {
			return r
		}
		if r := cmp.Compare(ui.EnLine, uj.EnLine); r != 0 {
			return r
		}
		if r := cmp.Compare(ui.StCol, uj.StCol); r != 0 {
			return r
		}
		if r := cmp.Compare(ui.EnCol, uj.EnCol); r != 0 {
			return r
		}
		return cmp.Compare(ui.NxStmts, uj.NxStmts)
	})
}

// EmitTextual writes the accumulated coverage data in the legacy
// cmd/cover text format to the writer 'w'. We sort the data items by
// importpath, source file, and line number before emitting (this sorting
// is not explicitly mandated by the format, but seems like a good idea
// for repeatable/deterministic dumps).
func (fm *Formatter) EmitTextual(w io.Writer) error {
	if fm.cm == coverage.CtrModeInvalid {
		panic("internal error, counter mode unset")
	}
	if _, err := fmt.Fprintf(w, "mode: %s\n", fm.cm.String()); err != nil {
		return err
	}
	pkgs := make([]string, 0, len(fm.pm))
	for importpath := range fm.pm {
		pkgs = append(pkgs, importpath)
	}
	slices.Sort(pkgs)
	for _, importpath := range pkgs {
		p := fm.pm[importpath]
		units := make([]extcu, 0, len(p.unitTable))
		for u := range p.unitTable {
			units = append(units, u)
		}
		p.sortUnits(units)
		for _, u := range units {
			count := p.unitTable[u]
			file := p.funcs[u.fnfid].file
			if _, err := fmt.Fprintf(w, "%s:%d.%d,%d.%d %d %d\n",
				file, u.StLine, u.StCol,
				u.EnLine, u.EnCol, u.NxStmts, count); err != nil {
				return err
			}
		}
	}
	return nil
}

// EmitPercent writes out a "percentage covered" string to the writer
// 'w', selecting the set of packages in 'pkgs' and suffixing the
// printed string with 'inpkgs'.
func (fm *Formatter) EmitPercent(w io.Writer, pkgs []string, inpkgs string, noteEmpty bool, aggregate bool) error {
	if len(pkgs) == 0 {
		pkgs = make([]string, 0, len(fm.pm))
		for importpath := range fm.pm {
			pkgs = append(pkgs, importpath)
		}
	}

	rep := func(cov, tot uint64) error {
		if tot != 0 {
			if _, err := fmt.Fprintf(w, "coverage: %.1f%% of statements%s\n",
				100.0*float64(cov)/float64(tot), inpkgs); err != nil {
				return err
			}
		} else if noteEmpty {
			if _, err := fmt.Fprintf(w, "coverage: [no statements]\n"); err != nil {
				return err
			}
		}
		return nil
	}

	slices.Sort(pkgs)
	var totalStmts, coveredStmts uint64
	for _, importpath := range pkgs {
		p := fm.pm[importpath]
		if p == nil {
			continue
		}
		if !aggregate {
			totalStmts, coveredStmts = 0, 0
		}
		for unit, count := range p.unitTable {
			nx := uint64(unit.NxStmts)
			totalStmts += nx
			if count != 0 {
				coveredStmts += nx
			}
		}
		if !aggregate {
			if _, err := fmt.Fprintf(w, "\t%s\t\t", importpath); err != nil {
				return err
			}
			if err := rep(coveredStmts, totalStmts); err != nil {
				return err
			}
		}
	}
	if aggregate {
		if err := rep(coveredStmts, totalStmts); err != nil {
			return err
		}
	}

	return nil
}

// EmitFuncs writes out a function-level summary to the writer 'w'. A
// note on handling function literals: although we collect coverage
// data for unnamed literals, it probably does not make sense to
// include them in the function summary since there isn't any good way
// to name them (this is also consistent with the legacy cmd/cover
// implementation). We do want to include their counts in the overall
// summary however.
func (fm *Formatter) EmitFuncs(w io.Writer) error {
	if fm.cm == coverage.CtrModeInvalid {
		panic("internal error, counter mode unset")
	}
	perc := func(covered, total uint64) float64 {
		if total == 0 {
			total = 1
		}
		return 100.0 * float64(covered) / float64(total)
	}
	tabber := tabwriter.NewWriter(w, 1, 8, 1, '\t', 0)
	defer tabber.Flush()
	allStmts := uint64(0)
	covStmts := uint64(0)

	pkgs := make([]string, 0, len(fm.pm))
	for importpath := range fm.pm {
		pkgs = append(pkgs, importpath)
	}
	slices.Sort(pkgs)

	// Emit functions for each package, sorted by import path.
	for _, importpath := range pkgs {
		p := fm.pm[importpath]
		if len(p.unitTable) == 0 {
			continue
		}
		units := make([]extcu, 0, len(p.unitTable))
		for u := range p.unitTable {
			units = append(units, u)
		}

		// Within a package, sort the units, then walk through the
		// sorted array. Each time we hit a new function, emit the
		// summary entry for the previous function, then make one last
		// emit call at the end of the loop.
		p.sortUnits(units)
		fname := ""
		ffile := ""
		flit := false
		var fline uint32
		var cstmts, tstmts uint64
		captureFuncStart := func(u extcu) {
			fname = p.funcs[u.fnfid].fname
			ffile = p.funcs[u.fnfid].file
			flit = p.funcs[u.fnfid].lit
			fline = u.StLine
		}
		emitFunc := func(u extcu) error {
			// Don't emit entries for function literals (see discussion
			// in function header comment above).
			if !flit {
				if _, err := fmt.Fprintf(tabber, "%s:%d:\t%s\t%.1f%%\n",
					ffile, fline, fname, perc(cstmts, tstmts)); err != nil {
					return err
				}
			}
			captureFuncStart(u)
			allStmts += tstmts
			covStmts += cstmts
			tstmts = 0
			cstmts = 0
			return nil
		}
		for k, u := range units {
			if k == 0 {
				captureFuncStart(u)
			} else {
				if fname != p.funcs[u.fnfid].fname {
					// New function; emit entry for previous one.
					if err := emitFunc(u); err != nil {
						return err
					}
				}
			}
			tstmts += uint64(u.NxStmts)
			count := p.unitTable[u]
			if count != 0 {
				cstmts += uint64(u.NxStmts)
			}
		}
		if err := emitFunc(extcu{}); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(tabber, "%s\t%s\t%.1f%%\n",
		"total", "(statements)", perc(covStmts, allStmts)); err != nil {
		return err
	}
	return nil
}
