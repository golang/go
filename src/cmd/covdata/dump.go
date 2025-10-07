// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This file contains functions and apis to support the "go tool
// covdata" sub-commands that relate to dumping text format summaries
// and reports: "pkglist", "func",  "debugdump", "percent", and
// "textfmt".

import (
	"flag"
	"fmt"
	"internal/coverage"
	"internal/coverage/calloc"
	"internal/coverage/cformat"
	"internal/coverage/cmerge"
	"internal/coverage/decodecounter"
	"internal/coverage/decodemeta"
	"internal/coverage/pods"
	"os"
	"slices"
	"strings"
)

var textfmtoutflag *string
var liveflag *bool

func makeDumpOp(cmd string) covOperation {
	if cmd == textfmtMode || cmd == percentMode {
		textfmtoutflag = flag.String("o", "", "Output text format to file")
	}
	if cmd == debugDumpMode {
		liveflag = flag.Bool("live", false, "Select only live (executed) functions for dump output.")
	}
	d := &dstate{
		cmd: cmd,
		cm:  &cmerge.Merger{},
	}
	// For these modes (percent, pkglist, func, etc), use a relaxed
	// policy when it comes to counter mode clashes. For a percent
	// report, for example, we only care whether a given line is
	// executed at least once, so it's ok to (effectively) merge
	// together runs derived from different counter modes.
	if d.cmd == percentMode || d.cmd == funcMode || d.cmd == pkglistMode {
		d.cm.SetModeMergePolicy(cmerge.ModeMergeRelaxed)
	}
	if d.cmd == pkglistMode {
		d.pkgpaths = make(map[string]struct{})
	}
	return d
}

// dstate encapsulates state and provides methods for implementing
// various dump operations. Specifically, dstate implements the
// CovDataVisitor interface, and is designed to be used in
// concert with the CovDataReader utility, which abstracts away most
// of the grubby details of reading coverage data files.
type dstate struct {
	// for batch allocation of counter arrays
	calloc.BatchCounterAlloc

	// counter merging state + methods
	cm *cmerge.Merger

	// counter data formatting helper
	format *cformat.Formatter

	// 'mm' stores values read from a counter data file; the pkfunc key
	// is a pkgid/funcid pair that uniquely identifies a function in
	// instrumented application.
	mm map[pkfunc]decodecounter.FuncPayload

	// pkm maps package ID to the number of functions in the package
	// with that ID. It is used to report inconsistencies in counter
	// data (for example, a counter data entry with pkgid=N funcid=10
	// where package N only has 3 functions).
	pkm map[uint32]uint32

	// pkgpaths records all package import paths encountered while
	// visiting coverage data files (used to implement the "pkglist"
	// subcommand).
	pkgpaths map[string]struct{}

	// Current package name and import path.
	pkgName       string
	pkgImportPath string

	// Module path for current package (may be empty).
	modulePath string

	// Dump subcommand (ex: "textfmt", "debugdump", etc).
	cmd string

	// File to which we will write text format output, if enabled.
	textfmtoutf *os.File

	// Total and covered statements (used by "debugdump" subcommand).
	totalStmts, coveredStmts int

	// Records whether preamble has been emitted for current pkg
	// (used when in "debugdump" mode)
	preambleEmitted bool
}

func (d *dstate) Usage(msg string) {
	if len(msg) > 0 {
		fmt.Fprintf(os.Stderr, "error: %s\n", msg)
	}
	fmt.Fprintf(os.Stderr, "usage: go tool covdata %s -i=<directories>\n\n", d.cmd)
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, "\nExamples:\n\n")
	switch d.cmd {
	case pkglistMode:
		fmt.Fprintf(os.Stderr, "  go tool covdata pkglist -i=dir1,dir2\n\n")
		fmt.Fprintf(os.Stderr, "  \treads coverage data files from dir1+dirs2\n")
		fmt.Fprintf(os.Stderr, "  \tand writes out a list of the import paths\n")
		fmt.Fprintf(os.Stderr, "  \tof all compiled packages.\n")
	case textfmtMode:
		fmt.Fprintf(os.Stderr, "  go tool covdata textfmt -i=dir1,dir2 -o=out.txt\n\n")
		fmt.Fprintf(os.Stderr, "  \tmerges data from input directories dir1+dir2\n")
		fmt.Fprintf(os.Stderr, "  \tand emits text format into file 'out.txt'\n")
	case percentMode:
		fmt.Fprintf(os.Stderr, "  go tool covdata percent -i=dir1,dir2\n\n")
		fmt.Fprintf(os.Stderr, "  \tmerges data from input directories dir1+dir2\n")
		fmt.Fprintf(os.Stderr, "  \tand emits percentage of statements covered\n\n")
	case funcMode:
		fmt.Fprintf(os.Stderr, "  go tool covdata func -i=dir1,dir2\n\n")
		fmt.Fprintf(os.Stderr, "  \treads coverage data files from dir1+dirs2\n")
		fmt.Fprintf(os.Stderr, "  \tand writes out coverage profile data for\n")
		fmt.Fprintf(os.Stderr, "  \teach function.\n")
	case debugDumpMode:
		fmt.Fprintf(os.Stderr, "  go tool covdata debugdump [flags] -i=dir1,dir2\n\n")
		fmt.Fprintf(os.Stderr, "  \treads coverage data from dir1+dir2 and dumps\n")
		fmt.Fprintf(os.Stderr, "  \tcontents in human-readable form to stdout, for\n")
		fmt.Fprintf(os.Stderr, "  \tdebugging purposes.\n")
	default:
		panic("unexpected")
	}
	Exit(2)
}

// Setup is called once at program startup time to vet flag values
// and do any necessary setup operations.
func (d *dstate) Setup() {
	if *indirsflag == "" {
		d.Usage("select input directories with '-i' option")
	}
	if d.cmd == textfmtMode || (d.cmd == percentMode && *textfmtoutflag != "") {
		if *textfmtoutflag == "" {
			d.Usage("select output file name with '-o' option")
		}
		var err error
		d.textfmtoutf, err = os.Create(*textfmtoutflag)
		if err != nil {
			d.Usage(fmt.Sprintf("unable to open textfmt output file %q: %v", *textfmtoutflag, err))
		}
	}
	if d.cmd == debugDumpMode {
		fmt.Printf("/* WARNING: the format of this dump is not stable and is\n")
		fmt.Printf(" * expected to change from one Go release to the next.\n")
		fmt.Printf(" *\n")
		fmt.Printf(" * produced by:\n")
		args := append([]string{os.Args[0]}, debugDumpMode)
		args = append(args, os.Args[1:]...)
		fmt.Printf(" *\t%s\n", strings.Join(args, " "))
		fmt.Printf(" */\n")
	}
}

func (d *dstate) BeginPod(p pods.Pod) {
	d.mm = make(map[pkfunc]decodecounter.FuncPayload)
}

func (d *dstate) EndPod(p pods.Pod) {
	if d.cmd == debugDumpMode {
		d.cm.ResetModeAndGranularity()
	}
}

func (d *dstate) BeginCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int) {
	dbgtrace(2, "visit counter data file %s dirIdx %d", cdf, dirIdx)
	if d.cmd == debugDumpMode {
		fmt.Printf("data file %s", cdf)
		if cdr.Goos() != "" {
			fmt.Printf(" GOOS=%s", cdr.Goos())
		}
		if cdr.Goarch() != "" {
			fmt.Printf(" GOARCH=%s", cdr.Goarch())
		}
		if len(cdr.OsArgs()) != 0 {
			fmt.Printf("  program args: %+v\n", cdr.OsArgs())
		}
		fmt.Printf("\n")
	}
}

func (d *dstate) EndCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int) {
}

func (d *dstate) VisitFuncCounterData(data decodecounter.FuncPayload) {
	if nf, ok := d.pkm[data.PkgIdx]; !ok || data.FuncIdx > nf {
		warn("func payload inconsistency: id [p=%d,f=%d] nf=%d len(ctrs)=%d in VisitFuncCounterData, ignored", data.PkgIdx, data.FuncIdx, nf, len(data.Counters))
		return
	}
	key := pkfunc{pk: data.PkgIdx, fcn: data.FuncIdx}
	val, found := d.mm[key]

	dbgtrace(5, "ctr visit pk=%d fid=%d found=%v len(val.ctrs)=%d len(data.ctrs)=%d", data.PkgIdx, data.FuncIdx, found, len(val.Counters), len(data.Counters))

	if len(val.Counters) < len(data.Counters) {
		t := val.Counters
		val.Counters = d.AllocateCounters(len(data.Counters))
		copy(val.Counters, t)
	}
	err, overflow := d.cm.MergeCounters(val.Counters, data.Counters)
	if err != nil {
		fatal("%v", err)
	}
	if overflow {
		warn("uint32 overflow during counter merge")
	}
	d.mm[key] = val
}

func (d *dstate) EndCounters() {
}

func (d *dstate) VisitMetaDataFile(mdf string, mfr *decodemeta.CoverageMetaFileReader) {
	newgran := mfr.CounterGranularity()
	newmode := mfr.CounterMode()
	if err := d.cm.SetModeAndGranularity(mdf, newmode, newgran); err != nil {
		fatal("%v", err)
	}
	if d.cmd == debugDumpMode {
		fmt.Printf("Cover mode: %s\n", newmode.String())
		fmt.Printf("Cover granularity: %s\n", newgran.String())
	}
	if d.format == nil {
		d.format = cformat.NewFormatter(mfr.CounterMode())
	}

	// To provide an additional layer of checking when reading counter
	// data, walk the meta-data file to determine the set of legal
	// package/function combinations. This will help catch bugs in the
	// counter file reader.
	d.pkm = make(map[uint32]uint32)
	np := uint32(mfr.NumPackages())
	payload := []byte{}
	for pkIdx := uint32(0); pkIdx < np; pkIdx++ {
		var pd *decodemeta.CoverageMetaDataDecoder
		var err error
		pd, payload, err = mfr.GetPackageDecoder(pkIdx, payload)
		if err != nil {
			fatal("reading pkg %d from meta-file %s: %s", pkIdx, mdf, err)
		}
		d.pkm[pkIdx] = pd.NumFuncs()
	}
}

func (d *dstate) BeginPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32) {
	d.preambleEmitted = false
	d.pkgImportPath = pd.PackagePath()
	d.pkgName = pd.PackageName()
	d.modulePath = pd.ModulePath()
	if d.cmd == pkglistMode {
		d.pkgpaths[d.pkgImportPath] = struct{}{}
	}
	d.format.SetPackage(pd.PackagePath())
}

func (d *dstate) EndPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32) {
}

func (d *dstate) VisitFunc(pkgIdx uint32, fnIdx uint32, fd *coverage.FuncDesc) {
	var counters []uint32
	key := pkfunc{pk: pkgIdx, fcn: fnIdx}
	v, haveCounters := d.mm[key]

	dbgtrace(5, "meta visit pk=%d fid=%d fname=%s file=%s found=%v len(val.ctrs)=%d", pkgIdx, fnIdx, fd.Funcname, fd.Srcfile, haveCounters, len(v.Counters))

	suppressOutput := false
	if haveCounters {
		counters = v.Counters
	} else if d.cmd == debugDumpMode && *liveflag {
		suppressOutput = true
	}

	if d.cmd == debugDumpMode && !suppressOutput {
		if !d.preambleEmitted {
			fmt.Printf("\nPackage path: %s\n", d.pkgImportPath)
			fmt.Printf("Package name: %s\n", d.pkgName)
			fmt.Printf("Module path: %s\n", d.modulePath)
			d.preambleEmitted = true
		}
		fmt.Printf("\nFunc: %s\n", fd.Funcname)
		fmt.Printf("Srcfile: %s\n", fd.Srcfile)
		fmt.Printf("Literal: %v\n", fd.Lit)
	}
	for i := 0; i < len(fd.Units); i++ {
		u := fd.Units[i]
		var count uint32
		if counters != nil {
			count = counters[i]
		}
		d.format.AddUnit(fd.Srcfile, fd.Funcname, fd.Lit, u, count)
		if d.cmd == debugDumpMode && !suppressOutput {
			fmt.Printf("%d: L%d:C%d -- L%d:C%d ",
				i, u.StLine, u.StCol, u.EnLine, u.EnCol)
			if u.Parent != 0 {
				fmt.Printf("Parent:%d = %d\n", u.Parent, count)
			} else {
				fmt.Printf("NS=%d = %d\n", u.NxStmts, count)
			}
		}
		d.totalStmts += int(u.NxStmts)
		if count != 0 {
			d.coveredStmts += int(u.NxStmts)
		}
	}
}

func (d *dstate) Finish() {
	// d.format maybe nil here if the specified input dir was empty.
	if d.format != nil {
		if d.cmd == percentMode {
			d.format.EmitPercent(os.Stdout, nil, "", false, false)
		}
		if d.cmd == funcMode {
			d.format.EmitFuncs(os.Stdout)
		}
		if d.textfmtoutf != nil {
			if err := d.format.EmitTextual(nil, d.textfmtoutf); err != nil {
				fatal("writing to %s: %v", *textfmtoutflag, err)
			}
		}
	}
	if d.textfmtoutf != nil {
		if err := d.textfmtoutf.Close(); err != nil {
			fatal("closing textfmt output file %s: %v", *textfmtoutflag, err)
		}
	}
	if d.cmd == debugDumpMode {
		fmt.Printf("totalStmts: %d coveredStmts: %d\n", d.totalStmts, d.coveredStmts)
	}
	if d.cmd == pkglistMode {
		pkgs := make([]string, 0, len(d.pkgpaths))
		for p := range d.pkgpaths {
			pkgs = append(pkgs, p)
		}
		slices.Sort(pkgs)
		for _, p := range pkgs {
			fmt.Printf("%s\n", p)
		}
	}
}
