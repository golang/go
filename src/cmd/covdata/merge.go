// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This file contains functions and apis to support the "merge"
// subcommand of "go tool covdata".

import (
	"flag"
	"fmt"
	"internal/coverage"
	"internal/coverage/cmerge"
	"internal/coverage/decodecounter"
	"internal/coverage/decodemeta"
	"internal/coverage/pods"
	"os"
)

var outdirflag *string
var pcombineflag *bool

func makeMergeOp() covOperation {
	outdirflag = flag.String("o", "", "Output directory to write")
	pcombineflag = flag.Bool("pcombine", false, "Combine profiles derived from distinct program executables")
	m := &mstate{
		mm: newMetaMerge(),
	}
	return m
}

// mstate encapsulates state and provides methods for implementing the
// merge operation. This type implements the CovDataVisitor interface,
// and is designed to be used in concert with the CovDataReader
// utility, which abstracts away most of the grubby details of reading
// coverage data files. Most of the heavy lifting for merging is done
// using apis from 'metaMerge' (this is mainly a wrapper around that
// functionality).
type mstate struct {
	mm *metaMerge
}

func (m *mstate) Usage(msg string) {
	if len(msg) > 0 {
		fmt.Fprintf(os.Stderr, "error: %s\n", msg)
	}
	fmt.Fprintf(os.Stderr, "usage: go tool covdata merge -i=<directories> -o=<dir>\n\n")
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, "\nExamples:\n\n")
	fmt.Fprintf(os.Stderr, "  go tool covdata merge -i=dir1,dir2,dir3 -o=outdir\n\n")
	fmt.Fprintf(os.Stderr, "  \tmerges all files in dir1/dir2/dir3\n")
	fmt.Fprintf(os.Stderr, "  \tinto output dir outdir\n")
	Exit(2)
}

func (m *mstate) Setup() {
	if *indirsflag == "" {
		m.Usage("select input directories with '-i' option")
	}
	if *outdirflag == "" {
		m.Usage("select output directory with '-o' option")
	}
	m.mm.SetModeMergePolicy(cmerge.ModeMergeRelaxed)
}

func (m *mstate) BeginPod(p pods.Pod) {
	m.mm.beginPod()
}

func (m *mstate) EndPod(p pods.Pod) {
	m.mm.endPod(*pcombineflag)
}

func (m *mstate) BeginCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int) {
	dbgtrace(2, "visit counter data file %s dirIdx %d", cdf, dirIdx)
	m.mm.beginCounterDataFile(cdr)
}

func (m *mstate) EndCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int) {
}

func (m *mstate) VisitFuncCounterData(data decodecounter.FuncPayload) {
	m.mm.visitFuncCounterData(data)
}

func (m *mstate) EndCounters() {
}

func (m *mstate) VisitMetaDataFile(mdf string, mfr *decodemeta.CoverageMetaFileReader) {
	m.mm.visitMetaDataFile(mdf, mfr)
}

func (m *mstate) BeginPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32) {
	dbgtrace(3, "VisitPackage(pk=%d path=%s)", pkgIdx, pd.PackagePath())
	m.mm.visitPackage(pd, pkgIdx, *pcombineflag)
}

func (m *mstate) EndPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32) {
}

func (m *mstate) VisitFunc(pkgIdx uint32, fnIdx uint32, fd *coverage.FuncDesc) {
	m.mm.visitFunc(pkgIdx, fnIdx, fd, mergeMode, *pcombineflag)
}

func (m *mstate) Finish() {
	if *pcombineflag {
		finalHash := m.mm.emitMeta(*outdirflag, true)
		m.mm.emitCounters(*outdirflag, finalHash)
	}
}
