// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This file contains functions and apis to support the "subtract" and
// "intersect" subcommands of "go tool covdata".

import (
	"flag"
	"fmt"
	"internal/coverage"
	"internal/coverage/decodecounter"
	"internal/coverage/decodemeta"
	"internal/coverage/pods"
	"os"
	"strings"
)

// makeSubtractIntersectOp creates a subtract or intersect operation.
// 'mode' here must be either "subtract" or "intersect".
func makeSubtractIntersectOp(mode string) covOperation {
	outdirflag = flag.String("o", "", "Output directory to write")
	s := &sstate{
		mode:  mode,
		mm:    newMetaMerge(),
		inidx: -1,
	}
	return s
}

// sstate holds state needed to implement subtraction and intersection
// operations on code coverage data files. This type provides methods
// to implement the CovDataVisitor interface, and is designed to be
// used in concert with the CovDataReader utility, which abstracts
// away most of the grubby details of reading coverage data files.
type sstate struct {
	mm    *metaMerge
	inidx int
	mode  string
	// Used only for intersection; keyed by pkg/fn ID, it keeps track of
	// just the set of functions for which we have data in the current
	// input directory.
	imm map[pkfunc]struct{}
}

func (s *sstate) Usage(msg string) {
	if len(msg) > 0 {
		fmt.Fprintf(os.Stderr, "error: %s\n", msg)
	}
	fmt.Fprintf(os.Stderr, "usage: go tool covdata %s -i=dir1,dir2 -o=<dir>\n\n", s.mode)
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, "\nExamples:\n\n")
	op := "from"
	if s.mode == intersectMode {
		op = "with"
	}
	fmt.Fprintf(os.Stderr, "  go tool covdata %s -i=dir1,dir2 -o=outdir\n\n", s.mode)
	fmt.Fprintf(os.Stderr, "  \t%ss dir2 %s dir1, writing result\n", s.mode, op)
	fmt.Fprintf(os.Stderr, "  \tinto output dir outdir.\n")
	os.Exit(2)
}

func (s *sstate) Setup() {
	if *indirsflag == "" {
		usage("select input directories with '-i' option")
	}
	indirs := strings.Split(*indirsflag, ",")
	if s.mode == subtractMode && len(indirs) != 2 {
		usage("supply exactly two input dirs for subtract operation")
	}
	if *outdirflag == "" {
		usage("select output directory with '-o' option")
	}
}

func (s *sstate) BeginPod(p pods.Pod) {
	s.mm.beginPod()
}

func (s *sstate) EndPod(p pods.Pod) {
	const pcombine = false
	s.mm.endPod(pcombine)
}

func (s *sstate) EndCounters() {
	if s.imm != nil {
		s.pruneCounters()
	}
}

// pruneCounters performs a function-level partial intersection using the
// current POD counter data (s.mm.pod.pmm) and the intersected data from
// PODs in previous dirs (s.imm).
func (s *sstate) pruneCounters() {
	pkeys := make([]pkfunc, 0, len(s.mm.pod.pmm))
	for k := range s.mm.pod.pmm {
		pkeys = append(pkeys, k)
	}
	// Remove anything from pmm not found in imm. We don't need to
	// go the other way (removing things from imm not found in pmm)
	// since we don't add anything to imm if there is no pmm entry.
	for _, k := range pkeys {
		if _, found := s.imm[k]; !found {
			delete(s.mm.pod.pmm, k)
		}
	}
	s.imm = nil
}

func (s *sstate) BeginCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int) {
	dbgtrace(2, "visiting counter data file %s diridx %d", cdf, dirIdx)
	if s.inidx != dirIdx {
		if s.inidx > dirIdx {
			// We're relying on having data files presented in
			// the order they appear in the inputs (e.g. first all
			// data files from input dir 0, then dir 1, etc).
			panic("decreasing dir index, internal error")
		}
		if dirIdx == 0 {
			// No need to keep track of the functions in the first
			// directory, since that info will be replicated in
			// s.mm.pod.pmm.
			s.imm = nil
		} else {
			// We're now starting to visit the Nth directory, N != 0.
			if s.mode == intersectMode {
				if s.imm != nil {
					s.pruneCounters()
				}
				s.imm = make(map[pkfunc]struct{})
			}
		}
		s.inidx = dirIdx
	}
}

func (s *sstate) EndCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int) {
}

func (s *sstate) VisitFuncCounterData(data decodecounter.FuncPayload) {
	key := pkfunc{pk: data.PkgIdx, fcn: data.FuncIdx}

	if *verbflag >= 5 {
		fmt.Printf("ctr visit fid=%d pk=%d inidx=%d data.Counters=%+v\n", data.FuncIdx, data.PkgIdx, s.inidx, data.Counters)
	}

	// If we're processing counter data from the initial (first) input
	// directory, then just install it into the counter data map
	// as usual.
	if s.inidx == 0 {
		s.mm.visitFuncCounterData(data)
		return
	}

	// If we're looking at counter data from a dir other than
	// the first, then perform the intersect/subtract.
	if val, ok := s.mm.pod.pmm[key]; ok {
		if s.mode == subtractMode {
			for i := 0; i < len(data.Counters); i++ {
				if data.Counters[i] != 0 {
					val.Counters[i] = 0
				}
			}
		} else if s.mode == intersectMode {
			s.imm[key] = struct{}{}
			for i := 0; i < len(data.Counters); i++ {
				if data.Counters[i] == 0 {
					val.Counters[i] = 0
				}
			}
		}
	}
}

func (s *sstate) VisitMetaDataFile(mdf string, mfr *decodemeta.CoverageMetaFileReader) {
	if s.mode == intersectMode {
		s.imm = make(map[pkfunc]struct{})
	}
	s.mm.visitMetaDataFile(mdf, mfr)
}

func (s *sstate) BeginPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32) {
	s.mm.visitPackage(pd, pkgIdx, false)
}

func (s *sstate) EndPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32) {
}

func (s *sstate) VisitFunc(pkgIdx uint32, fnIdx uint32, fd *coverage.FuncDesc) {
	s.mm.visitFunc(pkgIdx, fnIdx, fd, s.mode, false)
}

func (s *sstate) Finish() {
}
