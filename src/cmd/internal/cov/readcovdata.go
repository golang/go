// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cov

import (
	"cmd/internal/bio"
	"fmt"
	"internal/coverage"
	"internal/coverage/decodecounter"
	"internal/coverage/decodemeta"
	"internal/coverage/pods"
	"io"
	"os"
)

// CovDataReader is a general-purpose helper/visitor object for
// reading coverage data files in a structured way. Clients create a
// CovDataReader to process a given collection of coverage data file
// directories, then pass in a visitor object with methods that get
// invoked at various important points. CovDataReader is intended
// to facilitate common coverage data file operations such as
// merging or intersecting data files, analyzing data files, or
// dumping data files.
type CovDataReader struct {
	vis            CovDataVisitor
	indirs         []string
	matchpkg       func(name string) bool
	flags          CovDataReaderFlags
	err            error
	verbosityLevel int
}

// MakeCovDataReader creates a CovDataReader object to process the
// given set of input directories. Here 'vis' is a visitor object
// providing methods to be invoked as we walk through the data,
// 'indirs' is the set of coverage data directories to examine,
// 'verbosityLevel' controls the level of debugging trace messages
// (zero for off, higher for more output), 'flags' stores flags that
// indicate what to do if errors are detected, and 'matchpkg' is a
// caller-provided function that can be used to select specific
// packages by name (if nil, then all packages are included).
func MakeCovDataReader(vis CovDataVisitor, indirs []string, verbosityLevel int, flags CovDataReaderFlags, matchpkg func(name string) bool) *CovDataReader {
	return &CovDataReader{
		vis:            vis,
		indirs:         indirs,
		matchpkg:       matchpkg,
		verbosityLevel: verbosityLevel,
		flags:          flags,
	}
}

// CovDataVisitor defines hooks for clients of CovDataReader. When the
// coverage data reader makes its way through a coverage meta-data
// file and counter data files, it will invoke the methods below to
// hand off info to the client. The normal sequence of expected
// visitor method invocations is:
//
//	for each pod P {
//		BeginPod(p)
//		let MF be the meta-data file for P
//		VisitMetaDataFile(MF)
//		for each counter data file D in P {
//			BeginCounterDataFile(D)
//			for each live function F in D {
//				VisitFuncCounterData(F)
//			}
//			EndCounterDataFile(D)
//		}
//		EndCounters(MF)
//		for each package PK in MF {
//			BeginPackage(PK)
//			if <PK matched according to package pattern and/or modpath> {
//				for each function PF in PK {
//					VisitFunc(PF)
//				}
//			}
//			EndPackage(PK)
//		}
//		EndPod(p)
//	}
//	Finish()

type CovDataVisitor interface {
	// Invoked at the start and end of a given pod (a pod here is a
	// specific coverage meta-data files with the counter data files
	// that correspond to it).
	BeginPod(p pods.Pod)
	EndPod(p pods.Pod)

	// Invoked when the reader is starting to examine the meta-data
	// file for a pod. Here 'mdf' is the path of the file, and 'mfr'
	// is an open meta-data reader.
	VisitMetaDataFile(mdf string, mfr *decodemeta.CoverageMetaFileReader)

	// Invoked when the reader processes a counter data file, first
	// the 'begin' method at the start, then the 'end' method when
	// we're done with the file.
	BeginCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int)
	EndCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int)

	// Invoked once for each live function in the counter data file.
	VisitFuncCounterData(payload decodecounter.FuncPayload)

	// Invoked when we've finished processing the counter files in a
	// POD (e.g. no more calls to VisitFuncCounterData).
	EndCounters()

	// Invoked for each package in the meta-data file for the pod,
	// first the 'begin' method when processing of the package starts,
	// then the 'end' method when we're done
	BeginPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32)
	EndPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32)

	// Invoked for each function  the package being visited.
	VisitFunc(pkgIdx uint32, fnIdx uint32, fd *coverage.FuncDesc)

	// Invoked when all counter + meta-data file processing is complete.
	Finish()
}

type CovDataReaderFlags uint32

const (
	CovDataReaderNoFlags CovDataReaderFlags = 0
	PanicOnError                            = 1 << iota
	PanicOnWarning
)

func (r *CovDataReader) Visit() error {
	podlist, err := pods.CollectPods(r.indirs, false)
	if err != nil {
		return fmt.Errorf("reading inputs: %v", err)
	}
	if len(podlist) == 0 {
		r.warn("no applicable files found in input directories")
	}
	for _, p := range podlist {
		if err := r.visitPod(p); err != nil {
			return err
		}
	}
	r.vis.Finish()
	return nil
}

func (r *CovDataReader) verb(vlevel int, s string, a ...interface{}) {
	if r.verbosityLevel >= vlevel {
		fmt.Fprintf(os.Stderr, s, a...)
		fmt.Fprintf(os.Stderr, "\n")
	}
}

func (r *CovDataReader) warn(s string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, "warning: ")
	fmt.Fprintf(os.Stderr, s, a...)
	fmt.Fprintf(os.Stderr, "\n")
	if (r.flags & PanicOnWarning) != 0 {
		panic("unexpected warning")
	}
}

func (r *CovDataReader) fatal(s string, a ...interface{}) error {
	if r.err != nil {
		return nil
	}
	errstr := "error: " + fmt.Sprintf(s, a...) + "\n"
	if (r.flags & PanicOnError) != 0 {
		fmt.Fprintf(os.Stderr, "%s", errstr)
		panic("fatal error")
	}
	r.err = fmt.Errorf("%s", errstr)
	return r.err
}

// visitPod examines a coverage data 'pod', that is, a meta-data file and
// zero or more counter data files that refer to that meta-data file.
func (r *CovDataReader) visitPod(p pods.Pod) error {
	r.verb(1, "visiting pod: metafile %s with %d counter files",
		p.MetaFile, len(p.CounterDataFiles))
	r.vis.BeginPod(p)

	// Open meta-file
	f, err := os.Open(p.MetaFile)
	if err != nil {
		return r.fatal("unable to open meta-file %s", p.MetaFile)
	}
	defer f.Close()
	br := bio.NewReader(f)
	fi, err := f.Stat()
	if err != nil {
		return r.fatal("unable to stat metafile %s: %v", p.MetaFile, err)
	}
	fileView := br.SliceRO(uint64(fi.Size()))
	br.MustSeek(0, io.SeekStart)

	r.verb(1, "fileView for pod is length %d", len(fileView))

	var mfr *decodemeta.CoverageMetaFileReader
	mfr, err = decodemeta.NewCoverageMetaFileReader(f, fileView)
	if err != nil {
		return r.fatal("decoding meta-file %s: %s", p.MetaFile, err)
	}
	r.vis.VisitMetaDataFile(p.MetaFile, mfr)

	processCounterDataFile := func(cdf string, k int) error {
		cf, err := os.Open(cdf)
		if err != nil {
			return r.fatal("opening counter data file %s: %s", cdf, err)
		}
		defer cf.Close()
		var mr *MReader
		mr, err = NewMreader(cf)
		if err != nil {
			return r.fatal("creating reader for counter data file %s: %s", cdf, err)
		}
		var cdr *decodecounter.CounterDataReader
		cdr, err = decodecounter.NewCounterDataReader(cdf, mr)
		if err != nil {
			return r.fatal("reading counter data file %s: %s", cdf, err)
		}
		r.vis.BeginCounterDataFile(cdf, cdr, p.Origins[k])
		var data decodecounter.FuncPayload
		for {
			ok, err := cdr.NextFunc(&data)
			if err != nil {
				return r.fatal("reading counter data file %s: %v", cdf, err)
			}
			if !ok {
				break
			}
			r.vis.VisitFuncCounterData(data)
		}
		r.vis.EndCounterDataFile(cdf, cdr, p.Origins[k])
		return nil
	}

	// Read counter data files.
	for k, cdf := range p.CounterDataFiles {
		if err := processCounterDataFile(cdf, k); err != nil {
			return err
		}
	}
	r.vis.EndCounters()

	// NB: packages in the meta-file will be in dependency order (basically
	// the order in which init files execute). Do we want an additional sort
	// pass here, say by packagepath?
	np := uint32(mfr.NumPackages())
	payload := []byte{}
	for pkIdx := uint32(0); pkIdx < np; pkIdx++ {
		var pd *decodemeta.CoverageMetaDataDecoder
		pd, payload, err = mfr.GetPackageDecoder(pkIdx, payload)
		if err != nil {
			return r.fatal("reading pkg %d from meta-file %s: %s", pkIdx, p.MetaFile, err)
		}
		r.processPackage(p.MetaFile, pd, pkIdx)
	}
	r.vis.EndPod(p)

	return nil
}

func (r *CovDataReader) processPackage(mfname string, pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32) error {
	if r.matchpkg != nil {
		if !r.matchpkg(pd.PackagePath()) {
			return nil
		}
	}
	r.vis.BeginPackage(pd, pkgIdx)
	nf := pd.NumFuncs()
	var fd coverage.FuncDesc
	for fidx := uint32(0); fidx < nf; fidx++ {
		if err := pd.ReadFunc(fidx, &fd); err != nil {
			return r.fatal("reading meta-data file %s: %v", mfname, err)
		}
		r.vis.VisitFunc(pkgIdx, fidx, &fd)
	}
	r.vis.EndPackage(pd, pkgIdx)
	return nil
}
