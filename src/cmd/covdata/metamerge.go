// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This file contains functions and apis that support merging of
// meta-data information.  It helps implement the "merge", "subtract",
// and "intersect" subcommands.

import (
	"crypto/md5"
	"fmt"
	"internal/coverage"
	"internal/coverage/calloc"
	"internal/coverage/cmerge"
	"internal/coverage/decodecounter"
	"internal/coverage/decodemeta"
	"internal/coverage/encodecounter"
	"internal/coverage/encodemeta"
	"internal/coverage/slicewriter"
	"io"
	"os"
	"path/filepath"
	"sort"
	"time"
	"unsafe"
)

// metaMerge provides state and methods to help manage the process
// of selecting or merging meta data files. There are three cases
// of interest here: the "-pcombine" flag provided by merge, the
// "-pkg" option provided by all merge/subtract/intersect, and
// a regular vanilla merge with no package selection
//
// In the -pcombine case, we're essentially glomming together all the
// meta-data for all packages and all functions, meaning that
// everything we see in a given package needs to be added into the
// meta-data file builder; we emit a single meta-data file at the end
// of the run.
//
// In the -pkg case, we will typically emit a single meta-data file
// per input pod, where that new meta-data file contains entries for
// just the selected packages.
//
// In the third case (vanilla merge with no combining or package
// selection) we can carry over meta-data files without touching them
// at all (only counter data files will be merged).
type metaMerge struct {
	calloc.BatchCounterAlloc
	cmerge.Merger
	// maps package import path to package state
	pkm map[string]*pkstate
	// list of packages
	pkgs []*pkstate
	// current package state
	p *pkstate
	// current pod state
	pod *podstate
	// counter data file osargs/goos/goarch state
	astate *argstate
}

// pkstate
type pkstate struct {
	// index of package within meta-data file.
	pkgIdx uint32
	// this maps function index within the package to counter data payload
	ctab map[uint32]decodecounter.FuncPayload
	// pointer to meta-data blob for package
	mdblob []byte
	// filled in only for -pcombine merges
	*pcombinestate
}

type podstate struct {
	pmm      map[pkfunc]decodecounter.FuncPayload
	mdf      string
	mfr      *decodemeta.CoverageMetaFileReader
	fileHash [16]byte
}

type pkfunc struct {
	pk, fcn uint32
}

// pcombinestate
type pcombinestate struct {
	// Meta-data builder for the package.
	cmdb *encodemeta.CoverageMetaDataBuilder
	// Maps function meta-data hash to new function index in the
	// new version of the package we're building.
	ftab map[[16]byte]uint32
}

func newMetaMerge() *metaMerge {
	return &metaMerge{
		pkm:    make(map[string]*pkstate),
		astate: &argstate{},
	}
}

func (mm *metaMerge) visitMetaDataFile(mdf string, mfr *decodemeta.CoverageMetaFileReader) {
	dbgtrace(2, "visitMetaDataFile(mdf=%s)", mdf)

	// Record meta-data file name.
	mm.pod.mdf = mdf
	// Keep a pointer to the file-level reader.
	mm.pod.mfr = mfr
	// Record file hash.
	mm.pod.fileHash = mfr.FileHash()
	// Counter mode and granularity -- detect and record clashes here.
	newgran := mfr.CounterGranularity()
	newmode := mfr.CounterMode()
	if err := mm.SetModeAndGranularity(mdf, newmode, newgran); err != nil {
		fatal("%v", err)
	}
}

func (mm *metaMerge) beginCounterDataFile(cdr *decodecounter.CounterDataReader) {
	state := argvalues{
		osargs: cdr.OsArgs(),
		goos:   cdr.Goos(),
		goarch: cdr.Goarch(),
	}
	mm.astate.Merge(state)
}

func copyMetaDataFile(inpath, outpath string) {
	inf, err := os.Open(inpath)
	if err != nil {
		fatal("opening input meta-data file %s: %v", inpath, err)
	}
	defer inf.Close()

	fi, err := inf.Stat()
	if err != nil {
		fatal("accessing input meta-data file %s: %v", inpath, err)
	}

	outf, err := os.OpenFile(outpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, fi.Mode())
	if err != nil {
		fatal("opening output meta-data file %s: %v", outpath, err)
	}

	_, err = io.Copy(outf, inf)
	outf.Close()
	if err != nil {
		fatal("writing output meta-data file %s: %v", outpath, err)
	}
}

func (mm *metaMerge) beginPod() {
	mm.pod = &podstate{
		pmm: make(map[pkfunc]decodecounter.FuncPayload),
	}
}

// metaEndPod handles actions needed when we're done visiting all of
// the things in a pod -- counter files and meta-data file. There are
// three cases of interest here:
//
// Case 1: in an unconditional merge (we're not selecting a specific set of
// packages using "-pkg", and the "-pcombine" option is not in use),
// we can simply copy over the meta-data file from input to output.
//
// Case 2: if this is a select merge (-pkg is in effect), then at
// this point we write out a new smaller meta-data file that includes
// only the packages of interest. At this point we also emit a merged
// counter data file as well.
//
// Case 3: if "-pcombine" is in effect, we don't write anything at
// this point (all writes will happen at the end of the run).
func (mm *metaMerge) endPod(pcombine bool) {
	if pcombine {
		// Just clear out the pod data, we'll do all the
		// heavy lifting at the end.
		mm.pod = nil
		return
	}

	finalHash := mm.pod.fileHash
	if matchpkg != nil {
		// Emit modified meta-data file for this pod.
		finalHash = mm.emitMeta(*outdirflag, pcombine)
	} else {
		// Copy meta-data file for this pod to the output directory.
		inpath := mm.pod.mdf
		mdfbase := filepath.Base(mm.pod.mdf)
		outpath := filepath.Join(*outdirflag, mdfbase)
		copyMetaDataFile(inpath, outpath)
	}

	// Emit acccumulated counter data for this pod.
	mm.emitCounters(*outdirflag, finalHash)

	// Reset package state.
	mm.pkm = make(map[string]*pkstate)
	mm.pkgs = nil
	mm.pod = nil

	// Reset counter mode and granularity
	mm.ResetModeAndGranularity()
}

// emitMeta encodes and writes out a new coverage meta-data file as
// part of a merge operation, specifically a merge with the
// "-pcombine" flag.
func (mm *metaMerge) emitMeta(outdir string, pcombine bool) [16]byte {
	fh := md5.New()
	blobs := [][]byte{}
	tlen := uint64(unsafe.Sizeof(coverage.MetaFileHeader{}))
	for _, p := range mm.pkgs {
		var blob []byte
		if pcombine {
			mdw := &slicewriter.WriteSeeker{}
			p.cmdb.Emit(mdw)
			blob = mdw.BytesWritten()
		} else {
			blob = p.mdblob
		}
		ph := md5.Sum(blob)
		blobs = append(blobs, blob)
		if _, err := fh.Write(ph[:]); err != nil {
			panic(fmt.Sprintf("internal error: md5 sum failed: %v", err))
		}
		tlen += uint64(len(blob))
	}
	var finalHash [16]byte
	fhh := fh.Sum(nil)
	copy(finalHash[:], fhh)

	// Open meta-file for writing.
	fn := fmt.Sprintf("%s.%x", coverage.MetaFilePref, finalHash)
	fpath := filepath.Join(outdir, fn)
	mf, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		fatal("unable to open output meta-data file %s: %v", fpath, err)
	}

	// Encode and write.
	mfw := encodemeta.NewCoverageMetaFileWriter(fpath, mf)
	err = mfw.Write(finalHash, blobs, mm.Mode(), mm.Granularity())
	if err != nil {
		fatal("error writing %s: %v\n", fpath, err)
	}
	return finalHash
}

func (mm *metaMerge) emitCounters(outdir string, metaHash [16]byte) {
	// Open output file. The file naming scheme is intended to mimic
	// that used when running a coverage-instrumented binary, for
	// consistency (however the process ID is not meaningful here, so
	// use a value of zero).
	var dummyPID int
	fn := fmt.Sprintf(coverage.CounterFileTempl, coverage.CounterFilePref, metaHash, dummyPID, time.Now().UnixNano())
	fpath := filepath.Join(outdir, fn)
	cf, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		fatal("opening counter data file %s: %v", fpath, err)
	}
	defer func() {
		if err := cf.Close(); err != nil {
			fatal("error closing output meta-data file %s: %v", fpath, err)
		}
	}()

	args := mm.astate.ArgsSummary()
	cfw := encodecounter.NewCoverageDataWriter(cf, coverage.CtrULeb128)
	if err := cfw.Write(metaHash, args, mm); err != nil {
		fatal("counter file write failed: %v", err)
	}
	mm.astate = &argstate{}
}

// VisitFuncs is used while writing the counter data files; it
// implements the 'VisitFuncs' method required by the interface
// internal/coverage/encodecounter/CounterVisitor.
func (mm *metaMerge) VisitFuncs(f encodecounter.CounterVisitorFn) error {
	if *verbflag >= 4 {
		fmt.Printf("counterVisitor invoked\n")
	}
	// For each package, for each function, construct counter
	// array and then call "f" on it.
	for pidx, p := range mm.pkgs {
		fids := make([]int, 0, len(p.ctab))
		for fid := range p.ctab {
			fids = append(fids, int(fid))
		}
		sort.Ints(fids)
		if *verbflag >= 4 {
			fmt.Printf("fids for pk=%d: %+v\n", pidx, fids)
		}
		for _, fid := range fids {
			fp := p.ctab[uint32(fid)]
			if *verbflag >= 4 {
				fmt.Printf("counter write for pk=%d fid=%d len(ctrs)=%d\n", pidx, fid, len(fp.Counters))
			}
			if err := f(uint32(pidx), uint32(fid), fp.Counters); err != nil {
				return err
			}
		}
	}
	return nil
}

func (mm *metaMerge) visitPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32, pcombine bool) {
	p, ok := mm.pkm[pd.PackagePath()]
	if !ok {
		p = &pkstate{
			pkgIdx: uint32(len(mm.pkgs)),
		}
		mm.pkgs = append(mm.pkgs, p)
		mm.pkm[pd.PackagePath()] = p
		if pcombine {
			p.pcombinestate = new(pcombinestate)
			cmdb, err := encodemeta.NewCoverageMetaDataBuilder(pd.PackagePath(), pd.PackageName(), pd.ModulePath())
			if err != nil {
				fatal("fatal error creating meta-data builder: %v", err)
			}
			dbgtrace(2, "install new pkm entry for package %s pk=%d", pd.PackagePath(), pkgIdx)
			p.cmdb = cmdb
			p.ftab = make(map[[16]byte]uint32)
		} else {
			var err error
			p.mdblob, err = mm.pod.mfr.GetPackagePayload(pkgIdx, nil)
			if err != nil {
				fatal("error extracting package %d payload from %s: %v",
					pkgIdx, mm.pod.mdf, err)
			}
		}
		p.ctab = make(map[uint32]decodecounter.FuncPayload)
	}
	mm.p = p
}

func (mm *metaMerge) visitFuncCounterData(data decodecounter.FuncPayload) {
	key := pkfunc{pk: data.PkgIdx, fcn: data.FuncIdx}
	val := mm.pod.pmm[key]
	// FIXME: in theory either A) len(val.Counters) is zero, or B)
	// the two lengths are equal. Assert if not? Of course, we could
	// see odd stuff if there is source file skew.
	if *verbflag > 4 {
		fmt.Printf("visit pk=%d fid=%d len(counters)=%d\n", data.PkgIdx, data.FuncIdx, len(data.Counters))
	}
	if len(val.Counters) < len(data.Counters) {
		t := val.Counters
		val.Counters = mm.AllocateCounters(len(data.Counters))
		copy(val.Counters, t)
	}
	err, overflow := mm.MergeCounters(val.Counters, data.Counters)
	if err != nil {
		fatal("%v", err)
	}
	if overflow {
		warn("uint32 overflow during counter merge")
	}
	mm.pod.pmm[key] = val
}

func (mm *metaMerge) visitFunc(pkgIdx uint32, fnIdx uint32, fd *coverage.FuncDesc, verb string, pcombine bool) {
	if *verbflag >= 3 {
		fmt.Printf("visit pk=%d fid=%d func %s\n", pkgIdx, fnIdx, fd.Funcname)
	}

	var counters []uint32
	key := pkfunc{pk: pkgIdx, fcn: fnIdx}
	v, haveCounters := mm.pod.pmm[key]
	if haveCounters {
		counters = v.Counters
	}

	if pcombine {
		// If the merge is running in "combine programs" mode, then hash
		// the function and look it up in the package ftab to see if we've
		// encountered it before. If we haven't, then register it with the
		// meta-data builder.
		fnhash := encodemeta.HashFuncDesc(fd)
		gfidx, ok := mm.p.ftab[fnhash]
		if !ok {
			// We haven't seen this function before, need to add it to
			// the meta data.
			gfidx = uint32(mm.p.cmdb.AddFunc(*fd))
			mm.p.ftab[fnhash] = gfidx
			if *verbflag >= 3 {
				fmt.Printf("new meta entry for fn %s fid=%d\n", fd.Funcname, gfidx)
			}
		}
		fnIdx = gfidx
	}
	if !haveCounters {
		return
	}

	// Install counters in package ctab.
	gfp, ok := mm.p.ctab[fnIdx]
	if ok {
		if verb == "subtract" || verb == "intersect" {
			panic("should never see this for intersect/subtract")
		}
		if *verbflag >= 3 {
			fmt.Printf("counter merge for %s fidx=%d\n", fd.Funcname, fnIdx)
		}
		// Merge.
		err, overflow := mm.MergeCounters(gfp.Counters, counters)
		if err != nil {
			fatal("%v", err)
		}
		if overflow {
			warn("uint32 overflow during counter merge")
		}
		mm.p.ctab[fnIdx] = gfp
	} else {
		if *verbflag >= 3 {
			fmt.Printf("null merge for %s fidx %d\n", fd.Funcname, fnIdx)
		}
		gfp := v
		gfp.PkgIdx = mm.p.pkgIdx
		gfp.FuncIdx = fnIdx
		mm.p.ctab[fnIdx] = gfp
	}
}
