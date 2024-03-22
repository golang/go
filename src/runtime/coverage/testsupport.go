// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package coverage

import (
	"encoding/json"
	"fmt"
	"internal/coverage"
	"internal/coverage/calloc"
	"internal/coverage/cformat"
	"internal/coverage/cmerge"
	"internal/coverage/decodecounter"
	"internal/coverage/decodemeta"
	"internal/coverage/pods"
	"internal/runtime/atomic"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

// processCoverTestDir is called (via a linknamed reference) from
// testmain code when "go test -cover" is in effect. It is not
// intended to be used other than internally by the Go command's
// generated code.
func processCoverTestDir(dir string, cfile string, cm string, cpkg string) error {
	return processCoverTestDirInternal(dir, cfile, cm, cpkg, os.Stdout)
}

// processCoverTestDirInternal is an io.Writer version of processCoverTestDir,
// exposed for unit testing.
func processCoverTestDirInternal(dir string, cfile string, cm string, cpkg string, w io.Writer) error {
	cmode := coverage.ParseCounterMode(cm)
	if cmode == coverage.CtrModeInvalid {
		return fmt.Errorf("invalid counter mode %q", cm)
	}

	// Emit meta-data and counter data.
	ml := getCovMetaList()
	if len(ml) == 0 {
		// This corresponds to the case where we have a package that
		// contains test code but no functions (which is fine). In this
		// case there is no need to emit anything.
	} else {
		if err := emitMetaDataToDirectory(dir, ml); err != nil {
			return err
		}
		if err := emitCounterDataToDirectory(dir); err != nil {
			return err
		}
	}

	// Collect pods from test run. For the majority of cases we would
	// expect to see a single pod here, but allow for multiple pods in
	// case the test harness is doing extra work to collect data files
	// from builds that it kicks off as part of the testing.
	podlist, err := pods.CollectPods([]string{dir}, false)
	if err != nil {
		return fmt.Errorf("reading from %s: %v", dir, err)
	}

	// Open text output file if appropriate.
	var tf *os.File
	var tfClosed bool
	if cfile != "" {
		var err error
		tf, err = os.Create(cfile)
		if err != nil {
			return fmt.Errorf("internal error: opening coverage data output file %q: %v", cfile, err)
		}
		defer func() {
			if !tfClosed {
				tfClosed = true
				tf.Close()
			}
		}()
	}

	// Read/process the pods.
	ts := &tstate{
		cm:    &cmerge.Merger{},
		cf:    cformat.NewFormatter(cmode),
		cmode: cmode,
	}
	// Generate the expected hash string based on the final meta-data
	// hash for this test, then look only for pods that refer to that
	// hash (just in case there are multiple instrumented executables
	// in play). See issue #57924 for more on this.
	hashstring := fmt.Sprintf("%x", finalHash)
	importpaths := make(map[string]struct{})
	for _, p := range podlist {
		if !strings.Contains(p.MetaFile, hashstring) {
			continue
		}
		if err := ts.processPod(p, importpaths); err != nil {
			return err
		}
	}

	metafilespath := filepath.Join(dir, coverage.MetaFilesFileName)
	if _, err := os.Stat(metafilespath); err == nil {
		if err := ts.readAuxMetaFiles(metafilespath, importpaths); err != nil {
			return err
		}
	}

	// Emit percent.
	if err := ts.cf.EmitPercent(w, cpkg, true, true); err != nil {
		return err
	}

	// Emit text output.
	if tf != nil {
		if err := ts.cf.EmitTextual(tf); err != nil {
			return err
		}
		tfClosed = true
		if err := tf.Close(); err != nil {
			return fmt.Errorf("closing %s: %v", cfile, err)
		}
	}

	return nil
}

type tstate struct {
	calloc.BatchCounterAlloc
	cm    *cmerge.Merger
	cf    *cformat.Formatter
	cmode coverage.CounterMode
}

// processPod reads coverage counter data for a specific pod.
func (ts *tstate) processPod(p pods.Pod, importpaths map[string]struct{}) error {
	// Open meta-data file
	f, err := os.Open(p.MetaFile)
	if err != nil {
		return fmt.Errorf("unable to open meta-data file %s: %v", p.MetaFile, err)
	}
	defer func() {
		f.Close()
	}()
	var mfr *decodemeta.CoverageMetaFileReader
	mfr, err = decodemeta.NewCoverageMetaFileReader(f, nil)
	if err != nil {
		return fmt.Errorf("error reading meta-data file %s: %v", p.MetaFile, err)
	}
	newmode := mfr.CounterMode()
	if newmode != ts.cmode {
		return fmt.Errorf("internal error: counter mode clash: %q from test harness, %q from data file %s", ts.cmode.String(), newmode.String(), p.MetaFile)
	}
	newgran := mfr.CounterGranularity()
	if err := ts.cm.SetModeAndGranularity(p.MetaFile, cmode, newgran); err != nil {
		return err
	}

	// A map to store counter data, indexed by pkgid/fnid tuple.
	pmm := make(map[pkfunc][]uint32)

	// Helper to read a single counter data file.
	readcdf := func(cdf string) error {
		cf, err := os.Open(cdf)
		if err != nil {
			return fmt.Errorf("opening counter data file %s: %s", cdf, err)
		}
		defer cf.Close()
		var cdr *decodecounter.CounterDataReader
		cdr, err = decodecounter.NewCounterDataReader(cdf, cf)
		if err != nil {
			return fmt.Errorf("reading counter data file %s: %s", cdf, err)
		}
		var data decodecounter.FuncPayload
		for {
			ok, err := cdr.NextFunc(&data)
			if err != nil {
				return fmt.Errorf("reading counter data file %s: %v", cdf, err)
			}
			if !ok {
				break
			}

			// NB: sanity check on pkg and func IDs?
			key := pkfunc{pk: data.PkgIdx, fcn: data.FuncIdx}
			if prev, found := pmm[key]; found {
				// Note: no overflow reporting here.
				if err, _ := ts.cm.MergeCounters(data.Counters, prev); err != nil {
					return fmt.Errorf("processing counter data file %s: %v", cdf, err)
				}
			}
			c := ts.AllocateCounters(len(data.Counters))
			copy(c, data.Counters)
			pmm[key] = c
		}
		return nil
	}

	// Read counter data files.
	for _, cdf := range p.CounterDataFiles {
		if err := readcdf(cdf); err != nil {
			return err
		}
	}

	// Visit meta-data file.
	np := uint32(mfr.NumPackages())
	payload := []byte{}
	for pkIdx := uint32(0); pkIdx < np; pkIdx++ {
		var pd *decodemeta.CoverageMetaDataDecoder
		pd, payload, err = mfr.GetPackageDecoder(pkIdx, payload)
		if err != nil {
			return fmt.Errorf("reading pkg %d from meta-file %s: %s", pkIdx, p.MetaFile, err)
		}
		ts.cf.SetPackage(pd.PackagePath())
		importpaths[pd.PackagePath()] = struct{}{}
		var fd coverage.FuncDesc
		nf := pd.NumFuncs()
		for fnIdx := uint32(0); fnIdx < nf; fnIdx++ {
			if err := pd.ReadFunc(fnIdx, &fd); err != nil {
				return fmt.Errorf("reading meta-data file %s: %v",
					p.MetaFile, err)
			}
			key := pkfunc{pk: pkIdx, fcn: fnIdx}
			counters, haveCounters := pmm[key]
			for i := 0; i < len(fd.Units); i++ {
				u := fd.Units[i]
				// Skip units with non-zero parent (no way to represent
				// these in the existing format).
				if u.Parent != 0 {
					continue
				}
				count := uint32(0)
				if haveCounters {
					count = counters[i]
				}
				ts.cf.AddUnit(fd.Srcfile, fd.Funcname, fd.Lit, u, count)
			}
		}
	}
	return nil
}

type pkfunc struct {
	pk, fcn uint32
}

func (ts *tstate) readAuxMetaFiles(metafiles string, importpaths map[string]struct{}) error {
	// Unmarshal the information on available aux metafiles into
	// a MetaFileCollection struct.
	var mfc coverage.MetaFileCollection
	data, err := os.ReadFile(metafiles)
	if err != nil {
		return fmt.Errorf("error reading auxmetafiles file %q: %v", metafiles, err)
	}
	if err := json.Unmarshal(data, &mfc); err != nil {
		return fmt.Errorf("error reading auxmetafiles file %q: %v", metafiles, err)
	}

	// Walk through each available aux meta-file. If we've already
	// seen the package path in question during the walk of the
	// "regular" meta-data file, then we can skip the package,
	// otherwise construct a dummy pod with the single meta-data file
	// (no counters) and invoke processPod on it.
	for i := range mfc.ImportPaths {
		p := mfc.ImportPaths[i]
		if _, ok := importpaths[p]; ok {
			continue
		}
		var pod pods.Pod
		pod.MetaFile = mfc.MetaFileFragments[i]
		if err := ts.processPod(pod, importpaths); err != nil {
			return err
		}
	}
	return nil
}

// snapshot returns a snapshot of coverage percentage at a moment of
// time within a running test, so as to support the testing.Coverage()
// function. This version doesn't examine coverage meta-data, so the
// result it returns will be less accurate (more "slop") due to the
// fact that we don't look at the meta data to see how many statements
// are associated with each counter.
func snapshot() float64 {
	cl := getCovCounterList()
	if len(cl) == 0 {
		// no work to do here.
		return 0.0
	}

	tot := uint64(0)
	totExec := uint64(0)
	for _, c := range cl {
		sd := unsafe.Slice((*atomic.Uint32)(unsafe.Pointer(c.Counters)), c.Len)
		tot += uint64(len(sd))
		for i := 0; i < len(sd); i++ {
			// Skip ahead until the next non-zero value.
			if sd[i].Load() == 0 {
				continue
			}
			// We found a function that was executed.
			nCtrs := sd[i+coverage.NumCtrsOffset].Load()
			cst := i + coverage.FirstCtrOffset

			if cst+int(nCtrs) > len(sd) {
				break
			}
			counters := sd[cst : cst+int(nCtrs)]
			for i := range counters {
				if counters[i].Load() != 0 {
					totExec++
				}
			}
			i += coverage.FirstCtrOffset + int(nCtrs) - 1
		}
	}
	if tot == 0 {
		return 0.0
	}
	return float64(totExec) / float64(tot)
}
