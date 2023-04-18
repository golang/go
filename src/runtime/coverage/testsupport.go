// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package coverage

import (
	"fmt"
	"internal/coverage"
	"internal/coverage/calloc"
	"internal/coverage/cformat"
	"internal/coverage/cmerge"
	"internal/coverage/decodecounter"
	"internal/coverage/decodemeta"
	"internal/coverage/pods"
	"io"
	"os"
	"strings"
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
	for _, p := range podlist {
		if !strings.Contains(p.MetaFile, hashstring) {
			continue
		}
		if err := ts.processPod(p); err != nil {
			return err
		}
	}

	// Emit percent.
	if err := ts.cf.EmitPercent(w, cpkg, true); err != nil {
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
func (ts *tstate) processPod(p pods.Pod) error {
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
