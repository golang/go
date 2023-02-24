// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cov_test

import (
	"cmd/internal/cov"
	"fmt"
	"internal/coverage"
	"internal/coverage/decodecounter"
	"internal/coverage/decodemeta"
	"internal/coverage/pods"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

// visitor implements the CovDataVisitor interface in a very stripped
// down way, just keeps track of interesting events.
type visitor struct {
	metaFileCount    int
	counterFileCount int
	funcCounterData  int
	metaFuncCount    int
}

func (v *visitor) BeginPod(p pods.Pod) {}
func (v *visitor) EndPod(p pods.Pod)   {}
func (v *visitor) VisitMetaDataFile(mdf string, mfr *decodemeta.CoverageMetaFileReader) {
	v.metaFileCount++
}
func (v *visitor) BeginCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int) {
	v.counterFileCount++
}
func (v *visitor) EndCounterDataFile(cdf string, cdr *decodecounter.CounterDataReader, dirIdx int) {}
func (v *visitor) VisitFuncCounterData(payload decodecounter.FuncPayload)                          { v.funcCounterData++ }
func (v *visitor) EndCounters()                                                                    {}
func (v *visitor) BeginPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32)              {}
func (v *visitor) EndPackage(pd *decodemeta.CoverageMetaDataDecoder, pkgIdx uint32)                {}
func (v *visitor) VisitFunc(pkgIdx uint32, fnIdx uint32, fd *coverage.FuncDesc)                    { v.metaFuncCount++ }
func (v *visitor) Finish()                                                                         {}

func TestIssue58411(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// Build a tiny test program with -cover. Smallness is important;
	// it is one of the factors that triggers issue 58411.
	d := t.TempDir()
	exepath := filepath.Join(d, "small.exe")
	path := filepath.Join("testdata", "small.go")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build",
		"-o", exepath, "-cover", path)
	b, err := cmd.CombinedOutput()
	if len(b) != 0 {
		t.Logf("## build output:\n%s", b)
	}
	if err != nil {
		t.Fatalf("build error: %v", err)
	}

	// Run to produce coverage data. Note the large argument; we need a large
	// argument (more than 4k) to trigger the bug, but the overall file
	// has to remain small (since large files will be read with mmap).
	covdir := filepath.Join(d, "covdata")
	if err = os.Mkdir(covdir, 0777); err != nil {
		t.Fatalf("creating covdir: %v", err)
	}
	large := fmt.Sprintf("%07999d", 0)
	cmd = testenv.Command(t, exepath, "1", "2", "3", large)
	cmd.Dir = covdir
	cmd.Env = append(os.Environ(), "GOCOVERDIR="+covdir)
	b, err = cmd.CombinedOutput()
	if err != nil {
		t.Logf("## run output:\n%s", b)
		t.Fatalf("build error: %v", err)
	}

	vis := &visitor{}

	// Read resulting coverage data. Without the fix, this would
	// yield a "short read" error.
	const verbosityLevel = 0
	const flags = 0
	cdr := cov.MakeCovDataReader(vis, []string{covdir}, verbosityLevel, flags, nil)
	err = cdr.Visit()
	if err != nil {
		t.Fatalf("visit failed: %v", err)
	}

	// make sure we saw a few things just for grins
	const want = "{metaFileCount:1 counterFileCount:1 funcCounterData:1 metaFuncCount:1}"
	got := fmt.Sprintf("%+v", *vis)
	if want != got {
		t.Errorf("visitor contents: want %v got %v\n", want, got)
	}
}
