// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"fmt"
	"internal/coverage"
	"internal/coverage/decodemeta"
	"internal/coverage/encodemeta"
	"internal/coverage/slicewriter"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func cmpFuncDesc(want, got coverage.FuncDesc) string {
	swant := fmt.Sprintf("%+v", want)
	sgot := fmt.Sprintf("%+v", got)
	if swant == sgot {
		return ""
	}
	return fmt.Sprintf("wanted %q got %q", swant, sgot)
}

func TestMetaDataEmptyPackage(t *testing.T) {
	// Make sure that encoding/decoding works properly with packages
	// that don't actually have any functions.
	p := "empty/package"
	pn := "package"
	mp := "m"
	b, err := encodemeta.NewCoverageMetaDataBuilder(p, pn, mp)
	if err != nil {
		t.Fatalf("making builder: %v", err)
	}
	drws := &slicewriter.WriteSeeker{}
	b.Emit(drws)
	drws.Seek(0, io.SeekStart)
	dec, err := decodemeta.NewCoverageMetaDataDecoder(drws.BytesWritten(), false)
	if err != nil {
		t.Fatalf("making decoder: %v", err)
	}
	nf := dec.NumFuncs()
	if nf != 0 {
		t.Errorf("dec.NumFuncs(): got %d want %d", nf, 0)
	}
	pp := dec.PackagePath()
	if pp != p {
		t.Errorf("dec.PackagePath(): got %s want %s", pp, p)
	}
	ppn := dec.PackageName()
	if ppn != pn {
		t.Errorf("dec.PackageName(): got %s want %s", ppn, pn)
	}
	pmp := dec.ModulePath()
	if pmp != mp {
		t.Errorf("dec.ModulePath(): got %s want %s", pmp, mp)
	}
}

func TestMetaDataEncoderDecoder(t *testing.T) {
	// Test encode path.
	pp := "foo/bar/pkg"
	pn := "pkg"
	mp := "barmod"
	b, err := encodemeta.NewCoverageMetaDataBuilder(pp, pn, mp)
	if err != nil {
		t.Fatalf("making builder: %v", err)
	}
	f1 := coverage.FuncDesc{
		Funcname: "func",
		Srcfile:  "foo.go",
		Units: []coverage.CoverableUnit{
			coverage.CoverableUnit{StLine: 1, StCol: 2, EnLine: 3, EnCol: 4, NxStmts: 5},
			coverage.CoverableUnit{StLine: 6, StCol: 7, EnLine: 8, EnCol: 9, NxStmts: 10},
		},
	}
	idx := b.AddFunc(f1)
	if idx != 0 {
		t.Errorf("b.AddFunc(f1) got %d want %d", idx, 0)
	}

	f2 := coverage.FuncDesc{
		Funcname: "xfunc",
		Srcfile:  "bar.go",
		Units: []coverage.CoverableUnit{
			coverage.CoverableUnit{StLine: 1, StCol: 2, EnLine: 3, EnCol: 4, NxStmts: 5},
			coverage.CoverableUnit{StLine: 6, StCol: 7, EnLine: 8, EnCol: 9, NxStmts: 10},
			coverage.CoverableUnit{StLine: 11, StCol: 12, EnLine: 13, EnCol: 14, NxStmts: 15},
		},
	}
	idx = b.AddFunc(f2)
	if idx != 1 {
		t.Errorf("b.AddFunc(f2) got %d want %d", idx, 0)
	}

	// Emit into a writer.
	drws := &slicewriter.WriteSeeker{}
	b.Emit(drws)

	// Test decode path.
	drws.Seek(0, io.SeekStart)
	dec, err := decodemeta.NewCoverageMetaDataDecoder(drws.BytesWritten(), false)
	if err != nil {
		t.Fatalf("NewCoverageMetaDataDecoder error: %v", err)
	}
	nf := dec.NumFuncs()
	if nf != 2 {
		t.Errorf("dec.NumFuncs(): got %d want %d", nf, 2)
	}

	gotpp := dec.PackagePath()
	if gotpp != pp {
		t.Errorf("packagepath: got %s want %s", gotpp, pp)
	}
	gotpn := dec.PackageName()
	if gotpn != pn {
		t.Errorf("packagename: got %s want %s", gotpn, pn)
	}

	cases := []coverage.FuncDesc{f1, f2}
	for i := uint32(0); i < uint32(len(cases)); i++ {
		var fn coverage.FuncDesc
		if err := dec.ReadFunc(i, &fn); err != nil {
			t.Fatalf("err reading function %d: %v", i, err)
		}
		res := cmpFuncDesc(cases[i], fn)
		if res != "" {
			t.Errorf("ReadFunc(%d): %s", i, res)
		}
	}
}

func createFuncs(i int) []coverage.FuncDesc {
	res := []coverage.FuncDesc{}
	lc := uint32(1)
	for fi := 0; fi < i+1; fi++ {
		units := []coverage.CoverableUnit{}
		for ui := 0; ui < (fi+1)*(i+1); ui++ {
			units = append(units,
				coverage.CoverableUnit{StLine: lc, StCol: lc + 1,
					EnLine: lc + 2, EnCol: lc + 3, NxStmts: lc + 4,
				})
			lc += 5
		}
		f := coverage.FuncDesc{
			Funcname: fmt.Sprintf("func_%d_%d", i, fi),
			Srcfile:  fmt.Sprintf("foo_%d.go", i),
			Units:    units,
		}
		res = append(res, f)
	}
	return res
}

func createBlob(t *testing.T, i int) []byte {
	nomodule := ""
	b, err := encodemeta.NewCoverageMetaDataBuilder("foo/pkg", "pkg", nomodule)
	if err != nil {
		t.Fatalf("making builder: %v", err)
	}

	funcs := createFuncs(i)
	for _, f := range funcs {
		b.AddFunc(f)
	}
	drws := &slicewriter.WriteSeeker{}
	b.Emit(drws)
	return drws.BytesWritten()
}

func createMetaDataBlobs(t *testing.T, nb int) [][]byte {
	res := [][]byte{}
	for i := 0; i < nb; i++ {
		res = append(res, createBlob(t, i))
	}
	return res
}

func TestMetaDataWriterReader(t *testing.T) {
	d := t.TempDir()

	// Emit a meta-file...
	mfpath := filepath.Join(d, "covmeta.hash.0")
	of, err := os.OpenFile(mfpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		t.Fatalf("opening covmeta: %v", err)
	}
	//t.Logf("meta-file path is %s", mfpath)
	blobs := createMetaDataBlobs(t, 7)
	gran := coverage.CtrGranularityPerBlock
	mfw := encodemeta.NewCoverageMetaFileWriter(mfpath, of)
	finalHash := [16]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	err = mfw.Write(finalHash, blobs, coverage.CtrModeAtomic, gran)
	if err != nil {
		t.Fatalf("writing meta-file: %v", err)
	}
	if err = of.Close(); err != nil {
		t.Fatalf("closing meta-file: %v", err)
	}

	// ... then read it back in, first time without setting fileView,
	// second time setting it.
	for k := 0; k < 2; k++ {
		var fileView []byte

		inf, err := os.Open(mfpath)
		if err != nil {
			t.Fatalf("open() on meta-file: %v", err)
		}

		if k != 0 {
			// Use fileview to exercise different paths in reader.
			fi, err := os.Stat(mfpath)
			if err != nil {
				t.Fatalf("stat() on meta-file: %v", err)
			}
			fileView = make([]byte, fi.Size())
			if _, err := inf.Read(fileView); err != nil {
				t.Fatalf("read() on meta-file: %v", err)
			}
			if _, err := inf.Seek(int64(0), io.SeekStart); err != nil {
				t.Fatalf("seek() on meta-file: %v", err)
			}
		}

		mfr, err := decodemeta.NewCoverageMetaFileReader(inf, fileView)
		if err != nil {
			t.Fatalf("k=%d NewCoverageMetaFileReader failed with: %v", k, err)
		}
		np := mfr.NumPackages()
		if np != 7 {
			t.Fatalf("k=%d wanted 7 packages got %d", k, np)
		}
		md := mfr.CounterMode()
		wmd := coverage.CtrModeAtomic
		if md != wmd {
			t.Fatalf("k=%d wanted mode %d got %d", k, wmd, md)
		}
		gran := mfr.CounterGranularity()
		wgran := coverage.CtrGranularityPerBlock
		if gran != wgran {
			t.Fatalf("k=%d wanted gran %d got %d", k, wgran, gran)
		}

		payload := []byte{}
		for pi := 0; pi < int(np); pi++ {
			var pd *decodemeta.CoverageMetaDataDecoder
			var err error
			pd, payload, err = mfr.GetPackageDecoder(uint32(pi), payload)
			if err != nil {
				t.Fatalf("GetPackageDecoder(%d) failed with: %v", pi, err)
			}
			efuncs := createFuncs(pi)
			nf := pd.NumFuncs()
			if len(efuncs) != int(nf) {
				t.Fatalf("decoding pk %d wanted %d funcs got %d",
					pi, len(efuncs), nf)
			}
			var f coverage.FuncDesc
			for fi := 0; fi < int(nf); fi++ {
				if err := pd.ReadFunc(uint32(fi), &f); err != nil {
					t.Fatalf("ReadFunc(%d) pk %d got error %v",
						fi, pi, err)
				}
				res := cmpFuncDesc(efuncs[fi], f)
				if res != "" {
					t.Errorf("ReadFunc(%d) pk %d: %s", fi, pi, res)
				}
			}
		}
		inf.Close()
	}
}

func TestMetaDataDecodeLitFlagIssue57942(t *testing.T) {

	// Encode a package with a few functions. The funcs alternate
	// between regular functions and function literals.
	pp := "foo/bar/pkg"
	pn := "pkg"
	mp := "barmod"
	b, err := encodemeta.NewCoverageMetaDataBuilder(pp, pn, mp)
	if err != nil {
		t.Fatalf("making builder: %v", err)
	}
	const NF = 6
	const NCU = 1
	ln := uint32(10)
	wantfds := []coverage.FuncDesc{}
	for fi := uint32(0); fi < NF; fi++ {
		fis := fmt.Sprintf("%d", fi)
		fd := coverage.FuncDesc{
			Funcname: "func" + fis,
			Srcfile:  "foo" + fis + ".go",
			Units: []coverage.CoverableUnit{
				coverage.CoverableUnit{StLine: ln + 1, StCol: 2, EnLine: ln + 3, EnCol: 4, NxStmts: fi + 2},
			},
			Lit: (fi % 2) == 0,
		}
		wantfds = append(wantfds, fd)
		b.AddFunc(fd)
	}

	// Emit into a writer.
	drws := &slicewriter.WriteSeeker{}
	b.Emit(drws)

	// Decode the result.
	drws.Seek(0, io.SeekStart)
	dec, err := decodemeta.NewCoverageMetaDataDecoder(drws.BytesWritten(), false)
	if err != nil {
		t.Fatalf("making decoder: %v", err)
	}
	nf := dec.NumFuncs()
	if nf != NF {
		t.Fatalf("decoder number of functions: got %d want %d", nf, NF)
	}
	var fn coverage.FuncDesc
	for i := uint32(0); i < uint32(NF); i++ {
		if err := dec.ReadFunc(i, &fn); err != nil {
			t.Fatalf("err reading function %d: %v", i, err)
		}
		res := cmpFuncDesc(wantfds[i], fn)
		if res != "" {
			t.Errorf("ReadFunc(%d): %s", i, res)
		}
	}
}
