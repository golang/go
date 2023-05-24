// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decodemeta

// This package contains APIs and helpers for decoding a single package's
// meta data "blob" emitted by the compiler when coverage instrumentation
// is turned on.

import (
	"encoding/binary"
	"fmt"
	"internal/coverage"
	"internal/coverage/slicereader"
	"internal/coverage/stringtab"
	"io"
	"os"
)

// See comments in the encodecovmeta package for details on the format.

type CoverageMetaDataDecoder struct {
	r      *slicereader.Reader
	hdr    coverage.MetaSymbolHeader
	strtab *stringtab.Reader
	tmp    []byte
	debug  bool
}

func NewCoverageMetaDataDecoder(b []byte, readonly bool) (*CoverageMetaDataDecoder, error) {
	slr := slicereader.NewReader(b, readonly)
	x := &CoverageMetaDataDecoder{
		r:   slr,
		tmp: make([]byte, 0, 256),
	}
	if err := x.readHeader(); err != nil {
		return nil, err
	}
	if err := x.readStringTable(); err != nil {
		return nil, err
	}
	return x, nil
}

func (d *CoverageMetaDataDecoder) readHeader() error {
	if err := binary.Read(d.r, binary.LittleEndian, &d.hdr); err != nil {
		return err
	}
	if d.debug {
		fmt.Fprintf(os.Stderr, "=-= after readHeader: %+v\n", d.hdr)
	}
	return nil
}

func (d *CoverageMetaDataDecoder) readStringTable() error {
	// Seek to the correct location to read the string table.
	stringTableLocation := int64(coverage.CovMetaHeaderSize + 4*d.hdr.NumFuncs)
	if _, err := d.r.Seek(stringTableLocation, io.SeekStart); err != nil {
		return err
	}

	// Read the table itself.
	d.strtab = stringtab.NewReader(d.r)
	d.strtab.Read()
	return nil
}

func (d *CoverageMetaDataDecoder) PackagePath() string {
	return d.strtab.Get(d.hdr.PkgPath)
}

func (d *CoverageMetaDataDecoder) PackageName() string {
	return d.strtab.Get(d.hdr.PkgName)
}

func (d *CoverageMetaDataDecoder) ModulePath() string {
	return d.strtab.Get(d.hdr.ModulePath)
}

func (d *CoverageMetaDataDecoder) NumFuncs() uint32 {
	return d.hdr.NumFuncs
}

// ReadFunc reads the coverage meta-data for the function with index
// 'findex', filling it into the FuncDesc pointed to by 'f'.
func (d *CoverageMetaDataDecoder) ReadFunc(fidx uint32, f *coverage.FuncDesc) error {
	if fidx >= d.hdr.NumFuncs {
		return fmt.Errorf("illegal function index")
	}

	// Seek to the correct location to read the function offset and read it.
	funcOffsetLocation := int64(coverage.CovMetaHeaderSize + 4*fidx)
	if _, err := d.r.Seek(funcOffsetLocation, io.SeekStart); err != nil {
		return err
	}
	foff := d.r.ReadUint32()

	// Check assumptions
	if foff < uint32(funcOffsetLocation) || foff > d.hdr.Length {
		return fmt.Errorf("malformed func offset %d", foff)
	}

	// Seek to the correct location to read the function.
	floc := int64(foff)
	if _, err := d.r.Seek(floc, io.SeekStart); err != nil {
		return err
	}

	// Preamble containing number of units, file, and function.
	numUnits := uint32(d.r.ReadULEB128())
	fnameidx := uint32(d.r.ReadULEB128())
	fileidx := uint32(d.r.ReadULEB128())

	f.Srcfile = d.strtab.Get(fileidx)
	f.Funcname = d.strtab.Get(fnameidx)

	// Now the units
	f.Units = f.Units[:0]
	if cap(f.Units) < int(numUnits) {
		f.Units = make([]coverage.CoverableUnit, 0, numUnits)
	}
	for k := uint32(0); k < numUnits; k++ {
		f.Units = append(f.Units,
			coverage.CoverableUnit{
				StLine:  uint32(d.r.ReadULEB128()),
				StCol:   uint32(d.r.ReadULEB128()),
				EnLine:  uint32(d.r.ReadULEB128()),
				EnCol:   uint32(d.r.ReadULEB128()),
				NxStmts: uint32(d.r.ReadULEB128()),
			})
	}
	lit := d.r.ReadULEB128()
	f.Lit = lit != 0
	return nil
}
