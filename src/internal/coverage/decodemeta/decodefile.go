// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decodemeta

// This package contains APIs and helpers for reading and decoding
// meta-data output files emitted by the runtime when a
// coverage-instrumented binary executes. A meta-data file contains
// top-level info (counter mode, number of packages) and then a
// separate self-contained meta-data section for each Go package.

import (
	"bufio"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"internal/coverage"
	"internal/coverage/slicereader"
	"internal/coverage/stringtab"
	"io"
	"os"
)

// CoverageMetaFileReader provides state and methods for reading
// a meta-data file from a code coverage run.
type CoverageMetaFileReader struct {
	f          *os.File
	hdr        coverage.MetaFileHeader
	tmp        []byte
	pkgOffsets []uint64
	pkgLengths []uint64
	strtab     *stringtab.Reader
	fileRdr    *bufio.Reader
	fileView   []byte
	debug      bool
}

// NewCoverageMetaFileReader returns a new helper object for reading
// the coverage meta-data output file 'f'. The param 'fileView' is a
// read-only slice containing the contents of 'f' obtained by mmap'ing
// the file read-only; 'fileView' may be nil, in which case the helper
// will read the contents of the file using regular file Read
// operations.
func NewCoverageMetaFileReader(f *os.File, fileView []byte) (*CoverageMetaFileReader, error) {
	r := &CoverageMetaFileReader{
		f:        f,
		fileView: fileView,
		tmp:      make([]byte, 256),
	}

	if err := r.readFileHeader(); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *CoverageMetaFileReader) readFileHeader() error {
	var err error

	r.fileRdr = bufio.NewReader(r.f)

	// Read file header.
	if err := binary.Read(r.fileRdr, binary.LittleEndian, &r.hdr); err != nil {
		return err
	}

	// Verify magic string
	m := r.hdr.Magic
	g := coverage.CovMetaMagic
	if m[0] != g[0] || m[1] != g[1] || m[2] != g[2] || m[3] != g[3] {
		return fmt.Errorf("invalid meta-data file magic string")
	}

	// Vet the version. If this is a meta-data file from the future,
	// we won't be able to read it.
	if r.hdr.Version > coverage.MetaFileVersion {
		return fmt.Errorf("meta-data file withn unknown version %d (expected %d)", r.hdr.Version, coverage.MetaFileVersion)
	}

	// Read package offsets for good measure
	r.pkgOffsets = make([]uint64, r.hdr.Entries)
	for i := uint64(0); i < r.hdr.Entries; i++ {
		if r.pkgOffsets[i], err = r.rdUint64(); err != nil {
			return err
		}
		if r.pkgOffsets[i] > r.hdr.TotalLength {
			return fmt.Errorf("insane pkg offset %d: %d > totlen %d",
				i, r.pkgOffsets[i], r.hdr.TotalLength)
		}
	}
	r.pkgLengths = make([]uint64, r.hdr.Entries)
	for i := uint64(0); i < r.hdr.Entries; i++ {
		if r.pkgLengths[i], err = r.rdUint64(); err != nil {
			return err
		}
		if r.pkgLengths[i] > r.hdr.TotalLength {
			return fmt.Errorf("insane pkg length %d: %d > totlen %d",
				i, r.pkgLengths[i], r.hdr.TotalLength)
		}
	}

	// Read string table.
	b := make([]byte, r.hdr.StrTabLength)
	nr, err := r.fileRdr.Read(b)
	if err != nil {
		return err
	}
	if nr != int(r.hdr.StrTabLength) {
		return fmt.Errorf("error: short read on string table")
	}
	slr := slicereader.NewReader(b, false /* not readonly */)
	r.strtab = stringtab.NewReader(slr)
	r.strtab.Read()

	if r.debug {
		fmt.Fprintf(os.Stderr, "=-= read-in header is: %+v\n", *r)
	}

	return nil
}

func (r *CoverageMetaFileReader) rdUint64() (uint64, error) {
	r.tmp = r.tmp[:0]
	r.tmp = append(r.tmp, make([]byte, 8)...)
	n, err := r.fileRdr.Read(r.tmp)
	if err != nil {
		return 0, err
	}
	if n != 8 {
		return 0, fmt.Errorf("premature end of file on read")
	}
	v := binary.LittleEndian.Uint64(r.tmp)
	return v, nil
}

// NumPackages returns the number of packages for which this file
// contains meta-data.
func (r *CoverageMetaFileReader) NumPackages() uint64 {
	return r.hdr.Entries
}

// CounterMode returns the counter mode (set, count, atomic) used
// when building for coverage for the program that produce this
// meta-data file.
func (r *CoverageMetaFileReader) CounterMode() coverage.CounterMode {
	return r.hdr.CMode
}

// CounterGranularity returns the counter granularity (single counter per
// function, or counter per block) selected when building for coverage
// for the program that produce this meta-data file.
func (r *CoverageMetaFileReader) CounterGranularity() coverage.CounterGranularity {
	return r.hdr.CGranularity
}

// FileHash returns the hash computed for all of the package meta-data
// blobs. Coverage counter data files refer to this hash, and the
// hash will be encoded into the meta-data file name.
func (r *CoverageMetaFileReader) FileHash() [16]byte {
	return r.hdr.MetaFileHash
}

// GetPackageDecoder requests a decoder object for the package within
// the meta-data file whose index is 'pkIdx'. If the
// CoverageMetaFileReader was set up with a read-only file view, a
// pointer into that file view will be returned, otherwise the buffer
// 'payloadbuf' will be written to (or if it is not of sufficient
// size, a new buffer will be allocated). Return value is the decoder,
// a byte slice with the encoded meta-data, and an error.
func (r *CoverageMetaFileReader) GetPackageDecoder(pkIdx uint32, payloadbuf []byte) (*CoverageMetaDataDecoder, []byte, error) {
	pp, err := r.GetPackagePayload(pkIdx, payloadbuf)
	if r.debug {
		fmt.Fprintf(os.Stderr, "=-= pkidx=%d payload length is %d hash=%s\n",
			pkIdx, len(pp), fmt.Sprintf("%x", md5.Sum(pp)))
	}
	if err != nil {
		return nil, nil, err
	}
	mdd, err := NewCoverageMetaDataDecoder(pp, r.fileView != nil)
	if err != nil {
		return nil, nil, err
	}
	return mdd, pp, nil
}

// GetPackagePayload returns the raw (encoded) meta-data payload for the
// package with index 'pkIdx'. As with GetPackageDecoder, if the
// CoverageMetaFileReader was set up with a read-only file view, a
// pointer into that file view will be returned, otherwise the buffer
// 'payloadbuf' will be written to (or if it is not of sufficient
// size, a new buffer will be allocated). Return value is the decoder,
// a byte slice with the encoded meta-data, and an error.
func (r *CoverageMetaFileReader) GetPackagePayload(pkIdx uint32, payloadbuf []byte) ([]byte, error) {

	// Determine correct offset/length.
	if uint64(pkIdx) >= r.hdr.Entries {
		return nil, fmt.Errorf("GetPackagePayload: illegal pkg index %d", pkIdx)
	}
	off := r.pkgOffsets[pkIdx]
	len := r.pkgLengths[pkIdx]

	if r.debug {
		fmt.Fprintf(os.Stderr, "=-= for pk %d, off=%d len=%d\n", pkIdx, off, len)
	}

	if r.fileView != nil {
		return r.fileView[off : off+len], nil
	}

	payload := payloadbuf[:0]
	if cap(payload) < int(len) {
		payload = make([]byte, 0, len)
	}
	payload = append(payload, make([]byte, len)...)
	if _, err := r.f.Seek(int64(off), io.SeekStart); err != nil {
		return nil, err
	}
	if _, err := io.ReadFull(r.f, payload); err != nil {
		return nil, err
	}
	return payload, nil
}
