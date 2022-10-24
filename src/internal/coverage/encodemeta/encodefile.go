// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encodemeta

import (
	"bufio"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"internal/coverage"
	"internal/coverage/stringtab"
	"io"
	"os"
	"unsafe"
)

// This package contains APIs and helpers for writing out a meta-data
// file (composed of a file header, offsets/lengths, and then a series of
// meta-data blobs emitted by the compiler, one per Go package).

type CoverageMetaFileWriter struct {
	stab   stringtab.Writer
	mfname string
	w      *bufio.Writer
	tmp    []byte
	debug  bool
}

func NewCoverageMetaFileWriter(mfname string, w io.Writer) *CoverageMetaFileWriter {
	r := &CoverageMetaFileWriter{
		mfname: mfname,
		w:      bufio.NewWriter(w),
		tmp:    make([]byte, 64),
	}
	r.stab.InitWriter()
	r.stab.Lookup("")
	return r
}

func (m *CoverageMetaFileWriter) Write(finalHash [16]byte, blobs [][]byte, mode coverage.CounterMode, granularity coverage.CounterGranularity) error {
	mhsz := uint64(unsafe.Sizeof(coverage.MetaFileHeader{}))
	stSize := m.stab.Size()
	stOffset := mhsz + uint64(16*len(blobs))
	preambleLength := stOffset + uint64(stSize)

	if m.debug {
		fmt.Fprintf(os.Stderr, "=+= sizeof(MetaFileHeader)=%d\n", mhsz)
		fmt.Fprintf(os.Stderr, "=+= preambleLength=%d stSize=%d\n", preambleLength, stSize)
	}

	// Compute total size
	tlen := preambleLength
	for i := 0; i < len(blobs); i++ {
		tlen += uint64(len(blobs[i]))
	}

	// Emit header
	mh := coverage.MetaFileHeader{
		Magic:        coverage.CovMetaMagic,
		Version:      coverage.MetaFileVersion,
		TotalLength:  tlen,
		Entries:      uint64(len(blobs)),
		MetaFileHash: finalHash,
		StrTabOffset: uint32(stOffset),
		StrTabLength: stSize,
		CMode:        mode,
		CGranularity: granularity,
	}
	var err error
	if err = binary.Write(m.w, binary.LittleEndian, mh); err != nil {
		return fmt.Errorf("error writing %s: %v", m.mfname, err)
	}

	if m.debug {
		fmt.Fprintf(os.Stderr, "=+= len(blobs) is %d\n", mh.Entries)
	}

	// Emit package offsets section followed by package lengths section.
	off := preambleLength
	off2 := mhsz
	buf := make([]byte, 8)
	for _, blob := range blobs {
		binary.LittleEndian.PutUint64(buf, off)
		if _, err = m.w.Write(buf); err != nil {
			return fmt.Errorf("error writing %s: %v", m.mfname, err)
		}
		if m.debug {
			fmt.Fprintf(os.Stderr, "=+= pkg offset %d 0x%x\n", off, off)
		}
		off += uint64(len(blob))
		off2 += 8
	}
	for _, blob := range blobs {
		bl := uint64(len(blob))
		binary.LittleEndian.PutUint64(buf, bl)
		if _, err = m.w.Write(buf); err != nil {
			return fmt.Errorf("error writing %s: %v", m.mfname, err)
		}
		if m.debug {
			fmt.Fprintf(os.Stderr, "=+= pkg len %d 0x%x\n", bl, bl)
		}
		off2 += 8
	}

	// Emit string table
	if err = m.stab.Write(m.w); err != nil {
		return err
	}

	// Now emit blobs themselves.
	for k, blob := range blobs {
		if m.debug {
			fmt.Fprintf(os.Stderr, "=+= writing blob %d len %d at off=%d hash %s\n", k, len(blob), off2, fmt.Sprintf("%x", md5.Sum(blob)))
		}
		if _, err = m.w.Write(blob); err != nil {
			return fmt.Errorf("error writing %s: %v", m.mfname, err)
		}
		if m.debug {
			fmt.Fprintf(os.Stderr, "=+= wrote package payload of %d bytes\n",
				len(blob))
		}
		off2 += uint64(len(blob))
	}

	// Flush writer, and we're done.
	if err = m.w.Flush(); err != nil {
		return fmt.Errorf("error writing %s: %v", m.mfname, err)
	}
	return nil
}
