// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encodemeta

// This package contains APIs and helpers for encoding the meta-data
// "blob" for a single Go package, created when coverage
// instrumentation is turned on.

import (
	"bytes"
	"crypto/md5"
	"encoding/binary"
	"fmt"
	"hash"
	"internal/coverage"
	"internal/coverage/stringtab"
	"internal/coverage/uleb128"
	"io"
	"os"
)

type CoverageMetaDataBuilder struct {
	stab    stringtab.Writer
	funcs   []funcDesc
	tmp     []byte // temp work slice
	h       hash.Hash
	pkgpath uint32
	pkgname uint32
	modpath uint32
	debug   bool
	werr    error
}

func NewCoverageMetaDataBuilder(pkgpath string, pkgname string, modulepath string) (*CoverageMetaDataBuilder, error) {
	if pkgpath == "" {
		return nil, fmt.Errorf("invalid empty package path")
	}
	x := &CoverageMetaDataBuilder{
		tmp: make([]byte, 0, 256),
		h:   md5.New(),
	}
	x.stab.InitWriter()
	x.stab.Lookup("")
	x.pkgpath = x.stab.Lookup(pkgpath)
	x.pkgname = x.stab.Lookup(pkgname)
	x.modpath = x.stab.Lookup(modulepath)
	io.WriteString(x.h, pkgpath)
	io.WriteString(x.h, pkgname)
	io.WriteString(x.h, modulepath)
	return x, nil
}

func h32(x uint32, h hash.Hash, tmp []byte) {
	tmp = tmp[:0]
	tmp = append(tmp, 0, 0, 0, 0)
	binary.LittleEndian.PutUint32(tmp, x)
	h.Write(tmp)
}

type funcDesc struct {
	encoded []byte
}

// AddFunc registers a new function with the meta data builder.
func (b *CoverageMetaDataBuilder) AddFunc(f coverage.FuncDesc) uint {
	hashFuncDesc(b.h, &f, b.tmp)
	fd := funcDesc{}
	b.tmp = b.tmp[:0]
	b.tmp = uleb128.AppendUleb128(b.tmp, uint(len(f.Units)))
	b.tmp = uleb128.AppendUleb128(b.tmp, uint(b.stab.Lookup(f.Funcname)))
	b.tmp = uleb128.AppendUleb128(b.tmp, uint(b.stab.Lookup(f.Srcfile)))
	for _, u := range f.Units {
		b.tmp = uleb128.AppendUleb128(b.tmp, uint(u.StLine))
		b.tmp = uleb128.AppendUleb128(b.tmp, uint(u.StCol))
		b.tmp = uleb128.AppendUleb128(b.tmp, uint(u.EnLine))
		b.tmp = uleb128.AppendUleb128(b.tmp, uint(u.EnCol))
		b.tmp = uleb128.AppendUleb128(b.tmp, uint(u.NxStmts))
	}
	lit := uint(0)
	if f.Lit {
		lit = 1
	}
	b.tmp = uleb128.AppendUleb128(b.tmp, lit)
	fd.encoded = bytes.Clone(b.tmp)
	rv := uint(len(b.funcs))
	b.funcs = append(b.funcs, fd)
	return rv
}

func (b *CoverageMetaDataBuilder) emitFuncOffsets(w io.WriteSeeker, off int64) int64 {
	nFuncs := len(b.funcs)
	var foff int64 = coverage.CovMetaHeaderSize + int64(b.stab.Size()) + int64(nFuncs)*4
	for idx := 0; idx < nFuncs; idx++ {
		b.wrUint32(w, uint32(foff))
		foff += int64(len(b.funcs[idx].encoded))
	}
	return off + (int64(len(b.funcs)) * 4)
}

func (b *CoverageMetaDataBuilder) emitFunc(w io.WriteSeeker, off int64, f funcDesc) (int64, error) {
	ew := len(f.encoded)
	if nw, err := w.Write(f.encoded); err != nil {
		return 0, err
	} else if ew != nw {
		return 0, fmt.Errorf("short write emitting coverage meta-data")
	}
	return off + int64(ew), nil
}

func (b *CoverageMetaDataBuilder) reportWriteError(err error) {
	if b.werr != nil {
		b.werr = err
	}
}

func (b *CoverageMetaDataBuilder) wrUint32(w io.WriteSeeker, v uint32) {
	b.tmp = b.tmp[:0]
	b.tmp = append(b.tmp, 0, 0, 0, 0)
	binary.LittleEndian.PutUint32(b.tmp, v)
	if nw, err := w.Write(b.tmp); err != nil {
		b.reportWriteError(err)
	} else if nw != 4 {
		b.reportWriteError(fmt.Errorf("short write"))
	}
}

// Emit writes the meta-data accumulated so far in this builder to 'w'.
// Returns a hash of the meta-data payload and an error.
func (b *CoverageMetaDataBuilder) Emit(w io.WriteSeeker) ([16]byte, error) {
	// Emit header.  Length will initially be zero, we'll
	// back-patch it later.
	var digest [16]byte
	copy(digest[:], b.h.Sum(nil))
	mh := coverage.MetaSymbolHeader{
		// hash and length initially zero, will be back-patched
		PkgPath:    uint32(b.pkgpath),
		PkgName:    uint32(b.pkgname),
		ModulePath: uint32(b.modpath),
		NumFiles:   uint32(b.stab.Nentries()),
		NumFuncs:   uint32(len(b.funcs)),
		MetaHash:   digest,
	}
	if b.debug {
		fmt.Fprintf(os.Stderr, "=-= writing header: %+v\n", mh)
	}
	if err := binary.Write(w, binary.LittleEndian, mh); err != nil {
		return digest, fmt.Errorf("error writing meta-file header: %v", err)
	}
	off := int64(coverage.CovMetaHeaderSize)

	// Write function offsets section
	off = b.emitFuncOffsets(w, off)

	// Check for any errors up to this point.
	if b.werr != nil {
		return digest, b.werr
	}

	// Write string table.
	if err := b.stab.Write(w); err != nil {
		return digest, err
	}
	off += int64(b.stab.Size())

	// Write functions
	for _, f := range b.funcs {
		var err error
		off, err = b.emitFunc(w, off, f)
		if err != nil {
			return digest, err
		}
	}

	// Back-patch the length.
	totalLength := uint32(off)
	if _, err := w.Seek(0, io.SeekStart); err != nil {
		return digest, err
	}
	b.wrUint32(w, totalLength)
	if b.werr != nil {
		return digest, b.werr
	}
	return digest, nil
}

// HashFuncDesc computes an md5 sum of a coverage.FuncDesc and returns
// a digest for it.
func HashFuncDesc(f *coverage.FuncDesc) [16]byte {
	h := md5.New()
	tmp := make([]byte, 0, 32)
	hashFuncDesc(h, f, tmp)
	var r [16]byte
	copy(r[:], h.Sum(nil))
	return r
}

// hashFuncDesc incorporates a given function 'f' into the hash 'h'.
func hashFuncDesc(h hash.Hash, f *coverage.FuncDesc, tmp []byte) {
	io.WriteString(h, f.Funcname)
	io.WriteString(h, f.Srcfile)
	for _, u := range f.Units {
		h32(u.StLine, h, tmp)
		h32(u.StCol, h, tmp)
		h32(u.EnLine, h, tmp)
		h32(u.EnCol, h, tmp)
		h32(u.NxStmts, h, tmp)
	}
	lit := uint32(0)
	if f.Lit {
		lit = 1
	}
	h32(lit, h, tmp)
}
