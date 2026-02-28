// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package codesign provides basic functionalities for
// ad-hoc code signing of Mach-O files.
//
// This is not a general tool for code-signing. It is made
// specifically for the Go toolchain. It uses the same
// ad-hoc signing algorithm as the Darwin linker.
package codesign

import (
	"debug/macho"
	"encoding/binary"
	"io"

	"cmd/internal/notsha256"
)

// Code signature layout.
//
// The code signature is a block of bytes that contains
// a SuperBlob, which contains one or more Blobs. For ad-hoc
// signing, a single CodeDirectory Blob suffices.
//
// A SuperBlob starts with its header (the binary representation
// of the SuperBlob struct), followed by a list of (in our case,
// one) Blobs (offset and size). A CodeDirectory Blob starts
// with its head (the binary representation of CodeDirectory struct),
// followed by the identifier (as a C string) and the hashes, at
// the corresponding offsets.
//
// The signature data must be included in the __LINKEDIT segment.
// In the Mach-O file header, an LC_CODE_SIGNATURE load command
// points to the data.

const (
	pageSizeBits = 12
	pageSize     = 1 << pageSizeBits
)

const LC_CODE_SIGNATURE = 0x1d

// Constants and struct layouts are from
// https://opensource.apple.com/source/xnu/xnu-4903.270.47/osfmk/kern/cs_blobs.h

const (
	CSMAGIC_REQUIREMENT        = 0xfade0c00 // single Requirement blob
	CSMAGIC_REQUIREMENTS       = 0xfade0c01 // Requirements vector (internal requirements)
	CSMAGIC_CODEDIRECTORY      = 0xfade0c02 // CodeDirectory blob
	CSMAGIC_EMBEDDED_SIGNATURE = 0xfade0cc0 // embedded form of signature data
	CSMAGIC_DETACHED_SIGNATURE = 0xfade0cc1 // multi-arch collection of embedded signatures

	CSSLOT_CODEDIRECTORY = 0 // slot index for CodeDirectory
)

const (
	CS_HASHTYPE_SHA1             = 1
	CS_HASHTYPE_SHA256           = 2
	CS_HASHTYPE_SHA256_TRUNCATED = 3
	CS_HASHTYPE_SHA384           = 4
)

const (
	CS_EXECSEG_MAIN_BINARY     = 0x1   // executable segment denotes main binary
	CS_EXECSEG_ALLOW_UNSIGNED  = 0x10  // allow unsigned pages (for debugging)
	CS_EXECSEG_DEBUGGER        = 0x20  // main binary is debugger
	CS_EXECSEG_JIT             = 0x40  // JIT enabled
	CS_EXECSEG_SKIP_LV         = 0x80  // skip library validation
	CS_EXECSEG_CAN_LOAD_CDHASH = 0x100 // can bless cdhash for execution
	CS_EXECSEG_CAN_EXEC_CDHASH = 0x200 // can execute blessed cdhash
)

type Blob struct {
	typ    uint32 // type of entry
	offset uint32 // offset of entry
	// data follows
}

func (b *Blob) put(out []byte) []byte {
	out = put32be(out, b.typ)
	out = put32be(out, b.offset)
	return out
}

const blobSize = 2 * 4

type SuperBlob struct {
	magic  uint32 // magic number
	length uint32 // total length of SuperBlob
	count  uint32 // number of index entries following
	// blobs []Blob
}

func (s *SuperBlob) put(out []byte) []byte {
	out = put32be(out, s.magic)
	out = put32be(out, s.length)
	out = put32be(out, s.count)
	return out
}

const superBlobSize = 3 * 4

type CodeDirectory struct {
	magic         uint32 // magic number (CSMAGIC_CODEDIRECTORY)
	length        uint32 // total length of CodeDirectory blob
	version       uint32 // compatibility version
	flags         uint32 // setup and mode flags
	hashOffset    uint32 // offset of hash slot element at index zero
	identOffset   uint32 // offset of identifier string
	nSpecialSlots uint32 // number of special hash slots
	nCodeSlots    uint32 // number of ordinary (code) hash slots
	codeLimit     uint32 // limit to main image signature range
	hashSize      uint8  // size of each hash in bytes
	hashType      uint8  // type of hash (cdHashType* constants)
	_pad1         uint8  // unused (must be zero)
	pageSize      uint8  // log2(page size in bytes); 0 => infinite
	_pad2         uint32 // unused (must be zero)
	scatterOffset uint32
	teamOffset    uint32
	_pad3         uint32
	codeLimit64   uint64
	execSegBase   uint64
	execSegLimit  uint64
	execSegFlags  uint64
	// data follows
}

func (c *CodeDirectory) put(out []byte) []byte {
	out = put32be(out, c.magic)
	out = put32be(out, c.length)
	out = put32be(out, c.version)
	out = put32be(out, c.flags)
	out = put32be(out, c.hashOffset)
	out = put32be(out, c.identOffset)
	out = put32be(out, c.nSpecialSlots)
	out = put32be(out, c.nCodeSlots)
	out = put32be(out, c.codeLimit)
	out = put8(out, c.hashSize)
	out = put8(out, c.hashType)
	out = put8(out, c._pad1)
	out = put8(out, c.pageSize)
	out = put32be(out, c._pad2)
	out = put32be(out, c.scatterOffset)
	out = put32be(out, c.teamOffset)
	out = put32be(out, c._pad3)
	out = put64be(out, c.codeLimit64)
	out = put64be(out, c.execSegBase)
	out = put64be(out, c.execSegLimit)
	out = put64be(out, c.execSegFlags)
	return out
}

const codeDirectorySize = 13*4 + 4 + 4*8

// CodeSigCmd is Mach-O LC_CODE_SIGNATURE load command.
type CodeSigCmd struct {
	Cmd      uint32 // LC_CODE_SIGNATURE
	Cmdsize  uint32 // sizeof this command (16)
	Dataoff  uint32 // file offset of data in __LINKEDIT segment
	Datasize uint32 // file size of data in __LINKEDIT segment
}

func FindCodeSigCmd(f *macho.File) (CodeSigCmd, bool) {
	get32 := f.ByteOrder.Uint32
	for _, l := range f.Loads {
		data := l.Raw()
		cmd := get32(data)
		if cmd == LC_CODE_SIGNATURE {
			return CodeSigCmd{
				cmd,
				get32(data[4:]),
				get32(data[8:]),
				get32(data[12:]),
			}, true
		}
	}
	return CodeSigCmd{}, false
}

func put32be(b []byte, x uint32) []byte { binary.BigEndian.PutUint32(b, x); return b[4:] }
func put64be(b []byte, x uint64) []byte { binary.BigEndian.PutUint64(b, x); return b[8:] }
func put8(b []byte, x uint8) []byte     { b[0] = x; return b[1:] }
func puts(b, s []byte) []byte           { n := copy(b, s); return b[n:] }

// Size computes the size of the code signature.
// id is the identifier used for signing (a field in CodeDirectory blob, which
// has no significance in ad-hoc signing).
func Size(codeSize int64, id string) int64 {
	nhashes := (codeSize + pageSize - 1) / pageSize
	idOff := int64(codeDirectorySize)
	hashOff := idOff + int64(len(id)+1)
	cdirSz := hashOff + nhashes*notsha256.Size
	return int64(superBlobSize+blobSize) + cdirSz
}

// Sign generates an ad-hoc code signature and writes it to out.
// out must have length at least Size(codeSize, id).
// data is the file content without the signature, of size codeSize.
// textOff and textSize is the file offset and size of the text segment.
// isMain is true if this is a main executable.
// id is the identifier used for signing (a field in CodeDirectory blob, which
// has no significance in ad-hoc signing).
func Sign(out []byte, data io.Reader, id string, codeSize, textOff, textSize int64, isMain bool) {
	nhashes := (codeSize + pageSize - 1) / pageSize
	idOff := int64(codeDirectorySize)
	hashOff := idOff + int64(len(id)+1)
	sz := Size(codeSize, id)

	// emit blob headers
	sb := SuperBlob{
		magic:  CSMAGIC_EMBEDDED_SIGNATURE,
		length: uint32(sz),
		count:  1,
	}
	blob := Blob{
		typ:    CSSLOT_CODEDIRECTORY,
		offset: superBlobSize + blobSize,
	}
	cdir := CodeDirectory{
		magic:        CSMAGIC_CODEDIRECTORY,
		length:       uint32(sz) - (superBlobSize + blobSize),
		version:      0x20400,
		flags:        0x20002, // adhoc | linkerSigned
		hashOffset:   uint32(hashOff),
		identOffset:  uint32(idOff),
		nCodeSlots:   uint32(nhashes),
		codeLimit:    uint32(codeSize),
		hashSize:     notsha256.Size,
		hashType:     CS_HASHTYPE_SHA256,
		pageSize:     uint8(pageSizeBits),
		execSegBase:  uint64(textOff),
		execSegLimit: uint64(textSize),
	}
	if isMain {
		cdir.execSegFlags = CS_EXECSEG_MAIN_BINARY
	}

	outp := out
	outp = sb.put(outp)
	outp = blob.put(outp)
	outp = cdir.put(outp)

	// emit the identifier
	outp = puts(outp, []byte(id+"\000"))

	// emit hashes
	// NOTE(rsc): These must be SHA256, but for cgo bootstrap reasons
	// we cannot import crypto/sha256 when GOEXPERIMENT=boringcrypto
	// and the host is linux/amd64. So we use NOT-SHA256
	// and then apply a NOT ourselves to get SHA256. Sigh.
	var buf [pageSize]byte
	h := notsha256.New()
	p := 0
	for p < int(codeSize) {
		n, err := io.ReadFull(data, buf[:])
		if err == io.EOF {
			break
		}
		if err != nil && err != io.ErrUnexpectedEOF {
			panic(err)
		}
		if p+n > int(codeSize) {
			n = int(codeSize) - p
		}
		p += n
		h.Reset()
		h.Write(buf[:n])
		b := h.Sum(nil)
		for i := range b {
			b[i] ^= 0xFF // convert notsha256 to sha256
		}
		outp = puts(outp, b[:])
	}
}
