// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj

import (
	"bytes"
	"cmd/internal/objabi"
	"encoding/binary"
)

// CUFileIndex is used to index the filenames that are stored in the
// per-package/per-CU FileList.
type CUFileIndex uint32

// FuncInfo is serialized as a symbol (aux symbol). The symbol data is
// the binary encoding of the struct below.
//
// TODO: make each pcdata a separate symbol?
type FuncInfo struct {
	Args   uint32
	Locals uint32
	FuncID objabi.FuncID

	Pcsp        SymRef
	Pcfile      SymRef
	Pcline      SymRef
	Pcinline    SymRef
	Pcdata      []SymRef
	Funcdataoff []uint32
	File        []CUFileIndex

	InlTree []InlTreeNode
}

func (a *FuncInfo) Write(w *bytes.Buffer) {
	var b [4]byte
	writeUint32 := func(x uint32) {
		binary.LittleEndian.PutUint32(b[:], x)
		w.Write(b[:])
	}
	writeSymRef := func(s SymRef) {
		writeUint32(s.PkgIdx)
		writeUint32(s.SymIdx)
	}

	writeUint32(a.Args)
	writeUint32(a.Locals)
	writeUint32(uint32(a.FuncID))

	writeSymRef(a.Pcsp)
	writeSymRef(a.Pcfile)
	writeSymRef(a.Pcline)
	writeSymRef(a.Pcinline)
	writeUint32(uint32(len(a.Pcdata)))
	for _, sym := range a.Pcdata {
		writeSymRef(sym)
	}

	writeUint32(uint32(len(a.Funcdataoff)))
	for _, x := range a.Funcdataoff {
		writeUint32(x)
	}
	writeUint32(uint32(len(a.File)))
	for _, f := range a.File {
		writeUint32(uint32(f))
	}
	writeUint32(uint32(len(a.InlTree)))
	for i := range a.InlTree {
		a.InlTree[i].Write(w)
	}
}

func (a *FuncInfo) Read(b []byte) {
	readUint32 := func() uint32 {
		x := binary.LittleEndian.Uint32(b)
		b = b[4:]
		return x
	}
	readSymIdx := func() SymRef {
		return SymRef{readUint32(), readUint32()}
	}

	a.Args = readUint32()
	a.Locals = readUint32()
	a.FuncID = objabi.FuncID(readUint32())

	a.Pcsp = readSymIdx()
	a.Pcfile = readSymIdx()
	a.Pcline = readSymIdx()
	a.Pcinline = readSymIdx()
	a.Pcdata = make([]SymRef, readUint32())
	for i := range a.Pcdata {
		a.Pcdata[i] = readSymIdx()
	}

	funcdataofflen := readUint32()
	a.Funcdataoff = make([]uint32, funcdataofflen)
	for i := range a.Funcdataoff {
		a.Funcdataoff[i] = readUint32()
	}
	filelen := readUint32()
	a.File = make([]CUFileIndex, filelen)
	for i := range a.File {
		a.File[i] = CUFileIndex(readUint32())
	}
	inltreelen := readUint32()
	a.InlTree = make([]InlTreeNode, inltreelen)
	for i := range a.InlTree {
		b = a.InlTree[i].Read(b)
	}
}

// FuncInfoLengths is a cache containing a roadmap of offsets and
// lengths for things within a serialized FuncInfo. Each length field
// stores the number of items (e.g. files, inltree nodes, etc), and the
// corresponding "off" field stores the byte offset of the start of
// the items in question.
type FuncInfoLengths struct {
	NumPcdata      uint32
	PcdataOff      uint32
	NumFuncdataoff uint32
	FuncdataoffOff uint32
	NumFile        uint32
	FileOff        uint32
	NumInlTree     uint32
	InlTreeOff     uint32
	Initialized    bool
}

func (*FuncInfo) ReadFuncInfoLengths(b []byte) FuncInfoLengths {
	var result FuncInfoLengths

	// Offset to the number of pcdata values. This value is determined by counting
	// the number of bytes until we write pcdata to the file.
	const numpcdataOff = 44
	result.NumPcdata = binary.LittleEndian.Uint32(b[numpcdataOff:])
	result.PcdataOff = numpcdataOff + 4

	numfuncdataoffOff := result.PcdataOff + 8*result.NumPcdata
	result.NumFuncdataoff = binary.LittleEndian.Uint32(b[numfuncdataoffOff:])
	result.FuncdataoffOff = numfuncdataoffOff + 4

	numfileOff := result.FuncdataoffOff + 4*result.NumFuncdataoff
	result.NumFile = binary.LittleEndian.Uint32(b[numfileOff:])
	result.FileOff = numfileOff + 4

	numinltreeOff := result.FileOff + 4*result.NumFile
	result.NumInlTree = binary.LittleEndian.Uint32(b[numinltreeOff:])
	result.InlTreeOff = numinltreeOff + 4

	result.Initialized = true

	return result
}

func (*FuncInfo) ReadArgs(b []byte) uint32 { return binary.LittleEndian.Uint32(b) }

func (*FuncInfo) ReadLocals(b []byte) uint32 { return binary.LittleEndian.Uint32(b[4:]) }

func (*FuncInfo) ReadFuncID(b []byte) uint32 { return binary.LittleEndian.Uint32(b[8:]) }

func (*FuncInfo) ReadPcsp(b []byte) SymRef {
	return SymRef{binary.LittleEndian.Uint32(b[12:]), binary.LittleEndian.Uint32(b[16:])}
}

func (*FuncInfo) ReadPcfile(b []byte) SymRef {
	return SymRef{binary.LittleEndian.Uint32(b[20:]), binary.LittleEndian.Uint32(b[24:])}
}

func (*FuncInfo) ReadPcline(b []byte) SymRef {
	return SymRef{binary.LittleEndian.Uint32(b[28:]), binary.LittleEndian.Uint32(b[32:])}
}

func (*FuncInfo) ReadPcinline(b []byte) SymRef {
	return SymRef{binary.LittleEndian.Uint32(b[36:]), binary.LittleEndian.Uint32(b[40:])}
}

func (*FuncInfo) ReadPcdata(b []byte) []SymRef {
	syms := make([]SymRef, binary.LittleEndian.Uint32(b[44:]))
	for i := range syms {
		syms[i] = SymRef{binary.LittleEndian.Uint32(b[48+i*8:]), binary.LittleEndian.Uint32(b[52+i*8:])}
	}
	return syms
}

func (*FuncInfo) ReadFuncdataoff(b []byte, funcdataofffoff uint32, k uint32) int64 {
	return int64(binary.LittleEndian.Uint32(b[funcdataofffoff+4*k:]))
}

func (*FuncInfo) ReadFile(b []byte, filesoff uint32, k uint32) CUFileIndex {
	return CUFileIndex(binary.LittleEndian.Uint32(b[filesoff+4*k:]))
}

func (*FuncInfo) ReadInlTree(b []byte, inltreeoff uint32, k uint32) InlTreeNode {
	const inlTreeNodeSize = 4 * 6
	var result InlTreeNode
	result.Read(b[inltreeoff+k*inlTreeNodeSize:])
	return result
}

// InlTreeNode is the serialized form of FileInfo.InlTree.
type InlTreeNode struct {
	Parent   int32
	File     CUFileIndex
	Line     int32
	Func     SymRef
	ParentPC int32
}

func (inl *InlTreeNode) Write(w *bytes.Buffer) {
	var b [4]byte
	writeUint32 := func(x uint32) {
		binary.LittleEndian.PutUint32(b[:], x)
		w.Write(b[:])
	}
	writeUint32(uint32(inl.Parent))
	writeUint32(uint32(inl.File))
	writeUint32(uint32(inl.Line))
	writeUint32(inl.Func.PkgIdx)
	writeUint32(inl.Func.SymIdx)
	writeUint32(uint32(inl.ParentPC))
}

// Read an InlTreeNode from b, return the remaining bytes.
func (inl *InlTreeNode) Read(b []byte) []byte {
	readUint32 := func() uint32 {
		x := binary.LittleEndian.Uint32(b)
		b = b[4:]
		return x
	}
	inl.Parent = int32(readUint32())
	inl.File = CUFileIndex(readUint32())
	inl.Line = int32(readUint32())
	inl.Func = SymRef{readUint32(), readUint32()}
	inl.ParentPC = int32(readUint32())
	return b
}
