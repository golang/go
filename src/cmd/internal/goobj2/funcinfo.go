// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goobj2

import (
	"bytes"
	"encoding/binary"
)

// FuncInfo is serialized as a symbol (aux symbol). The symbol data is
// the binary encoding of the struct below.
//
// TODO: make each pcdata a separate symbol?
type FuncInfo struct {
	NoSplit uint8

	Args   uint32
	Locals uint32

	Pcsp        uint32
	Pcfile      uint32
	Pcline      uint32
	Pcinline    uint32
	Pcdata      []uint32
	PcdataEnd   uint32
	Funcdataoff []uint32
	File        []SymRef // TODO: just use string?

	InlTree []InlTreeNode
}

func (a *FuncInfo) Write(w *bytes.Buffer) {
	w.WriteByte(a.NoSplit)

	var b [4]byte
	writeUint32 := func(x uint32) {
		binary.LittleEndian.PutUint32(b[:], x)
		w.Write(b[:])
	}

	writeUint32(a.Args)
	writeUint32(a.Locals)

	writeUint32(a.Pcsp)
	writeUint32(a.Pcfile)
	writeUint32(a.Pcline)
	writeUint32(a.Pcinline)
	writeUint32(uint32(len(a.Pcdata)))
	for _, x := range a.Pcdata {
		writeUint32(x)
	}
	writeUint32(a.PcdataEnd)
	writeUint32(uint32(len(a.Funcdataoff)))
	for _, x := range a.Funcdataoff {
		writeUint32(x)
	}
	writeUint32(uint32(len(a.File)))
	for _, f := range a.File {
		writeUint32(f.PkgIdx)
		writeUint32(f.SymIdx)
	}
	writeUint32(uint32(len(a.InlTree)))
	for i := range a.InlTree {
		a.InlTree[i].Write(w)
	}
}

func (a *FuncInfo) Read(b []byte) {
	a.NoSplit = b[0]
	b = b[1:]

	readUint32 := func() uint32 {
		x := binary.LittleEndian.Uint32(b)
		b = b[4:]
		return x
	}

	a.Args = readUint32()
	a.Locals = readUint32()

	a.Pcsp = readUint32()
	a.Pcfile = readUint32()
	a.Pcline = readUint32()
	a.Pcinline = readUint32()
	pcdatalen := readUint32()
	a.Pcdata = make([]uint32, pcdatalen)
	for i := range a.Pcdata {
		a.Pcdata[i] = readUint32()
	}
	a.PcdataEnd = readUint32()
	funcdataofflen := readUint32()
	a.Funcdataoff = make([]uint32, funcdataofflen)
	for i := range a.Funcdataoff {
		a.Funcdataoff[i] = readUint32()
	}
	filelen := readUint32()
	a.File = make([]SymRef, filelen)
	for i := range a.File {
		a.File[i] = SymRef{readUint32(), readUint32()}
	}
	inltreelen := readUint32()
	a.InlTree = make([]InlTreeNode, inltreelen)
	for i := range a.InlTree {
		b = a.InlTree[i].Read(b)
	}
}

// InlTreeNode is the serialized form of FileInfo.InlTree.
type InlTreeNode struct {
	Parent   int32
	File     SymRef
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
	writeUint32(inl.File.PkgIdx)
	writeUint32(inl.File.SymIdx)
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
	inl.File = SymRef{readUint32(), readUint32()}
	inl.Line = int32(readUint32())
	inl.Func = SymRef{readUint32(), readUint32()}
	inl.ParentPC = int32(readUint32())
	return b
}
