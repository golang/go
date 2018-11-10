// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"debug/elf"
	"fmt"
)

// Decoding the type.* symbols.	 This has to be in sync with
// ../../runtime/type.go, or more specifically, with what
// cmd/compile/internal/gc/reflect.go stuffs in these.

// tflag is documented in reflect/type.go.
//
// tflag values must be kept in sync with copies in:
//	cmd/compile/internal/gc/reflect.go
//	cmd/link/internal/ld/decodesym.go
//	reflect/type.go
//	runtime/type.go
const (
	tflagUncommon  = 1 << 0
	tflagExtraStar = 1 << 1
)

func decodeReloc(s *Symbol, off int32) *Reloc {
	for i := range s.R {
		if s.R[i].Off == off {
			return &s.R[i]
		}
	}
	return nil
}

func decodeRelocSym(s *Symbol, off int32) *Symbol {
	r := decodeReloc(s, off)
	if r == nil {
		return nil
	}
	return r.Sym
}

func decodeInuxi(arch *sys.Arch, p []byte, sz int) uint64 {
	switch sz {
	case 2:
		return uint64(arch.ByteOrder.Uint16(p))
	case 4:
		return uint64(arch.ByteOrder.Uint32(p))
	case 8:
		return arch.ByteOrder.Uint64(p)
	default:
		Exitf("dwarf: decode inuxi %d", sz)
		panic("unreachable")
	}
}

func commonsize() int      { return 4*SysArch.PtrSize + 8 + 8 } // runtime._type
func structfieldSize() int { return 3 * SysArch.PtrSize }       // runtime.structfield
func uncommonSize() int    { return 4 + 2 + 2 + 4 + 4 }         // runtime.uncommontype

// Type.commonType.kind
func decodetypeKind(s *Symbol) uint8 {
	return s.P[2*SysArch.PtrSize+7] & objabi.KindMask //  0x13 / 0x1f
}

// Type.commonType.kind
func decodetypeUsegcprog(s *Symbol) uint8 {
	return s.P[2*SysArch.PtrSize+7] & objabi.KindGCProg //  0x13 / 0x1f
}

// Type.commonType.size
func decodetypeSize(arch *sys.Arch, s *Symbol) int64 {
	return int64(decodeInuxi(arch, s.P, SysArch.PtrSize)) // 0x8 / 0x10
}

// Type.commonType.ptrdata
func decodetypePtrdata(arch *sys.Arch, s *Symbol) int64 {
	return int64(decodeInuxi(arch, s.P[SysArch.PtrSize:], SysArch.PtrSize)) // 0x8 / 0x10
}

// Type.commonType.tflag
func decodetypeHasUncommon(s *Symbol) bool {
	return s.P[2*SysArch.PtrSize+4]&tflagUncommon != 0
}

// Find the elf.Section of a given shared library that contains a given address.
func findShlibSection(ctxt *Link, path string, addr uint64) *elf.Section {
	for _, shlib := range ctxt.Shlibs {
		if shlib.Path == path {
			for _, sect := range shlib.File.Sections {
				if sect.Addr <= addr && addr <= sect.Addr+sect.Size {
					return sect
				}
			}
		}
	}
	return nil
}

// Type.commonType.gc
func decodetypeGcprog(ctxt *Link, s *Symbol) []byte {
	if s.Type == SDYNIMPORT {
		addr := decodetypeGcprogShlib(ctxt, s)
		sect := findShlibSection(ctxt, s.File, addr)
		if sect != nil {
			// A gcprog is a 4-byte uint32 indicating length, followed by
			// the actual program.
			progsize := make([]byte, 4)
			sect.ReadAt(progsize, int64(addr-sect.Addr))
			progbytes := make([]byte, ctxt.Arch.ByteOrder.Uint32(progsize))
			sect.ReadAt(progbytes, int64(addr-sect.Addr+4))
			return append(progsize, progbytes...)
		}
		Exitf("cannot find gcprog for %s", s.Name)
		return nil
	}
	return decodeRelocSym(s, 2*int32(SysArch.PtrSize)+8+1*int32(SysArch.PtrSize)).P
}

func decodetypeGcprogShlib(ctxt *Link, s *Symbol) uint64 {
	if SysArch.Family == sys.ARM64 {
		for _, shlib := range ctxt.Shlibs {
			if shlib.Path == s.File {
				return shlib.gcdataAddresses[s]
			}
		}
		return 0
	}
	return decodeInuxi(ctxt.Arch, s.P[2*int32(SysArch.PtrSize)+8+1*int32(SysArch.PtrSize):], SysArch.PtrSize)
}

func decodetypeGcmask(ctxt *Link, s *Symbol) []byte {
	if s.Type == SDYNIMPORT {
		addr := decodetypeGcprogShlib(ctxt, s)
		ptrdata := decodetypePtrdata(ctxt.Arch, s)
		sect := findShlibSection(ctxt, s.File, addr)
		if sect != nil {
			r := make([]byte, ptrdata/int64(SysArch.PtrSize))
			sect.ReadAt(r, int64(addr-sect.Addr))
			return r
		}
		Exitf("cannot find gcmask for %s", s.Name)
		return nil
	}
	mask := decodeRelocSym(s, 2*int32(SysArch.PtrSize)+8+1*int32(SysArch.PtrSize))
	return mask.P
}

// Type.ArrayType.elem and Type.SliceType.Elem
func decodetypeArrayElem(s *Symbol) *Symbol {
	return decodeRelocSym(s, int32(commonsize())) // 0x1c / 0x30
}

func decodetypeArrayLen(arch *sys.Arch, s *Symbol) int64 {
	return int64(decodeInuxi(arch, s.P[commonsize()+2*SysArch.PtrSize:], SysArch.PtrSize))
}

// Type.PtrType.elem
func decodetypePtrElem(s *Symbol) *Symbol {
	return decodeRelocSym(s, int32(commonsize())) // 0x1c / 0x30
}

// Type.MapType.key, elem
func decodetypeMapKey(s *Symbol) *Symbol {
	return decodeRelocSym(s, int32(commonsize())) // 0x1c / 0x30
}

func decodetypeMapValue(s *Symbol) *Symbol {
	return decodeRelocSym(s, int32(commonsize())+int32(SysArch.PtrSize)) // 0x20 / 0x38
}

// Type.ChanType.elem
func decodetypeChanElem(s *Symbol) *Symbol {
	return decodeRelocSym(s, int32(commonsize())) // 0x1c / 0x30
}

// Type.FuncType.dotdotdot
func decodetypeFuncDotdotdot(arch *sys.Arch, s *Symbol) bool {
	return uint16(decodeInuxi(arch, s.P[commonsize()+2:], 2))&(1<<15) != 0
}

// Type.FuncType.inCount
func decodetypeFuncInCount(arch *sys.Arch, s *Symbol) int {
	return int(decodeInuxi(arch, s.P[commonsize():], 2))
}

func decodetypeFuncOutCount(arch *sys.Arch, s *Symbol) int {
	return int(uint16(decodeInuxi(arch, s.P[commonsize()+2:], 2)) & (1<<15 - 1))
}

func decodetypeFuncInType(s *Symbol, i int) *Symbol {
	uadd := commonsize() + 4
	if SysArch.PtrSize == 8 {
		uadd += 4
	}
	if decodetypeHasUncommon(s) {
		uadd += uncommonSize()
	}
	return decodeRelocSym(s, int32(uadd+i*SysArch.PtrSize))
}

func decodetypeFuncOutType(arch *sys.Arch, s *Symbol, i int) *Symbol {
	return decodetypeFuncInType(s, i+decodetypeFuncInCount(arch, s))
}

// Type.StructType.fields.Slice::length
func decodetypeStructFieldCount(arch *sys.Arch, s *Symbol) int {
	return int(decodeInuxi(arch, s.P[commonsize()+2*SysArch.PtrSize:], SysArch.PtrSize))
}

func decodetypeStructFieldArrayOff(s *Symbol, i int) int {
	off := commonsize() + 4*SysArch.PtrSize
	if decodetypeHasUncommon(s) {
		off += uncommonSize()
	}
	off += i * structfieldSize()
	return off
}

// decodetypeStr returns the contents of an rtype's str field (a nameOff).
func decodetypeStr(s *Symbol) string {
	str := decodetypeName(s, 4*SysArch.PtrSize+8)
	if s.P[2*SysArch.PtrSize+4]&tflagExtraStar != 0 {
		return str[1:]
	}
	return str
}

// decodetypeName decodes the name from a reflect.name.
func decodetypeName(s *Symbol, off int) string {
	r := decodeReloc(s, int32(off))
	if r == nil {
		return ""
	}

	data := r.Sym.P
	namelen := int(uint16(data[1])<<8 | uint16(data[2]))
	return string(data[3 : 3+namelen])
}

func decodetypeStructFieldName(s *Symbol, i int) string {
	off := decodetypeStructFieldArrayOff(s, i)
	return decodetypeName(s, off)
}

func decodetypeStructFieldType(s *Symbol, i int) *Symbol {
	off := decodetypeStructFieldArrayOff(s, i)
	return decodeRelocSym(s, int32(off+SysArch.PtrSize))
}

func decodetypeStructFieldOffs(arch *sys.Arch, s *Symbol, i int) int64 {
	return decodetypeStructFieldOffsAnon(arch, s, i) >> 1
}

func decodetypeStructFieldOffsAnon(arch *sys.Arch, s *Symbol, i int) int64 {
	off := decodetypeStructFieldArrayOff(s, i)
	return int64(decodeInuxi(arch, s.P[off+2*SysArch.PtrSize:], SysArch.PtrSize))
}

// InterfaceType.methods.length
func decodetypeIfaceMethodCount(arch *sys.Arch, s *Symbol) int64 {
	return int64(decodeInuxi(arch, s.P[commonsize()+2*SysArch.PtrSize:], SysArch.PtrSize))
}

// methodsig is a fully qualified typed method signature, like
// "Visit(type.go/ast.Node) (type.go/ast.Visitor)".
type methodsig string

// Matches runtime/typekind.go and reflect.Kind.
const (
	kindArray     = 17
	kindChan      = 18
	kindFunc      = 19
	kindInterface = 20
	kindMap       = 21
	kindPtr       = 22
	kindSlice     = 23
	kindStruct    = 25
	kindMask      = (1 << 5) - 1
)

// decodeMethodSig decodes an array of method signature information.
// Each element of the array is size bytes. The first 4 bytes is a
// nameOff for the method name, and the next 4 bytes is a typeOff for
// the function type.
//
// Conveniently this is the layout of both runtime.method and runtime.imethod.
func decodeMethodSig(arch *sys.Arch, s *Symbol, off, size, count int) []methodsig {
	var buf bytes.Buffer
	var methods []methodsig
	for i := 0; i < count; i++ {
		buf.WriteString(decodetypeName(s, off))
		mtypSym := decodeRelocSym(s, int32(off+4))

		buf.WriteRune('(')
		inCount := decodetypeFuncInCount(arch, mtypSym)
		for i := 0; i < inCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(decodetypeFuncInType(mtypSym, i).Name)
		}
		buf.WriteString(") (")
		outCount := decodetypeFuncOutCount(arch, mtypSym)
		for i := 0; i < outCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(decodetypeFuncOutType(arch, mtypSym, i).Name)
		}
		buf.WriteRune(')')

		off += size
		methods = append(methods, methodsig(buf.String()))
		buf.Reset()
	}
	return methods
}

func decodeIfaceMethods(arch *sys.Arch, s *Symbol) []methodsig {
	if decodetypeKind(s)&kindMask != kindInterface {
		panic(fmt.Sprintf("symbol %q is not an interface", s.Name))
	}
	r := decodeReloc(s, int32(commonsize()+SysArch.PtrSize))
	if r == nil {
		return nil
	}
	if r.Sym != s {
		panic(fmt.Sprintf("imethod slice pointer in %q leads to a different symbol", s.Name))
	}
	off := int(r.Add) // array of reflect.imethod values
	numMethods := int(decodetypeIfaceMethodCount(arch, s))
	sizeofIMethod := 4 + 4
	return decodeMethodSig(arch, s, off, sizeofIMethod, numMethods)
}

func decodetypeMethods(arch *sys.Arch, s *Symbol) []methodsig {
	if !decodetypeHasUncommon(s) {
		panic(fmt.Sprintf("no methods on %q", s.Name))
	}
	off := commonsize() // reflect.rtype
	switch decodetypeKind(s) & kindMask {
	case kindStruct: // reflect.structType
		off += 4 * SysArch.PtrSize
	case kindPtr: // reflect.ptrType
		off += SysArch.PtrSize
	case kindFunc: // reflect.funcType
		off += SysArch.PtrSize // 4 bytes, pointer aligned
	case kindSlice: // reflect.sliceType
		off += SysArch.PtrSize
	case kindArray: // reflect.arrayType
		off += 3 * SysArch.PtrSize
	case kindChan: // reflect.chanType
		off += 2 * SysArch.PtrSize
	case kindMap: // reflect.mapType
		off += 4*SysArch.PtrSize + 8
	case kindInterface: // reflect.interfaceType
		off += 3 * SysArch.PtrSize
	default:
		// just Sizeof(rtype)
	}

	mcount := int(decodeInuxi(arch, s.P[off+4:], 2))
	moff := int(decodeInuxi(arch, s.P[off+4+2+2:], 4))
	off += moff                // offset to array of reflect.method values
	const sizeofMethod = 4 * 4 // sizeof reflect.method in program
	return decodeMethodSig(arch, s, off, sizeofMethod, mcount)
}
