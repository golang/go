// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/obj"
	"cmd/internal/sys"
	"debug/elf"
	"fmt"
)

// Decoding the type.* symbols.	 This has to be in sync with
// ../../runtime/type.go, or more specifically, with what
// ../gc/reflect.c stuffs in these.

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

func decode_reloc(s *LSym, off int32) *Reloc {
	for i := range s.R {
		if s.R[i].Off == off {
			return &s.R[i]
		}
	}
	return nil
}

func decode_reloc_sym(s *LSym, off int32) *LSym {
	r := decode_reloc(s, off)
	if r == nil {
		return nil
	}
	return r.Sym
}

func decode_inuxi(p []byte, sz int) uint64 {
	switch sz {
	case 2:
		return uint64(Ctxt.Arch.ByteOrder.Uint16(p))
	case 4:
		return uint64(Ctxt.Arch.ByteOrder.Uint32(p))
	case 8:
		return Ctxt.Arch.ByteOrder.Uint64(p)
	default:
		Exitf("dwarf: decode inuxi %d", sz)
		panic("unreachable")
	}
}

func commonsize() int      { return 4*SysArch.PtrSize + 8 + 8 } // runtime._type
func structfieldSize() int { return 3 * SysArch.PtrSize }       // runtime.structfield
func uncommonSize() int    { return 4 + 2 + 2 + 4 + 4 }         // runtime.uncommontype

// Type.commonType.kind
func decodetype_kind(s *LSym) uint8 {
	return s.P[2*SysArch.PtrSize+7] & obj.KindMask //  0x13 / 0x1f
}

// Type.commonType.kind
func decodetype_usegcprog(s *LSym) uint8 {
	return s.P[2*SysArch.PtrSize+7] & obj.KindGCProg //  0x13 / 0x1f
}

// Type.commonType.size
func decodetype_size(s *LSym) int64 {
	return int64(decode_inuxi(s.P, SysArch.PtrSize)) // 0x8 / 0x10
}

// Type.commonType.ptrdata
func decodetype_ptrdata(s *LSym) int64 {
	return int64(decode_inuxi(s.P[SysArch.PtrSize:], SysArch.PtrSize)) // 0x8 / 0x10
}

// Type.commonType.tflag
func decodetype_hasUncommon(s *LSym) bool {
	return s.P[2*SysArch.PtrSize+4]&tflagUncommon != 0
}

// Find the elf.Section of a given shared library that contains a given address.
func findShlibSection(path string, addr uint64) *elf.Section {
	for _, shlib := range Ctxt.Shlibs {
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
func decodetype_gcprog(s *LSym) []byte {
	if s.Type == obj.SDYNIMPORT {
		addr := decodetype_gcprog_shlib(s)
		sect := findShlibSection(s.File, addr)
		if sect != nil {
			// A gcprog is a 4-byte uint32 indicating length, followed by
			// the actual program.
			progsize := make([]byte, 4)
			sect.ReadAt(progsize, int64(addr-sect.Addr))
			progbytes := make([]byte, Ctxt.Arch.ByteOrder.Uint32(progsize))
			sect.ReadAt(progbytes, int64(addr-sect.Addr+4))
			return append(progsize, progbytes...)
		}
		Exitf("cannot find gcprog for %s", s.Name)
		return nil
	}
	return decode_reloc_sym(s, 2*int32(SysArch.PtrSize)+8+1*int32(SysArch.PtrSize)).P
}

func decodetype_gcprog_shlib(s *LSym) uint64 {
	if SysArch.Family == sys.ARM64 {
		for _, shlib := range Ctxt.Shlibs {
			if shlib.Path == s.File {
				return shlib.gcdata_addresses[s]
			}
		}
		return 0
	}
	return decode_inuxi(s.P[2*int32(SysArch.PtrSize)+8+1*int32(SysArch.PtrSize):], SysArch.PtrSize)
}

func decodetype_gcmask(s *LSym) []byte {
	if s.Type == obj.SDYNIMPORT {
		addr := decodetype_gcprog_shlib(s)
		ptrdata := decodetype_ptrdata(s)
		sect := findShlibSection(s.File, addr)
		if sect != nil {
			r := make([]byte, ptrdata/int64(SysArch.PtrSize))
			sect.ReadAt(r, int64(addr-sect.Addr))
			return r
		}
		Exitf("cannot find gcmask for %s", s.Name)
		return nil
	}
	mask := decode_reloc_sym(s, 2*int32(SysArch.PtrSize)+8+1*int32(SysArch.PtrSize))
	return mask.P
}

// Type.ArrayType.elem and Type.SliceType.Elem
func decodetype_arrayelem(s *LSym) *LSym {
	return decode_reloc_sym(s, int32(commonsize())) // 0x1c / 0x30
}

func decodetype_arraylen(s *LSym) int64 {
	return int64(decode_inuxi(s.P[commonsize()+2*SysArch.PtrSize:], SysArch.PtrSize))
}

// Type.PtrType.elem
func decodetype_ptrelem(s *LSym) *LSym {
	return decode_reloc_sym(s, int32(commonsize())) // 0x1c / 0x30
}

// Type.MapType.key, elem
func decodetype_mapkey(s *LSym) *LSym {
	return decode_reloc_sym(s, int32(commonsize())) // 0x1c / 0x30
}

func decodetype_mapvalue(s *LSym) *LSym {
	return decode_reloc_sym(s, int32(commonsize())+int32(SysArch.PtrSize)) // 0x20 / 0x38
}

// Type.ChanType.elem
func decodetype_chanelem(s *LSym) *LSym {
	return decode_reloc_sym(s, int32(commonsize())) // 0x1c / 0x30
}

// Type.FuncType.dotdotdot
func decodetype_funcdotdotdot(s *LSym) bool {
	return uint16(decode_inuxi(s.P[commonsize()+2:], 2))&(1<<15) != 0
}

// Type.FuncType.inCount
func decodetype_funcincount(s *LSym) int {
	return int(decode_inuxi(s.P[commonsize():], 2))
}

func decodetype_funcoutcount(s *LSym) int {
	return int(uint16(decode_inuxi(s.P[commonsize()+2:], 2)) & (1<<15 - 1))
}

func decodetype_funcintype(s *LSym, i int) *LSym {
	uadd := commonsize() + 4
	if SysArch.PtrSize == 8 {
		uadd += 4
	}
	if decodetype_hasUncommon(s) {
		uadd += uncommonSize()
	}
	return decode_reloc_sym(s, int32(uadd+i*SysArch.PtrSize))
}

func decodetype_funcouttype(s *LSym, i int) *LSym {
	return decodetype_funcintype(s, i+decodetype_funcincount(s))
}

// Type.StructType.fields.Slice::length
func decodetype_structfieldcount(s *LSym) int {
	return int(decode_inuxi(s.P[commonsize()+2*SysArch.PtrSize:], SysArch.IntSize))
}

func decodetype_structfieldarrayoff(s *LSym, i int) int {
	off := commonsize() + 2*SysArch.PtrSize + 2*SysArch.IntSize
	if decodetype_hasUncommon(s) {
		off += uncommonSize()
	}
	off += i * structfieldSize()
	return off
}

// decodetype_str returns the contents of an rtype's str field (a nameOff).
func decodetype_str(s *LSym) string {
	str := decodetype_name(s, 4*SysArch.PtrSize+8)
	if s.P[2*SysArch.PtrSize+4]&tflagExtraStar != 0 {
		return str[1:]
	}
	return str
}

// decodetype_name decodes the name from a reflect.name.
func decodetype_name(s *LSym, off int) string {
	r := decode_reloc(s, int32(off))
	if r == nil {
		return ""
	}

	data := r.Sym.P
	namelen := int(uint16(data[1])<<8 | uint16(data[2]))
	return string(data[3 : 3+namelen])
}

func decodetype_structfieldname(s *LSym, i int) string {
	off := decodetype_structfieldarrayoff(s, i)
	return decodetype_name(s, off)
}

func decodetype_structfieldtype(s *LSym, i int) *LSym {
	off := decodetype_structfieldarrayoff(s, i)
	return decode_reloc_sym(s, int32(off+SysArch.PtrSize))
}

func decodetype_structfieldoffs(s *LSym, i int) int64 {
	off := decodetype_structfieldarrayoff(s, i)
	return int64(decode_inuxi(s.P[off+2*SysArch.PtrSize:], SysArch.IntSize))
}

// InterfaceType.methods.length
func decodetype_ifacemethodcount(s *LSym) int64 {
	return int64(decode_inuxi(s.P[commonsize()+2*SysArch.PtrSize:], SysArch.IntSize))
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

// decode_methodsig decodes an array of method signature information.
// Each element of the array is size bytes. The first 4 bytes is a
// nameOff for the method name, and the next 4 bytes is a typeOff for
// the function type.
//
// Conveniently this is the layout of both runtime.method and runtime.imethod.
func decode_methodsig(s *LSym, off, size, count int) []methodsig {
	var buf bytes.Buffer
	var methods []methodsig
	for i := 0; i < count; i++ {
		buf.WriteString(decodetype_name(s, off))
		mtypSym := decode_reloc_sym(s, int32(off+4))

		buf.WriteRune('(')
		inCount := decodetype_funcincount(mtypSym)
		for i := 0; i < inCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(decodetype_funcintype(mtypSym, i).Name)
		}
		buf.WriteString(") (")
		outCount := decodetype_funcoutcount(mtypSym)
		for i := 0; i < outCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(decodetype_funcouttype(mtypSym, i).Name)
		}
		buf.WriteRune(')')

		off += size
		methods = append(methods, methodsig(buf.String()))
		buf.Reset()
	}
	return methods
}

func decodetype_ifacemethods(s *LSym) []methodsig {
	if decodetype_kind(s)&kindMask != kindInterface {
		panic(fmt.Sprintf("symbol %q is not an interface", s.Name))
	}
	r := decode_reloc(s, int32(commonsize()+SysArch.PtrSize))
	if r == nil {
		return nil
	}
	if r.Sym != s {
		panic(fmt.Sprintf("imethod slice pointer in %q leads to a different symbol", s.Name))
	}
	off := int(r.Add) // array of reflect.imethod values
	numMethods := int(decodetype_ifacemethodcount(s))
	sizeofIMethod := 4 + 4
	return decode_methodsig(s, off, sizeofIMethod, numMethods)
}

func decodetype_methods(s *LSym) []methodsig {
	if !decodetype_hasUncommon(s) {
		panic(fmt.Sprintf("no methods on %q", s.Name))
	}
	off := commonsize() // reflect.rtype
	switch decodetype_kind(s) & kindMask {
	case kindStruct: // reflect.structType
		off += 2*SysArch.PtrSize + 2*SysArch.IntSize
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
		off += SysArch.PtrSize + 2*SysArch.IntSize
	default:
		// just Sizeof(rtype)
	}

	mcount := int(decode_inuxi(s.P[off+4:], 2))
	moff := int(decode_inuxi(s.P[off+4+2+2:], 4))
	off += moff                // offset to array of reflect.method values
	const sizeofMethod = 4 * 4 // sizeof reflect.method in program
	return decode_methodsig(s, off, sizeofMethod, mcount)
}
