// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
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

func decodeReloc(s *sym.Symbol, off int32) *sym.Reloc {
	for i := range s.R {
		if s.R[i].Off == off {
			return &s.R[i]
		}
	}
	return nil
}

func decodeRelocSym(s *sym.Symbol, off int32) *sym.Symbol {
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

func commonsize(arch *sys.Arch) int      { return 4*arch.PtrSize + 8 + 8 } // runtime._type
func structfieldSize(arch *sys.Arch) int { return 3 * arch.PtrSize }       // runtime.structfield
func uncommonSize() int                  { return 4 + 2 + 2 + 4 + 4 }      // runtime.uncommontype

// Type.commonType.kind
func decodetypeKind(arch *sys.Arch, p []byte) uint8 {
	return p[2*arch.PtrSize+7] & objabi.KindMask //  0x13 / 0x1f
}

// Type.commonType.kind
func decodetypeUsegcprog(arch *sys.Arch, p []byte) uint8 {
	return p[2*arch.PtrSize+7] & objabi.KindGCProg //  0x13 / 0x1f
}

// Type.commonType.size
func decodetypeSize(arch *sys.Arch, p []byte) int64 {
	return int64(decodeInuxi(arch, p, arch.PtrSize)) // 0x8 / 0x10
}

// Type.commonType.ptrdata
func decodetypePtrdata(arch *sys.Arch, p []byte) int64 {
	return int64(decodeInuxi(arch, p[arch.PtrSize:], arch.PtrSize)) // 0x8 / 0x10
}

// Type.commonType.tflag
func decodetypeHasUncommon(arch *sys.Arch, p []byte) bool {
	return p[2*arch.PtrSize+4]&tflagUncommon != 0
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
func decodetypeGcprog(ctxt *Link, s *sym.Symbol) []byte {
	if s.Type == sym.SDYNIMPORT {
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
	return decodeRelocSym(s, 2*int32(ctxt.Arch.PtrSize)+8+1*int32(ctxt.Arch.PtrSize)).P
}

func decodetypeGcprogShlib(ctxt *Link, s *sym.Symbol) uint64 {
	if ctxt.Arch.Family == sys.ARM64 {
		for _, shlib := range ctxt.Shlibs {
			if shlib.Path == s.File {
				return shlib.gcdataAddresses[s]
			}
		}
		return 0
	}
	return decodeInuxi(ctxt.Arch, s.P[2*int32(ctxt.Arch.PtrSize)+8+1*int32(ctxt.Arch.PtrSize):], ctxt.Arch.PtrSize)
}

func decodetypeGcmask(ctxt *Link, s *sym.Symbol) []byte {
	if s.Type == sym.SDYNIMPORT {
		addr := decodetypeGcprogShlib(ctxt, s)
		ptrdata := decodetypePtrdata(ctxt.Arch, s.P)
		sect := findShlibSection(ctxt, s.File, addr)
		if sect != nil {
			r := make([]byte, ptrdata/int64(ctxt.Arch.PtrSize))
			sect.ReadAt(r, int64(addr-sect.Addr))
			return r
		}
		Exitf("cannot find gcmask for %s", s.Name)
		return nil
	}
	mask := decodeRelocSym(s, 2*int32(ctxt.Arch.PtrSize)+8+1*int32(ctxt.Arch.PtrSize))
	return mask.P
}

// Type.ArrayType.elem and Type.SliceType.Elem
func decodetypeArrayElem(arch *sys.Arch, s *sym.Symbol) *sym.Symbol {
	return decodeRelocSym(s, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeArrayLen(arch *sys.Arch, s *sym.Symbol) int64 {
	return int64(decodeInuxi(arch, s.P[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
}

// Type.PtrType.elem
func decodetypePtrElem(arch *sys.Arch, s *sym.Symbol) *sym.Symbol {
	return decodeRelocSym(s, int32(commonsize(arch))) // 0x1c / 0x30
}

// Type.MapType.key, elem
func decodetypeMapKey(arch *sys.Arch, s *sym.Symbol) *sym.Symbol {
	return decodeRelocSym(s, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeMapValue(arch *sys.Arch, s *sym.Symbol) *sym.Symbol {
	return decodeRelocSym(s, int32(commonsize(arch))+int32(arch.PtrSize)) // 0x20 / 0x38
}

// Type.ChanType.elem
func decodetypeChanElem(arch *sys.Arch, s *sym.Symbol) *sym.Symbol {
	return decodeRelocSym(s, int32(commonsize(arch))) // 0x1c / 0x30
}

// Type.FuncType.dotdotdot
func decodetypeFuncDotdotdot(arch *sys.Arch, p []byte) bool {
	return uint16(decodeInuxi(arch, p[commonsize(arch)+2:], 2))&(1<<15) != 0
}

// Type.FuncType.inCount
func decodetypeFuncInCount(arch *sys.Arch, p []byte) int {
	return int(decodeInuxi(arch, p[commonsize(arch):], 2))
}

func decodetypeFuncOutCount(arch *sys.Arch, p []byte) int {
	return int(uint16(decodeInuxi(arch, p[commonsize(arch)+2:], 2)) & (1<<15 - 1))
}

func decodetypeFuncInType(arch *sys.Arch, s *sym.Symbol, i int) *sym.Symbol {
	uadd := commonsize(arch) + 4
	if arch.PtrSize == 8 {
		uadd += 4
	}
	if decodetypeHasUncommon(arch, s.P) {
		uadd += uncommonSize()
	}
	return decodeRelocSym(s, int32(uadd+i*arch.PtrSize))
}

func decodetypeFuncOutType(arch *sys.Arch, s *sym.Symbol, i int) *sym.Symbol {
	return decodetypeFuncInType(arch, s, i+decodetypeFuncInCount(arch, s.P))
}

// Type.StructType.fields.Slice::length
func decodetypeStructFieldCount(arch *sys.Arch, s *sym.Symbol) int {
	return int(decodeInuxi(arch, s.P[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
}

func decodetypeStructFieldArrayOff(arch *sys.Arch, s *sym.Symbol, i int) int {
	off := commonsize(arch) + 4*arch.PtrSize
	if decodetypeHasUncommon(arch, s.P) {
		off += uncommonSize()
	}
	off += i * structfieldSize(arch)
	return off
}

// decodetypeStr returns the contents of an rtype's str field (a nameOff).
func decodetypeStr(arch *sys.Arch, s *sym.Symbol) string {
	str := decodetypeName(s, 4*arch.PtrSize+8)
	if s.P[2*arch.PtrSize+4]&tflagExtraStar != 0 {
		return str[1:]
	}
	return str
}

// decodetypeName decodes the name from a reflect.name.
func decodetypeName(s *sym.Symbol, off int) string {
	r := decodeReloc(s, int32(off))
	if r == nil {
		return ""
	}

	data := r.Sym.P
	namelen := int(uint16(data[1])<<8 | uint16(data[2]))
	return string(data[3 : 3+namelen])
}

func decodetypeStructFieldName(arch *sys.Arch, s *sym.Symbol, i int) string {
	off := decodetypeStructFieldArrayOff(arch, s, i)
	return decodetypeName(s, off)
}

func decodetypeStructFieldType(arch *sys.Arch, s *sym.Symbol, i int) *sym.Symbol {
	off := decodetypeStructFieldArrayOff(arch, s, i)
	return decodeRelocSym(s, int32(off+arch.PtrSize))
}

func decodetypeStructFieldOffs(arch *sys.Arch, s *sym.Symbol, i int) int64 {
	return decodetypeStructFieldOffsAnon(arch, s, i) >> 1
}

func decodetypeStructFieldOffsAnon(arch *sys.Arch, s *sym.Symbol, i int) int64 {
	off := decodetypeStructFieldArrayOff(arch, s, i)
	return int64(decodeInuxi(arch, s.P[off+2*arch.PtrSize:], arch.PtrSize))
}

// InterfaceType.methods.length
func decodetypeIfaceMethodCount(arch *sys.Arch, p []byte) int64 {
	return int64(decodeInuxi(arch, p[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
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
func decodeMethodSig(arch *sys.Arch, s *sym.Symbol, off, size, count int) []methodsig {
	var buf bytes.Buffer
	var methods []methodsig
	for i := 0; i < count; i++ {
		buf.WriteString(decodetypeName(s, off))
		mtypSym := decodeRelocSym(s, int32(off+4))

		buf.WriteRune('(')
		inCount := decodetypeFuncInCount(arch, mtypSym.P)
		for i := 0; i < inCount; i++ {
			if i > 0 {
				buf.WriteString(", ")
			}
			buf.WriteString(decodetypeFuncInType(arch, mtypSym, i).Name)
		}
		buf.WriteString(") (")
		outCount := decodetypeFuncOutCount(arch, mtypSym.P)
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

func decodeIfaceMethods(arch *sys.Arch, s *sym.Symbol) []methodsig {
	if decodetypeKind(arch, s.P)&kindMask != kindInterface {
		panic(fmt.Sprintf("symbol %q is not an interface", s.Name))
	}
	r := decodeReloc(s, int32(commonsize(arch)+arch.PtrSize))
	if r == nil {
		return nil
	}
	if r.Sym != s {
		panic(fmt.Sprintf("imethod slice pointer in %q leads to a different symbol", s.Name))
	}
	off := int(r.Add) // array of reflect.imethod values
	numMethods := int(decodetypeIfaceMethodCount(arch, s.P))
	sizeofIMethod := 4 + 4
	return decodeMethodSig(arch, s, off, sizeofIMethod, numMethods)
}

func decodetypeMethods(arch *sys.Arch, s *sym.Symbol) []methodsig {
	if !decodetypeHasUncommon(arch, s.P) {
		panic(fmt.Sprintf("no methods on %q", s.Name))
	}
	off := commonsize(arch) // reflect.rtype
	switch decodetypeKind(arch, s.P) & kindMask {
	case kindStruct: // reflect.structType
		off += 4 * arch.PtrSize
	case kindPtr: // reflect.ptrType
		off += arch.PtrSize
	case kindFunc: // reflect.funcType
		off += arch.PtrSize // 4 bytes, pointer aligned
	case kindSlice: // reflect.sliceType
		off += arch.PtrSize
	case kindArray: // reflect.arrayType
		off += 3 * arch.PtrSize
	case kindChan: // reflect.chanType
		off += 2 * arch.PtrSize
	case kindMap: // reflect.mapType
		off += 4*arch.PtrSize + 8
	case kindInterface: // reflect.interfaceType
		off += 3 * arch.PtrSize
	default:
		// just Sizeof(rtype)
	}

	mcount := int(decodeInuxi(arch, s.P[off+4:], 2))
	moff := int(decodeInuxi(arch, s.P[off+4+2+2:], 4))
	off += moff                // offset to array of reflect.method values
	const sizeofMethod = 4 * 4 // sizeof reflect.method in program
	return decodeMethodSig(arch, s, off, sizeofMethod, mcount)
}
