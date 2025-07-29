// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"encoding/binary"
	"internal/abi"
	"log"
)

// Decoding the type.* symbols.	 This has to be in sync with
// ../../runtime/type.go, or more specifically, with what
// cmd/compile/internal/reflectdata/reflect.go stuffs in these.

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

func commonsize(arch *sys.Arch) int      { return abi.CommonSize(arch.PtrSize) }      // runtime._type
func structfieldSize(arch *sys.Arch) int { return abi.StructFieldSize(arch.PtrSize) } // runtime.structfield
func uncommonSize(arch *sys.Arch) int    { return int(abi.UncommonSize()) }           // runtime.uncommontype

// Type.commonType.kind
func decodetypeKind(arch *sys.Arch, p []byte) abi.Kind {
	return abi.Kind(p[2*arch.PtrSize+7]) //  0x13
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
	return abi.TFlag(p[abi.TFlagOff(arch.PtrSize)])&abi.TFlagUncommon != 0
}

// Type.commonType.tflag
func decodetypeGCMaskOnDemand(arch *sys.Arch, p []byte) bool {
	return abi.TFlag(p[abi.TFlagOff(arch.PtrSize)])&abi.TFlagGCMaskOnDemand != 0
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

// InterfaceType.methods.length
func decodetypeIfaceMethodCount(arch *sys.Arch, p []byte) int64 {
	return int64(decodeInuxi(arch, p[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
}

func decodeReloc(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int32) loader.Reloc {
	for j := 0; j < relocs.Count(); j++ {
		rel := relocs.At(j)
		if rel.Off() == off {
			return rel
		}
	}
	return loader.Reloc{}
}

func decodeRelocSym(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int32) loader.Sym {
	return decodeReloc(ldr, symIdx, relocs, off).Sym()
}

// decodetypeName decodes the name from a reflect.name.
func decodetypeName(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int) string {
	r := decodeRelocSym(ldr, symIdx, relocs, int32(off))
	if r == 0 {
		return ""
	}

	data := ldr.DataString(r)
	n := 1 + binary.MaxVarintLen64
	if len(data) < n {
		n = len(data)
	}
	nameLen, nameLenLen := binary.Uvarint([]byte(data[1:n]))
	return data[1+nameLenLen : 1+nameLenLen+int(nameLen)]
}

func decodetypeNameEmbedded(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int) bool {
	r := decodeRelocSym(ldr, symIdx, relocs, int32(off))
	if r == 0 {
		return false
	}
	data := ldr.Data(r)
	return data[0]&(1<<3) != 0
}

func decodetypeFuncInType(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs, i int) loader.Sym {
	uadd := commonsize(arch) + 4
	if arch.PtrSize == 8 {
		uadd += 4
	}
	if decodetypeHasUncommon(arch, ldr.Data(symIdx)) {
		uadd += uncommonSize(arch)
	}
	return decodeRelocSym(ldr, symIdx, relocs, int32(uadd+i*arch.PtrSize))
}

func decodetypeFuncOutType(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs, i int) loader.Sym {
	return decodetypeFuncInType(ldr, arch, symIdx, relocs, i+decodetypeFuncInCount(arch, ldr.Data(symIdx)))
}

func decodetypeArrayElem(ctxt *Link, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	return decodeTargetSym(ctxt, arch, symIdx, int64(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeArrayLen(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) int64 {
	data := ldr.Data(symIdx)
	return int64(decodeInuxi(arch, data[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
}

func decodetypeChanElem(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym(ldr, symIdx, &relocs, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeMapKey(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym(ldr, symIdx, &relocs, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeMapValue(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym(ldr, symIdx, &relocs, int32(commonsize(arch))+int32(arch.PtrSize)) // 0x20 / 0x38
}

func decodetypeMapGroup(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym(ldr, symIdx, &relocs, int32(commonsize(arch))+2*int32(arch.PtrSize)) // 0x24 / 0x40
}

func decodetypePtrElem(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym(ldr, symIdx, &relocs, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeStructFieldCount(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) int {
	data := ldr.Data(symIdx)
	return int(decodeInuxi(arch, data[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
}

func decodetypeStructFieldArrayOff(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) int {
	data := ldr.Data(symIdx)
	off := commonsize(arch) + 4*arch.PtrSize
	if decodetypeHasUncommon(arch, data) {
		off += uncommonSize(arch)
	}
	off += i * structfieldSize(arch)
	return off
}

func decodetypeStructFieldName(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) string {
	off := decodetypeStructFieldArrayOff(ldr, arch, symIdx, i)
	relocs := ldr.Relocs(symIdx)
	return decodetypeName(ldr, symIdx, &relocs, off)
}

func decodetypeStructFieldType(ctxt *Link, arch *sys.Arch, symIdx loader.Sym, i int) loader.Sym {
	ldr := ctxt.loader
	off := decodetypeStructFieldArrayOff(ldr, arch, symIdx, i)
	return decodeTargetSym(ctxt, arch, symIdx, int64(off+arch.PtrSize))
}

func decodetypeStructFieldOffset(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) int64 {
	off := decodetypeStructFieldArrayOff(ldr, arch, symIdx, i)
	data := ldr.Data(symIdx)
	return int64(decodeInuxi(arch, data[off+2*arch.PtrSize:], arch.PtrSize))
}

func decodetypeStructFieldEmbedded(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) bool {
	off := decodetypeStructFieldArrayOff(ldr, arch, symIdx, i)
	relocs := ldr.Relocs(symIdx)
	return decodetypeNameEmbedded(ldr, symIdx, &relocs, off)
}

// decodetypeStr returns the contents of an rtype's str field (a nameOff).
func decodetypeStr(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) string {
	relocs := ldr.Relocs(symIdx)
	str := decodetypeName(ldr, symIdx, &relocs, 4*arch.PtrSize+8)
	data := ldr.Data(symIdx)
	if data[abi.TFlagOff(arch.PtrSize)]&byte(abi.TFlagExtraStar) != 0 {
		return str[1:]
	}
	return str
}

func decodetypeGcmask(ctxt *Link, s loader.Sym) []byte {
	if ctxt.loader.SymType(s) == sym.SDYNIMPORT {
		symData := ctxt.loader.Data(s)
		addr := decodetypeGcprogShlib(ctxt, symData)
		ptrdata := decodetypePtrdata(ctxt.Arch, symData)
		sect := findShlibSection(ctxt, ctxt.loader.SymPkg(s), addr)
		if sect != nil {
			bits := ptrdata / int64(ctxt.Arch.PtrSize)
			r := make([]byte, (bits+7)/8)
			// ldshlibsyms avoids closing the ELF file so sect.ReadAt works.
			// If we remove this read (and the ones in decodetypeGcprog), we
			// can close the file.
			_, err := sect.ReadAt(r, int64(addr-sect.Addr))
			if err != nil {
				log.Fatal(err)
			}
			return r
		}
		Exitf("cannot find gcmask for %s", ctxt.loader.SymName(s))
		return nil
	}
	relocs := ctxt.loader.Relocs(s)
	mask := decodeRelocSym(ctxt.loader, s, &relocs, 2*int32(ctxt.Arch.PtrSize)+8+1*int32(ctxt.Arch.PtrSize))
	return ctxt.loader.Data(mask)
}

// Type.commonType.gc
func decodetypeGcprog(ctxt *Link, s loader.Sym) []byte {
	if ctxt.loader.SymType(s) == sym.SDYNIMPORT {
		symData := ctxt.loader.Data(s)
		addr := decodetypeGcprogShlib(ctxt, symData)
		sect := findShlibSection(ctxt, ctxt.loader.SymPkg(s), addr)
		if sect != nil {
			// A gcprog is a 4-byte uint32 indicating length, followed by
			// the actual program.
			progsize := make([]byte, 4)
			_, err := sect.ReadAt(progsize, int64(addr-sect.Addr))
			if err != nil {
				log.Fatal(err)
			}
			progbytes := make([]byte, ctxt.Arch.ByteOrder.Uint32(progsize))
			_, err = sect.ReadAt(progbytes, int64(addr-sect.Addr+4))
			if err != nil {
				log.Fatal(err)
			}
			return append(progsize, progbytes...)
		}
		Exitf("cannot find gcprog for %s", ctxt.loader.SymName(s))
		return nil
	}
	relocs := ctxt.loader.Relocs(s)
	rs := decodeRelocSym(ctxt.loader, s, &relocs, 2*int32(ctxt.Arch.PtrSize)+8+1*int32(ctxt.Arch.PtrSize))
	return ctxt.loader.Data(rs)
}

// Find the elf.Section of a given shared library that contains a given address.
func findShlibSection(ctxt *Link, path string, addr uint64) *elf.Section {
	for _, shlib := range ctxt.Shlibs {
		if shlib.Path == path {
			for _, sect := range shlib.File.Sections[1:] { // skip the NULL section
				if sect.Addr <= addr && addr < sect.Addr+sect.Size {
					return sect
				}
			}
		}
	}
	return nil
}

func decodetypeGcprogShlib(ctxt *Link, data []byte) uint64 {
	return decodeInuxi(ctxt.Arch, data[2*int32(ctxt.Arch.PtrSize)+8+1*int32(ctxt.Arch.PtrSize):], ctxt.Arch.PtrSize)
}

// decodeItabType returns the itab.Type field from an itab.
func decodeItabType(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym(ldr, symIdx, &relocs, int32(abi.ITabTypeOff(arch.PtrSize)))
}

// decodeTargetSym finds the symbol pointed to by the pointer slot at offset off in s.
func decodeTargetSym(ctxt *Link, arch *sys.Arch, s loader.Sym, off int64) loader.Sym {
	ldr := ctxt.loader
	if ldr.SymType(s) == sym.SDYNIMPORT {
		// In this case, relocations are not associated with a
		// particular symbol. Instead, they are all listed together
		// in the containing shared library. Find the relocation
		// in that shared library record.
		name := ldr.SymName(s)
		for _, sh := range ctxt.Shlibs {
			addr, ok := sh.symAddr[name]
			if !ok {
				continue
			}
			addr += uint64(off)
			target := sh.relocTarget[addr]
			if target == "" {
				Exitf("can't find relocation in %s at offset %d", name, off)
			}
			t := ldr.Lookup(target, 0)
			if t == 0 {
				Exitf("can't find target of relocation in %s at offset %d: %s", name, off, target)
			}
			return t
		}
	}

	// For the normal case, just find the relocation within the symbol that
	// lives at the requested offset.
	relocs := ldr.Relocs(s)
	return decodeRelocSym(ldr, s, &relocs, int32(off))
}
