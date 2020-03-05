// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/sys"
	"cmd/link/internal/loader"
)

// This file contains utilities to decode type.* symbols, for
// loader.Sym symbols (uses new loader interfaces).

// At some point we'll want to migrate the contents of this file
// to decodesym.go once the rouetines there have been decprecated + removed.

func decodeReloc2(ldr *loader.Loader, symIdx loader.Sym, symRelocs []loader.Reloc, off int32) loader.Reloc {
	for j := 0; j < len(symRelocs); j++ {
		rel := symRelocs[j]
		if rel.Off == off {
			return rel
		}
	}
	return loader.Reloc{}
}

func decodeReloc3(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int32) loader.Reloc2 {
	for j := 0; j < relocs.Count; j++ {
		rel := relocs.At2(j)
		if rel.Off() == off {
			return rel
		}
	}
	return loader.Reloc2{}
}

func decodeRelocSym2(ldr *loader.Loader, symIdx loader.Sym, symRelocs []loader.Reloc, off int32) loader.Sym {
	return decodeReloc2(ldr, symIdx, symRelocs, off).Sym
}

func decodeRelocSym3(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int32) loader.Sym {
	return decodeReloc3(ldr, symIdx, relocs, off).Sym()
}

// decodetypeName2 decodes the name from a reflect.name.
func decodetypeName2(ldr *loader.Loader, symIdx loader.Sym, symRelocs []loader.Reloc, off int) string {
	r := decodeRelocSym2(ldr, symIdx, symRelocs, int32(off))
	if r == 0 {
		return ""
	}

	data := ldr.Data(r)
	namelen := int(uint16(data[1])<<8 | uint16(data[2]))
	return string(data[3 : 3+namelen])
}

func decodetypeName3(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int) string {
	r := decodeRelocSym3(ldr, symIdx, relocs, int32(off))
	if r == 0 {
		return ""
	}

	data := ldr.Data(r)
	namelen := int(uint16(data[1])<<8 | uint16(data[2]))
	return string(data[3 : 3+namelen])
}

func decodetypeFuncInType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, symRelocs []loader.Reloc, i int) loader.Sym {
	uadd := commonsize(arch) + 4
	if arch.PtrSize == 8 {
		uadd += 4
	}
	if decodetypeHasUncommon(arch, ldr.Data(symIdx)) {
		uadd += uncommonSize()
	}
	return decodeRelocSym2(ldr, symIdx, symRelocs, int32(uadd+i*arch.PtrSize))
}

func decodetypeFuncInType3(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs, i int) loader.Sym {
	uadd := commonsize(arch) + 4
	if arch.PtrSize == 8 {
		uadd += 4
	}
	if decodetypeHasUncommon(arch, ldr.Data(symIdx)) {
		uadd += uncommonSize()
	}
	return decodeRelocSym3(ldr, symIdx, relocs, int32(uadd+i*arch.PtrSize))
}

func decodetypeFuncOutType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, symRelocs []loader.Reloc, i int) loader.Sym {
	return decodetypeFuncInType2(ldr, arch, symIdx, symRelocs, i+decodetypeFuncInCount(arch, ldr.Data(symIdx)))
}

func decodetypeFuncOutType3(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs, i int) loader.Sym {
	return decodetypeFuncInType3(ldr, arch, symIdx, relocs, i+decodetypeFuncInCount(arch, ldr.Data(symIdx)))
}

func decodetypeArrayElem2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	// FIXME: it's inefficient to read the relocations each time. Add some
	// sort of cache here, or pass in the relocs. Alternatively we could
	// switch to relocs.At() to see if that performs better.
	relocs := ldr.Relocs(symIdx)
	rslice := relocs.ReadAll(nil)
	return decodeRelocSym2(ldr, symIdx, rslice, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeArrayLen2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) int64 {
	data := ldr.Data(symIdx)
	return int64(decodeInuxi(arch, data[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
}

func decodetypeChanElem2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	// FIXME: it's inefficient to read the relocations each time. Add some
	// sort of cache here, or pass in the relocs.
	relocs := ldr.Relocs(symIdx)
	rslice := relocs.ReadAll(nil)
	return decodeRelocSym2(ldr, symIdx, rslice, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeMapKey2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	// FIXME: it's inefficient to read the relocations each time. Add some
	// sort of cache here, or pass in the relocs. Alternatively we could
	// switch to relocs.At() to see if that performs better.
	relocs := ldr.Relocs(symIdx)
	rslice := relocs.ReadAll(nil)
	return decodeRelocSym2(ldr, symIdx, rslice, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeMapValue2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	// FIXME: it's inefficient to read the relocations each time. Add some
	// sort of cache here, or pass in the relocs. Alternatively we could
	// switch to relocs.At() to see if that performs better.
	relocs := ldr.Relocs(symIdx)
	rslice := relocs.ReadAll(nil)
	return decodeRelocSym2(ldr, symIdx, rslice, int32(commonsize(arch))+int32(arch.PtrSize)) // 0x20 / 0x38
}

func decodetypePtrElem2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	// FIXME: it's inefficient to read the relocations each time. Add some
	// sort of cache here, or pass in the relocs. Alternatively we could
	// switch to relocs.At() to see if that performs better.
	relocs := ldr.Relocs(symIdx)
	rslice := relocs.ReadAll(nil)
	return decodeRelocSym2(ldr, symIdx, rslice, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeStructFieldCount2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) int {
	data := ldr.Data(symIdx)
	return int(decodeInuxi(arch, data[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
}

func decodetypeStructFieldArrayOff2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) int {
	data := ldr.Data(symIdx)
	off := commonsize(arch) + 4*arch.PtrSize
	if decodetypeHasUncommon(arch, data) {
		off += uncommonSize()
	}
	off += i * structfieldSize(arch)
	return off
}

func decodetypeStructFieldName2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) string {
	off := decodetypeStructFieldArrayOff2(ldr, arch, symIdx, i)
	// FIXME: it's inefficient to read the relocations each time. Add some
	// sort of cache here, or pass in the relocs. Alternatively we could
	// switch to relocs.At() to see if that performs better.
	relocs := ldr.Relocs(symIdx)
	rslice := relocs.ReadAll(nil)
	return decodetypeName2(ldr, symIdx, rslice, off)
}

func decodetypeStructFieldType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) loader.Sym {
	off := decodetypeStructFieldArrayOff2(ldr, arch, symIdx, i)
	// FIXME: it's inefficient to read the relocations each time. Add some
	// sort of cache here, or pass in the relocs. Alternatively we could
	// switch to relocs.At() to see if that performs better.
	relocs := ldr.Relocs(symIdx)
	rslice := relocs.ReadAll(nil)
	return decodeRelocSym2(ldr, symIdx, rslice, int32(off+arch.PtrSize))
}

func decodetypeStructFieldOffsAnon2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) int64 {
	off := decodetypeStructFieldArrayOff2(ldr, arch, symIdx, i)
	data := ldr.Data(symIdx)
	return int64(decodeInuxi(arch, data[off+2*arch.PtrSize:], arch.PtrSize))
}
