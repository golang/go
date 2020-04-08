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

func decodeReloc2(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int32) loader.Reloc2 {
	for j := 0; j < relocs.Count(); j++ {
		rel := relocs.At2(j)
		if rel.Off() == off {
			return rel
		}
	}
	return loader.Reloc2{}
}

func decodeRelocSym2(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int32) loader.Sym {
	return decodeReloc2(ldr, symIdx, relocs, off).Sym()
}

// decodetypeName2 decodes the name from a reflect.name.
func decodetypeName2(ldr *loader.Loader, symIdx loader.Sym, relocs *loader.Relocs, off int) string {
	r := decodeRelocSym2(ldr, symIdx, relocs, int32(off))
	if r == 0 {
		return ""
	}

	data := ldr.Data(r)
	namelen := int(uint16(data[1])<<8 | uint16(data[2]))
	return string(data[3 : 3+namelen])
}

func decodetypeFuncInType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs, i int) loader.Sym {
	uadd := commonsize(arch) + 4
	if arch.PtrSize == 8 {
		uadd += 4
	}
	if decodetypeHasUncommon(arch, ldr.Data(symIdx)) {
		uadd += uncommonSize()
	}
	return decodeRelocSym2(ldr, symIdx, relocs, int32(uadd+i*arch.PtrSize))
}

func decodetypeFuncOutType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, relocs *loader.Relocs, i int) loader.Sym {
	return decodetypeFuncInType2(ldr, arch, symIdx, relocs, i+decodetypeFuncInCount(arch, ldr.Data(symIdx)))
}

func decodetypeArrayElem2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym2(ldr, symIdx, &relocs, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeArrayLen2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) int64 {
	data := ldr.Data(symIdx)
	return int64(decodeInuxi(arch, data[commonsize(arch)+2*arch.PtrSize:], arch.PtrSize))
}

func decodetypeChanElem2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym2(ldr, symIdx, &relocs, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeMapKey2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym2(ldr, symIdx, &relocs, int32(commonsize(arch))) // 0x1c / 0x30
}

func decodetypeMapValue2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym2(ldr, symIdx, &relocs, int32(commonsize(arch))+int32(arch.PtrSize)) // 0x20 / 0x38
}

func decodetypePtrElem2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) loader.Sym {
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym2(ldr, symIdx, &relocs, int32(commonsize(arch))) // 0x1c / 0x30
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
	relocs := ldr.Relocs(symIdx)
	return decodetypeName2(ldr, symIdx, &relocs, off)
}

func decodetypeStructFieldType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) loader.Sym {
	off := decodetypeStructFieldArrayOff2(ldr, arch, symIdx, i)
	relocs := ldr.Relocs(symIdx)
	return decodeRelocSym2(ldr, symIdx, &relocs, int32(off+arch.PtrSize))
}

func decodetypeStructFieldOffsAnon2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, i int) int64 {
	off := decodetypeStructFieldArrayOff2(ldr, arch, symIdx, i)
	data := ldr.Data(symIdx)
	return int64(decodeInuxi(arch, data[off+2*arch.PtrSize:], arch.PtrSize))
}

// decodetypeStr2 returns the contents of an rtype's str field (a nameOff).
func decodetypeStr2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym) string {
	relocs := ldr.Relocs(symIdx)
	str := decodetypeName2(ldr, symIdx, &relocs, 4*arch.PtrSize+8)
	data := ldr.Data(symIdx)
	if data[2*arch.PtrSize+4]&tflagExtraStar != 0 {
		return str[1:]
	}
	return str
}
