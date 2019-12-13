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

func decodeRelocSym2(ldr *loader.Loader, symIdx loader.Sym, symRelocs []loader.Reloc, off int32) loader.Sym {
	return decodeReloc2(ldr, symIdx, symRelocs, off).Sym
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

func decodetypeFuncOutType2(ldr *loader.Loader, arch *sys.Arch, symIdx loader.Sym, symRelocs []loader.Reloc, i int) loader.Sym {
	return decodetypeFuncInType2(ldr, arch, symIdx, symRelocs, i+decodetypeFuncInCount(arch, ldr.Data(symIdx)))
}
