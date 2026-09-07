// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
)

// Init returns the architecture configuration for the wasm linker.
func Init(arch *sys.Arch) (*sys.Arch, ld.Arch) {
	if arch != sys.ArchWasm && arch != sys.ArchWasm32 {
		panic("invalid WebAssembly architecture")
	}
	st := new(wasm32State)
	theArch := ld.Arch{
		Funcalign: 16,
		Maxalign:  32,
		Minalign:  1,

		Archinit: archinit,
		AssignAddress: func(ldr *loader.Loader, sect *sym.Section, n int, s loader.Sym, va uint64, isTramp bool) (*sym.Section, int, uint64) {
			return st.assignAddress(arch, ldr, sect, n, s, va, isTramp)
		},
		Asmb:      asmb,
		Asmb2:     st.asmb2,
		Gentext:   st.gentext,
		Archreloc: st.archreloc,
		Extreloc:  extreloc,
	}

	return arch, theArch
}

func archinit(ctxt *ld.Link) {
	if *ld.FlagRound == -1 {
		*ld.FlagRound = 4096
	}
	if *ld.FlagTextAddr == -1 && ctxt.Arch.PtrSize == 8 {
		// See asm.go:assignAddress for the meaning of PC on Wasm
		// and the value here.
		*ld.FlagTextAddr = pcBase
	}
}
