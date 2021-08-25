package ld

import (
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
)

func (ctxt *Link) efacetypes() {
	if !ctxt.efaceTypes {
		return
	}

	ldr := ctxt.loader
	var efacetypes []loader.Sym

	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if ldr.AttrReachable(s) && ldr.AttrUsedInEface(s) {
			efacetypes = append(efacetypes, s)
		}
	}

	et := ldr.CreateSymForUpdate(".debug_efacetypes", 0)
	et.SetType(sym.SDWARFSECT)
	et.SetReachable(true)
	et.SetSize(int64(ctxt.Arch.PtrSize * len(efacetypes)))
	et.Grow(et.Size())

	relocs := et.AddRelocs(len(efacetypes))

	for i, s := range efacetypes {
		r := relocs.At(i)
		r.SetSym(s)
		r.SetOff(int32(i * ctxt.Arch.PtrSize))
		r.SetSiz(uint8(ctxt.Arch.PtrSize))
		r.SetType(objabi.R_ADDR)
	}
}
