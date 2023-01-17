// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
)

var sehp struct {
	pdata loader.Sym
}

func writeSEH(ctxt *Link) {
	switch ctxt.Arch.Family {
	case sys.AMD64:
		writeSEHAMD64(ctxt)
	}
}

func writeSEHAMD64(ctxt *Link) {
	ldr := ctxt.loader
	mkSecSym := func(name string, kind sym.SymKind) *loader.SymbolBuilder {
		s := ldr.CreateSymForUpdate(name, 0)
		s.SetType(kind)
		s.SetAlign(4)
		return s
	}
	pdata := mkSecSym(".pdata", sym.SPDATASECT)
	// TODO: the following 12 bytes represent a dummy unwind info,
	// remove once unwind infos are encoded in the .xdata section.
	pdata.AddUint64(ctxt.Arch, 0)
	pdata.AddUint32(ctxt.Arch, 0)
	for _, s := range ctxt.Textp {
		if fi := ldr.FuncInfo(s); !fi.Valid() || fi.TopFrame() {
			continue
		}
		uw := ldr.SEHUnwindSym(s)
		if uw == 0 {
			continue
		}

		// Reference:
		// https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64#struct-runtime_function
		pdata.AddPEImageRelativeAddrPlus(ctxt.Arch, s, 0)
		pdata.AddPEImageRelativeAddrPlus(ctxt.Arch, s, ldr.SymSize(s))
		// TODO: reference the .xdata symbol.
		pdata.AddPEImageRelativeAddrPlus(ctxt.Arch, pdata.Sym(), 0)
	}
	sehp.pdata = pdata.Sym()
}
