// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loadpe

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"sort"
)

const (
	UNW_FLAG_EHANDLER  = 1 << 3
	UNW_FLAG_UHANDLER  = 2 << 3
	UNW_FLAG_CHAININFO = 4 << 3
	unwStaticDataSize  = 4 // Bytes of unwind data before the variable length part.
	unwCodeSize        = 2 // Bytes per unwind code.
)

// processSEH walks all pdata relocations looking for exception handler function symbols.
// We want to mark these as reachable if the function that they protect is reachable
// in the final binary.
func processSEH(ldr *loader.Loader, arch *sys.Arch, pdata sym.LoaderSym, xdata sym.LoaderSym) error {
	switch arch.Family {
	case sys.AMD64:
		ldr.SetAttrReachable(pdata, true)
		if xdata != 0 {
			ldr.SetAttrReachable(xdata, true)
		}
		return processSEHAMD64(ldr, pdata)
	default:
		// TODO: support SEH on other architectures.
		return fmt.Errorf("unsupported architecture for SEH: %v", arch.Family)
	}
}

func processSEHAMD64(ldr *loader.Loader, pdata sym.LoaderSym) error {
	// The following loop traverses a list of pdata entries,
	// each entry being 3 relocations long. The first relocation
	// is a pointer to the function symbol to which the pdata entry
	// corresponds. The third relocation is a pointer to the
	// corresponding .xdata entry.
	// Reference:
	// https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64#struct-runtime_function
	rels := ldr.Relocs(pdata)
	if rels.Count()%3 != 0 {
		return fmt.Errorf(".pdata symbol %q has invalid relocation count", ldr.SymName(pdata))
	}
	for i := 0; i < rels.Count(); i += 3 {
		xrel := rels.At(i + 2)
		handler := findHandlerInXDataAMD64(ldr, xrel.Sym(), xrel.Add())
		if handler != 0 {
			sb := ldr.MakeSymbolUpdater(rels.At(i).Sym())
			r, _ := sb.AddRel(objabi.R_KEEP)
			r.SetSym(handler)
		}
	}
	return nil
}

// findHandlerInXDataAMD64 finds the symbol in the .xdata section that
// corresponds to the exception handler.
// Reference:
// https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64#struct-unwind_info
func findHandlerInXDataAMD64(ldr *loader.Loader, xsym sym.LoaderSym, add int64) loader.Sym {
	data := ldr.Data(xsym)
	if add < 0 || add+unwStaticDataSize > int64(len(data)) {
		return 0
	}
	data = data[add:]
	var isChained bool
	switch flag := data[0]; {
	case flag&UNW_FLAG_EHANDLER != 0 || flag&UNW_FLAG_UHANDLER != 0:
		// Exception handler.
	case flag&UNW_FLAG_CHAININFO != 0:
		isChained = true
	default:
		// Nothing to do.
		return 0
	}
	codes := data[2]
	if codes%2 != 0 {
		// There are always an even number of unwind codes, even if the last one is unused.
		codes += 1
	}
	// The exception handler relocation is the first relocation after the unwind codes,
	// unless it is chained, but we will handle this case later.
	targetOff := add + unwStaticDataSize + unwCodeSize*int64(codes)
	xrels := ldr.Relocs(xsym)
	xrelsCount := xrels.Count()
	idx := sort.Search(xrelsCount, func(i int) bool {
		return int64(xrels.At(i).Off()) >= targetOff
	})
	if idx == xrelsCount {
		return 0
	}
	if isChained {
		// The third relocations references the next .xdata entry in the chain, recurse.
		idx += 2
		if idx >= xrelsCount {
			return 0
		}
		r := xrels.At(idx)
		return findHandlerInXDataAMD64(ldr, r.Sym(), r.Add())
	}
	return xrels.At(idx).Sym()
}
