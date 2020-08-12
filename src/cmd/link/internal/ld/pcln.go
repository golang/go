// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// oldPclnState holds state information used during pclntab generation.  Here
// 'ldr' is just a pointer to the context's loader, 'deferReturnSym' is the
// index for the symbol "runtime.deferreturn",
//
// NB: This is deprecated, and will be eliminated when pclntab_old is
// eliminated.
type oldPclnState struct {
	ldr            *loader.Loader
	deferReturnSym loader.Sym
}

// pclntab holds the state needed for pclntab generation.
type pclntab struct {
	// The first and last functions found.
	firstFunc, lastFunc loader.Sym

	// Running total size of pclntab.
	size int64

	// runtime.pclntab's symbols
	carrier     loader.Sym
	pclntab     loader.Sym
	pcheader    loader.Sym
	funcnametab loader.Sym
	findfunctab loader.Sym
	cutab       loader.Sym
	filetab     loader.Sym
	pctab       loader.Sym

	// The number of functions + number of TEXT sections - 1. This is such an
	// unexpected value because platforms that have more than one TEXT section
	// get a dummy function inserted between because the external linker can place
	// functions in those areas. We mark those areas as not covered by the Go
	// runtime.
	//
	// On most platforms this is the number of reachable functions.
	nfunc int32

	// The number of filenames in runtime.filetab.
	nfiles uint32

	// maps the function symbol to offset in runtime.funcnametab
	// This doesn't need to reside in the state once pclntab_old's been
	// deleted -- it can live in generateFuncnametab.
	// TODO(jfaller): Delete me!
	funcNameOffset map[loader.Sym]int32
}

// addGeneratedSym adds a generator symbol to pclntab, returning the new Sym.
// It is the caller's responsibilty to save they symbol in state.
func (state *pclntab) addGeneratedSym(ctxt *Link, name string, size int64, f generatorFunc) loader.Sym {
	size = Rnd(size, int64(ctxt.Arch.PtrSize))
	state.size += size
	s := ctxt.createGeneratorSymbol(name, 0, sym.SPCLNTAB, size, f)
	ctxt.loader.SetAttrReachable(s, true)
	ctxt.loader.SetCarrierSym(s, state.carrier)
	ctxt.loader.SetAttrNotInSymbolTable(s, true)
	return s
}

func makeOldPclnState(ctxt *Link) *oldPclnState {
	ldr := ctxt.loader
	drs := ldr.Lookup("runtime.deferreturn", sym.SymVerABIInternal)
	state := &oldPclnState{
		ldr:            ldr,
		deferReturnSym: drs,
	}

	return state
}

// makePclntab makes a pclntab object, and assembles all the compilation units
// we'll need to write pclntab.
func makePclntab(ctxt *Link, container loader.Bitmap) (*pclntab, []*sym.CompilationUnit) {
	ldr := ctxt.loader

	state := &pclntab{
		funcNameOffset: make(map[loader.Sym]int32),
	}

	// Gather some basic stats and info.
	seenCUs := make(map[*sym.CompilationUnit]struct{})
	prevSect := ldr.SymSect(ctxt.Textp[0])
	compUnits := []*sym.CompilationUnit{}

	for _, s := range ctxt.Textp {
		if !emitPcln(ctxt, s, container) {
			continue
		}
		state.nfunc++
		if state.firstFunc == 0 {
			state.firstFunc = s
		}
		state.lastFunc = s
		ss := ldr.SymSect(s)
		if ss != prevSect {
			// With multiple text sections, the external linker may
			// insert functions between the sections, which are not
			// known by Go. This leaves holes in the PC range covered
			// by the func table. We need to generate an entry to mark
			// the hole.
			state.nfunc++
			prevSect = ss
		}

		// We need to keep track of all compilation units we see. Some symbols
		// (eg, go.buildid, _cgoexp_, etc) won't have a compilation unit.
		cu := ldr.SymUnit(s)
		if _, ok := seenCUs[cu]; cu != nil && !ok {
			seenCUs[cu] = struct{}{}
			cu.PclnIndex = len(compUnits)
			compUnits = append(compUnits, cu)
		}
	}
	return state, compUnits
}

func ftabaddstring(ftab *loader.SymbolBuilder, s string) int32 {
	start := len(ftab.Data())
	ftab.Grow(int64(start + len(s) + 1)) // make room for s plus trailing NUL
	ftd := ftab.Data()
	copy(ftd[start:], s)
	return int32(start)
}

// onlycsymbol looks at a symbol's name to report whether this is a
// symbol that is referenced by C code
func onlycsymbol(sname string) bool {
	switch sname {
	case "_cgo_topofstack", "__cgo_topofstack", "_cgo_panic", "crosscall2":
		return true
	}
	if strings.HasPrefix(sname, "_cgoexp_") {
		return true
	}
	return false
}

func emitPcln(ctxt *Link, s loader.Sym, container loader.Bitmap) bool {
	if ctxt.BuildMode == BuildModePlugin && ctxt.HeadType == objabi.Hdarwin && onlycsymbol(ctxt.loader.SymName(s)) {
		return false
	}
	// We want to generate func table entries only for the "lowest
	// level" symbols, not containers of subsymbols.
	return !container.Has(s)
}

func (state *oldPclnState) computeDeferReturn(target *Target, s loader.Sym) uint32 {
	deferreturn := uint32(0)
	lastWasmAddr := uint32(0)

	relocs := state.ldr.Relocs(s)
	for ri := 0; ri < relocs.Count(); ri++ {
		r := relocs.At(ri)
		if target.IsWasm() && r.Type() == objabi.R_ADDR {
			// Wasm does not have a live variable set at the deferreturn
			// call itself. Instead it has one identified by the
			// resumption point immediately preceding the deferreturn.
			// The wasm code has a R_ADDR relocation which is used to
			// set the resumption point to PC_B.
			lastWasmAddr = uint32(r.Add())
		}
		if r.Type().IsDirectCall() && (r.Sym() == state.deferReturnSym || state.ldr.IsDeferReturnTramp(r.Sym())) {
			if target.IsWasm() {
				deferreturn = lastWasmAddr - 1
			} else {
				// Note: the relocation target is in the call instruction, but
				// is not necessarily the whole instruction (for instance, on
				// x86 the relocation applies to bytes [1:5] of the 5 byte call
				// instruction).
				deferreturn = uint32(r.Off())
				switch target.Arch.Family {
				case sys.AMD64, sys.I386:
					deferreturn--
				case sys.PPC64, sys.ARM, sys.ARM64, sys.MIPS, sys.MIPS64:
					// no change
				case sys.RISCV64:
					// TODO(jsing): The JALR instruction is marked with
					// R_CALLRISCV, whereas the actual reloc is currently
					// one instruction earlier starting with the AUIPC.
					deferreturn -= 4
				case sys.S390X:
					deferreturn -= 2
				default:
					panic(fmt.Sprint("Unhandled architecture:", target.Arch.Family))
				}
			}
			break // only need one
		}
	}
	return deferreturn
}

// genInlTreeSym generates the InlTree sym for a function with the
// specified FuncInfo.
func (state *oldPclnState) genInlTreeSym(cu *sym.CompilationUnit, fi loader.FuncInfo, arch *sys.Arch, newState *pclntab) loader.Sym {
	ldr := state.ldr
	its := ldr.CreateExtSym("", 0)
	inlTreeSym := ldr.MakeSymbolUpdater(its)
	// Note: the generated symbol is given a type of sym.SGOFUNC, as a
	// signal to the symtab() phase that it needs to be grouped in with
	// other similar symbols (gcdata, etc); the dodata() phase will
	// eventually switch the type back to SRODATA.
	inlTreeSym.SetType(sym.SGOFUNC)
	ldr.SetAttrReachable(its, true)
	ninl := fi.NumInlTree()
	for i := 0; i < int(ninl); i++ {
		call := fi.InlTree(i)
		val := call.File
		nameoff, ok := newState.funcNameOffset[call.Func]
		if !ok {
			panic("couldn't find function name offset")
		}

		inlTreeSym.SetUint16(arch, int64(i*20+0), uint16(call.Parent))
		inlFunc := ldr.FuncInfo(call.Func)

		var funcID objabi.FuncID
		if inlFunc.Valid() {
			funcID = inlFunc.FuncID()
		}
		inlTreeSym.SetUint8(arch, int64(i*20+2), uint8(funcID))

		// byte 3 is unused
		inlTreeSym.SetUint32(arch, int64(i*20+4), uint32(val))
		inlTreeSym.SetUint32(arch, int64(i*20+8), uint32(call.Line))
		inlTreeSym.SetUint32(arch, int64(i*20+12), uint32(nameoff))
		inlTreeSym.SetUint32(arch, int64(i*20+16), uint32(call.ParentPC))
	}
	return its
}

// generatePCHeader creates the runtime.pcheader symbol, setting it up as a
// generator to fill in its data later.
func (state *pclntab) generatePCHeader(ctxt *Link) {
	writeHeader := func(ctxt *Link, s loader.Sym) {
		ldr := ctxt.loader
		header := ctxt.loader.MakeSymbolUpdater(s)

		writeSymOffset := func(off int64, ws loader.Sym) int64 {
			diff := ldr.SymValue(ws) - ldr.SymValue(s)
			if diff <= 0 {
				name := ldr.SymName(ws)
				panic(fmt.Sprintf("expected runtime.pcheader(%x) to be placed before %s(%x)", ldr.SymValue(s), name, ldr.SymValue(ws)))
			}
			return header.SetUintptr(ctxt.Arch, off, uintptr(diff))
		}

		// Write header.
		// Keep in sync with runtime/symtab.go:pcHeader.
		header.SetUint32(ctxt.Arch, 0, 0xfffffffa)
		header.SetUint8(ctxt.Arch, 6, uint8(ctxt.Arch.MinLC))
		header.SetUint8(ctxt.Arch, 7, uint8(ctxt.Arch.PtrSize))
		off := header.SetUint(ctxt.Arch, 8, uint64(state.nfunc))
		off = header.SetUint(ctxt.Arch, off, uint64(state.nfiles))
		off = writeSymOffset(off, state.funcnametab)
		off = writeSymOffset(off, state.cutab)
		off = writeSymOffset(off, state.filetab)
		off = writeSymOffset(off, state.pctab)
		off = writeSymOffset(off, state.pclntab)
	}

	size := int64(8 + 7*ctxt.Arch.PtrSize)
	state.pcheader = state.addGeneratedSym(ctxt, "runtime.pcheader", size, writeHeader)
}

// walkFuncs iterates over the Textp, calling a function for each unique
// function and inlined function.
func (state *pclntab) walkFuncs(ctxt *Link, container loader.Bitmap, f func(loader.Sym)) {
	ldr := ctxt.loader
	seen := make(map[loader.Sym]struct{})
	for _, ls := range ctxt.Textp {
		s := loader.Sym(ls)
		if !emitPcln(ctxt, s, container) {
			continue
		}
		if _, ok := seen[s]; !ok {
			f(s)
			seen[s] = struct{}{}
		}

		fi := ldr.FuncInfo(s)
		if !fi.Valid() {
			continue
		}
		fi.Preload()
		for i, ni := 0, fi.NumInlTree(); i < int(ni); i++ {
			call := fi.InlTree(i).Func
			if _, ok := seen[call]; !ok {
				f(call)
				seen[call] = struct{}{}
			}
		}
	}
}

// generateFuncnametab creates the function name table.
func (state *pclntab) generateFuncnametab(ctxt *Link, container loader.Bitmap) {
	// Write the null terminated strings.
	writeFuncNameTab := func(ctxt *Link, s loader.Sym) {
		symtab := ctxt.loader.MakeSymbolUpdater(s)
		for s, off := range state.funcNameOffset {
			symtab.AddStringAt(int64(off), ctxt.loader.SymName(s))
		}
	}

	// Loop through the CUs, and calculate the size needed.
	var size int64
	state.walkFuncs(ctxt, container, func(s loader.Sym) {
		state.funcNameOffset[s] = int32(size)
		size += int64(ctxt.loader.SymNameLen(s)) + 1 // NULL terminate
	})

	state.funcnametab = state.addGeneratedSym(ctxt, "runtime.funcnametab", size, writeFuncNameTab)
}

// walkFilenames walks the filenames in the all reachable functions.
func walkFilenames(ctxt *Link, container loader.Bitmap, f func(*sym.CompilationUnit, goobj.CUFileIndex)) {
	ldr := ctxt.loader

	// Loop through all functions, finding the filenames we need.
	for _, ls := range ctxt.Textp {
		s := loader.Sym(ls)
		if !emitPcln(ctxt, s, container) {
			continue
		}

		fi := ldr.FuncInfo(s)
		if !fi.Valid() {
			continue
		}
		fi.Preload()

		cu := ldr.SymUnit(s)
		for i, nf := 0, int(fi.NumFile()); i < nf; i++ {
			f(cu, fi.File(i))
		}
		for i, ninl := 0, int(fi.NumInlTree()); i < ninl; i++ {
			call := fi.InlTree(i)
			f(cu, call.File)
		}
	}
}

// generateFilenameTabs creates LUTs needed for filename lookup. Returns a slice
// of the index at which each CU begins in runtime.cutab.
//
// Function objects keep track of the files they reference to print the stack.
// This function creates a per-CU list of filenames if CU[M] references
// files[1-N], the following is generated:
//
//  runtime.cutab:
//    CU[M]
//     offsetToFilename[0]
//     offsetToFilename[1]
//     ..
//
//  runtime.filetab
//     filename[0]
//     filename[1]
//
// Looking up a filename then becomes:
//  0) Given a func, and filename index [K]
//  1) Get Func.CUIndex:       M := func.cuOffset
//  2) Find filename offset:   fileOffset := runtime.cutab[M+K]
//  3) Get the filename:       getcstring(runtime.filetab[fileOffset])
func (state *pclntab) generateFilenameTabs(ctxt *Link, compUnits []*sym.CompilationUnit, container loader.Bitmap) []uint32 {
	// On a per-CU basis, keep track of all the filenames we need.
	//
	// Note, that we store the filenames in a separate section in the object
	// files, and deduplicate based on the actual value. It would be better to
	// store the filenames as symbols, using content addressable symbols (and
	// then not loading extra filenames), and just use the hash value of the
	// symbol name to do this cataloging.
	//
	// TOOD: Store filenames as symbols. (Note this would be easiest if you
	// also move strings to ALWAYS using the larger content addressable hash
	// function, and use that hash value for uniqueness testing.)
	cuEntries := make([]goobj.CUFileIndex, len(compUnits))
	fileOffsets := make(map[string]uint32)

	// Walk the filenames.
	// We store the total filename string length we need to load, and the max
	// file index we've seen per CU so we can calculate how large the
	// CU->global table needs to be.
	var fileSize int64
	walkFilenames(ctxt, container, func(cu *sym.CompilationUnit, i goobj.CUFileIndex) {
		// Note we use the raw filename for lookup, but use the expanded filename
		// when we save the size.
		filename := cu.FileTable[i]
		if _, ok := fileOffsets[filename]; !ok {
			fileOffsets[filename] = uint32(fileSize)
			fileSize += int64(len(expandFile(filename)) + 1) // NULL terminate
		}

		// Find the maximum file index we've seen.
		if cuEntries[cu.PclnIndex] < i+1 {
			cuEntries[cu.PclnIndex] = i + 1 // Store max + 1
		}
	})

	// Calculate the size of the runtime.cutab variable.
	var totalEntries uint32
	cuOffsets := make([]uint32, len(cuEntries))
	for i, entries := range cuEntries {
		// Note, cutab is a slice of uint32, so an offset to a cu's entry is just the
		// running total of all cu indices we've needed to store so far, not the
		// number of bytes we've stored so far.
		cuOffsets[i] = totalEntries
		totalEntries += uint32(entries)
	}

	// Write cutab.
	writeCutab := func(ctxt *Link, s loader.Sym) {
		sb := ctxt.loader.MakeSymbolUpdater(s)

		var off int64
		for i, max := range cuEntries {
			// Write the per CU LUT.
			cu := compUnits[i]
			for j := goobj.CUFileIndex(0); j < max; j++ {
				fileOffset, ok := fileOffsets[cu.FileTable[j]]
				if !ok {
					// We're looping through all possible file indices. It's possible a file's
					// been deadcode eliminated, and although it's a valid file in the CU, it's
					// not needed in this binary. When that happens, use an invalid offset.
					fileOffset = ^uint32(0)
				}
				off = sb.SetUint32(ctxt.Arch, off, fileOffset)
			}
		}
	}
	state.cutab = state.addGeneratedSym(ctxt, "runtime.cutab", int64(totalEntries*4), writeCutab)

	// Write filetab.
	writeFiletab := func(ctxt *Link, s loader.Sym) {
		sb := ctxt.loader.MakeSymbolUpdater(s)

		// Write the strings.
		for filename, loc := range fileOffsets {
			sb.AddStringAt(int64(loc), expandFile(filename))
		}
	}
	state.nfiles = uint32(len(fileOffsets))
	state.filetab = state.addGeneratedSym(ctxt, "runtime.filetab", fileSize, writeFiletab)

	return cuOffsets
}

// generatePctab creates the runtime.pctab variable, holding all the
// deduplicated pcdata.
func (state *pclntab) generatePctab(ctxt *Link, container loader.Bitmap) {
	ldr := ctxt.loader

	// Pctab offsets of 0 are considered invalid in the runtime. We respect
	// that by just padding a single byte at the beginning of runtime.pctab,
	// that way no real offsets can be zero.
	size := int64(1)

	// Walk the functions, finding offset to store each pcdata.
	seen := make(map[loader.Sym]struct{})
	saveOffset := func(pcSym loader.Sym) {
		if _, ok := seen[pcSym]; !ok {
			datSize := ldr.SymSize(pcSym)
			if datSize != 0 {
				ldr.SetSymValue(pcSym, size)
			} else {
				// Invalid PC data, record as zero.
				ldr.SetSymValue(pcSym, 0)
			}
			size += datSize
			seen[pcSym] = struct{}{}
		}
	}
	for _, s := range ctxt.Textp {
		if !emitPcln(ctxt, s, container) {
			continue
		}
		fi := ldr.FuncInfo(s)
		if !fi.Valid() {
			continue
		}
		fi.Preload()

		pcSyms := []loader.Sym{fi.Pcsp(), fi.Pcfile(), fi.Pcline()}
		for _, pcSym := range pcSyms {
			saveOffset(pcSym)
		}
		for _, pcSym := range fi.Pcdata() {
			saveOffset(pcSym)
		}
		if fi.NumInlTree() > 0 {
			saveOffset(fi.Pcinline())
		}
	}

	// TODO: There is no reason we need a generator for this variable, and it
	// could be moved to a carrier symbol. However, carrier symbols containing
	// carrier symbols don't work yet (as of Aug 2020). Once this is fixed,
	// runtime.pctab could just be a carrier sym.
	writePctab := func(ctxt *Link, s loader.Sym) {
		ldr := ctxt.loader
		sb := ldr.MakeSymbolUpdater(s)
		for sym := range seen {
			sb.SetBytesAt(ldr.SymValue(sym), ldr.Data(sym))
		}
	}

	state.pctab = state.addGeneratedSym(ctxt, "runtime.pctab", size, writePctab)
}

// pclntab initializes the pclntab symbol with
// runtime function and file name information.

// pclntab generates the pcln table for the link output.
func (ctxt *Link) pclntab(container loader.Bitmap) *pclntab {
	// Go 1.2's symtab layout is documented in golang.org/s/go12symtab, but the
	// layout and data has changed since that time.
	//
	// As of August 2020, here's the layout of pclntab:
	//
	//  .gopclntab/__gopclntab [elf/macho section]
	//    runtime.pclntab
	//      Carrier symbol for the entire pclntab section.
	//
	//      runtime.pcheader  (see: runtime/symtab.go:pcHeader)
	//        8-byte magic
	//        nfunc [thearch.ptrsize bytes]
	//        offset to runtime.funcnametab from the beginning of runtime.pcheader
	//        offset to runtime.pclntab_old from beginning of runtime.pcheader
	//
	//      runtime.funcnametab
	//        []list of null terminated function names
	//
	//      runtime.cutab
	//        for i=0..#CUs
	//          for j=0..#max used file index in CU[i]
	//            uint32 offset into runtime.filetab for the filename[j]
	//
	//      runtime.filetab
	//        []null terminated filename strings
	//
	//      runtime.pctab
	//        []byte of deduplicated pc data.
	//
	//      runtime.pclntab_old
	//        function table, alternating PC and offset to func struct [each entry thearch.ptrsize bytes]
	//        end PC [thearch.ptrsize bytes]
	//        func structures, pcdata tables.

	oldState := makeOldPclnState(ctxt)
	state, compUnits := makePclntab(ctxt, container)

	ldr := ctxt.loader
	state.carrier = ldr.LookupOrCreateSym("runtime.pclntab", 0)
	ldr.MakeSymbolUpdater(state.carrier).SetType(sym.SPCLNTAB)
	ldr.SetAttrReachable(state.carrier, true)

	// runtime.pclntab_old is just a placeholder,and will eventually be deleted.
	// It contains the pieces of runtime.pclntab that haven't moved to a more
	// rational form.
	state.pclntab = ldr.LookupOrCreateSym("runtime.pclntab_old", 0)
	state.generatePCHeader(ctxt)
	state.generateFuncnametab(ctxt, container)
	cuOffsets := state.generateFilenameTabs(ctxt, compUnits, container)
	state.generatePctab(ctxt, container)

	funcdataBytes := int64(0)
	ldr.SetCarrierSym(state.pclntab, state.carrier)
	ldr.SetAttrNotInSymbolTable(state.pclntab, true)
	ftab := ldr.MakeSymbolUpdater(state.pclntab)
	ftab.SetValue(state.size)
	ftab.SetType(sym.SPCLNTAB)
	ftab.SetReachable(true)

	ftab.Grow(int64(state.nfunc)*2*int64(ctxt.Arch.PtrSize) + int64(ctxt.Arch.PtrSize) + 4)

	setAddr := (*loader.SymbolBuilder).SetAddrPlus
	if ctxt.IsExe() && ctxt.IsInternal() {
		// Internal linking static executable. At this point the function
		// addresses are known, so we can just use them instead of emitting
		// relocations.
		// For other cases we are generating a relocatable binary so we
		// still need to emit relocations.
		setAddr = func(s *loader.SymbolBuilder, arch *sys.Arch, off int64, tgt loader.Sym, add int64) int64 {
			if v := ldr.SymValue(tgt); v != 0 {
				return s.SetUint(arch, off, uint64(v+add))
			}
			return s.SetAddrPlus(arch, off, tgt, add)
		}
	}

	funcdata := []loader.Sym{}
	funcdataoff := []int64{}

	var nfunc int32
	prevFunc := ctxt.Textp[0]
	for _, s := range ctxt.Textp {
		if !emitPcln(ctxt, s, container) {
			continue
		}

		thisSect := ldr.SymSect(s)
		prevSect := ldr.SymSect(prevFunc)
		if thisSect != prevSect {
			// With multiple text sections, there may be a hole here
			// in the address space (see the comment above). We use an
			// invalid funcoff value to mark the hole. See also
			// runtime/symtab.go:findfunc
			prevFuncSize := int64(ldr.SymSize(prevFunc))
			setAddr(ftab, ctxt.Arch, int64(nfunc)*2*int64(ctxt.Arch.PtrSize), prevFunc, prevFuncSize)
			ftab.SetUint(ctxt.Arch, int64(nfunc)*2*int64(ctxt.Arch.PtrSize)+int64(ctxt.Arch.PtrSize), ^uint64(0))
			nfunc++
		}
		prevFunc = s

		var numPCData int32
		funcdataoff = funcdataoff[:0]
		funcdata = funcdata[:0]
		fi := ldr.FuncInfo(s)
		if fi.Valid() {
			fi.Preload()
			numPCData = int32(len(fi.Pcdata()))
			nfd := fi.NumFuncdataoff()
			for i := uint32(0); i < nfd; i++ {
				funcdataoff = append(funcdataoff, fi.Funcdataoff(int(i)))
			}
			funcdata = fi.Funcdata(funcdata)
		}

		writeInlPCData := false
		if fi.Valid() && fi.NumInlTree() > 0 {
			writeInlPCData = true
			if numPCData <= objabi.PCDATA_InlTreeIndex {
				numPCData = objabi.PCDATA_InlTreeIndex + 1
			}
			if len(funcdataoff) <= objabi.FUNCDATA_InlTree {
				// Create inline tree funcdata.
				newfuncdata := make([]loader.Sym, objabi.FUNCDATA_InlTree+1)
				newfuncdataoff := make([]int64, objabi.FUNCDATA_InlTree+1)
				copy(newfuncdata, funcdata)
				copy(newfuncdataoff, funcdataoff)
				funcdata = newfuncdata
				funcdataoff = newfuncdataoff
			}
		}

		dSize := len(ftab.Data())
		funcstart := int32(dSize)
		funcstart += int32(-dSize) & (int32(ctxt.Arch.PtrSize) - 1) // align to ptrsize

		setAddr(ftab, ctxt.Arch, int64(nfunc)*2*int64(ctxt.Arch.PtrSize), s, 0)
		ftab.SetUint(ctxt.Arch, int64(nfunc)*2*int64(ctxt.Arch.PtrSize)+int64(ctxt.Arch.PtrSize), uint64(funcstart))

		// Write runtime._func. Keep in sync with ../../../../runtime/runtime2.go:/_func
		// and package debug/gosym.

		// fixed size of struct, checked below
		off := funcstart

		end := funcstart + int32(ctxt.Arch.PtrSize) + 3*4 + 6*4 + numPCData*4 + int32(len(funcdata))*int32(ctxt.Arch.PtrSize)
		if len(funcdata) > 0 && (end&int32(ctxt.Arch.PtrSize-1) != 0) {
			end += 4
		}
		ftab.Grow(int64(end))

		// entry uintptr
		off = int32(setAddr(ftab, ctxt.Arch, int64(off), s, 0))

		// name int32
		nameoff, ok := state.funcNameOffset[s]
		if !ok {
			panic("couldn't find function name offset")
		}
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), uint32(nameoff)))

		// args int32
		// TODO: Move into funcinfo.
		args := uint32(0)
		if fi.Valid() {
			args = uint32(fi.Args())
		}
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), args))

		// deferreturn
		deferreturn := oldState.computeDeferReturn(&ctxt.Target, s)
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), deferreturn))

		cu := ldr.SymUnit(s)

		if fi.Valid() && fi.NumInlTree() > 0 {
			its := oldState.genInlTreeSym(cu, fi, ctxt.Arch, state)
			funcdata[objabi.FUNCDATA_InlTree] = its
		}

		// pcdata
		if fi.Valid() {
			off = int32(ftab.SetUint32(ctxt.Arch, int64(off), uint32(ldr.SymValue(fi.Pcsp()))))
			off = int32(ftab.SetUint32(ctxt.Arch, int64(off), uint32(ldr.SymValue(fi.Pcfile()))))
			off = int32(ftab.SetUint32(ctxt.Arch, int64(off), uint32(ldr.SymValue(fi.Pcline()))))
		} else {
			off += 12
		}
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), uint32(numPCData)))

		// Store the offset to compilation unit's file table.
		cuIdx := ^uint32(0)
		if cu := ldr.SymUnit(s); cu != nil {
			cuIdx = cuOffsets[cu.PclnIndex]
		}
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), cuIdx))

		// funcID uint8
		var funcID objabi.FuncID
		if fi.Valid() {
			funcID = fi.FuncID()
		}
		off = int32(ftab.SetUint8(ctxt.Arch, int64(off), uint8(funcID)))

		off += 2 // pad

		// nfuncdata must be the final entry.
		off = int32(ftab.SetUint8(ctxt.Arch, int64(off), uint8(len(funcdata))))

		// Output the pcdata.
		if fi.Valid() {
			for i, pcSym := range fi.Pcdata() {
				ftab.SetUint32(ctxt.Arch, int64(off+int32(i*4)), uint32(ldr.SymValue(pcSym)))
			}
			if writeInlPCData {
				ftab.SetUint32(ctxt.Arch, int64(off+objabi.PCDATA_InlTreeIndex*4), uint32(ldr.SymValue(fi.Pcinline())))
			}
		}
		off += numPCData * 4

		// funcdata, must be pointer-aligned and we're only int32-aligned.
		// Missing funcdata will be 0 (nil pointer).
		if len(funcdata) > 0 {
			if off&int32(ctxt.Arch.PtrSize-1) != 0 {
				off += 4
			}
			for i := range funcdata {
				dataoff := int64(off) + int64(ctxt.Arch.PtrSize)*int64(i)
				if funcdata[i] == 0 {
					ftab.SetUint(ctxt.Arch, dataoff, uint64(funcdataoff[i]))
					continue
				}
				// TODO: Dedup.
				funcdataBytes += int64(len(ldr.Data(funcdata[i])))
				setAddr(ftab, ctxt.Arch, dataoff, funcdata[i], funcdataoff[i])
			}
			off += int32(len(funcdata)) * int32(ctxt.Arch.PtrSize)
		}

		if off != end {
			ctxt.Errorf(s, "bad math in functab: funcstart=%d off=%d but end=%d (npcdata=%d nfuncdata=%d ptrsize=%d)", funcstart, off, end, numPCData, len(funcdata), ctxt.Arch.PtrSize)
			errorexit()
		}

		nfunc++
	}

	// Final entry of table is just end pc.
	setAddr(ftab, ctxt.Arch, int64(nfunc)*2*int64(ctxt.Arch.PtrSize), state.lastFunc, ldr.SymSize(state.lastFunc))

	ftab.SetSize(int64(len(ftab.Data())))

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("pclntab=%d bytes, funcdata total %d bytes\n", ftab.Size(), funcdataBytes)
	}

	return state
}

func gorootFinal() string {
	root := objabi.GOROOT
	if final := os.Getenv("GOROOT_FINAL"); final != "" {
		root = final
	}
	return root
}

func expandGoroot(s string) string {
	const n = len("$GOROOT")
	if len(s) >= n+1 && s[:n] == "$GOROOT" && (s[n] == '/' || s[n] == '\\') {
		return filepath.ToSlash(filepath.Join(gorootFinal(), s[n:]))
	}
	return s
}

const (
	BUCKETSIZE    = 256 * MINFUNC
	SUBBUCKETS    = 16
	SUBBUCKETSIZE = BUCKETSIZE / SUBBUCKETS
	NOIDX         = 0x7fffffff
)

// findfunctab generates a lookup table to quickly find the containing
// function for a pc. See src/runtime/symtab.go:findfunc for details.
func (ctxt *Link) findfunctab(state *pclntab, container loader.Bitmap) {
	ldr := ctxt.loader

	// find min and max address
	min := ldr.SymValue(ctxt.Textp[0])
	lastp := ctxt.Textp[len(ctxt.Textp)-1]
	max := ldr.SymValue(lastp) + ldr.SymSize(lastp)

	// for each subbucket, compute the minimum of all symbol indexes
	// that map to that subbucket.
	n := int32((max - min + SUBBUCKETSIZE - 1) / SUBBUCKETSIZE)

	nbuckets := int32((max - min + BUCKETSIZE - 1) / BUCKETSIZE)

	size := 4*int64(nbuckets) + int64(n)

	writeFindFuncTab := func(_ *Link, s loader.Sym) {
		t := ldr.MakeSymbolUpdater(s)

		indexes := make([]int32, n)
		for i := int32(0); i < n; i++ {
			indexes[i] = NOIDX
		}
		idx := int32(0)
		for i, s := range ctxt.Textp {
			if !emitPcln(ctxt, s, container) {
				continue
			}
			p := ldr.SymValue(s)
			var e loader.Sym
			i++
			if i < len(ctxt.Textp) {
				e = ctxt.Textp[i]
			}
			for e != 0 && !emitPcln(ctxt, e, container) && i < len(ctxt.Textp) {
				e = ctxt.Textp[i]
				i++
			}
			q := max
			if e != 0 {
				q = ldr.SymValue(e)
			}

			//print("%d: [%lld %lld] %s\n", idx, p, q, s->name);
			for ; p < q; p += SUBBUCKETSIZE {
				i = int((p - min) / SUBBUCKETSIZE)
				if indexes[i] > idx {
					indexes[i] = idx
				}
			}

			i = int((q - 1 - min) / SUBBUCKETSIZE)
			if indexes[i] > idx {
				indexes[i] = idx
			}
			idx++
		}

		// fill in table
		for i := int32(0); i < nbuckets; i++ {
			base := indexes[i*SUBBUCKETS]
			if base == NOIDX {
				Errorf(nil, "hole in findfunctab")
			}
			t.SetUint32(ctxt.Arch, int64(i)*(4+SUBBUCKETS), uint32(base))
			for j := int32(0); j < SUBBUCKETS && i*SUBBUCKETS+j < n; j++ {
				idx = indexes[i*SUBBUCKETS+j]
				if idx == NOIDX {
					Errorf(nil, "hole in findfunctab")
				}
				if idx-base >= 256 {
					Errorf(nil, "too many functions in a findfunc bucket! %d/%d %d %d", i, nbuckets, j, idx-base)
				}

				t.SetUint8(ctxt.Arch, int64(i)*(4+SUBBUCKETS)+4+int64(j), uint8(idx-base))
			}
		}
	}

	state.findfunctab = ctxt.createGeneratorSymbol("runtime.findfunctab", 0, sym.SRODATA, size, writeFindFuncTab)
	ldr.SetAttrReachable(state.findfunctab, true)
	ldr.SetAttrLocal(state.findfunctab, true)
}

// findContainerSyms returns a bitmap, indexed by symbol number, where there's
// a 1 for every container symbol.
func (ctxt *Link) findContainerSyms() loader.Bitmap {
	ldr := ctxt.loader
	container := loader.MakeBitmap(ldr.NSym())
	// Find container symbols and mark them as such.
	for _, s := range ctxt.Textp {
		outer := ldr.OuterSym(s)
		if outer != 0 {
			container.Set(outer)
		}
	}
	return container
}
