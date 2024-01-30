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
	"internal/abi"
	"internal/buildcfg"
	"os"
	"path/filepath"
	"strings"
)

const funcSize = 11 * 4 // funcSize is the size of the _func object in runtime/runtime2.go

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
}

// addGeneratedSym adds a generator symbol to pclntab, returning the new Sym.
// It is the caller's responsibility to save the symbol in state.
func (state *pclntab) addGeneratedSym(ctxt *Link, name string, size int64, f generatorFunc) loader.Sym {
	size = Rnd(size, int64(ctxt.Arch.PtrSize))
	state.size += size
	s := ctxt.createGeneratorSymbol(name, 0, sym.SPCLNTAB, size, f)
	ctxt.loader.SetAttrReachable(s, true)
	ctxt.loader.SetCarrierSym(s, state.carrier)
	ctxt.loader.SetAttrNotInSymbolTable(s, true)
	return s
}

// makePclntab makes a pclntab object, and assembles all the compilation units
// we'll need to write pclntab. Returns the pclntab structure, a slice of the
// CompilationUnits we need, and a slice of the function symbols we need to
// generate pclntab.
func makePclntab(ctxt *Link, container loader.Bitmap) (*pclntab, []*sym.CompilationUnit, []loader.Sym) {
	ldr := ctxt.loader
	state := new(pclntab)

	// Gather some basic stats and info.
	seenCUs := make(map[*sym.CompilationUnit]struct{})
	compUnits := []*sym.CompilationUnit{}
	funcs := []loader.Sym{}

	for _, s := range ctxt.Textp {
		if !emitPcln(ctxt, s, container) {
			continue
		}
		funcs = append(funcs, s)
		state.nfunc++
		if state.firstFunc == 0 {
			state.firstFunc = s
		}
		state.lastFunc = s

		// We need to keep track of all compilation units we see. Some symbols
		// (eg, go.buildid, _cgoexp_, etc) won't have a compilation unit.
		cu := ldr.SymUnit(s)
		if _, ok := seenCUs[cu]; cu != nil && !ok {
			seenCUs[cu] = struct{}{}
			cu.PclnIndex = len(compUnits)
			compUnits = append(compUnits, cu)
		}
	}
	return state, compUnits, funcs
}

func emitPcln(ctxt *Link, s loader.Sym, container loader.Bitmap) bool {
	if ctxt.Target.IsRISCV64() {
		// Avoid adding local symbols to the pcln table - RISC-V
		// linking generates a very large number of these, particularly
		// for HI20 symbols (which we need to load in order to be able
		// to resolve relocations). Unnecessarily including all of
		// these symbols quickly blows out the size of the pcln table
		// and overflows hash buckets.
		symName := ctxt.loader.SymName(s)
		if symName == "" || strings.HasPrefix(symName, ".L") {
			return false
		}
	}

	// We want to generate func table entries only for the "lowest
	// level" symbols, not containers of subsymbols.
	return !container.Has(s)
}

func computeDeferReturn(ctxt *Link, deferReturnSym, s loader.Sym) uint32 {
	ldr := ctxt.loader
	target := ctxt.Target
	deferreturn := uint32(0)
	lastWasmAddr := uint32(0)

	relocs := ldr.Relocs(s)
	for ri := 0; ri < relocs.Count(); ri++ {
		r := relocs.At(ri)
		if target.IsWasm() && r.Type() == objabi.R_ADDR {
			// wasm/ssa.go generates an ARESUMEPOINT just
			// before the deferreturn call. The "PC" of
			// the deferreturn call is stored in the
			// R_ADDR relocation on the ARESUMEPOINT.
			lastWasmAddr = uint32(r.Add())
		}
		if r.Type().IsDirectCall() && (r.Sym() == deferReturnSym || ldr.IsDeferReturnTramp(r.Sym())) {
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
				case sys.ARM, sys.ARM64, sys.Loong64, sys.MIPS, sys.MIPS64, sys.PPC64, sys.RISCV64:
					// no change
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
func genInlTreeSym(ctxt *Link, cu *sym.CompilationUnit, fi loader.FuncInfo, arch *sys.Arch, nameOffsets map[loader.Sym]uint32) loader.Sym {
	ldr := ctxt.loader
	its := ldr.CreateExtSym("", 0)
	inlTreeSym := ldr.MakeSymbolUpdater(its)
	// Note: the generated symbol is given a type of sym.SGOFUNC, as a
	// signal to the symtab() phase that it needs to be grouped in with
	// other similar symbols (gcdata, etc); the dodata() phase will
	// eventually switch the type back to SRODATA.
	inlTreeSym.SetType(sym.SGOFUNC)
	ldr.SetAttrReachable(its, true)
	ldr.SetSymAlign(its, 4) // it has 32-bit fields
	ninl := fi.NumInlTree()
	for i := 0; i < int(ninl); i++ {
		call := fi.InlTree(i)
		nameOff, ok := nameOffsets[call.Func]
		if !ok {
			panic("couldn't find function name offset")
		}

		inlFunc := ldr.FuncInfo(call.Func)
		var funcID abi.FuncID
		startLine := int32(0)
		if inlFunc.Valid() {
			funcID = inlFunc.FuncID()
			startLine = inlFunc.StartLine()
		} else if !ctxt.linkShared {
			// Inlined functions are always Go functions, and thus
			// must have FuncInfo.
			//
			// Unfortunately, with -linkshared, the inlined
			// function may be external symbols (from another
			// shared library), and we don't load FuncInfo from the
			// shared library. We will report potentially incorrect
			// FuncID in this case. See https://go.dev/issue/55954.
			panic(fmt.Sprintf("inlined function %s missing func info", ldr.SymName(call.Func)))
		}

		// Construct runtime.inlinedCall value.
		const size = 16
		inlTreeSym.SetUint8(arch, int64(i*size+0), uint8(funcID))
		// Bytes 1-3 are unused.
		inlTreeSym.SetUint32(arch, int64(i*size+4), uint32(nameOff))
		inlTreeSym.SetUint32(arch, int64(i*size+8), uint32(call.ParentPC))
		inlTreeSym.SetUint32(arch, int64(i*size+12), uint32(startLine))
	}
	return its
}

// makeInlSyms returns a map of loader.Sym that are created inlSyms.
func makeInlSyms(ctxt *Link, funcs []loader.Sym, nameOffsets map[loader.Sym]uint32) map[loader.Sym]loader.Sym {
	ldr := ctxt.loader
	// Create the inline symbols we need.
	inlSyms := make(map[loader.Sym]loader.Sym)
	for _, s := range funcs {
		if fi := ldr.FuncInfo(s); fi.Valid() {
			fi.Preload()
			if fi.NumInlTree() > 0 {
				inlSyms[s] = genInlTreeSym(ctxt, ldr.SymUnit(s), fi, ctxt.Arch, nameOffsets)
			}
		}
	}
	return inlSyms
}

// generatePCHeader creates the runtime.pcheader symbol, setting it up as a
// generator to fill in its data later.
func (state *pclntab) generatePCHeader(ctxt *Link) {
	ldr := ctxt.loader
	textStartOff := int64(8 + 2*ctxt.Arch.PtrSize)
	size := int64(8 + 8*ctxt.Arch.PtrSize)
	writeHeader := func(ctxt *Link, s loader.Sym) {
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
		// Keep in sync with runtime/symtab.go:pcHeader and package debug/gosym.
		header.SetUint32(ctxt.Arch, 0, 0xfffffff1)
		header.SetUint8(ctxt.Arch, 6, uint8(ctxt.Arch.MinLC))
		header.SetUint8(ctxt.Arch, 7, uint8(ctxt.Arch.PtrSize))
		off := header.SetUint(ctxt.Arch, 8, uint64(state.nfunc))
		off = header.SetUint(ctxt.Arch, off, uint64(state.nfiles))
		if off != textStartOff {
			panic(fmt.Sprintf("pcHeader textStartOff: %d != %d", off, textStartOff))
		}
		off += int64(ctxt.Arch.PtrSize) // skip runtimeText relocation
		off = writeSymOffset(off, state.funcnametab)
		off = writeSymOffset(off, state.cutab)
		off = writeSymOffset(off, state.filetab)
		off = writeSymOffset(off, state.pctab)
		off = writeSymOffset(off, state.pclntab)
		if off != size {
			panic(fmt.Sprintf("pcHeader size: %d != %d", off, size))
		}
	}

	state.pcheader = state.addGeneratedSym(ctxt, "runtime.pcheader", size, writeHeader)
	// Create the runtimeText relocation.
	sb := ldr.MakeSymbolUpdater(state.pcheader)
	sb.SetAddr(ctxt.Arch, textStartOff, ldr.Lookup("runtime.text", 0))
}

// walkFuncs iterates over the funcs, calling a function for each unique
// function and inlined function.
func walkFuncs(ctxt *Link, funcs []loader.Sym, f func(loader.Sym)) {
	ldr := ctxt.loader
	seen := make(map[loader.Sym]struct{})
	for _, s := range funcs {
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

// generateFuncnametab creates the function name table. Returns a map of
// func symbol to the name offset in runtime.funcnamtab.
func (state *pclntab) generateFuncnametab(ctxt *Link, funcs []loader.Sym) map[loader.Sym]uint32 {
	nameOffsets := make(map[loader.Sym]uint32, state.nfunc)

	// Write the null terminated strings.
	writeFuncNameTab := func(ctxt *Link, s loader.Sym) {
		symtab := ctxt.loader.MakeSymbolUpdater(s)
		for s, off := range nameOffsets {
			symtab.AddCStringAt(int64(off), ctxt.loader.SymName(s))
		}
	}

	// Loop through the CUs, and calculate the size needed.
	var size int64
	walkFuncs(ctxt, funcs, func(s loader.Sym) {
		nameOffsets[s] = uint32(size)
		size += int64(len(ctxt.loader.SymName(s)) + 1) // NULL terminate
	})

	state.funcnametab = state.addGeneratedSym(ctxt, "runtime.funcnametab", size, writeFuncNameTab)
	return nameOffsets
}

// walkFilenames walks funcs, calling a function for each filename used in each
// function's line table.
func walkFilenames(ctxt *Link, funcs []loader.Sym, f func(*sym.CompilationUnit, goobj.CUFileIndex)) {
	ldr := ctxt.loader

	// Loop through all functions, finding the filenames we need.
	for _, s := range funcs {
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
//	runtime.cutab:
//	  CU[M]
//	   offsetToFilename[0]
//	   offsetToFilename[1]
//	   ..
//
//	runtime.filetab
//	   filename[0]
//	   filename[1]
//
// Looking up a filename then becomes:
//  0. Given a func, and filename index [K]
//  1. Get Func.CUIndex:       M := func.cuOffset
//  2. Find filename offset:   fileOffset := runtime.cutab[M+K]
//  3. Get the filename:       getcstring(runtime.filetab[fileOffset])
func (state *pclntab) generateFilenameTabs(ctxt *Link, compUnits []*sym.CompilationUnit, funcs []loader.Sym) []uint32 {
	// On a per-CU basis, keep track of all the filenames we need.
	//
	// Note, that we store the filenames in a separate section in the object
	// files, and deduplicate based on the actual value. It would be better to
	// store the filenames as symbols, using content addressable symbols (and
	// then not loading extra filenames), and just use the hash value of the
	// symbol name to do this cataloging.
	//
	// TODO: Store filenames as symbols. (Note this would be easiest if you
	// also move strings to ALWAYS using the larger content addressable hash
	// function, and use that hash value for uniqueness testing.)
	cuEntries := make([]goobj.CUFileIndex, len(compUnits))
	fileOffsets := make(map[string]uint32)

	// Walk the filenames.
	// We store the total filename string length we need to load, and the max
	// file index we've seen per CU so we can calculate how large the
	// CU->global table needs to be.
	var fileSize int64
	walkFilenames(ctxt, funcs, func(cu *sym.CompilationUnit, i goobj.CUFileIndex) {
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
func (state *pclntab) generatePctab(ctxt *Link, funcs []loader.Sym) {
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
	var pcsp, pcline, pcfile, pcinline loader.Sym
	var pcdata []loader.Sym
	for _, s := range funcs {
		fi := ldr.FuncInfo(s)
		if !fi.Valid() {
			continue
		}
		fi.Preload()
		pcsp, pcfile, pcline, pcinline, pcdata = ldr.PcdataAuxs(s, pcdata)

		pcSyms := []loader.Sym{pcsp, pcfile, pcline}
		for _, pcSym := range pcSyms {
			saveOffset(pcSym)
		}
		for _, pcSym := range pcdata {
			saveOffset(pcSym)
		}
		if fi.NumInlTree() > 0 {
			saveOffset(pcinline)
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

// numPCData returns the number of PCData syms for the FuncInfo.
// NB: Preload must be called on valid FuncInfos before calling this function.
func numPCData(ldr *loader.Loader, s loader.Sym, fi loader.FuncInfo) uint32 {
	if !fi.Valid() {
		return 0
	}
	numPCData := uint32(ldr.NumPcdata(s))
	if fi.NumInlTree() > 0 {
		if numPCData < abi.PCDATA_InlTreeIndex+1 {
			numPCData = abi.PCDATA_InlTreeIndex + 1
		}
	}
	return numPCData
}

// generateFunctab creates the runtime.functab
//
// runtime.functab contains two things:
//
//   - pc->func look up table.
//   - array of func objects, interleaved with pcdata and funcdata
func (state *pclntab) generateFunctab(ctxt *Link, funcs []loader.Sym, inlSyms map[loader.Sym]loader.Sym, cuOffsets []uint32, nameOffsets map[loader.Sym]uint32) {
	// Calculate the size of the table.
	size, startLocations := state.calculateFunctabSize(ctxt, funcs)
	writePcln := func(ctxt *Link, s loader.Sym) {
		ldr := ctxt.loader
		sb := ldr.MakeSymbolUpdater(s)
		// Write the data.
		writePCToFunc(ctxt, sb, funcs, startLocations)
		writeFuncs(ctxt, sb, funcs, inlSyms, startLocations, cuOffsets, nameOffsets)
	}
	state.pclntab = state.addGeneratedSym(ctxt, "runtime.functab", size, writePcln)
}

// funcData returns the funcdata and offsets for the FuncInfo.
// The funcdata are written into runtime.functab after each func
// object. This is a helper function to make querying the FuncInfo object
// cleaner.
//
// NB: Preload must be called on the FuncInfo before calling.
// NB: fdSyms is used as scratch space.
func funcData(ldr *loader.Loader, s loader.Sym, fi loader.FuncInfo, inlSym loader.Sym, fdSyms []loader.Sym) []loader.Sym {
	fdSyms = fdSyms[:0]
	if fi.Valid() {
		fdSyms = ldr.Funcdata(s, fdSyms)
		if fi.NumInlTree() > 0 {
			if len(fdSyms) < abi.FUNCDATA_InlTree+1 {
				fdSyms = append(fdSyms, make([]loader.Sym, abi.FUNCDATA_InlTree+1-len(fdSyms))...)
			}
			fdSyms[abi.FUNCDATA_InlTree] = inlSym
		}
	}
	return fdSyms
}

// calculateFunctabSize calculates the size of the pclntab, and the offsets in
// the output buffer for individual func entries.
func (state pclntab) calculateFunctabSize(ctxt *Link, funcs []loader.Sym) (int64, []uint32) {
	ldr := ctxt.loader
	startLocations := make([]uint32, len(funcs))

	// Allocate space for the pc->func table. This structure consists of a pc offset
	// and an offset to the func structure. After that, we have a single pc
	// value that marks the end of the last function in the binary.
	size := int64(int(state.nfunc)*2*4 + 4)

	// Now find the space for the func objects. We do this in a running manner,
	// so that we can find individual starting locations.
	for i, s := range funcs {
		size = Rnd(size, int64(ctxt.Arch.PtrSize))
		startLocations[i] = uint32(size)
		fi := ldr.FuncInfo(s)
		size += funcSize
		if fi.Valid() {
			fi.Preload()
			numFuncData := ldr.NumFuncdata(s)
			if fi.NumInlTree() > 0 {
				if numFuncData < abi.FUNCDATA_InlTree+1 {
					numFuncData = abi.FUNCDATA_InlTree + 1
				}
			}
			size += int64(numPCData(ldr, s, fi) * 4)
			size += int64(numFuncData * 4)
		}
	}

	return size, startLocations
}

// writePCToFunc writes the PC->func lookup table.
func writePCToFunc(ctxt *Link, sb *loader.SymbolBuilder, funcs []loader.Sym, startLocations []uint32) {
	ldr := ctxt.loader
	textStart := ldr.SymValue(ldr.Lookup("runtime.text", 0))
	pcOff := func(s loader.Sym) uint32 {
		off := ldr.SymValue(s) - textStart
		if off < 0 {
			panic(fmt.Sprintf("expected func %s(%x) to be placed at or after textStart (%x)", ldr.SymName(s), ldr.SymValue(s), textStart))
		}
		return uint32(off)
	}
	for i, s := range funcs {
		sb.SetUint32(ctxt.Arch, int64(i*2*4), pcOff(s))
		sb.SetUint32(ctxt.Arch, int64((i*2+1)*4), startLocations[i])
	}

	// Final entry of table is just end pc offset.
	lastFunc := funcs[len(funcs)-1]
	sb.SetUint32(ctxt.Arch, int64(len(funcs))*2*4, pcOff(lastFunc)+uint32(ldr.SymSize(lastFunc)))
}

// writeFuncs writes the func structures and pcdata to runtime.functab.
func writeFuncs(ctxt *Link, sb *loader.SymbolBuilder, funcs []loader.Sym, inlSyms map[loader.Sym]loader.Sym, startLocations, cuOffsets []uint32, nameOffsets map[loader.Sym]uint32) {
	ldr := ctxt.loader
	deferReturnSym := ldr.Lookup("runtime.deferreturn", abiInternalVer)
	gofunc := ldr.Lookup("go:func.*", 0)
	gofuncBase := ldr.SymValue(gofunc)
	textStart := ldr.SymValue(ldr.Lookup("runtime.text", 0))
	funcdata := []loader.Sym{}
	var pcsp, pcfile, pcline, pcinline loader.Sym
	var pcdata []loader.Sym

	// Write the individual func objects.
	for i, s := range funcs {
		startLine := int32(0)
		fi := ldr.FuncInfo(s)
		if fi.Valid() {
			fi.Preload()
			pcsp, pcfile, pcline, pcinline, pcdata = ldr.PcdataAuxs(s, pcdata)
			startLine = fi.StartLine()
		}

		off := int64(startLocations[i])
		// entryOff uint32 (offset of func entry PC from textStart)
		entryOff := ldr.SymValue(s) - textStart
		if entryOff < 0 {
			panic(fmt.Sprintf("expected func %s(%x) to be placed before or at textStart (%x)", ldr.SymName(s), ldr.SymValue(s), textStart))
		}
		off = sb.SetUint32(ctxt.Arch, off, uint32(entryOff))

		// nameOff int32
		nameOff, ok := nameOffsets[s]
		if !ok {
			panic("couldn't find function name offset")
		}
		off = sb.SetUint32(ctxt.Arch, off, uint32(nameOff))

		// args int32
		// TODO: Move into funcinfo.
		args := uint32(0)
		if fi.Valid() {
			args = uint32(fi.Args())
		}
		off = sb.SetUint32(ctxt.Arch, off, args)

		// deferreturn
		deferreturn := computeDeferReturn(ctxt, deferReturnSym, s)
		off = sb.SetUint32(ctxt.Arch, off, deferreturn)

		// pcdata
		if fi.Valid() {
			off = sb.SetUint32(ctxt.Arch, off, uint32(ldr.SymValue(pcsp)))
			off = sb.SetUint32(ctxt.Arch, off, uint32(ldr.SymValue(pcfile)))
			off = sb.SetUint32(ctxt.Arch, off, uint32(ldr.SymValue(pcline)))
		} else {
			off += 12
		}
		off = sb.SetUint32(ctxt.Arch, off, uint32(numPCData(ldr, s, fi)))

		// Store the offset to compilation unit's file table.
		cuIdx := ^uint32(0)
		if cu := ldr.SymUnit(s); cu != nil {
			cuIdx = cuOffsets[cu.PclnIndex]
		}
		off = sb.SetUint32(ctxt.Arch, off, cuIdx)

		// startLine int32
		off = sb.SetUint32(ctxt.Arch, off, uint32(startLine))

		// funcID uint8
		var funcID abi.FuncID
		if fi.Valid() {
			funcID = fi.FuncID()
		}
		off = sb.SetUint8(ctxt.Arch, off, uint8(funcID))

		// flag uint8
		var flag abi.FuncFlag
		if fi.Valid() {
			flag = fi.FuncFlag()
		}
		off = sb.SetUint8(ctxt.Arch, off, uint8(flag))

		off += 1 // pad

		// nfuncdata must be the final entry.
		funcdata = funcData(ldr, s, fi, 0, funcdata)
		off = sb.SetUint8(ctxt.Arch, off, uint8(len(funcdata)))

		// Output the pcdata.
		if fi.Valid() {
			for j, pcSym := range pcdata {
				sb.SetUint32(ctxt.Arch, off+int64(j*4), uint32(ldr.SymValue(pcSym)))
			}
			if fi.NumInlTree() > 0 {
				sb.SetUint32(ctxt.Arch, off+abi.PCDATA_InlTreeIndex*4, uint32(ldr.SymValue(pcinline)))
			}
		}

		// Write funcdata refs as offsets from go:func.* and go:funcrel.*.
		funcdata = funcData(ldr, s, fi, inlSyms[s], funcdata)
		// Missing funcdata will be ^0. See runtime/symtab.go:funcdata.
		off = int64(startLocations[i] + funcSize + numPCData(ldr, s, fi)*4)
		for j := range funcdata {
			dataoff := off + int64(4*j)
			fdsym := funcdata[j]

			// cmd/internal/obj optimistically populates ArgsPointerMaps and
			// ArgInfo for assembly functions, hoping that the compiler will
			// emit appropriate symbols from their Go stub declarations. If
			// it didn't though, just ignore it.
			//
			// TODO(cherryyz): Fix arg map generation (see discussion on CL 523335).
			if fdsym != 0 && (j == abi.FUNCDATA_ArgsPointerMaps || j == abi.FUNCDATA_ArgInfo) && ldr.IsFromAssembly(s) && ldr.Data(fdsym) == nil {
				fdsym = 0
			}

			if fdsym == 0 {
				sb.SetUint32(ctxt.Arch, dataoff, ^uint32(0)) // ^0 is a sentinel for "no value"
				continue
			}

			if outer := ldr.OuterSym(fdsym); outer != gofunc {
				panic(fmt.Sprintf("bad carrier sym for symbol %s (funcdata %s#%d), want go:func.* got %s", ldr.SymName(fdsym), ldr.SymName(s), j, ldr.SymName(outer)))
			}
			sb.SetUint32(ctxt.Arch, dataoff, uint32(ldr.SymValue(fdsym)-gofuncBase))
		}
	}
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
	//      runtime.functab
	//        function table, alternating PC and offset to func struct [each entry thearch.ptrsize bytes]
	//        end PC [thearch.ptrsize bytes]
	//        func structures, pcdata offsets, func data.

	state, compUnits, funcs := makePclntab(ctxt, container)

	ldr := ctxt.loader
	state.carrier = ldr.LookupOrCreateSym("runtime.pclntab", 0)
	ldr.MakeSymbolUpdater(state.carrier).SetType(sym.SPCLNTAB)
	ldr.SetAttrReachable(state.carrier, true)
	setCarrierSym(sym.SPCLNTAB, state.carrier)

	state.generatePCHeader(ctxt)
	nameOffsets := state.generateFuncnametab(ctxt, funcs)
	cuOffsets := state.generateFilenameTabs(ctxt, compUnits, funcs)
	state.generatePctab(ctxt, funcs)
	inlSyms := makeInlSyms(ctxt, funcs, nameOffsets)
	state.generateFunctab(ctxt, funcs, inlSyms, cuOffsets, nameOffsets)

	return state
}

func gorootFinal() string {
	root := buildcfg.GOROOT
	if final := os.Getenv("GOROOT_FINAL"); final != "" {
		root = final
	}
	return root
}

func expandGoroot(s string) string {
	const n = len("$GOROOT")
	if len(s) >= n+1 && s[:n] == "$GOROOT" && (s[n] == '/' || s[n] == '\\') {
		if final := gorootFinal(); final != "" {
			return filepath.ToSlash(filepath.Join(final, s[n:]))
		}
	}
	return s
}

const (
	SUBBUCKETS    = 16
	SUBBUCKETSIZE = abi.Pcbucketsize / SUBBUCKETS
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

	nbuckets := int32((max - min + abi.Pcbucketsize - 1) / abi.Pcbucketsize)

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
