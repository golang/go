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
)

// pclntab holds the state needed for pclntab generation.
type pclntab struct {
	// The size of the func object in the runtime.
	funcSize uint32

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

// makePclntab makes a pclntab object, and assembles all the compilation units
// we'll need to write pclntab. Returns the pclntab structure, a slice of the
// CompilationUnits we need, and a slice of the function symbols we need to
// generate pclntab.
func makePclntab(ctxt *Link, container loader.Bitmap) (*pclntab, []*sym.CompilationUnit, []loader.Sym) {
	ldr := ctxt.loader

	state := &pclntab{
		// This is the size of the _func object in runtime/runtime2.go.
		funcSize: uint32(ctxt.Arch.PtrSize + 9*4),
	}

	// Gather some basic stats and info.
	seenCUs := make(map[*sym.CompilationUnit]struct{})
	prevSect := ldr.SymSect(ctxt.Textp[0])
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
	return state, compUnits, funcs
}

func emitPcln(ctxt *Link, s loader.Sym, container loader.Bitmap) bool {
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
			// Wasm does not have a live variable set at the deferreturn
			// call itself. Instead it has one identified by the
			// resumption point immediately preceding the deferreturn.
			// The wasm code has a R_ADDR relocation which is used to
			// set the resumption point to PC_B.
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
	ninl := fi.NumInlTree()
	for i := 0; i < int(ninl); i++ {
		call := fi.InlTree(i)
		val := call.File
		nameoff, ok := nameOffsets[call.Func]
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
			symtab.AddStringAt(int64(off), ctxt.loader.SymName(s))
		}
	}

	// Loop through the CUs, and calculate the size needed.
	var size int64
	walkFuncs(ctxt, funcs, func(s loader.Sym) {
		nameOffsets[s] = uint32(size)
		size += int64(ctxt.loader.SymNameLen(s)) + 1 // NULL terminate
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
func (state *pclntab) generateFilenameTabs(ctxt *Link, compUnits []*sym.CompilationUnit, funcs []loader.Sym) []uint32 {
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
	for _, s := range funcs {
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

// numPCData returns the number of PCData syms for the FuncInfo.
// NB: Preload must be called on valid FuncInfos before calling this function.
func numPCData(fi loader.FuncInfo) uint32 {
	if !fi.Valid() {
		return 0
	}
	numPCData := uint32(len(fi.Pcdata()))
	if fi.NumInlTree() > 0 {
		if numPCData < objabi.PCDATA_InlTreeIndex+1 {
			numPCData = objabi.PCDATA_InlTreeIndex + 1
		}
	}
	return numPCData
}

// Helper types for iterating pclntab.
type pclnSetAddr func(*loader.SymbolBuilder, *sys.Arch, int64, loader.Sym, int64) int64
type pclnSetUint func(*loader.SymbolBuilder, *sys.Arch, int64, uint64) int64

// generateFunctab creates the runtime.functab
//
// runtime.functab contains two things:
//
//   - pc->func look up table.
//   - array of func objects, interleaved with pcdata and funcdata
//
// Because of timing in the linker, generating this table takes two passes.
// The first pass is executed early in the link, and it creates any needed
// relocations to layout the data. The pieces that need relocations are:
//   1) the PC->func table.
//   2) The entry points in the func objects.
//   3) The funcdata.
// (1) and (2) are handled in walkPCToFunc. (3) is handled in walkFuncdata.
//
// After relocations, once we know where to write things in the output buffer,
// we execute the second pass, which is actually writing the data.
func (state *pclntab) generateFunctab(ctxt *Link, funcs []loader.Sym, inlSyms map[loader.Sym]loader.Sym, cuOffsets []uint32, nameOffsets map[loader.Sym]uint32) {
	// Calculate the size of the table.
	size, startLocations := state.calculateFunctabSize(ctxt, funcs)

	// If we are internally linking a static executable, the function addresses
	// are known, so we can just use them instead of emitting relocations. For
	// other cases we still need to emit relocations.
	//
	// This boolean just helps us figure out which callback to use.
	useSymValue := ctxt.IsExe() && ctxt.IsInternal()

	writePcln := func(ctxt *Link, s loader.Sym) {
		ldr := ctxt.loader
		sb := ldr.MakeSymbolUpdater(s)

		// Create our callbacks.
		var setAddr pclnSetAddr
		if useSymValue {
			// We need to write the offset.
			setAddr = func(s *loader.SymbolBuilder, arch *sys.Arch, off int64, tgt loader.Sym, add int64) int64 {
				if v := ldr.SymValue(tgt); v != 0 {
					s.SetUint(arch, off, uint64(v+add))
				}
				return 0
			}
		} else {
			// We already wrote relocations.
			setAddr = func(s *loader.SymbolBuilder, arch *sys.Arch, off int64, tgt loader.Sym, add int64) int64 { return 0 }
		}

		// Write the data.
		writePcToFunc(ctxt, sb, funcs, startLocations, setAddr, (*loader.SymbolBuilder).SetUint)
		writeFuncs(ctxt, sb, funcs, inlSyms, startLocations, cuOffsets, nameOffsets)
		state.writeFuncData(ctxt, sb, funcs, inlSyms, startLocations, setAddr, (*loader.SymbolBuilder).SetUint)
	}

	state.pclntab = state.addGeneratedSym(ctxt, "runtime.functab", size, writePcln)

	// Create the relocations we need.
	ldr := ctxt.loader
	sb := ldr.MakeSymbolUpdater(state.pclntab)

	var setAddr pclnSetAddr
	if useSymValue {
		// If we should use the symbol value, and we don't have one, write a relocation.
		setAddr = func(sb *loader.SymbolBuilder, arch *sys.Arch, off int64, tgt loader.Sym, add int64) int64 {
			if v := ldr.SymValue(tgt); v == 0 {
				sb.SetAddrPlus(arch, off, tgt, add)
			}
			return 0
		}
	} else {
		// If we're externally linking, write a relocation.
		setAddr = (*loader.SymbolBuilder).SetAddrPlus
	}
	setUintNOP := func(*loader.SymbolBuilder, *sys.Arch, int64, uint64) int64 { return 0 }
	writePcToFunc(ctxt, sb, funcs, startLocations, setAddr, setUintNOP)
	if !useSymValue {
		// Generate relocations for funcdata when externally linking.
		state.writeFuncData(ctxt, sb, funcs, inlSyms, startLocations, setAddr, setUintNOP)
	}
}

// funcData returns the funcdata and offsets for the FuncInfo.
// The funcdata and offsets are written into runtime.functab after each func
// object. This is a helper function to make querying the FuncInfo object
// cleaner.
//
// Note, the majority of fdOffsets are 0, meaning there is no offset between
// the compiler's generated symbol, and what the runtime needs. They are
// plumbed through for no loss of generality.
//
// NB: Preload must be called on the FuncInfo before calling.
// NB: fdSyms and fdOffs are used as scratch space.
func funcData(fi loader.FuncInfo, inlSym loader.Sym, fdSyms []loader.Sym, fdOffs []int64) ([]loader.Sym, []int64) {
	fdSyms, fdOffs = fdSyms[:0], fdOffs[:0]
	if fi.Valid() {
		numOffsets := int(fi.NumFuncdataoff())
		for i := 0; i < numOffsets; i++ {
			fdOffs = append(fdOffs, fi.Funcdataoff(i))
		}
		fdSyms = fi.Funcdata(fdSyms)
		if fi.NumInlTree() > 0 {
			if len(fdSyms) < objabi.FUNCDATA_InlTree+1 {
				fdSyms = append(fdSyms, make([]loader.Sym, objabi.FUNCDATA_InlTree+1-len(fdSyms))...)
				fdOffs = append(fdOffs, make([]int64, objabi.FUNCDATA_InlTree+1-len(fdOffs))...)
			}
			fdSyms[objabi.FUNCDATA_InlTree] = inlSym
		}
	}
	return fdSyms, fdOffs
}

// calculateFunctabSize calculates the size of the pclntab, and the offsets in
// the output buffer for individual func entries.
func (state pclntab) calculateFunctabSize(ctxt *Link, funcs []loader.Sym) (int64, []uint32) {
	ldr := ctxt.loader
	startLocations := make([]uint32, len(funcs))

	// Allocate space for the pc->func table. This structure consists of a pc
	// and an offset to the func structure. After that, we have a single pc
	// value that marks the end of the last function in the binary.
	size := int64(int(state.nfunc)*2*ctxt.Arch.PtrSize + ctxt.Arch.PtrSize)

	// Now find the space for the func objects. We do this in a running manner,
	// so that we can find individual starting locations, and because funcdata
	// requires alignment.
	for i, s := range funcs {
		size = Rnd(size, int64(ctxt.Arch.PtrSize))
		startLocations[i] = uint32(size)
		fi := ldr.FuncInfo(s)
		size += int64(state.funcSize)
		if fi.Valid() {
			fi.Preload()
			numFuncData := int(fi.NumFuncdataoff())
			if fi.NumInlTree() > 0 {
				if numFuncData < objabi.FUNCDATA_InlTree+1 {
					numFuncData = objabi.FUNCDATA_InlTree + 1
				}
			}
			size += int64(numPCData(fi) * 4)
			if numFuncData > 0 { // Func data is aligned.
				size = Rnd(size, int64(ctxt.Arch.PtrSize))
			}
			size += int64(numFuncData * ctxt.Arch.PtrSize)
		}
	}

	return size, startLocations
}

// writePcToFunc writes the PC->func lookup table.
// This function walks the pc->func lookup table, executing callbacks
// to generate relocations and writing the values for the table.
func writePcToFunc(ctxt *Link, sb *loader.SymbolBuilder, funcs []loader.Sym, startLocations []uint32, setAddr pclnSetAddr, setUint pclnSetUint) {
	ldr := ctxt.loader
	var prevFunc loader.Sym
	prevSect := ldr.SymSect(funcs[0])
	funcIndex := 0
	for i, s := range funcs {
		if thisSect := ldr.SymSect(s); thisSect != prevSect {
			// With multiple text sections, there may be a hole here in the
			// address space. We use an invalid funcoff value to mark the hole.
			// See also runtime/symtab.go:findfunc
			prevFuncSize := int64(ldr.SymSize(prevFunc))
			setAddr(sb, ctxt.Arch, int64(funcIndex*2*ctxt.Arch.PtrSize), prevFunc, prevFuncSize)
			setUint(sb, ctxt.Arch, int64((funcIndex*2+1)*ctxt.Arch.PtrSize), ^uint64(0))
			funcIndex++
			prevSect = thisSect
		}
		prevFunc = s
		// TODO: We don't actually need these relocations, provided we go to a
		// module->func look-up-table like we do for filenames. We could have a
		// single relocation for the module, and have them all laid out as
		// offsets from the beginning of that module.
		setAddr(sb, ctxt.Arch, int64(funcIndex*2*ctxt.Arch.PtrSize), s, 0)
		setUint(sb, ctxt.Arch, int64((funcIndex*2+1)*ctxt.Arch.PtrSize), uint64(startLocations[i]))
		funcIndex++

		// Write the entry location.
		setAddr(sb, ctxt.Arch, int64(startLocations[i]), s, 0)
	}

	// Final entry of table is just end pc.
	setAddr(sb, ctxt.Arch, int64(funcIndex)*2*int64(ctxt.Arch.PtrSize), prevFunc, ldr.SymSize(prevFunc))
}

// writeFuncData writes the funcdata tables.
//
// This function executes a callback for each funcdata needed in
// runtime.functab. It should be called once for internally linked static
// binaries, or twice (once to generate the needed relocations) for other
// build modes.
//
// Note the output of this function is interwoven with writeFuncs, but this is
// a separate function, because it's needed in different passes in
// generateFunctab.
func (state *pclntab) writeFuncData(ctxt *Link, sb *loader.SymbolBuilder, funcs []loader.Sym, inlSyms map[loader.Sym]loader.Sym, startLocations []uint32, setAddr pclnSetAddr, setUint pclnSetUint) {
	ldr := ctxt.loader
	funcdata, funcdataoff := []loader.Sym{}, []int64{}
	for i, s := range funcs {
		fi := ldr.FuncInfo(s)
		if !fi.Valid() {
			continue
		}
		fi.Preload()

		// funcdata, must be pointer-aligned and we're only int32-aligned.
		// Missing funcdata will be 0 (nil pointer).
		funcdata, funcdataoff := funcData(fi, inlSyms[s], funcdata, funcdataoff)
		if len(funcdata) > 0 {
			off := int64(startLocations[i] + state.funcSize + numPCData(fi)*4)
			off = Rnd(off, int64(ctxt.Arch.PtrSize))
			for j := range funcdata {
				dataoff := off + int64(ctxt.Arch.PtrSize*j)
				if funcdata[j] == 0 {
					setUint(sb, ctxt.Arch, dataoff, uint64(funcdataoff[j]))
					continue
				}
				// TODO: Does this need deduping?
				setAddr(sb, ctxt.Arch, dataoff, funcdata[j], funcdataoff[j])
			}
		}
	}
}

// writeFuncs writes the func structures and pcdata to runtime.functab.
func writeFuncs(ctxt *Link, sb *loader.SymbolBuilder, funcs []loader.Sym, inlSyms map[loader.Sym]loader.Sym, startLocations, cuOffsets []uint32, nameOffsets map[loader.Sym]uint32) {
	ldr := ctxt.loader
	deferReturnSym := ldr.Lookup("runtime.deferreturn", sym.SymVerABIInternal)
	funcdata, funcdataoff := []loader.Sym{}, []int64{}

	// Write the individual func objects.
	for i, s := range funcs {
		fi := ldr.FuncInfo(s)
		if fi.Valid() {
			fi.Preload()
		}

		// Note we skip the space for the entry value -- that's handled inn
		// walkPCToFunc. We don't write it here, because it might require a
		// relocation.
		off := startLocations[i] + uint32(ctxt.Arch.PtrSize) // entry

		// name int32
		nameoff, ok := nameOffsets[s]
		if !ok {
			panic("couldn't find function name offset")
		}
		off = uint32(sb.SetUint32(ctxt.Arch, int64(off), uint32(nameoff)))

		// args int32
		// TODO: Move into funcinfo.
		args := uint32(0)
		if fi.Valid() {
			args = uint32(fi.Args())
		}
		off = uint32(sb.SetUint32(ctxt.Arch, int64(off), args))

		// deferreturn
		deferreturn := computeDeferReturn(ctxt, deferReturnSym, s)
		off = uint32(sb.SetUint32(ctxt.Arch, int64(off), deferreturn))

		// pcdata
		if fi.Valid() {
			off = uint32(sb.SetUint32(ctxt.Arch, int64(off), uint32(ldr.SymValue(fi.Pcsp()))))
			off = uint32(sb.SetUint32(ctxt.Arch, int64(off), uint32(ldr.SymValue(fi.Pcfile()))))
			off = uint32(sb.SetUint32(ctxt.Arch, int64(off), uint32(ldr.SymValue(fi.Pcline()))))
		} else {
			off += 12
		}
		off = uint32(sb.SetUint32(ctxt.Arch, int64(off), uint32(numPCData(fi))))

		// Store the offset to compilation unit's file table.
		cuIdx := ^uint32(0)
		if cu := ldr.SymUnit(s); cu != nil {
			cuIdx = cuOffsets[cu.PclnIndex]
		}
		off = uint32(sb.SetUint32(ctxt.Arch, int64(off), cuIdx))

		// funcID uint8
		var funcID objabi.FuncID
		if fi.Valid() {
			funcID = fi.FuncID()
		}
		off = uint32(sb.SetUint8(ctxt.Arch, int64(off), uint8(funcID)))

		off += 2 // pad

		// nfuncdata must be the final entry.
		funcdata, funcdataoff = funcData(fi, 0, funcdata, funcdataoff)
		off = uint32(sb.SetUint8(ctxt.Arch, int64(off), uint8(len(funcdata))))

		// Output the pcdata.
		if fi.Valid() {
			for j, pcSym := range fi.Pcdata() {
				sb.SetUint32(ctxt.Arch, int64(off+uint32(j*4)), uint32(ldr.SymValue(pcSym)))
			}
			if fi.NumInlTree() > 0 {
				sb.SetUint32(ctxt.Arch, int64(off+objabi.PCDATA_InlTreeIndex*4), uint32(ldr.SymValue(fi.Pcinline())))
			}
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
