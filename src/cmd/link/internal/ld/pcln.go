// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/goobj"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// oldPclnState holds state information used during pclntab generation.  Here
// 'ldr' is just a pointer to the context's loader, 'deferReturnSym' is the
// index for the symbol "runtime.deferreturn", 'nameToOffset' is a helper
// function for capturing function names, 'numberedFiles' records the file
// number assigned to a given file symbol, 'filepaths' is a slice of expanded
// paths (indexed by file number).
//
// NB: This is deprecated, and will be eliminated when pclntab_old is
// eliminated.
type oldPclnState struct {
	ldr            *loader.Loader
	deferReturnSym loader.Sym
	numberedFiles  map[string]int64
	filepaths      []string
}

// pclntab holds the state needed for pclntab generation.
type pclntab struct {
	// The first and last functions found.
	firstFunc, lastFunc loader.Sym

	// The offset to the filetab.
	filetabOffset int32

	// Running total size of pclntab.
	size int64

	// runtime.pclntab's symbols
	carrier     loader.Sym
	pclntab     loader.Sym
	pcheader    loader.Sym
	funcnametab loader.Sym
	findfunctab loader.Sym

	// The number of functions + number of TEXT sections - 1. This is such an
	// unexpected value because platforms that have more than one TEXT section
	// get a dummy function inserted between because the external linker can place
	// functions in those areas. We mark those areas as not covered by the Go
	// runtime.
	//
	// On most platforms this is the number of reachable functions.
	nfunc int32

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
		numberedFiles:  make(map[string]int64),
		// NB: initial entry in filepaths below is to reserve the zero value,
		// so that when we do a map lookup in numberedFiles fails, it will not
		// return a value slot in filepaths.
		filepaths: []string{""},
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

// numberfile assigns a file number to the file if it hasn't been assigned already.
// This funciton looks at a CU's file at index [i], and if it's a new filename,
// stores that filename in the global file table, and adds it to the map lookup
// for renumbering pcfile.
func (state *oldPclnState) numberfile(cu *sym.CompilationUnit, i goobj.CUFileIndex) int64 {
	file := cu.FileTable[i]
	if val, ok := state.numberedFiles[file]; ok {
		return val
	}
	path := file
	if strings.HasPrefix(path, src.FileSymPrefix) {
		path = file[len(src.FileSymPrefix):]
	}
	val := int64(len(state.filepaths))
	state.numberedFiles[file] = val
	state.filepaths = append(state.filepaths, expandGoroot(path))
	return val
}

func (state *oldPclnState) fileVal(cu *sym.CompilationUnit, i int32) int64 {
	file := cu.FileTable[i]
	if val, ok := state.numberedFiles[file]; ok {
		return val
	}
	panic("should have been numbered first")
}

func (state *oldPclnState) renumberfiles(ctxt *Link, cu *sym.CompilationUnit, fi loader.FuncInfo, d *sym.Pcdata) {
	// Give files numbers.
	nf := fi.NumFile()
	for i := uint32(0); i < nf; i++ {
		state.numberfile(cu, fi.File(int(i)))
	}

	buf := make([]byte, binary.MaxVarintLen32)
	newval := int32(-1)
	var out sym.Pcdata
	it := obj.NewPCIter(uint32(ctxt.Arch.MinLC))
	for it.Init(d.P); !it.Done; it.Next() {
		// value delta
		oldval := it.Value

		var val int32
		if oldval == -1 {
			val = -1
		} else {
			if oldval < 0 || oldval >= int32(len(cu.FileTable)) {
				log.Fatalf("bad pcdata %d", oldval)
			}
			val = int32(state.fileVal(cu, oldval))
		}

		dv := val - newval
		newval = val

		// value
		n := binary.PutVarint(buf, int64(dv))
		out.P = append(out.P, buf[:n]...)

		// pc delta
		pc := (it.NextPC - it.PC) / it.PCScale
		n = binary.PutUvarint(buf, uint64(pc))
		out.P = append(out.P, buf[:n]...)
	}

	// terminating value delta
	// we want to write varint-encoded 0, which is just 0
	out.P = append(out.P, 0)

	*d = out
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
		// Usually, call.File is already numbered since the file
		// shows up in the Pcfile table. However, two inlined calls
		// might overlap exactly so that only the innermost file
		// appears in the Pcfile table. In that case, this assigns
		// the outer file a number.
		val := state.numberfile(cu, call.File)
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
		off = writeSymOffset(off, state.funcnametab)
		off = writeSymOffset(off, state.pclntab)
	}

	size := int64(8 + 3*ctxt.Arch.PtrSize)
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

// pclntab initializes the pclntab symbol with
// runtime function and file name information.

// pclntab generates the pcln table for the link output.
func (ctxt *Link) pclntab(container loader.Bitmap) *pclntab {
	// Go 1.2's symtab layout is documented in golang.org/s/go12symtab, but the
	// layout and data has changed since that time.
	//
	// As of July 2020, here's the layout of pclntab:
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
	//         []list of null terminated function names
	//
	//      runtime.pclntab_old
	//        function table, alternating PC and offset to func struct [each entry thearch.ptrsize bytes]
	//        end PC [thearch.ptrsize bytes]
	//        offset to file table [4 bytes]
	//        func structures, pcdata tables.
	//        filetable

	oldState := makeOldPclnState(ctxt)
	state, _ := makePclntab(ctxt, container)

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

	funcdataBytes := int64(0)
	ldr.SetCarrierSym(state.pclntab, state.carrier)
	ldr.SetAttrNotInSymbolTable(state.pclntab, true)
	ftab := ldr.MakeSymbolUpdater(state.pclntab)
	ftab.SetValue(state.size)
	ftab.SetType(sym.SPCLNTAB)
	ftab.SetReachable(true)

	ftab.Grow(int64(state.nfunc)*2*int64(ctxt.Arch.PtrSize) + int64(ctxt.Arch.PtrSize) + 4)

	szHint := len(ctxt.Textp) * 2
	pctaboff := make(map[string]uint32, szHint)
	writepctab := func(off int32, p []byte) int32 {
		start, ok := pctaboff[string(p)]
		if !ok {
			if len(p) > 0 {
				start = uint32(len(ftab.Data()))
				ftab.AddBytes(p)
			}
			pctaboff[string(p)] = start
		}
		newoff := int32(ftab.SetUint32(ctxt.Arch, int64(off), start))
		return newoff
	}

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

	pcsp := sym.Pcdata{}
	pcfile := sym.Pcdata{}
	pcline := sym.Pcdata{}
	pcdata := []sym.Pcdata{}
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

		pcsp.P = pcsp.P[:0]
		pcline.P = pcline.P[:0]
		pcfile.P = pcfile.P[:0]
		pcdata = pcdata[:0]
		funcdataoff = funcdataoff[:0]
		funcdata = funcdata[:0]
		fi := ldr.FuncInfo(s)
		if fi.Valid() {
			fi.Preload()
			npc := fi.NumPcdata()
			for i := uint32(0); i < npc; i++ {
				pcdata = append(pcdata, sym.Pcdata{P: fi.Pcdata(int(i))})
			}
			nfd := fi.NumFuncdataoff()
			for i := uint32(0); i < nfd; i++ {
				funcdataoff = append(funcdataoff, fi.Funcdataoff(int(i)))
			}
			funcdata = fi.Funcdata(funcdata)
		}

		if fi.Valid() && fi.NumInlTree() > 0 {

			if len(pcdata) <= objabi.PCDATA_InlTreeIndex {
				// Create inlining pcdata table.
				newpcdata := make([]sym.Pcdata, objabi.PCDATA_InlTreeIndex+1)
				copy(newpcdata, pcdata)
				pcdata = newpcdata
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

		end := funcstart + int32(ctxt.Arch.PtrSize) + 3*4 + 5*4 + int32(len(pcdata))*4 + int32(len(funcdata))*int32(ctxt.Arch.PtrSize)
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
		if fi.Valid() {
			pcsp = sym.Pcdata{P: fi.Pcsp()}
			pcfile = sym.Pcdata{P: fi.Pcfile()}
			pcline = sym.Pcdata{P: fi.Pcline()}
			oldState.renumberfiles(ctxt, cu, fi, &pcfile)
			if false {
				// Sanity check the new numbering
				it := obj.NewPCIter(uint32(ctxt.Arch.MinLC))
				for it.Init(pcfile.P); !it.Done; it.Next() {
					if it.Value < 1 || it.Value > int32(len(oldState.numberedFiles)) {
						ctxt.Errorf(s, "bad file number in pcfile: %d not in range [1, %d]\n", it.Value, len(oldState.numberedFiles))
						errorexit()
					}
				}
			}
		}

		if fi.Valid() && fi.NumInlTree() > 0 {
			its := oldState.genInlTreeSym(cu, fi, ctxt.Arch, state)
			funcdata[objabi.FUNCDATA_InlTree] = its
			pcdata[objabi.PCDATA_InlTreeIndex] = sym.Pcdata{P: fi.Pcinline()}
		}

		// pcdata
		off = writepctab(off, pcsp.P)
		off = writepctab(off, pcfile.P)
		off = writepctab(off, pcline.P)
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), uint32(len(pcdata))))

		// Store the compilation unit index.
		cuIdx := ^uint16(0)
		if cu := ldr.SymUnit(s); cu != nil {
			if cu.PclnIndex > math.MaxUint16 {
				panic("cu limit reached.")
			}
			cuIdx = uint16(cu.PclnIndex)
		}
		off = int32(ftab.SetUint16(ctxt.Arch, int64(off), cuIdx))

		// funcID uint8
		var funcID objabi.FuncID
		if fi.Valid() {
			funcID = fi.FuncID()
		}
		off = int32(ftab.SetUint8(ctxt.Arch, int64(off), uint8(funcID)))

		// nfuncdata must be the final entry.
		off = int32(ftab.SetUint8(ctxt.Arch, int64(off), uint8(len(funcdata))))
		for i := range pcdata {
			off = writepctab(off, pcdata[i].P)
		}

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
			ctxt.Errorf(s, "bad math in functab: funcstart=%d off=%d but end=%d (npcdata=%d nfuncdata=%d ptrsize=%d)", funcstart, off, end, len(pcdata), len(funcdata), ctxt.Arch.PtrSize)
			errorexit()
		}

		nfunc++
	}

	// Final entry of table is just end pc.
	setAddr(ftab, ctxt.Arch, int64(nfunc)*2*int64(ctxt.Arch.PtrSize), state.lastFunc, ldr.SymSize(state.lastFunc))

	// Start file table.
	dSize := len(ftab.Data())
	start := int32(dSize)
	start += int32(-dSize) & (int32(ctxt.Arch.PtrSize) - 1)
	state.filetabOffset = start
	ftab.SetUint32(ctxt.Arch, int64(nfunc)*2*int64(ctxt.Arch.PtrSize)+int64(ctxt.Arch.PtrSize), uint32(start))

	nf := len(oldState.numberedFiles)
	ftab.Grow(int64(start) + int64((nf+1)*4))
	ftab.SetUint32(ctxt.Arch, int64(start), uint32(nf+1))
	for i := nf; i > 0; i-- {
		path := oldState.filepaths[i]
		val := int64(i)
		ftab.SetUint32(ctxt.Arch, int64(start)+val*4, uint32(ftabaddstring(ftab, path)))
	}

	ftab.SetSize(int64(len(ftab.Data())))

	ctxt.NumFilesyms = len(oldState.numberedFiles)

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
