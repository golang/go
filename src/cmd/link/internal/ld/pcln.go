// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// pclnState holds state information used during pclntab generation.
// Here 'ldr' is just a pointer to the context's loader, 'container'
// is a bitmap holding whether a given symbol index is an outer or
// container symbol, 'deferReturnSym' is the index for the symbol
// "runtime.deferreturn", 'nameToOffset' is a helper function for
// capturing function names, 'numberedFiles' records the file number
// assigned to a given file symbol, 'filepaths' is a slice of
// expanded paths (indexed by file number).
type pclnState struct {
	ldr            *loader.Loader
	container      loader.Bitmap
	deferReturnSym loader.Sym
	nameToOffset   func(name string) int32
	numberedFiles  map[loader.Sym]int64
	filepaths      []string
}

func makepclnState(ctxt *Link) pclnState {
	ldr := ctxt.loader
	drs := ldr.Lookup("runtime.deferreturn", sym.SymVerABIInternal)
	return pclnState{
		container:      loader.MakeBitmap(ldr.NSym()),
		ldr:            ldr,
		deferReturnSym: drs,
		numberedFiles:  make(map[loader.Sym]int64),
		// NB: initial entry in filepaths below is to reserve the zero value,
		// so that when we do a map lookup in numberedFiles fails, it will not
		// return a value slot in filepaths.
		filepaths: []string{""},
	}
}

func (state *pclnState) ftabaddstring(ftab *loader.SymbolBuilder, s string) int32 {
	start := len(ftab.Data())
	ftab.Grow(int64(start + len(s) + 1)) // make room for s plus trailing NUL
	ftd := ftab.Data()
	copy(ftd[start:], s)
	return int32(start)
}

// numberfile assigns a file number to the file if it hasn't been assigned already.
func (state *pclnState) numberfile(file loader.Sym) int64 {
	if val, ok := state.numberedFiles[file]; ok {
		return val
	}
	sn := state.ldr.SymName(file)
	path := sn[len(src.FileSymPrefix):]
	val := int64(len(state.filepaths))
	state.numberedFiles[file] = val
	state.filepaths = append(state.filepaths, expandGoroot(path))
	return val
}

func (state *pclnState) fileVal(file loader.Sym) int64 {
	if val, ok := state.numberedFiles[file]; ok {
		return val
	}
	panic("should have been numbered first")
}

func (state *pclnState) renumberfiles(ctxt *Link, fi loader.FuncInfo, d *sym.Pcdata) {
	// Give files numbers.
	nf := fi.NumFile()
	for i := uint32(0); i < nf; i++ {
		state.numberfile(fi.File(int(i)))
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
			if oldval < 0 || oldval >= int32(nf) {
				log.Fatalf("bad pcdata %d", oldval)
			}
			val = int32(state.fileVal(fi.File(int(oldval))))
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

func (state *pclnState) computeDeferReturn(target *Target, s loader.Sym) uint32 {
	deferreturn := uint32(0)
	lastWasmAddr := uint32(0)

	relocs := state.ldr.Relocs(s)
	for ri := 0; ri < relocs.Count(); ri++ {
		r := relocs.At2(ri)
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
func (state *pclnState) genInlTreeSym(fi loader.FuncInfo, arch *sys.Arch) loader.Sym {
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
		val := state.numberfile(call.File)
		fn := ldr.SymName(call.Func)
		nameoff := state.nameToOffset(fn)

		inlTreeSym.SetUint16(arch, int64(i*20+0), uint16(call.Parent))
		inlTreeSym.SetUint8(arch, int64(i*20+2), uint8(objabi.GetFuncID(fn, "")))
		// byte 3 is unused
		inlTreeSym.SetUint32(arch, int64(i*20+4), uint32(val))
		inlTreeSym.SetUint32(arch, int64(i*20+8), uint32(call.Line))
		inlTreeSym.SetUint32(arch, int64(i*20+12), uint32(nameoff))
		inlTreeSym.SetUint32(arch, int64(i*20+16), uint32(call.ParentPC))
	}
	return its
}

// pclntab initializes the pclntab symbol with
// runtime function and file name information.

// These variables are used to initialize runtime.firstmoduledata, see symtab.go:symtab.
var pclntabNfunc int32
var pclntabFiletabOffset int32
var pclntabPclntabOffset int32
var pclntabFirstFunc loader.Sym
var pclntabLastFunc loader.Sym

// pclntab generates the pcln table for the link output. Return value
// is a bitmap indexed by global symbol that marks 'container' text
// symbols, e.g. the set of all symbols X such that Outer(S) = X for
// some other text symbol S.
func (ctxt *Link) pclntab() loader.Bitmap {
	funcdataBytes := int64(0)
	ldr := ctxt.loader
	ftabsym := ldr.LookupOrCreateSym("runtime.pclntab", 0)
	ftab := ldr.MakeSymbolUpdater(ftabsym)
	ftab.SetType(sym.SPCLNTAB)
	ldr.SetAttrReachable(ftabsym, true)

	state := makepclnState(ctxt)

	// See golang.org/s/go12symtab for the format. Briefly:
	//	8-byte header
	//	nfunc [thearch.ptrsize bytes]
	//	function table, alternating PC and offset to func struct [each entry thearch.ptrsize bytes]
	//	end PC [thearch.ptrsize bytes]
	//	offset to file table [4 bytes]

	// Find container symbols and mark them as such.
	for _, s := range ctxt.Textp2 {
		outer := ldr.OuterSym(s)
		if outer != 0 {
			state.container.Set(outer)
		}
	}

	// Gather some basic stats and info.
	var nfunc int32
	prevSect := ldr.SymSect(ctxt.Textp2[0])
	for _, s := range ctxt.Textp2 {
		if !emitPcln(ctxt, s, state.container) {
			continue
		}
		nfunc++
		if pclntabFirstFunc == 0 {
			pclntabFirstFunc = s
		}
		ss := ldr.SymSect(s)
		if ss != prevSect {
			// With multiple text sections, the external linker may
			// insert functions between the sections, which are not
			// known by Go. This leaves holes in the PC range covered
			// by the func table. We need to generate an entry to mark
			// the hole.
			nfunc++
			prevSect = ss
		}
	}

	pclntabNfunc = nfunc
	ftab.Grow(8 + int64(ctxt.Arch.PtrSize) + int64(nfunc)*2*int64(ctxt.Arch.PtrSize) + int64(ctxt.Arch.PtrSize) + 4)
	ftab.SetUint32(ctxt.Arch, 0, 0xfffffffb)
	ftab.SetUint8(ctxt.Arch, 6, uint8(ctxt.Arch.MinLC))
	ftab.SetUint8(ctxt.Arch, 7, uint8(ctxt.Arch.PtrSize))
	ftab.SetUint(ctxt.Arch, 8, uint64(nfunc))
	pclntabPclntabOffset = int32(8 + ctxt.Arch.PtrSize)

	szHint := len(ctxt.Textp2) * 2
	funcnameoff := make(map[string]int32, szHint)
	nameToOffset := func(name string) int32 {
		nameoff, ok := funcnameoff[name]
		if !ok {
			nameoff = state.ftabaddstring(ftab, name)
			funcnameoff[name] = nameoff
		}
		return nameoff
	}
	state.nameToOffset = nameToOffset

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
	if ctxt.IsExe() && ctxt.IsInternal() && !ctxt.DynlinkingGo() {
		// Internal linking static executable. At this point the function
		// addresses are known, so we can just use them instead of emitting
		// relocations.
		// For other cases we are generating a relocatable binary so we
		// still need to emit relocations.
		//
		// Also not do this optimization when using plugins (DynlinkingGo),
		// as on darwin it does weird things with runtime.etext symbol.
		// TODO: remove the weird thing and remove this condition.
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

	nfunc = 0 // repurpose nfunc as a running index
	prevFunc := ctxt.Textp2[0]
	for _, s := range ctxt.Textp2 {
		if !emitPcln(ctxt, s, state.container) {
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
			setAddr(ftab, ctxt.Arch, 8+int64(ctxt.Arch.PtrSize)+int64(nfunc)*2*int64(ctxt.Arch.PtrSize), prevFunc, prevFuncSize)
			ftab.SetUint(ctxt.Arch, 8+int64(ctxt.Arch.PtrSize)+int64(nfunc)*2*int64(ctxt.Arch.PtrSize)+int64(ctxt.Arch.PtrSize), ^uint64(0))
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

		setAddr(ftab, ctxt.Arch, 8+int64(ctxt.Arch.PtrSize)+int64(nfunc)*2*int64(ctxt.Arch.PtrSize), s, 0)
		ftab.SetUint(ctxt.Arch, 8+int64(ctxt.Arch.PtrSize)+int64(nfunc)*2*int64(ctxt.Arch.PtrSize)+int64(ctxt.Arch.PtrSize), uint64(funcstart))

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
		sn := ldr.SymName(s)
		nameoff := nameToOffset(sn)
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), uint32(nameoff)))

		// args int32
		// TODO: Move into funcinfo.
		args := uint32(0)
		if fi.Valid() {
			args = uint32(fi.Args())
		}
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), args))

		// deferreturn
		deferreturn := state.computeDeferReturn(&ctxt.Target, s)
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), deferreturn))

		if fi.Valid() {
			pcsp = sym.Pcdata{P: fi.Pcsp()}
			pcfile = sym.Pcdata{P: fi.Pcfile()}
			pcline = sym.Pcdata{P: fi.Pcline()}
			state.renumberfiles(ctxt, fi, &pcfile)
			if false {
				// Sanity check the new numbering
				it := obj.NewPCIter(uint32(ctxt.Arch.MinLC))
				for it.Init(pcfile.P); !it.Done; it.Next() {
					if it.Value < 1 || it.Value > int32(len(state.numberedFiles)) {
						ctxt.Errorf(s, "bad file number in pcfile: %d not in range [1, %d]\n", it.Value, len(state.numberedFiles))
						errorexit()
					}
				}
			}
		}

		if fi.Valid() && fi.NumInlTree() > 0 {
			its := state.genInlTreeSym(fi, ctxt.Arch)
			funcdata[objabi.FUNCDATA_InlTree] = its
			pcdata[objabi.PCDATA_InlTreeIndex] = sym.Pcdata{P: fi.Pcinline()}
		}

		// pcdata
		off = writepctab(off, pcsp.P)
		off = writepctab(off, pcfile.P)
		off = writepctab(off, pcline.P)
		off = int32(ftab.SetUint32(ctxt.Arch, int64(off), uint32(len(pcdata))))

		// funcID uint8
		var file string
		if fi.Valid() && fi.NumFile() > 0 {
			filesymname := ldr.SymName(fi.File(0))
			file = filesymname[len(src.FileSymPrefix):]
		}
		funcID := objabi.GetFuncID(sn, file)

		off = int32(ftab.SetUint8(ctxt.Arch, int64(off), uint8(funcID)))

		// unused
		off += 2

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

	last := ctxt.Textp2[len(ctxt.Textp2)-1]
	pclntabLastFunc = last
	// Final entry of table is just end pc.
	setAddr(ftab, ctxt.Arch, 8+int64(ctxt.Arch.PtrSize)+int64(nfunc)*2*int64(ctxt.Arch.PtrSize), last, ldr.SymSize(last))

	// Start file table.
	dSize := len(ftab.Data())
	start := int32(dSize)
	start += int32(-dSize) & (int32(ctxt.Arch.PtrSize) - 1)
	pclntabFiletabOffset = start
	ftab.SetUint32(ctxt.Arch, 8+int64(ctxt.Arch.PtrSize)+int64(nfunc)*2*int64(ctxt.Arch.PtrSize)+int64(ctxt.Arch.PtrSize), uint32(start))

	nf := len(state.numberedFiles)
	ftab.Grow(int64(start) + int64((nf+1)*4))
	ftab.SetUint32(ctxt.Arch, int64(start), uint32(nf+1))
	for i := nf; i > 0; i-- {
		path := state.filepaths[i]
		val := int64(i)
		ftab.SetUint32(ctxt.Arch, int64(start)+val*4, uint32(state.ftabaddstring(ftab, path)))
	}

	ftab.SetSize(int64(len(ftab.Data())))

	ctxt.NumFilesyms = len(state.numberedFiles)

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("pclntab=%d bytes, funcdata total %d bytes\n", ftab.Size(), funcdataBytes)
	}

	return state.container
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
// 'container' is a bitmap indexed by global symbol holding whether
// a given text symbols is a container (outer sym).
func (ctxt *Link) findfunctab(container loader.Bitmap) {
	ldr := ctxt.loader
	tsym := ldr.LookupOrCreateSym("runtime.findfunctab", 0)
	t := ldr.MakeSymbolUpdater(tsym)
	t.SetType(sym.SRODATA)
	ldr.SetAttrReachable(tsym, true)
	ldr.SetAttrLocal(tsym, true)

	// find min and max address
	min := ldr.SymValue(ctxt.Textp2[0])
	lastp := ctxt.Textp2[len(ctxt.Textp2)-1]
	max := ldr.SymValue(lastp) + ldr.SymSize(lastp)

	// for each subbucket, compute the minimum of all symbol indexes
	// that map to that subbucket.
	n := int32((max - min + SUBBUCKETSIZE - 1) / SUBBUCKETSIZE)

	indexes := make([]int32, n)
	for i := int32(0); i < n; i++ {
		indexes[i] = NOIDX
	}
	idx := int32(0)
	for i, s := range ctxt.Textp2 {
		if !emitPcln(ctxt, s, container) {
			continue
		}
		p := ldr.SymValue(s)
		var e loader.Sym
		i++
		if i < len(ctxt.Textp2) {
			e = ctxt.Textp2[i]
		}
		for e != 0 && !emitPcln(ctxt, e, container) && i < len(ctxt.Textp2) {
			e = ctxt.Textp2[i]
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

	// allocate table
	nbuckets := int32((max - min + BUCKETSIZE - 1) / BUCKETSIZE)

	t.Grow(4*int64(nbuckets) + int64(n))

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
