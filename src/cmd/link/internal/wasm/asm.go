// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"bytes"
	"cmd/internal/obj"
	"cmd/internal/obj/wasm"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/ld"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"internal/abi"
	"internal/buildcfg"
	"io"
	"regexp"
)

const (
	I32  = 0x7F
	I64  = 0x7E
	F32  = 0x7D
	F64  = 0x7C
	V128 = 0x7B

	// contRef is a sentinel value in wasmFuncType.Params and Results standing for
	// the nullable continuation reference (ref null $ContinuationTypeIndex)
	// under GOWASM=stackswitch.
	contRef = 0x00
)

const (
	sectionCustom   = 0
	sectionType     = 1
	sectionImport   = 2
	sectionFunction = 3
	sectionTable    = 4
	sectionMemory   = 5
	sectionTag      = 13
	sectionGlobal   = 6
	sectionExport   = 7
	sectionStart    = 8
	sectionElement  = 9
	sectionCode     = 10
	sectionData     = 11
)

// wasm32State holds function-table state for one linker invocation.
// assignAddress maps each text symbol to its call_indirect handle in funcIndex.
// Handle 0 denotes nil. archreloc and asmb2 use funcIndex for R_WASMFCALL
// relocations. Parallel relocations only read funcIndex.
//
// In asmb2, funcIndexes maps entry PCs to ctxt.Textp indices; direct calls
// add the import count. writeElementSec maps handle h to module function index
// numImports+h-1.
//
// gentext creates runtime.wasmFuncEntry, a uint32 table of entry PCs indexed
// by handle. assignAddress fills it with relocations in PC order; entry 0 is
// zero. At runtime, funcValuePC and funcHandle use this table to convert
// between handles and PCs (see runtime/gobuf_wasm32.go).
type wasm32State struct {
	funcIndex         map[loader.Sym]uint32
	funcEntries       loader.Sym
	unreachableMethod loader.Sym
}

func (st *wasm32State) gentext(ctxt *ld.Link, ldr *loader.Loader) {
	if ctxt.Arch.PtrSize != 4 {
		return
	}

	setWasm32TextAddr(ctxt, ldr)

	st.funcIndex = make(map[loader.Sym]uint32, len(ctxt.Textp))
	abiVersion := sym.SymVerABIInternal
	if !buildcfg.Experiment.RegabiWrappers {
		abiVersion = 0
	}
	st.unreachableMethod = ldr.Lookup("runtime.unreachableMethod", abiVersion)
	entries := ldr.CreateSymForUpdate("runtime.wasmFuncEntry", 0)
	entries.SetType(sym.SRODATA)
	entries.SetAlign(4)
	// Reserve handle 0 for nil. assignAddress fills the remaining slots
	// after text ordering, using relocations to the final function PCs.
	entries.SetSize(int64(len(ctxt.Textp)+1) * 4)
	entries.Grow(entries.Size())
	st.funcEntries = entries.Sym()

	count := ldr.CreateSymForUpdate("runtime.wasmFuncCountData", 0)
	count.SetType(sym.SRODATA)
	count.SetAlign(4)
	count.AddUint32(ctxt.Arch, uint32(len(ctxt.Textp)))

	addHandle := func(name, target string) {
		s := ldr.CreateSymForUpdate(name, 0)
		s.SetType(sym.SRODATA)
		s.SetAlign(4)
		s.AddSymRef(ctxt.Arch, ldr.Lookup(target, 0), 0, objabi.R_WASMFCALL, 4)
	}
	addHandle("runtime.wasmMstartF", "runtime.mstart")
	addHandle("runtime.wasmSystemstackSwitchF", "runtime.systemstack_switch")
	addHandle("runtime.wasmGoexitF", "runtime.goexit")
}

func (st *wasm32State) archreloc(target *ld.Target, ldr *loader.Loader, _ *ld.ArchSyms, r loader.Reloc, s loader.Sym, val int64) (int64, int, bool) {
	if r.Type() != objabi.R_WASMFCALL {
		return val, 0, false
	}
	if target.Arch.PtrSize != 4 {
		ldr.Errorf(s, "R_WASMFCALL used on non-wasm32 target")
		return val, 0, true
	}
	rs := r.Sym()
	if r.Weak() && !ldr.AttrReachable(rs) {
		rs = st.unreachableMethod
	}
	handle, ok := st.funcIndex[rs]
	if !ok {
		ldr.Errorf(s, "R_WASMFCALL target is not a defined WebAssembly function: %s", ldr.SymName(rs))
		return val, 0, true
	}
	return int64(handle), 0, true
}

func extreloc(_ *ld.Target, _ *loader.Loader, r loader.Reloc, _ loader.Sym) (loader.ExtReloc, bool) {
	if r.Type() == objabi.R_WASMFCALL {
		return loader.ExtReloc{}, false
	}
	panic("unexpected external WebAssembly relocation")
}

type wasmFunc struct {
	Module string
	Name   string
	Type   uint32
	Code   []byte
}

type wasmFuncType struct {
	Params  []byte
	Results []byte
	isCont  bool // the entry represent a continuation type.
	contOf  uint32
}

func readWasmImport(ldr *loader.Loader, s loader.Sym) obj.WasmImport {
	var wi obj.WasmImport
	wi.Read(ldr.Data(s))
	return wi
}

// wasmFuncTypes gives the Wasm signatures of functions that do not use the
// Go calling convention. Keep in sync with notUsePC_B and the special cases
// in assemble in cmd/internal/obj/wasm.
var wasmFuncTypes = map[string]*wasmFuncType{
	"_rt0_wasm32_wasip1":      {Params: []byte{}},
	"_rt0_wasm32_wasip1_lib":  {Params: []byte{}},
	"_rt0_wasm_js":            {Params: []byte{}},                                         //
	"_rt0_wasm_wasip1":        {Params: []byte{}},                                         //
	"_rt0_wasm_wasip1_lib":    {Params: []byte{}},                                         //
	"wasm_export__start":      {},                                                         //
	"wasm_export_run":         {Params: []byte{I32, I32}},                                 // argc, argv
	"wasm_export_resume":      {Params: []byte{}},                                         //
	"wasm_export_getsp":       {Results: []byte{I32}},                                     // sp
	"wasm_pc_f_loop":          {Params: []byte{}},                                         //
	"wasm_pc_f_loop_export":   {Params: []byte{I32}},                                      // stop PC_F or logical PC token
	"runtime.wasmDiv":         {Params: []byte{I64, I64}, Results: []byte{I64}},           // x, y -> x/y
	"runtime.wasmTruncS":      {Params: []byte{F64}, Results: []byte{I64}},                // x -> int(x)
	"runtime.wasmTruncU":      {Params: []byte{F64}, Results: []byte{I64}},                // x -> uint(x)
	"gcWriteBarrier":          {Params: []byte{I64}, Results: []byte{I64}},                // #bytes -> bufptr
	"runtime.gcWriteBarrier1": {Results: []byte{I64}},                                     // -> bufptr
	"runtime.gcWriteBarrier2": {Results: []byte{I64}},                                     // -> bufptr
	"runtime.gcWriteBarrier3": {Results: []byte{I64}},                                     // -> bufptr
	"runtime.gcWriteBarrier4": {Results: []byte{I64}},                                     // -> bufptr
	"runtime.gcWriteBarrier5": {Results: []byte{I64}},                                     // -> bufptr
	"runtime.gcWriteBarrier6": {Results: []byte{I64}},                                     // -> bufptr
	"runtime.gcWriteBarrier7": {Results: []byte{I64}},                                     // -> bufptr
	"runtime.gcWriteBarrier8": {Results: []byte{I64}},                                     // -> bufptr
	"runtime.notInitialized":  {},                                                         //
	"cmpbody":                 {Params: []byte{I64, I64, I64, I64}, Results: []byte{I64}}, // a, alen, b, blen -> -1/0/1
	"memeqbody":               {Params: []byte{I64, I64, I64}, Results: []byte{I64}},      // a, b, len -> 0/1
	"memcmp":                  {Params: []byte{I32, I32, I32}, Results: []byte{I32}},      // a, b, len -> <0/0/>0
	"memchr":                  {Params: []byte{I32, I32, I32}, Results: []byte{I32}},      // s, c, len -> index
	"wasm_scheduler_loop":     {Params: []byte{}},                                         //
	"wasm_resume":             {Params: []byte{I32}, Results: []byte{contRef}},            // slot -> cont
	"runtime.wasm_goentry":    {Params: []byte{}},                                         //
	"runtime.wasm_g0entry":    {Params: []byte{}},                                         //
}

// pcBase is the base value for PCs on GOARCH=wasm.
// We set the highest bit, so PCs are distinct from data addresses.
const pcBase = -1 << 63

const (
	wasmPageSize    = 64 << 10
	wasm32AddrLimit = 1 << 32
)

func wasm32FuncSize(size int64) uint64 {
	return uint64(max(size, abi.MINFUNC))
}

func setWasm32TextAddr(ctxt *ld.Link, ldr *loader.Loader) {
	const maxPC = wasm32AddrLimit - 1
	// Include the padding after runtime.etext. Round the start down so
	// linear memory can end on a Wasm page boundary below the PCs.
	span := uint64(ctxt.Arch.MinLC)
	for _, s := range ctxt.Textp {
		size := wasm32FuncSize(ldr.SymSize(s))
		if size > maxPC-span {
			ld.Exitf("wasm32 logical PC space exhausted")
		}
		span += size
	}
	align := uint64(max(wasmPageSize, ld.Funcalign))
	start := (maxPC - span) &^ (align - 1)
	if *ld.FlagTextAddr != -1 {
		start = uint64(ld.Rnd(*ld.FlagTextAddr, int64(ld.Funcalign)))
	}
	if start < wasmPageSize || start > maxPC-span {
		ld.Exitf("wasm32 logical PC range does not fit at %#x", start)
	}
	*ld.FlagTextAddr = int64(start)
}

func nextFuncAddress(arch *sys.Arch, pc uint64, size int64) (uint64, bool) {
	if arch.PtrSize == 8 {
		return pc + 1<<16, true
	}
	next := pc + wasm32FuncSize(size)
	if next < pc || next >= wasm32AddrLimit {
		return 0, false
	}
	return next, true
}

func wasmRelocValue(entry, add int64, size uint8) int64 {
	value := entry + add
	if size == 4 {
		return int64(int32(value))
	}
	return value
}

// wasmPCRelocValue returns the SLEB value for a continuation relocation. A
// wasm32 continuation occupies an 8-byte PC slot: its low word is
// the logical PC token and its high word is the PC_F table handle.
func wasmPCRelocValue(arch *sys.Arch, entry, add int64, size uint8, handle uint32) int64 {
	value := entry + add
	if arch.PtrSize == 4 && size == 8 {
		return int64(uint64(handle)<<32 | uint64(uint32(value)))
	}
	return wasmRelocValue(entry, add, size)
}

func (st *wasm32State) assignAddress(arch *sys.Arch, ldr *loader.Loader, sect *sym.Section, n int, s loader.Sym, va uint64, isTramp bool) (*sym.Section, int, uint64) {
	// WebAssembly functions do not live in the same address space as the linear memory.
	// Instead, WebAssembly assigns function indices: n imported functions have
	// indices [0, n), followed by the functions defined in the module.
	//
	// On GOARCH=wasm, s.Value is 1<<63 + PC_F<<16. On wasm32, s.Value
	// is a logical PC. Each function has SymSize(s) consecutive PC tokens,
	// so s.Value+PC_B identifies a resume point. A separate
	// PC_F identifies the function's call_indirect table entry.
	// See wasm32State for the mappings between these representations.
	//
	// The field "s.Value" corresponds to the concept of PC at runtime.
	// However, there is no PC register, only PC_F and PC_B. PC_F denotes the function,
	// PC_B the resume point inside of that function. The entry of the function has PC_B = 0.
	next, ok := nextFuncAddress(arch, va, ldr.SymSize(s))
	if !ok {
		ldr.Errorf(s, "wasm32 logical PC space exhausted")
		return sect, n, va
	}
	ldr.SetSymSect(s, sect)
	ldr.SetSymValue(s, int64(va))
	if arch.PtrSize == 4 {
		// Address assignment may revisit a function. Preserve its handle
		// and let the entry relocation resolve to its final PC.
		if _, ok := st.funcIndex[s]; !ok {
			handle := uint32(len(st.funcIndex) + 1)
			st.funcIndex[s] = handle
			entries := ldr.MakeSymbolUpdater(st.funcEntries)
			entries.SetAddr(arch, int64(handle)*4, s)
		}
	}
	return sect, n, next
}

type wasmDataSect struct {
	sect *sym.Section
	data []byte
}

var dataSects []wasmDataSect

func asmb(ctxt *ld.Link, ldr *loader.Loader) {
	sections := []*sym.Section{
		ldr.SymSect(ldr.Lookup("runtime.rodata", 0)),
		ldr.SymSect(ldr.Lookup("runtime.types", 0)),
		ldr.SymSect(ldr.Lookup("go:funcdesc", 0)),
		ldr.SymSect(ldr.Lookup("runtime.firstmoduledata", 0)),
		ldr.SymSect(ldr.Lookup("runtime.pclntab", 0)),
		ldr.SymSect(ldr.Lookup("runtime.noptrdata", 0)),
		ldr.SymSect(ldr.Lookup("runtime.data", 0)),
	}

	dataSects = make([]wasmDataSect, len(sections))
	for i, sect := range sections {
		data := ld.DatblkBytes(ctxt, int64(sect.Vaddr), int64(sect.Length))
		dataSects[i] = wasmDataSect{sect, data}
	}
}

// asmb writes the final WebAssembly module binary.
// Spec: https://webassembly.github.io/spec/core/binary/modules.html
func (st *wasm32State) asmb2(ctxt *ld.Link, ldr *loader.Loader) {
	types := []*wasmFuncType{
		// For normal Go functions, the single parameter is PC_B,
		// the return value is
		// 0 if the function returned normally or
		// 1 if the stack needs to be unwound.
		{Params: []byte{I32}, Results: []byte{I32}},
	}

	if buildcfg.GOWASM.StackSwitch {
		initStackSwitchingTypes(&types)
	}

	// collect host imports (functions that get imported from the WebAssembly host, usually JavaScript)
	// we store the import index of each imported function, so the R_WASMIMPORT relocation
	// can write the correct index after a "call" instruction
	// these are added as import statements to the top of the WebAssembly binary
	var hostImports []*wasmFunc
	hostImportMap := make(map[loader.Sym]int64)
	for _, fn := range ctxt.Textp {
		relocs := ldr.Relocs(fn)
		for ri := 0; ri < relocs.Count(); ri++ {
			r := relocs.At(ri)
			if r.Type() == objabi.R_WASMIMPORT {
				if wsym := ldr.WasmImportSym(fn); wsym != 0 {
					wi := readWasmImport(ldr, wsym)
					hostImportMap[fn] = int64(len(hostImports))
					hostImports = append(hostImports, &wasmFunc{
						Module: wi.Module,
						Name:   wi.Name,
						Type: lookupType(&wasmFuncType{
							Params:  fieldsToTypes(wi.Params),
							Results: fieldsToTypes(wi.Results),
						}, &types),
					})
				} else {
					panic(fmt.Sprintf("missing wasm symbol for %s", ldr.SymName(r.Sym())))
				}
			}
		}
	}

	// collect functions with WebAssembly body
	var buildid []byte
	fns := make([]*wasmFunc, len(ctxt.Textp))
	funcIndexes := make(map[int64]uint32, len(ctxt.Textp))
	for i, fn := range ctxt.Textp {
		funcIndexes[ldr.SymValue(fn)] = uint32(i)
	}
	// Resolve aliases through the same PC index used for direct calls.
	funcHandle := func(s loader.Sym) (uint32, bool) {
		if handle, ok := st.funcIndex[s]; ok {
			return handle, true
		}
		if index, ok := funcIndexes[ldr.SymValue(s)]; ok {
			handle, ok := st.funcIndex[ctxt.Textp[index]]
			return handle, ok
		}
		return 0, false
	}
	for i, fn := range ctxt.Textp {
		wfn := new(bytes.Buffer)
		if ldr.SymName(fn) == "go:buildid" {
			writeUleb128(wfn, 0) // number of sets of locals
			writeI32Const(wfn, 0)
			wfn.WriteByte(0x0b) // end
			buildid = ldr.Data(fn)
		} else {
			// Relocations have variable length, handle them here.
			relocs := ldr.Relocs(fn)
			P := ldr.Data(fn)
			off := int32(0)
			for ri := 0; ri < relocs.Count(); ri++ {
				r := relocs.At(ri)
				if r.Siz() == 0 {
					continue // skip marker relocations
				}
				wfn.Write(P[off:r.Off()])
				off = r.Off()
				rs := r.Sym()
				switch r.Type() {
				case objabi.R_ADDR:
					writeSleb128(wfn, wasmRelocValue(ldr.SymValue(rs), r.Add(), r.Siz()))
				case objabi.R_WASMPC:
					var handle uint32
					if ctxt.Arch.PtrSize == 4 {
						var ok bool
						handle, ok = funcHandle(rs)
						if !ok {
							ldr.Errorf(fn, "continuation target is not a defined WebAssembly function: %s", ldr.SymName(rs))
							continue
						}
					}
					writeSleb128(wfn, wasmPCRelocValue(ctxt.Arch, ldr.SymValue(rs), r.Add(), r.Siz(), handle))
				case objabi.R_WASMFCALL:
					handle, ok := funcHandle(rs)
					if !ok {
						ldr.Errorf(fn, "function handle target is not a defined WebAssembly function: %s", ldr.SymName(rs))
						continue
					}
					writeSleb128(wfn, int64(handle))
				case objabi.R_CALL:
					idx, ok := funcIndexes[ldr.SymValue(rs)]
					if !ok {
						ldr.Errorf(fn, "direct call target is not a defined WebAssembly function: %s", ldr.SymName(rs))
						continue
					}
					writeSleb128(wfn, int64(len(hostImports))+int64(idx))
				case objabi.R_WASMIMPORT:
					writeSleb128(wfn, hostImportMap[rs])
				default:
					ldr.Errorf(fn, "bad reloc type %d (%s)", r.Type(), sym.RelocName(ctxt.Arch, r.Type()))
					continue
				}
			}
			wfn.Write(P[off:])
		}

		typ := uint32(0)
		if sig, ok := wasmFuncTypes[ldr.SymName(fn)]; ok {
			typ = lookupType(sig, &types)
		}
		if s := ldr.WasmTypeSym(fn); s != 0 {
			var o obj.WasmFuncType
			o.Read(ldr.Data(s))
			t := &wasmFuncType{
				Params:  fieldsToTypes(o.Params),
				Results: fieldsToTypes(o.Results),
			}
			typ = lookupType(t, &types)
		}

		name := nameRegexp.ReplaceAllString(ldr.SymName(fn), "_")
		fns[i] = &wasmFunc{Name: name, Type: typ, Code: wfn.Bytes()}
	}

	ctxt.Out.Write([]byte{0x00, 0x61, 0x73, 0x6d}) // magic
	ctxt.Out.Write([]byte{0x01, 0x00, 0x00, 0x00}) // version

	// Add any buildid early in the binary:
	if len(buildid) != 0 {
		writeBuildID(ctxt, buildid)
	}

	writeTypeSec(ctxt, types)
	writeImportSec(ctxt, hostImports)
	writeFunctionSec(ctxt, fns)
	tableOffset := uint32(0)
	if ctxt.Arch.PtrSize == 4 {
		tableOffset = 1 // zero is the nil function handle
	}
	writeTableSec(ctxt, fns, tableOffset)
	writeMemorySec(ctxt, ldr)
	writeTagSec(ctxt)
	writeGlobalSec(ctxt)
	writeExportSec(ctxt, ldr, len(hostImports), funcIndexes)
	writeElementSec(ctxt, uint64(len(hostImports)), tableOffset, uint64(len(fns)))
	writeCodeSec(ctxt, fns)
	writeDataSec(ctxt)
	writeProducerSec(ctxt)
	if !*ld.FlagS {
		writeNameSec(ctxt, len(hostImports), fns)
	}
}

func lookupType(sig *wasmFuncType, types *[]*wasmFuncType) uint32 {
	for i, t := range *types {
		if !t.isCont && bytes.Equal(sig.Params, t.Params) && bytes.Equal(sig.Results, t.Results) {
			return uint32(i)
		}
	}
	*types = append(*types, sig)
	return uint32(len(*types) - 1)
}

// contTableInitialSize is the initial capacity of the continuation table.
const contTableInitialSize = 1024

// initStackSwitchingTypes registers the fixed-index WasmFX continuation types
// at the top of types so their indices remain constant across all Go programs.
func initStackSwitchingTypes(types *[]*wasmFuncType) {
	// Force () -> () function type to be at index 1.
	emptyFuncType := lookupType(&wasmFuncType{Params: []byte{}, Results: []byte{}}, types)

	// Continuation type (cont $emptyFuncType) at index 2 (wasm.ContinuationTypeIndex).
	if uint32(len(*types)) != wasm.ContinuationTypeIndex {
		panic("wasm: ContinuationTypeIndex is not at the expected index")
	}
	*types = append(*types, &wasmFuncType{isCont: true, contOf: emptyFuncType})
}

func writeSecHeader(ctxt *ld.Link, id uint8) int64 {
	ctxt.Out.WriteByte(id)
	sizeOffset := ctxt.Out.Offset()
	ctxt.Out.Write(make([]byte, 5)) // placeholder for length
	return sizeOffset
}

func writeSecSize(ctxt *ld.Link, sizeOffset int64) {
	endOffset := ctxt.Out.Offset()
	ctxt.Out.SeekSet(sizeOffset)
	writeUleb128FixedLength(ctxt.Out, uint64(endOffset-sizeOffset-5), 5)
	ctxt.Out.SeekSet(endOffset)
}

func writeBuildID(ctxt *ld.Link, buildid []byte) {
	sizeOffset := writeSecHeader(ctxt, sectionCustom)
	writeName(ctxt.Out, "go:buildid")
	ctxt.Out.Write(buildid)
	writeSecSize(ctxt, sizeOffset)
}

// writeTypeSec writes the section that declares all function types
// (and, under GOWASM=stackswitch, continuation types) so they can be
// referenced by index.
func writeTypeSec(ctxt *ld.Link, types []*wasmFuncType) {
	sizeOffset := writeSecHeader(ctxt, sectionType)

	writeUleb128(ctxt.Out, uint64(len(types)))

	for _, t := range types {
		if t.isCont {
			ctxt.Out.WriteByte(0x5d)                // conttype
			writeSleb128(ctxt.Out, int64(t.contOf)) // underlying functype index (s33)
			continue
		}
		ctxt.Out.WriteByte(0x60) // functype
		writeValTypes(ctxt.Out, t.Params)
		writeValTypes(ctxt.Out, t.Results)
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeContRefType writes the Wasm reference type (ref null $ContinuationTypeIndex).
func writeContRefType(w io.ByteWriter) {
	w.WriteByte(0x63)                           // ref null
	writeSleb128(w, wasm.ContinuationTypeIndex) // heaptype $ContinuationTypeIndex (s33)
}

// writeValTypes writes a Wasm resulttype vector, expanding the contRef sentinel
// into the nullable concrete reference type (ref null $ContinuationTypeIndex).
func writeValTypes(w io.ByteWriter, vs []byte) {
	writeUleb128(w, uint64(len(vs)))
	for _, v := range vs {
		if v == contRef {
			writeContRefType(w)
			continue
		}
		w.WriteByte(v)
	}
}

// writeImportSec writes the section that lists the functions that get
// imported from the WebAssembly host, usually JavaScript.
func writeImportSec(ctxt *ld.Link, hostImports []*wasmFunc) {
	sizeOffset := writeSecHeader(ctxt, sectionImport)

	writeUleb128(ctxt.Out, uint64(len(hostImports))) // number of imports
	for _, fn := range hostImports {
		if fn.Module != "" {
			writeName(ctxt.Out, fn.Module)
		} else {
			writeName(ctxt.Out, wasm.GojsModule) // provided by the import object in wasm_exec.js
		}
		writeName(ctxt.Out, fn.Name)
		ctxt.Out.WriteByte(0x00) // func import
		writeUleb128(ctxt.Out, uint64(fn.Type))
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeFunctionSec writes the section that declares the types of functions.
// The bodies of these functions will later be provided in the "code" section.
func writeFunctionSec(ctxt *ld.Link, fns []*wasmFunc) {
	sizeOffset := writeSecHeader(ctxt, sectionFunction)

	writeUleb128(ctxt.Out, uint64(len(fns)))
	for _, fn := range fns {
		writeUleb128(ctxt.Out, uint64(fn.Type))
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeTableSec writes the section that declares tables. Table 0 holds funcrefs
// used by CallIndirect; under GOWASM=stackswitch, Table 1 holds
// (ref null $ContinuationTypeIndex) entries for suspended goroutine continuations.
func writeTableSec(ctxt *ld.Link, fns []*wasmFunc, tableOffset uint32) {
	sizeOffset := writeSecHeader(ctxt, sectionTable)
	numElements := uint64(len(fns)) + uint64(tableOffset)

	numTables := uint64(1)
	if buildcfg.GOWASM.StackSwitch {
		numTables = 2
	}
	writeUleb128(ctxt.Out, numTables) // number of tables

	ctxt.Out.WriteByte(0x70)            // type: funcref
	ctxt.Out.WriteByte(0x00)            // no max
	writeUleb128(ctxt.Out, numElements) // min

	if buildcfg.GOWASM.StackSwitch {
		// Table 1: nullable continuation references initialized to null.
		writeContRefType(ctxt.Out)
		ctxt.Out.WriteByte(0x00)                     // limits: no max
		writeUleb128(ctxt.Out, contTableInitialSize) // initial capacity
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeMemorySec writes the section that declares linear memories. Currently one linear memory is being used.
// Linear memory always starts at address zero. More memory can be requested with the GrowMemory instruction.
func writeMemorySec(ctxt *ld.Link, ldr *loader.Loader) {
	dataEnd := uint64(ldr.SymValue(ldr.Lookup("runtime.end", 0)))
	initialPages := (dataEnd + 1<<20) / wasmPageSize // 1 MB for runtime init
	sizeOffset := writeSecHeader(ctxt, sectionMemory)

	writeUleb128(ctxt.Out, 1) // number of memories
	if ctxt.Arch.PtrSize == 4 {
		// Keep linear memory below the logical PCs, including when a
		// memory.grow instruction extends it.
		text := uint64(ldr.SymValue(ldr.Lookup("runtime.text", 0)))
		maxPages := text / wasmPageSize
		if initialPages > maxPages {
			ld.Exitf("wasm32 initial linear memory needs %d pages, only %d fit below logical PCs", initialPages, maxPages)
		}
		ctxt.Out.WriteByte(0x01) // minimum and maximum memory size
		writeUleb128(ctxt.Out, initialPages)
		writeUleb128(ctxt.Out, maxPages)
	} else {
		ctxt.Out.WriteByte(0x00) // no maximum memory size
		writeUleb128(ctxt.Out, initialPages)
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeTagSec writes the Wasm Tag section (ID 13) declaring effect tags
// used to implement goroutines with stack-switching instructions (suspend/resume).
func writeTagSec(ctxt *ld.Link) {
	if !buildcfg.GOWASM.StackSwitch {
		return
	}
	sizeOffset := writeSecHeader(ctxt, sectionTag)

	writeUleb128(ctxt.Out, 1) // number of tags
	ctxt.Out.WriteByte(0x00)  // A tag we will call $yield, index 0
	ctxt.Out.WriteByte(0x01)  // the tag's function type (forced at index 1 by initStackSwitchingTypes)

	writeSecSize(ctxt, sizeOffset)
}

// writeGlobalSec writes the section that declares global variables.
func writeGlobalSec(ctxt *ld.Link) {
	sizeOffset := writeSecHeader(ctxt, sectionGlobal)

	globalRegs := []byte{
		I32, // 0: SP
		I64, // 1: CTXT
		I64, // 2: g
		I64, // 3: RET0
		I64, // 4: RET1
		I64, // 5: RET2
		I64, // 6: RET3
		I32, // 7: PAUSE
	}

	writeUleb128(ctxt.Out, uint64(len(globalRegs))) // number of globals

	for _, typ := range globalRegs {
		ctxt.Out.WriteByte(typ)
		ctxt.Out.WriteByte(0x01) // var
		switch typ {
		case I32:
			writeI32Const(ctxt.Out, 0)
		case I64:
			writeI64Const(ctxt.Out, 0)
		}
		ctxt.Out.WriteByte(0x0b) // end
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeExportSec writes the section that declares exports.
// Exports can be accessed by the WebAssembly host, usually JavaScript.
// The wasm_export_* functions and the linear memory get exported.
func writeExportSec(ctxt *ld.Link, ldr *loader.Loader, lenHostImports int, funcIndexes map[int64]uint32) {
	sizeOffset := writeSecHeader(ctxt, sectionExport)
	funcIndex := func(s loader.Sym) uint32 {
		idx, ok := funcIndexes[ldr.SymValue(s)]
		if !ok {
			ldr.Errorf(s, "export is not a defined WebAssembly function")
		}
		return uint32(lenHostImports) + idx
	}

	switch buildcfg.GOOS {
	case "wasip1":
		writeUleb128(ctxt.Out, uint64(2+len(ldr.WasmExports))) // number of exports
		var entry, entryExpName string
		switch ctxt.BuildMode {
		case ld.BuildModeExe:
			entry = "_rt0_" + ctxt.Arch.Name + "_wasip1"
			entryExpName = "_start"
		case ld.BuildModeCShared:
			entry = "_rt0_" + ctxt.Arch.Name + "_wasip1_lib"
			entryExpName = "_initialize"
		}
		s := ldr.Lookup(entry, 0)
		if s == 0 {
			ld.Errorf("export symbol %s not defined", entry)
		}
		idx := funcIndex(s)
		writeName(ctxt.Out, entryExpName)   // the wasi entrypoint
		ctxt.Out.WriteByte(0x00)            // func export
		writeUleb128(ctxt.Out, uint64(idx)) // funcidx
		for _, s := range ldr.WasmExports {
			idx := funcIndex(s)
			writeName(ctxt.Out, ldr.SymName(s))
			ctxt.Out.WriteByte(0x00)            // func export
			writeUleb128(ctxt.Out, uint64(idx)) // funcidx
		}
		writeName(ctxt.Out, "memory") // memory in wasi
		ctxt.Out.WriteByte(0x02)      // mem export
		writeUleb128(ctxt.Out, 0)     // memidx
	case "js":
		writeUleb128(ctxt.Out, uint64(4+len(ldr.WasmExports))) // number of exports
		for _, name := range []string{"run", "resume", "getsp"} {
			s := ldr.Lookup("wasm_export_"+name, 0)
			if s == 0 {
				ld.Errorf("export symbol %s not defined", "wasm_export_"+name)
			}
			idx := funcIndex(s)
			writeName(ctxt.Out, name)           // inst.exports.run/resume/getsp in wasm_exec.js
			ctxt.Out.WriteByte(0x00)            // func export
			writeUleb128(ctxt.Out, uint64(idx)) // funcidx
		}
		for _, s := range ldr.WasmExports {
			idx := funcIndex(s)
			writeName(ctxt.Out, ldr.SymName(s))
			ctxt.Out.WriteByte(0x00)            // func export
			writeUleb128(ctxt.Out, uint64(idx)) // funcidx
		}
		writeName(ctxt.Out, "mem") // inst.exports.mem in wasm_exec.js
		ctxt.Out.WriteByte(0x02)   // mem export
		writeUleb128(ctxt.Out, 0)  // memidx
	default:
		ld.Exitf("internal error: writeExportSec: unrecognized GOOS %s", buildcfg.GOOS)
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeElementSec writes the section that initializes the tables declared by the "table" section.
// The table for CallIndirect gets initialized in a very simple way so that each table index (PC_F value)
// maps linearly to the function index (numImports + PC_F - tableOffset).
func writeElementSec(ctxt *ld.Link, numImports uint64, tableOffset uint32, numFns uint64) {
	sizeOffset := writeSecHeader(ctxt, sectionElement)

	writeUleb128(ctxt.Out, 1) // number of element segments

	writeUleb128(ctxt.Out, 0) // tableidx
	writeI32Const(ctxt.Out, int32(tableOffset))
	ctxt.Out.WriteByte(0x0b) // end

	writeUleb128(ctxt.Out, numFns) // number of entries
	for i := uint64(0); i < numFns; i++ {
		writeUleb128(ctxt.Out, numImports+i)
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeCodeSec writes the section that provides the function bodies for the functions
// declared by the "func" section.
func writeCodeSec(ctxt *ld.Link, fns []*wasmFunc) {
	sizeOffset := writeSecHeader(ctxt, sectionCode)

	writeUleb128(ctxt.Out, uint64(len(fns))) // number of code entries
	for _, fn := range fns {
		writeUleb128(ctxt.Out, uint64(len(fn.Code)))
		ctxt.Out.Write(fn.Code)
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeDataSec writes the section that provides data that will be used to initialize the linear memory.
func writeDataSec(ctxt *ld.Link) {
	sizeOffset := writeSecHeader(ctxt, sectionData)

	type dataSegment struct {
		offset int32
		data   []byte
	}

	// Omit blocks of zeroes and instead emit data segments with offsets skipping the zeroes.
	// This reduces the size of the WebAssembly binary. We use 8 bytes as an estimate for the
	// overhead of adding a new segment (same as wasm-opt's memory-packing optimization uses).
	const segmentOverhead = 8

	// Generate at most this many segments. A higher number of segments gets rejected by some WebAssembly runtimes.
	const maxNumSegments = 100000

	var segments []*dataSegment
	for secIndex, ds := range dataSects {
		data := ds.data
		offset := int32(ds.sect.Vaddr)

		// skip leading zeroes
		for len(data) > 0 && data[0] == 0 {
			data = data[1:]
			offset++
		}

		for len(data) > 0 {
			dataLen := int32(len(data))
			var segmentEnd, zeroEnd int32
			if len(segments)+(len(dataSects)-secIndex) == maxNumSegments {
				segmentEnd = dataLen
				zeroEnd = dataLen
			} else {
				for {
					// look for beginning of zeroes
					for segmentEnd < dataLen && data[segmentEnd] != 0 {
						segmentEnd++
					}
					// look for end of zeroes
					zeroEnd = segmentEnd
					for zeroEnd < dataLen && data[zeroEnd] == 0 {
						zeroEnd++
					}
					// emit segment if omitting zeroes reduces the output size
					if zeroEnd-segmentEnd >= segmentOverhead || zeroEnd == dataLen {
						break
					}
					segmentEnd = zeroEnd
				}
			}

			segments = append(segments, &dataSegment{
				offset: offset,
				data:   data[:segmentEnd],
			})
			data = data[zeroEnd:]
			offset += zeroEnd
		}
	}

	writeUleb128(ctxt.Out, uint64(len(segments))) // number of data entries
	for _, seg := range segments {
		writeUleb128(ctxt.Out, 0) // memidx
		writeI32Const(ctxt.Out, seg.offset)
		ctxt.Out.WriteByte(0x0b) // end
		writeUleb128(ctxt.Out, uint64(len(seg.data)))
		ctxt.Out.Write(seg.data)
	}

	writeSecSize(ctxt, sizeOffset)
}

// writeProducerSec writes an optional section that reports the source language and compiler version.
func writeProducerSec(ctxt *ld.Link) {
	sizeOffset := writeSecHeader(ctxt, sectionCustom)
	writeName(ctxt.Out, "producers")

	writeUleb128(ctxt.Out, 2) // number of fields

	writeName(ctxt.Out, "language")       // field name
	writeUleb128(ctxt.Out, 1)             // number of values
	writeName(ctxt.Out, "Go")             // value: name
	writeName(ctxt.Out, buildcfg.Version) // value: version

	writeName(ctxt.Out, "processed-by")   // field name
	writeUleb128(ctxt.Out, 1)             // number of values
	writeName(ctxt.Out, "Go cmd/compile") // value: name
	writeName(ctxt.Out, buildcfg.Version) // value: version

	writeSecSize(ctxt, sizeOffset)
}

var nameRegexp = regexp.MustCompile(`[^\w.]`)

// writeNameSec writes an optional section that assigns names to the functions declared by the "func" section.
// The names are only used by WebAssembly stack traces, debuggers and decompilers.
// TODO(neelance): add symbol table of DATA symbols
func writeNameSec(ctxt *ld.Link, firstFnIndex int, fns []*wasmFunc) {
	sizeOffset := writeSecHeader(ctxt, sectionCustom)
	writeName(ctxt.Out, "name")

	sizeOffset2 := writeSecHeader(ctxt, 0x01) // function names
	writeUleb128(ctxt.Out, uint64(len(fns)))
	for i, fn := range fns {
		writeUleb128(ctxt.Out, uint64(firstFnIndex+i))
		writeName(ctxt.Out, fn.Name)
	}
	writeSecSize(ctxt, sizeOffset2)

	writeSecSize(ctxt, sizeOffset)
}

type nameWriter interface {
	io.ByteWriter
	io.Writer
}

func writeI32Const(w io.ByteWriter, v int32) {
	w.WriteByte(0x41) // i32.const
	writeSleb128(w, int64(v))
}

func writeI64Const(w io.ByteWriter, v int64) {
	w.WriteByte(0x42) // i64.const
	writeSleb128(w, v)
}

func writeName(w nameWriter, name string) {
	writeUleb128(w, uint64(len(name)))
	w.Write([]byte(name))
}

func writeUleb128(w io.ByteWriter, v uint64) {
	if v < 128 {
		w.WriteByte(uint8(v))
		return
	}
	more := true
	for more {
		c := uint8(v & 0x7f)
		v >>= 7
		more = v != 0
		if more {
			c |= 0x80
		}
		w.WriteByte(c)
	}
}

func writeUleb128FixedLength(w io.ByteWriter, v uint64, length int) {
	for i := 0; i < length; i++ {
		c := uint8(v & 0x7f)
		v >>= 7
		if i < length-1 {
			c |= 0x80
		}
		w.WriteByte(c)
	}
	if v != 0 {
		panic("writeUleb128FixedLength: length too small")
	}
}

func writeSleb128(w io.ByteWriter, v int64) {
	more := true
	for more {
		c := uint8(v & 0x7f)
		s := uint8(v & 0x40)
		v >>= 7
		more = !((v == 0 && s == 0) || (v == -1 && s != 0))
		if more {
			c |= 0x80
		}
		w.WriteByte(c)
	}
}

func fieldsToTypes(fields []obj.WasmField) []byte {
	b := make([]byte, len(fields))
	for i, f := range fields {
		switch f.Type {
		case obj.WasmI32, obj.WasmPtr, obj.WasmBool:
			b[i] = I32
		case obj.WasmI64:
			b[i] = I64
		case obj.WasmF32:
			b[i] = F32
		case obj.WasmF64:
			b[i] = F64
		case obj.WasmV128:
			b[i] = V128
		default:
			panic(fmt.Sprintf("fieldsToTypes: unknown field type: %d", f.Type))
		}
	}
	return b
}
