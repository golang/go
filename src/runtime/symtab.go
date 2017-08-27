// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Frames may be used to get function/file/line information for a
// slice of PC values returned by Callers.
type Frames struct {
	// callers is a slice of PCs that have not yet been expanded.
	callers []uintptr

	// stackExpander expands callers into a sequence of Frames,
	// tracking the necessary state across PCs.
	stackExpander stackExpander
}

// Frame is the information returned by Frames for each call frame.
type Frame struct {
	// PC is the program counter for the location in this frame.
	// For a frame that calls another frame, this will be the
	// program counter of a call instruction. Because of inlining,
	// multiple frames may have the same PC value, but different
	// symbolic information.
	PC uintptr

	// Func is the Func value of this call frame. This may be nil
	// for non-Go code or fully inlined functions.
	Func *Func

	// Function is the package path-qualified function name of
	// this call frame. If non-empty, this string uniquely
	// identifies a single function in the program.
	// This may be the empty string if not known.
	// If Func is not nil then Function == Func.Name().
	Function string

	// File and Line are the file name and line number of the
	// location in this frame. For non-leaf frames, this will be
	// the location of a call. These may be the empty string and
	// zero, respectively, if not known.
	File string
	Line int

	// Entry point program counter for the function; may be zero
	// if not known. If Func is not nil then Entry ==
	// Func.Entry().
	Entry uintptr
}

// stackExpander expands a call stack of PCs into a sequence of
// Frames. It tracks state across PCs necessary to perform this
// expansion.
//
// This is the core of the Frames implementation, but is a separate
// internal API to make it possible to use within the runtime without
// heap-allocating the PC slice. The only difference with the public
// Frames API is that the caller is responsible for threading the PC
// slice between expansion steps in this API. If escape analysis were
// smarter, we may not need this (though it may have to be a lot
// smarter).
type stackExpander struct {
	// pcExpander expands the current PC into a sequence of Frames.
	pcExpander pcExpander

	// If previous caller in iteration was a panic, then the next
	// PC in the call stack is the address of the faulting
	// instruction instead of the return address of the call.
	wasPanic bool

	// skip > 0 indicates that skip frames in the expansion of the
	// first PC should be skipped over and callers[1] should also
	// be skipped.
	skip int
}

// CallersFrames takes a slice of PC values returned by Callers and
// prepares to return function/file/line information.
// Do not change the slice until you are done with the Frames.
func CallersFrames(callers []uintptr) *Frames {
	ci := &Frames{}
	ci.callers = ci.stackExpander.init(callers)
	return ci
}

func (se *stackExpander) init(callers []uintptr) []uintptr {
	if len(callers) >= 1 {
		pc := callers[0]
		s := pc - skipPC
		if s >= 0 && s < sizeofSkipFunction {
			// Ignore skip frame callers[0] since this means the caller trimmed the PC slice.
			return callers[1:]
		}
	}
	if len(callers) >= 2 {
		pc := callers[1]
		s := pc - skipPC
		if s > 0 && s < sizeofSkipFunction {
			// Skip the first s inlined frames when we expand the first PC.
			se.skip = int(s)
		}
	}
	return callers
}

// Next returns frame information for the next caller.
// If more is false, there are no more callers (the Frame value is valid).
func (ci *Frames) Next() (frame Frame, more bool) {
	ci.callers, frame, more = ci.stackExpander.next(ci.callers)
	return
}

func (se *stackExpander) next(callers []uintptr) (ncallers []uintptr, frame Frame, more bool) {
	ncallers = callers
	if !se.pcExpander.more {
		// Expand the next PC.
		if len(ncallers) == 0 {
			se.wasPanic = false
			return ncallers, Frame{}, false
		}
		se.pcExpander.init(ncallers[0], se.wasPanic)
		ncallers = ncallers[1:]
		se.wasPanic = se.pcExpander.funcInfo.valid() && se.pcExpander.funcInfo.entry == sigpanicPC
		if se.skip > 0 {
			for ; se.skip > 0; se.skip-- {
				se.pcExpander.next()
			}
			se.skip = 0
			// Drop skipPleaseUseCallersFrames.
			ncallers = ncallers[1:]
		}
		if !se.pcExpander.more {
			// No symbolic information for this PC.
			// However, we return at least one frame for
			// every PC, so return an invalid frame.
			return ncallers, Frame{}, len(ncallers) > 0
		}
	}

	frame = se.pcExpander.next()
	return ncallers, frame, se.pcExpander.more || len(ncallers) > 0
}

// A pcExpander expands a single PC into a sequence of Frames.
type pcExpander struct {
	// more indicates that the next call to next will return a
	// valid frame.
	more bool

	// pc is the pc being expanded.
	pc uintptr

	// frames is a pre-expanded set of Frames to return from the
	// iterator. If this is set, then this is everything that will
	// be returned from the iterator.
	frames []Frame

	// funcInfo is the funcInfo of the function containing pc.
	funcInfo funcInfo

	// inlTree is the inlining tree of the function containing pc.
	inlTree *[1 << 20]inlinedCall

	// file and line are the file name and line number of the next
	// frame.
	file string
	line int32

	// inlIndex is the inlining index of the next frame, or -1 if
	// the next frame is an outermost frame.
	inlIndex int32
}

// init initializes this pcExpander to expand pc. It sets ex.more if
// pc expands to any Frames.
//
// A pcExpander can be reused by calling init again.
//
// If pc was a "call" to sigpanic, panicCall should be true. In this
// case, pc is treated as the address of a faulting instruction
// instead of the return address of a call.
func (ex *pcExpander) init(pc uintptr, panicCall bool) {
	ex.more = false

	ex.funcInfo = findfunc(pc)
	if !ex.funcInfo.valid() {
		if cgoSymbolizer != nil {
			// Pre-expand cgo frames. We could do this
			// incrementally, too, but there's no way to
			// avoid allocation in this case anyway.
			ex.frames = expandCgoFrames(pc)
			ex.more = len(ex.frames) > 0
		}
		return
	}

	ex.more = true
	entry := ex.funcInfo.entry
	ex.pc = pc
	if ex.pc > entry && !panicCall {
		ex.pc--
	}

	// file and line are the innermost position at pc.
	ex.file, ex.line = funcline1(ex.funcInfo, ex.pc, false)

	// Get inlining tree at pc
	inldata := funcdata(ex.funcInfo, _FUNCDATA_InlTree)
	if inldata != nil {
		ex.inlTree = (*[1 << 20]inlinedCall)(inldata)
		ex.inlIndex = pcdatavalue(ex.funcInfo, _PCDATA_InlTreeIndex, ex.pc, nil)
	} else {
		ex.inlTree = nil
		ex.inlIndex = -1
	}
}

// next returns the next Frame in the expansion of pc and sets ex.more
// if there are more Frames to follow.
func (ex *pcExpander) next() Frame {
	if !ex.more {
		return Frame{}
	}

	if len(ex.frames) > 0 {
		// Return pre-expended frame.
		frame := ex.frames[0]
		ex.frames = ex.frames[1:]
		ex.more = len(ex.frames) > 0
		return frame
	}

	if ex.inlIndex >= 0 {
		// Return inner inlined frame.
		call := ex.inlTree[ex.inlIndex]
		frame := Frame{
			PC:       ex.pc,
			Func:     nil, // nil for inlined functions
			Function: funcnameFromNameoff(ex.funcInfo, call.func_),
			File:     ex.file,
			Line:     int(ex.line),
			Entry:    ex.funcInfo.entry,
		}
		ex.file = funcfile(ex.funcInfo, call.file)
		ex.line = call.line
		ex.inlIndex = call.parent
		return frame
	}

	// No inlining or pre-expanded frames.
	ex.more = false
	return Frame{
		PC:       ex.pc,
		Func:     ex.funcInfo._Func(),
		Function: funcname(ex.funcInfo),
		File:     ex.file,
		Line:     int(ex.line),
		Entry:    ex.funcInfo.entry,
	}
}

// expandCgoFrames expands frame information for pc, known to be
// a non-Go function, using the cgoSymbolizer hook. expandCgoFrames
// returns nil if pc could not be expanded.
func expandCgoFrames(pc uintptr) []Frame {
	arg := cgoSymbolizerArg{pc: pc}
	callCgoSymbolizer(&arg)

	if arg.file == nil && arg.funcName == nil {
		// No useful information from symbolizer.
		return nil
	}

	var frames []Frame
	for {
		frames = append(frames, Frame{
			PC:       pc,
			Func:     nil,
			Function: gostring(arg.funcName),
			File:     gostring(arg.file),
			Line:     int(arg.lineno),
			Entry:    arg.entry,
		})
		if arg.more == 0 {
			break
		}
		callCgoSymbolizer(&arg)
	}

	// No more frames for this PC. Tell the symbolizer we are done.
	// We don't try to maintain a single cgoSymbolizerArg for the
	// whole use of Frames, because there would be no good way to tell
	// the symbolizer when we are done.
	arg.pc = 0
	callCgoSymbolizer(&arg)

	return frames
}

// NOTE: Func does not expose the actual unexported fields, because we return *Func
// values to users, and we want to keep them from being able to overwrite the data
// with (say) *f = Func{}.
// All code operating on a *Func must call raw() to get the *_func
// or funcInfo() to get the funcInfo instead.

// A Func represents a Go function in the running binary.
type Func struct {
	opaque struct{} // unexported field to disallow conversions
}

func (f *Func) raw() *_func {
	return (*_func)(unsafe.Pointer(f))
}

func (f *Func) funcInfo() funcInfo {
	fn := f.raw()
	return funcInfo{fn, findmoduledatap(fn.entry)}
}

// PCDATA and FUNCDATA table indexes.
//
// See funcdata.h and ../cmd/internal/obj/funcdata.go.
const (
	_PCDATA_StackMapIndex       = 0
	_PCDATA_InlTreeIndex        = 1
	_FUNCDATA_ArgsPointerMaps   = 0
	_FUNCDATA_LocalsPointerMaps = 1
	_FUNCDATA_InlTree           = 2
	_ArgsSizeUnknown            = -0x80000000
)

// moduledata records information about the layout of the executable
// image. It is written by the linker. Any changes here must be
// matched changes to the code in cmd/internal/ld/symtab.go:symtab.
// moduledata is stored in read-only memory; none of the pointers here
// are visible to the garbage collector.
type moduledata struct {
	pclntable    []byte
	ftab         []functab
	filetab      []uint32
	findfunctab  uintptr
	minpc, maxpc uintptr

	text, etext           uintptr
	noptrdata, enoptrdata uintptr
	data, edata           uintptr
	bss, ebss             uintptr
	noptrbss, enoptrbss   uintptr
	end, gcdata, gcbss    uintptr
	types, etypes         uintptr

	textsectmap []textsect
	typelinks   []int32 // offsets from types
	itablinks   []*itab

	ptab []ptabEntry

	pluginpath string
	pkghashes  []modulehash

	modulename   string
	modulehashes []modulehash

	gcdatamask, gcbssmask bitvector

	typemap map[typeOff]*_type // offset to *_rtype in previous module

	next *moduledata
}

// A modulehash is used to compare the ABI of a new module or a
// package in a new module with the loaded program.
//
// For each shared library a module links against, the linker creates an entry in the
// moduledata.modulehashes slice containing the name of the module, the abi hash seen
// at link time and a pointer to the runtime abi hash. These are checked in
// moduledataverify1 below.
//
// For each loaded plugin, the pkghashes slice has a modulehash of the
// newly loaded package that can be used to check the plugin's version of
// a package against any previously loaded version of the package.
// This is done in plugin.lastmoduleinit.
type modulehash struct {
	modulename   string
	linktimehash string
	runtimehash  *string
}

// pinnedTypemaps are the map[typeOff]*_type from the moduledata objects.
//
// These typemap objects are allocated at run time on the heap, but the
// only direct reference to them is in the moduledata, created by the
// linker and marked SNOPTRDATA so it is ignored by the GC.
//
// To make sure the map isn't collected, we keep a second reference here.
var pinnedTypemaps []map[typeOff]*_type

var firstmoduledata moduledata  // linker symbol
var lastmoduledatap *moduledata // linker symbol
var modulesSlice unsafe.Pointer // see activeModules

// activeModules returns a slice of active modules.
//
// A module is active once its gcdatamask and gcbssmask have been
// assembled and it is usable by the GC.
//
// This is nosplit/nowritebarrier because it is called by the
// cgo pointer checking code.
//go:nosplit
//go:nowritebarrier
func activeModules() []*moduledata {
	p := (*[]*moduledata)(atomic.Loadp(unsafe.Pointer(&modulesSlice)))
	if p == nil {
		return nil
	}
	return *p
}

// modulesinit creates the active modules slice out of all loaded modules.
//
// When a module is first loaded by the dynamic linker, an .init_array
// function (written by cmd/link) is invoked to call addmoduledata,
// appending to the module to the linked list that starts with
// firstmoduledata.
//
// There are two times this can happen in the lifecycle of a Go
// program. First, if compiled with -linkshared, a number of modules
// built with -buildmode=shared can be loaded at program initialization.
// Second, a Go program can load a module while running that was built
// with -buildmode=plugin.
//
// After loading, this function is called which initializes the
// moduledata so it is usable by the GC and creates a new activeModules
// list.
//
// Only one goroutine may call modulesinit at a time.
func modulesinit() {
	modules := new([]*moduledata)
	for md := &firstmoduledata; md != nil; md = md.next {
		*modules = append(*modules, md)
		if md.gcdatamask == (bitvector{}) {
			md.gcdatamask = progToPointerMask((*byte)(unsafe.Pointer(md.gcdata)), md.edata-md.data)
			md.gcbssmask = progToPointerMask((*byte)(unsafe.Pointer(md.gcbss)), md.ebss-md.bss)
		}
	}

	// Modules appear in the moduledata linked list in the order they are
	// loaded by the dynamic loader, with one exception: the
	// firstmoduledata itself the module that contains the runtime. This
	// is not always the first module (when using -buildmode=shared, it
	// is typically libstd.so, the second module). The order matters for
	// typelinksinit, so we swap the first module with whatever module
	// contains the main function.
	//
	// See Issue #18729.
	mainText := funcPC(main_main)
	for i, md := range *modules {
		if md.text <= mainText && mainText <= md.etext {
			(*modules)[0] = md
			(*modules)[i] = &firstmoduledata
			break
		}
	}

	atomicstorep(unsafe.Pointer(&modulesSlice), unsafe.Pointer(modules))
}

type functab struct {
	entry   uintptr
	funcoff uintptr
}

// Mapping information for secondary text sections

type textsect struct {
	vaddr    uintptr // prelinked section vaddr
	length   uintptr // section length
	baseaddr uintptr // relocated section address
}

const minfunc = 16                 // minimum function size
const pcbucketsize = 256 * minfunc // size of bucket in the pc->func lookup table

// findfunctab is an array of these structures.
// Each bucket represents 4096 bytes of the text segment.
// Each subbucket represents 256 bytes of the text segment.
// To find a function given a pc, locate the bucket and subbucket for
// that pc. Add together the idx and subbucket value to obtain a
// function index. Then scan the functab array starting at that
// index to find the target function.
// This table uses 20 bytes for every 4096 bytes of code, or ~0.5% overhead.
type findfuncbucket struct {
	idx        uint32
	subbuckets [16]byte
}

func moduledataverify() {
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		moduledataverify1(datap)
	}
}

const debugPcln = false

func moduledataverify1(datap *moduledata) {
	// See golang.org/s/go12symtab for header: 0xfffffffb,
	// two zero bytes, a byte giving the PC quantum,
	// and a byte giving the pointer width in bytes.
	pcln := *(**[8]byte)(unsafe.Pointer(&datap.pclntable))
	pcln32 := *(**[2]uint32)(unsafe.Pointer(&datap.pclntable))
	if pcln32[0] != 0xfffffffb || pcln[4] != 0 || pcln[5] != 0 || pcln[6] != sys.PCQuantum || pcln[7] != sys.PtrSize {
		println("runtime: function symbol table header:", hex(pcln32[0]), hex(pcln[4]), hex(pcln[5]), hex(pcln[6]), hex(pcln[7]))
		throw("invalid function symbol table\n")
	}

	// ftab is lookup table for function by program counter.
	nftab := len(datap.ftab) - 1
	var pcCache pcvalueCache
	for i := 0; i < nftab; i++ {
		// NOTE: ftab[nftab].entry is legal; it is the address beyond the final function.
		if datap.ftab[i].entry > datap.ftab[i+1].entry {
			f1 := funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[i].funcoff])), datap}
			f2 := funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[i+1].funcoff])), datap}
			f2name := "end"
			if i+1 < nftab {
				f2name = funcname(f2)
			}
			println("function symbol table not sorted by program counter:", hex(datap.ftab[i].entry), funcname(f1), ">", hex(datap.ftab[i+1].entry), f2name)
			for j := 0; j <= i; j++ {
				print("\t", hex(datap.ftab[j].entry), " ", funcname(funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[j].funcoff])), datap}), "\n")
			}
			throw("invalid runtime symbol table")
		}

		if debugPcln || nftab-i < 5 {
			// Check a PC near but not at the very end.
			// The very end might be just padding that is not covered by the tables.
			// No architecture rounds function entries to more than 16 bytes,
			// but if one came along we'd need to subtract more here.
			// But don't use the next PC if it corresponds to a foreign object chunk
			// (no pcln table, f2.pcln == 0). That chunk might have an alignment
			// more than 16 bytes.
			f := funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[i].funcoff])), datap}
			end := f.entry
			if i+1 < nftab {
				f2 := funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[i+1].funcoff])), datap}
				if f2.pcln != 0 {
					end = f2.entry - 16
					if end < f.entry {
						end = f.entry
					}
				}
			}
			pcvalue(f, f.pcfile, end, &pcCache, true)
			pcvalue(f, f.pcln, end, &pcCache, true)
			pcvalue(f, f.pcsp, end, &pcCache, true)
		}
	}

	if datap.minpc != datap.ftab[0].entry ||
		datap.maxpc != datap.ftab[nftab].entry {
		throw("minpc or maxpc invalid")
	}

	for _, modulehash := range datap.modulehashes {
		if modulehash.linktimehash != *modulehash.runtimehash {
			println("abi mismatch detected between", datap.modulename, "and", modulehash.modulename)
			throw("abi mismatch")
		}
	}
}

// FuncForPC returns a *Func describing the function that contains the
// given program counter address, or else nil.
//
// If pc represents multiple functions because of inlining, it returns
// the *Func describing the outermost function.
func FuncForPC(pc uintptr) *Func {
	return findfunc(pc)._Func()
}

// Name returns the name of the function.
func (f *Func) Name() string {
	if f == nil {
		return ""
	}
	return funcname(f.funcInfo())
}

// Entry returns the entry address of the function.
func (f *Func) Entry() uintptr {
	return f.raw().entry
}

// FileLine returns the file name and line number of the
// source code corresponding to the program counter pc.
// The result will not be accurate if pc is not a program
// counter within f.
func (f *Func) FileLine(pc uintptr) (file string, line int) {
	// Pass strict=false here, because anyone can call this function,
	// and they might just be wrong about targetpc belonging to f.
	file, line32 := funcline1(f.funcInfo(), pc, false)
	return file, int(line32)
}

func findmoduledatap(pc uintptr) *moduledata {
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		if datap.minpc <= pc && pc < datap.maxpc {
			return datap
		}
	}
	return nil
}

type funcInfo struct {
	*_func
	datap *moduledata
}

func (f funcInfo) valid() bool {
	return f._func != nil
}

func (f funcInfo) _Func() *Func {
	return (*Func)(unsafe.Pointer(f._func))
}

func findfunc(pc uintptr) funcInfo {
	datap := findmoduledatap(pc)
	if datap == nil {
		return funcInfo{}
	}
	const nsub = uintptr(len(findfuncbucket{}.subbuckets))

	x := pc - datap.minpc
	b := x / pcbucketsize
	i := x % pcbucketsize / (pcbucketsize / nsub)

	ffb := (*findfuncbucket)(add(unsafe.Pointer(datap.findfunctab), b*unsafe.Sizeof(findfuncbucket{})))
	idx := ffb.idx + uint32(ffb.subbuckets[i])

	// If the idx is beyond the end of the ftab, set it to the end of the table and search backward.
	// This situation can occur if multiple text sections are generated to handle large text sections
	// and the linker has inserted jump tables between them.

	if idx >= uint32(len(datap.ftab)) {
		idx = uint32(len(datap.ftab) - 1)
	}
	if pc < datap.ftab[idx].entry {

		// With multiple text sections, the idx might reference a function address that
		// is higher than the pc being searched, so search backward until the matching address is found.

		for datap.ftab[idx].entry > pc && idx > 0 {
			idx--
		}
		if idx == 0 {
			throw("findfunc: bad findfunctab entry idx")
		}
	} else {

		// linear search to find func with pc >= entry.
		for datap.ftab[idx+1].entry <= pc {
			idx++
		}
	}
	return funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[idx].funcoff])), datap}
}

type pcvalueCache struct {
	entries [16]pcvalueCacheEnt
}

type pcvalueCacheEnt struct {
	// targetpc and off together are the key of this cache entry.
	targetpc uintptr
	off      int32
	// val is the value of this cached pcvalue entry.
	val int32
}

func pcvalue(f funcInfo, off int32, targetpc uintptr, cache *pcvalueCache, strict bool) int32 {
	if off == 0 {
		return -1
	}

	// Check the cache. This speeds up walks of deep stacks, which
	// tend to have the same recursive functions over and over.
	//
	// This cache is small enough that full associativity is
	// cheaper than doing the hashing for a less associative
	// cache.
	if cache != nil {
		for i := range cache.entries {
			// We check off first because we're more
			// likely to have multiple entries with
			// different offsets for the same targetpc
			// than the other way around, so we'll usually
			// fail in the first clause.
			ent := &cache.entries[i]
			if ent.off == off && ent.targetpc == targetpc {
				return ent.val
			}
		}
	}

	if !f.valid() {
		if strict && panicking == 0 {
			print("runtime: no module data for ", hex(f.entry), "\n")
			throw("no module data")
		}
		return -1
	}
	datap := f.datap
	p := datap.pclntable[off:]
	pc := f.entry
	val := int32(-1)
	for {
		var ok bool
		p, ok = step(p, &pc, &val, pc == f.entry)
		if !ok {
			break
		}
		if targetpc < pc {
			// Replace a random entry in the cache. Random
			// replacement prevents a performance cliff if
			// a recursive stack's cycle is slightly
			// larger than the cache.
			if cache != nil {
				ci := fastrandn(uint32(len(cache.entries)))
				cache.entries[ci] = pcvalueCacheEnt{
					targetpc: targetpc,
					off:      off,
					val:      val,
				}
			}

			return val
		}
	}

	// If there was a table, it should have covered all program counters.
	// If not, something is wrong.
	if panicking != 0 || !strict {
		return -1
	}

	print("runtime: invalid pc-encoded table f=", funcname(f), " pc=", hex(pc), " targetpc=", hex(targetpc), " tab=", p, "\n")

	p = datap.pclntable[off:]
	pc = f.entry
	val = -1
	for {
		var ok bool
		p, ok = step(p, &pc, &val, pc == f.entry)
		if !ok {
			break
		}
		print("\tvalue=", val, " until pc=", hex(pc), "\n")
	}

	throw("invalid runtime symbol table")
	return -1
}

func cfuncname(f funcInfo) *byte {
	if !f.valid() || f.nameoff == 0 {
		return nil
	}
	return &f.datap.pclntable[f.nameoff]
}

func funcname(f funcInfo) string {
	return gostringnocopy(cfuncname(f))
}

func funcnameFromNameoff(f funcInfo, nameoff int32) string {
	datap := f.datap
	if !f.valid() {
		return ""
	}
	cstr := &datap.pclntable[nameoff]
	return gostringnocopy(cstr)
}

func funcfile(f funcInfo, fileno int32) string {
	datap := f.datap
	if !f.valid() {
		return "?"
	}
	return gostringnocopy(&datap.pclntable[datap.filetab[fileno]])
}

func funcline1(f funcInfo, targetpc uintptr, strict bool) (file string, line int32) {
	datap := f.datap
	if !f.valid() {
		return "?", 0
	}
	fileno := int(pcvalue(f, f.pcfile, targetpc, nil, strict))
	line = pcvalue(f, f.pcln, targetpc, nil, strict)
	if fileno == -1 || line == -1 || fileno >= len(datap.filetab) {
		// print("looking for ", hex(targetpc), " in ", funcname(f), " got file=", fileno, " line=", lineno, "\n")
		return "?", 0
	}
	file = gostringnocopy(&datap.pclntable[datap.filetab[fileno]])
	return
}

func funcline(f funcInfo, targetpc uintptr) (file string, line int32) {
	return funcline1(f, targetpc, true)
}

func funcspdelta(f funcInfo, targetpc uintptr, cache *pcvalueCache) int32 {
	x := pcvalue(f, f.pcsp, targetpc, cache, true)
	if x&(sys.PtrSize-1) != 0 {
		print("invalid spdelta ", funcname(f), " ", hex(f.entry), " ", hex(targetpc), " ", hex(f.pcsp), " ", x, "\n")
	}
	return x
}

func pcdatavalue(f funcInfo, table int32, targetpc uintptr, cache *pcvalueCache) int32 {
	if table < 0 || table >= f.npcdata {
		return -1
	}
	off := *(*int32)(add(unsafe.Pointer(&f.nfuncdata), unsafe.Sizeof(f.nfuncdata)+uintptr(table)*4))
	return pcvalue(f, off, targetpc, cache, true)
}

func funcdata(f funcInfo, i int32) unsafe.Pointer {
	if i < 0 || i >= f.nfuncdata {
		return nil
	}
	p := add(unsafe.Pointer(&f.nfuncdata), unsafe.Sizeof(f.nfuncdata)+uintptr(f.npcdata)*4)
	if sys.PtrSize == 8 && uintptr(p)&4 != 0 {
		if uintptr(unsafe.Pointer(f._func))&4 != 0 {
			println("runtime: misaligned func", f._func)
		}
		p = add(p, 4)
	}
	return *(*unsafe.Pointer)(add(p, uintptr(i)*sys.PtrSize))
}

// step advances to the next pc, value pair in the encoded table.
func step(p []byte, pc *uintptr, val *int32, first bool) (newp []byte, ok bool) {
	// For both uvdelta and pcdelta, the common case (~70%)
	// is that they are a single byte. If so, avoid calling readvarint.
	uvdelta := uint32(p[0])
	if uvdelta == 0 && !first {
		return nil, false
	}
	n := uint32(1)
	if uvdelta&0x80 != 0 {
		n, uvdelta = readvarint(p)
	}
	p = p[n:]
	if uvdelta&1 != 0 {
		uvdelta = ^(uvdelta >> 1)
	} else {
		uvdelta >>= 1
	}
	vdelta := int32(uvdelta)
	pcdelta := uint32(p[0])
	n = 1
	if pcdelta&0x80 != 0 {
		n, pcdelta = readvarint(p)
	}
	p = p[n:]
	*pc += uintptr(pcdelta * sys.PCQuantum)
	*val += vdelta
	return p, true
}

// readvarint reads a varint from p.
func readvarint(p []byte) (read uint32, val uint32) {
	var v, shift, n uint32
	for {
		b := p[n]
		n++
		v |= uint32(b&0x7F) << (shift & 31)
		if b&0x80 == 0 {
			break
		}
		shift += 7
	}
	return n, v
}

type stackmap struct {
	n        int32   // number of bitmaps
	nbit     int32   // number of bits in each bitmap
	bytedata [1]byte // bitmaps, each starting on a byte boundary
}

//go:nowritebarrier
func stackmapdata(stkmap *stackmap, n int32) bitvector {
	if n < 0 || n >= stkmap.n {
		throw("stackmapdata: index out of range")
	}
	return bitvector{stkmap.nbit, (*byte)(add(unsafe.Pointer(&stkmap.bytedata), uintptr(n*((stkmap.nbit+7)>>3))))}
}

// inlinedCall is the encoding of entries in the FUNCDATA_InlTree table.
type inlinedCall struct {
	parent int32 // index of parent in the inltree, or < 0
	file   int32 // fileno index into filetab
	line   int32 // line number of the call site
	func_  int32 // offset into pclntab for name of called function
}
