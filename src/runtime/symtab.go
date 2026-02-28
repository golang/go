// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goarch"
	"runtime/internal/atomic"
	"runtime/internal/sys"
	"unsafe"
)

// Frames may be used to get function/file/line information for a
// slice of PC values returned by Callers.
type Frames struct {
	// callers is a slice of PCs that have not yet been expanded to frames.
	callers []uintptr

	// frames is a slice of Frames that have yet to be returned.
	frames     []Frame
	frameStore [2]Frame
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

	// The runtime's internal view of the function. This field
	// is set (funcInfo.valid() returns true) only for Go functions,
	// not for C functions.
	funcInfo funcInfo
}

// CallersFrames takes a slice of PC values returned by Callers and
// prepares to return function/file/line information.
// Do not change the slice until you are done with the Frames.
func CallersFrames(callers []uintptr) *Frames {
	f := &Frames{callers: callers}
	f.frames = f.frameStore[:0]
	return f
}

// Next returns a Frame representing the next call frame in the slice
// of PC values. If it has already returned all call frames, Next
// returns a zero Frame.
//
// The more result indicates whether the next call to Next will return
// a valid Frame. It does not necessarily indicate whether this call
// returned one.
//
// See the Frames example for idiomatic usage.
func (ci *Frames) Next() (frame Frame, more bool) {
	for len(ci.frames) < 2 {
		// Find the next frame.
		// We need to look for 2 frames so we know what
		// to return for the "more" result.
		if len(ci.callers) == 0 {
			break
		}
		pc := ci.callers[0]
		ci.callers = ci.callers[1:]
		funcInfo := findfunc(pc)
		if !funcInfo.valid() {
			if cgoSymbolizer != nil {
				// Pre-expand cgo frames. We could do this
				// incrementally, too, but there's no way to
				// avoid allocation in this case anyway.
				ci.frames = append(ci.frames, expandCgoFrames(pc)...)
			}
			continue
		}
		f := funcInfo._Func()
		entry := f.Entry()
		if pc > entry {
			// We store the pc of the start of the instruction following
			// the instruction in question (the call or the inline mark).
			// This is done for historical reasons, and to make FuncForPC
			// work correctly for entries in the result of runtime.Callers.
			pc--
		}
		name := funcname(funcInfo)
		if inldata := funcdata(funcInfo, _FUNCDATA_InlTree); inldata != nil {
			inltree := (*[1 << 20]inlinedCall)(inldata)
			// Non-strict as cgoTraceback may have added bogus PCs
			// with a valid funcInfo but invalid PCDATA.
			ix := pcdatavalue1(funcInfo, _PCDATA_InlTreeIndex, pc, nil, false)
			if ix >= 0 {
				// Note: entry is not modified. It always refers to a real frame, not an inlined one.
				f = nil
				name = funcnameFromNameoff(funcInfo, inltree[ix].func_)
				// File/line is already correct.
				// TODO: remove file/line from InlinedCall?
			}
		}
		ci.frames = append(ci.frames, Frame{
			PC:       pc,
			Func:     f,
			Function: name,
			Entry:    entry,
			funcInfo: funcInfo,
			// Note: File,Line set below
		})
	}

	// Pop one frame from the frame list. Keep the rest.
	// Avoid allocation in the common case, which is 1 or 2 frames.
	switch len(ci.frames) {
	case 0: // In the rare case when there are no frames at all, we return Frame{}.
		return
	case 1:
		frame = ci.frames[0]
		ci.frames = ci.frameStore[:0]
	case 2:
		frame = ci.frames[0]
		ci.frameStore[0] = ci.frames[1]
		ci.frames = ci.frameStore[:1]
	default:
		frame = ci.frames[0]
		ci.frames = ci.frames[1:]
	}
	more = len(ci.frames) > 0
	if frame.funcInfo.valid() {
		// Compute file/line just before we need to return it,
		// as it can be expensive. This avoids computing file/line
		// for the Frame we find but don't return. See issue 32093.
		file, line := funcline1(frame.funcInfo, frame.PC, false)
		frame.File, frame.Line = file, int(line)
	}
	return
}

// runtime_expandFinalInlineFrame expands the final pc in stk to include all
// "callers" if pc is inline.
//
//go:linkname runtime_expandFinalInlineFrame runtime/pprof.runtime_expandFinalInlineFrame
func runtime_expandFinalInlineFrame(stk []uintptr) []uintptr {
	if len(stk) == 0 {
		return stk
	}
	pc := stk[len(stk)-1]
	tracepc := pc - 1

	f := findfunc(tracepc)
	if !f.valid() {
		// Not a Go function.
		return stk
	}

	inldata := funcdata(f, _FUNCDATA_InlTree)
	if inldata == nil {
		// Nothing inline in f.
		return stk
	}

	// Treat the previous func as normal. We haven't actually checked, but
	// since this pc was included in the stack, we know it shouldn't be
	// elided.
	lastFuncID := funcID_normal

	// Remove pc from stk; we'll re-add it below.
	stk = stk[:len(stk)-1]

	// See inline expansion in gentraceback.
	var cache pcvalueCache
	inltree := (*[1 << 20]inlinedCall)(inldata)
	for {
		// Non-strict as cgoTraceback may have added bogus PCs
		// with a valid funcInfo but invalid PCDATA.
		ix := pcdatavalue1(f, _PCDATA_InlTreeIndex, tracepc, &cache, false)
		if ix < 0 {
			break
		}
		if inltree[ix].funcID == funcID_wrapper && elideWrapperCalling(lastFuncID) {
			// ignore wrappers
		} else {
			stk = append(stk, pc)
		}
		lastFuncID = inltree[ix].funcID
		// Back up to an instruction in the "caller".
		tracepc = f.entry() + uintptr(inltree[ix].parentPc)
		pc = tracepc + 1
	}

	// N.B. we want to keep the last parentPC which is not inline.
	stk = append(stk, pc)

	return stk
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
			// funcInfo is zero, which implies !funcInfo.valid().
			// That ensures that we use the File/Line info given here.
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
	return f.raw().funcInfo()
}

func (f *_func) funcInfo() funcInfo {
	// Find the module containing fn. fn is located in the pclntable.
	// The unsafe.Pointer to uintptr conversions and arithmetic
	// are safe because we are working with module addresses.
	ptr := uintptr(unsafe.Pointer(f))
	var mod *moduledata
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		if len(datap.pclntable) == 0 {
			continue
		}
		base := uintptr(unsafe.Pointer(&datap.pclntable[0]))
		if base <= ptr && ptr < base+uintptr(len(datap.pclntable)) {
			mod = datap
			break
		}
	}
	return funcInfo{f, mod}
}

// PCDATA and FUNCDATA table indexes.
//
// See funcdata.h and ../cmd/internal/objabi/funcdata.go.
const (
	_PCDATA_UnsafePoint   = 0
	_PCDATA_StackMapIndex = 1
	_PCDATA_InlTreeIndex  = 2
	_PCDATA_ArgLiveIndex  = 3

	_FUNCDATA_ArgsPointerMaps    = 0
	_FUNCDATA_LocalsPointerMaps  = 1
	_FUNCDATA_StackObjects       = 2
	_FUNCDATA_InlTree            = 3
	_FUNCDATA_OpenCodedDeferInfo = 4
	_FUNCDATA_ArgInfo            = 5
	_FUNCDATA_ArgLiveInfo        = 6
	_FUNCDATA_WrapInfo           = 7

	_ArgsSizeUnknown = -0x80000000
)

const (
	// PCDATA_UnsafePoint values.
	_PCDATA_UnsafePointSafe   = -1 // Safe for async preemption
	_PCDATA_UnsafePointUnsafe = -2 // Unsafe for async preemption

	// _PCDATA_Restart1(2) apply on a sequence of instructions, within
	// which if an async preemption happens, we should back off the PC
	// to the start of the sequence when resume.
	// We need two so we can distinguish the start/end of the sequence
	// in case that two sequences are next to each other.
	_PCDATA_Restart1 = -3
	_PCDATA_Restart2 = -4

	// Like _PCDATA_RestartAtEntry, but back to function entry if async
	// preempted.
	_PCDATA_RestartAtEntry = -5
)

// A FuncID identifies particular functions that need to be treated
// specially by the runtime.
// Note that in some situations involving plugins, there may be multiple
// copies of a particular special runtime function.
// Note: this list must match the list in cmd/internal/objabi/funcid.go.
type funcID uint8

const (
	funcID_normal funcID = iota // not a special function
	funcID_abort
	funcID_asmcgocall
	funcID_asyncPreempt
	funcID_cgocallback
	funcID_debugCallV2
	funcID_gcBgMarkWorker
	funcID_goexit
	funcID_gogo
	funcID_gopanic
	funcID_handleAsyncEvent
	funcID_mcall
	funcID_morestack
	funcID_mstart
	funcID_panicwrap
	funcID_rt0_go
	funcID_runfinq
	funcID_runtime_main
	funcID_sigpanic
	funcID_systemstack
	funcID_systemstack_switch
	funcID_wrapper // any autogenerated code (hash/eq algorithms, method wrappers, etc.)
)

// A FuncFlag holds bits about a function.
// This list must match the list in cmd/internal/objabi/funcid.go.
type funcFlag uint8

const (
	// TOPFRAME indicates a function that appears at the top of its stack.
	// The traceback routine stop at such a function and consider that a
	// successful, complete traversal of the stack.
	// Examples of TOPFRAME functions include goexit, which appears
	// at the top of a user goroutine stack, and mstart, which appears
	// at the top of a system goroutine stack.
	funcFlag_TOPFRAME funcFlag = 1 << iota

	// SPWRITE indicates a function that writes an arbitrary value to SP
	// (any write other than adding or subtracting a constant amount).
	// The traceback routines cannot encode such changes into the
	// pcsp tables, so the function traceback cannot safely unwind past
	// SPWRITE functions. Stopping at an SPWRITE function is considered
	// to be an incomplete unwinding of the stack. In certain contexts
	// (in particular garbage collector stack scans) that is a fatal error.
	funcFlag_SPWRITE

	// ASM indicates that a function was implemented in assembly.
	funcFlag_ASM
)

// pcHeader holds data used by the pclntab lookups.
type pcHeader struct {
	magic          uint32  // 0xFFFFFFF0
	pad1, pad2     uint8   // 0,0
	minLC          uint8   // min instruction size
	ptrSize        uint8   // size of a ptr in bytes
	nfunc          int     // number of functions in the module
	nfiles         uint    // number of entries in the file tab
	textStart      uintptr // base for function entry PC offsets in this module, equal to moduledata.text
	funcnameOffset uintptr // offset to the funcnametab variable from pcHeader
	cuOffset       uintptr // offset to the cutab variable from pcHeader
	filetabOffset  uintptr // offset to the filetab variable from pcHeader
	pctabOffset    uintptr // offset to the pctab variable from pcHeader
	pclnOffset     uintptr // offset to the pclntab variable from pcHeader
}

// moduledata records information about the layout of the executable
// image. It is written by the linker. Any changes here must be
// matched changes to the code in cmd/link/internal/ld/symtab.go:symtab.
// moduledata is stored in statically allocated non-pointer memory;
// none of the pointers here are visible to the garbage collector.
type moduledata struct {
	pcHeader     *pcHeader
	funcnametab  []byte
	cutab        []uint32
	filetab      []byte
	pctab        []byte
	pclntable    []byte
	ftab         []functab
	findfunctab  uintptr
	minpc, maxpc uintptr

	text, etext           uintptr
	noptrdata, enoptrdata uintptr
	data, edata           uintptr
	bss, ebss             uintptr
	noptrbss, enoptrbss   uintptr
	end, gcdata, gcbss    uintptr
	types, etypes         uintptr
	rodata                uintptr
	gofunc                uintptr // go.func.*

	textsectmap []textsect
	typelinks   []int32 // offsets from types
	itablinks   []*itab

	ptab []ptabEntry

	pluginpath string
	pkghashes  []modulehash

	modulename   string
	modulehashes []modulehash

	hasmain uint8 // 1 if module contains the main function, 0 otherwise

	gcdatamask, gcbssmask bitvector

	typemap map[typeOff]*_type // offset to *_rtype in previous module

	bad bool // module failed to load and should be ignored

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
var modulesSlice *[]*moduledata // see activeModules

// activeModules returns a slice of active modules.
//
// A module is active once its gcdatamask and gcbssmask have been
// assembled and it is usable by the GC.
//
// This is nosplit/nowritebarrier because it is called by the
// cgo pointer checking code.
//
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
		if md.bad {
			continue
		}
		*modules = append(*modules, md)
		if md.gcdatamask == (bitvector{}) {
			scanDataSize := md.edata - md.data
			md.gcdatamask = progToPointerMask((*byte)(unsafe.Pointer(md.gcdata)), scanDataSize)
			scanBSSSize := md.ebss - md.bss
			md.gcbssmask = progToPointerMask((*byte)(unsafe.Pointer(md.gcbss)), scanBSSSize)
			gcController.addGlobals(int64(scanDataSize + scanBSSSize))
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
	for i, md := range *modules {
		if md.hasmain != 0 {
			(*modules)[0] = md
			(*modules)[i] = &firstmoduledata
			break
		}
	}

	atomicstorep(unsafe.Pointer(&modulesSlice), unsafe.Pointer(modules))
}

type functab struct {
	entryoff uint32 // relative to runtime.text
	funcoff  uint32
}

// Mapping information for secondary text sections

type textsect struct {
	vaddr    uintptr // prelinked section vaddr
	end      uintptr // vaddr + section length
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
	// Check that the pclntab's format is valid.
	hdr := datap.pcHeader
	if hdr.magic != 0xfffffff0 || hdr.pad1 != 0 || hdr.pad2 != 0 ||
		hdr.minLC != sys.PCQuantum || hdr.ptrSize != goarch.PtrSize || hdr.textStart != datap.text {
		println("runtime: pcHeader: magic=", hex(hdr.magic), "pad1=", hdr.pad1, "pad2=", hdr.pad2,
			"minLC=", hdr.minLC, "ptrSize=", hdr.ptrSize, "pcHeader.textStart=", hex(hdr.textStart),
			"text=", hex(datap.text), "pluginpath=", datap.pluginpath)
		throw("invalid function symbol table")
	}

	// ftab is lookup table for function by program counter.
	nftab := len(datap.ftab) - 1
	for i := 0; i < nftab; i++ {
		// NOTE: ftab[nftab].entry is legal; it is the address beyond the final function.
		if datap.ftab[i].entryoff > datap.ftab[i+1].entryoff {
			f1 := funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[i].funcoff])), datap}
			f2 := funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[i+1].funcoff])), datap}
			f2name := "end"
			if i+1 < nftab {
				f2name = funcname(f2)
			}
			println("function symbol table not sorted by PC offset:", hex(datap.ftab[i].entryoff), funcname(f1), ">", hex(datap.ftab[i+1].entryoff), f2name, ", plugin:", datap.pluginpath)
			for j := 0; j <= i; j++ {
				println("\t", hex(datap.ftab[j].entryoff), funcname(funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[datap.ftab[j].funcoff])), datap}))
			}
			if GOOS == "aix" && isarchive {
				println("-Wl,-bnoobjreorder is mandatory on aix/ppc64 with c-archive")
			}
			throw("invalid runtime symbol table")
		}
	}

	min := datap.textAddr(datap.ftab[0].entryoff)
	max := datap.textAddr(datap.ftab[nftab].entryoff)
	if datap.minpc != min || datap.maxpc != max {
		println("minpc=", hex(datap.minpc), "min=", hex(min), "maxpc=", hex(datap.maxpc), "max=", hex(max))
		throw("minpc or maxpc invalid")
	}

	for _, modulehash := range datap.modulehashes {
		if modulehash.linktimehash != *modulehash.runtimehash {
			println("abi mismatch detected between", datap.modulename, "and", modulehash.modulename)
			throw("abi mismatch")
		}
	}
}

// textAddr returns md.text + off, with special handling for multiple text sections.
// off is a (virtual) offset computed at internal linking time,
// before the external linker adjusts the sections' base addresses.
//
// The text, or instruction stream is generated as one large buffer.
// The off (offset) for a function is its offset within this buffer.
// If the total text size gets too large, there can be issues on platforms like ppc64
// if the target of calls are too far for the call instruction.
// To resolve the large text issue, the text is split into multiple text sections
// to allow the linker to generate long calls when necessary.
// When this happens, the vaddr for each text section is set to its offset within the text.
// Each function's offset is compared against the section vaddrs and ends to determine the containing section.
// Then the section relative offset is added to the section's
// relocated baseaddr to compute the function address.
//
// It is nosplit because it is part of the findfunc implementation.
//
//go:nosplit
func (md *moduledata) textAddr(off32 uint32) uintptr {
	off := uintptr(off32)
	res := md.text + off
	if len(md.textsectmap) > 1 {
		for i, sect := range md.textsectmap {
			// For the last section, include the end address (etext), as it is included in the functab.
			if off >= sect.vaddr && off < sect.end || (i == len(md.textsectmap)-1 && off == sect.end) {
				res = sect.baseaddr + off - sect.vaddr
				break
			}
		}
		if res > md.etext && GOARCH != "wasm" { // on wasm, functions do not live in the same address space as the linear memory
			println("runtime: textAddr", hex(res), "out of range", hex(md.text), "-", hex(md.etext))
			throw("runtime: text offset out of range")
		}
	}
	return res
}

// textOff is the opposite of textAddr. It converts a PC to a (virtual) offset
// to md.text, and returns if the PC is in any Go text section.
//
// It is nosplit because it is part of the findfunc implementation.
//
//go:nosplit
func (md *moduledata) textOff(pc uintptr) (uint32, bool) {
	res := uint32(pc - md.text)
	if len(md.textsectmap) > 1 {
		for i, sect := range md.textsectmap {
			if sect.baseaddr > pc {
				// pc is not in any section.
				return 0, false
			}
			end := sect.baseaddr + (sect.end - sect.vaddr)
			// For the last section, include the end address (etext), as it is included in the functab.
			if i == len(md.textsectmap) {
				end++
			}
			if pc < end {
				res = uint32(pc - sect.baseaddr + sect.vaddr)
				break
			}
		}
	}
	return res, true
}

// FuncForPC returns a *Func describing the function that contains the
// given program counter address, or else nil.
//
// If pc represents multiple functions because of inlining, it returns
// the *Func describing the innermost function, but with an entry of
// the outermost function.
func FuncForPC(pc uintptr) *Func {
	f := findfunc(pc)
	if !f.valid() {
		return nil
	}
	if inldata := funcdata(f, _FUNCDATA_InlTree); inldata != nil {
		// Note: strict=false so bad PCs (those between functions) don't crash the runtime.
		// We just report the preceding function in that situation. See issue 29735.
		// TODO: Perhaps we should report no function at all in that case.
		// The runtime currently doesn't have function end info, alas.
		if ix := pcdatavalue1(f, _PCDATA_InlTreeIndex, pc, nil, false); ix >= 0 {
			inltree := (*[1 << 20]inlinedCall)(inldata)
			name := funcnameFromNameoff(f, inltree[ix].func_)
			file, line := funcline(f, pc)
			fi := &funcinl{
				ones:  ^uint32(0),
				entry: f.entry(), // entry of the real (the outermost) function.
				name:  name,
				file:  file,
				line:  int(line),
			}
			return (*Func)(unsafe.Pointer(fi))
		}
	}
	return f._Func()
}

// Name returns the name of the function.
func (f *Func) Name() string {
	if f == nil {
		return ""
	}
	fn := f.raw()
	if fn.isInlined() { // inlined version
		fi := (*funcinl)(unsafe.Pointer(fn))
		return fi.name
	}
	return funcname(f.funcInfo())
}

// Entry returns the entry address of the function.
func (f *Func) Entry() uintptr {
	fn := f.raw()
	if fn.isInlined() { // inlined version
		fi := (*funcinl)(unsafe.Pointer(fn))
		return fi.entry
	}
	return fn.funcInfo().entry()
}

// FileLine returns the file name and line number of the
// source code corresponding to the program counter pc.
// The result will not be accurate if pc is not a program
// counter within f.
func (f *Func) FileLine(pc uintptr) (file string, line int) {
	fn := f.raw()
	if fn.isInlined() { // inlined version
		fi := (*funcinl)(unsafe.Pointer(fn))
		return fi.file, fi.line
	}
	// Pass strict=false here, because anyone can call this function,
	// and they might just be wrong about targetpc belonging to f.
	file, line32 := funcline1(f.funcInfo(), pc, false)
	return file, int(line32)
}

// findmoduledatap looks up the moduledata for a PC.
//
// It is nosplit because it's part of the isgoexception
// implementation.
//
//go:nosplit
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

// isInlined reports whether f should be re-interpreted as a *funcinl.
func (f *_func) isInlined() bool {
	return f.entryoff == ^uint32(0) // see comment for funcinl.ones
}

// entry returns the entry PC for f.
func (f funcInfo) entry() uintptr {
	return f.datap.textAddr(f.entryoff)
}

// findfunc looks up function metadata for a PC.
//
// It is nosplit because it's part of the isgoexception
// implementation.
//
//go:nosplit
func findfunc(pc uintptr) funcInfo {
	datap := findmoduledatap(pc)
	if datap == nil {
		return funcInfo{}
	}
	const nsub = uintptr(len(findfuncbucket{}.subbuckets))

	pcOff, ok := datap.textOff(pc)
	if !ok {
		return funcInfo{}
	}

	x := uintptr(pcOff) + datap.text - datap.minpc // TODO: are datap.text and datap.minpc always equal?
	b := x / pcbucketsize
	i := x % pcbucketsize / (pcbucketsize / nsub)

	ffb := (*findfuncbucket)(add(unsafe.Pointer(datap.findfunctab), b*unsafe.Sizeof(findfuncbucket{})))
	idx := ffb.idx + uint32(ffb.subbuckets[i])

	// Find the ftab entry.
	for datap.ftab[idx+1].entryoff <= pcOff {
		idx++
	}

	funcoff := datap.ftab[idx].funcoff
	return funcInfo{(*_func)(unsafe.Pointer(&datap.pclntable[funcoff])), datap}
}

type pcvalueCache struct {
	entries [2][8]pcvalueCacheEnt
}

type pcvalueCacheEnt struct {
	// targetpc and off together are the key of this cache entry.
	targetpc uintptr
	off      uint32
	// val is the value of this cached pcvalue entry.
	val int32
}

// pcvalueCacheKey returns the outermost index in a pcvalueCache to use for targetpc.
// It must be very cheap to calculate.
// For now, align to goarch.PtrSize and reduce mod the number of entries.
// In practice, this appears to be fairly randomly and evenly distributed.
func pcvalueCacheKey(targetpc uintptr) uintptr {
	return (targetpc / goarch.PtrSize) % uintptr(len(pcvalueCache{}.entries))
}

// Returns the PCData value, and the PC where this value starts.
// TODO: the start PC is returned only when cache is nil.
func pcvalue(f funcInfo, off uint32, targetpc uintptr, cache *pcvalueCache, strict bool) (int32, uintptr) {
	if off == 0 {
		return -1, 0
	}

	// Check the cache. This speeds up walks of deep stacks, which
	// tend to have the same recursive functions over and over.
	//
	// This cache is small enough that full associativity is
	// cheaper than doing the hashing for a less associative
	// cache.
	if cache != nil {
		x := pcvalueCacheKey(targetpc)
		for i := range cache.entries[x] {
			// We check off first because we're more
			// likely to have multiple entries with
			// different offsets for the same targetpc
			// than the other way around, so we'll usually
			// fail in the first clause.
			ent := &cache.entries[x][i]
			if ent.off == off && ent.targetpc == targetpc {
				return ent.val, 0
			}
		}
	}

	if !f.valid() {
		if strict && panicking == 0 {
			println("runtime: no module data for", hex(f.entry()))
			throw("no module data")
		}
		return -1, 0
	}
	datap := f.datap
	p := datap.pctab[off:]
	pc := f.entry()
	prevpc := pc
	val := int32(-1)
	for {
		var ok bool
		p, ok = step(p, &pc, &val, pc == f.entry())
		if !ok {
			break
		}
		if targetpc < pc {
			// Replace a random entry in the cache. Random
			// replacement prevents a performance cliff if
			// a recursive stack's cycle is slightly
			// larger than the cache.
			// Put the new element at the beginning,
			// since it is the most likely to be newly used.
			if cache != nil {
				x := pcvalueCacheKey(targetpc)
				e := &cache.entries[x]
				ci := fastrandn(uint32(len(cache.entries[x])))
				e[ci] = e[0]
				e[0] = pcvalueCacheEnt{
					targetpc: targetpc,
					off:      off,
					val:      val,
				}
			}

			return val, prevpc
		}
		prevpc = pc
	}

	// If there was a table, it should have covered all program counters.
	// If not, something is wrong.
	if panicking != 0 || !strict {
		return -1, 0
	}

	print("runtime: invalid pc-encoded table f=", funcname(f), " pc=", hex(pc), " targetpc=", hex(targetpc), " tab=", p, "\n")

	p = datap.pctab[off:]
	pc = f.entry()
	val = -1
	for {
		var ok bool
		p, ok = step(p, &pc, &val, pc == f.entry())
		if !ok {
			break
		}
		print("\tvalue=", val, " until pc=", hex(pc), "\n")
	}

	throw("invalid runtime symbol table")
	return -1, 0
}

func cfuncname(f funcInfo) *byte {
	if !f.valid() || f.nameoff == 0 {
		return nil
	}
	return &f.datap.funcnametab[f.nameoff]
}

func funcname(f funcInfo) string {
	return gostringnocopy(cfuncname(f))
}

func funcpkgpath(f funcInfo) string {
	name := funcname(f)
	i := len(name) - 1
	for ; i > 0; i-- {
		if name[i] == '/' {
			break
		}
	}
	for ; i < len(name); i++ {
		if name[i] == '.' {
			break
		}
	}
	return name[:i]
}

func cfuncnameFromNameoff(f funcInfo, nameoff int32) *byte {
	if !f.valid() {
		return nil
	}
	return &f.datap.funcnametab[nameoff]
}

func funcnameFromNameoff(f funcInfo, nameoff int32) string {
	return gostringnocopy(cfuncnameFromNameoff(f, nameoff))
}

func funcfile(f funcInfo, fileno int32) string {
	datap := f.datap
	if !f.valid() {
		return "?"
	}
	// Make sure the cu index and file offset are valid
	if fileoff := datap.cutab[f.cuOffset+uint32(fileno)]; fileoff != ^uint32(0) {
		return gostringnocopy(&datap.filetab[fileoff])
	}
	// pcln section is corrupt.
	return "?"
}

func funcline1(f funcInfo, targetpc uintptr, strict bool) (file string, line int32) {
	datap := f.datap
	if !f.valid() {
		return "?", 0
	}
	fileno, _ := pcvalue(f, f.pcfile, targetpc, nil, strict)
	line, _ = pcvalue(f, f.pcln, targetpc, nil, strict)
	if fileno == -1 || line == -1 || int(fileno) >= len(datap.filetab) {
		// print("looking for ", hex(targetpc), " in ", funcname(f), " got file=", fileno, " line=", lineno, "\n")
		return "?", 0
	}
	file = funcfile(f, fileno)
	return
}

func funcline(f funcInfo, targetpc uintptr) (file string, line int32) {
	return funcline1(f, targetpc, true)
}

func funcspdelta(f funcInfo, targetpc uintptr, cache *pcvalueCache) int32 {
	x, _ := pcvalue(f, f.pcsp, targetpc, cache, true)
	if debugPcln && x&(goarch.PtrSize-1) != 0 {
		print("invalid spdelta ", funcname(f), " ", hex(f.entry()), " ", hex(targetpc), " ", hex(f.pcsp), " ", x, "\n")
		throw("bad spdelta")
	}
	return x
}

// funcMaxSPDelta returns the maximum spdelta at any point in f.
func funcMaxSPDelta(f funcInfo) int32 {
	datap := f.datap
	p := datap.pctab[f.pcsp:]
	pc := f.entry()
	val := int32(-1)
	max := int32(0)
	for {
		var ok bool
		p, ok = step(p, &pc, &val, pc == f.entry())
		if !ok {
			return max
		}
		if val > max {
			max = val
		}
	}
}

func pcdatastart(f funcInfo, table uint32) uint32 {
	return *(*uint32)(add(unsafe.Pointer(&f.nfuncdata), unsafe.Sizeof(f.nfuncdata)+uintptr(table)*4))
}

func pcdatavalue(f funcInfo, table uint32, targetpc uintptr, cache *pcvalueCache) int32 {
	if table >= f.npcdata {
		return -1
	}
	r, _ := pcvalue(f, pcdatastart(f, table), targetpc, cache, true)
	return r
}

func pcdatavalue1(f funcInfo, table uint32, targetpc uintptr, cache *pcvalueCache, strict bool) int32 {
	if table >= f.npcdata {
		return -1
	}
	r, _ := pcvalue(f, pcdatastart(f, table), targetpc, cache, strict)
	return r
}

// Like pcdatavalue, but also return the start PC of this PCData value.
// It doesn't take a cache.
func pcdatavalue2(f funcInfo, table uint32, targetpc uintptr) (int32, uintptr) {
	if table >= f.npcdata {
		return -1, 0
	}
	return pcvalue(f, pcdatastart(f, table), targetpc, nil, true)
}

// funcdata returns a pointer to the ith funcdata for f.
// funcdata should be kept in sync with cmd/link:writeFuncs.
func funcdata(f funcInfo, i uint8) unsafe.Pointer {
	if i < 0 || i >= f.nfuncdata {
		return nil
	}
	base := f.datap.gofunc // load gofunc address early so that we calculate during cache misses
	p := uintptr(unsafe.Pointer(&f.nfuncdata)) + unsafe.Sizeof(f.nfuncdata) + uintptr(f.npcdata)*4 + uintptr(i)*4
	off := *(*uint32)(unsafe.Pointer(p))
	// Return off == ^uint32(0) ? 0 : f.datap.gofunc + uintptr(off), but without branches.
	// The compiler calculates mask on most architectures using conditional assignment.
	var mask uintptr
	if off == ^uint32(0) {
		mask = 1
	}
	mask--
	raw := base + uintptr(off)
	return unsafe.Pointer(raw & mask)
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
	*val += int32(-(uvdelta & 1) ^ (uvdelta >> 1))
	p = p[n:]

	pcdelta := uint32(p[0])
	n = 1
	if pcdelta&0x80 != 0 {
		n, pcdelta = readvarint(p)
	}
	p = p[n:]
	*pc += uintptr(pcdelta * sys.PCQuantum)
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
	// Check this invariant only when stackDebug is on at all.
	// The invariant is already checked by many of stackmapdata's callers,
	// and disabling it by default allows stackmapdata to be inlined.
	if stackDebug > 0 && (n < 0 || n >= stkmap.n) {
		throw("stackmapdata: index out of range")
	}
	return bitvector{stkmap.nbit, addb(&stkmap.bytedata[0], uintptr(n*((stkmap.nbit+7)>>3)))}
}

// inlinedCall is the encoding of entries in the FUNCDATA_InlTree table.
type inlinedCall struct {
	parent   int16  // index of parent in the inltree, or < 0
	funcID   funcID // type of the called function
	_        byte
	file     int32 // perCU file index for inlined call. See cmd/link:pcln.go
	line     int32 // line number of the call site
	func_    int32 // offset into pclntab for name of called function
	parentPc int32 // position of an instruction whose source position is the call site (offset from entry)
}
