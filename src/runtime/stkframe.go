// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/runtime/sys"
	"unsafe"
)

// A stkframe holds information about a single physical stack frame.
type stkframe struct {
	// fn is the function being run in this frame. If there is
	// inlining, this is the outermost function.
	fn funcInfo

	// pc is the program counter within fn.
	//
	// The meaning of this is subtle:
	//
	// - Typically, this frame performed a regular function call
	//   and this is the return PC (just after the CALL
	//   instruction). In this case, pc-1 reflects the CALL
	//   instruction itself and is the correct source of symbolic
	//   information.
	//
	// - If this frame "called" sigpanic, then pc is the
	//   instruction that panicked, and pc is the correct address
	//   to use for symbolic information.
	//
	// - If this is the innermost frame, then PC is where
	//   execution will continue, but it may not be the
	//   instruction following a CALL. This may be from
	//   cooperative preemption, in which case this is the
	//   instruction after the call to morestack. Or this may be
	//   from a signal or an un-started goroutine, in which case
	//   PC could be any instruction, including the first
	//   instruction in a function. Conventionally, we use pc-1
	//   for symbolic information, unless pc == fn.entry(), in
	//   which case we use pc.
	pc uintptr

	// continpc is the PC where execution will continue in fn, or
	// 0 if execution will not continue in this frame.
	//
	// This is usually the same as pc, unless this frame "called"
	// sigpanic, in which case it's either the address of
	// deferreturn or 0 if this frame will never execute again.
	//
	// This is the PC to use to look up GC liveness for this frame.
	continpc uintptr

	lr   uintptr // program counter at caller aka link register
	sp   uintptr // stack pointer at pc
	fp   uintptr // stack pointer at caller aka frame pointer
	varp uintptr // top of local variables
	argp uintptr // pointer to function arguments
}

// reflectMethodValue is a partial duplicate of reflect.makeFuncImpl
// and reflect.methodValue.
type reflectMethodValue struct {
	fn     uintptr
	stack  *bitvector // ptrmap for both args and results
	argLen uintptr    // just args
}

// argBytes returns the argument frame size for a call to frame.fn.
func (frame *stkframe) argBytes() uintptr {
	if frame.fn.args != abi.ArgsSizeUnknown {
		return uintptr(frame.fn.args)
	}
	// This is an uncommon and complicated case. Fall back to fully
	// fetching the argument map to compute its size.
	argMap, _ := frame.argMapInternal()
	return uintptr(argMap.n) * goarch.PtrSize
}

// argMapInternal is used internally by stkframe to fetch special
// argument maps.
//
// argMap.n is always populated with the size of the argument map.
//
// argMap.bytedata is only populated for dynamic argument maps (used
// by reflect). If the caller requires the argument map, it should use
// this if non-nil, and otherwise fetch the argument map using the
// current PC.
//
// hasReflectStackObj indicates that this frame also has a reflect
// function stack object, which the caller must synthesize.
func (frame *stkframe) argMapInternal() (argMap bitvector, hasReflectStackObj bool) {
	f := frame.fn
	if f.args != abi.ArgsSizeUnknown {
		argMap.n = f.args / goarch.PtrSize
		return
	}
	// Extract argument bitmaps for reflect stubs from the calls they made to reflect.
	switch funcname(f) {
	case "reflect.makeFuncStub", "reflect.methodValueCall":
		// These take a *reflect.methodValue as their
		// context register and immediately save it to 0(SP).
		// Get the methodValue from 0(SP).
		arg0 := frame.sp + sys.MinFrameSize

		minSP := frame.fp
		if !usesLR {
			// The CALL itself pushes a word.
			// Undo that adjustment.
			minSP -= goarch.PtrSize
		}
		if arg0 >= minSP {
			// The function hasn't started yet.
			// This only happens if f was the
			// start function of a new goroutine
			// that hasn't run yet *and* f takes
			// no arguments and has no results
			// (otherwise it will get wrapped in a
			// closure). In this case, we can't
			// reach into its locals because it
			// doesn't have locals yet, but we
			// also know its argument map is
			// empty.
			if frame.pc != f.entry() {
				print("runtime: confused by ", funcname(f), ": no frame (sp=", hex(frame.sp), " fp=", hex(frame.fp), ") at entry+", hex(frame.pc-f.entry()), "\n")
				throw("reflect mismatch")
			}
			return bitvector{}, false // No locals, so also no stack objects
		}
		hasReflectStackObj = true
		mv := *(**reflectMethodValue)(unsafe.Pointer(arg0))
		// Figure out whether the return values are valid.
		// Reflect will update this value after it copies
		// in the return values.
		retValid := *(*bool)(unsafe.Pointer(arg0 + 4*goarch.PtrSize))
		if mv.fn != f.entry() {
			print("runtime: confused by ", funcname(f), "\n")
			throw("reflect mismatch")
		}
		argMap = *mv.stack
		if !retValid {
			// argMap.n includes the results, but
			// those aren't valid, so drop them.
			n := int32((mv.argLen &^ (goarch.PtrSize - 1)) / goarch.PtrSize)
			if n < argMap.n {
				argMap.n = n
			}
		}
	}
	return
}

// getStackMap returns the locals and arguments live pointer maps, and
// stack object list for frame.
func (frame *stkframe) getStackMap(debug bool) (locals, args bitvector, objs []stackObjectRecord) {
	targetpc := frame.continpc
	if targetpc == 0 {
		// Frame is dead. Return empty bitvectors.
		return
	}

	f := frame.fn
	pcdata := int32(-1)
	if targetpc != f.entry() {
		// Back up to the CALL. If we're at the function entry
		// point, we want to use the entry map (-1), even if
		// the first instruction of the function changes the
		// stack map.
		targetpc--
		pcdata = pcdatavalue(f, abi.PCDATA_StackMapIndex, targetpc)
	}
	if pcdata == -1 {
		// We do not have a valid pcdata value but there might be a
		// stackmap for this function. It is likely that we are looking
		// at the function prologue, assume so and hope for the best.
		pcdata = 0
	}

	// Local variables.
	size := frame.varp - frame.sp
	var minsize uintptr
	switch goarch.ArchFamily {
	case goarch.ARM64:
		minsize = sys.StackAlign
	default:
		minsize = sys.MinFrameSize
	}
	if size > minsize {
		stackid := pcdata
		stkmap := (*stackmap)(funcdata(f, abi.FUNCDATA_LocalsPointerMaps))
		if stkmap == nil || stkmap.n <= 0 {
			print("runtime: frame ", funcname(f), " untyped locals ", hex(frame.varp-size), "+", hex(size), "\n")
			throw("missing stackmap")
		}
		// If nbit == 0, there's no work to do.
		if stkmap.nbit > 0 {
			if stackid < 0 || stackid >= stkmap.n {
				// don't know where we are
				print("runtime: pcdata is ", stackid, " and ", stkmap.n, " locals stack map entries for ", funcname(f), " (targetpc=", hex(targetpc), ")\n")
				throw("bad symbol table")
			}
			locals = stackmapdata(stkmap, stackid)
			if stackDebug >= 3 && debug {
				print("      locals ", stackid, "/", stkmap.n, " ", locals.n, " words ", locals.bytedata, "\n")
			}
		} else if stackDebug >= 3 && debug {
			print("      no locals to adjust\n")
		}
	}

	// Arguments. First fetch frame size and special-case argument maps.
	var isReflect bool
	args, isReflect = frame.argMapInternal()
	if args.n > 0 && args.bytedata == nil {
		// Non-empty argument frame, but not a special map.
		// Fetch the argument map at pcdata.
		stackmap := (*stackmap)(funcdata(f, abi.FUNCDATA_ArgsPointerMaps))
		if stackmap == nil || stackmap.n <= 0 {
			print("runtime: frame ", funcname(f), " untyped args ", hex(frame.argp), "+", hex(args.n*goarch.PtrSize), "\n")
			throw("missing stackmap")
		}
		if pcdata < 0 || pcdata >= stackmap.n {
			// don't know where we are
			print("runtime: pcdata is ", pcdata, " and ", stackmap.n, " args stack map entries for ", funcname(f), " (targetpc=", hex(targetpc), ")\n")
			throw("bad symbol table")
		}
		if stackmap.nbit == 0 {
			args.n = 0
		} else {
			args = stackmapdata(stackmap, pcdata)
		}
	}

	// stack objects.
	if (GOARCH == "amd64" || GOARCH == "arm64" || GOARCH == "loong64" || GOARCH == "ppc64" || GOARCH == "ppc64le" || GOARCH == "riscv64") &&
		unsafe.Sizeof(abi.RegArgs{}) > 0 && isReflect {
		// For reflect.makeFuncStub and reflect.methodValueCall,
		// we need to fake the stack object record.
		// These frames contain an internal/abi.RegArgs at a hard-coded offset.
		// This offset matches the assembly code on amd64 and arm64.
		objs = methodValueCallFrameObjs[:]
	} else {
		p := funcdata(f, abi.FUNCDATA_StackObjects)
		if p != nil {
			n := *(*uintptr)(p)
			p = add(p, goarch.PtrSize)
			r0 := (*stackObjectRecord)(noescape(p))
			objs = unsafe.Slice(r0, int(n))
			// Note: the noescape above is needed to keep
			// getStackMap from "leaking param content:
			// frame".  That leak propagates up to getgcmask, then
			// GCMask, then verifyGCInfo, which converts the stack
			// gcinfo tests into heap gcinfo tests :(
		}
	}

	return
}

var methodValueCallFrameObjs [1]stackObjectRecord // initialized in stackobjectinit

func stkobjinit() {
	var abiRegArgsEface any = abi.RegArgs{}
	abiRegArgsType := efaceOf(&abiRegArgsEface)._type
	if abiRegArgsType.Kind_&abi.KindGCProg != 0 {
		throw("abiRegArgsType needs GC Prog, update methodValueCallFrameObjs")
	}
	// Set methodValueCallFrameObjs[0].gcdataoff so that
	// stackObjectRecord.gcdata() will work correctly with it.
	ptr := uintptr(unsafe.Pointer(&methodValueCallFrameObjs[0]))
	var mod *moduledata
	for datap := &firstmoduledata; datap != nil; datap = datap.next {
		if datap.gofunc <= ptr && ptr < datap.end {
			mod = datap
			break
		}
	}
	if mod == nil {
		throw("methodValueCallFrameObjs is not in a module")
	}
	methodValueCallFrameObjs[0] = stackObjectRecord{
		off:       -int32(alignUp(abiRegArgsType.Size_, 8)), // It's always the highest address local.
		size:      int32(abiRegArgsType.Size_),
		ptrBytes:  int32(abiRegArgsType.PtrBytes),
		gcdataoff: uint32(uintptr(unsafe.Pointer(abiRegArgsType.GCData)) - mod.rodata),
	}
}
