// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	_ "unsafe" // for linkname
)

// inlinedCall is the encoding of entries in the FUNCDATA_InlTree table.
type inlinedCall struct {
	funcID    abi.FuncID // type of the called function
	_         [3]byte
	nameOff   int32 // offset into pclntab for name of called function
	parentPc  int32 // position of an instruction whose source position is the call site (offset from entry)
	startLine int32 // line number of start of function (func keyword/TEXT directive)
}

// An inlineUnwinder iterates over the stack of inlined calls at a PC by
// decoding the inline table. The last step of iteration is always the frame of
// the physical function, so there's always at least one frame.
//
// This is typically used as:
//
//	for u, uf := newInlineUnwinder(...); uf.valid(); uf = u.next(uf) { ... }
//
// Implementation note: This is used in contexts that disallow write barriers.
// Hence, the constructor returns this by value and pointer receiver methods
// must not mutate pointer fields. Also, we keep the mutable state in a separate
// struct mostly to keep both structs SSA-able, which generates much better
// code.
type inlineUnwinder struct {
	f       funcInfo
	inlTree *[1 << 20]inlinedCall
}

// An inlineFrame is a position in an inlineUnwinder.
type inlineFrame struct {
	// pc is the PC giving the file/line metadata of the current frame. This is
	// always a "call PC" (not a "return PC"). This is 0 when the iterator is
	// exhausted.
	pc uintptr

	// index is the index of the current record in inlTree, or -1 if we are in
	// the outermost function.
	index int32
}

// newInlineUnwinder creates an inlineUnwinder initially set to the inner-most
// inlined frame at PC. PC should be a "call PC" (not a "return PC").
//
// This unwinder uses non-strict handling of PC because it's assumed this is
// only ever used for symbolic debugging. If things go really wrong, it'll just
// fall back to the outermost frame.
//
// newInlineUnwinder should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/phuslu/log
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname newInlineUnwinder
func newInlineUnwinder(f funcInfo, pc uintptr) (inlineUnwinder, inlineFrame) {
	inldata := funcdata(f, abi.FUNCDATA_InlTree)
	if inldata == nil {
		return inlineUnwinder{f: f}, inlineFrame{pc: pc, index: -1}
	}
	inlTree := (*[1 << 20]inlinedCall)(inldata)
	u := inlineUnwinder{f: f, inlTree: inlTree}
	return u, u.resolveInternal(pc)
}

func (u *inlineUnwinder) resolveInternal(pc uintptr) inlineFrame {
	return inlineFrame{
		pc: pc,
		// Conveniently, this returns -1 if there's an error, which is the same
		// value we use for the outermost frame.
		index: pcdatavalue1(u.f, abi.PCDATA_InlTreeIndex, pc, false),
	}
}

func (uf inlineFrame) valid() bool {
	return uf.pc != 0
}

// next returns the frame representing uf's logical caller.
func (u *inlineUnwinder) next(uf inlineFrame) inlineFrame {
	if uf.index < 0 {
		uf.pc = 0
		return uf
	}
	parentPc := u.inlTree[uf.index].parentPc
	return u.resolveInternal(u.f.entry() + uintptr(parentPc))
}

// isInlined returns whether uf is an inlined frame.
func (u *inlineUnwinder) isInlined(uf inlineFrame) bool {
	return uf.index >= 0
}

// srcFunc returns the srcFunc representing the given frame.
//
// srcFunc should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/phuslu/log
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
// The go:linkname is below.
func (u *inlineUnwinder) srcFunc(uf inlineFrame) srcFunc {
	if uf.index < 0 {
		return u.f.srcFunc()
	}
	t := &u.inlTree[uf.index]
	return srcFunc{
		u.f.datap,
		t.nameOff,
		t.startLine,
		t.funcID,
	}
}

//go:linkname badSrcFunc runtime.(*inlineUnwinder).srcFunc
func badSrcFunc(*inlineUnwinder, inlineFrame) srcFunc

// fileLine returns the file name and line number of the call within the given
// frame. As a convenience, for the innermost frame, it returns the file and
// line of the PC this unwinder was started at (often this is a call to another
// physical function).
//
// It returns "?", 0 if something goes wrong.
func (u *inlineUnwinder) fileLine(uf inlineFrame) (file string, line int) {
	file, line32 := funcline1(u.f, uf.pc, false)
	return file, int(line32)
}
