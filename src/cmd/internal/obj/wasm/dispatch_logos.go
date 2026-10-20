// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"cmd/internal/obj"
	"internal/abi"
)

// This file holds the selective resume points of the WASM profile
// (loGo/s M16-S1). preprocess keeps its upstream behavior unless
// ctxt.WasmProfile is set, which only the compiler sets, through
// -logos.wasmprofile.
//
// Two decisions live here:
//
//   - A function needs the br_table dispatch only when it has a call that
//     keeps a resume point (obj.ACALL) or a jump between basic blocks. A
//     straight-line function whose calls all compile to ACALLNORESUME needs
//     neither, so it is emitted as plain wasm code. The stack check at its
//     entry resumes at the function start, which needs no dispatch.
//   - A leaf function with a small frame skips the stack check and becomes
//     nosplit, as the other ports do (x86 obj6.go). Stack growth unwinds
//     the wasm stack through runtime.morestack, so a function with a stack
//     check can never be called without a resume point; the small leaf can.

// LeafNoSplit reports whether the profile drops the stack check of the
// function whose first Prog is text: it has a frame below StackSmall and
// no call of any kind. It reads only the Prog list, so the compiler asks it
// before preprocess runs.
func LeafNoSplit(ctxt *obj.Link, text *obj.Prog) bool {
	if !ctxt.WasmProfile || ctxt.IsAsm {
		return false
	}
	if text.To.Offset >= abi.StackSmall {
		return false
	}
	for p := text; p != nil; p = p.Link {
		switch p.As {
		case obj.ACALL, ACALLNORESUME, ACall, ACallIndirect, ARETUNWIND:
			return false
		case obj.ARET:
			if p.To.Type == obj.TYPE_MEM {
				return false
			}
		case obj.AJMP:
			if p.To.Type != obj.TYPE_BRANCH {
				return false
			}
		}
	}
	return true
}

// NeedMoreStack reports whether preprocess emits the stack check for the
// function whose first Prog is text, so the compiler can tell which
// functions may unwind at entry.
func NeedMoreStack(ctxt *obj.Link, text *obj.Prog) bool {
	return text.To.Offset > 0 && !text.From.Sym.NoSplit() && !LeafNoSplit(ctxt, text)
}

// markLeafNoSplit drops the stack check of a leaf function and records the
// nosplit attribute so the linker's stack check sees the same function.
func markLeafNoSplit(ctxt *obj.Link, s *obj.LSym) bool {
	if !LeafNoSplit(ctxt, s.Func().Text) {
		return false
	}
	s.Set(obj.AttrNoSplit, true)
	return true
}

// needsDispatch reports whether s keeps the br_table dispatch prologue.
// Assembly functions, import and export wrappers, and builds with
// maymorestack keep it unconditionally.
func needsDispatch(ctxt *obj.Link, s *obj.LSym) bool {
	if !ctxt.WasmProfile || ctxt.IsAsm || ctxt.Flag_maymorestack != "" {
		return true
	}
	if s.Func().WasmImport != nil || s.Func().WasmExport != nil {
		return true
	}
	for p := s.Func().Text; p != nil; p = p.Link {
		if p.As == obj.ACALL || p.As == obj.AJMP && p.To.Type == obj.TYPE_BRANCH {
			return true
		}
	}
	return false
}
