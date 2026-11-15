// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"cmd/internal/obj"
	"internal/abi"
)

func leafNoSplit(ctxt *obj.Link, text *obj.Prog) bool {
	if !ctxt.WasmProfile || ctxt.IsAsm {
		return false
	}
	if text.To.Offset >= abi.StackSmall {
		return false
	}
	for p := text; p != nil; p = p.Link {
		switch p.As {
		case obj.ACALL, ACALLNORESUME, ACALLCONT, ACall, ACallIndirect, ARETUNWIND:
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

func NeedMoreStack(ctxt *obj.Link, text *obj.Prog) bool {
	return text.To.Offset > 0 && !text.From.Sym.NoSplit() && !leafNoSplit(ctxt, text)
}

func markLeafNoSplit(ctxt *obj.Link, s *obj.LSym) bool {
	if !leafNoSplit(ctxt, s.Func().Text) {
		return false
	}
	s.Set(obj.AttrNoSplit, true)
	return true
}

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
