// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"encoding/base64"
	"fmt"
	"math"
)

type sehbuf struct {
	ctxt *obj.Link
	data []byte
	off  int
}

func newsehbuf(ctxt *obj.Link, nodes uint8) sehbuf {
	// - 8 bytes for the header
	// - 2 bytes for each node
	// - 2 bytes in case nodes is not even
	size := 8 + nodes*2
	if nodes%2 != 0 {
		size += 2
	}
	return sehbuf{ctxt, make([]byte, size), 0}
}

func (b *sehbuf) write8(v uint8) {
	b.data[b.off] = v
	b.off++
}

func (b *sehbuf) write32(v uint32) {
	b.ctxt.Arch.ByteOrder.PutUint32(b.data[b.off:], v)
	b.off += 4
}

func (b *sehbuf) writecode(op, value uint8) {
	b.write8(value<<4 | op)
}

// populateSeh generates the SEH unwind information for s.
func populateSeh(ctxt *obj.Link, s *obj.LSym) (sehsym *obj.LSym) {
	if s.NoFrame() {
		return
	}

	// This implementation expects the following function prologue layout:
	// - Stack split code (optional)
	// - PUSHQ	BP
	// - MOVQ	SP,	BP
	//
	// If the prologue layout change, the unwind information should be updated
	// accordingly.

	// Search for the PUSHQ BP instruction inside the prologue.
	var pushbp *obj.Prog
	for p := s.Func().Text; p != nil; p = p.Link {
		if p.As == APUSHQ && p.From.Type == obj.TYPE_REG && p.From.Reg == REG_BP {
			pushbp = p
			break
		}
		if p.Pos.Xlogue() == src.PosPrologueEnd {
			break
		}
	}
	if pushbp == nil {
		ctxt.Diag("missing frame pointer instruction: PUSHQ BP")
		return
	}

	// It must be followed by a MOVQ SP, BP.
	movbp := pushbp.Link
	if movbp == nil {
		ctxt.Diag("missing frame pointer instruction: MOVQ SP, BP")
		return
	}
	if !(movbp.As == AMOVQ && movbp.From.Type == obj.TYPE_REG && movbp.From.Reg == REG_SP &&
		movbp.To.Type == obj.TYPE_REG && movbp.To.Reg == REG_BP && movbp.From.Offset == 0) {
		ctxt.Diag("unexpected frame pointer instruction\n%v", movbp)
		return
	}
	if movbp.Link.Pc > math.MaxUint8 {
		// SEH unwind information don't support prologues that are more than 255 bytes long.
		// These are very rare, but still possible, e.g., when compiling functions with many
		// parameters with -gcflags=-d=maymorestack=runtime.mayMoreStackPreempt.
		// Return without reporting an error.
		return
	}

	// Reference:
	// https://learn.microsoft.com/en-us/cpp/build/exception-handling-x64#struct-unwind_info

	const (
		UWOP_PUSH_NONVOL  = 0
		UWOP_SET_FPREG    = 3
		SEH_REG_BP        = 5
		UNW_FLAG_EHANDLER = 1 << 3
	)

	var exceptionHandler *obj.LSym
	var flags uint8
	if s.Name == "runtime.asmcgocall_landingpad" {
		// Most cgo calls go through runtime.asmcgocall_landingpad,
		// we can use it to catch exceptions from C code.
		// TODO: use a more generic approach to identify which calls need an exception handler.
		exceptionHandler = ctxt.Lookup("runtime.sehtramp")
		if exceptionHandler == nil {
			ctxt.Diag("missing runtime.sehtramp\n")
			return
		}
		flags = UNW_FLAG_EHANDLER
	}

	// Fow now we only support operations which are encoded
	// using a single 2-byte node, so the number of nodes
	// is the number of operations.
	nodes := uint8(2)
	buf := newsehbuf(ctxt, nodes)
	buf.write8(flags | 1)            // Flags + version
	buf.write8(uint8(movbp.Link.Pc)) // Size of prolog
	buf.write8(nodes)                // Count of nodes
	buf.write8(SEH_REG_BP)           // FP register

	// Notes are written in reverse order of appearance.
	buf.write8(uint8(movbp.Link.Pc))
	buf.writecode(UWOP_SET_FPREG, 0)

	buf.write8(uint8(pushbp.Link.Pc))
	buf.writecode(UWOP_PUSH_NONVOL, SEH_REG_BP)

	// The following 4 bytes reference the RVA of the exception handler.
	// The value is set to 0 for now, if an exception handler is needed,
	// it will be updated later with a R_PEIMAGEOFF relocation to the
	// exception handler.
	buf.write32(0)

	// The list of unwind infos in a PE binary have very low cardinality
	// as each info only contains frame pointer operations,
	// which are very similar across functions.
	// Dedup them when possible.
	hash := base64.StdEncoding.EncodeToString(buf.data)
	symname := fmt.Sprintf("%d.%s", len(buf.data), hash)
	return ctxt.LookupInit("go:sehuw."+symname, func(s *obj.LSym) {
		s.WriteBytes(ctxt, 0, buf.data)
		s.Type = objabi.SSEHUNWINDINFO
		s.Set(obj.AttrDuplicateOK, true)
		s.Set(obj.AttrLocal, true)
		if exceptionHandler != nil {
			r := obj.Addrel(s)
			r.Off = int32(len(buf.data) - 4)
			r.Siz = 4
			r.Sym = exceptionHandler
			r.Type = objabi.R_PEIMAGEOFF
		}
		// Note: AttrContentAddressable cannot be set here,
		// because the content-addressable-handling code
		// does not know about aux symbols.
	})
}
