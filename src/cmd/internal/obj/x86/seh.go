// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86

import (
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

type sehbuf struct {
	ctxt   *obj.Link
	data   []byte
	relocs []obj.Reloc
}

func (b *sehbuf) write8(v uint8) {
	b.data = append(b.data, v)
}

func (b *sehbuf) write32(v uint32) {
	b.data = binary.LittleEndian.AppendUint32(b.data, v)
}

func (b *sehbuf) writecode(op, value uint8) {
	b.write8(value<<4 | op)
}

func (b *sehbuf) writeaddr(s *obj.LSym) {
	if s != nil {
		b.relocs = append(b.relocs, obj.Reloc{
			Type: objabi.R_PEIMAGEOFF,
			Off:  int32(len(b.data)),
			Siz:  4,
			Sym:  s,
		})
	}
	b.write32(0)
}

func (b *sehbuf) sym(prefix string) *obj.LSym {
	// Include relocation targets so records with the same bytes but
	// different unwind information or exception handlers remain distinct.
	var name strings.Builder
	fmt.Fprintf(&name, "%s%d.%s", prefix, len(b.data), base64.StdEncoding.EncodeToString(b.data))
	for _, r := range b.relocs {
		fmt.Fprintf(&name, ".%d.%d:%s", r.Off, len(r.Sym.Name), r.Sym.Name)
	}
	return b.ctxt.LookupInit(name.String(), func(s *obj.LSym) {
		s.WriteBytes(b.ctxt, 0, b.data)
		s.Align = 4
		s.Type = objabi.SSEHUNWINDINFO
		s.Set(obj.AttrDuplicateOK, true)
		s.Set(obj.AttrLocal, true)
		s.Set(obj.AttrContentAddressable, true)
		for _, r := range b.relocs {
			s.AddRel(b.ctxt, r)
		}
		b.ctxt.SEHSyms = append(b.ctxt.SEHSyms, s)
	})
}

// populateSeh generates function-relative .pdata records and their unwind information for s.
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
	if s.Size > math.MaxUint32 {
		ctxt.Diag("function too large for SEH: %s", s.Name)
		return
	}

	end := s.Size
	if !s.NoSplit() {
		split := movbp.Link
		for split != nil && split.Mark&sehStackSplit == 0 {
			split = split.Link
		}
		if split == nil {
			ctxt.Diag("missing stack-split code: %s", s.Name)
			return
		}
		end = split.Pc
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
	const nodes = 2
	buf := sehbuf{ctxt: ctxt, data: make([]byte, 0, 8+2*nodes)}
	buf.write8(flags | 1)            // Flags + version
	buf.write8(uint8(movbp.Link.Pc)) // Size of prolog
	buf.write8(nodes)                // Count of nodes
	buf.write8(SEH_REG_BP)           // FP register

	// Notes are written in reverse order of appearance.
	buf.write8(uint8(movbp.Link.Pc))
	buf.writecode(UWOP_SET_FPREG, 0)

	buf.write8(uint8(pushbp.Link.Pc))
	buf.writecode(UWOP_PUSH_NONVOL, SEH_REG_BP)

	buf.writeaddr(exceptionHandler)
	unwind := buf.sym("go:sehuw.")

	size := 12
	if end < s.Size {
		size += 12
	}
	pdata := sehbuf{ctxt: ctxt, data: make([]byte, 0, size)}
	pdata.write32(0)
	pdata.write32(uint32(end))
	pdata.writeaddr(unwind)
	if end < s.Size {
		// The stack-split code has no frame. Give it a separate .pdata
		// record with empty unwind info, so Windows pops the return
		// address from SP without changing BP.
		leaf := sehbuf{ctxt: ctxt, data: []byte{1, 0, 0, 0}} // UNWIND_INFO version 1.
		pdata.write32(uint32(end))
		pdata.write32(uint32(s.Size))
		pdata.writeaddr(leaf.sym("go:sehuw."))
	}
	return pdata.sym("go:sehpdata.")
}
