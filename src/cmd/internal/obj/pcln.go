// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import "log"

func addvarint(d *Pcdata, v uint32) {
	for ; v >= 0x80; v >>= 7 {
		d.P = append(d.P, uint8(v|0x80))
	}
	d.P = append(d.P, uint8(v))
}

// funcpctab writes to dst a pc-value table mapping the code in func to the values
// returned by valfunc parameterized by arg. The invocation of valfunc to update the
// current value is, for each p,
//
//	val = valfunc(func, val, p, 0, arg);
//	record val as value at p->pc;
//	val = valfunc(func, val, p, 1, arg);
//
// where func is the function, val is the current value, p is the instruction being
// considered, and arg can be used to further parameterize valfunc.
func funcpctab(ctxt *Link, dst *Pcdata, func_ *LSym, desc string, valfunc func(*Link, *LSym, int32, *Prog, int32, interface{}) int32, arg interface{}) {
	dbg := desc == ctxt.Debugpcln

	dst.P = dst.P[:0]

	if dbg {
		ctxt.Logf("funcpctab %s [valfunc=%s]\n", func_.Name, desc)
	}

	val := int32(-1)
	oldval := val
	if func_.Func.Text == nil {
		return
	}

	pc := func_.Func.Text.Pc

	if dbg {
		ctxt.Logf("%6x %6d %v\n", uint64(pc), val, func_.Func.Text)
	}

	started := false
	var delta uint32
	for p := func_.Func.Text; p != nil; p = p.Link {
		// Update val. If it's not changing, keep going.
		val = valfunc(ctxt, func_, val, p, 0, arg)

		if val == oldval && started {
			val = valfunc(ctxt, func_, val, p, 1, arg)
			if dbg {
				ctxt.Logf("%6x %6s %v\n", uint64(p.Pc), "", p)
			}
			continue
		}

		// If the pc of the next instruction is the same as the
		// pc of this instruction, this instruction is not a real
		// instruction. Keep going, so that we only emit a delta
		// for a true instruction boundary in the program.
		if p.Link != nil && p.Link.Pc == p.Pc {
			val = valfunc(ctxt, func_, val, p, 1, arg)
			if dbg {
				ctxt.Logf("%6x %6s %v\n", uint64(p.Pc), "", p)
			}
			continue
		}

		// The table is a sequence of (value, pc) pairs, where each
		// pair states that the given value is in effect from the current position
		// up to the given pc, which becomes the new current position.
		// To generate the table as we scan over the program instructions,
		// we emit a "(value" when pc == func->value, and then
		// each time we observe a change in value we emit ", pc) (value".
		// When the scan is over, we emit the closing ", pc)".
		//
		// The table is delta-encoded. The value deltas are signed and
		// transmitted in zig-zag form, where a complement bit is placed in bit 0,
		// and the pc deltas are unsigned. Both kinds of deltas are sent
		// as variable-length little-endian base-128 integers,
		// where the 0x80 bit indicates that the integer continues.

		if dbg {
			ctxt.Logf("%6x %6d %v\n", uint64(p.Pc), val, p)
		}

		if started {
			addvarint(dst, uint32((p.Pc-pc)/int64(ctxt.Arch.MinLC)))
			pc = p.Pc
		}

		delta = uint32(val) - uint32(oldval)
		if delta>>31 != 0 {
			delta = 1 | ^(delta << 1)
		} else {
			delta <<= 1
		}
		addvarint(dst, delta)
		oldval = val
		started = true
		val = valfunc(ctxt, func_, val, p, 1, arg)
	}

	if started {
		if dbg {
			ctxt.Logf("%6x done\n", uint64(func_.Func.Text.Pc+func_.Size))
		}
		addvarint(dst, uint32((func_.Size-pc)/int64(ctxt.Arch.MinLC)))
		addvarint(dst, 0) // terminator
	}

	if dbg {
		ctxt.Logf("wrote %d bytes to %p\n", len(dst.P), dst)
		for i := 0; i < len(dst.P); i++ {
			ctxt.Logf(" %02x", dst.P[i])
		}
		ctxt.Logf("\n")
	}
}

// pctofileline computes either the file number (arg == 0)
// or the line number (arg == 1) to use at p.
// Because p.Pos applies to p, phase == 0 (before p)
// takes care of the update.
func pctofileline(ctxt *Link, sym *LSym, oldval int32, p *Prog, phase int32, arg interface{}) int32 {
	if p.As == ATEXT || p.As == ANOP || p.Pos.Line() == 0 || phase == 1 {
		return oldval
	}
	f, l := linkgetlineFromPos(ctxt, p.Pos)
	if arg == nil {
		return l
	}
	pcln := arg.(*Pcln)

	if f == pcln.Lastfile {
		return int32(pcln.Lastindex)
	}

	for i, file := range pcln.File {
		if file == f {
			pcln.Lastfile = f
			pcln.Lastindex = i
			return int32(i)
		}
	}
	i := len(pcln.File)
	pcln.File = append(pcln.File, f)
	pcln.Lastfile = f
	pcln.Lastindex = i
	return int32(i)
}

// pcinlineState holds the state used to create a function's inlining
// tree and the PC-value table that maps PCs to nodes in that tree.
type pcinlineState struct {
	globalToLocal map[int]int
	localTree     InlTree
}

// addBranch adds a branch from the global inlining tree in ctxt to
// the function's local inlining tree, returning the index in the local tree.
func (s *pcinlineState) addBranch(ctxt *Link, globalIndex int) int {
	if globalIndex < 0 {
		return -1
	}

	localIndex, ok := s.globalToLocal[globalIndex]
	if ok {
		return localIndex
	}

	// Since tracebacks don't include column information, we could
	// use one node for multiple calls of the same function on the
	// same line (e.g., f(x) + f(y)). For now, we use one node for
	// each inlined call.
	call := ctxt.InlTree.nodes[globalIndex]
	call.Parent = s.addBranch(ctxt, call.Parent)
	localIndex = len(s.localTree.nodes)
	s.localTree.nodes = append(s.localTree.nodes, call)
	s.globalToLocal[globalIndex] = localIndex
	return localIndex
}

// pctoinline computes the index into the local inlining tree to use at p.
// If p is not the result of inlining, pctoinline returns -1. Because p.Pos
// applies to p, phase == 0 (before p) takes care of the update.
func (s *pcinlineState) pctoinline(ctxt *Link, sym *LSym, oldval int32, p *Prog, phase int32, arg interface{}) int32 {
	if phase == 1 {
		return oldval
	}

	posBase := ctxt.PosTable.Pos(p.Pos).Base()
	if posBase == nil {
		return -1
	}

	globalIndex := posBase.InliningIndex()
	if globalIndex < 0 {
		return -1
	}

	if s.globalToLocal == nil {
		s.globalToLocal = make(map[int]int)
	}

	return int32(s.addBranch(ctxt, globalIndex))
}

// pctospadj computes the sp adjustment in effect.
// It is oldval plus any adjustment made by p itself.
// The adjustment by p takes effect only after p, so we
// apply the change during phase == 1.
func pctospadj(ctxt *Link, sym *LSym, oldval int32, p *Prog, phase int32, arg interface{}) int32 {
	if oldval == -1 { // starting
		oldval = 0
	}
	if phase == 0 {
		return oldval
	}
	if oldval+p.Spadj < -10000 || oldval+p.Spadj > 1100000000 {
		ctxt.Diag("overflow in spadj: %d + %d = %d", oldval, p.Spadj, oldval+p.Spadj)
		log.Fatalf("bad code")
	}

	return oldval + p.Spadj
}

// pctopcdata computes the pcdata value in effect at p.
// A PCDATA instruction sets the value in effect at future
// non-PCDATA instructions.
// Since PCDATA instructions have no width in the final code,
// it does not matter which phase we use for the update.
func pctopcdata(ctxt *Link, sym *LSym, oldval int32, p *Prog, phase int32, arg interface{}) int32 {
	if phase == 0 || p.As != APCDATA || p.From.Offset != int64(arg.(uint32)) {
		return oldval
	}
	if int64(int32(p.To.Offset)) != p.To.Offset {
		ctxt.Diag("overflow in PCDATA instruction: %v", p)
		log.Fatalf("bad code")
	}

	return int32(p.To.Offset)
}

func linkpcln(ctxt *Link, cursym *LSym) {
	pcln := &cursym.Func.Pcln

	npcdata := 0
	nfuncdata := 0
	for p := cursym.Func.Text; p != nil; p = p.Link {
		// Find the highest ID of any used PCDATA table. This ignores PCDATA table
		// that consist entirely of "-1", since that's the assumed default value.
		//   From.Offset is table ID
		//   To.Offset is data
		if p.As == APCDATA && p.From.Offset >= int64(npcdata) && p.To.Offset != -1 { // ignore -1 as we start at -1, if we only see -1, nothing changed
			npcdata = int(p.From.Offset + 1)
		}
		// Find the highest ID of any FUNCDATA table.
		//   From.Offset is table ID
		if p.As == AFUNCDATA && p.From.Offset >= int64(nfuncdata) {
			nfuncdata = int(p.From.Offset + 1)
		}
	}

	pcln.Pcdata = make([]Pcdata, npcdata)
	pcln.Pcdata = pcln.Pcdata[:npcdata]
	pcln.Funcdata = make([]*LSym, nfuncdata)
	pcln.Funcdataoff = make([]int64, nfuncdata)
	pcln.Funcdataoff = pcln.Funcdataoff[:nfuncdata]

	funcpctab(ctxt, &pcln.Pcsp, cursym, "pctospadj", pctospadj, nil)
	funcpctab(ctxt, &pcln.Pcfile, cursym, "pctofile", pctofileline, pcln)
	funcpctab(ctxt, &pcln.Pcline, cursym, "pctoline", pctofileline, nil)

	pcinlineState := new(pcinlineState)
	funcpctab(ctxt, &pcln.Pcinline, cursym, "pctoinline", pcinlineState.pctoinline, nil)
	pcln.InlTree = pcinlineState.localTree
	if ctxt.Debugpcln == "pctoinline" && len(pcln.InlTree.nodes) > 0 {
		ctxt.Logf("-- inlining tree for %s:\n", cursym)
		dumpInlTree(ctxt, pcln.InlTree)
		ctxt.Logf("--\n")
	}

	// tabulate which pc and func data we have.
	havepc := make([]uint32, (npcdata+31)/32)
	havefunc := make([]uint32, (nfuncdata+31)/32)
	for p := cursym.Func.Text; p != nil; p = p.Link {
		if p.As == AFUNCDATA {
			if (havefunc[p.From.Offset/32]>>uint64(p.From.Offset%32))&1 != 0 {
				ctxt.Diag("multiple definitions for FUNCDATA $%d", p.From.Offset)
			}
			havefunc[p.From.Offset/32] |= 1 << uint64(p.From.Offset%32)
		}

		if p.As == APCDATA && p.To.Offset != -1 {
			havepc[p.From.Offset/32] |= 1 << uint64(p.From.Offset%32)
		}
	}

	// pcdata.
	for i := 0; i < npcdata; i++ {
		if (havepc[i/32]>>uint(i%32))&1 == 0 {
			continue
		}
		funcpctab(ctxt, &pcln.Pcdata[i], cursym, "pctopcdata", pctopcdata, interface{}(uint32(i)))
	}

	// funcdata
	if nfuncdata > 0 {
		var i int
		for p := cursym.Func.Text; p != nil; p = p.Link {
			if p.As == AFUNCDATA {
				i = int(p.From.Offset)
				pcln.Funcdataoff[i] = p.To.Offset
				if p.To.Type != TYPE_CONST {
					// TODO: Dedup.
					//funcdata_bytes += p->to.sym->size;
					pcln.Funcdata[i] = p.To.Sym
				}
			}
		}
	}
}
