// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mips64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
)

func blockcopy(n, res *gc.Node, osrc, odst, w int64) {
	// determine alignment.
	// want to avoid unaligned access, so have to use
	// smaller operations for less aligned types.
	// for example moving [4]byte must use 4 MOVB not 1 MOVW.
	align := int(n.Type.Align)

	var op int
	switch align {
	default:
		gc.Fatalf("sgen: invalid alignment %d for %v", align, n.Type)

	case 1:
		op = mips.AMOVB

	case 2:
		op = mips.AMOVH

	case 4:
		op = mips.AMOVW

	case 8:
		op = mips.AMOVV
	}

	if w%int64(align) != 0 {
		gc.Fatalf("sgen: unaligned size %d (align=%d) for %v", w, align, n.Type)
	}
	c := int32(w / int64(align))

	// if we are copying forward on the stack and
	// the src and dst overlap, then reverse direction
	dir := align

	if osrc < odst && int64(odst) < int64(osrc)+w {
		dir = -dir
	}

	var dst gc.Node
	var src gc.Node
	if n.Ullman >= res.Ullman {
		gc.Agenr(n, &dst, res) // temporarily use dst
		gc.Regalloc(&src, gc.Types[gc.Tptr], nil)
		gins(mips.AMOVV, &dst, &src)
		if res.Op == gc.ONAME {
			gc.Gvardef(res)
		}
		gc.Agen(res, &dst)
	} else {
		if res.Op == gc.ONAME {
			gc.Gvardef(res)
		}
		gc.Agenr(res, &dst, res)
		gc.Agenr(n, &src, nil)
	}

	var tmp gc.Node
	gc.Regalloc(&tmp, gc.Types[gc.Tptr], nil)

	// set up end marker
	var nend gc.Node

	// move src and dest to the end of block if necessary
	if dir < 0 {
		if c >= 4 {
			gc.Regalloc(&nend, gc.Types[gc.Tptr], nil)
			gins(mips.AMOVV, &src, &nend)
		}

		p := gins(mips.AADDV, nil, &src)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = w

		p = gins(mips.AADDV, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = w
	} else {
		p := gins(mips.AADDV, nil, &src)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(-dir)

		p = gins(mips.AADDV, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(-dir)

		if c >= 4 {
			gc.Regalloc(&nend, gc.Types[gc.Tptr], nil)
			p := gins(mips.AMOVV, &src, &nend)
			p.From.Type = obj.TYPE_ADDR
			p.From.Offset = w
		}
	}

	// move
	// TODO: enable duffcopy for larger copies.
	if c >= 4 {
		p := gins(op, &src, &tmp)
		p.From.Type = obj.TYPE_MEM
		p.From.Offset = int64(dir)
		ploop := p

		p = gins(mips.AADDV, nil, &src)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(dir)

		p = gins(op, &tmp, &dst)
		p.To.Type = obj.TYPE_MEM
		p.To.Offset = int64(dir)

		p = gins(mips.AADDV, nil, &dst)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(dir)

		gc.Patch(ginsbranch(mips.ABNE, nil, &src, &nend, 0), ploop)
		gc.Regfree(&nend)
	} else {
		// TODO: Instead of generating ADDV $-8,R8; ADDV
		// $-8,R7; n*(MOVV 8(R8),R9; ADDV $8,R8; MOVV R9,8(R7);
		// ADDV $8,R7;) just generate the offsets directly and
		// eliminate the ADDs.  That will produce shorter, more
		// pipeline-able code.
		var p *obj.Prog
		for ; c > 0; c-- {
			p = gins(op, &src, &tmp)
			p.From.Type = obj.TYPE_MEM
			p.From.Offset = int64(dir)

			p = gins(mips.AADDV, nil, &src)
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = int64(dir)

			p = gins(op, &tmp, &dst)
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = int64(dir)

			p = gins(mips.AADDV, nil, &dst)
			p.From.Type = obj.TYPE_CONST
			p.From.Offset = int64(dir)
		}
	}

	gc.Regfree(&dst)
	gc.Regfree(&src)
	gc.Regfree(&tmp)
}
