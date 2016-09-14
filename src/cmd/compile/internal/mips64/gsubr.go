// Derived from Inferno utils/6c/txt.c
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6c/txt.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package mips64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/mips"
	"fmt"
)

var resvd = []int{
	mips.REGZERO,
	mips.REGSP,   // reserved for SP
	mips.REGSB,   // reserved for SB
	mips.REGLINK, // reserved for link
	mips.REGG,
	mips.REGTMP,
	mips.REG_R26, // kernel
	mips.REG_R27, // kernel
}

/*
 * generate
 *      as $c, n
 */
func ginscon(as obj.As, c int64, n2 *gc.Node) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	if as != mips.AMOVV && (c < -mips.BIG || c > mips.BIG) || n2.Op != gc.OREGISTER || as == mips.AMUL || as == mips.AMULU || as == mips.AMULV || as == mips.AMULVU {
		// cannot have more than 16-bit of immediate in ADD, etc.
		// instead, MOV into register first.
		var ntmp gc.Node
		gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

		rawgins(mips.AMOVV, &n1, &ntmp)
		rawgins(as, &ntmp, n2)
		gc.Regfree(&ntmp)
		return
	}

	rawgins(as, &n1, n2)
}

// gins is called by the front end.
// It synthesizes some multiple-instruction sequences
// so the front end can stay simpler.
func gins(as obj.As, f, t *gc.Node) *obj.Prog {
	if as >= obj.A_ARCHSPECIFIC {
		if x, ok := f.IntLiteral(); ok {
			ginscon(as, x, t)
			return nil // caller must not use
		}
	}
	return rawgins(as, f, t)
}

/*
 * generate one instruction:
 *	as f, t
 */
func rawgins(as obj.As, f *gc.Node, t *gc.Node) *obj.Prog {
	// TODO(austin): Add self-move test like in 6g (but be careful
	// of truncation moves)

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	switch as {
	case obj.ACALL:
		if p.To.Type == obj.TYPE_REG {
			// Allow front end to emit CALL REG, and rewrite into CALL (REG).
			p.From = obj.Addr{}
			p.To.Type = obj.TYPE_MEM
			p.To.Offset = 0

			if gc.Debug['g'] != 0 {
				fmt.Printf("%v\n", p)
			}

			return p
		}

	// Bad things the front end has done to us. Crash to find call stack.
	case mips.AAND:
		if p.From.Type == obj.TYPE_CONST {
			gc.Debug['h'] = 1
			gc.Fatalf("bad inst: %v", p)
		}
	case mips.ASGT, mips.ASGTU:
		if p.From.Type == obj.TYPE_MEM || p.To.Type == obj.TYPE_MEM {
			gc.Debug['h'] = 1
			gc.Fatalf("bad inst: %v", p)
		}

	// Special cases
	case mips.AMUL, mips.AMULU, mips.AMULV, mips.AMULVU:
		if p.From.Type == obj.TYPE_CONST {
			gc.Debug['h'] = 1
			gc.Fatalf("bad inst: %v", p)
		}

		pp := gc.Prog(mips.AMOVV)
		pp.From.Type = obj.TYPE_REG
		pp.From.Reg = mips.REG_LO
		pp.To = p.To

		p.Reg = p.To.Reg
		p.To = obj.Addr{}

	case mips.ASUBVU:
		// unary
		if f == nil {
			p.From = p.To
			p.Reg = mips.REGZERO
		}
	}

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	w := int32(0)
	switch as {
	case mips.AMOVB,
		mips.AMOVBU:
		w = 1

	case mips.AMOVH,
		mips.AMOVHU:
		w = 2

	case mips.AMOVW,
		mips.AMOVWU:
		w = 4

	case mips.AMOVV:
		if p.From.Type == obj.TYPE_CONST || p.From.Type == obj.TYPE_ADDR {
			break
		}
		w = 8
	}

	if w != 0 && ((f != nil && p.From.Width < int64(w)) || (t != nil && p.To.Type != obj.TYPE_REG && p.To.Width > int64(w))) {
		gc.Dump("f", f)
		gc.Dump("t", t)
		gc.Fatalf("bad width: %v (%d, %d)\n", p, p.From.Width, p.To.Width)
	}

	return p
}
