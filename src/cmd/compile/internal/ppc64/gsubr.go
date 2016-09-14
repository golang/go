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

package ppc64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/ppc64"
	"fmt"
)

var resvd = []int{
	ppc64.REGZERO,
	ppc64.REGSP, // reserved for SP
	// We need to preserve the C ABI TLS pointer because sigtramp
	// may happen during C code and needs to access the g. C
	// clobbers REGG, so if Go were to clobber REGTLS, sigtramp
	// won't know which convention to use. By preserving REGTLS,
	// we can just retrieve g from TLS when we aren't sure.
	ppc64.REGTLS,

	// TODO(austin): Consolidate REGTLS and REGG?
	ppc64.REGG,
	ppc64.REGTMP, // REGTMP
}

/*
 * generate
 *	as $c, n
 */
func ginscon(as obj.As, c int64, n2 *gc.Node) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	if as != ppc64.AMOVD && (c < -ppc64.BIG || c > ppc64.BIG) || n2.Op != gc.OREGISTER || as == ppc64.AMULLD {
		// cannot have more than 16-bit of immediate in ADD, etc.
		// instead, MOV into register first.
		var ntmp gc.Node
		gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

		rawgins(ppc64.AMOVD, &n1, &ntmp)
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
		if p.To.Type == obj.TYPE_REG && p.To.Reg != ppc64.REG_CTR {
			// Allow front end to emit CALL REG, and rewrite into MOV REG, CTR; CALL CTR.
			if gc.Ctxt.Flag_shared {
				// Make sure function pointer is in R12 as well when
				// compiling Go into PIC.
				// TODO(mwhudson): it would obviously be better to
				// change the register allocation to put the value in
				// R12 already, but I don't know how to do that.
				q := gc.Prog(as)
				q.As = ppc64.AMOVD
				q.From = p.To
				q.To.Type = obj.TYPE_REG
				q.To.Reg = ppc64.REG_R12
			}
			pp := gc.Prog(as)
			pp.From = p.From
			pp.To.Type = obj.TYPE_REG
			pp.To.Reg = ppc64.REG_CTR

			p.As = ppc64.AMOVD
			p.From = p.To
			p.To.Type = obj.TYPE_REG
			p.To.Reg = ppc64.REG_CTR

			if gc.Ctxt.Flag_shared {
				// When compiling Go into PIC, the function we just
				// called via pointer might have been implemented in
				// a separate module and so overwritten the TOC
				// pointer in R2; reload it.
				q := gc.Prog(ppc64.AMOVD)
				q.From.Type = obj.TYPE_MEM
				q.From.Offset = 24
				q.From.Reg = ppc64.REGSP
				q.To.Type = obj.TYPE_REG
				q.To.Reg = ppc64.REG_R2
			}

			if gc.Debug['g'] != 0 {
				fmt.Printf("%v\n", p)
				fmt.Printf("%v\n", pp)
			}

			return pp
		}

	// Bad things the front end has done to us. Crash to find call stack.
	case ppc64.AAND, ppc64.AMULLD:
		if p.From.Type == obj.TYPE_CONST {
			gc.Debug['h'] = 1
			gc.Fatalf("bad inst: %v", p)
		}
	case ppc64.ACMP, ppc64.ACMPU:
		if p.From.Type == obj.TYPE_MEM || p.To.Type == obj.TYPE_MEM {
			gc.Debug['h'] = 1
			gc.Fatalf("bad inst: %v", p)
		}
	}

	if gc.Debug['g'] != 0 {
		fmt.Printf("%v\n", p)
	}

	w := int32(0)
	switch as {
	case ppc64.AMOVB,
		ppc64.AMOVBU,
		ppc64.AMOVBZ,
		ppc64.AMOVBZU:
		w = 1

	case ppc64.AMOVH,
		ppc64.AMOVHU,
		ppc64.AMOVHZ,
		ppc64.AMOVHZU:
		w = 2

	case ppc64.AMOVW,
		ppc64.AMOVWU,
		ppc64.AMOVWZ,
		ppc64.AMOVWZU:
		w = 4

	case ppc64.AMOVD,
		ppc64.AMOVDU:
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
