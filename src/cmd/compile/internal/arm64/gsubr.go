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

package arm64

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"fmt"
)

var resvd = []int{
	arm64.REGTMP,
	arm64.REGG,
	arm64.REGRT1,
	arm64.REGRT2,
	arm64.REG_R31, // REGZERO and REGSP
}

/*
 * generate
 *	as $c, n
 */
func ginscon(as obj.As, c int64, n2 *gc.Node) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	if as != arm64.AMOVD && (c < -arm64.BIG || c > arm64.BIG) || as == arm64.AMUL || n2 != nil && n2.Op != gc.OREGISTER {
		// cannot have more than 16-bit of immediate in ADD, etc.
		// instead, MOV into register first.
		var ntmp gc.Node
		gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

		gins(arm64.AMOVD, &n1, &ntmp)
		gins(as, &ntmp, n2)
		gc.Regfree(&ntmp)
		return
	}

	rawgins(as, &n1, n2)
}

/*
 * generate
 *	as n, $c (CMP)
 */
func ginscon2(as obj.As, n2 *gc.Node, c int64) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	switch as {
	default:
		gc.Fatalf("ginscon2")

	case arm64.ACMP:
		if -arm64.BIG <= c && c <= arm64.BIG {
			gcmp(as, n2, &n1)
			return
		}
	}

	// MOV n1 into register first
	var ntmp gc.Node
	gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

	rawgins(arm64.AMOVD, &n1, &ntmp)
	gcmp(as, n2, &ntmp)
	gc.Regfree(&ntmp)
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
	if as == arm64.ACMP {
		if x, ok := t.IntLiteral(); ok {
			ginscon2(as, f, x)
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
	case arm64.ACMP, arm64.AFCMPS, arm64.AFCMPD:
		if t != nil {
			if f.Op != gc.OREGISTER {
				gc.Fatalf("bad operands to gcmp")
			}
			p.From = p.To
			p.To = obj.Addr{}
			raddr(f, p)
		}
	}

	// Bad things the front end has done to us. Crash to find call stack.
	switch as {
	case arm64.AAND, arm64.AMUL:
		if p.From.Type == obj.TYPE_CONST {
			gc.Debug['h'] = 1
			gc.Fatalf("bad inst: %v", p)
		}
	case arm64.ACMP:
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
	case arm64.AMOVB,
		arm64.AMOVBU:
		w = 1

	case arm64.AMOVH,
		arm64.AMOVHU:
		w = 2

	case arm64.AMOVW,
		arm64.AMOVWU:
		w = 4

	case arm64.AMOVD:
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

/*
 * insert n into reg slot of p
 */
func raddr(n *gc.Node, p *obj.Prog) {
	var a obj.Addr

	gc.Naddr(&a, n)
	if a.Type != obj.TYPE_REG {
		if n != nil {
			gc.Fatalf("bad in raddr: %v", n.Op)
		} else {
			gc.Fatalf("bad in raddr: <null>")
		}
		p.Reg = 0
	} else {
		p.Reg = a.Reg
	}
}

func gcmp(as obj.As, lhs *gc.Node, rhs *gc.Node) *obj.Prog {
	if lhs.Op != gc.OREGISTER {
		gc.Fatalf("bad operands to gcmp: %v %v", lhs.Op, rhs.Op)
	}

	p := rawgins(as, rhs, nil)
	raddr(lhs, p)
	return p
}
