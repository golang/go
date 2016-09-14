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
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

package s390x

import (
	"cmd/compile/internal/gc"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"fmt"
)

var resvd = []int{
	s390x.REGZERO, // R0
	s390x.REGTMP,  // R10
	s390x.REGTMP2, // R11
	s390x.REGCTXT, // R12
	s390x.REGG,    // R13
	s390x.REG_LR,  // R14
	s390x.REGSP,   // R15
}

// generate
//	as $c, n
func ginscon(as obj.As, c int64, n2 *gc.Node) {
	var n1 gc.Node

	gc.Nodconst(&n1, gc.Types[gc.TINT64], c)

	if as != s390x.AMOVD && (c < -s390x.BIG || c > s390x.BIG) || n2.Op != gc.OREGISTER {
		// cannot have more than 16-bit of immediate in ADD, etc.
		// instead, MOV into register first.
		var ntmp gc.Node
		gc.Regalloc(&ntmp, gc.Types[gc.TINT64], nil)

		rawgins(s390x.AMOVD, &n1, &ntmp)
		rawgins(as, &ntmp, n2)
		gc.Regfree(&ntmp)
		return
	}

	rawgins(as, &n1, n2)
}

// gmvc tries to move f to t using a mvc instruction.
// If successful it returns true, otherwise it returns false.
func gmvc(f, t *gc.Node) bool {
	ft := int(gc.Simsimtype(f.Type))
	tt := int(gc.Simsimtype(t.Type))

	if ft != tt {
		return false
	}

	if f.Op != gc.OINDREG || t.Op != gc.OINDREG {
		return false
	}

	if f.Xoffset < 0 || f.Xoffset >= 4096-8 {
		return false
	}

	if t.Xoffset < 0 || t.Xoffset >= 4096-8 {
		return false
	}

	var len int64
	switch ft {
	case gc.TUINT8, gc.TINT8, gc.TBOOL:
		len = 1
	case gc.TUINT16, gc.TINT16:
		len = 2
	case gc.TUINT32, gc.TINT32, gc.TFLOAT32:
		len = 4
	case gc.TUINT64, gc.TINT64, gc.TFLOAT64, gc.TPTR64:
		len = 8
	case gc.TUNSAFEPTR:
		len = int64(gc.Widthptr)
	default:
		return false
	}

	p := gc.Prog(s390x.AMVC)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)
	p.From3 = new(obj.Addr)
	p.From3.Offset = len
	p.From3.Type = obj.TYPE_CONST
	return true
}

func intLiteral(n *gc.Node) (x int64, ok bool) {
	switch {
	case n == nil:
		return
	case gc.Isconst(n, gc.CTINT):
		return n.Int64(), true
	case gc.Isconst(n, gc.CTBOOL):
		return int64(obj.Bool2int(n.Bool())), true
	}
	return
}

// gins is called by the front end.
// It synthesizes some multiple-instruction sequences
// so the front end can stay simpler.
func gins(as obj.As, f, t *gc.Node) *obj.Prog {
	if t != nil {
		if as >= obj.A_ARCHSPECIFIC {
			if x, ok := intLiteral(f); ok {
				ginscon(as, x, t)
				return nil // caller must not use
			}
		}
	}
	return rawgins(as, f, t)
}

// generate one instruction:
//	as f, t
func rawgins(as obj.As, f *gc.Node, t *gc.Node) *obj.Prog {
	// self move check
	// TODO(mundaym): use sized math and extend to MOVB, MOVWZ etc.
	switch as {
	case s390x.AMOVD, s390x.AFMOVS, s390x.AFMOVD:
		if f != nil && t != nil &&
			f.Op == gc.OREGISTER && t.Op == gc.OREGISTER &&
			f.Reg == t.Reg {
			return nil
		}
	}

	p := gc.Prog(as)
	gc.Naddr(&p.From, f)
	gc.Naddr(&p.To, t)

	switch as {
	// Bad things the front end has done to us. Crash to find call stack.
	case s390x.ACMP, s390x.ACMPU:
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
	case s390x.AMOVB, s390x.AMOVBZ:
		w = 1

	case s390x.AMOVH, s390x.AMOVHZ:
		w = 2

	case s390x.AMOVW, s390x.AMOVWZ:
		w = 4

	case s390x.AMOVD:
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
