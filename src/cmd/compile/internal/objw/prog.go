// Derived from Inferno utils/6c/txt.c
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6c/txt.c
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

package objw

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/internal/obj"
	"cmd/internal/src"
	"internal/abi"
)

var sharedProgArray = new([10000]obj.Prog) // *T instead of T to work around issue 19839

// NewProgs returns a new Progs for fn.
// worker indicates which of the backend workers will use the Progs.
func NewProgs(fn *ir.Func, worker int) *Progs {
	pp := new(Progs)
	if base.Ctxt.CanReuseProgs() {
		sz := len(sharedProgArray) / base.Flag.LowerC
		pp.Cache = sharedProgArray[sz*worker : sz*(worker+1)]
	}
	pp.CurFunc = fn

	// prime the pump
	pp.Next = pp.NewProg()
	pp.Clear(pp.Next)

	pp.Pos = fn.Pos()
	pp.SetText(fn)
	// PCDATA tables implicitly start with index -1.
	pp.PrevLive = -1
	pp.NextLive = pp.PrevLive
	pp.NextUnsafe = pp.PrevUnsafe
	return pp
}

// Progs accumulates Progs for a function and converts them into machine code.
type Progs struct {
	Text       *obj.Prog  // ATEXT Prog for this function
	Next       *obj.Prog  // next Prog
	PC         int64      // virtual PC; count of Progs
	Pos        src.XPos   // position to use for new Progs
	CurFunc    *ir.Func   // fn these Progs are for
	Cache      []obj.Prog // local progcache
	CacheIndex int        // first free element of progcache

	NextLive StackMapIndex // liveness index for the next Prog
	PrevLive StackMapIndex // last emitted liveness index

	NextUnsafe bool // unsafe mark for the next Prog
	PrevUnsafe bool // last emitted unsafe mark
}

type StackMapIndex int

// StackMapDontCare indicates that the stack map index at a Value
// doesn't matter.
//
// This is a sentinel value that should never be emitted to the PCDATA
// stream. We use -1000 because that's obviously never a valid stack
// index (but -1 is).
const StackMapDontCare StackMapIndex = -1000

func (s StackMapIndex) StackMapValid() bool {
	return s != StackMapDontCare
}

func (pp *Progs) NewProg() *obj.Prog {
	var p *obj.Prog
	if pp.CacheIndex < len(pp.Cache) {
		p = &pp.Cache[pp.CacheIndex]
		pp.CacheIndex++
	} else {
		p = new(obj.Prog)
	}
	p.Ctxt = base.Ctxt
	return p
}

// Flush converts from pp to machine code.
func (pp *Progs) Flush() {
	plist := &obj.Plist{Firstpc: pp.Text, Curfn: pp.CurFunc}
	obj.Flushplist(base.Ctxt, plist, pp.NewProg)
}

// Free clears pp and any associated resources.
func (pp *Progs) Free() {
	if base.Ctxt.CanReuseProgs() {
		// Clear progs to enable GC and avoid abuse.
		s := pp.Cache[:pp.CacheIndex]
		for i := range s {
			s[i] = obj.Prog{}
		}
	}
	// Clear pp to avoid abuse.
	*pp = Progs{}
}

// Prog adds a Prog with instruction As to pp.
func (pp *Progs) Prog(as obj.As) *obj.Prog {
	if pp.NextLive != StackMapDontCare && pp.NextLive != pp.PrevLive {
		// Emit stack map index change.
		idx := pp.NextLive
		pp.PrevLive = idx
		p := pp.Prog(obj.APCDATA)
		p.From.SetConst(abi.PCDATA_StackMapIndex)
		p.To.SetConst(int64(idx))
	}
	if pp.NextUnsafe != pp.PrevUnsafe {
		// Emit unsafe-point marker.
		pp.PrevUnsafe = pp.NextUnsafe
		p := pp.Prog(obj.APCDATA)
		p.From.SetConst(abi.PCDATA_UnsafePoint)
		if pp.NextUnsafe {
			p.To.SetConst(abi.UnsafePointUnsafe)
		} else {
			p.To.SetConst(abi.UnsafePointSafe)
		}
	}

	p := pp.Next
	pp.Next = pp.NewProg()
	pp.Clear(pp.Next)
	p.Link = pp.Next

	if !pp.Pos.IsKnown() && base.Flag.K != 0 {
		base.Warn("prog: unknown position (line 0)")
	}

	p.As = as
	p.Pos = pp.Pos
	if pp.Pos.IsStmt() == src.PosIsStmt {
		// Clear IsStmt for later Progs at this pos provided that as can be marked as a stmt
		if LosesStmtMark(as) {
			return p
		}
		pp.Pos = pp.Pos.WithNotStmt()
	}
	return p
}

func (pp *Progs) Clear(p *obj.Prog) {
	obj.Nopout(p)
	p.As = obj.AEND
	p.Pc = pp.PC
	pp.PC++
}

func (pp *Progs) Append(p *obj.Prog, as obj.As, ftype obj.AddrType, freg int16, foffset int64, ttype obj.AddrType, treg int16, toffset int64) *obj.Prog {
	q := pp.NewProg()
	pp.Clear(q)
	q.As = as
	q.Pos = p.Pos
	q.From.Type = ftype
	q.From.Reg = freg
	q.From.Offset = foffset
	q.To.Type = ttype
	q.To.Reg = treg
	q.To.Offset = toffset
	q.Link = p.Link
	p.Link = q
	return q
}

func (pp *Progs) SetText(fn *ir.Func) {
	if pp.Text != nil {
		base.Fatalf("Progs.SetText called twice")
	}
	ptxt := pp.Prog(obj.ATEXT)
	pp.Text = ptxt

	fn.LSym.Func().Text = ptxt
	ptxt.From.Type = obj.TYPE_MEM
	ptxt.From.Name = obj.NAME_EXTERN
	ptxt.From.Sym = fn.LSym
}

// LosesStmtMark reports whether a prog with op as loses its statement mark on the way to DWARF.
// The attributes from some opcodes are lost in translation.
// TODO: this is an artifact of how funcpctab combines information for instructions at a single PC.
// Should try to fix it there.
func LosesStmtMark(as obj.As) bool {
	// is_stmt does not work for these; it DOES for ANOP even though that generates no code.
	return as == obj.APCDATA || as == obj.AFUNCDATA
}
