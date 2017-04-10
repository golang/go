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

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
)

var sharedProgArray *[10000]obj.Prog // *T instead of T to work around issue 19839

func init() {
	sharedProgArray = new([10000]obj.Prog)
}

// Progs accumulates Progs for a function and converts them into machine code.
type Progs struct {
	Text      *obj.Prog  // ATEXT Prog for this function
	next      *obj.Prog  // next Prog
	pc        int64      // virtual PC; count of Progs
	pos       src.XPos   // position to use for new Progs
	curfn     *Node      // fn these Progs are for
	progcache []obj.Prog // local progcache
	cacheidx  int        // first free element of progcache
}

// newProgs returns a new Progs for fn.
func newProgs(fn *Node) *Progs {
	pp := new(Progs)
	if Ctxt.CanReuseProgs() {
		pp.progcache = sharedProgArray[:]
	}
	pp.curfn = fn

	// prime the pump
	pp.next = pp.NewProg()
	pp.clearp(pp.next)

	pp.pos = fn.Pos
	pp.settext(fn)
	return pp
}

func (pp *Progs) NewProg() *obj.Prog {
	if pp.cacheidx < len(pp.progcache) {
		p := &pp.progcache[pp.cacheidx]
		p.Ctxt = Ctxt
		pp.cacheidx++
		return p
	}
	p := new(obj.Prog)
	p.Ctxt = Ctxt
	return p
}

// Flush converts from pp to machine code.
func (pp *Progs) Flush() {
	plist := &obj.Plist{Firstpc: pp.Text, Curfn: pp.curfn}
	obj.Flushplist(Ctxt, plist, pp.NewProg)
}

// Free clears pp and any associated resources.
func (pp *Progs) Free() {
	if Ctxt.CanReuseProgs() {
		// Clear progs to enable GC and avoid abuse.
		s := pp.progcache[:pp.cacheidx]
		for i := range s {
			s[i] = obj.Prog{}
		}
	}
	// Clear pp to avoid abuse.
	*pp = Progs{}
}

// Prog adds a Prog with instruction As to pp.
func (pp *Progs) Prog(as obj.As) *obj.Prog {
	p := pp.next
	pp.next = pp.NewProg()
	pp.clearp(pp.next)
	p.Link = pp.next

	if !pp.pos.IsKnown() && Debug['K'] != 0 {
		Warn("prog: unknown position (line 0)")
	}

	p.As = as
	p.Pos = pp.pos
	return p
}

func (pp *Progs) clearp(p *obj.Prog) {
	obj.Nopout(p)
	p.As = obj.AEND
	p.Pc = pp.pc
	pp.pc++
}

func (pp *Progs) Appendpp(p *obj.Prog, as obj.As, ftype obj.AddrType, freg int16, foffset int64, ttype obj.AddrType, treg int16, toffset int64) *obj.Prog {
	q := pp.NewProg()
	pp.clearp(q)
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

func (pp *Progs) settext(fn *Node) {
	if pp.Text != nil {
		Fatalf("Progs.settext called twice")
	}

	ptxt := pp.Prog(obj.ATEXT)
	if nam := fn.Func.Nname; !isblank(nam) {
		ptxt.From.Type = obj.TYPE_MEM
		ptxt.From.Name = obj.NAME_EXTERN
		ptxt.From.Sym = Linksym(nam.Sym)
		if fn.Func.Pragma&Systemstack != 0 {
			ptxt.From.Sym.Set(obj.AttrCFunc, true)
		}
	}

	ptxt.From3 = new(obj.Addr)
	if fn.Func.Dupok() {
		ptxt.From3.Offset |= obj.DUPOK
	}
	if fn.Func.Wrapper() {
		ptxt.From3.Offset |= obj.WRAPPER
	}
	if fn.Func.NoFramePointer() {
		ptxt.From3.Offset |= obj.NOFRAME
	}
	if fn.Func.Needctxt() {
		ptxt.From3.Offset |= obj.NEEDCTXT
	}
	if fn.Func.Pragma&Nosplit != 0 {
		ptxt.From3.Offset |= obj.NOSPLIT
	}
	if fn.Func.ReflectMethod() {
		ptxt.From3.Offset |= obj.REFLECTMETHOD
	}

	// Clumsy but important.
	// See test/recover.go for test cases and src/reflect/value.go
	// for the actual functions being considered.
	if myimportpath == "reflect" {
		switch fn.Func.Nname.Sym.Name {
		case "callReflect", "callMethod":
			ptxt.From3.Offset |= obj.WRAPPER
		}
	}

	Ctxt.InitTextSym(ptxt)

	pp.Text = ptxt
}

func ggloblnod(nam *Node) {
	s := Linksym(nam.Sym)
	s.Gotype = Linksym(ngotype(nam))
	flags := 0
	if nam.Name.Readonly() {
		flags = obj.RODATA
	}
	if nam.Type != nil && !types.Haspointers(nam.Type) {
		flags |= obj.NOPTR
	}
	Ctxt.Globl(s, nam.Type.Width, flags)
}

func ggloblsym(s *types.Sym, width int32, flags int16) {
	ggloblLSym(Linksym(s), width, flags)
}

func ggloblLSym(s *obj.LSym, width int32, flags int16) {
	if flags&obj.LOCAL != 0 {
		s.Set(obj.AttrLocal, true)
		flags &^= obj.LOCAL
	}
	Ctxt.Globl(s, int64(width), int(flags))
}

func isfat(t *types.Type) bool {
	if t != nil {
		switch t.Etype {
		case TSTRUCT, TARRAY, TSLICE, TSTRING,
			TINTER: // maybe remove later
			return true
		}
	}

	return false
}

func Addrconst(a *obj.Addr, v int64) {
	a.Sym = nil
	a.Type = obj.TYPE_CONST
	a.Offset = v
}

// nodarg returns a Node for the function argument denoted by t,
// which is either the entire function argument or result struct (t is a  struct *types.Type)
// or a specific argument (t is a *types.Field within a struct *types.Type).
//
// If fp is 0, the node is for use by a caller invoking the given
// function, preparing the arguments before the call
// or retrieving the results after the call.
// In this case, the node will correspond to an outgoing argument
// slot like 8(SP).
//
// If fp is 1, the node is for use by the function itself
// (the callee), to retrieve its arguments or write its results.
// In this case the node will be an ONAME with an appropriate
// type and offset.
func nodarg(t interface{}, fp int) *Node {
	var n *Node

	var funarg types.Funarg
	switch t := t.(type) {
	default:
		Fatalf("bad nodarg %T(%v)", t, t)

	case *types.Type:
		// Entire argument struct, not just one arg
		if !t.IsFuncArgStruct() {
			Fatalf("nodarg: bad type %v", t)
		}
		funarg = t.StructType().Funarg

		// Build fake variable name for whole arg struct.
		n = newname(lookup(".args"))
		n.Type = t
		first := t.Field(0)
		if first == nil {
			Fatalf("nodarg: bad struct")
		}
		if first.Offset == BADWIDTH {
			Fatalf("nodarg: offset not computed for %v", t)
		}
		n.Xoffset = first.Offset

	case *types.Field:
		funarg = t.Funarg
		if fp == 1 {
			// NOTE(rsc): This should be using t.Nname directly,
			// except in the case where t.Nname.Sym is the blank symbol and
			// so the assignment would be discarded during code generation.
			// In that case we need to make a new node, and there is no harm
			// in optimization passes to doing so. But otherwise we should
			// definitely be using the actual declaration and not a newly built node.
			// The extra Fatalf checks here are verifying that this is the case,
			// without changing the actual logic (at time of writing, it's getting
			// toward time for the Go 1.7 beta).
			// At some quieter time (assuming we've never seen these Fatalfs happen)
			// we could change this code to use "expect" directly.
			expect := asNode(t.Nname)
			if expect.isParamHeapCopy() {
				expect = expect.Name.Param.Stackcopy
			}

			for _, n := range Curfn.Func.Dcl {
				if (n.Class == PPARAM || n.Class == PPARAMOUT) && !isblanksym(t.Sym) && n.Sym == t.Sym {
					if n != expect {
						Fatalf("nodarg: unexpected node: %v (%p %v) vs %v (%p %v)", n, n, n.Op, asNode(t.Nname), asNode(t.Nname), asNode(t.Nname).Op)
					}
					return n
				}
			}

			if !isblanksym(expect.Sym) {
				Fatalf("nodarg: did not find node in dcl list: %v", expect)
			}
		}

		// Build fake name for individual variable.
		// This is safe because if there was a real declared name
		// we'd have used it above.
		n = newname(lookup("__"))
		n.Type = t.Type
		if t.Offset == BADWIDTH {
			Fatalf("nodarg: offset not computed for %v", t)
		}
		n.Xoffset = t.Offset
		n.Orig = asNode(t.Nname)
	}

	// Rewrite argument named _ to __,
	// or else the assignment to _ will be
	// discarded during code generation.
	if isblank(n) {
		n.Sym = lookup("__")
	}

	switch fp {
	default:
		Fatalf("bad fp")

	case 0: // preparing arguments for call
		n.Op = OINDREGSP
		n.Xoffset += Ctxt.FixedFrameSize()

	case 1: // reading arguments inside call
		n.Class = PPARAM
		if funarg == types.FunargResults {
			n.Class = PPARAMOUT
		}
	}

	n.Typecheck = 1
	n.SetAddrtaken(true) // keep optimizers at bay
	return n
}

func Patch(p *obj.Prog, to *obj.Prog) {
	if p.To.Type != obj.TYPE_BRANCH {
		Fatalf("patch: not a branch")
	}
	p.To.Val = to
	p.To.Offset = to.Pc
}
