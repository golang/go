// Derived from Inferno utils/6c/reg.c
// http://code.google.com/p/inferno-os/source/browse/utils/6c/reg.c
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

package gc

import (
	"bytes"
	"cmd/internal/obj"
	"fmt"
	"sort"
	"strings"
)

// A Var represents a single variable that may be stored in a register.
// That variable may itself correspond to a hardware register,
// to represent the use of registers in the unoptimized instruction stream.
type Var struct {
	offset     int64
	node       *Node
	nextinnode *Var
	width      int
	id         int // index in vars
	name       int8
	etype      EType
	addr       int8
}

// Bits represents a set of Vars, stored as a bit set of var numbers
// (the index in vars, or equivalently v.id).
type Bits struct {
	b [BITS]uint64
}

const (
	BITS = 3
	NVAR = BITS * 64
)

var (
	vars [NVAR]Var // variables under consideration
	nvar int       // number of vars

	regbits uint64 // bits for hardware registers

	zbits   Bits // zero
	externs Bits // global variables
	params  Bits // function parameters and results
	ivar    Bits // function parameters (inputs)
	ovar    Bits // function results (outputs)
	consts  Bits // constant values
	addrs   Bits // variables with address taken
)

// A Reg is a wrapper around a single Prog (one instruction) that holds
// register optimization information while the optimizer runs.
// r->prog is the instruction.
type Reg struct {
	set  Bits // regopt variables written by this instruction.
	use1 Bits // regopt variables read by prog->from.
	use2 Bits // regopt variables read by prog->to.

	// refahead/refbehind are the regopt variables whose current
	// value may be used in the following/preceding instructions
	// up to a CALL (or the value is clobbered).
	refbehind Bits
	refahead  Bits

	// calahead/calbehind are similar, but for variables in
	// instructions that are reachable after hitting at least one
	// CALL.
	calbehind Bits
	calahead  Bits

	regdiff Bits
	act     Bits
	regu    uint64 // register used bitmap
}

// A Rgn represents a single regopt variable over a region of code
// where a register could potentially be dedicated to that variable.
// The code encompassed by a Rgn is defined by the flow graph,
// starting at enter, flood-filling forward while varno is refahead
// and backward while varno is refbehind, and following branches.
// A single variable may be represented by multiple disjoint Rgns and
// each Rgn may choose a different register for that variable.
// Registers are allocated to regions greedily in order of descending
// cost.
type Rgn struct {
	enter *Flow
	cost  int16
	varno int16
	regno int16
}

// The Plan 9 C compilers used a limit of 600 regions,
// but the yacc-generated parser in y.go has 3100 regions.
// We set MaxRgn large enough to handle that.
// There's not a huge cost to having too many regions:
// the main processing traces the live area for each variable,
// which is limited by the number of variables times the area,
// not the raw region count. If there are many regions, they
// are almost certainly small and easy to trace.
// The only operation that scales with region count is the
// sorting by cost, which uses sort.Sort and is therefore
// guaranteed n log n.
const MaxRgn = 6000

var (
	region  []Rgn
	nregion int
)

type rcmp []Rgn

func (x rcmp) Len() int {
	return len(x)
}

func (x rcmp) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}

func (x rcmp) Less(i, j int) bool {
	p1 := &x[i]
	p2 := &x[j]
	if p1.cost != p2.cost {
		return int(p2.cost)-int(p1.cost) < 0
	}
	if p1.varno != p2.varno {
		return int(p2.varno)-int(p1.varno) < 0
	}
	if p1.enter != p2.enter {
		return int(p2.enter.Id-p1.enter.Id) < 0
	}
	return false
}

func setaddrs(bit Bits) {
	var i int
	var n int
	var v *Var
	var node *Node

	for bany(&bit) {
		// convert each bit to a variable
		i = bnum(&bit)

		node = vars[i].node
		n = int(vars[i].name)
		biclr(&bit, uint(i))

		// disable all pieces of that variable
		for i = 0; i < nvar; i++ {
			v = &vars[i]
			if v.node == node && int(v.name) == n {
				v.addr = 2
			}
		}
	}
}

var regnodes [64]*Node

func walkvardef(n *Node, f *Flow, active int) {
	var f1 *Flow
	var bn int
	var v *Var

	for f1 = f; f1 != nil; f1 = f1.S1 {
		if f1.Active == int32(active) {
			break
		}
		f1.Active = int32(active)
		if f1.Prog.As == obj.AVARKILL && f1.Prog.To.Node == n {
			break
		}
		for v, _ = n.Opt().(*Var); v != nil; v = v.nextinnode {
			bn = v.id
			biset(&(f1.Data.(*Reg)).act, uint(bn))
		}

		if f1.Prog.As == obj.ACALL {
			break
		}
	}

	for f2 := f; f2 != f1; f2 = f2.S1 {
		if f2.S2 != nil {
			walkvardef(n, f2.S2, active)
		}
	}
}

// add mov b,rn
// just after r
func addmove(r *Flow, bn int, rn int, f int) {
	p1 := Ctxt.NewProg()
	Clearp(p1)
	p1.Pc = 9999

	p := r.Prog
	p1.Link = p.Link
	p.Link = p1
	p1.Lineno = p.Lineno

	v := &vars[bn]

	a := &p1.To
	a.Offset = v.offset
	a.Etype = uint8(v.etype)
	a.Type = obj.TYPE_MEM
	a.Name = v.name
	a.Node = v.node
	a.Sym = Linksym(v.node.Sym)

	/* NOTE(rsc): 9g did
	if(a->etype == TARRAY)
		a->type = TYPE_ADDR;
	else if(a->sym == nil)
		a->type = TYPE_CONST;
	*/
	p1.As = int16(Thearch.Optoas(OAS, Types[uint8(v.etype)]))

	// TODO(rsc): Remove special case here.
	if (Thearch.Thechar == '0' || Thearch.Thechar == '5' || Thearch.Thechar == '7' || Thearch.Thechar == '9') && v.etype == TBOOL {
		p1.As = int16(Thearch.Optoas(OAS, Types[TUINT8]))
	}
	p1.From.Type = obj.TYPE_REG
	p1.From.Reg = int16(rn)
	p1.From.Name = obj.NAME_NONE
	if f == 0 {
		p1.From = *a
		*a = obj.Addr{}
		a.Type = obj.TYPE_REG
		a.Reg = int16(rn)
	}

	if Debug['R'] != 0 && Debug['v'] != 0 {
		fmt.Printf("%v ===add=== %v\n", p, p1)
	}
	Ostats.Nspill++
}

func overlap_reg(o1 int64, w1 int, o2 int64, w2 int) bool {
	t1 := o1 + int64(w1)
	t2 := o2 + int64(w2)

	if t1 <= o2 || t2 <= o1 {
		return false
	}

	return true
}

func mkvar(f *Flow, a *obj.Addr) Bits {
	// mark registers used
	if a.Type == obj.TYPE_NONE {
		return zbits
	}

	r := f.Data.(*Reg)
	r.use1.b[0] |= Thearch.Doregbits(int(a.Index)) // TODO: Use RtoB

	var n int
	switch a.Type {
	default:
		regu := Thearch.Doregbits(int(a.Reg)) | Thearch.RtoB(int(a.Reg)) // TODO: Use RtoB
		if regu == 0 {
			return zbits
		}
		bit := zbits
		bit.b[0] = regu
		return bit

		// TODO(rsc): Remove special case here.
	case obj.TYPE_ADDR:
		var bit Bits
		if Thearch.Thechar == '0' || Thearch.Thechar == '5' || Thearch.Thechar == '7' || Thearch.Thechar == '9' {
			goto memcase
		}
		a.Type = obj.TYPE_MEM
		bit = mkvar(f, a)
		setaddrs(bit)
		a.Type = obj.TYPE_ADDR
		Ostats.Naddr++
		return zbits

	memcase:
		fallthrough

	case obj.TYPE_MEM:
		if r != nil {
			r.use1.b[0] |= Thearch.RtoB(int(a.Reg))
		}

		/* NOTE: 5g did
		if(r->f.prog->scond & (C_PBIT|C_WBIT))
			r->set.b[0] |= RtoB(a->reg);
		*/
		switch a.Name {
		default:
			// Note: This case handles NAME_EXTERN and NAME_STATIC.
			// We treat these as requiring eager writes to memory, due to
			// the possibility of a fault handler looking at them, so there is
			// not much point in registerizing the loads.
			// If we later choose the set of candidate variables from a
			// larger list, these cases could be deprioritized instead of
			// removed entirely.
			return zbits

		case obj.NAME_PARAM,
			obj.NAME_AUTO:
			n = int(a.Name)
		}
	}

	node, _ := a.Node.(*Node)
	if node == nil || node.Op != ONAME || node.Orig == nil {
		return zbits
	}
	node = node.Orig
	if node.Orig != node {
		Fatalf("%v: bad node", Ctxt.Dconv(a))
	}
	if node.Sym == nil || node.Sym.Name[0] == '.' {
		return zbits
	}
	et := EType(a.Etype)
	o := a.Offset
	w := a.Width
	if w < 0 {
		Fatalf("bad width %d for %v", w, Ctxt.Dconv(a))
	}

	flag := 0
	var v *Var
	for i := 0; i < nvar; i++ {
		v = &vars[i]
		if v.node == node && int(v.name) == n {
			if v.offset == o {
				if v.etype == et {
					if int64(v.width) == w {
						// TODO(rsc): Remove special case for arm here.
						if flag == 0 || Thearch.Thechar != '5' {
							return blsh(uint(i))
						}
					}
				}
			}

			// if they overlap, disable both
			if overlap_reg(v.offset, v.width, o, int(w)) {
				//				print("disable overlap %s %d %d %d %d, %E != %E\n", s->name, v->offset, v->width, o, w, v->etype, et);
				v.addr = 1

				flag = 1
			}
		}
	}

	switch et {
	case 0, TFUNC:
		return zbits
	}

	if nvar >= NVAR {
		if Debug['w'] > 1 && node != nil {
			Fatalf("variable not optimized: %v", Nconv(node, obj.FmtSharp))
		}
		if Debug['v'] > 0 {
			Warn("variable not optimized: %v", Nconv(node, obj.FmtSharp))
		}

		// If we're not tracking a word in a variable, mark the rest as
		// having its address taken, so that we keep the whole thing
		// live at all calls. otherwise we might optimize away part of
		// a variable but not all of it.
		var v *Var
		for i := 0; i < nvar; i++ {
			v = &vars[i]
			if v.node == node {
				v.addr = 1
			}
		}

		return zbits
	}

	i := nvar
	nvar++
	v = &vars[i]
	v.id = i
	v.offset = o
	v.name = int8(n)
	v.etype = et
	v.width = int(w)
	v.addr = int8(flag) // funny punning
	v.node = node

	// node->opt is the head of a linked list
	// of Vars within the given Node, so that
	// we can start at a Var and find all the other
	// Vars in the same Go variable.
	v.nextinnode, _ = node.Opt().(*Var)

	node.SetOpt(v)

	bit := blsh(uint(i))
	if n == obj.NAME_EXTERN || n == obj.NAME_STATIC {
		for z := 0; z < BITS; z++ {
			externs.b[z] |= bit.b[z]
		}
	}
	if n == obj.NAME_PARAM {
		for z := 0; z < BITS; z++ {
			params.b[z] |= bit.b[z]
		}
	}

	if node.Class == PPARAM {
		for z := 0; z < BITS; z++ {
			ivar.b[z] |= bit.b[z]
		}
	}
	if node.Class == PPARAMOUT {
		for z := 0; z < BITS; z++ {
			ovar.b[z] |= bit.b[z]
		}
	}

	// Treat values with their address taken as live at calls,
	// because the garbage collector's liveness analysis in plive.go does.
	// These must be consistent or else we will elide stores and the garbage
	// collector will see uninitialized data.
	// The typical case where our own analysis is out of sync is when the
	// node appears to have its address taken but that code doesn't actually
	// get generated and therefore doesn't show up as an address being
	// taken when we analyze the instruction stream.
	// One instance of this case is when a closure uses the same name as
	// an outer variable for one of its own variables declared with :=.
	// The parser flags the outer variable as possibly shared, and therefore
	// sets addrtaken, even though it ends up not being actually shared.
	// If we were better about _ elision, _ = &x would suffice too.
	// The broader := in a closure problem is mentioned in a comment in
	// closure.go:/^typecheckclosure and dcl.go:/^oldname.
	if node.Addrtaken {
		v.addr = 1
	}

	// Disable registerization for globals, because:
	// (1) we might panic at any time and we want the recovery code
	// to see the latest values (issue 1304).
	// (2) we don't know what pointers might point at them and we want
	// loads via those pointers to see updated values and vice versa (issue 7995).
	//
	// Disable registerization for results if using defer, because the deferred func
	// might recover and return, causing the current values to be used.
	if node.Class == PEXTERN || (hasdefer && node.Class == PPARAMOUT) {
		v.addr = 1
	}

	if Debug['R'] != 0 {
		fmt.Printf("bit=%2d et=%v w=%d+%d %v %v flag=%d\n", i, Econv(et), o, w, Nconv(node, obj.FmtSharp), Ctxt.Dconv(a), v.addr)
	}
	Ostats.Nvar++

	return bit
}

var change int

func prop(f *Flow, ref Bits, cal Bits) {
	var f1 *Flow
	var r1 *Reg
	var z int
	var i int
	var v *Var
	var v1 *Var

	for f1 = f; f1 != nil; f1 = f1.P1 {
		r1 = f1.Data.(*Reg)
		for z = 0; z < BITS; z++ {
			ref.b[z] |= r1.refahead.b[z]
			if ref.b[z] != r1.refahead.b[z] {
				r1.refahead.b[z] = ref.b[z]
				change = 1
			}

			cal.b[z] |= r1.calahead.b[z]
			if cal.b[z] != r1.calahead.b[z] {
				r1.calahead.b[z] = cal.b[z]
				change = 1
			}
		}

		switch f1.Prog.As {
		case obj.ACALL:
			if Noreturn(f1.Prog) {
				break
			}

			// Mark all input variables (ivar) as used, because that's what the
			// liveness bitmaps say. The liveness bitmaps say that so that a
			// panic will not show stale values in the parameter dump.
			// Mark variables with a recent VARDEF (r1->act) as used,
			// so that the optimizer flushes initializations to memory,
			// so that if a garbage collection happens during this CALL,
			// the collector will see initialized memory. Again this is to
			// match what the liveness bitmaps say.
			for z = 0; z < BITS; z++ {
				cal.b[z] |= ref.b[z] | externs.b[z] | ivar.b[z] | r1.act.b[z]
				ref.b[z] = 0
			}

			// cal.b is the current approximation of what's live across the call.
			// Every bit in cal.b is a single stack word. For each such word,
			// find all the other tracked stack words in the same Go variable
			// (struct/slice/string/interface) and mark them live too.
			// This is necessary because the liveness analysis for the garbage
			// collector works at variable granularity, not at word granularity.
			// It is fundamental for slice/string/interface: the garbage collector
			// needs the whole value, not just some of the words, in order to
			// interpret the other bits correctly. Specifically, slice needs a consistent
			// ptr and cap, string needs a consistent ptr and len, and interface
			// needs a consistent type word and data word.
			for z = 0; z < BITS; z++ {
				if cal.b[z] == 0 {
					continue
				}
				for i = 0; i < 64; i++ {
					if z*64+i >= nvar || (cal.b[z]>>uint(i))&1 == 0 {
						continue
					}
					v = &vars[z*64+i]
					if v.node.Opt() == nil { // v represents fixed register, not Go variable
						continue
					}

					// v->node->opt is the head of a linked list of Vars
					// corresponding to tracked words from the Go variable v->node.
					// Walk the list and set all the bits.
					// For a large struct this could end up being quadratic:
					// after the first setting, the outer loop (for z, i) would see a 1 bit
					// for all of the remaining words in the struct, and for each such
					// word would go through and turn on all the bits again.
					// To avoid the quadratic behavior, we only turn on the bits if
					// v is the head of the list or if the head's bit is not yet turned on.
					// This will set the bits at most twice, keeping the overall loop linear.
					v1, _ = v.node.Opt().(*Var)

					if v == v1 || !btest(&cal, uint(v1.id)) {
						for ; v1 != nil; v1 = v1.nextinnode {
							biset(&cal, uint(v1.id))
						}
					}
				}
			}

		case obj.ATEXT:
			for z = 0; z < BITS; z++ {
				cal.b[z] = 0
				ref.b[z] = 0
			}

		case obj.ARET:
			for z = 0; z < BITS; z++ {
				cal.b[z] = externs.b[z] | ovar.b[z]
				ref.b[z] = 0
			}
		}

		for z = 0; z < BITS; z++ {
			ref.b[z] = ref.b[z]&^r1.set.b[z] | r1.use1.b[z] | r1.use2.b[z]
			cal.b[z] &^= (r1.set.b[z] | r1.use1.b[z] | r1.use2.b[z])
			r1.refbehind.b[z] = ref.b[z]
			r1.calbehind.b[z] = cal.b[z]
		}

		if f1.Active != 0 {
			break
		}
		f1.Active = 1
	}

	var r *Reg
	var f2 *Flow
	for ; f != f1; f = f.P1 {
		r = f.Data.(*Reg)
		for f2 = f.P2; f2 != nil; f2 = f2.P2link {
			prop(f2, r.refbehind, r.calbehind)
		}
	}
}

func synch(f *Flow, dif Bits) {
	var r1 *Reg
	var z int

	for f1 := f; f1 != nil; f1 = f1.S1 {
		r1 = f1.Data.(*Reg)
		for z = 0; z < BITS; z++ {
			dif.b[z] = dif.b[z]&^(^r1.refbehind.b[z]&r1.refahead.b[z]) | r1.set.b[z] | r1.regdiff.b[z]
			if dif.b[z] != r1.regdiff.b[z] {
				r1.regdiff.b[z] = dif.b[z]
				change = 1
			}
		}

		if f1.Active != 0 {
			break
		}
		f1.Active = 1
		for z = 0; z < BITS; z++ {
			dif.b[z] &^= (^r1.calbehind.b[z] & r1.calahead.b[z])
		}
		if f1.S2 != nil {
			synch(f1.S2, dif)
		}
	}
}

func allreg(b uint64, r *Rgn) uint64 {
	v := &vars[r.varno]
	r.regno = 0
	switch v.etype {
	default:
		Fatalf("unknown etype %d/%v", Bitno(b), Econv(v.etype))

	case TINT8,
		TUINT8,
		TINT16,
		TUINT16,
		TINT32,
		TUINT32,
		TINT64,
		TUINT64,
		TINT,
		TUINT,
		TUINTPTR,
		TBOOL,
		TPTR32,
		TPTR64:
		i := Thearch.BtoR(^b)
		if i != 0 && r.cost > 0 {
			r.regno = int16(i)
			return Thearch.RtoB(i)
		}

	case TFLOAT32, TFLOAT64:
		i := Thearch.BtoF(^b)
		if i != 0 && r.cost > 0 {
			r.regno = int16(i)
			return Thearch.FtoB(i)
		}
	}

	return 0
}

func LOAD(r *Reg, z int) uint64 {
	return ^r.refbehind.b[z] & r.refahead.b[z]
}

func STORE(r *Reg, z int) uint64 {
	return ^r.calbehind.b[z] & r.calahead.b[z]
}

// Cost parameters
const (
	CLOAD = 5 // cost of load
	CREF  = 5 // cost of reference if not registerized
	LOOP  = 3 // loop execution count (applied in popt.go)
)

func paint1(f *Flow, bn int) {
	z := bn / 64
	bb := uint64(1 << uint(bn%64))
	r := f.Data.(*Reg)
	if r.act.b[z]&bb != 0 {
		return
	}
	var f1 *Flow
	var r1 *Reg
	for {
		if r.refbehind.b[z]&bb == 0 {
			break
		}
		f1 = f.P1
		if f1 == nil {
			break
		}
		r1 = f1.Data.(*Reg)
		if r1.refahead.b[z]&bb == 0 {
			break
		}
		if r1.act.b[z]&bb != 0 {
			break
		}
		f = f1
		r = r1
	}

	if LOAD(r, z)&^(r.set.b[z]&^(r.use1.b[z]|r.use2.b[z]))&bb != 0 {
		change -= CLOAD * int(f.Loop)
	}

	for {
		r.act.b[z] |= bb

		if f.Prog.As != obj.ANOP { // don't give credit for NOPs
			if r.use1.b[z]&bb != 0 {
				change += CREF * int(f.Loop)
			}
			if (r.use2.b[z]|r.set.b[z])&bb != 0 {
				change += CREF * int(f.Loop)
			}
		}

		if STORE(r, z)&r.regdiff.b[z]&bb != 0 {
			change -= CLOAD * int(f.Loop)
		}

		if r.refbehind.b[z]&bb != 0 {
			for f1 = f.P2; f1 != nil; f1 = f1.P2link {
				if (f1.Data.(*Reg)).refahead.b[z]&bb != 0 {
					paint1(f1, bn)
				}
			}
		}

		if r.refahead.b[z]&bb == 0 {
			break
		}
		f1 = f.S2
		if f1 != nil {
			if (f1.Data.(*Reg)).refbehind.b[z]&bb != 0 {
				paint1(f1, bn)
			}
		}
		f = f.S1
		if f == nil {
			break
		}
		r = f.Data.(*Reg)
		if r.act.b[z]&bb != 0 {
			break
		}
		if r.refbehind.b[z]&bb == 0 {
			break
		}
	}
}

func paint2(f *Flow, bn int, depth int) uint64 {
	z := bn / 64
	bb := uint64(1 << uint(bn%64))
	vreg := regbits
	r := f.Data.(*Reg)
	if r.act.b[z]&bb == 0 {
		return vreg
	}
	var r1 *Reg
	var f1 *Flow
	for {
		if r.refbehind.b[z]&bb == 0 {
			break
		}
		f1 = f.P1
		if f1 == nil {
			break
		}
		r1 = f1.Data.(*Reg)
		if r1.refahead.b[z]&bb == 0 {
			break
		}
		if r1.act.b[z]&bb == 0 {
			break
		}
		f = f1
		r = r1
	}

	for {
		if Debug['R'] != 0 && Debug['v'] != 0 {
			fmt.Printf("  paint2 %d %v\n", depth, f.Prog)
		}

		r.act.b[z] &^= bb

		vreg |= r.regu

		if r.refbehind.b[z]&bb != 0 {
			for f1 = f.P2; f1 != nil; f1 = f1.P2link {
				if (f1.Data.(*Reg)).refahead.b[z]&bb != 0 {
					vreg |= paint2(f1, bn, depth+1)
				}
			}
		}

		if r.refahead.b[z]&bb == 0 {
			break
		}
		f1 = f.S2
		if f1 != nil {
			if (f1.Data.(*Reg)).refbehind.b[z]&bb != 0 {
				vreg |= paint2(f1, bn, depth+1)
			}
		}
		f = f.S1
		if f == nil {
			break
		}
		r = f.Data.(*Reg)
		if r.act.b[z]&bb == 0 {
			break
		}
		if r.refbehind.b[z]&bb == 0 {
			break
		}
	}

	return vreg
}

func paint3(f *Flow, bn int, rb uint64, rn int) {
	z := bn / 64
	bb := uint64(1 << uint(bn%64))
	r := f.Data.(*Reg)
	if r.act.b[z]&bb != 0 {
		return
	}
	var r1 *Reg
	var f1 *Flow
	for {
		if r.refbehind.b[z]&bb == 0 {
			break
		}
		f1 = f.P1
		if f1 == nil {
			break
		}
		r1 = f1.Data.(*Reg)
		if r1.refahead.b[z]&bb == 0 {
			break
		}
		if r1.act.b[z]&bb != 0 {
			break
		}
		f = f1
		r = r1
	}

	if LOAD(r, z)&^(r.set.b[z]&^(r.use1.b[z]|r.use2.b[z]))&bb != 0 {
		addmove(f, bn, rn, 0)
	}
	var p *obj.Prog
	for {
		r.act.b[z] |= bb
		p = f.Prog

		if r.use1.b[z]&bb != 0 {
			if Debug['R'] != 0 && Debug['v'] != 0 {
				fmt.Printf("%v", p)
			}
			addreg(&p.From, rn)
			if Debug['R'] != 0 && Debug['v'] != 0 {
				fmt.Printf(" ===change== %v\n", p)
			}
		}

		if (r.use2.b[z]|r.set.b[z])&bb != 0 {
			if Debug['R'] != 0 && Debug['v'] != 0 {
				fmt.Printf("%v", p)
			}
			addreg(&p.To, rn)
			if Debug['R'] != 0 && Debug['v'] != 0 {
				fmt.Printf(" ===change== %v\n", p)
			}
		}

		if STORE(r, z)&r.regdiff.b[z]&bb != 0 {
			addmove(f, bn, rn, 1)
		}
		r.regu |= rb

		if r.refbehind.b[z]&bb != 0 {
			for f1 = f.P2; f1 != nil; f1 = f1.P2link {
				if (f1.Data.(*Reg)).refahead.b[z]&bb != 0 {
					paint3(f1, bn, rb, rn)
				}
			}
		}

		if r.refahead.b[z]&bb == 0 {
			break
		}
		f1 = f.S2
		if f1 != nil {
			if (f1.Data.(*Reg)).refbehind.b[z]&bb != 0 {
				paint3(f1, bn, rb, rn)
			}
		}
		f = f.S1
		if f == nil {
			break
		}
		r = f.Data.(*Reg)
		if r.act.b[z]&bb != 0 {
			break
		}
		if r.refbehind.b[z]&bb == 0 {
			break
		}
	}
}

func addreg(a *obj.Addr, rn int) {
	a.Sym = nil
	a.Node = nil
	a.Offset = 0
	a.Type = obj.TYPE_REG
	a.Reg = int16(rn)
	a.Name = 0

	Ostats.Ncvtreg++
}

func dumpone(f *Flow, isreg int) {
	fmt.Printf("%d:%v", f.Loop, f.Prog)
	if isreg != 0 {
		r := f.Data.(*Reg)
		var bit Bits
		for z := 0; z < BITS; z++ {
			bit.b[z] = r.set.b[z] | r.use1.b[z] | r.use2.b[z] | r.refbehind.b[z] | r.refahead.b[z] | r.calbehind.b[z] | r.calahead.b[z] | r.regdiff.b[z] | r.act.b[z] | 0
		}
		if bany(&bit) {
			fmt.Printf("\t")
			if bany(&r.set) {
				fmt.Printf(" s:%v", &r.set)
			}
			if bany(&r.use1) {
				fmt.Printf(" u1:%v", &r.use1)
			}
			if bany(&r.use2) {
				fmt.Printf(" u2:%v", &r.use2)
			}
			if bany(&r.refbehind) {
				fmt.Printf(" rb:%v ", &r.refbehind)
			}
			if bany(&r.refahead) {
				fmt.Printf(" ra:%v ", &r.refahead)
			}
			if bany(&r.calbehind) {
				fmt.Printf(" cb:%v ", &r.calbehind)
			}
			if bany(&r.calahead) {
				fmt.Printf(" ca:%v ", &r.calahead)
			}
			if bany(&r.regdiff) {
				fmt.Printf(" d:%v ", &r.regdiff)
			}
			if bany(&r.act) {
				fmt.Printf(" a:%v ", &r.act)
			}
		}
	}

	fmt.Printf("\n")
}

func Dumpit(str string, r0 *Flow, isreg int) {
	var r1 *Flow

	fmt.Printf("\n%s\n", str)
	for r := r0; r != nil; r = r.Link {
		dumpone(r, isreg)
		r1 = r.P2
		if r1 != nil {
			fmt.Printf("\tpred:")
			for ; r1 != nil; r1 = r1.P2link {
				fmt.Printf(" %.4d", uint(int(r1.Prog.Pc)))
			}
			if r.P1 != nil {
				fmt.Printf(" (and %.4d)", uint(int(r.P1.Prog.Pc)))
			} else {
				fmt.Printf(" (only)")
			}
			fmt.Printf("\n")
		}

		// Print successors if it's not just the next one
		if r.S1 != r.Link || r.S2 != nil {
			fmt.Printf("\tsucc:")
			if r.S1 != nil {
				fmt.Printf(" %.4d", uint(int(r.S1.Prog.Pc)))
			}
			if r.S2 != nil {
				fmt.Printf(" %.4d", uint(int(r.S2.Prog.Pc)))
			}
			fmt.Printf("\n")
		}
	}
}

func regopt(firstp *obj.Prog) {
	mergetemp(firstp)

	// control flow is more complicated in generated go code
	// than in generated c code.  define pseudo-variables for
	// registers, so we have complete register usage information.
	var nreg int
	regnames := Thearch.Regnames(&nreg)

	nvar = nreg
	for i := 0; i < nreg; i++ {
		vars[i] = Var{}
	}
	for i := 0; i < nreg; i++ {
		if regnodes[i] == nil {
			regnodes[i] = newname(Lookup(regnames[i]))
		}
		vars[i].node = regnodes[i]
	}

	regbits = Thearch.Excludedregs()
	externs = zbits
	params = zbits
	consts = zbits
	addrs = zbits
	ivar = zbits
	ovar = zbits

	// pass 1
	// build aux data structure
	// allocate pcs
	// find use and set of variables
	g := Flowstart(firstp, func() interface{} { return new(Reg) })
	if g == nil {
		for i := 0; i < nvar; i++ {
			vars[i].node.SetOpt(nil)
		}
		return
	}

	firstf := g.Start

	for f := firstf; f != nil; f = f.Link {
		p := f.Prog
		// AVARLIVE must be considered a use, do not skip it.
		// Otherwise the variable will be optimized away,
		// and the whole point of AVARLIVE is to keep it on the stack.
		if p.As == obj.AVARDEF || p.As == obj.AVARKILL {
			continue
		}

		// Avoid making variables for direct-called functions.
		if p.As == obj.ACALL && p.To.Type == obj.TYPE_MEM && p.To.Name == obj.NAME_EXTERN {
			continue
		}

		// from vs to doesn't matter for registers.
		r := f.Data.(*Reg)
		r.use1.b[0] |= p.Info.Reguse | p.Info.Regindex
		r.set.b[0] |= p.Info.Regset

		bit := mkvar(f, &p.From)
		if bany(&bit) {
			if p.Info.Flags&LeftAddr != 0 {
				setaddrs(bit)
			}
			if p.Info.Flags&LeftRead != 0 {
				for z := 0; z < BITS; z++ {
					r.use1.b[z] |= bit.b[z]
				}
			}
			if p.Info.Flags&LeftWrite != 0 {
				for z := 0; z < BITS; z++ {
					r.set.b[z] |= bit.b[z]
				}
			}
		}

		// Compute used register for reg
		if p.Info.Flags&RegRead != 0 {
			r.use1.b[0] |= Thearch.RtoB(int(p.Reg))
		}

		// Currently we never generate three register forms.
		// If we do, this will need to change.
		if p.From3Type() != obj.TYPE_NONE {
			Fatalf("regopt not implemented for from3")
		}

		bit = mkvar(f, &p.To)
		if bany(&bit) {
			if p.Info.Flags&RightAddr != 0 {
				setaddrs(bit)
			}
			if p.Info.Flags&RightRead != 0 {
				for z := 0; z < BITS; z++ {
					r.use2.b[z] |= bit.b[z]
				}
			}
			if p.Info.Flags&RightWrite != 0 {
				for z := 0; z < BITS; z++ {
					r.set.b[z] |= bit.b[z]
				}
			}
		}
	}

	for i := 0; i < nvar; i++ {
		v := &vars[i]
		if v.addr != 0 {
			bit := blsh(uint(i))
			for z := 0; z < BITS; z++ {
				addrs.b[z] |= bit.b[z]
			}
		}

		if Debug['R'] != 0 && Debug['v'] != 0 {
			fmt.Printf("bit=%2d addr=%d et=%v w=%-2d s=%v + %d\n", i, v.addr, Econv(v.etype), v.width, v.node, v.offset)
		}
	}

	if Debug['R'] != 0 && Debug['v'] != 0 {
		Dumpit("pass1", firstf, 1)
	}

	// pass 2
	// find looping structure
	flowrpo(g)

	if Debug['R'] != 0 && Debug['v'] != 0 {
		Dumpit("pass2", firstf, 1)
	}

	// pass 2.5
	// iterate propagating fat vardef covering forward
	// r->act records vars with a VARDEF since the last CALL.
	// (r->act will be reused in pass 5 for something else,
	// but we'll be done with it by then.)
	active := 0

	for f := firstf; f != nil; f = f.Link {
		f.Active = 0
		r := f.Data.(*Reg)
		r.act = zbits
	}

	for f := firstf; f != nil; f = f.Link {
		p := f.Prog
		if p.As == obj.AVARDEF && Isfat(((p.To.Node).(*Node)).Type) && ((p.To.Node).(*Node)).Opt() != nil {
			active++
			walkvardef(p.To.Node.(*Node), f, active)
		}
	}

	// pass 3
	// iterate propagating usage
	// 	back until flow graph is complete
	var f1 *Flow
	var i int
	var f *Flow
loop1:
	change = 0

	for f = firstf; f != nil; f = f.Link {
		f.Active = 0
	}
	for f = firstf; f != nil; f = f.Link {
		if f.Prog.As == obj.ARET {
			prop(f, zbits, zbits)
		}
	}

	// pick up unreachable code
loop11:
	i = 0

	for f = firstf; f != nil; f = f1 {
		f1 = f.Link
		if f1 != nil && f1.Active != 0 && f.Active == 0 {
			prop(f, zbits, zbits)
			i = 1
		}
	}

	if i != 0 {
		goto loop11
	}
	if change != 0 {
		goto loop1
	}

	if Debug['R'] != 0 && Debug['v'] != 0 {
		Dumpit("pass3", firstf, 1)
	}

	// pass 4
	// iterate propagating register/variable synchrony
	// 	forward until graph is complete
loop2:
	change = 0

	for f = firstf; f != nil; f = f.Link {
		f.Active = 0
	}
	synch(firstf, zbits)
	if change != 0 {
		goto loop2
	}

	if Debug['R'] != 0 && Debug['v'] != 0 {
		Dumpit("pass4", firstf, 1)
	}

	// pass 4.5
	// move register pseudo-variables into regu.
	mask := uint64((1 << uint(nreg)) - 1)
	for f := firstf; f != nil; f = f.Link {
		r := f.Data.(*Reg)
		r.regu = (r.refbehind.b[0] | r.set.b[0]) & mask
		r.set.b[0] &^= mask
		r.use1.b[0] &^= mask
		r.use2.b[0] &^= mask
		r.refbehind.b[0] &^= mask
		r.refahead.b[0] &^= mask
		r.calbehind.b[0] &^= mask
		r.calahead.b[0] &^= mask
		r.regdiff.b[0] &^= mask
		r.act.b[0] &^= mask
	}

	if Debug['R'] != 0 && Debug['v'] != 0 {
		Dumpit("pass4.5", firstf, 1)
	}

	// pass 5
	// isolate regions
	// calculate costs (paint1)
	var bit Bits
	if f := firstf; f != nil {
		r := f.Data.(*Reg)
		for z := 0; z < BITS; z++ {
			bit.b[z] = (r.refahead.b[z] | r.calahead.b[z]) &^ (externs.b[z] | params.b[z] | addrs.b[z] | consts.b[z])
		}
		if bany(&bit) && !f.Refset {
			// should never happen - all variables are preset
			if Debug['w'] != 0 {
				fmt.Printf("%v: used and not set: %v\n", f.Prog.Line(), &bit)
			}
			f.Refset = true
		}
	}

	for f := firstf; f != nil; f = f.Link {
		(f.Data.(*Reg)).act = zbits
	}
	nregion = 0
	region = region[:0]
	var rgp *Rgn
	for f := firstf; f != nil; f = f.Link {
		r := f.Data.(*Reg)
		for z := 0; z < BITS; z++ {
			bit.b[z] = r.set.b[z] &^ (r.refahead.b[z] | r.calahead.b[z] | addrs.b[z])
		}
		if bany(&bit) && !f.Refset {
			if Debug['w'] != 0 {
				fmt.Printf("%v: set and not used: %v\n", f.Prog.Line(), &bit)
			}
			f.Refset = true
			Thearch.Excise(f)
		}

		for z := 0; z < BITS; z++ {
			bit.b[z] = LOAD(r, z) &^ (r.act.b[z] | addrs.b[z])
		}
		for bany(&bit) {
			i = bnum(&bit)
			change = 0
			paint1(f, i)
			biclr(&bit, uint(i))
			if change <= 0 {
				continue
			}
			if nregion >= MaxRgn {
				nregion++
				continue
			}

			region = append(region, Rgn{
				enter: f,
				cost:  int16(change),
				varno: int16(i),
			})
			nregion++
		}
	}

	if false && Debug['v'] != 0 && strings.Contains(Curfn.Func.Nname.Sym.Name, "Parse") {
		Warn("regions: %d\n", nregion)
	}
	if nregion >= MaxRgn {
		if Debug['v'] != 0 {
			Warn("too many regions: %d\n", nregion)
		}
		nregion = MaxRgn
	}

	sort.Sort(rcmp(region[:nregion]))

	if Debug['R'] != 0 && Debug['v'] != 0 {
		Dumpit("pass5", firstf, 1)
	}

	// pass 6
	// determine used registers (paint2)
	// replace code (paint3)
	if Debug['R'] != 0 && Debug['v'] != 0 {
		fmt.Printf("\nregisterizing\n")
	}
	var usedreg uint64
	var vreg uint64
	for i := 0; i < nregion; i++ {
		rgp = &region[i]
		if Debug['R'] != 0 && Debug['v'] != 0 {
			fmt.Printf("region %d: cost %d varno %d enter %d\n", i, rgp.cost, rgp.varno, rgp.enter.Prog.Pc)
		}
		bit = blsh(uint(rgp.varno))
		usedreg = paint2(rgp.enter, int(rgp.varno), 0)
		vreg = allreg(usedreg, rgp)
		if rgp.regno != 0 {
			if Debug['R'] != 0 && Debug['v'] != 0 {
				v := &vars[rgp.varno]
				fmt.Printf("registerize %v+%d (bit=%2d et=%v) in %v usedreg=%#x vreg=%#x\n", v.node, v.offset, rgp.varno, Econv(v.etype), obj.Rconv(int(rgp.regno)), usedreg, vreg)
			}

			paint3(rgp.enter, int(rgp.varno), vreg, int(rgp.regno))
		}
	}

	// free aux structures. peep allocates new ones.
	for i := 0; i < nvar; i++ {
		vars[i].node.SetOpt(nil)
	}
	Flowend(g)
	firstf = nil

	if Debug['R'] != 0 && Debug['v'] != 0 {
		// Rebuild flow graph, since we inserted instructions
		g := Flowstart(firstp, nil)
		firstf = g.Start
		Dumpit("pass6", firstf, 0)
		Flowend(g)
		firstf = nil
	}

	// pass 7
	// peep-hole on basic block
	if Debug['R'] == 0 || Debug['P'] != 0 {
		Thearch.Peep(firstp)
	}

	// eliminate nops
	for p := firstp; p != nil; p = p.Link {
		for p.Link != nil && p.Link.As == obj.ANOP {
			p.Link = p.Link.Link
		}
		if p.To.Type == obj.TYPE_BRANCH {
			for p.To.Val.(*obj.Prog) != nil && p.To.Val.(*obj.Prog).As == obj.ANOP {
				p.To.Val = p.To.Val.(*obj.Prog).Link
			}
		}
	}

	if Debug['R'] != 0 {
		if Ostats.Ncvtreg != 0 || Ostats.Nspill != 0 || Ostats.Nreload != 0 || Ostats.Ndelmov != 0 || Ostats.Nvar != 0 || Ostats.Naddr != 0 || false {
			fmt.Printf("\nstats\n")
		}

		if Ostats.Ncvtreg != 0 {
			fmt.Printf("\t%4d cvtreg\n", Ostats.Ncvtreg)
		}
		if Ostats.Nspill != 0 {
			fmt.Printf("\t%4d spill\n", Ostats.Nspill)
		}
		if Ostats.Nreload != 0 {
			fmt.Printf("\t%4d reload\n", Ostats.Nreload)
		}
		if Ostats.Ndelmov != 0 {
			fmt.Printf("\t%4d delmov\n", Ostats.Ndelmov)
		}
		if Ostats.Nvar != 0 {
			fmt.Printf("\t%4d var\n", Ostats.Nvar)
		}
		if Ostats.Naddr != 0 {
			fmt.Printf("\t%4d addr\n", Ostats.Naddr)
		}

		Ostats = OptStats{}
	}
}

// bany reports whether any bits in a are set.
func bany(a *Bits) bool {
	for _, x := range &a.b { // & to avoid making a copy of a.b
		if x != 0 {
			return true
		}
	}
	return false
}

// bnum reports the lowest index of a 1 bit in a.
func bnum(a *Bits) int {
	for i, x := range &a.b { // & to avoid making a copy of a.b
		if x != 0 {
			return 64*i + Bitno(x)
		}
	}

	Fatalf("bad in bnum")
	return 0
}

// blsh returns a Bits with 1 at index n, 0 elsewhere (1<<n).
func blsh(n uint) Bits {
	c := zbits
	c.b[n/64] = 1 << (n % 64)
	return c
}

// btest reports whether bit n is 1.
func btest(a *Bits, n uint) bool {
	return a.b[n/64]&(1<<(n%64)) != 0
}

// biset sets bit n to 1.
func biset(a *Bits, n uint) {
	a.b[n/64] |= 1 << (n % 64)
}

// biclr sets bit n to 0.
func biclr(a *Bits, n uint) {
	a.b[n/64] &^= (1 << (n % 64))
}

// Bitno reports the lowest index of a 1 bit in b.
// It calls Fatalf if there is no 1 bit.
func Bitno(b uint64) int {
	if b == 0 {
		Fatalf("bad in bitno")
	}
	n := 0
	if b&(1<<32-1) == 0 {
		n += 32
		b >>= 32
	}
	if b&(1<<16-1) == 0 {
		n += 16
		b >>= 16
	}
	if b&(1<<8-1) == 0 {
		n += 8
		b >>= 8
	}
	if b&(1<<4-1) == 0 {
		n += 4
		b >>= 4
	}
	if b&(1<<2-1) == 0 {
		n += 2
		b >>= 2
	}
	if b&1 == 0 {
		n++
	}
	return n
}

// String returns a space-separated list of the variables represented by bits.
func (bits Bits) String() string {
	// Note: This method takes a value receiver, both for convenience
	// and to make it safe to modify the bits as we process them.
	// Even so, most prints above use &bits, because then the value
	// being stored in the interface{} is a pointer and does not require
	// an allocation and copy to create the interface{}.
	var buf bytes.Buffer
	sep := ""
	for bany(&bits) {
		i := bnum(&bits)
		buf.WriteString(sep)
		sep = " "
		v := &vars[i]
		if v.node == nil || v.node.Sym == nil {
			fmt.Fprintf(&buf, "$%d", i)
		} else {
			fmt.Fprintf(&buf, "%s(%d)", v.node.Sym.Name, i)
			if v.offset != 0 {
				fmt.Fprintf(&buf, "%+d", int64(v.offset))
			}
		}
		biclr(&bits, uint(i))
	}
	return buf.String()
}
