// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

/*
 * portable half of code generator.
 * mainly statements and control flow.
 */
var labellist *Label

var lastlabel *Label

func Sysfunc(name string) *Node {
	n := newname(Pkglookup(name, Runtimepkg))
	n.Class = PFUNC
	return n
}

/*
 * the address of n has been taken and might be used after
 * the current function returns.  mark any local vars
 * as needing to move to the heap.
 */
func addrescapes(n *Node) {
	switch n.Op {
	// probably a type error already.
	// dump("addrescapes", n);
	default:
		break

	case ONAME:
		if n == nodfp {
			break
		}

		// if this is a tmpname (PAUTO), it was tagged by tmpname as not escaping.
		// on PPARAM it means something different.
		if n.Class == PAUTO && n.Esc == EscNever {
			break
		}

		switch n.Class {
		case PPARAMREF:
			addrescapes(n.Defn)

			// if func param, need separate temporary
		// to hold heap pointer.
		// the function type has already been checked
		// (we're in the function body)
		// so the param already has a valid xoffset.

		// expression to refer to stack copy
		case PPARAM, PPARAMOUT:
			n.Stackparam = Nod(OPARAM, n, nil)

			n.Stackparam.Type = n.Type
			n.Stackparam.Addable = true
			if n.Xoffset == BADWIDTH {
				Fatal("addrescapes before param assignment")
			}
			n.Stackparam.Xoffset = n.Xoffset
			fallthrough

			// fallthrough

		case PAUTO:
			n.Class |= PHEAP

			n.Addable = false
			n.Ullman = 2
			n.Xoffset = 0

			// create stack variable to hold pointer to heap
			oldfn := Curfn

			Curfn = n.Curfn
			n.Heapaddr = temp(Ptrto(n.Type))
			buf := fmt.Sprintf("&%v", Sconv(n.Sym, 0))
			n.Heapaddr.Sym = Lookup(buf)
			n.Heapaddr.Orig.Sym = n.Heapaddr.Sym
			n.Esc = EscHeap
			if Debug['m'] != 0 {
				fmt.Printf("%v: moved to heap: %v\n", n.Line(), Nconv(n, 0))
			}
			Curfn = oldfn
		}

	case OIND, ODOTPTR:
		break

		// ODOTPTR has already been introduced,
	// so these are the non-pointer ODOT and OINDEX.
	// In &x[0], if x is a slice, then x does not
	// escape--the pointer inside x does, but that
	// is always a heap pointer anyway.
	case ODOT, OINDEX:
		if !Isslice(n.Left.Type) {
			addrescapes(n.Left)
		}
	}
}

func clearlabels() {
	for l := labellist; l != nil; l = l.Link {
		l.Sym.Label = nil
	}

	labellist = nil
	lastlabel = nil
}

func newlab(n *Node) *Label {
	s := n.Left.Sym
	lab := s.Label
	if lab == nil {
		lab = new(Label)
		if lastlabel == nil {
			labellist = lab
		} else {
			lastlabel.Link = lab
		}
		lastlabel = lab
		lab.Sym = s
		s.Label = lab
	}

	if n.Op == OLABEL {
		if lab.Def != nil {
			Yyerror("label %v already defined at %v", Sconv(s, 0), lab.Def.Line())
		} else {
			lab.Def = n
		}
	} else {
		lab.Use = list(lab.Use, n)
	}

	return lab
}

func checkgoto(from *Node, to *Node) {
	if from.Sym == to.Sym {
		return
	}

	nf := 0
	for fs := from.Sym; fs != nil; fs = fs.Link {
		nf++
	}
	nt := 0
	for fs := to.Sym; fs != nil; fs = fs.Link {
		nt++
	}
	fs := from.Sym
	for ; nf > nt; nf-- {
		fs = fs.Link
	}
	if fs != to.Sym {
		lno := int(lineno)
		setlineno(from)

		// decide what to complain about.
		// prefer to complain about 'into block' over declarations,
		// so scan backward to find most recent block or else dcl.
		var block *Sym

		var dcl *Sym
		ts := to.Sym
		for ; nt > nf; nt-- {
			if ts.Pkg == nil {
				block = ts
			} else {
				dcl = ts
			}
			ts = ts.Link
		}

		for ts != fs {
			if ts.Pkg == nil {
				block = ts
			} else {
				dcl = ts
			}
			ts = ts.Link
			fs = fs.Link
		}

		if block != nil {
			Yyerror("goto %v jumps into block starting at %v", Sconv(from.Left.Sym, 0), Ctxt.Line(int(block.Lastlineno)))
		} else {
			Yyerror("goto %v jumps over declaration of %v at %v", Sconv(from.Left.Sym, 0), Sconv(dcl, 0), Ctxt.Line(int(dcl.Lastlineno)))
		}
		lineno = int32(lno)
	}
}

func stmtlabel(n *Node) *Label {
	if n.Sym != nil {
		lab := n.Sym.Label
		if lab != nil {
			if lab.Def != nil {
				if lab.Def.Defn == n {
					return lab
				}
			}
		}
	}
	return nil
}

/*
 * compile statements
 */
func Genlist(l *NodeList) {
	for ; l != nil; l = l.Next {
		gen(l.N)
	}
}

/*
 * generate code to start new proc running call n.
 */
func cgen_proc(n *Node, proc int) {
	switch n.Left.Op {
	default:
		Fatal("cgen_proc: unknown call %v", Oconv(int(n.Left.Op), 0))

	case OCALLMETH:
		cgen_callmeth(n.Left, proc)

	case OCALLINTER:
		cgen_callinter(n.Left, nil, proc)

	case OCALLFUNC:
		cgen_call(n.Left, proc)
	}
}

/*
 * generate declaration.
 * have to allocate heap copy
 * for escaped variables.
 */
func cgen_dcl(n *Node) {
	if Debug['g'] != 0 {
		Dump("\ncgen-dcl", n)
	}
	if n.Op != ONAME {
		Dump("cgen_dcl", n)
		Fatal("cgen_dcl")
	}

	if n.Class&PHEAP == 0 {
		return
	}
	if compiling_runtime != 0 {
		Yyerror("%v escapes to heap, not allowed in runtime.", Nconv(n, 0))
	}
	if n.Alloc == nil {
		n.Alloc = callnew(n.Type)
	}
	Cgen_as(n.Heapaddr, n.Alloc)
}

/*
 * generate discard of value
 */
func cgen_discard(nr *Node) {
	if nr == nil {
		return
	}

	switch nr.Op {
	case ONAME:
		if nr.Class&PHEAP == 0 && nr.Class != PEXTERN && nr.Class != PFUNC && nr.Class != PPARAMREF {
			gused(nr)
		}

		// unary
	case OADD,
		OAND,
		ODIV,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLSH,
		OLT,
		OMOD,
		OMUL,
		ONE,
		OOR,
		ORSH,
		OSUB,
		OXOR:
		cgen_discard(nr.Left)

		cgen_discard(nr.Right)

		// binary
	case OCAP,
		OCOM,
		OLEN,
		OMINUS,
		ONOT,
		OPLUS:
		cgen_discard(nr.Left)

	case OIND:
		Cgen_checknil(nr.Left)

		// special enough to just evaluate
	default:
		var tmp Node
		Tempname(&tmp, nr.Type)

		Cgen_as(&tmp, nr)
		gused(&tmp)
	}
}

/*
 * clearslim generates code to zero a slim node.
 */
func Clearslim(n *Node) {
	var z Node
	z.Op = OLITERAL
	z.Type = n.Type
	z.Addable = true

	switch Simtype[n.Type.Etype] {
	case TCOMPLEX64, TCOMPLEX128:
		z.Val.U.Cval = new(Mpcplx)
		Mpmovecflt(&z.Val.U.Cval.Real, 0.0)
		Mpmovecflt(&z.Val.U.Cval.Imag, 0.0)

	case TFLOAT32, TFLOAT64:
		var zero Mpflt
		Mpmovecflt(&zero, 0.0)
		z.Val.Ctype = CTFLT
		z.Val.U.Fval = &zero

	case TPTR32, TPTR64, TCHAN, TMAP:
		z.Val.Ctype = CTNIL

	case TBOOL:
		z.Val.Ctype = CTBOOL

	case TINT8,
		TINT16,
		TINT32,
		TINT64,
		TUINT8,
		TUINT16,
		TUINT32,
		TUINT64:
		z.Val.Ctype = CTINT
		z.Val.U.Xval = new(Mpint)
		Mpmovecfix(z.Val.U.Xval, 0)

	default:
		Fatal("clearslim called on type %v", Tconv(n.Type, 0))
	}

	ullmancalc(&z)
	Cgen(&z, n)
}

/*
 * generate:
 *	res = iface{typ, data}
 * n->left is typ
 * n->right is data
 */
func Cgen_eface(n *Node, res *Node) {
	/*
	 * the right node of an eface may contain function calls that uses res as an argument,
	 * so it's important that it is done first
	 */

	tmp := temp(Types[Tptr])
	Cgen(n.Right, tmp)

	Gvardef(res)

	dst := *res
	dst.Type = Types[Tptr]
	dst.Xoffset += int64(Widthptr)
	Cgen(tmp, &dst)

	dst.Xoffset -= int64(Widthptr)
	Cgen(n.Left, &dst)
}

/*
 * generate one of:
 *	res, resok = x.(T)
 *	res = x.(T) (when resok == nil)
 * n.Left is x
 * n.Type is T
 */
func cgen_dottype(n *Node, res, resok *Node) {
	if Debug_typeassert > 0 {
		Warn("type assertion inlined")
	}
	//	iface := n.Left
	//	r1 := iword(iface)
	//	if n.Left is non-empty interface {
	//		r1 = *r1
	//	}
	//	if r1 == T {
	//		res = idata(iface)
	//		resok = true
	//	} else {
	//		assert[EI]2T(x, T, nil) // (when resok == nil; does not return)
	//		resok = false // (when resok != nil)
	//	}
	//
	var iface Node
	Igen(n.Left, &iface, res)
	var r1, r2 Node
	byteptr := Ptrto(Types[TUINT8]) // type used in runtime prototypes for runtime type (*byte)
	Regalloc(&r1, byteptr, nil)
	iface.Type = byteptr
	Cgen(&iface, &r1)
	if !isnilinter(n.Left.Type) {
		// Holding itab, want concrete type in second word.
		Thearch.Gins(Thearch.Optoas(OCMP, byteptr), &r1, Nodintconst(0))
		p := Gbranch(Thearch.Optoas(OEQ, byteptr), nil, -1)
		r2 = r1
		r2.Op = OINDREG
		r2.Xoffset = int64(Widthptr)
		Cgen(&r2, &r1)
		Patch(p, Pc)
	}
	Regalloc(&r2, byteptr, nil)
	Cgen(typename(n.Type), &r2)
	Thearch.Gins(Thearch.Optoas(OCMP, byteptr), &r1, &r2)
	p := Gbranch(Thearch.Optoas(ONE, byteptr), nil, -1)
	iface.Xoffset += int64(Widthptr)
	Cgen(&iface, &r1)
	Regfree(&iface)

	if resok == nil {
		r1.Type = res.Type
		Cgen(&r1, res)
		q := Gbranch(obj.AJMP, nil, 0)
		Patch(p, Pc)

		fn := syslook("panicdottype", 0)
		dowidth(fn.Type)
		call := Nod(OCALLFUNC, fn, nil)
		r1.Type = byteptr
		r2.Type = byteptr
		call.List = list(list(list1(&r1), &r2), typename(n.Left.Type))
		call.List = ascompatte(OCALLFUNC, call, false, getinarg(fn.Type), call.List, 0, nil)
		gen(call)
		Regfree(&r1)
		Regfree(&r2)
		Thearch.Gins(obj.AUNDEF, nil, nil)
		Patch(q, Pc)
	} else {
		// This half is handling the res, resok = x.(T) case,
		// which is called from gen, not cgen, and is consequently fussier
		// about blank assignments. We have to avoid calling cgen for those.
		Regfree(&r2)
		r1.Type = res.Type
		if !isblank(res) {
			Cgen(&r1, res)
		}
		Regfree(&r1)
		if !isblank(resok) {
			Cgen(Nodbool(true), resok)
		}
		q := Gbranch(obj.AJMP, nil, 0)
		Patch(p, Pc)
		if !isblank(res) {
			n := nodnil()
			n.Type = res.Type
			Cgen(n, res)
		}
		if !isblank(resok) {
			Cgen(Nodbool(false), resok)
		}
		Patch(q, Pc)
	}
}

/*
 * generate:
 *	res, resok = x.(T)
 * n.Left is x
 * n.Type is T
 */
func Cgen_As2dottype(n, res, resok *Node) {
	if Debug_typeassert > 0 {
		Warn("type assertion inlined")
	}
	//	iface := n.Left
	//	r1 := iword(iface)
	//	if n.Left is non-empty interface {
	//		r1 = *r1
	//	}
	//	if r1 == T {
	//		res = idata(iface)
	//		resok = true
	//	} else {
	//		res = nil
	//		resok = false
	//	}
	//
	var iface Node
	Igen(n.Left, &iface, nil)
	var r1, r2 Node
	byteptr := Ptrto(Types[TUINT8]) // type used in runtime prototypes for runtime type (*byte)
	Regalloc(&r1, byteptr, res)
	iface.Type = byteptr
	Cgen(&iface, &r1)
	if !isnilinter(n.Left.Type) {
		// Holding itab, want concrete type in second word.
		Thearch.Gins(Thearch.Optoas(OCMP, byteptr), &r1, Nodintconst(0))
		p := Gbranch(Thearch.Optoas(OEQ, byteptr), nil, -1)
		r2 = r1
		r2.Op = OINDREG
		r2.Xoffset = int64(Widthptr)
		Cgen(&r2, &r1)
		Patch(p, Pc)
	}
	Regalloc(&r2, byteptr, nil)
	Cgen(typename(n.Type), &r2)
	Thearch.Gins(Thearch.Optoas(OCMP, byteptr), &r1, &r2)
	p := Gbranch(Thearch.Optoas(ONE, byteptr), nil, -1)
	iface.Type = n.Type
	iface.Xoffset += int64(Widthptr)
	Cgen(&iface, &r1)
	if iface.Op != 0 {
		Regfree(&iface)
	}
	Cgen(&r1, res)
	q := Gbranch(obj.AJMP, nil, 0)
	Patch(p, Pc)

	fn := syslook("panicdottype", 0)
	dowidth(fn.Type)
	call := Nod(OCALLFUNC, fn, nil)
	call.List = list(list(list1(&r1), &r2), typename(n.Left.Type))
	call.List = ascompatte(OCALLFUNC, call, false, getinarg(fn.Type), call.List, 0, nil)
	gen(call)
	Regfree(&r1)
	Regfree(&r2)
	Thearch.Gins(obj.AUNDEF, nil, nil)
	Patch(q, Pc)
}

/*
 * generate:
 *	res = s[lo, hi];
 * n->left is s
 * n->list is (cap(s)-lo(TUINT), hi-lo(TUINT)[, lo*width(TUINTPTR)])
 * caller (cgen) guarantees res is an addable ONAME.
 *
 * called for OSLICE, OSLICE3, OSLICEARR, OSLICE3ARR, OSLICESTR.
 */
func Cgen_slice(n *Node, res *Node) {
	cap := n.List.N
	len := n.List.Next.N
	var offs *Node
	if n.List.Next.Next != nil {
		offs = n.List.Next.Next.N
	}

	// evaluate base pointer first, because it is the only
	// possibly complex expression. once that is evaluated
	// and stored, updating the len and cap can be done
	// without making any calls, so without doing anything that
	// might cause preemption or garbage collection.
	// this makes the whole slice update atomic as far as the
	// garbage collector can see.
	base := temp(Types[TUINTPTR])

	tmplen := temp(Types[TINT])
	var tmpcap *Node
	if n.Op != OSLICESTR {
		tmpcap = temp(Types[TINT])
	} else {
		tmpcap = tmplen
	}

	var src Node
	if isnil(n.Left) {
		Tempname(&src, n.Left.Type)
		Cgen(n.Left, &src)
	} else {
		src = *n.Left
	}
	if n.Op == OSLICE || n.Op == OSLICE3 || n.Op == OSLICESTR {
		src.Xoffset += int64(Array_array)
	}

	if n.Op == OSLICEARR || n.Op == OSLICE3ARR {
		if !Isptr[n.Left.Type.Etype] {
			Fatal("slicearr is supposed to work on pointer: %v\n", Nconv(n, obj.FmtSign))
		}
		Cgen(&src, base)
		Cgen_checknil(base)
	} else {
		src.Type = Types[Tptr]
		Cgen(&src, base)
	}

	// committed to the update
	Gvardef(res)

	// compute len and cap.
	// len = n-i, cap = m-i, and offs = i*width.
	// computing offs last lets the multiply overwrite i.
	Cgen((*Node)(len), tmplen)

	if n.Op != OSLICESTR {
		Cgen(cap, tmpcap)
	}

	// if new cap != 0 { base += add }
	// This avoids advancing base past the end of the underlying array/string,
	// so that it cannot point at the next object in memory.
	// If cap == 0, the base doesn't matter except insofar as it is 0 or non-zero.
	// In essence we are replacing x[i:j:k] where i == j == k
	// or x[i:j] where i == j == cap(x) with x[0:0:0].
	if offs != nil {
		p1 := gjmp(nil)
		p2 := gjmp(nil)
		Patch(p1, Pc)

		var con Node
		Nodconst(&con, tmpcap.Type, 0)
		cmp := Nod(OEQ, tmpcap, &con)
		typecheck(&cmp, Erv)
		Bgen(cmp, true, -1, p2)

		add := Nod(OADD, base, offs)
		typecheck(&add, Erv)
		Cgen(add, base)

		Patch(p2, Pc)
	}

	// dst.array = src.array  [ + lo *width ]
	dst := *res

	dst.Xoffset += int64(Array_array)
	dst.Type = Types[Tptr]
	Cgen(base, &dst)

	// dst.len = hi [ - lo ]
	dst = *res

	dst.Xoffset += int64(Array_nel)
	dst.Type = Types[Simtype[TUINT]]
	Cgen(tmplen, &dst)

	if n.Op != OSLICESTR {
		// dst.cap = cap [ - lo ]
		dst = *res

		dst.Xoffset += int64(Array_cap)
		dst.Type = Types[Simtype[TUINT]]
		Cgen(tmpcap, &dst)
	}
}

/*
 * gather series of offsets
 * >=0 is direct addressed field
 * <0 is pointer to next field (+1)
 */
func Dotoffset(n *Node, oary []int64, nn **Node) int {
	var i int

	switch n.Op {
	case ODOT:
		if n.Xoffset == BADWIDTH {
			Dump("bad width in dotoffset", n)
			Fatal("bad width in dotoffset")
		}

		i = Dotoffset(n.Left, oary, nn)
		if i > 0 {
			if oary[i-1] >= 0 {
				oary[i-1] += n.Xoffset
			} else {
				oary[i-1] -= n.Xoffset
			}
			break
		}

		if i < 10 {
			oary[i] = n.Xoffset
			i++
		}

	case ODOTPTR:
		if n.Xoffset == BADWIDTH {
			Dump("bad width in dotoffset", n)
			Fatal("bad width in dotoffset")
		}

		i = Dotoffset(n.Left, oary, nn)
		if i < 10 {
			oary[i] = -(n.Xoffset + 1)
			i++
		}

	default:
		*nn = n
		return 0
	}

	if i >= 10 {
		*nn = nil
	}
	return i
}

/*
 * make a new off the books
 */
func Tempname(nn *Node, t *Type) {
	if Curfn == nil {
		Fatal("no curfn for tempname")
	}

	if t == nil {
		Yyerror("tempname called with nil type")
		t = Types[TINT32]
	}

	// give each tmp a different name so that there
	// a chance to registerizer them
	s := Lookupf("autotmp_%.4d", statuniqgen)
	statuniqgen++
	n := Nod(ONAME, nil, nil)
	n.Sym = s
	s.Def = n
	n.Type = t
	n.Class = PAUTO
	n.Addable = true
	n.Ullman = 1
	n.Esc = EscNever
	n.Curfn = Curfn
	Curfn.Func.Dcl = list(Curfn.Func.Dcl, n)

	dowidth(t)
	n.Xoffset = 0
	*nn = *n
}

func temp(t *Type) *Node {
	n := Nod(OXXX, nil, nil)
	Tempname(n, t)
	n.Sym.Def.Used = true
	return n.Orig
}

func gen(n *Node) {
	//dump("gen", n);

	lno := setlineno(n)

	wasregalloc := Anyregalloc()

	if n == nil {
		goto ret
	}

	if n.Ninit != nil {
		Genlist(n.Ninit)
	}

	setlineno(n)

	switch n.Op {
	default:
		Fatal("gen: unknown op %v", Nconv(n, obj.FmtShort|obj.FmtSign))

	case OCASE,
		OFALL,
		OXCASE,
		OXFALL,
		ODCLCONST,
		ODCLFUNC,
		ODCLTYPE:
		break

	case OEMPTY:
		break

	case OBLOCK:
		Genlist(n.List)

	case OLABEL:
		if isblanksym(n.Left.Sym) {
			break
		}

		lab := newlab(n)

		// if there are pending gotos, resolve them all to the current pc.
		var p2 *obj.Prog
		for p1 := lab.Gotopc; p1 != nil; p1 = p2 {
			p2 = unpatch(p1)
			Patch(p1, Pc)
		}

		lab.Gotopc = nil
		if lab.Labelpc == nil {
			lab.Labelpc = Pc
		}

		if n.Defn != nil {
			switch n.Defn.Op {
			// so stmtlabel can find the label
			case OFOR, OSWITCH, OSELECT:
				n.Defn.Sym = lab.Sym
			}
		}

		// if label is defined, emit jump to it.
	// otherwise save list of pending gotos in lab->gotopc.
	// the list is linked through the normal jump target field
	// to avoid a second list.  (the jumps are actually still
	// valid code, since they're just going to another goto
	// to the same label.  we'll unwind it when we learn the pc
	// of the label in the OLABEL case above.)
	case OGOTO:
		lab := newlab(n)

		if lab.Labelpc != nil {
			gjmp(lab.Labelpc)
		} else {
			lab.Gotopc = gjmp(lab.Gotopc)
		}

	case OBREAK:
		if n.Left != nil {
			lab := n.Left.Sym.Label
			if lab == nil {
				Yyerror("break label not defined: %v", Sconv(n.Left.Sym, 0))
				break
			}

			lab.Used = 1
			if lab.Breakpc == nil {
				Yyerror("invalid break label %v", Sconv(n.Left.Sym, 0))
				break
			}

			gjmp(lab.Breakpc)
			break
		}

		if breakpc == nil {
			Yyerror("break is not in a loop")
			break
		}

		gjmp(breakpc)

	case OCONTINUE:
		if n.Left != nil {
			lab := n.Left.Sym.Label
			if lab == nil {
				Yyerror("continue label not defined: %v", Sconv(n.Left.Sym, 0))
				break
			}

			lab.Used = 1
			if lab.Continpc == nil {
				Yyerror("invalid continue label %v", Sconv(n.Left.Sym, 0))
				break
			}

			gjmp(lab.Continpc)
			break
		}

		if continpc == nil {
			Yyerror("continue is not in a loop")
			break
		}

		gjmp(continpc)

	case OFOR:
		sbreak := breakpc
		p1 := gjmp(nil)     //		goto test
		breakpc = gjmp(nil) // break:	goto done
		scontin := continpc
		continpc = Pc

		// define break and continue labels
		lab := stmtlabel(n)
		if lab != nil {
			lab.Breakpc = breakpc
			lab.Continpc = continpc
		}

		gen(n.Nincr)                      // contin:	incr
		Patch(p1, Pc)                     // test:
		Bgen(n.Ntest, false, -1, breakpc) //		if(!test) goto break
		Genlist(n.Nbody)                  //		body
		gjmp(continpc)
		Patch(breakpc, Pc) // done:
		continpc = scontin
		breakpc = sbreak
		if lab != nil {
			lab.Breakpc = nil
			lab.Continpc = nil
		}

	case OIF:
		p1 := gjmp(nil)                          //		goto test
		p2 := gjmp(nil)                          // p2:		goto else
		Patch(p1, Pc)                            // test:
		Bgen(n.Ntest, false, int(-n.Likely), p2) //		if(!test) goto p2
		Genlist(n.Nbody)                         //		then
		p3 := gjmp(nil)                          //		goto done
		Patch(p2, Pc)                            // else:
		Genlist(n.Nelse)                         //		else
		Patch(p3, Pc)                            // done:

	case OSWITCH:
		sbreak := breakpc
		p1 := gjmp(nil)     //		goto test
		breakpc = gjmp(nil) // break:	goto done

		// define break label
		lab := stmtlabel(n)
		if lab != nil {
			lab.Breakpc = breakpc
		}

		Patch(p1, Pc)      // test:
		Genlist(n.Nbody)   //		switch(test) body
		Patch(breakpc, Pc) // done:
		breakpc = sbreak
		if lab != nil {
			lab.Breakpc = nil
		}

	case OSELECT:
		sbreak := breakpc
		p1 := gjmp(nil)     //		goto test
		breakpc = gjmp(nil) // break:	goto done

		// define break label
		lab := stmtlabel(n)
		if lab != nil {
			lab.Breakpc = breakpc
		}

		Patch(p1, Pc)      // test:
		Genlist(n.Nbody)   //		select() body
		Patch(breakpc, Pc) // done:
		breakpc = sbreak
		if lab != nil {
			lab.Breakpc = nil
		}

	case ODCL:
		cgen_dcl(n.Left)

	case OAS:
		if gen_as_init(n) {
			break
		}
		Cgen_as(n.Left, n.Right)

	case OAS2DOTTYPE:
		cgen_dottype(n.Rlist.N, n.List.N, n.List.Next.N)

	case OCALLMETH:
		cgen_callmeth(n, 0)

	case OCALLINTER:
		cgen_callinter(n, nil, 0)

	case OCALLFUNC:
		cgen_call(n, 0)

	case OPROC:
		cgen_proc(n, 1)

	case ODEFER:
		cgen_proc(n, 2)

	case ORETURN, ORETJMP:
		cgen_ret(n)

	// Function calls turned into compiler intrinsics.
	// At top level, can just ignore the call and make sure to preserve side effects in the argument, if any.
	case OGETG:
		// nothing
	case OSQRT:
		cgen_discard(n.Left)

	case OCHECKNIL:
		Cgen_checknil(n.Left)

	case OVARKILL:
		gvarkill(n.Left)
	}

ret:
	if Anyregalloc() != wasregalloc {
		Dump("node", n)
		Fatal("registers left allocated")
	}

	lineno = lno
}

func Cgen_as(nl *Node, nr *Node) {
	if Debug['g'] != 0 {
		Dump("cgen_as", nl)
		Dump("cgen_as = ", nr)
	}

	for nr != nil && nr.Op == OCONVNOP {
		nr = nr.Left
	}

	if nl == nil || isblank(nl) {
		cgen_discard(nr)
		return
	}

	if nr == nil || iszero(nr) {
		// heaps should already be clear
		if nr == nil && (nl.Class&PHEAP != 0) {
			return
		}

		tl := nl.Type
		if tl == nil {
			return
		}
		if Isfat(tl) {
			if nl.Op == ONAME {
				Gvardef(nl)
			}
			Thearch.Clearfat(nl)
			return
		}

		Clearslim(nl)
		return
	}

	tl := nl.Type
	if tl == nil {
		return
	}

	Cgen(nr, nl)
}

func cgen_callmeth(n *Node, proc int) {
	// generate a rewrite in n2 for the method call
	// (p.f)(...) goes to (f)(p,...)

	l := n.Left

	if l.Op != ODOTMETH {
		Fatal("cgen_callmeth: not dotmethod: %v")
	}

	n2 := *n
	n2.Op = OCALLFUNC
	n2.Left = l.Right
	n2.Left.Type = l.Type

	if n2.Left.Op == ONAME {
		n2.Left.Class = PFUNC
	}
	cgen_call(&n2, proc)
}

func checklabels() {
	var l *NodeList

	for lab := labellist; lab != nil; lab = lab.Link {
		if lab.Def == nil {
			for l = lab.Use; l != nil; l = l.Next {
				yyerrorl(int(l.N.Lineno), "label %v not defined", Sconv(lab.Sym, 0))
			}
			continue
		}

		if lab.Use == nil && lab.Used == 0 {
			yyerrorl(int(lab.Def.Lineno), "label %v defined and not used", Sconv(lab.Sym, 0))
			continue
		}

		if lab.Gotopc != nil {
			Fatal("label %v never resolved", Sconv(lab.Sym, 0))
		}
		for l = lab.Use; l != nil; l = l.Next {
			checkgoto(l.N, lab.Def)
		}
	}
}

// Componentgen copies a composite value by moving its individual components.
// Slices, strings and interfaces are supported. Small structs or arrays with
// elements of basic type are also supported.
// nr is nil when assigning a zero value.
func Componentgen(nr *Node, nl *Node) bool {
	var nodl, nodr Node

	switch nl.Type.Etype {
	default:
		return false

	case TARRAY:
		t := nl.Type

		// Slices are ok.
		if Isslice(t) {
			break
		}

		// Small arrays are ok.
		if t.Bound > 0 && t.Bound <= 3 && !Isfat(t.Type) {
			break
		}

		return false

	case TSTRUCT:
		// Small structs with non-fat types are ok.
		// Zero-sized structs are treated separately elsewhere.
		fldcount := int64(0)

		for t := nl.Type.Type; t != nil; t = t.Down {
			if Isfat(t.Type) && !Isslice(t) {
				return false
			}
			if t.Etype != TFIELD {
				Fatal("componentgen: not a TFIELD: %v", Tconv(t, obj.FmtLong))
			}
			fldcount++
		}

		if fldcount == 0 || fldcount > 4 {
			return false
		}

	case TSTRING, TINTER:
		break
	}

	isConstString := Isconst(nr, CTSTR)
	nodl = *nl
	if !cadable(nl) {
		if nr != nil && !cadable(nr) && !isConstString {
			return false
		}
		Igen(nl, &nodl, nil)
		defer Regfree(&nodl)
	}

	if nr != nil {
		nodr = *nr
		if !cadable(nr) && !isConstString {
			Igen(nr, &nodr, nil)
			defer Regfree(&nodr)
		}
	} else {
		// When zeroing, prepare a register containing zero.
		if Thearch.REGZERO != 0 {
			// cpu has a dedicated zero register
			Nodreg(&nodr, Types[TUINT], Thearch.REGZERO)
		} else {
			// no dedicated zero register
			var tmp Node
			Nodconst(&tmp, nl.Type, 0)

			Regalloc(&nodr, Types[TUINT], nil)
			Thearch.Gmove(&tmp, &nodr)
			defer Regfree(&nodr)
		}
	}

	// nl and nr are 'cadable' which basically means they are names (variables) now.
	// If they are the same variable, don't generate any code, because the
	// VARDEF we generate will mark the old value as dead incorrectly.
	// (And also the assignments are useless.)
	if nr != nil && nl.Op == ONAME && nr.Op == ONAME && nl == nr {
		return true
	}

	switch nl.Type.Etype {
	default:
		return false

	case TARRAY:
		// componentgen for arrays.
		if nl.Op == ONAME {
			Gvardef(nl)
		}
		t := nl.Type
		if !Isslice(t) {
			nodl.Type = t.Type
			nodr.Type = nodl.Type
			for fldcount := int64(0); fldcount < t.Bound; fldcount++ {
				if nr == nil {
					Clearslim(&nodl)
				} else {
					Thearch.Gmove(&nodr, &nodl)
				}
				nodl.Xoffset += t.Type.Width
				nodr.Xoffset += t.Type.Width
			}
			return true
		}

		// componentgen for slices.
		nodl.Xoffset += int64(Array_array)

		nodl.Type = Ptrto(nl.Type.Type)

		if nr != nil {
			nodr.Xoffset += int64(Array_array)
			nodr.Type = nodl.Type
		}

		Thearch.Gmove(&nodr, &nodl)

		nodl.Xoffset += int64(Array_nel) - int64(Array_array)
		nodl.Type = Types[Simtype[TUINT]]

		if nr != nil {
			nodr.Xoffset += int64(Array_nel) - int64(Array_array)
			nodr.Type = nodl.Type
		}

		Thearch.Gmove(&nodr, &nodl)

		nodl.Xoffset += int64(Array_cap) - int64(Array_nel)
		nodl.Type = Types[Simtype[TUINT]]

		if nr != nil {
			nodr.Xoffset += int64(Array_cap) - int64(Array_nel)
			nodr.Type = nodl.Type
		}

		Thearch.Gmove(&nodr, &nodl)
		return true

	case TSTRING:
		if nl.Op == ONAME {
			Gvardef(nl)
		}
		nodl.Xoffset += int64(Array_array)
		nodl.Type = Ptrto(Types[TUINT8])

		if isConstString {
			Regalloc(&nodr, Types[Tptr], nil)
			p := Thearch.Gins(Thearch.Optoas(OAS, Types[Tptr]), nil, &nodr)
			Datastring(nr.Val.U.Sval, &p.From)
			p.From.Type = obj.TYPE_ADDR
			Regfree(&nodr)
		} else if nr != nil {
			nodr.Xoffset += int64(Array_array)
			nodr.Type = nodl.Type
		}

		Thearch.Gmove(&nodr, &nodl)

		nodl.Xoffset += int64(Array_nel) - int64(Array_array)
		nodl.Type = Types[Simtype[TUINT]]

		if isConstString {
			Nodconst(&nodr, nodl.Type, int64(len(nr.Val.U.Sval)))
		} else if nr != nil {
			nodr.Xoffset += int64(Array_nel) - int64(Array_array)
			nodr.Type = nodl.Type
		}

		Thearch.Gmove(&nodr, &nodl)
		return true

	case TINTER:
		if nl.Op == ONAME {
			Gvardef(nl)
		}
		nodl.Xoffset += int64(Array_array)
		nodl.Type = Ptrto(Types[TUINT8])

		if nr != nil {
			nodr.Xoffset += int64(Array_array)
			nodr.Type = nodl.Type
		}

		Thearch.Gmove(&nodr, &nodl)

		nodl.Xoffset += int64(Array_nel) - int64(Array_array)
		nodl.Type = Ptrto(Types[TUINT8])

		if nr != nil {
			nodr.Xoffset += int64(Array_nel) - int64(Array_array)
			nodr.Type = nodl.Type
		}

		Thearch.Gmove(&nodr, &nodl)
		return true

	case TSTRUCT:
		if nl.Op == ONAME {
			Gvardef(nl)
		}
		loffset := nodl.Xoffset
		roffset := nodr.Xoffset

		// funarg structs may not begin at offset zero.
		if nl.Type.Etype == TSTRUCT && nl.Type.Funarg != 0 && nl.Type.Type != nil {
			loffset -= nl.Type.Type.Width
		}
		if nr != nil && nr.Type.Etype == TSTRUCT && nr.Type.Funarg != 0 && nr.Type.Type != nil {
			roffset -= nr.Type.Type.Width
		}

		for t := nl.Type.Type; t != nil; t = t.Down {
			nodl.Xoffset = loffset + t.Width
			nodl.Type = t.Type

			if nr == nil {
				Clearslim(&nodl)
			} else {
				nodr.Xoffset = roffset + t.Width
				nodr.Type = nodl.Type
				Thearch.Gmove(&nodr, &nodl)
			}
		}
		return true
	}
}

func cadable(n *Node) bool {
	if !n.Addable {
		// dont know how it happens,
		// but it does
		return false
	}

	switch n.Op {
	case ONAME:
		return true
	}

	return false
}
