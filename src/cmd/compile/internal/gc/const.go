// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/big"
	"cmd/internal/obj"
	"strings"
)

// IntLiteral returns the Node's literal value as an interger.
func (n *Node) IntLiteral() (x int64, ok bool) {
	switch {
	case n == nil:
		return
	case Isconst(n, CTINT):
		return n.Int(), true
	case Isconst(n, CTBOOL):
		return int64(obj.Bool2int(n.Bool())), true
	}
	return
}

// Int returns n as an int.
// n must be an integer constant.
func (n *Node) Int() int64 {
	if !Isconst(n, CTINT) {
		Fatalf("Int(%v)", n)
	}
	return Mpgetfix(n.Val().U.(*Mpint))
}

// SetInt sets n's value to i.
// n must be an integer constant.
func (n *Node) SetInt(i int64) {
	if !Isconst(n, CTINT) {
		Fatalf("SetInt(%v)", n)
	}
	Mpmovecfix(n.Val().U.(*Mpint), i)
}

// SetBigInt sets n's value to x.
// n must be an integer constant.
func (n *Node) SetBigInt(x *big.Int) {
	if !Isconst(n, CTINT) {
		Fatalf("SetBigInt(%v)", n)
	}
	n.Val().U.(*Mpint).Val.Set(x)
}

// Bool returns n as an bool.
// n must be an boolean constant.
func (n *Node) Bool() bool {
	if !Isconst(n, CTBOOL) {
		Fatalf("Int(%v)", n)
	}
	return n.Val().U.(bool)
}

// truncate float literal fv to 32-bit or 64-bit precision
// according to type; return truncated value.
func truncfltlit(oldv *Mpflt, t *Type) *Mpflt {
	if t == nil {
		return oldv
	}

	var v Val
	v.U = oldv
	overflow(v, t)

	fv := newMpflt()
	mpmovefltflt(fv, oldv)

	// convert large precision literal floating
	// into limited precision (float64 or float32)
	switch t.Etype {
	case TFLOAT64:
		d := mpgetflt(fv)
		Mpmovecflt(fv, d)

	case TFLOAT32:
		d := mpgetflt32(fv)
		Mpmovecflt(fv, d)
	}

	return fv
}

// convert n, if literal, to type t.
// implicit conversion.
func Convlit(np **Node, t *Type) {
	convlit1(np, t, false)
}

// convert n, if literal, to type t.
// return a new node if necessary
//(if n is a named constant, can't edit n->type directly).
func convlit1(np **Node, t *Type, explicit bool) {
	n := *np
	if n == nil || t == nil || n.Type == nil || isideal(t) || n.Type == t {
		return
	}
	if !explicit && !isideal(n.Type) {
		return
	}

	if n.Op == OLITERAL {
		nn := Nod(OXXX, nil, nil)
		*nn = *n
		n = nn
		*np = n
	}

	switch n.Op {
	default:
		if n.Type == idealbool {
			if t.Etype == TBOOL {
				n.Type = t
			} else {
				n.Type = Types[TBOOL]
			}
		}

		if n.Type.Etype == TIDEAL {
			Convlit(&n.Left, t)
			Convlit(&n.Right, t)
			n.Type = t
		}

		return

		// target is invalid type for a constant?  leave alone.
	case OLITERAL:
		if !okforconst[t.Etype] && n.Type.Etype != TNIL {
			defaultlit(&n, nil)
			*np = n
			return
		}

	case OLSH, ORSH:
		convlit1(&n.Left, t, explicit && isideal(n.Left.Type))
		t = n.Left.Type
		if t != nil && t.Etype == TIDEAL && n.Val().Ctype() != CTINT {
			n.SetVal(toint(n.Val()))
		}
		if t != nil && !Isint[t.Etype] {
			Yyerror("invalid operation: %v (shift of type %v)", n, t)
			t = nil
		}

		n.Type = t
		return

	case OCOMPLEX:
		if n.Type.Etype == TIDEAL {
			switch t.Etype {
			// If trying to convert to non-complex type,
			// leave as complex128 and let typechecker complain.
			default:
				t = Types[TCOMPLEX128]
				fallthrough

				//fallthrough
			case TCOMPLEX128:
				n.Type = t

				Convlit(&n.Left, Types[TFLOAT64])
				Convlit(&n.Right, Types[TFLOAT64])

			case TCOMPLEX64:
				n.Type = t
				Convlit(&n.Left, Types[TFLOAT32])
				Convlit(&n.Right, Types[TFLOAT32])
			}
		}

		return
	}

	// avoided repeated calculations, errors
	if Eqtype(n.Type, t) {
		return
	}

	ct := consttype(n)
	var et int
	if ct < 0 {
		goto bad
	}

	et = int(t.Etype)
	if et == TINTER {
		if ct == CTNIL && n.Type == Types[TNIL] {
			n.Type = t
			return
		}

		defaultlit(np, nil)
		return
	}

	switch ct {
	default:
		goto bad

	case CTNIL:
		switch et {
		default:
			n.Type = nil
			goto bad

			// let normal conversion code handle it
		case TSTRING:
			return

		case TARRAY:
			if !Isslice(t) {
				goto bad
			}

		case TPTR32,
			TPTR64,
			TINTER,
			TMAP,
			TCHAN,
			TFUNC,
			TUNSAFEPTR:
			break

			// A nil literal may be converted to uintptr
		// if it is an unsafe.Pointer
		case TUINTPTR:
			if n.Type.Etype == TUNSAFEPTR {
				n.SetVal(Val{new(Mpint)})
				Mpmovecfix(n.Val().U.(*Mpint), 0)
			} else {
				goto bad
			}
		}

	case CTSTR, CTBOOL:
		if et != int(n.Type.Etype) {
			goto bad
		}

	case CTINT, CTRUNE, CTFLT, CTCPLX:
		if n.Type.Etype == TUNSAFEPTR && t.Etype != TUINTPTR {
			goto bad
		}
		ct := n.Val().Ctype()
		if Isint[et] {
			switch ct {
			default:
				goto bad

			case CTCPLX, CTFLT, CTRUNE:
				n.SetVal(toint(n.Val()))
				fallthrough

			case CTINT:
				overflow(n.Val(), t)
			}
		} else if Isfloat[et] {
			switch ct {
			default:
				goto bad

			case CTCPLX, CTINT, CTRUNE:
				n.SetVal(toflt(n.Val()))
				fallthrough

			case CTFLT:
				n.SetVal(Val{truncfltlit(n.Val().U.(*Mpflt), t)})
			}
		} else if Iscomplex[et] {
			switch ct {
			default:
				goto bad

			case CTFLT, CTINT, CTRUNE:
				n.SetVal(tocplx(n.Val()))
				fallthrough

			case CTCPLX:
				overflow(n.Val(), t)
			}
		} else if et == TSTRING && (ct == CTINT || ct == CTRUNE) && explicit {
			n.SetVal(tostr(n.Val()))
		} else {
			goto bad
		}
	}

	n.Type = t
	return

bad:
	if n.Diag == 0 {
		if !t.Broke {
			Yyerror("cannot convert %v to type %v", n, t)
		}
		n.Diag = 1
	}

	if isideal(n.Type) {
		defaultlit(&n, nil)
		*np = n
	}
}

func copyval(v Val) Val {
	switch v.Ctype() {
	case CTINT, CTRUNE:
		i := new(Mpint)
		mpmovefixfix(i, v.U.(*Mpint))
		i.Rune = v.U.(*Mpint).Rune
		v.U = i

	case CTFLT:
		f := newMpflt()
		mpmovefltflt(f, v.U.(*Mpflt))
		v.U = f

	case CTCPLX:
		c := new(Mpcplx)
		mpmovefltflt(&c.Real, &v.U.(*Mpcplx).Real)
		mpmovefltflt(&c.Imag, &v.U.(*Mpcplx).Imag)
		v.U = c
	}

	return v
}

func tocplx(v Val) Val {
	switch v.Ctype() {
	case CTINT, CTRUNE:
		c := new(Mpcplx)
		Mpmovefixflt(&c.Real, v.U.(*Mpint))
		Mpmovecflt(&c.Imag, 0.0)
		v.U = c

	case CTFLT:
		c := new(Mpcplx)
		mpmovefltflt(&c.Real, v.U.(*Mpflt))
		Mpmovecflt(&c.Imag, 0.0)
		v.U = c
	}

	return v
}

func toflt(v Val) Val {
	switch v.Ctype() {
	case CTINT, CTRUNE:
		f := newMpflt()
		Mpmovefixflt(f, v.U.(*Mpint))
		v.U = f

	case CTCPLX:
		f := newMpflt()
		mpmovefltflt(f, &v.U.(*Mpcplx).Real)
		if mpcmpfltc(&v.U.(*Mpcplx).Imag, 0) != 0 {
			Yyerror("constant %v%vi truncated to real", Fconv(&v.U.(*Mpcplx).Real, obj.FmtSharp), Fconv(&v.U.(*Mpcplx).Imag, obj.FmtSharp|obj.FmtSign))
		}
		v.U = f
	}

	return v
}

func toint(v Val) Val {
	switch v.Ctype() {
	case CTRUNE:
		i := new(Mpint)
		mpmovefixfix(i, v.U.(*Mpint))
		v.U = i

	case CTFLT:
		i := new(Mpint)
		if f := v.U.(*Mpflt); mpmovefltfix(i, f) < 0 {
			msg := "constant %v truncated to integer"
			// provide better error message if mpmovefltfix failed because f was too large
			if f.Val.IsInt() {
				msg = "constant %v overflows integer"
			}
			Yyerror(msg, Fconv(f, obj.FmtSharp))
		}
		v.U = i

	case CTCPLX:
		i := new(Mpint)
		if mpmovefltfix(i, &v.U.(*Mpcplx).Real) < 0 {
			Yyerror("constant %v%vi truncated to integer", Fconv(&v.U.(*Mpcplx).Real, obj.FmtSharp), Fconv(&v.U.(*Mpcplx).Imag, obj.FmtSharp|obj.FmtSign))
		}
		if mpcmpfltc(&v.U.(*Mpcplx).Imag, 0) != 0 {
			Yyerror("constant %v%vi truncated to real", Fconv(&v.U.(*Mpcplx).Real, obj.FmtSharp), Fconv(&v.U.(*Mpcplx).Imag, obj.FmtSharp|obj.FmtSign))
		}
		v.U = i
	}

	return v
}

func doesoverflow(v Val, t *Type) bool {
	switch v.Ctype() {
	case CTINT, CTRUNE:
		if !Isint[t.Etype] {
			Fatalf("overflow: %v integer constant", t)
		}
		if Mpcmpfixfix(v.U.(*Mpint), Minintval[t.Etype]) < 0 || Mpcmpfixfix(v.U.(*Mpint), Maxintval[t.Etype]) > 0 {
			return true
		}

	case CTFLT:
		if !Isfloat[t.Etype] {
			Fatalf("overflow: %v floating-point constant", t)
		}
		if mpcmpfltflt(v.U.(*Mpflt), minfltval[t.Etype]) <= 0 || mpcmpfltflt(v.U.(*Mpflt), maxfltval[t.Etype]) >= 0 {
			return true
		}

	case CTCPLX:
		if !Iscomplex[t.Etype] {
			Fatalf("overflow: %v complex constant", t)
		}
		if mpcmpfltflt(&v.U.(*Mpcplx).Real, minfltval[t.Etype]) <= 0 || mpcmpfltflt(&v.U.(*Mpcplx).Real, maxfltval[t.Etype]) >= 0 || mpcmpfltflt(&v.U.(*Mpcplx).Imag, minfltval[t.Etype]) <= 0 || mpcmpfltflt(&v.U.(*Mpcplx).Imag, maxfltval[t.Etype]) >= 0 {
			return true
		}
	}

	return false
}

func overflow(v Val, t *Type) {
	// v has already been converted
	// to appropriate form for t.
	if t == nil || t.Etype == TIDEAL {
		return
	}

	// Only uintptrs may be converted to unsafe.Pointer, which cannot overflow.
	if t.Etype == TUNSAFEPTR {
		return
	}

	if doesoverflow(v, t) {
		Yyerror("constant %s overflows %v", Vconv(v, 0), t)
	}
}

func tostr(v Val) Val {
	switch v.Ctype() {
	case CTINT, CTRUNE:
		if Mpcmpfixfix(v.U.(*Mpint), Minintval[TINT]) < 0 || Mpcmpfixfix(v.U.(*Mpint), Maxintval[TINT]) > 0 {
			Yyerror("overflow in int -> string")
		}
		r := uint(Mpgetfix(v.U.(*Mpint)))
		v = Val{}
		v.U = string(r)

	case CTFLT:
		Yyerror("no float -> string")
		fallthrough

	case CTNIL:
		v = Val{}
		v.U = ""
	}

	return v
}

func consttype(n *Node) Ctype {
	if n == nil || n.Op != OLITERAL {
		return -1
	}
	return n.Val().Ctype()
}

func Isconst(n *Node, ct Ctype) bool {
	t := consttype(n)

	// If the caller is asking for CTINT, allow CTRUNE too.
	// Makes life easier for back ends.
	return t == ct || (ct == CTINT && t == CTRUNE)
}

func saveorig(n *Node) *Node {
	if n == n.Orig {
		// duplicate node for n->orig.
		n1 := Nod(OLITERAL, nil, nil)

		n.Orig = n1
		*n1 = *n
	}

	return n.Orig
}

// if n is constant, rewrite as OLITERAL node.
func evconst(n *Node) {
	// pick off just the opcodes that can be
	// constant evaluated.
	switch n.Op {
	default:
		return

	case OADD,
		OAND,
		OANDAND,
		OANDNOT,
		OARRAYBYTESTR,
		OCOM,
		ODIV,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLSH,
		OLT,
		OMINUS,
		OMOD,
		OMUL,
		ONE,
		ONOT,
		OOR,
		OOROR,
		OPLUS,
		ORSH,
		OSUB,
		OXOR:
		break

	case OCONV:
		if n.Type == nil {
			return
		}
		if !okforconst[n.Type.Etype] && n.Type.Etype != TNIL {
			return
		}

		// merge adjacent constants in the argument list.
	case OADDSTR:
		var nr *Node
		var nl *Node
		var l2 *NodeList
		for l1 := n.List; l1 != nil; l1 = l1.Next {
			if Isconst(l1.N, CTSTR) && l1.Next != nil && Isconst(l1.Next.N, CTSTR) {
				// merge from l1 up to but not including l2
				var strs []string
				l2 = l1
				for l2 != nil && Isconst(l2.N, CTSTR) {
					nr = l2.N
					strs = append(strs, nr.Val().U.(string))
					l2 = l2.Next
				}

				nl = Nod(OXXX, nil, nil)
				*nl = *l1.N
				nl.Orig = nl
				nl.SetVal(Val{strings.Join(strs, "")})
				l1.N = nl
				l1.Next = l2
			}
		}

		// fix list end pointer.
		for l2 := n.List; l2 != nil; l2 = l2.Next {
			n.List.End = l2
		}

		// collapse single-constant list to single constant.
		if count(n.List) == 1 && Isconst(n.List.N, CTSTR) {
			n.Op = OLITERAL
			n.SetVal(n.List.N.Val())
		}

		return
	}

	nl := n.Left
	if nl == nil || nl.Type == nil {
		return
	}
	if consttype(nl) < 0 {
		return
	}
	wl := int(nl.Type.Etype)
	if Isint[wl] || Isfloat[wl] || Iscomplex[wl] {
		wl = TIDEAL
	}

	// avoid constant conversions in switches below
	const (
		CTINT_         = uint32(CTINT)
		CTRUNE_        = uint32(CTRUNE)
		CTFLT_         = uint32(CTFLT)
		CTCPLX_        = uint32(CTCPLX)
		CTSTR_         = uint32(CTSTR)
		CTBOOL_        = uint32(CTBOOL)
		CTNIL_         = uint32(CTNIL)
		OCONV_         = uint32(OCONV) << 16
		OARRAYBYTESTR_ = uint32(OARRAYBYTESTR) << 16
		OPLUS_         = uint32(OPLUS) << 16
		OMINUS_        = uint32(OMINUS) << 16
		OCOM_          = uint32(OCOM) << 16
		ONOT_          = uint32(ONOT) << 16
		OLSH_          = uint32(OLSH) << 16
		ORSH_          = uint32(ORSH) << 16
		OADD_          = uint32(OADD) << 16
		OSUB_          = uint32(OSUB) << 16
		OMUL_          = uint32(OMUL) << 16
		ODIV_          = uint32(ODIV) << 16
		OMOD_          = uint32(OMOD) << 16
		OOR_           = uint32(OOR) << 16
		OAND_          = uint32(OAND) << 16
		OANDNOT_       = uint32(OANDNOT) << 16
		OXOR_          = uint32(OXOR) << 16
		OEQ_           = uint32(OEQ) << 16
		ONE_           = uint32(ONE) << 16
		OLT_           = uint32(OLT) << 16
		OLE_           = uint32(OLE) << 16
		OGE_           = uint32(OGE) << 16
		OGT_           = uint32(OGT) << 16
		OOROR_         = uint32(OOROR) << 16
		OANDAND_       = uint32(OANDAND) << 16
	)

	nr := n.Right
	var rv Val
	var lno int
	var wr int
	var v Val
	var norig *Node
	var nn *Node
	if nr == nil {
		// copy numeric value to avoid modifying
		// nl, in case someone still refers to it (e.g. iota).
		v = nl.Val()

		if wl == TIDEAL {
			v = copyval(v)
		}

		switch uint32(n.Op)<<16 | uint32(v.Ctype()) {
		default:
			if n.Diag == 0 {
				Yyerror("illegal constant expression %v %v", Oconv(int(n.Op), 0), nl.Type)
				n.Diag = 1
			}
			return

		case OCONV_ | CTNIL_,
			OARRAYBYTESTR_ | CTNIL_:
			if n.Type.Etype == TSTRING {
				v = tostr(v)
				nl.Type = n.Type
				break
			}
			fallthrough

			// fall through
		case OCONV_ | CTINT_,
			OCONV_ | CTRUNE_,
			OCONV_ | CTFLT_,
			OCONV_ | CTSTR_,
			OCONV_ | CTBOOL_:
			convlit1(&nl, n.Type, true)

			v = nl.Val()

		case OPLUS_ | CTINT_,
			OPLUS_ | CTRUNE_:
			break

		case OMINUS_ | CTINT_,
			OMINUS_ | CTRUNE_:
			mpnegfix(v.U.(*Mpint))

		case OCOM_ | CTINT_,
			OCOM_ | CTRUNE_:
			et := Txxx
			if nl.Type != nil {
				et = int(nl.Type.Etype)
			}

			// calculate the mask in b
			// result will be (a ^ mask)
			var b Mpint
			switch et {
			// signed guys change sign
			default:
				Mpmovecfix(&b, -1)

				// unsigned guys invert their bits
			case TUINT8,
				TUINT16,
				TUINT32,
				TUINT64,
				TUINT,
				TUINTPTR:
				mpmovefixfix(&b, Maxintval[et])
			}

			mpxorfixfix(v.U.(*Mpint), &b)

		case OPLUS_ | CTFLT_:
			break

		case OMINUS_ | CTFLT_:
			mpnegflt(v.U.(*Mpflt))

		case OPLUS_ | CTCPLX_:
			break

		case OMINUS_ | CTCPLX_:
			mpnegflt(&v.U.(*Mpcplx).Real)
			mpnegflt(&v.U.(*Mpcplx).Imag)

		case ONOT_ | CTBOOL_:
			if !v.U.(bool) {
				goto settrue
			}
			goto setfalse
		}
		goto ret
	}
	if nr.Type == nil {
		return
	}
	if consttype(nr) < 0 {
		return
	}
	wr = int(nr.Type.Etype)
	if Isint[wr] || Isfloat[wr] || Iscomplex[wr] {
		wr = TIDEAL
	}

	// check for compatible general types (numeric, string, etc)
	if wl != wr {
		goto illegal
	}

	// check for compatible types.
	switch n.Op {
	// ideal const mixes with anything but otherwise must match.
	default:
		if nl.Type.Etype != TIDEAL {
			defaultlit(&nr, nl.Type)
			n.Right = nr
		}

		if nr.Type.Etype != TIDEAL {
			defaultlit(&nl, nr.Type)
			n.Left = nl
		}

		if nl.Type.Etype != nr.Type.Etype {
			goto illegal
		}

		// right must be unsigned.
	// left can be ideal.
	case OLSH, ORSH:
		defaultlit(&nr, Types[TUINT])

		n.Right = nr
		if nr.Type != nil && (Issigned[nr.Type.Etype] || !Isint[nr.Type.Etype]) {
			goto illegal
		}
		if nl.Val().Ctype() != CTRUNE {
			nl.SetVal(toint(nl.Val()))
		}
		nr.SetVal(toint(nr.Val()))
	}

	// copy numeric value to avoid modifying
	// n->left, in case someone still refers to it (e.g. iota).
	v = nl.Val()

	if wl == TIDEAL {
		v = copyval(v)
	}

	rv = nr.Val()

	// convert to common ideal
	if v.Ctype() == CTCPLX || rv.Ctype() == CTCPLX {
		v = tocplx(v)
		rv = tocplx(rv)
	}

	if v.Ctype() == CTFLT || rv.Ctype() == CTFLT {
		v = toflt(v)
		rv = toflt(rv)
	}

	// Rune and int turns into rune.
	if v.Ctype() == CTRUNE && rv.Ctype() == CTINT {
		i := new(Mpint)
		mpmovefixfix(i, rv.U.(*Mpint))
		i.Rune = true
		rv.U = i
	}
	if v.Ctype() == CTINT && rv.Ctype() == CTRUNE {
		if n.Op == OLSH || n.Op == ORSH {
			i := new(Mpint)
			mpmovefixfix(i, rv.U.(*Mpint))
			rv.U = i
		} else {
			i := new(Mpint)
			mpmovefixfix(i, v.U.(*Mpint))
			i.Rune = true
			v.U = i
		}
	}

	if v.Ctype() != rv.Ctype() {
		// Use of undefined name as constant?
		if (v.Ctype() == 0 || rv.Ctype() == 0) && nerrors > 0 {
			return
		}
		Fatalf("constant type mismatch %v(%d) %v(%d)", nl.Type, v.Ctype(), nr.Type, rv.Ctype())
	}

	// run op
	switch uint32(n.Op)<<16 | uint32(v.Ctype()) {
	default:
		goto illegal

	case OADD_ | CTINT_,
		OADD_ | CTRUNE_:
		mpaddfixfix(v.U.(*Mpint), rv.U.(*Mpint), 0)

	case OSUB_ | CTINT_,
		OSUB_ | CTRUNE_:
		mpsubfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case OMUL_ | CTINT_,
		OMUL_ | CTRUNE_:
		mpmulfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case ODIV_ | CTINT_,
		ODIV_ | CTRUNE_:
		if mpcmpfixc(rv.U.(*Mpint), 0) == 0 {
			Yyerror("division by zero")
			mpsetovf(v.U.(*Mpint))
			break
		}

		mpdivfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case OMOD_ | CTINT_,
		OMOD_ | CTRUNE_:
		if mpcmpfixc(rv.U.(*Mpint), 0) == 0 {
			Yyerror("division by zero")
			mpsetovf(v.U.(*Mpint))
			break
		}

		mpmodfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case OLSH_ | CTINT_,
		OLSH_ | CTRUNE_:
		mplshfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case ORSH_ | CTINT_,
		ORSH_ | CTRUNE_:
		mprshfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case OOR_ | CTINT_,
		OOR_ | CTRUNE_:
		mporfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case OAND_ | CTINT_,
		OAND_ | CTRUNE_:
		mpandfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case OANDNOT_ | CTINT_,
		OANDNOT_ | CTRUNE_:
		mpandnotfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case OXOR_ | CTINT_,
		OXOR_ | CTRUNE_:
		mpxorfixfix(v.U.(*Mpint), rv.U.(*Mpint))

	case OADD_ | CTFLT_:
		mpaddfltflt(v.U.(*Mpflt), rv.U.(*Mpflt))

	case OSUB_ | CTFLT_:
		mpsubfltflt(v.U.(*Mpflt), rv.U.(*Mpflt))

	case OMUL_ | CTFLT_:
		mpmulfltflt(v.U.(*Mpflt), rv.U.(*Mpflt))

	case ODIV_ | CTFLT_:
		if mpcmpfltc(rv.U.(*Mpflt), 0) == 0 {
			Yyerror("division by zero")
			Mpmovecflt(v.U.(*Mpflt), 1.0)
			break
		}

		mpdivfltflt(v.U.(*Mpflt), rv.U.(*Mpflt))

		// The default case above would print 'ideal % ideal',
	// which is not quite an ideal error.
	case OMOD_ | CTFLT_:
		if n.Diag == 0 {
			Yyerror("illegal constant expression: floating-point %% operation")
			n.Diag = 1
		}

		return

	case OADD_ | CTCPLX_:
		mpaddfltflt(&v.U.(*Mpcplx).Real, &rv.U.(*Mpcplx).Real)
		mpaddfltflt(&v.U.(*Mpcplx).Imag, &rv.U.(*Mpcplx).Imag)

	case OSUB_ | CTCPLX_:
		mpsubfltflt(&v.U.(*Mpcplx).Real, &rv.U.(*Mpcplx).Real)
		mpsubfltflt(&v.U.(*Mpcplx).Imag, &rv.U.(*Mpcplx).Imag)

	case OMUL_ | CTCPLX_:
		cmplxmpy(v.U.(*Mpcplx), rv.U.(*Mpcplx))

	case ODIV_ | CTCPLX_:
		if mpcmpfltc(&rv.U.(*Mpcplx).Real, 0) == 0 && mpcmpfltc(&rv.U.(*Mpcplx).Imag, 0) == 0 {
			Yyerror("complex division by zero")
			Mpmovecflt(&rv.U.(*Mpcplx).Real, 1.0)
			Mpmovecflt(&rv.U.(*Mpcplx).Imag, 0.0)
			break
		}

		cmplxdiv(v.U.(*Mpcplx), rv.U.(*Mpcplx))

	case OEQ_ | CTNIL_:
		goto settrue

	case ONE_ | CTNIL_:
		goto setfalse

	case OEQ_ | CTINT_,
		OEQ_ | CTRUNE_:
		if Mpcmpfixfix(v.U.(*Mpint), rv.U.(*Mpint)) == 0 {
			goto settrue
		}
		goto setfalse

	case ONE_ | CTINT_,
		ONE_ | CTRUNE_:
		if Mpcmpfixfix(v.U.(*Mpint), rv.U.(*Mpint)) != 0 {
			goto settrue
		}
		goto setfalse

	case OLT_ | CTINT_,
		OLT_ | CTRUNE_:
		if Mpcmpfixfix(v.U.(*Mpint), rv.U.(*Mpint)) < 0 {
			goto settrue
		}
		goto setfalse

	case OLE_ | CTINT_,
		OLE_ | CTRUNE_:
		if Mpcmpfixfix(v.U.(*Mpint), rv.U.(*Mpint)) <= 0 {
			goto settrue
		}
		goto setfalse

	case OGE_ | CTINT_,
		OGE_ | CTRUNE_:
		if Mpcmpfixfix(v.U.(*Mpint), rv.U.(*Mpint)) >= 0 {
			goto settrue
		}
		goto setfalse

	case OGT_ | CTINT_,
		OGT_ | CTRUNE_:
		if Mpcmpfixfix(v.U.(*Mpint), rv.U.(*Mpint)) > 0 {
			goto settrue
		}
		goto setfalse

	case OEQ_ | CTFLT_:
		if mpcmpfltflt(v.U.(*Mpflt), rv.U.(*Mpflt)) == 0 {
			goto settrue
		}
		goto setfalse

	case ONE_ | CTFLT_:
		if mpcmpfltflt(v.U.(*Mpflt), rv.U.(*Mpflt)) != 0 {
			goto settrue
		}
		goto setfalse

	case OLT_ | CTFLT_:
		if mpcmpfltflt(v.U.(*Mpflt), rv.U.(*Mpflt)) < 0 {
			goto settrue
		}
		goto setfalse

	case OLE_ | CTFLT_:
		if mpcmpfltflt(v.U.(*Mpflt), rv.U.(*Mpflt)) <= 0 {
			goto settrue
		}
		goto setfalse

	case OGE_ | CTFLT_:
		if mpcmpfltflt(v.U.(*Mpflt), rv.U.(*Mpflt)) >= 0 {
			goto settrue
		}
		goto setfalse

	case OGT_ | CTFLT_:
		if mpcmpfltflt(v.U.(*Mpflt), rv.U.(*Mpflt)) > 0 {
			goto settrue
		}
		goto setfalse

	case OEQ_ | CTCPLX_:
		if mpcmpfltflt(&v.U.(*Mpcplx).Real, &rv.U.(*Mpcplx).Real) == 0 && mpcmpfltflt(&v.U.(*Mpcplx).Imag, &rv.U.(*Mpcplx).Imag) == 0 {
			goto settrue
		}
		goto setfalse

	case ONE_ | CTCPLX_:
		if mpcmpfltflt(&v.U.(*Mpcplx).Real, &rv.U.(*Mpcplx).Real) != 0 || mpcmpfltflt(&v.U.(*Mpcplx).Imag, &rv.U.(*Mpcplx).Imag) != 0 {
			goto settrue
		}
		goto setfalse

	case OEQ_ | CTSTR_:
		if strlit(nl) == strlit(nr) {
			goto settrue
		}
		goto setfalse

	case ONE_ | CTSTR_:
		if strlit(nl) != strlit(nr) {
			goto settrue
		}
		goto setfalse

	case OLT_ | CTSTR_:
		if strlit(nl) < strlit(nr) {
			goto settrue
		}
		goto setfalse

	case OLE_ | CTSTR_:
		if strlit(nl) <= strlit(nr) {
			goto settrue
		}
		goto setfalse

	case OGE_ | CTSTR_:
		if strlit(nl) >= strlit(nr) {
			goto settrue
		}
		goto setfalse

	case OGT_ | CTSTR_:
		if strlit(nl) > strlit(nr) {
			goto settrue
		}
		goto setfalse

	case OOROR_ | CTBOOL_:
		if v.U.(bool) || rv.U.(bool) {
			goto settrue
		}
		goto setfalse

	case OANDAND_ | CTBOOL_:
		if v.U.(bool) && rv.U.(bool) {
			goto settrue
		}
		goto setfalse

	case OEQ_ | CTBOOL_:
		if v.U.(bool) == rv.U.(bool) {
			goto settrue
		}
		goto setfalse

	case ONE_ | CTBOOL_:
		if v.U.(bool) != rv.U.(bool) {
			goto settrue
		}
		goto setfalse
	}

	goto ret

ret:
	norig = saveorig(n)
	*n = *nl

	// restore value of n->orig.
	n.Orig = norig

	n.SetVal(v)

	// check range.
	lno = int(setlineno(n))

	overflow(v, n.Type)
	lineno = int32(lno)

	// truncate precision for non-ideal float.
	if v.Ctype() == CTFLT && n.Type.Etype != TIDEAL {
		n.SetVal(Val{truncfltlit(v.U.(*Mpflt), n.Type)})
	}
	return

settrue:
	nn = Nodbool(true)
	nn.Orig = saveorig(n)
	if !iscmp[n.Op] {
		nn.Type = nl.Type
	}
	*n = *nn
	return

setfalse:
	nn = Nodbool(false)
	nn.Orig = saveorig(n)
	if !iscmp[n.Op] {
		nn.Type = nl.Type
	}
	*n = *nn
	return

illegal:
	if n.Diag == 0 {
		Yyerror("illegal constant expression: %v %v %v", nl.Type, Oconv(int(n.Op), 0), nr.Type)
		n.Diag = 1
	}
}

func nodlit(v Val) *Node {
	n := Nod(OLITERAL, nil, nil)
	n.SetVal(v)
	switch v.Ctype() {
	default:
		Fatalf("nodlit ctype %d", v.Ctype())

	case CTSTR:
		n.Type = idealstring

	case CTBOOL:
		n.Type = idealbool

	case CTINT, CTRUNE, CTFLT, CTCPLX:
		n.Type = Types[TIDEAL]

	case CTNIL:
		n.Type = Types[TNIL]
	}

	return n
}

func nodcplxlit(r Val, i Val) *Node {
	r = toflt(r)
	i = toflt(i)

	c := new(Mpcplx)
	n := Nod(OLITERAL, nil, nil)
	n.Type = Types[TIDEAL]
	n.SetVal(Val{c})

	if r.Ctype() != CTFLT || i.Ctype() != CTFLT {
		Fatalf("nodcplxlit ctype %d/%d", r.Ctype(), i.Ctype())
	}

	mpmovefltflt(&c.Real, r.U.(*Mpflt))
	mpmovefltflt(&c.Imag, i.U.(*Mpflt))
	return n
}

// idealkind returns a constant kind like consttype
// but for an arbitrary "ideal" (untyped constant) expression.
func idealkind(n *Node) Ctype {
	if n == nil || !isideal(n.Type) {
		return CTxxx
	}

	switch n.Op {
	default:
		return CTxxx

	case OLITERAL:
		return n.Val().Ctype()

		// numeric kinds.
	case OADD,
		OAND,
		OANDNOT,
		OCOM,
		ODIV,
		OMINUS,
		OMOD,
		OMUL,
		OSUB,
		OXOR,
		OOR,
		OPLUS:
		k1 := idealkind(n.Left)

		k2 := idealkind(n.Right)
		if k1 > k2 {
			return k1
		} else {
			return k2
		}

	case OREAL, OIMAG:
		return CTFLT

	case OCOMPLEX:
		return CTCPLX

	case OADDSTR:
		return CTSTR

	case OANDAND,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLT,
		ONE,
		ONOT,
		OOROR,
		OCMPSTR,
		OCMPIFACE:
		return CTBOOL

		// shifts (beware!).
	case OLSH, ORSH:
		return idealkind(n.Left)
	}
}

func defaultlit(np **Node, t *Type) {
	n := *np
	if n == nil || !isideal(n.Type) {
		return
	}

	if n.Op == OLITERAL {
		nn := Nod(OXXX, nil, nil)
		*nn = *n
		n = nn
		*np = n
	}

	lno := int(setlineno(n))
	ctype := idealkind(n)
	var t1 *Type
	switch ctype {
	default:
		if t != nil {
			Convlit(np, t)
			return
		}

		if n.Val().Ctype() == CTNIL {
			lineno = int32(lno)
			if n.Diag == 0 {
				Yyerror("use of untyped nil")
				n.Diag = 1
			}

			n.Type = nil
			break
		}

		if n.Val().Ctype() == CTSTR {
			t1 := Types[TSTRING]
			Convlit(np, t1)
			break
		}

		Yyerror("defaultlit: unknown literal: %v", n)

	case CTxxx:
		Fatalf("defaultlit: idealkind is CTxxx: %v", Nconv(n, obj.FmtSign))

	case CTBOOL:
		t1 := Types[TBOOL]
		if t != nil && t.Etype == TBOOL {
			t1 = t
		}
		Convlit(np, t1)

	case CTINT:
		t1 = Types[TINT]
		goto num

	case CTRUNE:
		t1 = runetype
		goto num

	case CTFLT:
		t1 = Types[TFLOAT64]
		goto num

	case CTCPLX:
		t1 = Types[TCOMPLEX128]
		goto num
	}

	lineno = int32(lno)
	return

num:
	// Note: n.Val().Ctype() can be CTxxx (not a constant) here
	// in the case of an untyped non-constant value, like 1<<i.
	v1 := n.Val()
	if t != nil {
		if Isint[t.Etype] {
			t1 = t
			v1 = toint(n.Val())
		} else if Isfloat[t.Etype] {
			t1 = t
			v1 = toflt(n.Val())
		} else if Iscomplex[t.Etype] {
			t1 = t
			v1 = tocplx(n.Val())
		}
		if n.Val().Ctype() != CTxxx {
			n.SetVal(v1)
		}
	}

	if n.Val().Ctype() != CTxxx {
		overflow(n.Val(), t1)
	}
	Convlit(np, t1)
	lineno = int32(lno)
	return
}

// defaultlit on both nodes simultaneously;
// if they're both ideal going in they better
// get the same type going out.
// force means must assign concrete (non-ideal) type.
func defaultlit2(lp **Node, rp **Node, force int) {
	l := *lp
	r := *rp
	if l.Type == nil || r.Type == nil {
		return
	}
	if !isideal(l.Type) {
		Convlit(rp, l.Type)
		return
	}

	if !isideal(r.Type) {
		Convlit(lp, r.Type)
		return
	}

	if force == 0 {
		return
	}
	if l.Type.Etype == TBOOL {
		Convlit(lp, Types[TBOOL])
		Convlit(rp, Types[TBOOL])
	}

	lkind := idealkind(l)
	rkind := idealkind(r)
	if lkind == CTCPLX || rkind == CTCPLX {
		Convlit(lp, Types[TCOMPLEX128])
		Convlit(rp, Types[TCOMPLEX128])
		return
	}

	if lkind == CTFLT || rkind == CTFLT {
		Convlit(lp, Types[TFLOAT64])
		Convlit(rp, Types[TFLOAT64])
		return
	}

	if lkind == CTRUNE || rkind == CTRUNE {
		Convlit(lp, runetype)
		Convlit(rp, runetype)
		return
	}

	Convlit(lp, Types[TINT])
	Convlit(rp, Types[TINT])
}

// strlit returns the value of a literal string Node as a string.
func strlit(n *Node) string {
	return n.Val().U.(string)
}

func Smallintconst(n *Node) bool {
	if n.Op == OLITERAL && Isconst(n, CTINT) && n.Type != nil {
		switch Simtype[n.Type.Etype] {
		case TINT8,
			TUINT8,
			TINT16,
			TUINT16,
			TINT32,
			TUINT32,
			TBOOL,
			TPTR32:
			return true

		case TIDEAL, TINT64, TUINT64, TPTR64:
			if Mpcmpfixfix(n.Val().U.(*Mpint), Minintval[TINT32]) < 0 || Mpcmpfixfix(n.Val().U.(*Mpint), Maxintval[TINT32]) > 0 {
				break
			}
			return true
		}
	}

	return false
}

func nonnegconst(n *Node) int {
	if n.Op == OLITERAL && n.Type != nil {
		switch Simtype[n.Type.Etype] {
		// check negative and 2^31
		case TINT8,
			TUINT8,
			TINT16,
			TUINT16,
			TINT32,
			TUINT32,
			TINT64,
			TUINT64,
			TIDEAL:
			if Mpcmpfixfix(n.Val().U.(*Mpint), Minintval[TUINT32]) < 0 || Mpcmpfixfix(n.Val().U.(*Mpint), Maxintval[TINT32]) > 0 {
				break
			}
			return int(Mpgetfix(n.Val().U.(*Mpint)))
		}
	}

	return -1
}

// convert x to type et and back to int64
// for sign extension and truncation.
func iconv(x int64, et EType) int64 {
	switch et {
	case TINT8:
		x = int64(int8(x))

	case TUINT8:
		x = int64(uint8(x))

	case TINT16:
		x = int64(int16(x))

	case TUINT16:
		x = int64(uint64(x))

	case TINT32:
		x = int64(int32(x))

	case TUINT32:
		x = int64(uint32(x))

	case TINT64, TUINT64:
		break
	}

	return x
}

// Convconst converts constant node n to type t and
// places the result in con.
func (n *Node) Convconst(con *Node, t *Type) {
	tt := Simsimtype(t)

	// copy the constant for conversion
	Nodconst(con, Types[TINT8], 0)

	con.Type = t
	con.SetVal(n.Val())

	if Isint[tt] {
		con.SetVal(Val{new(Mpint)})
		var i int64
		switch n.Val().Ctype() {
		default:
			Fatalf("convconst ctype=%d %v", n.Val().Ctype(), Tconv(t, obj.FmtLong))

		case CTINT, CTRUNE:
			i = Mpgetfix(n.Val().U.(*Mpint))

		case CTBOOL:
			i = int64(obj.Bool2int(n.Val().U.(bool)))

		case CTNIL:
			i = 0
		}

		i = iconv(i, tt)
		Mpmovecfix(con.Val().U.(*Mpint), i)
		return
	}

	if Isfloat[tt] {
		con.SetVal(toflt(con.Val()))
		if con.Val().Ctype() != CTFLT {
			Fatalf("convconst ctype=%d %v", con.Val().Ctype(), t)
		}
		if tt == TFLOAT32 {
			con.SetVal(Val{truncfltlit(con.Val().U.(*Mpflt), t)})
		}
		return
	}

	if Iscomplex[tt] {
		con.SetVal(tocplx(con.Val()))
		if tt == TCOMPLEX64 {
			con.Val().U.(*Mpcplx).Real = *truncfltlit(&con.Val().U.(*Mpcplx).Real, Types[TFLOAT32])
			con.Val().U.(*Mpcplx).Imag = *truncfltlit(&con.Val().U.(*Mpcplx).Imag, Types[TFLOAT32])
		}
		return
	}

	Fatalf("convconst %v constant", Tconv(t, obj.FmtLong))
}

// complex multiply v *= rv
//	(a, b) * (c, d) = (a*c - b*d, b*c + a*d)
func cmplxmpy(v *Mpcplx, rv *Mpcplx) {
	var ac Mpflt
	var bd Mpflt
	var bc Mpflt
	var ad Mpflt

	mpmovefltflt(&ac, &v.Real)
	mpmulfltflt(&ac, &rv.Real) // ac

	mpmovefltflt(&bd, &v.Imag)

	mpmulfltflt(&bd, &rv.Imag) // bd

	mpmovefltflt(&bc, &v.Imag)

	mpmulfltflt(&bc, &rv.Real) // bc

	mpmovefltflt(&ad, &v.Real)

	mpmulfltflt(&ad, &rv.Imag) // ad

	mpmovefltflt(&v.Real, &ac)

	mpsubfltflt(&v.Real, &bd) // ac-bd

	mpmovefltflt(&v.Imag, &bc)

	mpaddfltflt(&v.Imag, &ad) // bc+ad
}

// complex divide v /= rv
//	(a, b) / (c, d) = ((a*c + b*d), (b*c - a*d))/(c*c + d*d)
func cmplxdiv(v *Mpcplx, rv *Mpcplx) {
	var ac Mpflt
	var bd Mpflt
	var bc Mpflt
	var ad Mpflt
	var cc_plus_dd Mpflt

	mpmovefltflt(&cc_plus_dd, &rv.Real)
	mpmulfltflt(&cc_plus_dd, &rv.Real) // cc

	mpmovefltflt(&ac, &rv.Imag)

	mpmulfltflt(&ac, &rv.Imag) // dd

	mpaddfltflt(&cc_plus_dd, &ac) // cc+dd

	mpmovefltflt(&ac, &v.Real)

	mpmulfltflt(&ac, &rv.Real) // ac

	mpmovefltflt(&bd, &v.Imag)

	mpmulfltflt(&bd, &rv.Imag) // bd

	mpmovefltflt(&bc, &v.Imag)

	mpmulfltflt(&bc, &rv.Real) // bc

	mpmovefltflt(&ad, &v.Real)

	mpmulfltflt(&ad, &rv.Imag) // ad

	mpmovefltflt(&v.Real, &ac)

	mpaddfltflt(&v.Real, &bd)         // ac+bd
	mpdivfltflt(&v.Real, &cc_plus_dd) // (ac+bd)/(cc+dd)

	mpmovefltflt(&v.Imag, &bc)

	mpsubfltflt(&v.Imag, &ad)         // bc-ad
	mpdivfltflt(&v.Imag, &cc_plus_dd) // (bc+ad)/(cc+dd)
}

// Is n a Go language constant (as opposed to a compile-time constant)?
// Expressions derived from nil, like string([]byte(nil)), while they
// may be known at compile time, are not Go language constants.
// Only called for expressions known to evaluated to compile-time
// constants.
func isgoconst(n *Node) bool {
	if n.Orig != nil {
		n = n.Orig
	}

	switch n.Op {
	case OADD,
		OADDSTR,
		OAND,
		OANDAND,
		OANDNOT,
		OCOM,
		ODIV,
		OEQ,
		OGE,
		OGT,
		OLE,
		OLSH,
		OLT,
		OMINUS,
		OMOD,
		OMUL,
		ONE,
		ONOT,
		OOR,
		OOROR,
		OPLUS,
		ORSH,
		OSUB,
		OXOR,
		OIOTA,
		OCOMPLEX,
		OREAL,
		OIMAG:
		if isgoconst(n.Left) && (n.Right == nil || isgoconst(n.Right)) {
			return true
		}

	case OCONV:
		if okforconst[n.Type.Etype] && isgoconst(n.Left) {
			return true
		}

	case OLEN, OCAP:
		l := n.Left
		if isgoconst(l) {
			return true
		}

		// Special case: len/cap is constant when applied to array or
		// pointer to array when the expression does not contain
		// function calls or channel receive operations.
		t := l.Type

		if t != nil && Isptr[t.Etype] {
			t = t.Type
		}
		if Isfixedarray(t) && !hascallchan(l) {
			return true
		}

	case OLITERAL:
		if n.Val().Ctype() != CTNIL {
			return true
		}

	case ONAME:
		l := n.Sym.Def
		if l != nil && l.Op == OLITERAL && n.Val().Ctype() != CTNIL {
			return true
		}

	case ONONAME:
		if n.Sym.Def != nil && n.Sym.Def.Op == OIOTA {
			return true
		}

		// Only constant calls are unsafe.Alignof, Offsetof, and Sizeof.
	case OCALL:
		l := n.Left

		for l.Op == OPAREN {
			l = l.Left
		}
		if l.Op != ONAME || l.Sym.Pkg != unsafepkg {
			break
		}
		if l.Sym.Name == "Alignof" || l.Sym.Name == "Offsetof" || l.Sym.Name == "Sizeof" {
			return true
		}
	}

	//dump("nonconst", n);
	return false
}

func hascallchan(n *Node) bool {
	if n == nil {
		return false
	}
	switch n.Op {
	case OAPPEND,
		OCALL,
		OCALLFUNC,
		OCALLINTER,
		OCALLMETH,
		OCAP,
		OCLOSE,
		OCOMPLEX,
		OCOPY,
		ODELETE,
		OIMAG,
		OLEN,
		OMAKE,
		ONEW,
		OPANIC,
		OPRINT,
		OPRINTN,
		OREAL,
		ORECOVER,
		ORECV:
		return true
	}

	if hascallchan(n.Left) || hascallchan(n.Right) {
		return true
	}

	for l := n.List; l != nil; l = l.Next {
		if hascallchan(l.N) {
			return true
		}
	}
	for l := n.Rlist; l != nil; l = l.Next {
		if hascallchan(l.N) {
			return true
		}
	}

	return false
}
