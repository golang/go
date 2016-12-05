// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "strings"

// Ctype describes the constant kind of an "ideal" (untyped) constant.
type Ctype int8

const (
	CTxxx Ctype = iota

	CTINT
	CTRUNE
	CTFLT
	CTCPLX
	CTSTR
	CTBOOL
	CTNIL
)

type Val struct {
	// U contains one of:
	// bool     bool when n.ValCtype() == CTBOOL
	// *Mpint   int when n.ValCtype() == CTINT, rune when n.ValCtype() == CTRUNE
	// *Mpflt   float when n.ValCtype() == CTFLT
	// *Mpcplx  pair of floats when n.ValCtype() == CTCPLX
	// string   string when n.ValCtype() == CTSTR
	// *Nilval  when n.ValCtype() == CTNIL
	U interface{}
}

func (v Val) Ctype() Ctype {
	switch x := v.U.(type) {
	default:
		Fatalf("unexpected Ctype for %T", v.U)
		panic("not reached")
	case nil:
		return 0
	case *NilVal:
		return CTNIL
	case bool:
		return CTBOOL
	case *Mpint:
		if x.Rune {
			return CTRUNE
		}
		return CTINT
	case *Mpflt:
		return CTFLT
	case *Mpcplx:
		return CTCPLX
	case string:
		return CTSTR
	}
}

func eqval(a, b Val) bool {
	if a.Ctype() != b.Ctype() {
		return false
	}
	switch x := a.U.(type) {
	default:
		Fatalf("unexpected Ctype for %T", a.U)
		panic("not reached")
	case *NilVal:
		return true
	case bool:
		y := b.U.(bool)
		return x == y
	case *Mpint:
		y := b.U.(*Mpint)
		return x.Cmp(y) == 0
	case *Mpflt:
		y := b.U.(*Mpflt)
		return x.Cmp(y) == 0
	case *Mpcplx:
		y := b.U.(*Mpcplx)
		return x.Real.Cmp(&y.Real) == 0 && x.Imag.Cmp(&y.Imag) == 0
	case string:
		y := b.U.(string)
		return x == y
	}
}

// Interface returns the constant value stored in v as an interface{}.
// It returns int64s for ints and runes, float64s for floats,
// complex128s for complex values, and nil for constant nils.
func (v Val) Interface() interface{} {
	switch x := v.U.(type) {
	default:
		Fatalf("unexpected Interface for %T", v.U)
		panic("not reached")
	case *NilVal:
		return nil
	case bool, string:
		return x
	case *Mpint:
		return x.Int64()
	case *Mpflt:
		return x.Float64()
	case *Mpcplx:
		return complex(x.Real.Float64(), x.Imag.Float64())
	}
}

type NilVal struct{}

// Int64 returns n as an int64.
// n must be an integer or rune constant.
func (n *Node) Int64() int64 {
	if !Isconst(n, CTINT) {
		Fatalf("Int(%v)", n)
	}
	return n.Val().U.(*Mpint).Int64()
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
	fv.Set(oldv)

	// convert large precision literal floating
	// into limited precision (float64 or float32)
	switch t.Etype {
	case TFLOAT64:
		d := fv.Float64()
		fv.SetFloat64(d)

	case TFLOAT32:
		d := fv.Float32()
		fv.SetFloat64(d)
	}

	return fv
}

// canReuseNode indicates whether it is known to be safe
// to reuse a Node.
type canReuseNode bool

const (
	noReuse canReuseNode = false // not necessarily safe to reuse
	reuseOK canReuseNode = true  // safe to reuse
)

// convert n, if literal, to type t.
// implicit conversion.
// The result of convlit MUST be assigned back to n, e.g.
// 	n.Left = convlit(n.Left, t)
func convlit(n *Node, t *Type) *Node {
	return convlit1(n, t, false, noReuse)
}

// convlit1 converts n, if literal, to type t.
// It returns a new node if necessary.
// The result of convlit1 MUST be assigned back to n, e.g.
// 	n.Left = convlit1(n.Left, t, explicit, reuse)
func convlit1(n *Node, t *Type, explicit bool, reuse canReuseNode) *Node {
	if n == nil || t == nil || n.Type == nil || t.IsUntyped() || n.Type == t {
		return n
	}
	if !explicit && !n.Type.IsUntyped() {
		return n
	}

	if n.Op == OLITERAL && !reuse {
		// Can't always set n.Type directly on OLITERAL nodes.
		// See discussion on CL 20813.
		nn := *n
		n = &nn
		reuse = true
	}

	switch n.Op {
	default:
		if n.Type == idealbool {
			if t.IsBoolean() {
				n.Type = t
			} else {
				n.Type = Types[TBOOL]
			}
		}

		if n.Type.Etype == TIDEAL {
			n.Left = convlit(n.Left, t)
			n.Right = convlit(n.Right, t)
			n.Type = t
		}

		return n

		// target is invalid type for a constant?  leave alone.
	case OLITERAL:
		if !okforconst[t.Etype] && n.Type.Etype != TNIL {
			return defaultlitreuse(n, nil, reuse)
		}

	case OLSH, ORSH:
		n.Left = convlit1(n.Left, t, explicit && n.Left.Type.IsUntyped(), noReuse)
		t = n.Left.Type
		if t != nil && t.Etype == TIDEAL && n.Val().Ctype() != CTINT {
			n.SetVal(toint(n.Val()))
		}
		if t != nil && !t.IsInteger() {
			yyerror("invalid operation: %v (shift of type %v)", n, t)
			t = nil
		}

		n.Type = t
		return n

	case OCOMPLEX:
		if n.Type.Etype == TIDEAL {
			switch t.Etype {
			default:
				// If trying to convert to non-complex type,
				// leave as complex128 and let typechecker complain.
				t = Types[TCOMPLEX128]
				fallthrough
			case TCOMPLEX128:
				n.Type = t
				n.Left = convlit(n.Left, Types[TFLOAT64])
				n.Right = convlit(n.Right, Types[TFLOAT64])

			case TCOMPLEX64:
				n.Type = t
				n.Left = convlit(n.Left, Types[TFLOAT32])
				n.Right = convlit(n.Right, Types[TFLOAT32])
			}
		}

		return n
	}

	// avoided repeated calculations, errors
	if eqtype(n.Type, t) {
		return n
	}

	ct := consttype(n)
	var et EType
	if ct < 0 {
		goto bad
	}

	et = t.Etype
	if et == TINTER {
		if ct == CTNIL && n.Type == Types[TNIL] {
			n.Type = t
			return n
		}
		return defaultlitreuse(n, nil, reuse)
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
			return n

		case TARRAY:
			goto bad

		case TPTR32,
			TPTR64,
			TINTER,
			TMAP,
			TCHAN,
			TFUNC,
			TSLICE,
			TUNSAFEPTR:
			break

		// A nil literal may be converted to uintptr
		// if it is an unsafe.Pointer
		case TUINTPTR:
			if n.Type.Etype == TUNSAFEPTR {
				n.SetVal(Val{new(Mpint)})
				n.Val().U.(*Mpint).SetInt64(0)
			} else {
				goto bad
			}
		}

	case CTSTR, CTBOOL:
		if et != n.Type.Etype {
			goto bad
		}

	case CTINT, CTRUNE, CTFLT, CTCPLX:
		if n.Type.Etype == TUNSAFEPTR && t.Etype != TUINTPTR {
			goto bad
		}
		ct := n.Val().Ctype()
		if isInt[et] {
			switch ct {
			default:
				goto bad

			case CTCPLX, CTFLT, CTRUNE:
				n.SetVal(toint(n.Val()))
				fallthrough

			case CTINT:
				overflow(n.Val(), t)
			}
		} else if isFloat[et] {
			switch ct {
			default:
				goto bad

			case CTCPLX, CTINT, CTRUNE:
				n.SetVal(toflt(n.Val()))
				fallthrough

			case CTFLT:
				n.SetVal(Val{truncfltlit(n.Val().U.(*Mpflt), t)})
			}
		} else if isComplex[et] {
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
	return n

bad:
	if !n.Diag {
		if !t.Broke {
			yyerror("cannot convert %v to type %v", n, t)
		}
		n.Diag = true
	}

	if n.Type.IsUntyped() {
		n = defaultlitreuse(n, nil, reuse)
	}
	return n
}

func copyval(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		i := new(Mpint)
		i.Set(u)
		i.Rune = u.Rune
		v.U = i

	case *Mpflt:
		f := newMpflt()
		f.Set(u)
		v.U = f

	case *Mpcplx:
		c := new(Mpcplx)
		c.Real.Set(&u.Real)
		c.Imag.Set(&u.Imag)
		v.U = c
	}

	return v
}

func tocplx(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		c := new(Mpcplx)
		c.Real.SetInt(u)
		c.Imag.SetFloat64(0.0)
		v.U = c

	case *Mpflt:
		c := new(Mpcplx)
		c.Real.Set(u)
		c.Imag.SetFloat64(0.0)
		v.U = c
	}

	return v
}

func toflt(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		f := newMpflt()
		f.SetInt(u)
		v.U = f

	case *Mpcplx:
		f := newMpflt()
		f.Set(&u.Real)
		if u.Imag.CmpFloat64(0) != 0 {
			yyerror("constant %v%vi truncated to real", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp|FmtSign))
		}
		v.U = f
	}

	return v
}

func toint(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		if u.Rune {
			i := new(Mpint)
			i.Set(u)
			v.U = i
		}

	case *Mpflt:
		i := new(Mpint)
		if i.SetFloat(u) < 0 {
			msg := "constant %v truncated to integer"
			// provide better error message if SetFloat failed because f was too large
			if u.Val.IsInt() {
				msg = "constant %v overflows integer"
			}
			yyerror(msg, fconv(u, FmtSharp))
		}
		v.U = i

	case *Mpcplx:
		i := new(Mpint)
		if i.SetFloat(&u.Real) < 0 {
			yyerror("constant %v%vi truncated to integer", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp|FmtSign))
		}
		if u.Imag.CmpFloat64(0) != 0 {
			yyerror("constant %v%vi truncated to real", fconv(&u.Real, FmtSharp), fconv(&u.Imag, FmtSharp|FmtSign))
		}
		v.U = i
	}

	return v
}

func doesoverflow(v Val, t *Type) bool {
	switch u := v.U.(type) {
	case *Mpint:
		if !t.IsInteger() {
			Fatalf("overflow: %v integer constant", t)
		}
		return u.Cmp(minintval[t.Etype]) < 0 || u.Cmp(maxintval[t.Etype]) > 0

	case *Mpflt:
		if !t.IsFloat() {
			Fatalf("overflow: %v floating-point constant", t)
		}
		return u.Cmp(minfltval[t.Etype]) <= 0 || u.Cmp(maxfltval[t.Etype]) >= 0

	case *Mpcplx:
		if !t.IsComplex() {
			Fatalf("overflow: %v complex constant", t)
		}
		return u.Real.Cmp(minfltval[t.Etype]) <= 0 || u.Real.Cmp(maxfltval[t.Etype]) >= 0 ||
			u.Imag.Cmp(minfltval[t.Etype]) <= 0 || u.Imag.Cmp(maxfltval[t.Etype]) >= 0
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
		yyerror("constant %v overflows %v", v, t)
	}
}

func tostr(v Val) Val {
	switch u := v.U.(type) {
	case *Mpint:
		var i int64 = 0xFFFD
		if u.Cmp(minintval[TUINT32]) >= 0 && u.Cmp(maxintval[TUINT32]) <= 0 {
			i = u.Int64()
		}
		v.U = string(i)

	case *NilVal:
		// Can happen because of string([]byte(nil)).
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
		n1 := nod(OLITERAL, nil, nil)

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
		s := n.List.Slice()
		for i1 := 0; i1 < len(s); i1++ {
			if Isconst(s[i1], CTSTR) && i1+1 < len(s) && Isconst(s[i1+1], CTSTR) {
				// merge from i1 up to but not including i2
				var strs []string
				i2 := i1
				for i2 < len(s) && Isconst(s[i2], CTSTR) {
					strs = append(strs, s[i2].Val().U.(string))
					i2++
				}

				nl := *s[i1]
				nl.Orig = &nl
				nl.SetVal(Val{strings.Join(strs, "")})
				s[i1] = &nl
				s = append(s[:i1+1], s[i2:]...)
			}
		}

		if len(s) == 1 && Isconst(s[0], CTSTR) {
			n.Op = OLITERAL
			n.SetVal(s[0].Val())
		} else {
			n.List.Set(s)
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
	wl := nl.Type.Etype
	if isInt[wl] || isFloat[wl] || isComplex[wl] {
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
	var lno int32
	var wr EType
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
			if !n.Diag {
				yyerror("illegal constant expression %v %v", n.Op, nl.Type)
				n.Diag = true
			}
			return

		case OCONV_ | CTNIL_,
			OARRAYBYTESTR_ | CTNIL_:
			if n.Type.IsString() {
				v = tostr(v)
				nl.Type = n.Type
				break
			}
			fallthrough
		case OCONV_ | CTINT_,
			OCONV_ | CTRUNE_,
			OCONV_ | CTFLT_,
			OCONV_ | CTSTR_,
			OCONV_ | CTBOOL_:
			nl = convlit1(nl, n.Type, true, false)
			v = nl.Val()

		case OPLUS_ | CTINT_,
			OPLUS_ | CTRUNE_:
			break

		case OMINUS_ | CTINT_,
			OMINUS_ | CTRUNE_:
			v.U.(*Mpint).Neg()

		case OCOM_ | CTINT_,
			OCOM_ | CTRUNE_:
			var et EType = Txxx
			if nl.Type != nil {
				et = nl.Type.Etype
			}

			// calculate the mask in b
			// result will be (a ^ mask)
			var b Mpint
			switch et {
			// signed guys change sign
			default:
				b.SetInt64(-1)

				// unsigned guys invert their bits
			case TUINT8,
				TUINT16,
				TUINT32,
				TUINT64,
				TUINT,
				TUINTPTR:
				b.Set(maxintval[et])
			}

			v.U.(*Mpint).Xor(&b)

		case OPLUS_ | CTFLT_:
			break

		case OMINUS_ | CTFLT_:
			v.U.(*Mpflt).Neg()

		case OPLUS_ | CTCPLX_:
			break

		case OMINUS_ | CTCPLX_:
			v.U.(*Mpcplx).Real.Neg()
			v.U.(*Mpcplx).Imag.Neg()

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
	wr = nr.Type.Etype
	if isInt[wr] || isFloat[wr] || isComplex[wr] {
		wr = TIDEAL
	}

	// check for compatible general types (numeric, string, etc)
	if wl != wr {
		if wl == TINTER || wr == TINTER {
			goto setfalse
		}
		goto illegal
	}

	// check for compatible types.
	switch n.Op {
	// ideal const mixes with anything but otherwise must match.
	default:
		if nl.Type.Etype != TIDEAL {
			nr = defaultlit(nr, nl.Type)
			n.Right = nr
		}

		if nr.Type.Etype != TIDEAL {
			nl = defaultlit(nl, nr.Type)
			n.Left = nl
		}

		if nl.Type.Etype != nr.Type.Etype {
			goto illegal
		}

	// right must be unsigned.
	// left can be ideal.
	case OLSH, ORSH:
		nr = defaultlit(nr, Types[TUINT])

		n.Right = nr
		if nr.Type != nil && (nr.Type.IsSigned() || !nr.Type.IsInteger()) {
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
		i.Set(rv.U.(*Mpint))
		i.Rune = true
		rv.U = i
	}
	if v.Ctype() == CTINT && rv.Ctype() == CTRUNE {
		if n.Op == OLSH || n.Op == ORSH {
			i := new(Mpint)
			i.Set(rv.U.(*Mpint))
			rv.U = i
		} else {
			i := new(Mpint)
			i.Set(v.U.(*Mpint))
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
		v.U.(*Mpint).Add(rv.U.(*Mpint))

	case OSUB_ | CTINT_,
		OSUB_ | CTRUNE_:
		v.U.(*Mpint).Sub(rv.U.(*Mpint))

	case OMUL_ | CTINT_,
		OMUL_ | CTRUNE_:
		v.U.(*Mpint).Mul(rv.U.(*Mpint))

	case ODIV_ | CTINT_,
		ODIV_ | CTRUNE_:
		if rv.U.(*Mpint).CmpInt64(0) == 0 {
			yyerror("division by zero")
			v.U.(*Mpint).SetOverflow()
			break
		}

		v.U.(*Mpint).Quo(rv.U.(*Mpint))

	case OMOD_ | CTINT_,
		OMOD_ | CTRUNE_:
		if rv.U.(*Mpint).CmpInt64(0) == 0 {
			yyerror("division by zero")
			v.U.(*Mpint).SetOverflow()
			break
		}

		v.U.(*Mpint).Rem(rv.U.(*Mpint))

	case OLSH_ | CTINT_,
		OLSH_ | CTRUNE_:
		v.U.(*Mpint).Lsh(rv.U.(*Mpint))

	case ORSH_ | CTINT_,
		ORSH_ | CTRUNE_:
		v.U.(*Mpint).Rsh(rv.U.(*Mpint))

	case OOR_ | CTINT_,
		OOR_ | CTRUNE_:
		v.U.(*Mpint).Or(rv.U.(*Mpint))

	case OAND_ | CTINT_,
		OAND_ | CTRUNE_:
		v.U.(*Mpint).And(rv.U.(*Mpint))

	case OANDNOT_ | CTINT_,
		OANDNOT_ | CTRUNE_:
		v.U.(*Mpint).AndNot(rv.U.(*Mpint))

	case OXOR_ | CTINT_,
		OXOR_ | CTRUNE_:
		v.U.(*Mpint).Xor(rv.U.(*Mpint))

	case OADD_ | CTFLT_:
		v.U.(*Mpflt).Add(rv.U.(*Mpflt))

	case OSUB_ | CTFLT_:
		v.U.(*Mpflt).Sub(rv.U.(*Mpflt))

	case OMUL_ | CTFLT_:
		v.U.(*Mpflt).Mul(rv.U.(*Mpflt))

	case ODIV_ | CTFLT_:
		if rv.U.(*Mpflt).CmpFloat64(0) == 0 {
			yyerror("division by zero")
			v.U.(*Mpflt).SetFloat64(1.0)
			break
		}

		v.U.(*Mpflt).Quo(rv.U.(*Mpflt))

	// The default case above would print 'ideal % ideal',
	// which is not quite an ideal error.
	case OMOD_ | CTFLT_:
		if !n.Diag {
			yyerror("illegal constant expression: floating-point %% operation")
			n.Diag = true
		}

		return

	case OADD_ | CTCPLX_:
		v.U.(*Mpcplx).Real.Add(&rv.U.(*Mpcplx).Real)
		v.U.(*Mpcplx).Imag.Add(&rv.U.(*Mpcplx).Imag)

	case OSUB_ | CTCPLX_:
		v.U.(*Mpcplx).Real.Sub(&rv.U.(*Mpcplx).Real)
		v.U.(*Mpcplx).Imag.Sub(&rv.U.(*Mpcplx).Imag)

	case OMUL_ | CTCPLX_:
		cmplxmpy(v.U.(*Mpcplx), rv.U.(*Mpcplx))

	case ODIV_ | CTCPLX_:
		if rv.U.(*Mpcplx).Real.CmpFloat64(0) == 0 && rv.U.(*Mpcplx).Imag.CmpFloat64(0) == 0 {
			yyerror("complex division by zero")
			rv.U.(*Mpcplx).Real.SetFloat64(1.0)
			rv.U.(*Mpcplx).Imag.SetFloat64(0.0)
			break
		}

		cmplxdiv(v.U.(*Mpcplx), rv.U.(*Mpcplx))

	case OEQ_ | CTNIL_:
		goto settrue

	case ONE_ | CTNIL_:
		goto setfalse

	case OEQ_ | CTINT_,
		OEQ_ | CTRUNE_:
		if v.U.(*Mpint).Cmp(rv.U.(*Mpint)) == 0 {
			goto settrue
		}
		goto setfalse

	case ONE_ | CTINT_,
		ONE_ | CTRUNE_:
		if v.U.(*Mpint).Cmp(rv.U.(*Mpint)) != 0 {
			goto settrue
		}
		goto setfalse

	case OLT_ | CTINT_,
		OLT_ | CTRUNE_:
		if v.U.(*Mpint).Cmp(rv.U.(*Mpint)) < 0 {
			goto settrue
		}
		goto setfalse

	case OLE_ | CTINT_,
		OLE_ | CTRUNE_:
		if v.U.(*Mpint).Cmp(rv.U.(*Mpint)) <= 0 {
			goto settrue
		}
		goto setfalse

	case OGE_ | CTINT_,
		OGE_ | CTRUNE_:
		if v.U.(*Mpint).Cmp(rv.U.(*Mpint)) >= 0 {
			goto settrue
		}
		goto setfalse

	case OGT_ | CTINT_,
		OGT_ | CTRUNE_:
		if v.U.(*Mpint).Cmp(rv.U.(*Mpint)) > 0 {
			goto settrue
		}
		goto setfalse

	case OEQ_ | CTFLT_:
		if v.U.(*Mpflt).Cmp(rv.U.(*Mpflt)) == 0 {
			goto settrue
		}
		goto setfalse

	case ONE_ | CTFLT_:
		if v.U.(*Mpflt).Cmp(rv.U.(*Mpflt)) != 0 {
			goto settrue
		}
		goto setfalse

	case OLT_ | CTFLT_:
		if v.U.(*Mpflt).Cmp(rv.U.(*Mpflt)) < 0 {
			goto settrue
		}
		goto setfalse

	case OLE_ | CTFLT_:
		if v.U.(*Mpflt).Cmp(rv.U.(*Mpflt)) <= 0 {
			goto settrue
		}
		goto setfalse

	case OGE_ | CTFLT_:
		if v.U.(*Mpflt).Cmp(rv.U.(*Mpflt)) >= 0 {
			goto settrue
		}
		goto setfalse

	case OGT_ | CTFLT_:
		if v.U.(*Mpflt).Cmp(rv.U.(*Mpflt)) > 0 {
			goto settrue
		}
		goto setfalse

	case OEQ_ | CTCPLX_:
		if v.U.(*Mpcplx).Real.Cmp(&rv.U.(*Mpcplx).Real) == 0 && v.U.(*Mpcplx).Imag.Cmp(&rv.U.(*Mpcplx).Imag) == 0 {
			goto settrue
		}
		goto setfalse

	case ONE_ | CTCPLX_:
		if v.U.(*Mpcplx).Real.Cmp(&rv.U.(*Mpcplx).Real) != 0 || v.U.(*Mpcplx).Imag.Cmp(&rv.U.(*Mpcplx).Imag) != 0 {
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
	lno = setlineno(n)
	overflow(v, n.Type)
	lineno = lno

	// truncate precision for non-ideal float.
	if v.Ctype() == CTFLT && n.Type.Etype != TIDEAL {
		n.SetVal(Val{truncfltlit(v.U.(*Mpflt), n.Type)})
	}
	return

settrue:
	nn = nodbool(true)
	nn.Orig = saveorig(n)
	if !iscmp[n.Op] {
		nn.Type = nl.Type
	}
	*n = *nn
	return

setfalse:
	nn = nodbool(false)
	nn.Orig = saveorig(n)
	if !iscmp[n.Op] {
		nn.Type = nl.Type
	}
	*n = *nn
	return

illegal:
	if !n.Diag {
		yyerror("illegal constant expression: %v %v %v", nl.Type, n.Op, nr.Type)
		n.Diag = true
	}
}

func nodlit(v Val) *Node {
	n := nod(OLITERAL, nil, nil)
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
	n := nod(OLITERAL, nil, nil)
	n.Type = Types[TIDEAL]
	n.SetVal(Val{c})

	if r.Ctype() != CTFLT || i.Ctype() != CTFLT {
		Fatalf("nodcplxlit ctype %d/%d", r.Ctype(), i.Ctype())
	}

	c.Real.Set(r.U.(*Mpflt))
	c.Imag.Set(i.U.(*Mpflt))
	return n
}

// idealkind returns a constant kind like consttype
// but for an arbitrary "ideal" (untyped constant) expression.
func idealkind(n *Node) Ctype {
	if n == nil || !n.Type.IsUntyped() {
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

// The result of defaultlit MUST be assigned back to n, e.g.
// 	n.Left = defaultlit(n.Left, t)
func defaultlit(n *Node, t *Type) *Node {
	return defaultlitreuse(n, t, noReuse)
}

// The result of defaultlitreuse MUST be assigned back to n, e.g.
// 	n.Left = defaultlitreuse(n.Left, t, reuse)
func defaultlitreuse(n *Node, t *Type, reuse canReuseNode) *Node {
	if n == nil || !n.Type.IsUntyped() {
		return n
	}

	if n.Op == OLITERAL && !reuse {
		nn := *n
		n = &nn
		reuse = true
	}

	lno := setlineno(n)
	ctype := idealkind(n)
	var t1 *Type
	switch ctype {
	default:
		if t != nil {
			return convlit(n, t)
		}

		if n.Val().Ctype() == CTNIL {
			lineno = lno
			if !n.Diag {
				yyerror("use of untyped nil")
				n.Diag = true
			}

			n.Type = nil
			break
		}

		if n.Val().Ctype() == CTSTR {
			t1 := Types[TSTRING]
			n = convlit1(n, t1, false, reuse)
			break
		}

		yyerror("defaultlit: unknown literal: %v", n)

	case CTxxx:
		Fatalf("defaultlit: idealkind is CTxxx: %+v", n)

	case CTBOOL:
		t1 := Types[TBOOL]
		if t != nil && t.IsBoolean() {
			t1 = t
		}
		n = convlit1(n, t1, false, reuse)

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

	lineno = lno
	return n

num:
	// Note: n.Val().Ctype() can be CTxxx (not a constant) here
	// in the case of an untyped non-constant value, like 1<<i.
	v1 := n.Val()
	if t != nil {
		if t.IsInteger() {
			t1 = t
			v1 = toint(n.Val())
		} else if t.IsFloat() {
			t1 = t
			v1 = toflt(n.Val())
		} else if t.IsComplex() {
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
	n = convlit1(n, t1, false, reuse)
	lineno = lno
	return n
}

// defaultlit on both nodes simultaneously;
// if they're both ideal going in they better
// get the same type going out.
// force means must assign concrete (non-ideal) type.
// The results of defaultlit2 MUST be assigned back to l and r, e.g.
// 	n.Left, n.Right = defaultlit2(n.Left, n.Right, force)
func defaultlit2(l *Node, r *Node, force bool) (*Node, *Node) {
	if l.Type == nil || r.Type == nil {
		return l, r
	}
	if !l.Type.IsUntyped() {
		r = convlit(r, l.Type)
		return l, r
	}

	if !r.Type.IsUntyped() {
		l = convlit(l, r.Type)
		return l, r
	}

	if !force {
		return l, r
	}

	if l.Type.IsBoolean() {
		l = convlit(l, Types[TBOOL])
		r = convlit(r, Types[TBOOL])
	}

	lkind := idealkind(l)
	rkind := idealkind(r)
	if lkind == CTCPLX || rkind == CTCPLX {
		l = convlit(l, Types[TCOMPLEX128])
		r = convlit(r, Types[TCOMPLEX128])
		return l, r
	}

	if lkind == CTFLT || rkind == CTFLT {
		l = convlit(l, Types[TFLOAT64])
		r = convlit(r, Types[TFLOAT64])
		return l, r
	}

	if lkind == CTRUNE || rkind == CTRUNE {
		l = convlit(l, runetype)
		r = convlit(r, runetype)
		return l, r
	}

	l = convlit(l, Types[TINT])
	r = convlit(r, Types[TINT])

	return l, r
}

// strlit returns the value of a literal string Node as a string.
func strlit(n *Node) string {
	return n.Val().U.(string)
}

func smallintconst(n *Node) bool {
	if n.Op == OLITERAL && Isconst(n, CTINT) && n.Type != nil {
		switch simtype[n.Type.Etype] {
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
			v, ok := n.Val().U.(*Mpint)
			if ok && v.Cmp(minintval[TINT32]) > 0 && v.Cmp(maxintval[TINT32]) < 0 {
				return true
			}
		}
	}

	return false
}

// nonnegintconst checks if Node n contains a constant expression
// representable as a non-negative small integer, and returns its
// (integer) value if that's the case. Otherwise, it returns -1.
func nonnegintconst(n *Node) int64 {
	if n.Op != OLITERAL {
		return -1
	}

	// toint will leave n.Val unchanged if it's not castable to an
	// Mpint, so we still have to guard the conversion.
	v := toint(n.Val())
	vi, ok := v.U.(*Mpint)
	if !ok || vi.Val.Sign() < 0 || vi.Cmp(maxintval[TINT32]) > 0 {
		return -1
	}

	return vi.Int64()
}

// complex multiply v *= rv
//	(a, b) * (c, d) = (a*c - b*d, b*c + a*d)
func cmplxmpy(v *Mpcplx, rv *Mpcplx) {
	var ac Mpflt
	var bd Mpflt
	var bc Mpflt
	var ad Mpflt

	ac.Set(&v.Real)
	ac.Mul(&rv.Real) // ac

	bd.Set(&v.Imag)

	bd.Mul(&rv.Imag) // bd

	bc.Set(&v.Imag)

	bc.Mul(&rv.Real) // bc

	ad.Set(&v.Real)

	ad.Mul(&rv.Imag) // ad

	v.Real.Set(&ac)

	v.Real.Sub(&bd) // ac-bd

	v.Imag.Set(&bc)

	v.Imag.Add(&ad) // bc+ad
}

// complex divide v /= rv
//	(a, b) / (c, d) = ((a*c + b*d), (b*c - a*d))/(c*c + d*d)
func cmplxdiv(v *Mpcplx, rv *Mpcplx) {
	var ac Mpflt
	var bd Mpflt
	var bc Mpflt
	var ad Mpflt
	var cc_plus_dd Mpflt

	cc_plus_dd.Set(&rv.Real)
	cc_plus_dd.Mul(&rv.Real) // cc

	ac.Set(&rv.Imag)

	ac.Mul(&rv.Imag) // dd

	cc_plus_dd.Add(&ac) // cc+dd

	ac.Set(&v.Real)

	ac.Mul(&rv.Real) // ac

	bd.Set(&v.Imag)

	bd.Mul(&rv.Imag) // bd

	bc.Set(&v.Imag)

	bc.Mul(&rv.Real) // bc

	ad.Set(&v.Real)

	ad.Mul(&rv.Imag) // ad

	v.Real.Set(&ac)

	v.Real.Add(&bd)         // ac+bd
	v.Real.Quo(&cc_plus_dd) // (ac+bd)/(cc+dd)

	v.Imag.Set(&bc)

	v.Imag.Sub(&ad)         // bc-ad
	v.Imag.Quo(&cc_plus_dd) // (bc+ad)/(cc+dd)
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

		if t != nil && t.IsPtr() {
			t = t.Elem()
		}
		if t != nil && t.IsArray() && !hascallchan(l) {
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

	case OALIGNOF, OOFFSETOF, OSIZEOF:
		return true
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
	for _, n1 := range n.List.Slice() {
		if hascallchan(n1) {
			return true
		}
	}
	for _, n2 := range n.Rlist.Slice() {
		if hascallchan(n2) {
			return true
		}
	}

	return false
}
