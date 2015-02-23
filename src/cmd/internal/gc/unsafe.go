// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/internal/obj"

/*
 * look for
 *	unsafe.Sizeof
 *	unsafe.Offsetof
 *	unsafe.Alignof
 * rewrite with a constant
 */
func unsafenmagic(nn *Node) *Node {
	var r *Node
	var s *Sym
	var v int64

	fn := nn.Left
	args := nn.List

	if safemode != 0 || fn == nil || fn.Op != ONAME {
		goto no
	}
	s = fn.Sym
	if s == nil {
		goto no
	}
	if s.Pkg != unsafepkg {
		goto no
	}

	if args == nil {
		Yyerror("missing argument for %v", Sconv(s, 0))
		goto no
	}

	r = args.N

	if s.Name == "Sizeof" {
		typecheck(&r, Erv)
		defaultlit(&r, nil)
		tr := r.Type
		if tr == nil {
			goto bad
		}
		dowidth(tr)
		v = tr.Width
		goto yes
	}

	if s.Name == "Offsetof" {
		// must be a selector.
		if r.Op != OXDOT {
			goto bad
		}

		// Remember base of selector to find it back after dot insertion.
		// Since r->left may be mutated by typechecking, check it explicitly
		// first to track it correctly.
		typecheck(&r.Left, Erv)

		base := r.Left
		typecheck(&r, Erv)
		switch r.Op {
		case ODOT,
			ODOTPTR:
			break

		case OCALLPART:
			Yyerror("invalid expression %v: argument is a method value", Nconv(nn, 0))
			v = 0
			goto ret

		default:
			goto bad
		}

		v = 0

		// add offsets for inserted dots.
		var r1 *Node
		for r1 = r; r1.Left != base; r1 = r1.Left {
			switch r1.Op {
			case ODOT:
				v += r1.Xoffset

			case ODOTPTR:
				Yyerror("invalid expression %v: selector implies indirection of embedded %v", Nconv(nn, 0), Nconv(r1.Left, 0))
				goto ret

			default:
				Dump("unsafenmagic", r)
				Fatal("impossible %v node after dot insertion", Oconv(int(r1.Op), obj.FmtSharp))
				goto bad
			}
		}

		v += r1.Xoffset
		goto yes
	}

	if s.Name == "Alignof" {
		typecheck(&r, Erv)
		defaultlit(&r, nil)
		tr := r.Type
		if tr == nil {
			goto bad
		}

		// make struct { byte; T; }
		t := typ(TSTRUCT)

		t.Type = typ(TFIELD)
		t.Type.Type = Types[TUINT8]
		t.Type.Down = typ(TFIELD)
		t.Type.Down.Type = tr

		// compute struct widths
		dowidth(t)

		// the offset of T is its required alignment
		v = t.Type.Down.Width

		goto yes
	}

no:
	return nil

bad:
	Yyerror("invalid expression %v", Nconv(nn, 0))
	v = 0
	goto ret

yes:
	if args.Next != nil {
		Yyerror("extra arguments for %v", Sconv(s, 0))
	}

	// any side effects disappear; ignore init
ret:
	var val Val
	val.Ctype = CTINT

	val.U.Xval = new(Mpint)
	Mpmovecfix(val.U.Xval, v)
	n := Nod(OLITERAL, nil, nil)
	n.Orig = nn
	n.Val = val
	n.Type = Types[TUINTPTR]
	nn.Type = Types[TUINTPTR]
	return n
}

func isunsafebuiltin(n *Node) bool {
	if n == nil || n.Op != ONAME || n.Sym == nil || n.Sym.Pkg != unsafepkg {
		return false
	}
	if n.Sym.Name == "Sizeof" {
		return true
	}
	if n.Sym.Name == "Offsetof" {
		return true
	}
	if n.Sym.Name == "Alignof" {
		return true
	}
	return false
}
