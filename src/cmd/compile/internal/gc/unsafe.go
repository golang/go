// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "cmd/internal/obj"

// look for
//	unsafe.Sizeof
//	unsafe.Offsetof
//	unsafe.Alignof
// rewrite with a constant
func unsafenmagic(nn *Node) *Node {
	fn := nn.Left
	args := nn.List

	if safemode != 0 || fn == nil || fn.Op != ONAME {
		return nil
	}
	s := fn.Sym
	if s == nil {
		return nil
	}
	if s.Pkg != unsafepkg {
		return nil
	}

	if args == nil {
		Yyerror("missing argument for %v", s)
		return nil
	}

	r := args.N

	var v int64
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
		case ODOT, ODOTPTR:
			break

		case OCALLPART:
			Yyerror("invalid expression %v: argument is a method value", nn)
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
				Yyerror("invalid expression %v: selector implies indirection of embedded %v", nn, r1.Left)
				goto ret

			default:
				Dump("unsafenmagic", r)
				Fatalf("impossible %v node after dot insertion", Oconv(int(r1.Op), obj.FmtSharp))
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

	return nil

bad:
	Yyerror("invalid expression %v", nn)
	v = 0
	goto ret

yes:
	if args.Next != nil {
		Yyerror("extra arguments for %v", s)
	}

	// any side effects disappear; ignore init
ret:
	var val Val
	val.U = new(Mpint)
	Mpmovecfix(val.U.(*Mpint), v)
	n := Nod(OLITERAL, nil, nil)
	n.Orig = nn
	n.SetVal(val)
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
