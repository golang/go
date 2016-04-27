// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

// unsafenmagic rewrites calls to package unsafe's functions into constants.
func unsafenmagic(nn *Node) *Node {
	fn := nn.Left
	args := nn.List

	if safemode || fn == nil || fn.Op != ONAME {
		return nil
	}
	s := fn.Sym
	if s == nil {
		return nil
	}
	if s.Pkg != unsafepkg {
		return nil
	}

	if args.Len() == 0 {
		Yyerror("missing argument for %v", s)
		return nil
	}

	r := args.First()

	var v int64
	switch s.Name {
	case "Alignof", "Sizeof":
		r = typecheck(r, Erv)
		r = defaultlit(r, nil)
		tr := r.Type
		if tr == nil {
			goto bad
		}
		dowidth(tr)
		if s.Name == "Alignof" {
			v = int64(tr.Align)
		} else {
			v = tr.Width
		}

	case "Offsetof":
		// must be a selector.
		if r.Op != OXDOT {
			goto bad
		}

		// Remember base of selector to find it back after dot insertion.
		// Since r->left may be mutated by typechecking, check it explicitly
		// first to track it correctly.
		r.Left = typecheck(r.Left, Erv)
		base := r.Left

		r = typecheck(r, Erv)
		switch r.Op {
		case ODOT, ODOTPTR:
			break
		case OCALLPART:
			Yyerror("invalid expression %v: argument is a method value", nn)
			goto ret
		default:
			goto bad
		}

		// Sum offsets for dots until we reach base.
		for r1 := r; r1 != base; r1 = r1.Left {
			switch r1.Op {
			case ODOTPTR:
				// For Offsetof(s.f), s may itself be a pointer,
				// but accessing f must not otherwise involve
				// indirection via embedded pointer types.
				if r1.Left != base {
					Yyerror("invalid expression %v: selector implies indirection of embedded %v", nn, r1.Left)
					goto ret
				}
				fallthrough
			case ODOT:
				v += r1.Xoffset
			default:
				Dump("unsafenmagic", r)
				Fatalf("impossible %#v node after dot insertion", r1.Op)
				goto bad
			}
		}

	default:
		return nil
	}

	if args.Len() > 1 {
		Yyerror("extra arguments for %v", s)
	}
	goto ret

bad:
	Yyerror("invalid expression %v", nn)

ret:
	// any side effects disappear; ignore init
	var val Val
	val.U = new(Mpint)
	val.U.(*Mpint).SetInt64(v)
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
