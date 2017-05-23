// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

//	case OADD:
//		if(n->right->op == OLITERAL) {
//			v = n->right->vconst;
//			naddr(n->left, a, canemitcode);
//		} else
//		if(n->left->op == OLITERAL) {
//			v = n->left->vconst;
//			naddr(n->right, a, canemitcode);
//		} else
//			goto bad;
//		a->offset += v;
//		break;

// a function named init is a special case.
// it is called by the initialization before
// main is run. to make it unique within a
// package and also uncallable, the name,
// normally "pkg.init", is altered to "pkg.init.1".

var renameinit_initgen int

func renameinit() *Sym {
	renameinit_initgen++
	return LookupN("init.", renameinit_initgen)
}

// hand-craft the following initialization code
//      var initdone· uint8                             (1)
//      func init() {                                   (2)
//              if initdone· > 1 {                      (3)
//                      return                          (3a)
//              }
//              if initdone· == 1 {                     (4)
//                      throw()                         (4a)
//              }
//              initdone· = 1                           (5)
//              // over all matching imported symbols
//                      <pkg>.init()                    (6)
//              { <init stmts> }                        (7)
//              init.<n>() // if any                    (8)
//              initdone· = 2                           (9)
//              return                                  (10)
//      }
func anyinit(n []*Node) bool {
	// are there any interesting init statements
	for _, ln := range n {
		switch ln.Op {
		case ODCLFUNC, ODCLCONST, ODCLTYPE, OEMPTY:
			break

		case OAS, OASWB:
			if isblank(ln.Left) && candiscard(ln.Right) {
				break
			}
			fallthrough
		default:
			return true
		}
	}

	// is this main
	if localpkg.Name == "main" {
		return true
	}

	// is there an explicit init function
	s := Lookup("init.1")

	if s.Def != nil {
		return true
	}

	// are there any imported init functions
	for _, s := range initSyms {
		if s.Def != nil {
			return true
		}
	}

	// then none
	return false
}

func fninit(n []*Node) {
	if Debug['A'] != 0 {
		// sys.go or unsafe.go during compiler build
		return
	}

	nf := initfix(n)
	if !anyinit(nf) {
		return
	}

	var r []*Node

	// (1)
	gatevar := newname(Lookup("initdone·"))
	addvar(gatevar, Types[TUINT8], PEXTERN)

	// (2)
	Maxarg = 0

	fn := Nod(ODCLFUNC, nil, nil)
	initsym := Lookup("init")
	fn.Func.Nname = newname(initsym)
	fn.Func.Nname.Name.Defn = fn
	fn.Func.Nname.Name.Param.Ntype = Nod(OTFUNC, nil, nil)
	declare(fn.Func.Nname, PFUNC)
	funchdr(fn)

	// (3)
	a := Nod(OIF, nil, nil)
	a.Left = Nod(OGT, gatevar, Nodintconst(1))
	a.Likely = 1
	r = append(r, a)
	// (3a)
	a.Nbody.Set1(Nod(ORETURN, nil, nil))

	// (4)
	b := Nod(OIF, nil, nil)
	b.Left = Nod(OEQ, gatevar, Nodintconst(1))
	// this actually isn't likely, but code layout is better
	// like this: no JMP needed after the call.
	b.Likely = 1
	r = append(r, b)
	// (4a)
	b.Nbody.Set1(Nod(OCALL, syslook("throwinit"), nil))

	// (5)
	a = Nod(OAS, gatevar, Nodintconst(1))

	r = append(r, a)

	// (6)
	for _, s := range initSyms {
		if s.Def != nil && s != initsym {
			// could check that it is fn of no args/returns
			a = Nod(OCALL, s.Def, nil)
			r = append(r, a)
		}
	}

	// (7)
	r = append(r, nf...)

	// (8)
	// could check that it is fn of no args/returns
	for i := 1; ; i++ {
		s := LookupN("init.", i)
		if s.Def == nil {
			break
		}
		a = Nod(OCALL, s.Def, nil)
		r = append(r, a)
	}

	// (9)
	a = Nod(OAS, gatevar, Nodintconst(2))

	r = append(r, a)

	// (10)
	a = Nod(ORETURN, nil, nil)

	r = append(r, a)
	exportsym(fn.Func.Nname)

	fn.Nbody.Set(r)
	funcbody(fn)

	Curfn = fn
	fn = typecheck(fn, Etop)
	typecheckslice(r, Etop)
	Curfn = nil
	funccompile(fn)
}
