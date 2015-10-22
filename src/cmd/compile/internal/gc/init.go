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
	return Lookupf("init.%d", renameinit_initgen)
}

// hand-craft the following initialization code
//	var initdone· uint8 				(1)
//	func init()					(2)
//		if initdone· != 0 {			(3)
//			if initdone· == 2		(4)
//				return
//			throw();			(5)
//		}
//		initdone· = 1;				(6)
//		// over all matching imported symbols
//			<pkg>.init()			(7)
//		{ <init stmts> }			(8)
//		init.<n>() // if any			(9)
//		initdone· = 2;				(10)
//		return					(11)
//	}
func anyinit(n *NodeList) bool {
	// are there any interesting init statements
	for l := n; l != nil; l = l.Next {
		switch l.N.Op {
		case ODCLFUNC, ODCLCONST, ODCLTYPE, OEMPTY:
			break

		case OAS, OASWB:
			if isblank(l.N.Left) && candiscard(l.N.Right) {
				break
			}
			fallthrough

			// fall through
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

func fninit(n *NodeList) {
	if Debug['A'] != 0 {
		// sys.go or unsafe.go during compiler build
		return
	}

	n = initfix(n)
	if !anyinit(n) {
		return
	}

	var r *NodeList

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

	a.Left = Nod(ONE, gatevar, Nodintconst(0))
	r = list(r, a)

	// (4)
	b := Nod(OIF, nil, nil)

	b.Left = Nod(OEQ, gatevar, Nodintconst(2))
	b.Nbody = list1(Nod(ORETURN, nil, nil))
	a.Nbody = list1(b)

	// (5)
	b = syslook("throwinit", 0)

	b = Nod(OCALL, b, nil)
	a.Nbody = list(a.Nbody, b)

	// (6)
	a = Nod(OAS, gatevar, Nodintconst(1))

	r = list(r, a)

	// (7)
	for _, s := range initSyms {
		if s.Def != nil && s != initsym {
			// could check that it is fn of no args/returns
			a = Nod(OCALL, s.Def, nil)
			r = list(r, a)
		}
	}

	// (8)
	r = concat(r, n)

	// (9)
	// could check that it is fn of no args/returns
	for i := 1; ; i++ {
		s := Lookupf("init.%d", i)
		if s.Def == nil {
			break
		}
		a = Nod(OCALL, s.Def, nil)
		r = list(r, a)
	}

	// (10)
	a = Nod(OAS, gatevar, Nodintconst(2))

	r = list(r, a)

	// (11)
	a = Nod(ORETURN, nil, nil)

	r = list(r, a)
	exportsym(fn.Func.Nname)

	fn.Nbody = r
	funcbody(fn)

	Curfn = fn
	typecheck(&fn, Etop)
	typechecklist(r, Etop)
	Curfn = nil
	funccompile(fn)
}
