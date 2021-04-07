// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file will evolve, since we plan to do a mix of stenciling and passing
// around dictionaries.

package noder

import (
	"bytes"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"strings"
)

// For catching problems as we add more features
// TODO(danscales): remove assertions or replace with base.FatalfAt()
func assert(p bool) {
	if !p {
		panic("assertion failed")
	}
}

// stencil scans functions for instantiated generic function calls and creates the
// required instantiations for simple generic functions. It also creates
// instantiated methods for all fully-instantiated generic types that have been
// encountered already or new ones that are encountered during the stenciling
// process.
func (g *irgen) stencil() {
	g.target.Stencils = make(map[*types.Sym]*ir.Func)

	// Instantiate the methods of instantiated generic types that we have seen so far.
	g.instantiateMethods()

	// Don't use range(g.target.Decls) - we also want to process any new instantiated
	// functions that are created during this loop, in order to handle generic
	// functions calling other generic functions.
	for i := 0; i < len(g.target.Decls); i++ {
		decl := g.target.Decls[i]

		// Look for function instantiations in bodies of non-generic
		// functions or in global assignments (ignore global type and
		// constant declarations).
		switch decl.Op() {
		case ir.ODCLFUNC:
			if decl.Type().HasTParam() {
				// Skip any generic functions
				continue
			}

		case ir.OAS:

		case ir.OAS2:

		default:
			continue
		}

		// For all non-generic code, search for any function calls using
		// generic function instantiations. Then create the needed
		// instantiated function if it hasn't been created yet, and change
		// to calling that function directly.
		modified := false
		foundFuncInst := false
		ir.Visit(decl, func(n ir.Node) {
			if n.Op() == ir.OFUNCINST {
				// We found a function instantiation that is not
				// immediately called.
				foundFuncInst = true
			}
			if n.Op() != ir.OCALL || n.(*ir.CallExpr).X.Op() != ir.OFUNCINST {
				return
			}
			// We have found a function call using a generic function
			// instantiation.
			call := n.(*ir.CallExpr)
			inst := call.X.(*ir.InstExpr)
			st := g.getInstantiationForNode(inst)
			// Replace the OFUNCINST with a direct reference to the
			// new stenciled function
			call.X = st.Nname
			if inst.X.Op() == ir.OCALLPART {
				// When we create an instantiation of a method
				// call, we make it a function. So, move the
				// receiver to be the first arg of the function
				// call.
				withRecv := make([]ir.Node, len(call.Args)+1)
				dot := inst.X.(*ir.SelectorExpr)
				withRecv[0] = dot.X
				copy(withRecv[1:], call.Args)
				call.Args = withRecv
			}
			// Transform the Call now, which changes OCALL
			// to OCALLFUNC and does typecheckaste/assignconvfn.
			transformCall(call)
			modified = true
		})

		// If we found an OFUNCINST without a corresponding call in the
		// above decl, then traverse the nodes of decl again (with
		// EditChildren rather than Visit), where we actually change the
		// OFUNCINST node to an ONAME for the instantiated function.
		// EditChildren is more expensive than Visit, so we only do this
		// in the infrequent case of an OFUNCINSt without a corresponding
		// call.
		if foundFuncInst {
			var edit func(ir.Node) ir.Node
			edit = func(x ir.Node) ir.Node {
				if x.Op() == ir.OFUNCINST {
					st := g.getInstantiationForNode(x.(*ir.InstExpr))
					return st.Nname
				}
				ir.EditChildren(x, edit)
				return x
			}
			edit(decl)
		}
		if base.Flag.W > 1 && modified {
			ir.Dump(fmt.Sprintf("\nmodified %v", decl), decl)
		}
		// We may have seen new fully-instantiated generic types while
		// instantiating any needed functions/methods in the above
		// function. If so, instantiate all the methods of those types
		// (which will then lead to more function/methods to scan in the loop).
		g.instantiateMethods()
	}

}

// instantiateMethods instantiates all the methods of all fully-instantiated
// generic types that have been added to g.instTypeList.
func (g *irgen) instantiateMethods() {
	for i := 0; i < len(g.instTypeList); i++ {
		typ := g.instTypeList[i]
		// Get the base generic type by looking up the symbol of the
		// generic (uninstantiated) name.
		baseSym := typ.Sym().Pkg.Lookup(genericTypeName(typ.Sym()))
		baseType := baseSym.Def.(*ir.Name).Type()
		for j, m := range typ.Methods().Slice() {
			name := m.Nname.(*ir.Name)
			targs := make([]ir.Node, len(typ.RParams()))
			for k, targ := range typ.RParams() {
				targs[k] = ir.TypeNode(targ)
			}
			baseNname := baseType.Methods().Slice()[j].Nname.(*ir.Name)
			name.Func = g.getInstantiation(baseNname, targs, true)
		}
	}
	g.instTypeList = nil

}

// genericSym returns the name of the base generic type for the type named by
// sym. It simply returns the name obtained by removing everything after the
// first bracket ("[").
func genericTypeName(sym *types.Sym) string {
	return sym.Name[0:strings.Index(sym.Name, "[")]
}

// getInstantiationForNode returns the function/method instantiation for a
// InstExpr node inst.
func (g *irgen) getInstantiationForNode(inst *ir.InstExpr) *ir.Func {
	if meth, ok := inst.X.(*ir.SelectorExpr); ok {
		return g.getInstantiation(meth.Selection.Nname.(*ir.Name), inst.Targs, true)
	} else {
		return g.getInstantiation(inst.X.(*ir.Name), inst.Targs, false)
	}
}

// getInstantiation gets the instantiantion of the function or method nameNode
// with the type arguments targs. If the instantiated function is not already
// cached, then it calls genericSubst to create the new instantiation.
func (g *irgen) getInstantiation(nameNode *ir.Name, targs []ir.Node, isMeth bool) *ir.Func {
	sym := makeInstName(nameNode.Sym(), targs, isMeth)
	st := g.target.Stencils[sym]
	if st == nil {
		// If instantiation doesn't exist yet, create it and add
		// to the list of decls.
		st = g.genericSubst(sym, nameNode, targs, isMeth)
		g.target.Stencils[sym] = st
		g.target.Decls = append(g.target.Decls, st)
		if base.Flag.W > 1 {
			ir.Dump(fmt.Sprintf("\nstenciled %v", st), st)
		}
	}
	return st
}

// makeInstName makes the unique name for a stenciled generic function or method,
// based on the name of the function fy=nsym and the targs. It replaces any
// existing bracket type list in the name. makeInstName asserts that fnsym has
// brackets in its name if and only if hasBrackets is true.
// TODO(danscales): remove the assertions and the hasBrackets argument later.
//
// Names of declared generic functions have no brackets originally, so hasBrackets
// should be false. Names of generic methods already have brackets, since the new
// type parameter is specified in the generic type of the receiver (e.g. func
// (func (v *value[T]).set(...) { ... } has the original name (*value[T]).set.
//
// The standard naming is something like: 'genFn[int,bool]' for functions and
// '(*genType[int,bool]).methodName' for methods
func makeInstName(fnsym *types.Sym, targs []ir.Node, hasBrackets bool) *types.Sym {
	b := bytes.NewBufferString("")
	name := fnsym.Name
	i := strings.Index(name, "[")
	assert(hasBrackets == (i >= 0))
	if i >= 0 {
		b.WriteString(name[0:i])
	} else {
		b.WriteString(name)
	}
	b.WriteString("[")
	for i, targ := range targs {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(targ.Type().String())
	}
	b.WriteString("]")
	if i >= 0 {
		i2 := strings.Index(name[i:], "]")
		assert(i2 >= 0)
		b.WriteString(name[i+i2+1:])
	}
	return typecheck.Lookup(b.String())
}

// Struct containing info needed for doing the substitution as we create the
// instantiation of a generic function with specified type arguments.
type subster struct {
	g        *irgen
	isMethod bool     // If a method is being instantiated
	newf     *ir.Func // Func node for the new stenciled function
	tparams  []*types.Field
	targs    []ir.Node
	// The substitution map from name nodes in the generic function to the
	// name nodes in the new stenciled function.
	vars map[*ir.Name]*ir.Name
}

// genericSubst returns a new function with name newsym. The function is an
// instantiation of a generic function or method specified by namedNode with type
// args targs. For a method with a generic receiver, it returns an instantiated
// function type where the receiver becomes the first parameter. Otherwise the
// instantiated method would still need to be transformed by later compiler
// phases.
func (g *irgen) genericSubst(newsym *types.Sym, nameNode *ir.Name, targs []ir.Node, isMethod bool) *ir.Func {
	var tparams []*types.Field
	if isMethod {
		// Get the type params from the method receiver (after skipping
		// over any pointer)
		recvType := nameNode.Type().Recv().Type
		recvType = deref(recvType)
		tparams = make([]*types.Field, len(recvType.RParams()))
		for i, rparam := range recvType.RParams() {
			tparams[i] = types.NewField(src.NoXPos, nil, rparam)
		}
	} else {
		tparams = nameNode.Type().TParams().Fields().Slice()
	}
	gf := nameNode.Func
	// Pos of the instantiated function is same as the generic function
	newf := ir.NewFunc(gf.Pos())
	newf.Nname = ir.NewNameAt(gf.Pos(), newsym)
	newf.Nname.Func = newf
	newf.Nname.Defn = newf
	newsym.Def = newf.Nname
	ir.CurFunc = newf

	assert(len(tparams) == len(targs))

	subst := &subster{
		g:        g,
		isMethod: isMethod,
		newf:     newf,
		tparams:  tparams,
		targs:    targs,
		vars:     make(map[*ir.Name]*ir.Name),
	}

	newf.Dcl = make([]*ir.Name, len(gf.Dcl))
	for i, n := range gf.Dcl {
		newf.Dcl[i] = subst.node(n).(*ir.Name)
	}

	// Ugly: we have to insert the Name nodes of the parameters/results into
	// the function type. The current function type has no Nname fields set,
	// because it came via conversion from the types2 type.
	oldt := nameNode.Type()
	// We also transform a generic method type to the corresponding
	// instantiated function type where the receiver is the first parameter.
	newt := types.NewSignature(oldt.Pkg(), nil, nil,
		subst.fields(ir.PPARAM, append(oldt.Recvs().FieldSlice(), oldt.Params().FieldSlice()...), newf.Dcl),
		subst.fields(ir.PPARAMOUT, oldt.Results().FieldSlice(), newf.Dcl))

	newf.Nname.Ntype = ir.TypeNode(newt)
	newf.Nname.SetType(newt)
	ir.MarkFunc(newf.Nname)
	newf.SetTypecheck(1)
	newf.Nname.SetTypecheck(1)

	// Make sure name/type of newf is set before substituting the body.
	newf.Body = subst.list(gf.Body)
	ir.CurFunc = nil

	return newf
}

// node is like DeepCopy(), but creates distinct ONAME nodes, and also descends
// into closures. It substitutes type arguments for type parameters in all the new
// nodes.
func (subst *subster) node(n ir.Node) ir.Node {
	// Use closure to capture all state needed by the ir.EditChildren argument.
	var edit func(ir.Node) ir.Node
	edit = func(x ir.Node) ir.Node {
		switch x.Op() {
		case ir.OTYPE:
			return ir.TypeNode(subst.typ(x.Type()))

		case ir.ONAME:
			name := x.(*ir.Name)
			if v := subst.vars[name]; v != nil {
				return v
			}
			m := ir.NewNameAt(name.Pos(), name.Sym())
			if name.IsClosureVar() {
				m.SetIsClosureVar(true)
			}
			t := x.Type()
			if t == nil {
				assert(name.BuiltinOp != 0)
			} else {
				newt := subst.typ(t)
				m.SetType(newt)
			}
			m.BuiltinOp = name.BuiltinOp
			m.Curfn = subst.newf
			m.Class = name.Class
			m.Func = name.Func
			subst.vars[name] = m
			m.SetTypecheck(1)
			return m
		case ir.OLITERAL, ir.ONIL:
			if x.Sym() != nil {
				return x
			}
		}
		m := ir.Copy(x)
		if _, isExpr := m.(ir.Expr); isExpr {
			t := x.Type()
			if t == nil {
				// t can be nil only if this is a call that has no
				// return values, so allow that and otherwise give
				// an error.
				_, isCallExpr := m.(*ir.CallExpr)
				_, isStructKeyExpr := m.(*ir.StructKeyExpr)
				if !isCallExpr && !isStructKeyExpr && x.Op() != ir.OPANIC &&
					x.Op() != ir.OCLOSE {
					base.Fatalf(fmt.Sprintf("Nil type for %v", x))
				}
			} else if x.Op() != ir.OCLOSURE {
				m.SetType(subst.typ(x.Type()))
			}
		}
		ir.EditChildren(m, edit)

		if x.Typecheck() == 3 {
			// These are nodes whose transforms were delayed until
			// their instantiated type was known.
			m.SetTypecheck(1)
			if typecheck.IsCmp(x.Op()) {
				transformCompare(m.(*ir.BinaryExpr))
			} else {
				switch x.Op() {
				case ir.OSLICE, ir.OSLICE3:
					transformSlice(m.(*ir.SliceExpr))

				case ir.OADD:
					m = transformAdd(m.(*ir.BinaryExpr))

				case ir.OINDEX:
					transformIndex(m.(*ir.IndexExpr))

				case ir.OAS2:
					as2 := m.(*ir.AssignListStmt)
					transformAssign(as2, as2.Lhs, as2.Rhs)

				case ir.OAS:
					as := m.(*ir.AssignStmt)
					lhs, rhs := []ir.Node{as.X}, []ir.Node{as.Y}
					transformAssign(as, lhs, rhs)

				case ir.OASOP:
					as := m.(*ir.AssignOpStmt)
					transformCheckAssign(as, as.X)

				case ir.ORETURN:
					transformReturn(m.(*ir.ReturnStmt))

				case ir.OSEND:
					transformSend(m.(*ir.SendStmt))

				default:
					base.Fatalf("Unexpected node with Typecheck() == 3")
				}
			}
		}

		switch x.Op() {
		case ir.OLITERAL:
			t := m.Type()
			if t != x.Type() {
				// types2 will give us a constant with a type T,
				// if an untyped constant is used with another
				// operand of type T (in a provably correct way).
				// When we substitute in the type args during
				// stenciling, we now know the real type of the
				// constant. We may then need to change the
				// BasicLit.val to be the correct type (e.g.
				// convert an int64Val constant to a floatVal
				// constant).
				m.SetType(types.UntypedInt) // use any untyped type for DefaultLit to work
				m = typecheck.DefaultLit(m, t)
			}

		case ir.OXDOT:
			// A method value/call via a type param will have been
			// left as an OXDOT. When we see this during stenciling,
			// finish the transformation, now that we have the
			// instantiated receiver type. We need to do this now,
			// since the access/selection to the method for the real
			// type is very different from the selection for the type
			// param. m will be transformed to an OCALLPART node. It
			// will be transformed to an ODOTMETH or ODOTINTER node if
			// we find in the OCALL case below that the method value
			// is actually called.
			transformDot(m.(*ir.SelectorExpr), false)
			m.SetTypecheck(1)

		case ir.OCALL:
			call := m.(*ir.CallExpr)
			switch call.X.Op() {
			case ir.OTYPE:
				// Transform the conversion, now that we know the
				// type argument.
				m = transformConvCall(m.(*ir.CallExpr))

			case ir.OCALLPART:
				// Redo the transformation of OXDOT, now that we
				// know the method value is being called. Then
				// transform the call.
				call.X.(*ir.SelectorExpr).SetOp(ir.OXDOT)
				transformDot(call.X.(*ir.SelectorExpr), true)
				transformCall(call)

			case ir.ODOT, ir.ODOTPTR:
				// An OXDOT for a generic receiver was resolved to
				// an access to a field which has a function
				// value. Transform the call to that function, now
				// that the OXDOT was resolved.
				transformCall(call)

			case ir.ONAME:
				name := call.X.Name()
				if name.BuiltinOp != ir.OXXX {
					switch name.BuiltinOp {
					case ir.OMAKE, ir.OREAL, ir.OIMAG, ir.OLEN, ir.OCAP, ir.OAPPEND:
						// Transform these builtins now that we
						// know the type of the args.
						m = transformBuiltin(call)
					default:
						base.FatalfAt(call.Pos(), "Unexpected builtin op")
					}
				} else {
					// This is the case of a function value that was a
					// type parameter (implied to be a function via a
					// structural constraint) which is now resolved.
					transformCall(call)
				}

			case ir.OCLOSURE:
				transformCall(call)

			case ir.OFUNCINST:
				// A call with an OFUNCINST will get transformed
				// in stencil() once we have created & attached the
				// instantiation to be called.

			default:
				base.FatalfAt(call.Pos(), fmt.Sprintf("Unexpected op with CALL during stenciling: %v", call.X.Op()))
			}

		case ir.OCLOSURE:
			x := x.(*ir.ClosureExpr)
			// Need to save/duplicate x.Func.Nname,
			// x.Func.Nname.Ntype, x.Func.Dcl, x.Func.ClosureVars, and
			// x.Func.Body.
			oldfn := x.Func
			newfn := ir.NewFunc(oldfn.Pos())
			if oldfn.ClosureCalled() {
				newfn.SetClosureCalled(true)
			}
			newfn.SetIsHiddenClosure(true)
			m.(*ir.ClosureExpr).Func = newfn
			// Closure name can already have brackets, if it derives
			// from a generic method
			newsym := makeInstName(oldfn.Nname.Sym(), subst.targs, subst.isMethod)
			newfn.Nname = ir.NewNameAt(oldfn.Nname.Pos(), newsym)
			newfn.Nname.Func = newfn
			newfn.Nname.Defn = newfn
			ir.MarkFunc(newfn.Nname)
			newfn.OClosure = m.(*ir.ClosureExpr)

			saveNewf := subst.newf
			ir.CurFunc = newfn
			subst.newf = newfn
			newfn.Dcl = subst.namelist(oldfn.Dcl)
			newfn.ClosureVars = subst.namelist(oldfn.ClosureVars)

			// Set Ntype for now to be compatible with later parts of compiler
			newfn.Nname.Ntype = subst.node(oldfn.Nname.Ntype).(ir.Ntype)
			typed(subst.typ(oldfn.Nname.Type()), newfn.Nname)
			typed(newfn.Nname.Type(), m)
			newfn.SetTypecheck(1)

			// Make sure type of closure function is set before doing body.
			newfn.Body = subst.list(oldfn.Body)
			subst.newf = saveNewf
			ir.CurFunc = saveNewf

			subst.g.target.Decls = append(subst.g.target.Decls, newfn)
		}
		return m
	}

	return edit(n)
}

func (subst *subster) namelist(l []*ir.Name) []*ir.Name {
	s := make([]*ir.Name, len(l))
	for i, n := range l {
		s[i] = subst.node(n).(*ir.Name)
		if n.Defn != nil {
			s[i].Defn = subst.node(n.Defn)
		}
		if n.Outer != nil {
			s[i].Outer = subst.node(n.Outer).(*ir.Name)
		}
	}
	return s
}

func (subst *subster) list(l []ir.Node) []ir.Node {
	s := make([]ir.Node, len(l))
	for i, n := range l {
		s[i] = subst.node(n)
	}
	return s
}

// tstruct substitutes type params in types of the fields of a structure type. For
// each field, if Nname is set, tstruct also translates the Nname using
// subst.vars, if Nname is in subst.vars. To always force the creation of a new
// (top-level) struct, regardless of whether anything changed with the types or
// names of the struct's fields, set force to true.
func (subst *subster) tstruct(t *types.Type, force bool) *types.Type {
	if t.NumFields() == 0 {
		if t.HasTParam() {
			// For an empty struct, we need to return a new type,
			// since it may now be fully instantiated (HasTParam
			// becomes false).
			return types.NewStruct(t.Pkg(), nil)
		}
		return t
	}
	var newfields []*types.Field
	if force {
		newfields = make([]*types.Field, t.NumFields())
	}
	for i, f := range t.Fields().Slice() {
		t2 := subst.typ(f.Type)
		if (t2 != f.Type || f.Nname != nil) && newfields == nil {
			newfields = make([]*types.Field, t.NumFields())
			for j := 0; j < i; j++ {
				newfields[j] = t.Field(j)
			}
		}
		if newfields != nil {
			// TODO(danscales): make sure this works for the field
			// names of embedded types (which should keep the name of
			// the type param, not the instantiated type).
			newfields[i] = types.NewField(f.Pos, f.Sym, t2)
			if f.Nname != nil {
				// f.Nname may not be in subst.vars[] if this is
				// a function name or a function instantiation type
				// that we are translating
				v := subst.vars[f.Nname.(*ir.Name)]
				// Be careful not to put a nil var into Nname,
				// since Nname is an interface, so it would be a
				// non-nil interface.
				if v != nil {
					newfields[i].Nname = v
				}
			}
		}
	}
	if newfields != nil {
		return types.NewStruct(t.Pkg(), newfields)
	}
	return t

}

// tinter substitutes type params in types of the methods of an interface type.
func (subst *subster) tinter(t *types.Type) *types.Type {
	if t.Methods().Len() == 0 {
		return t
	}
	var newfields []*types.Field
	for i, f := range t.Methods().Slice() {
		t2 := subst.typ(f.Type)
		if (t2 != f.Type || f.Nname != nil) && newfields == nil {
			newfields = make([]*types.Field, t.Methods().Len())
			for j := 0; j < i; j++ {
				newfields[j] = t.Methods().Index(j)
			}
		}
		if newfields != nil {
			newfields[i] = types.NewField(f.Pos, f.Sym, t2)
		}
	}
	if newfields != nil {
		return types.NewInterface(t.Pkg(), newfields)
	}
	return t
}

// instTypeName creates a name for an instantiated type, based on the name of the
// generic type and the type args
func instTypeName(name string, targs []*types.Type) string {
	b := bytes.NewBufferString(name)
	b.WriteByte('[')
	for i, targ := range targs {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(targ.String())
	}
	b.WriteByte(']')
	return b.String()
}

// typ computes the type obtained by substituting any type parameter in t with the
// corresponding type argument in subst. If t contains no type parameters, the
// result is t; otherwise the result is a new type. It deals with recursive types
// by using TFORW types and finding partially or fully created types via sym.Def.
func (subst *subster) typ(t *types.Type) *types.Type {
	if !t.HasTParam() {
		return t
	}

	if t.Kind() == types.TTYPEPARAM {
		for i, tp := range subst.tparams {
			if tp.Type == t {
				return subst.targs[i].Type()
			}
		}
		// If t is a simple typeparam T, then t has the name/symbol 'T'
		// and t.Underlying() == t.
		//
		// However, consider the type definition: 'type P[T any] T'. We
		// might use this definition so we can have a variant of type T
		// that we can add new methods to. Suppose t is a reference to
		// P[T]. t has the name 'P[T]', but its kind is TTYPEPARAM,
		// because P[T] is defined as T. If we look at t.Underlying(), it
		// is different, because the name of t.Underlying() is 'T' rather
		// than 'P[T]'. But the kind of t.Underlying() is also TTYPEPARAM.
		// In this case, we do the needed recursive substitution in the
		// case statement below.
		if t.Underlying() == t {
			// t is a simple typeparam that didn't match anything in tparam
			return t
		}
		// t is a more complex typeparam (e.g. P[T], as above, whose
		// definition is just T).
		assert(t.Sym() != nil)
	}

	var newsym *types.Sym
	var neededTargs []*types.Type
	var forw *types.Type

	if t.Sym() != nil {
		// Translate the type params for this type according to
		// the tparam/targs mapping from subst.
		neededTargs = make([]*types.Type, len(t.RParams()))
		for i, rparam := range t.RParams() {
			neededTargs[i] = subst.typ(rparam)
		}
		// For a named (defined) type, we have to change the name of the
		// type as well. We do this first, so we can look up if we've
		// already seen this type during this substitution or other
		// definitions/substitutions.
		genName := genericTypeName(t.Sym())
		newsym = t.Sym().Pkg.Lookup(instTypeName(genName, neededTargs))
		if newsym.Def != nil {
			// We've already created this instantiated defined type.
			return newsym.Def.Type()
		}

		// In order to deal with recursive generic types, create a TFORW
		// type initially and set the Def field of its sym, so it can be
		// found if this type appears recursively within the type.
		forw = newIncompleteNamedType(t.Pos(), newsym)
		//println("Creating new type by sub", newsym.Name, forw.HasTParam())
		forw.SetRParams(neededTargs)
	}

	var newt *types.Type

	switch t.Kind() {
	case types.TTYPEPARAM:
		if t.Sym() == newsym {
			// The substitution did not change the type.
			return t
		}
		// Substitute the underlying typeparam (e.g. T in P[T], see
		// the example describing type P[T] above).
		newt = subst.typ(t.Underlying())
		assert(newt != t)

	case types.TARRAY:
		elem := t.Elem()
		newelem := subst.typ(elem)
		if newelem != elem {
			newt = types.NewArray(newelem, t.NumElem())
		}

	case types.TPTR:
		elem := t.Elem()
		newelem := subst.typ(elem)
		if newelem != elem {
			newt = types.NewPtr(newelem)
		}

	case types.TSLICE:
		elem := t.Elem()
		newelem := subst.typ(elem)
		if newelem != elem {
			newt = types.NewSlice(newelem)
		}

	case types.TSTRUCT:
		newt = subst.tstruct(t, false)
		if newt == t {
			newt = nil
		}

	case types.TFUNC:
		newrecvs := subst.tstruct(t.Recvs(), false)
		newparams := subst.tstruct(t.Params(), false)
		newresults := subst.tstruct(t.Results(), false)
		if newrecvs != t.Recvs() || newparams != t.Params() || newresults != t.Results() {
			// If any types have changed, then the all the fields of
			// of recv, params, and results must be copied, because they have
			// offset fields that are dependent, and so must have an
			// independent copy for each new signature.
			var newrecv *types.Field
			if newrecvs.NumFields() > 0 {
				if newrecvs == t.Recvs() {
					newrecvs = subst.tstruct(t.Recvs(), true)
				}
				newrecv = newrecvs.Field(0)
			}
			if newparams == t.Params() {
				newparams = subst.tstruct(t.Params(), true)
			}
			if newresults == t.Results() {
				newresults = subst.tstruct(t.Results(), true)
			}
			newt = types.NewSignature(t.Pkg(), newrecv, t.TParams().FieldSlice(), newparams.FieldSlice(), newresults.FieldSlice())
		}

	case types.TINTER:
		newt = subst.tinter(t)
		if newt == t {
			newt = nil
		}

	case types.TMAP:
		newkey := subst.typ(t.Key())
		newval := subst.typ(t.Elem())
		if newkey != t.Key() || newval != t.Elem() {
			newt = types.NewMap(newkey, newval)
		}

	case types.TCHAN:
		elem := t.Elem()
		newelem := subst.typ(elem)
		if newelem != elem {
			newt = types.NewChan(newelem, t.ChanDir())
			if !newt.HasTParam() {
				// TODO(danscales): not sure why I have to do this
				// only for channels.....
				types.CheckSize(newt)
			}
		}
	}
	if newt == nil {
		// Even though there were typeparams in the type, there may be no
		// change if this is a function type for a function call (which will
		// have its own tparams/targs in the function instantiation).
		return t
	}

	if t.Sym() == nil {
		// Not a named type, so there was no forwarding type and there are
		// no methods to substitute.
		assert(t.Methods().Len() == 0)
		return newt
	}

	forw.SetUnderlying(newt)
	newt = forw

	if t.Kind() != types.TINTER && t.Methods().Len() > 0 {
		// Fill in the method info for the new type.
		var newfields []*types.Field
		newfields = make([]*types.Field, t.Methods().Len())
		for i, f := range t.Methods().Slice() {
			t2 := subst.typ(f.Type)
			oldsym := f.Nname.Sym()
			newsym := makeInstName(oldsym, subst.targs, true)
			var nname *ir.Name
			if newsym.Def != nil {
				nname = newsym.Def.(*ir.Name)
			} else {
				nname = ir.NewNameAt(f.Pos, newsym)
				nname.SetType(t2)
				newsym.Def = nname
			}
			newfields[i] = types.NewField(f.Pos, f.Sym, t2)
			newfields[i].Nname = nname
		}
		newt.Methods().Set(newfields)
		if !newt.HasTParam() {
			// Generate all the methods for a new fully-instantiated type.
			subst.g.instTypeList = append(subst.g.instTypeList, newt)
		}
	}
	return newt
}

// fields sets the Nname field for the Field nodes inside a type signature, based
// on the corresponding in/out parameters in dcl. It depends on the in and out
// parameters being in order in dcl.
func (subst *subster) fields(class ir.Class, oldfields []*types.Field, dcl []*ir.Name) []*types.Field {
	// Find the starting index in dcl of declarations of the class (either
	// PPARAM or PPARAMOUT).
	var i int
	for i = range dcl {
		if dcl[i].Class == class {
			break
		}
	}

	// Create newfields nodes that are copies of the oldfields nodes, but
	// with substitution for any type params, and with Nname set to be the node in
	// Dcl for the corresponding PPARAM or PPARAMOUT.
	newfields := make([]*types.Field, len(oldfields))
	for j := range oldfields {
		newfields[j] = oldfields[j].Copy()
		newfields[j].Type = subst.typ(oldfields[j].Type)
		// A param field will be missing from dcl if its name is
		// unspecified or specified as "_". So, we compare the dcl sym
		// with the field sym. If they don't match, this dcl (if there is
		// one left) must apply to a later field.
		if i < len(dcl) && dcl[i].Sym() == oldfields[j].Sym {
			newfields[j].Nname = dcl[i]
			i++
		}
	}
	return newfields
}

// defer does a single defer of type t, if it is a pointer type.
func deref(t *types.Type) *types.Type {
	if t.IsPtr() {
		return t.Elem()
	}
	return t
}

// newIncompleteNamedType returns a TFORW type t with name specified by sym, such
// that t.nod and sym.Def are set correctly.
func newIncompleteNamedType(pos src.XPos, sym *types.Sym) *types.Type {
	name := ir.NewDeclNameAt(pos, ir.OTYPE, sym)
	forw := types.NewNamed(name)
	name.SetType(forw)
	sym.Def = name
	return forw
}
