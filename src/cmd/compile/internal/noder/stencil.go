// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file will evolve, since we plan to do a mix of stenciling and passing
// around dictionaries.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"go/constant"
)

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
			// transformCall() below depends on CurFunc being set.
			ir.CurFunc = decl.(*ir.Func)

		case ir.OAS, ir.OAS2, ir.OAS2DOTTYPE, ir.OAS2FUNC, ir.OAS2MAPR, ir.OAS2RECV, ir.OASOP:
			// These are all the various kinds of global assignments,
			// whose right-hand-sides might contain a function
			// instantiation.

		default:
			// The other possible ops at the top level are ODCLCONST
			// and ODCLTYPE, which don't have any function
			// instantiations.
			continue
		}

		// For all non-generic code, search for any function calls using
		// generic function instantiations. Then create the needed
		// instantiated function if it hasn't been created yet, and change
		// to calling that function directly.
		modified := false
		closureRequired := false
		ir.Visit(decl, func(n ir.Node) {
			if n.Op() == ir.OFUNCINST {
				// generic F, not immediately called
				closureRequired = true
			}
			if n.Op() == ir.OMETHEXPR && len(n.(*ir.SelectorExpr).X.Type().RParams()) > 0 {
				// T.M, T a type which is generic, not immediately called
				closureRequired = true
			}
			if n.Op() == ir.OCALL && n.(*ir.CallExpr).X.Op() == ir.OFUNCINST {
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
					call.Args.Prepend(inst.X.(*ir.SelectorExpr).X)
				}
				// Add dictionary to argument list.
				dict := reflectdata.GetDictionaryForInstantiation(inst)
				call.Args.Prepend(dict)
				// Transform the Call now, which changes OCALL
				// to OCALLFUNC and does typecheckaste/assignconvfn.
				transformCall(call)
				modified = true
			}
			if n.Op() == ir.OCALLMETH && n.(*ir.CallExpr).X.Op() == ir.ODOTMETH && len(deref(n.(*ir.CallExpr).X.Type().Recv().Type).RParams()) > 0 {
				// Method call on a generic type, which was instantiated by stenciling.
				// Method calls on explicitly instantiated types will have an OFUNCINST
				// and are handled above.
				call := n.(*ir.CallExpr)
				meth := call.X.(*ir.SelectorExpr)
				targs := deref(meth.Type().Recv().Type).RParams()

				t := meth.X.Type()
				baseSym := deref(t).OrigSym
				baseType := baseSym.Def.(*ir.Name).Type()
				var gf *ir.Name
				for _, m := range baseType.Methods().Slice() {
					if meth.Sel == m.Sym {
						gf = m.Nname.(*ir.Name)
						break
					}
				}

				st := g.getInstantiation(gf, targs, true)
				call.SetOp(ir.OCALL)
				call.X = st.Nname
				dict := reflectdata.GetDictionaryForMethod(gf, targs)
				call.Args.Prepend(dict, meth.X)
				// Transform the Call now, which changes OCALL
				// to OCALLFUNC and does typecheckaste/assignconvfn.
				transformCall(call)
				modified = true
			}
		})

		// If we found a reference to a generic instantiation that wasn't an
		// immediate call, then traverse the nodes of decl again (with
		// EditChildren rather than Visit), where we actually change the
		// reference to the instantiation to a closure that captures the
		// dictionary, then does a direct call.
		// EditChildren is more expensive than Visit, so we only do this
		// in the infrequent case of an OFUNCINST without a corresponding
		// call.
		if closureRequired {
			var edit func(ir.Node) ir.Node
			var outer *ir.Func
			if f, ok := decl.(*ir.Func); ok {
				outer = f
			}
			edit = func(x ir.Node) ir.Node {
				ir.EditChildren(x, edit)
				switch {
				case x.Op() == ir.OFUNCINST:
					// TODO: only set outer!=nil if this instantiation uses
					// a type parameter from outer. See comment in buildClosure.
					return g.buildClosure(outer, x)
				case x.Op() == ir.OMETHEXPR && len(deref(x.(*ir.SelectorExpr).X.Type()).RParams()) > 0: // TODO: test for ptr-to-method case
					return g.buildClosure(outer, x)
				}
				return x
			}
			edit(decl)
		}
		if base.Flag.W > 1 && modified {
			ir.Dump(fmt.Sprintf("\nmodified %v", decl), decl)
		}
		ir.CurFunc = nil
		// We may have seen new fully-instantiated generic types while
		// instantiating any needed functions/methods in the above
		// function. If so, instantiate all the methods of those types
		// (which will then lead to more function/methods to scan in the loop).
		g.instantiateMethods()
	}

}

// buildClosure makes a closure to implement x, a OFUNCINST or OMETHEXPR
// of generic type. outer is the containing function (or nil if closure is
// in a global assignment instead of a function).
func (g *irgen) buildClosure(outer *ir.Func, x ir.Node) ir.Node {
	pos := x.Pos()
	var target *ir.Func   // target instantiated function/method
	var dictValue ir.Node // dictionary to use
	var rcvrValue ir.Node // receiver, if a method value
	typ := x.Type()       // type of the closure
	if x.Op() == ir.OFUNCINST {
		inst := x.(*ir.InstExpr)

		// Type arguments we're instantiating with.
		targs := typecheck.TypesOf(inst.Targs)

		// Find the generic function/method.
		var gf *ir.Name
		if inst.X.Op() == ir.ONAME {
			// Instantiating a generic function call.
			gf = inst.X.(*ir.Name)
		} else if inst.X.Op() == ir.OCALLPART {
			// Instantiating a method value x.M.
			se := inst.X.(*ir.SelectorExpr)
			rcvrValue = se.X
			gf = se.Selection.Nname.(*ir.Name)
		} else {
			panic("unhandled")
		}

		// target is the instantiated function we're trying to call.
		// For functions, the target expects a dictionary as its first argument.
		// For method values, the target expects a dictionary and the receiver
		// as its first two arguments.
		target = g.getInstantiation(gf, targs, rcvrValue != nil)

		// The value to use for the dictionary argument.
		if rcvrValue == nil {
			dictValue = reflectdata.GetDictionaryForFunc(gf, targs)
		} else {
			dictValue = reflectdata.GetDictionaryForMethod(gf, targs)
		}
	} else { // ir.OMETHEXPR
		// Method expression T.M where T is a generic type.
		// TODO: Is (*T).M right?
		se := x.(*ir.SelectorExpr)
		targs := se.X.Type().RParams()
		if len(targs) == 0 {
			if se.X.Type().IsPtr() {
				targs = se.X.Type().Elem().RParams()
				if len(targs) == 0 {
					panic("bad")
				}
			}
		}
		t := se.X.Type()
		baseSym := t.OrigSym
		baseType := baseSym.Def.(*ir.Name).Type()
		var gf *ir.Name
		for _, m := range baseType.Methods().Slice() {
			if se.Sel == m.Sym {
				gf = m.Nname.(*ir.Name)
				break
			}
		}
		target = g.getInstantiation(gf, targs, true)
		dictValue = reflectdata.GetDictionaryForMethod(gf, targs)
	}

	// Build a closure to implement a function instantiation.
	//
	//   func f[T any] (int, int) (int, int) { ...whatever... }
	//
	// Then any reference to f[int] not directly called gets rewritten to
	//
	//   .dictN := ... dictionary to use ...
	//   func(a0, a1 int) (r0, r1 int) {
	//     return .inst.f[int](.dictN, a0, a1)
	//   }
	//
	// Similarly for method expressions,
	//
	//   type g[T any] ....
	//   func (rcvr g[T]) f(a0, a1 int) (r0, r1 int) { ... }
	//
	// Any reference to g[int].f not directly called gets rewritten to
	//
	//   .dictN := ... dictionary to use ...
	//   func(rcvr g[int], a0, a1 int) (r0, r1 int) {
	//     return .inst.g[int].f(.dictN, rcvr, a0, a1)
	//   }
	//
	// Also method values
	//
	//   var x g[int]
	//
	// Any reference to x.f not directly called gets rewritten to
	//
	//   .dictN := ... dictionary to use ...
	//   x2 := x
	//   func(a0, a1 int) (r0, r1 int) {
	//     return .inst.g[int].f(.dictN, x2, a0, a1)
	//   }

	// Make a new internal function.
	fn := ir.NewFunc(pos)
	fn.SetIsHiddenClosure(true)

	// This is the dictionary we want to use.
	// It may be a constant, or it may be a dictionary acquired from the outer function's dictionary.
	// For the latter, dictVar is a variable in the outer function's scope, set to the subdictionary
	// read from the outer function's dictionary.
	var dictVar *ir.Name
	var dictAssign *ir.AssignStmt
	if outer != nil {
		// Note: for now this is a compile-time constant, so we don't really need a closure
		// to capture it (a wrapper function would work just as well). But eventually it
		// will be a read of a subdictionary from the parent dictionary.
		dictVar = ir.NewNameAt(pos, typecheck.LookupNum(".dict", g.dnum))
		g.dnum++
		dictVar.Class = ir.PAUTO
		typed(types.Types[types.TUINTPTR], dictVar)
		dictVar.Curfn = outer
		dictAssign = ir.NewAssignStmt(pos, dictVar, dictValue)
		dictAssign.SetTypecheck(1)
		dictVar.Defn = dictAssign
		outer.Dcl = append(outer.Dcl, dictVar)
	}
	// assign the receiver to a temporary.
	var rcvrVar *ir.Name
	var rcvrAssign ir.Node
	if rcvrValue != nil {
		rcvrVar = ir.NewNameAt(pos, typecheck.LookupNum(".rcvr", g.dnum))
		g.dnum++
		rcvrVar.Class = ir.PAUTO
		typed(rcvrValue.Type(), rcvrVar)
		rcvrVar.Curfn = outer
		rcvrAssign = ir.NewAssignStmt(pos, rcvrVar, rcvrValue)
		rcvrAssign.SetTypecheck(1)
		rcvrVar.Defn = rcvrAssign
		outer.Dcl = append(outer.Dcl, rcvrVar)
	}

	// Build formal argument and return lists.
	var formalParams []*types.Field  // arguments of closure
	var formalResults []*types.Field // returns of closure
	for i := 0; i < typ.NumParams(); i++ {
		t := typ.Params().Field(i).Type
		arg := ir.NewNameAt(pos, typecheck.LookupNum("a", i))
		arg.Class = ir.PPARAM
		typed(t, arg)
		arg.Curfn = fn
		fn.Dcl = append(fn.Dcl, arg)
		f := types.NewField(pos, arg.Sym(), t)
		f.Nname = arg
		formalParams = append(formalParams, f)
	}
	for i := 0; i < typ.NumResults(); i++ {
		t := typ.Results().Field(i).Type
		result := ir.NewNameAt(pos, typecheck.LookupNum("r", i)) // TODO: names not needed?
		result.Class = ir.PPARAMOUT
		typed(t, result)
		result.Curfn = fn
		fn.Dcl = append(fn.Dcl, result)
		f := types.NewField(pos, result.Sym(), t)
		f.Nname = result
		formalResults = append(formalResults, f)
	}

	// Build an internal function with the right signature.
	closureType := types.NewSignature(x.Type().Pkg(), nil, nil, formalParams, formalResults)
	sym := typecheck.ClosureName(outer)
	sym.SetFunc(true)
	fn.Nname = ir.NewNameAt(pos, sym)
	fn.Nname.Class = ir.PFUNC
	fn.Nname.Func = fn
	fn.Nname.Defn = fn
	typed(closureType, fn.Nname)
	fn.SetTypecheck(1)

	// Build body of closure. This involves just calling the wrapped function directly
	// with the additional dictionary argument.

	// First, figure out the dictionary argument.
	var dict2Var ir.Node
	if outer != nil {
		// If there's an outer function, the dictionary value will be read from
		// the dictionary of the outer function.
		// TODO: only use a subdictionary if any of the instantiating types
		// depend on the type params of the outer function.
		dict2Var = ir.CaptureName(pos, fn, dictVar)
	} else {
		// No outer function, instantiating types are known concrete types.
		dict2Var = dictValue
	}
	// Also capture the receiver variable.
	var rcvr2Var *ir.Name
	if rcvrValue != nil {
		rcvr2Var = ir.CaptureName(pos, fn, rcvrVar)
	}

	// Build arguments to call inside the closure.
	var args []ir.Node

	// First the dictionary argument.
	args = append(args, dict2Var)
	// Then the receiver.
	if rcvrValue != nil {
		args = append(args, rcvr2Var)
	}
	// Then all the other arguments (including receiver for method expressions).
	for i := 0; i < typ.NumParams(); i++ {
		args = append(args, formalParams[i].Nname.(*ir.Name))
	}

	// Build call itself.
	var innerCall ir.Node = ir.NewCallExpr(pos, ir.OCALL, target.Nname, args)
	if len(formalResults) > 0 {
		innerCall = ir.NewReturnStmt(pos, []ir.Node{innerCall})
	}
	// Finish building body of closure.
	ir.CurFunc = fn
	// TODO: set types directly here instead of using typecheck.Stmt
	typecheck.Stmt(innerCall)
	ir.CurFunc = nil
	fn.Body = []ir.Node{innerCall}
	if outer == nil {
		g.target.Decls = append(g.target.Decls, fn)
	}

	// We're all done with the captured dictionary (and receiver, for method values).
	ir.FinishCaptureNames(pos, outer, fn)

	// Make a closure referencing our new internal function.
	c := ir.NewClosureExpr(pos, fn)
	var init []ir.Node
	if outer != nil {
		init = append(init, dictAssign)
	}
	if rcvrValue != nil {
		init = append(init, rcvrAssign)
	}
	c.SetInit(init)
	typed(x.Type(), c)
	return c
}

// instantiateMethods instantiates all the methods of all fully-instantiated
// generic types that have been added to g.instTypeList.
func (g *irgen) instantiateMethods() {
	for i := 0; i < len(g.instTypeList); i++ {
		typ := g.instTypeList[i]
		// Mark runtime type as needed, since this ensures that the
		// compiler puts out the needed DWARF symbols, when this
		// instantiated type has a different package from the local
		// package.
		typecheck.NeedRuntimeType(typ)
		// Lookup the method on the base generic type, since methods may
		// not be set on imported instantiated types.
		baseSym := typ.OrigSym
		baseType := baseSym.Def.(*ir.Name).Type()
		for j, _ := range typ.Methods().Slice() {
			baseNname := baseType.Methods().Slice()[j].Nname.(*ir.Name)
			// Eagerly generate the instantiations that implement these methods.
			// We don't use the instantiations here, just generate them (and any
			// further instantiations those generate, etc.).
			// Note that we don't set the Func for any methods on instantiated
			// types. Their signatures don't match so that would be confusing.
			// Direct method calls go directly to the instantiations, implemented above.
			// Indirect method calls use wrappers generated in reflectcall. Those wrappers
			// will use these instantiations if they are needed (for interface tables or reflection).
			_ = g.getInstantiation(baseNname, typ.RParams(), true)
		}
	}
	g.instTypeList = nil

}

// getInstantiationForNode returns the function/method instantiation for a
// InstExpr node inst.
func (g *irgen) getInstantiationForNode(inst *ir.InstExpr) *ir.Func {
	if meth, ok := inst.X.(*ir.SelectorExpr); ok {
		return g.getInstantiation(meth.Selection.Nname.(*ir.Name), typecheck.TypesOf(inst.Targs), true)
	} else {
		return g.getInstantiation(inst.X.(*ir.Name), typecheck.TypesOf(inst.Targs), false)
	}
}

// getInstantiation gets the instantiantion of the function or method nameNode
// with the type arguments targs. If the instantiated function is not already
// cached, then it calls genericSubst to create the new instantiation.
func (g *irgen) getInstantiation(nameNode *ir.Name, targs []*types.Type, isMeth bool) *ir.Func {
	if nameNode.Func.Body == nil && nameNode.Func.Inl != nil {
		// If there is no body yet but Func.Inl exists, then we can can
		// import the whole generic body.
		assert(nameNode.Func.Inl.Cost == 1 && nameNode.Sym().Pkg != types.LocalPkg)
		typecheck.ImportBody(nameNode.Func)
		assert(nameNode.Func.Inl.Body != nil)
		nameNode.Func.Body = nameNode.Func.Inl.Body
		nameNode.Func.Dcl = nameNode.Func.Inl.Dcl
	}
	sym := typecheck.MakeInstName(nameNode.Sym(), targs, isMeth)
	st := g.target.Stencils[sym]
	if st == nil {
		// If instantiation doesn't exist yet, create it and add
		// to the list of decls.
		st = g.genericSubst(sym, nameNode, targs, isMeth)
		// This ensures that the linker drops duplicates of this instantiation.
		// All just works!
		st.SetDupok(true)
		g.target.Stencils[sym] = st
		g.target.Decls = append(g.target.Decls, st)
		if base.Flag.W > 1 {
			ir.Dump(fmt.Sprintf("\nstenciled %v", st), st)
		}
	}
	return st
}

// Struct containing info needed for doing the substitution as we create the
// instantiation of a generic function with specified type arguments.
type subster struct {
	g          *irgen
	isMethod   bool     // If a method is being instantiated
	newf       *ir.Func // Func node for the new stenciled function
	ts         typecheck.Tsubster
	dictionary *ir.Name // Name of dictionary variable
}

// genericSubst returns a new function with name newsym. The function is an
// instantiation of a generic function or method specified by namedNode with type
// args targs. For a method with a generic receiver, it returns an instantiated
// function type where the receiver becomes the first parameter. Otherwise the
// instantiated method would still need to be transformed by later compiler
// phases.
func (g *irgen) genericSubst(newsym *types.Sym, nameNode *ir.Name, targs []*types.Type, isMethod bool) *ir.Func {
	var tparams []*types.Type
	if isMethod {
		// Get the type params from the method receiver (after skipping
		// over any pointer)
		recvType := nameNode.Type().Recv().Type
		recvType = deref(recvType)
		tparams = recvType.RParams()
	} else {
		fields := nameNode.Type().TParams().Fields().Slice()
		tparams = make([]*types.Type, len(fields))
		for i, f := range fields {
			tparams[i] = f.Type
		}
	}
	gf := nameNode.Func
	// Pos of the instantiated function is same as the generic function
	newf := ir.NewFunc(gf.Pos())
	newf.Pragma = gf.Pragma // copy over pragmas from generic function to stenciled implementation.
	newf.Nname = ir.NewNameAt(gf.Pos(), newsym)
	newf.Nname.Func = newf
	newf.Nname.Defn = newf
	newsym.Def = newf.Nname
	savef := ir.CurFunc
	// transformCall/transformReturn (called during stenciling of the body)
	// depend on ir.CurFunc being set.
	ir.CurFunc = newf

	assert(len(tparams) == len(targs))

	subst := &subster{
		g:        g,
		isMethod: isMethod,
		newf:     newf,
		ts: typecheck.Tsubster{
			Tparams: tparams,
			Targs:   targs,
			Vars:    make(map[*ir.Name]*ir.Name),
		},
	}

	newf.Dcl = make([]*ir.Name, 0, len(gf.Dcl)+1)

	// Replace the types in the function signature.
	// Ugly: also, we have to insert the Name nodes of the parameters/results into
	// the function type. The current function type has no Nname fields set,
	// because it came via conversion from the types2 type.
	oldt := nameNode.Type()
	// We also transform a generic method type to the corresponding
	// instantiated function type where the dictionary is the first parameter.
	dictionarySym := newsym.Pkg.Lookup(".dict")
	dictionaryType := types.Types[types.TUINTPTR]
	dictionaryName := ir.NewNameAt(gf.Pos(), dictionarySym)
	typed(dictionaryType, dictionaryName)
	dictionaryName.Class = ir.PPARAM
	dictionaryName.Curfn = newf
	newf.Dcl = append(newf.Dcl, dictionaryName)
	for _, n := range gf.Dcl {
		if n.Sym().Name == ".dict" {
			panic("already has dictionary")
		}
		newf.Dcl = append(newf.Dcl, subst.localvar(n))
	}
	dictionaryArg := types.NewField(gf.Pos(), dictionarySym, dictionaryType)
	dictionaryArg.Nname = dictionaryName
	subst.dictionary = dictionaryName
	var args []*types.Field
	args = append(args, dictionaryArg)
	args = append(args, oldt.Recvs().FieldSlice()...)
	args = append(args, oldt.Params().FieldSlice()...)
	newt := types.NewSignature(oldt.Pkg(), nil, nil,
		subst.fields(ir.PPARAM, args, newf.Dcl),
		subst.fields(ir.PPARAMOUT, oldt.Results().FieldSlice(), newf.Dcl))

	typed(newt, newf.Nname)
	ir.MarkFunc(newf.Nname)
	newf.SetTypecheck(1)

	// Make sure name/type of newf is set before substituting the body.
	newf.Body = subst.list(gf.Body)

	// Add code to check that the dictionary is correct.
	newf.Body.Prepend(g.checkDictionary(dictionaryName, targs)...)

	ir.CurFunc = savef
	// Add any new, fully instantiated types seen during the substitution to
	// g.instTypeList.
	g.instTypeList = append(g.instTypeList, subst.ts.InstTypeList...)

	return newf
}

// localvar creates a new name node for the specified local variable and enters it
// in subst.vars. It substitutes type arguments for type parameters in the type of
// name as needed.
func (subst *subster) localvar(name *ir.Name) *ir.Name {
	m := ir.NewNameAt(name.Pos(), name.Sym())
	if name.IsClosureVar() {
		m.SetIsClosureVar(true)
	}
	m.SetType(subst.ts.Typ(name.Type()))
	m.BuiltinOp = name.BuiltinOp
	m.Curfn = subst.newf
	m.Class = name.Class
	assert(name.Class != ir.PEXTERN && name.Class != ir.PFUNC)
	m.Func = name.Func
	subst.ts.Vars[name] = m
	m.SetTypecheck(1)
	return m
}

// checkDictionary returns code that does runtime consistency checks
// between the dictionary and the types it should contain.
func (g *irgen) checkDictionary(name *ir.Name, targs []*types.Type) (code []ir.Node) {
	if false {
		return // checking turned off
	}
	// TODO: when moving to GCshape, this test will become harder. Call into
	// runtime to check the expected shape is correct?
	pos := name.Pos()
	// Convert dictionary to *[N]uintptr
	d := ir.NewConvExpr(pos, ir.OCONVNOP, types.Types[types.TUNSAFEPTR], name)
	d.SetTypecheck(1)
	d = ir.NewConvExpr(pos, ir.OCONVNOP, types.NewArray(types.Types[types.TUINTPTR], int64(len(targs))).PtrTo(), d)
	d.SetTypecheck(1)

	// Check that each type entry in the dictionary is correct.
	for i, t := range targs {
		want := reflectdata.TypePtr(t)
		typed(types.Types[types.TUINTPTR], want)
		deref := ir.NewStarExpr(pos, d)
		typed(d.Type().Elem(), deref)
		idx := ir.NewConstExpr(constant.MakeUint64(uint64(i)), name) // TODO: what to set orig to?
		typed(types.Types[types.TUINTPTR], idx)
		got := ir.NewIndexExpr(pos, deref, idx)
		typed(types.Types[types.TUINTPTR], got)
		cond := ir.NewBinaryExpr(pos, ir.ONE, want, got)
		typed(types.Types[types.TBOOL], cond)
		panicArg := ir.NewNilExpr(pos)
		typed(types.NewInterface(types.LocalPkg, nil), panicArg)
		then := ir.NewUnaryExpr(pos, ir.OPANIC, panicArg)
		then.SetTypecheck(1)
		x := ir.NewIfStmt(pos, cond, []ir.Node{then}, nil)
		x.SetTypecheck(1)
		code = append(code, x)
	}
	return
}

// getDictionaryType returns a *runtime._type from the dictionary corresponding to the input type.
// The input type must be a type parameter (TODO: or a local derived type).
func (subst *subster) getDictionaryType(pos src.XPos, t *types.Type) ir.Node {
	tparams := subst.ts.Tparams
	var i = 0
	for i = range tparams {
		if t == tparams[i] {
			break
		}
	}
	if i == len(tparams) {
		base.Fatalf(fmt.Sprintf("couldn't find type param %+v", t))
	}

	// Convert dictionary to *[N]uintptr
	// All entries in the dictionary are pointers. They all point to static data, though, so we
	// treat them as uintptrs so the GC doesn't need to keep track of them.
	d := ir.NewConvExpr(pos, ir.OCONVNOP, types.Types[types.TUNSAFEPTR], subst.dictionary)
	d.SetTypecheck(1)
	d = ir.NewConvExpr(pos, ir.OCONVNOP, types.NewArray(types.Types[types.TUINTPTR], int64(len(tparams))).PtrTo(), d)
	d.SetTypecheck(1)

	// Load entry i out of the dictionary.
	deref := ir.NewStarExpr(pos, d)
	typed(d.Type().Elem(), deref)
	idx := ir.NewConstExpr(constant.MakeUint64(uint64(i)), subst.dictionary) // TODO: what to set orig to?
	typed(types.Types[types.TUINTPTR], idx)
	r := ir.NewIndexExpr(pos, deref, idx)
	typed(types.Types[types.TUINT8].PtrTo(), r) // standard typing of a *runtime._type in the compiler is *byte
	return r
}

// node is like DeepCopy(), but substitutes ONAME nodes based on subst.ts.vars, and
// also descends into closures. It substitutes type arguments for type parameters
// in all the new nodes.
func (subst *subster) node(n ir.Node) ir.Node {
	// Use closure to capture all state needed by the ir.EditChildren argument.
	var edit func(ir.Node) ir.Node
	edit = func(x ir.Node) ir.Node {
		switch x.Op() {
		case ir.OTYPE:
			return ir.TypeNode(subst.ts.Typ(x.Type()))

		case ir.ONAME:
			if v := subst.ts.Vars[x.(*ir.Name)]; v != nil {
				return v
			}
			return x
		case ir.ONONAME:
			// This handles the identifier in a type switch guard
			fallthrough
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
				m.SetType(subst.ts.Typ(x.Type()))
			}
		}
		ir.EditChildren(m, edit)

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
				if as.Y != nil {
					// transformAssign doesn't handle the case
					// of zeroing assignment of a dcl (rhs[0] is nil).
					lhs, rhs := []ir.Node{as.X}, []ir.Node{as.Y}
					transformAssign(as, lhs, rhs)
				}

			case ir.OASOP:
				as := m.(*ir.AssignOpStmt)
				transformCheckAssign(as, as.X)

			case ir.ORETURN:
				transformReturn(m.(*ir.ReturnStmt))

			case ir.OSEND:
				transformSend(m.(*ir.SendStmt))

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
			// Need to duplicate x.Func.Nname, x.Func.Dcl, x.Func.ClosureVars, and
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
			newsym := typecheck.MakeInstName(oldfn.Nname.Sym(), subst.ts.Targs, subst.isMethod)
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

			typed(subst.ts.Typ(oldfn.Nname.Type()), newfn.Nname)
			typed(newfn.Nname.Type(), m)
			newfn.SetTypecheck(1)

			// Make sure type of closure function is set before doing body.
			newfn.Body = subst.list(oldfn.Body)
			subst.newf = saveNewf
			ir.CurFunc = saveNewf

			subst.g.target.Decls = append(subst.g.target.Decls, newfn)

		case ir.OCONVIFACE:
			x := x.(*ir.ConvExpr)
			// TODO: handle converting from derived types. For now, just from naked
			// type parameters.
			if x.X.Type().IsTypeParam() {
				// Load the actual runtime._type of the type parameter from the dictionary.
				rt := subst.getDictionaryType(m.Pos(), x.X.Type())

				// At this point, m is an interface type with a data word we want.
				// But the type word represents a gcshape type, which we don't want.
				// Replace with the instantiated type loaded from the dictionary.
				m = ir.NewUnaryExpr(m.Pos(), ir.OIDATA, m)
				typed(types.Types[types.TUNSAFEPTR], m)
				m = ir.NewBinaryExpr(m.Pos(), ir.OEFACE, rt, m)
				if !x.Type().IsEmptyInterface() {
					// We just built an empty interface{}. Type it as such,
					// then assert it to the required non-empty interface.
					typed(types.NewInterface(types.LocalPkg, nil), m)
					m = ir.NewTypeAssertExpr(m.Pos(), m, nil)
				}
				typed(x.Type(), m)
				// TODO: we're throwing away the type word of the original version
				// of m here (it would be OITAB(m)), which probably took some
				// work to generate. Can we avoid generating it at all?
				// (The linker will throw them away if not needed, so it would just
				// save toolchain work, not binary size.)
			}
		}
		return m
	}

	return edit(n)
}

func (subst *subster) namelist(l []*ir.Name) []*ir.Name {
	s := make([]*ir.Name, len(l))
	for i, n := range l {
		s[i] = subst.localvar(n)
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
		newfields[j].Type = subst.ts.Typ(oldfields[j].Type)
		// A PPARAM field will be missing from dcl if its name is
		// unspecified or specified as "_". So, we compare the dcl sym
		// with the field sym (or sym of the field's Nname node). (Unnamed
		// results still have a name like ~r2 in their Nname node.) If
		// they don't match, this dcl (if there is one left) must apply to
		// a later field.
		if i < len(dcl) && (dcl[i].Sym() == oldfields[j].Sym ||
			(oldfields[j].Nname != nil && dcl[i].Sym() == oldfields[j].Nname.Sym())) {
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
