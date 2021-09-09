// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file will evolve, since we plan to do a mix of stenciling and passing
// around dictionaries.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"go/constant"
)

// Enable extra consistency checks.
const doubleCheck = true

func assert(p bool) {
	base.Assert(p)
}

// Temporary - for outputting information on derived types, dictionaries, sub-dictionaries.
// Turn off when running tests.
var infoPrintMode = false

func infoPrint(format string, a ...interface{}) {
	if infoPrintMode {
		fmt.Printf(format, a...)
	}
}

// stencil scans functions for instantiated generic function calls and creates the
// required instantiations for simple generic functions. It also creates
// instantiated methods for all fully-instantiated generic types that have been
// encountered already or new ones that are encountered during the stenciling
// process.
func (g *irgen) stencil() {
	g.instInfoMap = make(map[*types.Sym]*instInfo)
	g.gfInfoMap = make(map[*types.Sym]*gfInfo)

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
		// declInfo will be non-nil exactly if we are scanning an instantiated function
		declInfo := g.instInfoMap[decl.Sym()]

		ir.Visit(decl, func(n ir.Node) {
			if n.Op() == ir.OFUNCINST {
				// generic F, not immediately called
				closureRequired = true
			}
			if (n.Op() == ir.OMETHEXPR || n.Op() == ir.OMETHVALUE) && len(deref(n.(*ir.SelectorExpr).X.Type()).RParams()) > 0 && !types.IsInterfaceMethod(n.(*ir.SelectorExpr).Selection.Type) {
				// T.M or x.M, where T or x is generic, but not immediately
				// called. Not necessary if the method selected is
				// actually for an embedded interface field.
				closureRequired = true
			}
			if n.Op() == ir.OCALL && n.(*ir.CallExpr).X.Op() == ir.OFUNCINST {
				// We have found a function call using a generic function
				// instantiation.
				call := n.(*ir.CallExpr)
				inst := call.X.(*ir.InstExpr)
				nameNode, isMeth := g.getInstNameNode(inst)
				targs := typecheck.TypesOf(inst.Targs)
				st := g.getInstantiation(nameNode, targs, isMeth)
				dictValue, usingSubdict := g.getDictOrSubdict(declInfo, n, nameNode, targs, isMeth)
				if infoPrintMode {
					dictkind := "Main dictionary"
					if usingSubdict {
						dictkind = "Sub-dictionary"
					}
					if inst.X.Op() == ir.OMETHVALUE {
						fmt.Printf("%s in %v at generic method call: %v - %v\n", dictkind, decl, inst.X, call)
					} else {
						fmt.Printf("%s in %v at generic function call: %v - %v\n", dictkind, decl, inst.X, call)
					}
				}

				// Transform the Call now, which changes OCALL to
				// OCALLFUNC and does typecheckaste/assignconvfn. Do
				// it before installing the instantiation, so we are
				// checking against non-shape param types in
				// typecheckaste.
				transformCall(call)

				// Replace the OFUNCINST with a direct reference to the
				// new stenciled function
				call.X = st.Nname
				if inst.X.Op() == ir.OMETHVALUE {
					// When we create an instantiation of a method
					// call, we make it a function. So, move the
					// receiver to be the first arg of the function
					// call.
					call.Args.Prepend(inst.X.(*ir.SelectorExpr).X)
				}

				// Add dictionary to argument list.
				call.Args.Prepend(dictValue)
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
				baseSym := deref(t).OrigSym()
				baseType := baseSym.Def.(*ir.Name).Type()
				var gf *ir.Name
				for _, m := range baseType.Methods().Slice() {
					if meth.Sel == m.Sym {
						gf = m.Nname.(*ir.Name)
						break
					}
				}

				// Transform the Call now, which changes OCALL
				// to OCALLFUNC and does typecheckaste/assignconvfn.
				transformCall(call)

				st := g.getInstantiation(gf, targs, true)
				dictValue, usingSubdict := g.getDictOrSubdict(declInfo, n, gf, targs, true)
				// We have to be using a subdictionary, since this is
				// a generic method call.
				assert(usingSubdict)

				// Transform to a function call, by appending the
				// dictionary and the receiver to the args.
				call.SetOp(ir.OCALLFUNC)
				call.X = st.Nname
				call.Args.Prepend(dictValue, meth.X)
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
			modified = true
			var edit func(ir.Node) ir.Node
			var outer *ir.Func
			if f, ok := decl.(*ir.Func); ok {
				outer = f
			}
			edit = func(x ir.Node) ir.Node {
				if x.Op() == ir.OFUNCINST {
					child := x.(*ir.InstExpr).X
					if child.Op() == ir.OMETHEXPR || child.Op() == ir.OMETHVALUE {
						// Call EditChildren on child (x.X),
						// not x, so that we don't do
						// buildClosure() on the
						// METHEXPR/METHVALUE nodes as well.
						ir.EditChildren(child, edit)
						return g.buildClosure(outer, x)
					}
				}
				ir.EditChildren(x, edit)
				switch {
				case x.Op() == ir.OFUNCINST:
					return g.buildClosure(outer, x)
				case (x.Op() == ir.OMETHEXPR || x.Op() == ir.OMETHVALUE) &&
					len(deref(x.(*ir.SelectorExpr).X.Type()).RParams()) > 0 &&
					!types.IsInterfaceMethod(x.(*ir.SelectorExpr).Selection.Type):
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

	g.finalizeSyms()
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
	var outerInfo *instInfo
	if outer != nil {
		outerInfo = g.instInfoMap[outer.Sym()]
	}
	usingSubdict := false
	valueMethod := false
	if x.Op() == ir.OFUNCINST {
		inst := x.(*ir.InstExpr)

		// Type arguments we're instantiating with.
		targs := typecheck.TypesOf(inst.Targs)

		// Find the generic function/method.
		var gf *ir.Name
		if inst.X.Op() == ir.ONAME {
			// Instantiating a generic function call.
			gf = inst.X.(*ir.Name)
		} else if inst.X.Op() == ir.OMETHVALUE {
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
		// dictValue is the value to use for the dictionary argument.
		target = g.getInstantiation(gf, targs, rcvrValue != nil)
		dictValue, usingSubdict = g.getDictOrSubdict(outerInfo, x, gf, targs, rcvrValue != nil)
		if infoPrintMode {
			dictkind := "Main dictionary"
			if usingSubdict {
				dictkind = "Sub-dictionary"
			}
			if rcvrValue == nil {
				fmt.Printf("%s in %v for generic function value %v\n", dictkind, outer, inst.X)
			} else {
				fmt.Printf("%s in %v for generic method value %v\n", dictkind, outer, inst.X)
			}
		}
	} else { // ir.OMETHEXPR or ir.METHVALUE
		// Method expression T.M where T is a generic type.
		se := x.(*ir.SelectorExpr)
		targs := deref(se.X.Type()).RParams()
		if len(targs) == 0 {
			panic("bad")
		}
		if x.Op() == ir.OMETHVALUE {
			rcvrValue = se.X
		}

		// se.X.Type() is the top-level type of the method expression. To
		// correctly handle method expressions involving embedded fields,
		// look up the generic method below using the type of the receiver
		// of se.Selection, since that will be the type that actually has
		// the method.
		recv := deref(se.Selection.Type.Recv().Type)
		if len(recv.RParams()) == 0 {
			// The embedded type that actually has the method is not
			// actually generic, so no need to build a closure.
			return x
		}
		baseType := recv.OrigSym().Def.Type()
		var gf *ir.Name
		for _, m := range baseType.Methods().Slice() {
			if se.Sel == m.Sym {
				gf = m.Nname.(*ir.Name)
				break
			}
		}
		if !gf.Type().Recv().Type.IsPtr() {
			// Remember if value method, so we can detect (*T).M case.
			valueMethod = true
		}
		target = g.getInstantiation(gf, targs, true)
		dictValue, usingSubdict = g.getDictOrSubdict(outerInfo, x, gf, targs, true)
		if infoPrintMode {
			dictkind := "Main dictionary"
			if usingSubdict {
				dictkind = "Sub-dictionary"
			}
			fmt.Printf("%s in %v for method expression %v\n", dictkind, outer, x)
		}
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
	fn, formalParams, formalResults := startClosure(pos, outer, typ)

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
		typed(rcvrValue.Type(), rcvrVar)
		rcvrAssign = ir.NewAssignStmt(pos, rcvrVar, rcvrValue)
		rcvrAssign.SetTypecheck(1)
		rcvrVar.Defn = rcvrAssign
		if outer == nil {
			rcvrVar.Class = ir.PEXTERN
			g.target.Decls = append(g.target.Decls, rcvrAssign)
			g.target.Externs = append(g.target.Externs, rcvrVar)
		} else {
			rcvrVar.Class = ir.PAUTO
			rcvrVar.Curfn = outer
			outer.Dcl = append(outer.Dcl, rcvrVar)
		}
	}

	// Build body of closure. This involves just calling the wrapped function directly
	// with the additional dictionary argument.

	// First, figure out the dictionary argument.
	var dict2Var ir.Node
	if usingSubdict {
		// Capture sub-dictionary calculated in the outer function
		dict2Var = ir.CaptureName(pos, fn, dictVar)
		typed(types.Types[types.TUINTPTR], dict2Var)
	} else {
		// Static dictionary, so can be used directly in the closure
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
		if x.Op() == ir.OMETHEXPR && i == 0 {
			// If we are doing a method expression, we need to
			// explicitly traverse any embedded fields in the receiver
			// argument in order to call the method instantiation.
			arg0 := formalParams[0].Nname.(ir.Node)
			arg0 = typecheck.AddImplicitDots(ir.NewSelectorExpr(base.Pos, ir.OXDOT, arg0, x.(*ir.SelectorExpr).Sel)).X
			if valueMethod && arg0.Type().IsPtr() {
				// For handling the (*T).M case: if we have a pointer
				// receiver after following all the embedded fields,
				// but it's a value method, add a star operator.
				arg0 = ir.NewStarExpr(arg0.Pos(), arg0)
			}
			args = append(args, arg0)
		} else {
			args = append(args, formalParams[i].Nname.(*ir.Name))
		}
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

	// We're all done with the captured dictionary (and receiver, for method values).
	ir.FinishCaptureNames(pos, outer, fn)

	// Make a closure referencing our new internal function.
	c := ir.UseClosure(fn.OClosure, g.target)
	var init []ir.Node
	if outer != nil {
		init = append(init, dictAssign)
	}
	if rcvrValue != nil {
		init = append(init, rcvrAssign)
	}
	return ir.InitExpr(init, c)
}

// instantiateMethods instantiates all the methods (and associated dictionaries) of
// all fully-instantiated generic types that have been added to typecheck.instTypeList.
// It continues until no more types are added to typecheck.instTypeList.
func (g *irgen) instantiateMethods() {
	for {
		instTypeList := typecheck.GetInstTypeList()
		if len(instTypeList) == 0 {
			break
		}
		for _, typ := range instTypeList {
			assert(!typ.HasShape())
			// Mark runtime type as needed, since this ensures that the
			// compiler puts out the needed DWARF symbols, when this
			// instantiated type has a different package from the local
			// package.
			typecheck.NeedRuntimeType(typ)
			// Lookup the method on the base generic type, since methods may
			// not be set on imported instantiated types.
			baseSym := typ.OrigSym()
			baseType := baseSym.Def.(*ir.Name).Type()
			for j, _ := range typ.Methods().Slice() {
				if baseType.Methods().Slice()[j].Nointerface() {
					typ.Methods().Slice()[j].SetNointerface(true)
				}
				baseNname := baseType.Methods().Slice()[j].Nname.(*ir.Name)
				// Eagerly generate the instantiations and dictionaries that implement these methods.
				// We don't use the instantiations here, just generate them (and any
				// further instantiations those generate, etc.).
				// Note that we don't set the Func for any methods on instantiated
				// types. Their signatures don't match so that would be confusing.
				// Direct method calls go directly to the instantiations, implemented above.
				// Indirect method calls use wrappers generated in reflectcall. Those wrappers
				// will use these instantiations if they are needed (for interface tables or reflection).
				_ = g.getInstantiation(baseNname, typ.RParams(), true)
				_ = g.getDictionarySym(baseNname, typ.RParams(), true)
			}
		}
	}
}

// getInstNameNode returns the name node for the method or function being instantiated, and a bool which is true if a method is being instantiated.
func (g *irgen) getInstNameNode(inst *ir.InstExpr) (*ir.Name, bool) {
	if meth, ok := inst.X.(*ir.SelectorExpr); ok {
		return meth.Selection.Nname.(*ir.Name), true
	} else {
		return inst.X.(*ir.Name), false
	}
}

// getDictOrSubdict returns, for a method/function call or reference (node n) in an
// instantiation (described by instInfo), a node which is accessing a sub-dictionary
// or main/static dictionary, as needed, and also returns a boolean indicating if a
// sub-dictionary was accessed. nameNode is the particular function or method being
// called/referenced, and targs are the type arguments.
func (g *irgen) getDictOrSubdict(declInfo *instInfo, n ir.Node, nameNode *ir.Name, targs []*types.Type, isMeth bool) (ir.Node, bool) {
	var dict ir.Node
	usingSubdict := false
	if declInfo != nil {
		// Get the dictionary arg via sub-dictionary reference
		entry, ok := declInfo.dictEntryMap[n]
		// If the entry is not found, it may be that this node did not have
		// any type args that depend on type params, so we need a main
		// dictionary, not a sub-dictionary.
		if ok {
			dict = getDictionaryEntry(n.Pos(), declInfo.dictParam, entry, declInfo.dictLen)
			usingSubdict = true
		}
	}
	if !usingSubdict {
		dict = g.getDictionaryValue(nameNode, targs, isMeth)
	}
	return dict, usingSubdict
}

// checkFetchBody checks if a generic body can be fetched, but hasn't been loaded
// yet. If so, it imports the body.
func checkFetchBody(nameNode *ir.Name) {
	if nameNode.Func.Body == nil && nameNode.Func.Inl != nil {
		// If there is no body yet but Func.Inl exists, then we can can
		// import the whole generic body.
		assert(nameNode.Func.Inl.Cost == 1 && nameNode.Sym().Pkg != types.LocalPkg)
		typecheck.ImportBody(nameNode.Func)
		assert(nameNode.Func.Inl.Body != nil)
		nameNode.Func.Body = nameNode.Func.Inl.Body
		nameNode.Func.Dcl = nameNode.Func.Inl.Dcl
	}
}

// getInstantiation gets the instantiantion and dictionary of the function or method nameNode
// with the type arguments shapes. If the instantiated function is not already
// cached, then it calls genericSubst to create the new instantiation.
func (g *irgen) getInstantiation(nameNode *ir.Name, shapes []*types.Type, isMeth bool) *ir.Func {
	checkFetchBody(nameNode)

	// Convert any non-shape type arguments to their shape, so we can reduce the
	// number of instantiations we have to generate. You can actually have a mix
	// of shape and non-shape arguments, because of inferred or explicitly
	// specified concrete type args.
	var s1 []*types.Type
	for i, t := range shapes {
		if !t.HasShape() {
			if s1 == nil {
				s1 = make([]*types.Type, len(shapes))
				copy(s1[0:i], shapes[0:i])
			}
			s1[i] = typecheck.Shapify(t)
		} else if s1 != nil {
			s1[i] = shapes[i]
		}
	}
	if s1 != nil {
		shapes = s1
	}

	sym := typecheck.MakeFuncInstSym(nameNode.Sym(), shapes, isMeth)
	info := g.instInfoMap[sym]
	if info == nil {
		// If instantiation doesn't exist yet, create it and add
		// to the list of decls.
		gfInfo := g.getGfInfo(nameNode)
		info = &instInfo{
			gf:            nameNode,
			gfInfo:        gfInfo,
			startSubDict:  len(shapes) + len(gfInfo.derivedTypes),
			startItabConv: len(shapes) + len(gfInfo.derivedTypes) + len(gfInfo.subDictCalls),
			dictLen:       len(shapes) + len(gfInfo.derivedTypes) + len(gfInfo.subDictCalls) + len(gfInfo.itabConvs),
			dictEntryMap:  make(map[ir.Node]int),
		}
		// genericSubst fills in info.dictParam and info.dictEntryMap.
		st := g.genericSubst(sym, nameNode, shapes, isMeth, info)
		info.fun = st
		g.instInfoMap[sym] = info
		// This ensures that the linker drops duplicates of this instantiation.
		// All just works!
		st.SetDupok(true)
		g.target.Decls = append(g.target.Decls, st)
		if base.Flag.W > 1 {
			ir.Dump(fmt.Sprintf("\nstenciled %v", st), st)
		}
	}
	return info.fun
}

// Struct containing info needed for doing the substitution as we create the
// instantiation of a generic function with specified type arguments.
type subster struct {
	g        *irgen
	isMethod bool     // If a method is being instantiated
	newf     *ir.Func // Func node for the new stenciled function
	ts       typecheck.Tsubster
	info     *instInfo // Place to put extra info in the instantiation

	// Map from non-nil, non-ONAME node n to slice of all m, where m.Defn = n
	defnMap map[ir.Node][]**ir.Name
}

// genericSubst returns a new function with name newsym. The function is an
// instantiation of a generic function or method specified by namedNode with type
// args shapes. For a method with a generic receiver, it returns an instantiated
// function type where the receiver becomes the first parameter. For either a generic
// method or function, a dictionary parameter is the added as the very first
// parameter. genericSubst fills in info.dictParam and info.dictEntryMap.
func (g *irgen) genericSubst(newsym *types.Sym, nameNode *ir.Name, shapes []*types.Type, isMethod bool, info *instInfo) *ir.Func {
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

	assert(len(tparams) == len(shapes))

	subst := &subster{
		g:        g,
		isMethod: isMethod,
		newf:     newf,
		info:     info,
		ts: typecheck.Tsubster{
			Tparams: tparams,
			Targs:   shapes,
			Vars:    make(map[*ir.Name]*ir.Name),
		},
		defnMap: make(map[ir.Node][]**ir.Name),
	}

	newf.Dcl = make([]*ir.Name, 0, len(gf.Dcl)+1)

	// Create the needed dictionary param
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
	info.dictParam = dictionaryName

	// We add the dictionary as the first parameter in the function signature.
	// We also transform a method type to the corresponding function type
	// (make the receiver be the next parameter after the dictionary).
	oldt := nameNode.Type()
	var args []*types.Field
	args = append(args, dictionaryArg)
	args = append(args, oldt.Recvs().FieldSlice()...)
	args = append(args, oldt.Params().FieldSlice()...)

	// Replace the types in the function signature via subst.fields.
	// Ugly: also, we have to insert the Name nodes of the parameters/results into
	// the function type. The current function type has no Nname fields set,
	// because it came via conversion from the types2 type.
	newt := types.NewSignature(oldt.Pkg(), nil, nil,
		subst.fields(ir.PPARAM, args, newf.Dcl),
		subst.fields(ir.PPARAMOUT, oldt.Results().FieldSlice(), newf.Dcl))

	typed(newt, newf.Nname)
	ir.MarkFunc(newf.Nname)
	newf.SetTypecheck(1)

	// Make sure name/type of newf is set before substituting the body.
	newf.Body = subst.list(gf.Body)

	// Add code to check that the dictionary is correct.
	// TODO: must be adjusted to deal with shapes, but will go away soon when we move
	// to many->1 shape to concrete mapping.
	// newf.Body.Prepend(subst.checkDictionary(dictionaryName, shapes)...)

	if len(subst.defnMap) > 0 {
		base.Fatalf("defnMap is not empty")
	}

	ir.CurFunc = savef

	if doubleCheck {
		ir.Visit(newf, func(n ir.Node) {
			if n.Op() != ir.OCONVIFACE {
				return
			}
			c := n.(*ir.ConvExpr)
			if c.X.Type().HasShape() && !c.X.Type().IsInterface() {
				ir.Dump("BAD FUNCTION", newf)
				ir.Dump("BAD CONVERSION", c)
				base.Fatalf("converting shape type to interface")
			}
		})
	}

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
	if name.Defn != nil {
		if name.Defn.Op() == ir.ONAME {
			// This is a closure variable, so its Defn is the outer
			// captured variable, which has already been substituted.
			m.Defn = subst.node(name.Defn)
		} else {
			// The other values of Defn are nodes in the body of the
			// function, so just remember the mapping so we can set Defn
			// properly in node() when we create the new body node. We
			// always call localvar() on all the local variables before
			// we substitute the body.
			slice := subst.defnMap[name.Defn]
			subst.defnMap[name.Defn] = append(slice, &m)
		}
	}
	if name.Outer != nil {
		m.Outer = subst.node(name.Outer).(*ir.Name)
	}

	return m
}

// checkDictionary returns code that does runtime consistency checks
// between the dictionary and the types it should contain.
func (subst *subster) checkDictionary(name *ir.Name, targs []*types.Type) (code []ir.Node) {
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
	types.CheckSize(d.Type().Elem())

	// Check that each type entry in the dictionary is correct.
	for i, t := range targs {
		if t.HasShape() {
			// Check the concrete type, not the shape type.
			base.Fatalf("shape type in dictionary %s %+v\n", name.Sym().Name, t)
		}
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

// getDictionaryEntry gets the i'th entry in the dictionary dict.
func getDictionaryEntry(pos src.XPos, dict *ir.Name, i int, size int) ir.Node {
	// Convert dictionary to *[N]uintptr
	// All entries in the dictionary are pointers. They all point to static data, though, so we
	// treat them as uintptrs so the GC doesn't need to keep track of them.
	d := ir.NewConvExpr(pos, ir.OCONVNOP, types.Types[types.TUNSAFEPTR], dict)
	d.SetTypecheck(1)
	d = ir.NewConvExpr(pos, ir.OCONVNOP, types.NewArray(types.Types[types.TUINTPTR], int64(size)).PtrTo(), d)
	d.SetTypecheck(1)
	types.CheckSize(d.Type().Elem())

	// Load entry i out of the dictionary.
	deref := ir.NewStarExpr(pos, d)
	typed(d.Type().Elem(), deref)
	idx := ir.NewConstExpr(constant.MakeUint64(uint64(i)), dict) // TODO: what to set orig to?
	typed(types.Types[types.TUINTPTR], idx)
	r := ir.NewIndexExpr(pos, deref, idx)
	typed(types.Types[types.TUINTPTR], r)
	return r
}

// getDictionaryType returns a *runtime._type from the dictionary entry i (which
// refers to a type param or a derived type that uses type params). It uses the
// specified dictionary dictParam, rather than the one in info.dictParam.
func getDictionaryType(info *instInfo, dictParam *ir.Name, pos src.XPos, i int) ir.Node {
	if i < 0 || i >= info.startSubDict {
		base.Fatalf(fmt.Sprintf("bad dict index %d", i))
	}

	r := getDictionaryEntry(pos, info.dictParam, i, info.startSubDict)
	// change type of retrieved dictionary entry to *byte, which is the
	// standard typing of a *runtime._type in the compiler
	typed(types.Types[types.TUINT8].PtrTo(), r)
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

		slice, ok := subst.defnMap[x]
		if ok {
			// We just copied a non-ONAME node which was the Defn value
			// of a local variable. Set the Defn value of the copied
			// local variable to this new Defn node.
			for _, ptr := range slice {
				(*ptr).Defn = m
			}
			delete(subst.defnMap, x)
		}

		if _, isExpr := m.(ir.Expr); isExpr {
			t := x.Type()
			if t == nil {
				// t can be nil only if this is a call that has no
				// return values, so allow that and otherwise give
				// an error.
				_, isCallExpr := m.(*ir.CallExpr)
				_, isStructKeyExpr := m.(*ir.StructKeyExpr)
				_, isKeyExpr := m.(*ir.KeyExpr)
				if !isCallExpr && !isStructKeyExpr && !isKeyExpr && x.Op() != ir.OPANIC &&
					x.Op() != ir.OCLOSE {
					base.Fatalf(fmt.Sprintf("Nil type for %v", x))
				}
			} else if x.Op() != ir.OCLOSURE {
				m.SetType(subst.ts.Typ(x.Type()))
			}
		}

		for i, de := range subst.info.gfInfo.subDictCalls {
			if de == x {
				// Remember the dictionary entry associated with this
				// node in the instantiated function
				// TODO: make sure this remains correct with respect to the
				// transformations below.
				subst.info.dictEntryMap[m] = subst.info.startSubDict + i
				break
			}
		}

		ir.EditChildren(m, edit)

		m.SetTypecheck(1)
		if x.Op().IsCmp() {
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
					as.X, as.Y = lhs[0], rhs[0]
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
			// param. m will be transformed to an OMETHVALUE node. It
			// will be transformed to an ODOTMETH or ODOTINTER node if
			// we find in the OCALL case below that the method value
			// is actually called.
			mse := m.(*ir.SelectorExpr)
			if src := mse.X.Type(); src.IsShape() {
				// The only dot on a shape type value are methods.
				if mse.X.Op() == ir.OTYPE {
					// Method expression T.M
					m = subst.g.buildClosure2(subst, m, x)
					// No need for transformDot - buildClosure2 has already
					// transformed to OCALLINTER/ODOTINTER.
				} else {
					// Implement x.M as a conversion-to-bound-interface
					//  1) convert x to the bound interface
					//  2) call M on that interface
					gsrc := x.(*ir.SelectorExpr).X.Type()
					bound := gsrc.Bound()
					dst := bound
					if dst.HasTParam() {
						dst = subst.ts.Typ(dst)
					}
					if src.IsInterface() {
						// If type arg is an interface (unusual case),
						// we do a type assert to the type bound.
						mse.X = assertToBound(subst.info, subst.info.dictParam, m.Pos(), mse.X, bound, dst)
					} else {
						mse.X = convertUsingDictionary(subst.info, subst.info.dictParam, m.Pos(), mse.X, x, dst, gsrc)
					}
					transformDot(mse, false)
				}
			} else {
				transformDot(mse, false)
			}
			m.SetTypecheck(1)

		case ir.OCALL:
			call := m.(*ir.CallExpr)
			switch call.X.Op() {
			case ir.OTYPE:
				// Transform the conversion, now that we know the
				// type argument.
				m = transformConvCall(call)
				// CONVIFACE transformation was already done in node2
				assert(m.Op() != ir.OCONVIFACE)

			case ir.OMETHVALUE, ir.OMETHEXPR:
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
					case ir.OMAKE, ir.OREAL, ir.OIMAG, ir.OAPPEND, ir.ODELETE, ir.OALIGNOF, ir.OOFFSETOF, ir.OSIZEOF:
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

			case ir.ODEREF, ir.OINDEX, ir.OINDEXMAP, ir.ORECV:
				// Transform a call that was delayed because of the
				// use of typeparam inside an expression that required
				// a pointer dereference, array indexing, map indexing,
				// or channel receive to compute function value.
				transformCall(call)

			case ir.OCALL, ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER, ir.ODYNAMICDOTTYPE:
				transformCall(call)

			case ir.OFUNCINST:
				// A call with an OFUNCINST will get transformed
				// in stencil() once we have created & attached the
				// instantiation to be called.

			default:
				base.FatalfAt(call.Pos(), fmt.Sprintf("Unexpected op with CALL during stenciling: %v", call.X.Op()))
			}

		case ir.OCLOSURE:
			// We're going to create a new closure from scratch, so clear m
			// to avoid using the ir.Copy by accident until we reassign it.
			m = nil

			x := x.(*ir.ClosureExpr)
			// Need to duplicate x.Func.Nname, x.Func.Dcl, x.Func.ClosureVars, and
			// x.Func.Body.
			oldfn := x.Func
			newfn := ir.NewClosureFunc(oldfn.Pos(), subst.newf != nil)
			ir.NameClosure(newfn.OClosure, subst.newf)

			saveNewf := subst.newf
			ir.CurFunc = newfn
			subst.newf = newfn
			newfn.Dcl = subst.namelist(oldfn.Dcl)

			// Make a closure variable for the dictionary of the
			// containing function.
			cdict := ir.CaptureName(oldfn.Pos(), newfn, subst.info.dictParam)
			typed(types.Types[types.TUINTPTR], cdict)
			ir.FinishCaptureNames(oldfn.Pos(), saveNewf, newfn)
			newfn.ClosureVars = append(newfn.ClosureVars, subst.namelist(oldfn.ClosureVars)...)

			// Copy that closure variable to a local one.
			// Note: this allows the dictionary to be captured by child closures.
			// See issue 47723.
			ldict := ir.NewNameAt(x.Pos(), subst.info.gf.Sym().Pkg.Lookup(".dict"))
			typed(types.Types[types.TUINTPTR], ldict)
			ldict.Class = ir.PAUTO
			ldict.Curfn = newfn
			newfn.Dcl = append(newfn.Dcl, ldict)
			as := ir.NewAssignStmt(x.Pos(), ldict, cdict)
			as.SetTypecheck(1)
			newfn.Body.Append(as)

			// Create inst info for the instantiated closure. The dict
			// param is the closure variable for the dictionary of the
			// outer function. Since the dictionary is shared, use the
			// same entries for startSubDict, dictLen, dictEntryMap.
			cinfo := &instInfo{
				fun:           newfn,
				dictParam:     ldict,
				gf:            subst.info.gf,
				gfInfo:        subst.info.gfInfo,
				startSubDict:  subst.info.startSubDict,
				startItabConv: subst.info.startItabConv,
				dictLen:       subst.info.dictLen,
				dictEntryMap:  subst.info.dictEntryMap,
			}
			subst.g.instInfoMap[newfn.Nname.Sym()] = cinfo

			typed(subst.ts.Typ(oldfn.Nname.Type()), newfn.Nname)
			typed(newfn.Nname.Type(), newfn.OClosure)
			newfn.SetTypecheck(1)

			outerinfo := subst.info
			subst.info = cinfo
			// Make sure type of closure function is set before doing body.
			newfn.Body.Append(subst.list(oldfn.Body)...)
			subst.info = outerinfo
			subst.newf = saveNewf
			ir.CurFunc = saveNewf

			m = ir.UseClosure(newfn.OClosure, subst.g.target)
			m.(*ir.ClosureExpr).SetInit(subst.list(x.Init()))

		case ir.OCONVIFACE:
			x := x.(*ir.ConvExpr)
			if m.Type().IsEmptyInterface() && m.(*ir.ConvExpr).X.Type().IsEmptyInterface() {
				// Was T->interface{}, after stenciling it is now interface{}->interface{}.
				// No longer need the conversion. See issue 48276.
				m.(*ir.ConvExpr).SetOp(ir.OCONVNOP)
				break
			}
			// Note: x's argument is still typed as a type parameter.
			// m's argument now has an instantiated type.
			if x.X.Type().HasTParam() || (x.X.Type().IsInterface() && x.Type().HasTParam()) {
				m = convertUsingDictionary(subst.info, subst.info.dictParam, m.Pos(), m.(*ir.ConvExpr).X, x, m.Type(), x.X.Type())
			}
		case ir.ODOTTYPE, ir.ODOTTYPE2:
			if !x.Type().HasTParam() {
				break
			}
			dt := m.(*ir.TypeAssertExpr)
			var rt ir.Node
			if dt.Type().IsInterface() || dt.X.Type().IsEmptyInterface() {
				ix := findDictType(subst.info, x.Type())
				assert(ix >= 0)
				rt = getDictionaryType(subst.info, subst.info.dictParam, dt.Pos(), ix)
			} else {
				// nonempty interface to noninterface. Need an itab.
				ix := -1
				for i, ic := range subst.info.gfInfo.itabConvs {
					if ic == x {
						ix = subst.info.startItabConv + i
						break
					}
				}
				assert(ix >= 0)
				rt = getDictionaryEntry(dt.Pos(), subst.info.dictParam, ix, subst.info.dictLen)
			}
			op := ir.ODYNAMICDOTTYPE
			if x.Op() == ir.ODOTTYPE2 {
				op = ir.ODYNAMICDOTTYPE2
			}
			m = ir.NewDynamicTypeAssertExpr(dt.Pos(), op, dt.X, rt)
			m.SetType(dt.Type())
			m.SetTypecheck(1)
		case ir.OCASE:
			if _, ok := x.(*ir.CommClause); ok {
				// This is not a type switch. TODO: Should we use an OSWITCH case here instead of OCASE?
				break
			}
			x := x.(*ir.CaseClause)
			m := m.(*ir.CaseClause)
			for i, c := range x.List {
				if c.Op() == ir.OTYPE && c.Type().HasTParam() {
					// Use a *runtime._type for the dynamic type.
					ix := findDictType(subst.info, c.Type())
					assert(ix >= 0)
					dt := ir.NewDynamicType(c.Pos(), getDictionaryEntry(c.Pos(), subst.info.dictParam, ix, subst.info.dictLen))

					// For type switch from nonempty interfaces to non-interfaces, we need an itab as well.
					if !m.List[i].Type().IsInterface() {
						if _, ok := subst.info.gfInfo.type2switchType[c]; ok {
							// Type switch from nonempty interface. We need a *runtime.itab
							// for the dynamic type.
							ix := -1
							for i, ic := range subst.info.gfInfo.itabConvs {
								if ic == c {
									ix = subst.info.startItabConv + i
									break
								}
							}
							assert(ix >= 0)
							dt.ITab = getDictionaryEntry(c.Pos(), subst.info.dictParam, ix, subst.info.dictLen)
						}
					}
					typed(m.List[i].Type(), dt)
					m.List[i] = dt
				}
			}
		}
		return m
	}

	return edit(n)
}

// findDictType looks for type t in the typeparams or derived types in the generic
// function info.gfInfo. This will indicate the dictionary entry with the
// correct concrete type for the associated instantiated function.
func findDictType(info *instInfo, t *types.Type) int {
	for i, dt := range info.gfInfo.tparams {
		if dt == t {
			return i
		}
	}
	for i, dt := range info.gfInfo.derivedTypes {
		if types.Identical(dt, t) {
			return i + len(info.gfInfo.tparams)
		}
	}
	return -1
}

// convertUsingDictionary converts value v from instantiated type src to an interface
// type dst, by returning a new set of nodes that make use of a dictionary entry. src
// is the generic (not shape) type, and gn is the original generic node of the
// CONVIFACE node or XDOT node (for a bound method call) that is causing the
// conversion.
func convertUsingDictionary(info *instInfo, dictParam *ir.Name, pos src.XPos, v ir.Node, gn ir.Node, dst, src *types.Type) ir.Node {
	assert(src.HasTParam() || src.IsInterface() && gn.Type().HasTParam())
	assert(dst.IsInterface())

	if v.Type().IsInterface() {
		// Converting from an interface. The shape-ness of the source doesn't really matter, as
		// we'll be using the concrete type from the first interface word.
		if dst.IsEmptyInterface() {
			// Converting I2E. OCONVIFACE does that for us, and doesn't depend
			// on what the empty interface was instantiated with. No dictionary entry needed.
			v = ir.NewConvExpr(pos, ir.OCONVIFACE, dst, v)
			v.SetTypecheck(1)
			return v
		}
		gdst := gn.Type() // pre-stenciled destination type
		if !gdst.HasTParam() {
			// Regular OCONVIFACE works if the destination isn't parameterized.
			v = ir.NewConvExpr(pos, ir.OCONVIFACE, dst, v)
			v.SetTypecheck(1)
			return v
		}

		// We get the destination interface type from the dictionary and the concrete
		// type from the argument's itab. Call runtime.convI2I to get the new itab.
		tmp := typecheck.Temp(v.Type())
		as := ir.NewAssignStmt(pos, tmp, v)
		as.SetTypecheck(1)
		itab := ir.NewUnaryExpr(pos, ir.OITAB, tmp)
		typed(types.Types[types.TUINTPTR].PtrTo(), itab)
		idata := ir.NewUnaryExpr(pos, ir.OIDATA, tmp)
		typed(types.Types[types.TUNSAFEPTR], idata)

		fn := typecheck.LookupRuntime("convI2I")
		fn.SetTypecheck(1)
		types.CalcSize(fn.Type())
		call := ir.NewCallExpr(pos, ir.OCALLFUNC, fn, nil)
		typed(types.Types[types.TUINT8].PtrTo(), call)
		ix := findDictType(info, gdst)
		assert(ix >= 0)
		inter := getDictionaryType(info, dictParam, pos, ix)
		call.Args = []ir.Node{inter, itab}
		i := ir.NewBinaryExpr(pos, ir.OEFACE, call, idata)
		typed(dst, i)
		i.PtrInit().Append(as)
		return i
	}

	var rt ir.Node
	if !dst.IsEmptyInterface() {
		// We should have an itab entry in the dictionary. Using this itab
		// will be more efficient than converting to an empty interface first
		// and then type asserting to dst.
		ix := -1
		for i, ic := range info.gfInfo.itabConvs {
			if ic == gn {
				ix = info.startItabConv + i
				break
			}
		}
		assert(ix >= 0)
		rt = getDictionaryEntry(pos, dictParam, ix, info.dictLen)
	} else {
		ix := findDictType(info, src)
		assert(ix >= 0)
		// Load the actual runtime._type of the type parameter from the dictionary.
		rt = getDictionaryType(info, dictParam, pos, ix)
	}

	// Figure out what the data field of the interface will be.
	data := ir.NewConvExpr(pos, ir.OCONVIDATA, nil, v)
	typed(types.Types[types.TUNSAFEPTR], data)

	// Build an interface from the type and data parts.
	var i ir.Node = ir.NewBinaryExpr(pos, ir.OEFACE, rt, data)
	typed(dst, i)
	return i
}

func (subst *subster) namelist(l []*ir.Name) []*ir.Name {
	s := make([]*ir.Name, len(l))
	for i, n := range l {
		s[i] = subst.localvar(n)
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

// deref does a single deref of type t, if it is a pointer type.
func deref(t *types.Type) *types.Type {
	if t.IsPtr() {
		return t.Elem()
	}
	return t
}

// markTypeUsed marks type t as used in order to help avoid dead-code elimination of
// needed methods.
func markTypeUsed(t *types.Type, lsym *obj.LSym) {
	if t.IsInterface() {
		// Mark all the methods of the interface as used.
		// TODO: we should really only mark the interface methods
		// that are actually called in the application.
		for i, _ := range t.AllMethods().Slice() {
			reflectdata.MarkUsedIfaceMethodIndex(lsym, t, i)
		}
	} else {
		// TODO: This is somewhat overkill, we really only need it
		// for types that are put into interfaces.
		reflectdata.MarkTypeUsedInInterface(t, lsym)
	}
}

// getDictionarySym returns the dictionary for the named generic function gf, which
// is instantiated with the type arguments targs.
func (g *irgen) getDictionarySym(gf *ir.Name, targs []*types.Type, isMeth bool) *types.Sym {
	if len(targs) == 0 {
		base.Fatalf("%s should have type arguments", gf.Sym().Name)
	}

	// Enforce that only concrete types can make it to here.
	for _, t := range targs {
		if t.HasShape() {
			panic(fmt.Sprintf("shape %+v in dictionary for %s", t, gf.Sym().Name))
		}
	}

	// Get a symbol representing the dictionary.
	sym := typecheck.MakeDictSym(gf.Sym(), targs, isMeth)

	// Initialize the dictionary, if we haven't yet already.
	lsym := sym.Linksym()
	if len(lsym.P) > 0 {
		// We already started creating this dictionary and its lsym.
		return sym
	}

	info := g.getGfInfo(gf)

	infoPrint("=== Creating dictionary %v\n", sym.Name)
	off := 0
	// Emit an entry for each targ (concrete type or gcshape).
	for _, t := range targs {
		infoPrint(" * %v\n", t)
		s := reflectdata.TypeLinksym(t)
		off = objw.SymPtr(lsym, off, s, 0)
		markTypeUsed(t, lsym)
	}
	subst := typecheck.Tsubster{
		Tparams: info.tparams,
		Targs:   targs,
	}
	// Emit an entry for each derived type (after substituting targs)
	for _, t := range info.derivedTypes {
		ts := subst.Typ(t)
		infoPrint(" - %v\n", ts)
		s := reflectdata.TypeLinksym(ts)
		off = objw.SymPtr(lsym, off, s, 0)
		markTypeUsed(ts, lsym)
	}
	// Emit an entry for each subdictionary (after substituting targs)
	for _, n := range info.subDictCalls {
		var sym *types.Sym
		switch n.Op() {
		case ir.OCALL:
			call := n.(*ir.CallExpr)
			if call.X.Op() == ir.OXDOT {
				var nameNode *ir.Name
				se := call.X.(*ir.SelectorExpr)
				if types.IsInterfaceMethod(se.Selection.Type) {
					// This is a method call enabled by a type bound.
					tmpse := ir.NewSelectorExpr(base.Pos, ir.OXDOT, se.X, se.Sel)
					tmpse = typecheck.AddImplicitDots(tmpse)
					tparam := tmpse.X.Type()
					assert(tparam.IsTypeParam())
					recvType := targs[tparam.Index()]
					if recvType.IsInterface() || len(recvType.RParams()) == 0 {
						// No sub-dictionary entry is
						// actually needed, since the
						// type arg is not an
						// instantiated type that
						// will have generic methods.
						break
					}
					// This is a method call for an
					// instantiated type, so we need a
					// sub-dictionary.
					targs := recvType.RParams()
					genRecvType := recvType.OrigSym().Def.Type()
					nameNode = typecheck.Lookdot1(call.X, se.Sel, genRecvType, genRecvType.Methods(), 1).Nname.(*ir.Name)
					sym = g.getDictionarySym(nameNode, targs, true)
				} else {
					// This is the case of a normal
					// method call on a generic type.
					nameNode = call.X.(*ir.SelectorExpr).Selection.Nname.(*ir.Name)
					subtargs := deref(call.X.(*ir.SelectorExpr).X.Type()).RParams()
					s2targs := make([]*types.Type, len(subtargs))
					for i, t := range subtargs {
						s2targs[i] = subst.Typ(t)
					}
					sym = g.getDictionarySym(nameNode, s2targs, true)
				}
			} else {
				inst := call.X.(*ir.InstExpr)
				var nameNode *ir.Name
				var meth *ir.SelectorExpr
				var isMeth bool
				if meth, isMeth = inst.X.(*ir.SelectorExpr); isMeth {
					nameNode = meth.Selection.Nname.(*ir.Name)
				} else {
					nameNode = inst.X.(*ir.Name)
				}
				subtargs := typecheck.TypesOf(inst.Targs)
				for i, t := range subtargs {
					subtargs[i] = subst.Typ(t)
				}
				sym = g.getDictionarySym(nameNode, subtargs, isMeth)
			}

		case ir.OFUNCINST:
			inst := n.(*ir.InstExpr)
			nameNode := inst.X.(*ir.Name)
			subtargs := typecheck.TypesOf(inst.Targs)
			for i, t := range subtargs {
				subtargs[i] = subst.Typ(t)
			}
			sym = g.getDictionarySym(nameNode, subtargs, false)

		case ir.OXDOT:
			selExpr := n.(*ir.SelectorExpr)
			subtargs := deref(selExpr.X.Type()).RParams()
			s2targs := make([]*types.Type, len(subtargs))
			for i, t := range subtargs {
				s2targs[i] = subst.Typ(t)
			}
			nameNode := selExpr.Selection.Nname.(*ir.Name)
			sym = g.getDictionarySym(nameNode, s2targs, true)

		default:
			assert(false)
		}

		if sym == nil {
			// Unused sub-dictionary entry, just emit 0.
			off = objw.Uintptr(lsym, off, 0)
			infoPrint(" - Unused subdict entry\n")
		} else {
			off = objw.SymPtr(lsym, off, sym.Linksym(), 0)
			infoPrint(" - Subdict %v\n", sym.Name)
		}
	}

	delay := &delayInfo{
		gf:    gf,
		targs: targs,
		sym:   sym,
		off:   off,
	}
	g.dictSymsToFinalize = append(g.dictSymsToFinalize, delay)
	return sym
}

// finalizeSyms finishes up all dictionaries on g.dictSymsToFinalize, by writing out
// any needed LSyms for itabs. The itab lsyms create wrappers which need various
// dictionaries and method instantiations to be complete, so, to avoid recursive
// dependencies, we finalize the itab lsyms only after all dictionaries syms and
// instantiations have been created.
func (g *irgen) finalizeSyms() {
	for _, d := range g.dictSymsToFinalize {
		infoPrint("=== Finalizing dictionary %s\n", d.sym.Name)

		lsym := d.sym.Linksym()
		info := g.getGfInfo(d.gf)

		subst := typecheck.Tsubster{
			Tparams: info.tparams,
			Targs:   d.targs,
		}

		// Emit an entry for each itab
		for _, n := range info.itabConvs {
			var srctype, dsttype *types.Type
			switch n.Op() {
			case ir.OXDOT:
				se := n.(*ir.SelectorExpr)
				srctype = subst.Typ(se.X.Type())
				dsttype = subst.Typ(se.X.Type().Bound())
				found := false
				for i, m := range dsttype.AllMethods().Slice() {
					if se.Sel == m.Sym {
						// Mark that this method se.Sel is
						// used for the dsttype interface, so
						// it won't get deadcoded.
						reflectdata.MarkUsedIfaceMethodIndex(lsym, dsttype, i)
						found = true
						break
					}
				}
				assert(found)
			case ir.ODOTTYPE, ir.ODOTTYPE2:
				srctype = subst.Typ(n.(*ir.TypeAssertExpr).Type())
				dsttype = subst.Typ(n.(*ir.TypeAssertExpr).X.Type())
			case ir.OCONVIFACE:
				srctype = subst.Typ(n.(*ir.ConvExpr).X.Type())
				dsttype = subst.Typ(n.Type())
			case ir.OTYPE:
				srctype = subst.Typ(n.Type())
				dsttype = subst.Typ(info.type2switchType[n])
			default:
				base.Fatalf("itab entry with unknown op %s", n.Op())
			}
			if srctype.IsInterface() || dsttype.IsEmptyInterface() {
				// No itab is wanted if src type is an interface. We
				// will use a type assert instead.
				d.off = objw.Uintptr(lsym, d.off, 0)
				infoPrint(" + Unused itab entry for %v\n", srctype)
			} else {
				itabLsym := reflectdata.ITabLsym(srctype, dsttype)
				d.off = objw.SymPtr(lsym, d.off, itabLsym, 0)
				infoPrint(" + Itab for (%v,%v)\n", srctype, dsttype)
			}
		}

		objw.Global(lsym, int32(d.off), obj.DUPOK|obj.RODATA)
		infoPrint("=== Finalized dictionary %s\n", d.sym.Name)
	}
	g.dictSymsToFinalize = nil
}

func (g *irgen) getDictionaryValue(gf *ir.Name, targs []*types.Type, isMeth bool) ir.Node {
	sym := g.getDictionarySym(gf, targs, isMeth)

	// Make (or reuse) a node referencing the dictionary symbol.
	var n *ir.Name
	if sym.Def != nil {
		n = sym.Def.(*ir.Name)
	} else {
		n = typecheck.NewName(sym)
		n.SetType(types.Types[types.TUINTPTR]) // should probably be [...]uintptr, but doesn't really matter
		n.SetTypecheck(1)
		n.Class = ir.PEXTERN
		sym.Def = n
	}

	// Return the address of the dictionary.
	np := typecheck.NodAddr(n)
	// Note: treat dictionary pointers as uintptrs, so they aren't pointers
	// with respect to GC. That saves on stack scanning work, write barriers, etc.
	// We can get away with it because dictionaries are global variables.
	// TODO: use a cast, or is typing directly ok?
	np.SetType(types.Types[types.TUINTPTR])
	np.SetTypecheck(1)
	return np
}

// hasTParamNodes returns true if the type of any node in targs has a typeparam.
func hasTParamNodes(targs []ir.Node) bool {
	for _, n := range targs {
		if n.Type().HasTParam() {
			return true
		}
	}
	return false
}

// hasTParamNodes returns true if any type in targs has a typeparam.
func hasTParamTypes(targs []*types.Type) bool {
	for _, t := range targs {
		if t.HasTParam() {
			return true
		}
	}
	return false
}

// getGfInfo get information for a generic function - type params, derived generic
// types, and subdictionaries.
func (g *irgen) getGfInfo(gn *ir.Name) *gfInfo {
	infop := g.gfInfoMap[gn.Sym()]
	if infop != nil {
		return infop
	}

	checkFetchBody(gn)
	var info gfInfo
	gf := gn.Func
	recv := gf.Type().Recv()
	if recv != nil {
		info.tparams = deref(recv.Type).RParams()
	} else {
		tparams := gn.Type().TParams().FieldSlice()
		info.tparams = make([]*types.Type, len(tparams))
		for i, f := range tparams {
			info.tparams[i] = f.Type
		}
	}

	for _, t := range info.tparams {
		b := t.Bound()
		if b.HasTParam() {
			// If a type bound is parameterized (unusual case), then we
			// may need its derived type to do a type assert when doing a
			// bound call for a type arg that is an interface.
			addType(&info, nil, b)
		}
	}

	for _, n := range gf.Dcl {
		addType(&info, n, n.Type())
	}

	if infoPrintMode {
		fmt.Printf(">>> GfInfo for %v\n", gn)
		for _, t := range info.tparams {
			fmt.Printf("  Typeparam %v\n", t)
		}
	}

	var visitFunc func(ir.Node)
	visitFunc = func(n ir.Node) {
		if n.Op() == ir.OFUNCINST && !n.(*ir.InstExpr).Implicit() {
			if hasTParamNodes(n.(*ir.InstExpr).Targs) {
				infoPrint("  Closure&subdictionary required at generic function value %v\n", n.(*ir.InstExpr).X)
				info.subDictCalls = append(info.subDictCalls, n)
			}
		} else if n.Op() == ir.OXDOT && !n.(*ir.SelectorExpr).Implicit() &&
			n.(*ir.SelectorExpr).Selection != nil &&
			len(deref(n.(*ir.SelectorExpr).X.Type()).RParams()) > 0 {
			if hasTParamTypes(deref(n.(*ir.SelectorExpr).X.Type()).RParams()) {
				if n.(*ir.SelectorExpr).X.Op() == ir.OTYPE {
					infoPrint("  Closure&subdictionary required at generic meth expr %v\n", n)
				} else {
					infoPrint("  Closure&subdictionary required at generic meth value %v\n", n)
				}
				info.subDictCalls = append(info.subDictCalls, n)
			}
		}
		if n.Op() == ir.OCALL && n.(*ir.CallExpr).X.Op() == ir.OFUNCINST {
			n.(*ir.CallExpr).X.(*ir.InstExpr).SetImplicit(true)
			if hasTParamNodes(n.(*ir.CallExpr).X.(*ir.InstExpr).Targs) {
				infoPrint("  Subdictionary at generic function/method call: %v - %v\n", n.(*ir.CallExpr).X.(*ir.InstExpr).X, n)
				info.subDictCalls = append(info.subDictCalls, n)
			}
		}
		if n.Op() == ir.OCALL && n.(*ir.CallExpr).X.Op() == ir.OXDOT &&
			n.(*ir.CallExpr).X.(*ir.SelectorExpr).Selection != nil &&
			len(deref(n.(*ir.CallExpr).X.(*ir.SelectorExpr).X.Type()).RParams()) > 0 {
			n.(*ir.CallExpr).X.(*ir.SelectorExpr).SetImplicit(true)
			if hasTParamTypes(deref(n.(*ir.CallExpr).X.(*ir.SelectorExpr).X.Type()).RParams()) {
				infoPrint("  Subdictionary at generic method call: %v\n", n)
				info.subDictCalls = append(info.subDictCalls, n)
			}
		}
		if n.Op() == ir.OCALL && n.(*ir.CallExpr).X.Op() == ir.OXDOT &&
			n.(*ir.CallExpr).X.(*ir.SelectorExpr).Selection != nil &&
			deref(n.(*ir.CallExpr).X.(*ir.SelectorExpr).X.Type()).IsTypeParam() {
			n.(*ir.CallExpr).X.(*ir.SelectorExpr).SetImplicit(true)
			infoPrint("  Optional subdictionary at generic bound call: %v\n", n)
			info.subDictCalls = append(info.subDictCalls, n)
		}
		if n.Op() == ir.OCONVIFACE && n.Type().IsInterface() &&
			!n.Type().IsEmptyInterface() &&
			n.(*ir.ConvExpr).X.Type().HasTParam() {
			infoPrint("  Itab for interface conv: %v\n", n)
			info.itabConvs = append(info.itabConvs, n)
		}
		if n.Op() == ir.OXDOT && n.(*ir.SelectorExpr).X.Type().IsTypeParam() {
			infoPrint("  Itab for bound call: %v\n", n)
			info.itabConvs = append(info.itabConvs, n)
		}
		if (n.Op() == ir.ODOTTYPE || n.Op() == ir.ODOTTYPE2) && !n.(*ir.TypeAssertExpr).Type().IsInterface() && !n.(*ir.TypeAssertExpr).X.Type().IsEmptyInterface() {
			infoPrint("  Itab for dot type: %v\n", n)
			info.itabConvs = append(info.itabConvs, n)
		}
		if n.Op() == ir.OCLOSURE {
			// Visit the closure body and add all relevant entries to the
			// dictionary of the outer function (closure will just use
			// the dictionary of the outer function).
			for _, n1 := range n.(*ir.ClosureExpr).Func.Body {
				ir.Visit(n1, visitFunc)
			}
		}
		if n.Op() == ir.OSWITCH && n.(*ir.SwitchStmt).Tag != nil && n.(*ir.SwitchStmt).Tag.Op() == ir.OTYPESW && !n.(*ir.SwitchStmt).Tag.(*ir.TypeSwitchGuard).X.Type().IsEmptyInterface() {
			for _, cc := range n.(*ir.SwitchStmt).Cases {
				for _, c := range cc.List {
					if c.Op() == ir.OTYPE && c.Type().HasTParam() {
						// Type switch from a non-empty interface - might need an itab.
						infoPrint("  Itab for type switch: %v\n", c)
						info.itabConvs = append(info.itabConvs, c)
						if info.type2switchType == nil {
							info.type2switchType = map[ir.Node]*types.Type{}
						}
						info.type2switchType[c] = n.(*ir.SwitchStmt).Tag.(*ir.TypeSwitchGuard).X.Type()
					}
				}
			}
		}
		addType(&info, n, n.Type())
	}

	for _, stmt := range gf.Body {
		ir.Visit(stmt, visitFunc)
	}
	if infoPrintMode {
		for _, t := range info.derivedTypes {
			fmt.Printf("  Derived type %v\n", t)
		}
		fmt.Printf(">>> Done Gfinfo\n")
	}
	g.gfInfoMap[gn.Sym()] = &info
	return &info
}

// addType adds t to info.derivedTypes if it is parameterized type (which is not
// just a simple type param) that is different from any existing type on
// info.derivedTypes.
func addType(info *gfInfo, n ir.Node, t *types.Type) {
	if t == nil || !t.HasTParam() {
		return
	}
	if t.IsTypeParam() && t.Underlying() == t {
		return
	}
	if t.Kind() == types.TFUNC && n != nil &&
		(t.Recv() != nil ||
			n.Op() == ir.ONAME && n.Name().Class == ir.PFUNC) {
		// Don't use the type of a named generic function or method,
		// since that is parameterized by other typeparams.
		// (They all come from arguments of a FUNCINST node.)
		return
	}
	if doubleCheck && !parameterizedBy(t, info.tparams) {
		base.Fatalf("adding type with invalid parameters %+v", t)
	}
	if t.Kind() == types.TSTRUCT && t.IsFuncArgStruct() {
		// Multiple return values are not a relevant new type (?).
		return
	}
	// Ignore a derived type we've already added.
	for _, et := range info.derivedTypes {
		if types.Identical(t, et) {
			return
		}
	}
	info.derivedTypes = append(info.derivedTypes, t)
}

// parameterizedBy returns true if t is parameterized by (at most) params.
func parameterizedBy(t *types.Type, params []*types.Type) bool {
	return parameterizedBy1(t, params, map[*types.Type]bool{})
}
func parameterizedBy1(t *types.Type, params []*types.Type, visited map[*types.Type]bool) bool {
	if visited[t] {
		return true
	}
	visited[t] = true

	if t.Sym() != nil && len(t.RParams()) > 0 {
		// This defined type is instantiated. Check the instantiating types.
		for _, r := range t.RParams() {
			if !parameterizedBy1(r, params, visited) {
				return false
			}
		}
		return true
	}
	switch t.Kind() {
	case types.TTYPEPARAM:
		// Check if t is one of the allowed parameters in scope.
		for _, p := range params {
			if p == t {
				return true
			}
		}
		// Couldn't find t in the list of allowed parameters.
		return false

	case types.TARRAY, types.TPTR, types.TSLICE, types.TCHAN:
		return parameterizedBy1(t.Elem(), params, visited)

	case types.TMAP:
		return parameterizedBy1(t.Key(), params, visited) && parameterizedBy1(t.Elem(), params, visited)

	case types.TFUNC:
		return parameterizedBy1(t.TParams(), params, visited) && parameterizedBy1(t.Recvs(), params, visited) && parameterizedBy1(t.Params(), params, visited) && parameterizedBy1(t.Results(), params, visited)

	case types.TSTRUCT:
		for _, f := range t.Fields().Slice() {
			if !parameterizedBy1(f.Type, params, visited) {
				return false
			}
		}
		return true

	case types.TINTER:
		for _, f := range t.Methods().Slice() {
			if !parameterizedBy1(f.Type, params, visited) {
				return false
			}
		}
		return true

	case types.TINT, types.TINT8, types.TINT16, types.TINT32, types.TINT64,
		types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64,
		types.TUINTPTR, types.TBOOL, types.TSTRING, types.TFLOAT32, types.TFLOAT64, types.TCOMPLEX64, types.TCOMPLEX128, types.TUNSAFEPTR:
		return true

	case types.TUNION:
		for i := 0; i < t.NumTerms(); i++ {
			tt, _ := t.Term(i)
			if !parameterizedBy1(tt, params, visited) {
				return false
			}
		}
		return true

	default:
		base.Fatalf("bad type kind %+v", t)
		return true
	}
}

// startClosures starts creation of a closure that has the function type typ. It
// creates all the formal params and results according to the type typ. On return,
// the body and closure variables of the closure must still be filled in, and
// ir.UseClosure() called.
func startClosure(pos src.XPos, outer *ir.Func, typ *types.Type) (*ir.Func, []*types.Field, []*types.Field) {
	// Make a new internal function.
	fn := ir.NewClosureFunc(pos, outer != nil)
	ir.NameClosure(fn.OClosure, outer)

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
	closureType := types.NewSignature(typ.Pkg(), nil, nil, formalParams, formalResults)
	typed(closureType, fn.Nname)
	typed(typ, fn.OClosure)
	fn.SetTypecheck(1)
	return fn, formalParams, formalResults

}

// assertToBound returns a new node that converts a node rcvr with interface type to
// the 'dst' interface type.  bound is the unsubstituted form of dst.
func assertToBound(info *instInfo, dictVar *ir.Name, pos src.XPos, rcvr ir.Node, bound, dst *types.Type) ir.Node {
	if bound.HasTParam() {
		ix := findDictType(info, bound)
		assert(ix >= 0)
		rt := getDictionaryType(info, dictVar, pos, ix)
		rcvr = ir.NewDynamicTypeAssertExpr(pos, ir.ODYNAMICDOTTYPE, rcvr, rt)
		typed(dst, rcvr)
	} else {
		rcvr = ir.NewTypeAssertExpr(pos, rcvr, nil)
		typed(bound, rcvr)
	}
	return rcvr
}

// buildClosure2 makes a closure to implement a method expression m (generic form x)
// which has a shape type as receiver. If the receiver is exactly a shape (i.e. from
// a typeparam), then the body of the closure converts m.X (the receiver) to the
// interface bound type, and makes an interface call with the remaining arguments.
//
// The returned closure is fully substituted and has already had any needed
// transformations done.
func (g *irgen) buildClosure2(subst *subster, m, x ir.Node) ir.Node {
	outer := subst.newf
	info := subst.info
	pos := m.Pos()
	typ := m.Type() // type of the closure

	fn, formalParams, formalResults := startClosure(pos, outer, typ)

	// Capture dictionary calculated in the outer function
	dictVar := ir.CaptureName(pos, fn, info.dictParam)
	typed(types.Types[types.TUINTPTR], dictVar)

	// Build arguments to call inside the closure.
	var args []ir.Node
	for i := 0; i < typ.NumParams(); i++ {
		args = append(args, formalParams[i].Nname.(*ir.Name))
	}

	// Build call itself. This involves converting the first argument to the
	// bound type (an interface) using the dictionary, and then making an
	// interface call with the remaining arguments.
	var innerCall ir.Node
	rcvr := args[0]
	args = args[1:]
	assert(m.(*ir.SelectorExpr).X.Type().IsShape())
	gsrc := x.(*ir.SelectorExpr).X.Type()
	bound := gsrc.Bound()
	dst := bound
	if dst.HasTParam() {
		dst = subst.ts.Typ(bound)
	}
	if m.(*ir.SelectorExpr).X.Type().IsInterface() {
		// If type arg is an interface (unusual case), we do a type assert to
		// the type bound.
		rcvr = assertToBound(info, dictVar, pos, rcvr, bound, dst)
	} else {
		rcvr = convertUsingDictionary(info, dictVar, pos, rcvr, x, dst, gsrc)
	}
	dot := ir.NewSelectorExpr(pos, ir.ODOTINTER, rcvr, x.(*ir.SelectorExpr).Sel)
	dot.Selection = typecheck.Lookdot1(dot, dot.Sel, dot.X.Type(), dot.X.Type().AllMethods(), 1)

	// Do a type substitution on the generic bound, in case it is parameterized.
	typed(subst.ts.Typ(x.(*ir.SelectorExpr).Selection.Type), dot)
	innerCall = ir.NewCallExpr(pos, ir.OCALLINTER, dot, args)
	t := m.Type()
	if t.NumResults() == 0 {
		innerCall.SetTypecheck(1)
	} else if t.NumResults() == 1 {
		typed(t.Results().Field(0).Type, innerCall)
	} else {
		typed(t.Results(), innerCall)
	}
	if len(formalResults) > 0 {
		innerCall = ir.NewReturnStmt(pos, []ir.Node{innerCall})
		innerCall.SetTypecheck(1)
	}
	fn.Body = []ir.Node{innerCall}

	// We're all done with the captured dictionary
	ir.FinishCaptureNames(pos, outer, fn)

	// Do final checks on closure and return it.
	return ir.UseClosure(fn.OClosure, g.target)
}
