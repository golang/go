// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
	"strings"
)

func (g *irgen) pkg(pkg *types2.Package) *types.Pkg {
	switch pkg {
	case nil:
		return types.BuiltinPkg
	case g.self:
		return types.LocalPkg
	case types2.Unsafe:
		return types.UnsafePkg
	}
	return types.NewPkg(pkg.Path(), pkg.Name())
}

var universeAny = types2.Universe.Lookup("any").Type()

// typ converts a types2.Type to a types.Type, including caching of previously
// translated types.
func (g *irgen) typ(typ types2.Type) *types.Type {
	// Defer the CheckSize calls until we have fully-defined a
	// (possibly-recursive) top-level type.
	types.DeferCheckSize()
	res := g.typ1(typ)
	types.ResumeCheckSize()

	// Finish up any types on typesToFinalize, now that we are at the top of a
	// fully-defined (possibly recursive) type. fillinMethods could create more
	// types to finalize.
	for len(g.typesToFinalize) > 0 {
		l := len(g.typesToFinalize)
		info := g.typesToFinalize[l-1]
		g.typesToFinalize = g.typesToFinalize[:l-1]
		types.DeferCheckSize()
		g.fillinMethods(info.typ, info.ntyp)
		types.ResumeCheckSize()
	}
	return res
}

// typ1 is like typ, but doesn't call CheckSize, since it may have only
// constructed part of a recursive type. Should not be called from outside this
// file (g.typ is the "external" entry point).
func (g *irgen) typ1(typ types2.Type) *types.Type {
	// See issue 49583: the type checker has trouble keeping track of aliases,
	// but for such a common alias as any we can improve things by preserving a
	// pointer identity that can be checked when formatting type strings.
	if typ == universeAny {
		return types.AnyType
	}
	// Cache type2-to-type mappings. Important so that each defined generic
	// type (instantiated or not) has a single types.Type representation.
	// Also saves a lot of computation and memory by avoiding re-translating
	// types2 types repeatedly.
	res, ok := g.typs[typ]
	if !ok {
		res = g.typ0(typ)
		// Calculate the size for all concrete types seen by the frontend.
		// This is the replacement for the CheckSize() calls in the types1
		// typechecker. These will be deferred until the top-level g.typ().
		if res != nil && !res.IsUntyped() && !res.IsFuncArgStruct() && !res.HasTParam() {
			types.CheckSize(res)
		}
		g.typs[typ] = res
	}
	return res
}

// instTypeName2 creates a name for an instantiated type, base on the type args
// (given as types2 types).
func (g *irgen) instTypeName2(name string, targs *types2.TypeList) string {
	rparams := make([]*types.Type, targs.Len())
	for i := range rparams {
		rparams[i] = g.typ(targs.At(i))
	}
	return typecheck.InstTypeName(name, rparams)
}

// typ0 converts a types2.Type to a types.Type, but doesn't do the caching check
// at the top level.
func (g *irgen) typ0(typ types2.Type) *types.Type {
	switch typ := typ.(type) {
	case *types2.Basic:
		return g.basic(typ)
	case *types2.Named:
		// If tparams is set, but targs is not, typ is a base generic
		// type. typ is appearing as part of the source type of an alias,
		// since that is the only use of a generic type that doesn't
		// involve instantiation. We just translate the named type in the
		// normal way below using g.obj().
		if typ.TypeParams() != nil && typ.TypeArgs() != nil {
			// typ is an instantiation of a defined (named) generic type.
			// This instantiation should also be a defined (named) type.
			// types2 gives us the substituted type in t.Underlying()
			// The substituted type may or may not still have type
			// params. We might, for example, be substituting one type
			// param for another type param.
			//
			// When converted to types.Type, typ has a unique name,
			// based on the names of the type arguments.
			instName := g.instTypeName2(typ.Obj().Name(), typ.TypeArgs())
			s := g.pkg(typ.Obj().Pkg()).Lookup(instName)

			// Make sure the base generic type exists in type1 (it may
			// not yet if we are referecing an imported generic type, as
			// opposed to a generic type declared in this package). Make
			// sure to do this lookup before checking s.Def, in case
			// s.Def gets defined while importing base (if an imported
			// type). (Issue #50486).
			base := g.obj(typ.Origin().Obj())

			if s.Def != nil {
				// We have already encountered this instantiation.
				// Use the type we previously created, since there
				// must be exactly one instance of a defined type.
				return s.Def.Type()
			}

			if base.Class == ir.PAUTO {
				// If the base type is a local type, we want to pop
				// this instantiated type symbol/definition when we
				// leave the containing block, so we don't use it
				// incorrectly later.
				types.Pushdcl(s)
			}

			// Create a forwarding type first and put it in the g.typs
			// map, in order to deal with recursive generic types
			// (including via method signatures). Set up the extra
			// ntyp information (Def, RParams, which may set
			// HasTParam) before translating the underlying type
			// itself, so we handle recursion correctly.
			ntyp := typecheck.NewIncompleteNamedType(g.pos(typ.Obj().Pos()), s)
			g.typs[typ] = ntyp

			// If ntyp still has type params, then we must be
			// referencing something like 'value[T2]', as when
			// specifying the generic receiver of a method, where
			// value was defined as "type value[T any] ...". Save the
			// type args, which will now be the new typeparams of the
			// current type.
			//
			// If ntyp does not have type params, we are saving the
			// non-generic types used to instantiate this type. We'll
			// use these when instantiating the methods of the
			// instantiated type.
			targs := typ.TypeArgs()
			rparams := make([]*types.Type, targs.Len())
			for i := range rparams {
				rparams[i] = g.typ1(targs.At(i))
			}
			ntyp.SetRParams(rparams)
			//fmt.Printf("Saw new type %v %v\n", instName, ntyp.HasTParam())

			// Save the symbol for the base generic type.
			ntyp.SetOrigType(base.Type())
			ntyp.SetUnderlying(g.typ1(typ.Underlying()))
			if typ.NumMethods() != 0 {
				// Save a delayed call to g.fillinMethods() (once
				// potentially recursive types have been fully
				// resolved).
				g.typesToFinalize = append(g.typesToFinalize,
					&typeDelayInfo{
						typ:  typ,
						ntyp: ntyp,
					})
			}
			return ntyp
		}
		obj := g.obj(typ.Obj())
		if obj.Op() != ir.OTYPE {
			base.FatalfAt(obj.Pos(), "expected type: %L", obj)
		}
		return obj.Type()

	case *types2.Array:
		return types.NewArray(g.typ1(typ.Elem()), typ.Len())
	case *types2.Chan:
		return types.NewChan(g.typ1(typ.Elem()), dirs[typ.Dir()])
	case *types2.Map:
		return types.NewMap(g.typ1(typ.Key()), g.typ1(typ.Elem()))
	case *types2.Pointer:
		return types.NewPtr(g.typ1(typ.Elem()))
	case *types2.Signature:
		return g.signature(nil, typ)
	case *types2.Slice:
		return types.NewSlice(g.typ1(typ.Elem()))

	case *types2.Struct:
		fields := make([]*types.Field, typ.NumFields())
		for i := range fields {
			v := typ.Field(i)
			f := types.NewField(g.pos(v), g.selector(v), g.typ1(v.Type()))
			f.Note = typ.Tag(i)
			if v.Embedded() {
				f.Embedded = 1
			}
			fields[i] = f
		}
		return types.NewStruct(g.tpkg(typ), fields)

	case *types2.Interface:
		embeddeds := make([]*types.Field, typ.NumEmbeddeds())
		j := 0
		for i := range embeddeds {
			// TODO(mdempsky): Get embedding position.
			e := typ.EmbeddedType(i)

			// With Go 1.18, an embedded element can be any type, not
			// just an interface.
			embeddeds[j] = types.NewField(src.NoXPos, nil, g.typ1(e))
			j++
		}
		embeddeds = embeddeds[:j]

		methods := make([]*types.Field, typ.NumExplicitMethods())
		for i := range methods {
			m := typ.ExplicitMethod(i)
			mtyp := g.signature(types.FakeRecv(), m.Type().(*types2.Signature))
			methods[i] = types.NewField(g.pos(m), g.selector(m), mtyp)
		}

		return types.NewInterface(g.tpkg(typ), append(embeddeds, methods...), typ.IsImplicit())

	case *types2.TypeParam:
		// Save the name of the type parameter in the sym of the type.
		// Include the types2 subscript in the sym name
		pkg := g.tpkg(typ)
		// Create the unique types1 name for a type param, using its context
		// with a function, type, or method declaration. Also, map blank type
		// param names to a unique name based on their type param index. The
		// unique blank names will be exported, but will be reverted during
		// types2 and gcimporter import.
		assert(g.curDecl != "")
		nm := typecheck.TparamExportName(g.curDecl, typ.Obj().Name(), typ.Index())
		sym := pkg.Lookup(nm)
		if sym.Def != nil {
			// Make sure we use the same type param type for the same
			// name, whether it is created during types1-import or
			// this types2-to-types1 translation.
			return sym.Def.Type()
		}
		tp := types.NewTypeParam(sym, typ.Index())
		nname := ir.NewDeclNameAt(g.pos(typ.Obj().Pos()), ir.OTYPE, sym)
		sym.Def = nname
		nname.SetType(tp)
		tp.SetNod(nname)
		// Set g.typs[typ] in case the bound methods reference typ.
		g.typs[typ] = tp

		bound := g.typ1(typ.Constraint())
		tp.SetBound(bound)
		return tp

	case *types2.Union:
		nt := typ.Len()
		tlist := make([]*types.Type, nt)
		tildes := make([]bool, nt)
		for i := range tlist {
			t := typ.Term(i)
			tlist[i] = g.typ1(t.Type())
			tildes[i] = t.Tilde()
		}
		return types.NewUnion(tlist, tildes)

	case *types2.Tuple:
		// Tuples are used for the type of a function call (i.e. the
		// return value of the function).
		if typ == nil {
			return (*types.Type)(nil)
		}
		fields := make([]*types.Field, typ.Len())
		for i := range fields {
			fields[i] = g.param(typ.At(i))
		}
		t := types.NewStruct(types.LocalPkg, fields)
		t.StructType().Funarg = types.FunargResults
		return t

	default:
		base.FatalfAt(src.NoXPos, "unhandled type: %v (%T)", typ, typ)
		panic("unreachable")
	}
}

// fillinMethods fills in the method name nodes and types for a defined type with at
// least one method. This is needed for later typechecking when looking up methods of
// instantiated types, and for actually generating the methods for instantiated
// types.
func (g *irgen) fillinMethods(typ *types2.Named, ntyp *types.Type) {
	targs2 := typ.TypeArgs()
	targs := make([]*types.Type, targs2.Len())
	for i := range targs {
		targs[i] = g.typ1(targs2.At(i))
	}

	methods := make([]*types.Field, typ.NumMethods())
	for i := range methods {
		m := typ.Method(i)
		recvType := deref2(types2.AsSignature(m.Type()).Recv().Type())
		var meth *ir.Name
		imported := false
		if m.Pkg() != g.self {
			// Imported methods cannot be loaded by name (what
			// g.obj() does) - they must be loaded via their
			// type.
			meth = g.obj(recvType.(*types2.Named).Obj()).Type().Methods().Index(i).Nname.(*ir.Name)
			// XXX Because Obj() returns the object of the base generic
			// type, we have to still do the method translation below.
			imported = true
		} else {
			meth = g.obj(m)
		}
		assert(recvType == types2.Type(typ))
		if imported {
			// Unfortunately, meth is the type of the method of the
			// generic type, so we have to do a substitution to get
			// the name/type of the method of the instantiated type,
			// using m.Type().RParams() and typ.TArgs()
			inst2 := g.instTypeName2("", typ.TypeArgs())
			name := meth.Sym().Name
			i1 := strings.Index(name, "[")
			i2 := strings.Index(name[i1:], "]")
			assert(i1 >= 0 && i2 >= 0)
			// Generate the name of the instantiated method.
			name = name[0:i1] + inst2 + name[i1+i2+1:]
			newsym := meth.Sym().Pkg.Lookup(name)
			var meth2 *ir.Name
			if newsym.Def != nil {
				meth2 = newsym.Def.(*ir.Name)
			} else {
				meth2 = ir.NewNameAt(meth.Pos(), newsym)
				rparams := types2.AsSignature(m.Type()).RecvTypeParams()
				tparams := make([]*types.Type, rparams.Len())
				// Set g.curDecl to be the method context, so type
				// params in the receiver of the method that we are
				// translating gets the right unique name. We could
				// be in a top-level typeDecl, so save and restore
				// the current contents of g.curDecl.
				savedCurDecl := g.curDecl
				g.curDecl = typ.Obj().Name() + "." + m.Name()
				for i := range tparams {
					tparams[i] = g.typ1(rparams.At(i))
				}
				g.curDecl = savedCurDecl
				assert(len(tparams) == len(targs))
				ts := typecheck.Tsubster{
					Tparams: tparams,
					Targs:   targs,
				}
				// Do the substitution of the type
				meth2.SetType(ts.Typ(meth.Type()))
				newsym.Def = meth2
			}
			meth = meth2
		}
		methods[i] = types.NewField(meth.Pos(), g.selector(m), meth.Type())
		methods[i].Nname = meth
	}
	ntyp.Methods().Set(methods)
	if !ntyp.HasTParam() && !ntyp.HasShape() {
		// Generate all the methods for a new fully-instantiated type.
		typecheck.NeedInstType(ntyp)
	}
}

func (g *irgen) signature(recv *types.Field, sig *types2.Signature) *types.Type {
	tparams2 := sig.TypeParams()
	tparams := make([]*types.Field, tparams2.Len())
	for i := range tparams {
		tp := tparams2.At(i).Obj()
		tparams[i] = types.NewField(g.pos(tp), g.sym(tp), g.typ1(tp.Type()))
	}

	do := func(typ *types2.Tuple) []*types.Field {
		fields := make([]*types.Field, typ.Len())
		for i := range fields {
			fields[i] = g.param(typ.At(i))
		}
		return fields
	}
	params := do(sig.Params())
	results := do(sig.Results())
	if sig.Variadic() {
		params[len(params)-1].SetIsDDD(true)
	}

	return types.NewSignature(g.tpkg(sig), recv, tparams, params, results)
}

func (g *irgen) param(v *types2.Var) *types.Field {
	return types.NewField(g.pos(v), g.sym(v), g.typ1(v.Type()))
}

func (g *irgen) sym(obj types2.Object) *types.Sym {
	if name := obj.Name(); name != "" {
		return g.pkg(obj.Pkg()).Lookup(obj.Name())
	}
	return nil
}

func (g *irgen) selector(obj types2.Object) *types.Sym {
	pkg, name := g.pkg(obj.Pkg()), obj.Name()
	if types.IsExported(name) {
		pkg = types.LocalPkg
	}
	return pkg.Lookup(name)
}

// tpkg returns the package that a function, interface, struct, or typeparam type
// expression appeared in.
//
// Caveat: For the degenerate types "func()", "interface{}", and
// "struct{}", tpkg always returns LocalPkg. However, we only need the
// package information so that go/types can report it via its API, and
// the reason we fail to return the original package for these
// particular types is because go/types does *not* report it for
// them. So in practice this limitation is probably moot.
func (g *irgen) tpkg(typ types2.Type) *types.Pkg {
	if obj := anyObj(typ); obj != nil {
		return g.pkg(obj.Pkg())
	}
	return types.LocalPkg
}

// anyObj returns some object accessible from typ, if any.
func anyObj(typ types2.Type) types2.Object {
	switch typ := typ.(type) {
	case *types2.Signature:
		if recv := typ.Recv(); recv != nil {
			return recv
		}
		if params := typ.Params(); params.Len() > 0 {
			return params.At(0)
		}
		if results := typ.Results(); results.Len() > 0 {
			return results.At(0)
		}
	case *types2.Struct:
		if typ.NumFields() > 0 {
			return typ.Field(0)
		}
	case *types2.Interface:
		if typ.NumExplicitMethods() > 0 {
			return typ.ExplicitMethod(0)
		}
	case *types2.TypeParam:
		return typ.Obj()
	}
	return nil
}

func (g *irgen) basic(typ *types2.Basic) *types.Type {
	switch typ.Name() {
	case "byte":
		return types.ByteType
	case "rune":
		return types.RuneType
	}
	return *basics[typ.Kind()]
}

var basics = [...]**types.Type{
	types2.Invalid:        new(*types.Type),
	types2.Bool:           &types.Types[types.TBOOL],
	types2.Int:            &types.Types[types.TINT],
	types2.Int8:           &types.Types[types.TINT8],
	types2.Int16:          &types.Types[types.TINT16],
	types2.Int32:          &types.Types[types.TINT32],
	types2.Int64:          &types.Types[types.TINT64],
	types2.Uint:           &types.Types[types.TUINT],
	types2.Uint8:          &types.Types[types.TUINT8],
	types2.Uint16:         &types.Types[types.TUINT16],
	types2.Uint32:         &types.Types[types.TUINT32],
	types2.Uint64:         &types.Types[types.TUINT64],
	types2.Uintptr:        &types.Types[types.TUINTPTR],
	types2.Float32:        &types.Types[types.TFLOAT32],
	types2.Float64:        &types.Types[types.TFLOAT64],
	types2.Complex64:      &types.Types[types.TCOMPLEX64],
	types2.Complex128:     &types.Types[types.TCOMPLEX128],
	types2.String:         &types.Types[types.TSTRING],
	types2.UnsafePointer:  &types.Types[types.TUNSAFEPTR],
	types2.UntypedBool:    &types.UntypedBool,
	types2.UntypedInt:     &types.UntypedInt,
	types2.UntypedRune:    &types.UntypedRune,
	types2.UntypedFloat:   &types.UntypedFloat,
	types2.UntypedComplex: &types.UntypedComplex,
	types2.UntypedString:  &types.UntypedString,
	types2.UntypedNil:     &types.Types[types.TNIL],
}

var dirs = [...]types.ChanDir{
	types2.SendRecv: types.Cboth,
	types2.SendOnly: types.Csend,
	types2.RecvOnly: types.Crecv,
}

// deref2 does a single deref of types2 type t, if it is a pointer type.
func deref2(t types2.Type) types2.Type {
	if ptr := types2.AsPointer(t); ptr != nil {
		t = ptr.Elem()
	}
	return t
}
