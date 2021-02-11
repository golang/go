// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"bytes"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

func (g *irgen) pkg(pkg *types2.Package) *types.Pkg {
	switch pkg {
	case nil:
		return types.BuiltinPkg
	case g.self:
		return types.LocalPkg
	case types2.Unsafe:
		return ir.Pkgs.Unsafe
	}
	return types.NewPkg(pkg.Path(), pkg.Name())
}

// typ converts a types2.Type to a types.Type, including caching of previously
// translated types.
func (g *irgen) typ(typ types2.Type) *types.Type {
	// Caching type mappings isn't strictly needed, because typ0 preserves
	// type identity; but caching minimizes memory blow-up from mapping the
	// same composite type multiple times, and also plays better with the
	// current state of cmd/compile (e.g., haphazard calculation of type
	// sizes).
	res, ok := g.typs[typ]
	if !ok {
		res = g.typ0(typ)
		g.typs[typ] = res

		// Ensure we calculate the size for all concrete types seen by
		// the frontend. This is another heavy hammer for something that
		// should really be the backend's responsibility instead.
		if res != nil && !res.IsUntyped() && !res.IsFuncArgStruct() {
			types.CheckSize(res)
		}
	}
	return res
}

// instTypeName2 creates a name for an instantiated type, base on the type args
// (given as types2 types).
func instTypeName2(name string, targs []types2.Type) string {
	b := bytes.NewBufferString(name)
	b.WriteByte('[')
	for i, targ := range targs {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(types2.TypeString(targ,
			func(*types2.Package) string { return "" }))
	}
	b.WriteByte(']')
	return b.String()
}

// typ0 converts a types2.Type to a types.Type, but doesn't do the caching check
// at the top level.
func (g *irgen) typ0(typ types2.Type) *types.Type {
	switch typ := typ.(type) {
	case *types2.Basic:
		return g.basic(typ)
	case *types2.Named:
		if typ.TParams() != nil {
			// typ is an instantiation of a defined (named) generic type.
			// This instantiation should also be a defined (named) type.
			// types2 gives us the substituted type in t.Underlying()
			// The substituted type may or may not still have type
			// params. We might, for example, be substituting one type
			// param for another type param.

			if typ.TArgs() == nil {
				base.Fatalf("In typ0, Targs should be set if TParams is set")
			}

			// When converted to types.Type, typ must have a name,
			// based on the names of the type arguments. We need a
			// name to deal with recursive generic types (and it also
			// looks better when printing types).
			instName := instTypeName2(typ.Obj().Name(), typ.TArgs())
			s := g.pkg(typ.Obj().Pkg()).Lookup(instName)
			if s.Def != nil {
				// We have already encountered this instantiation,
				// so use the type we previously created, since there
				// must be exactly one instance of a defined type.
				return s.Def.Type()
			}

			// Create a forwarding type first and put it in the g.typs
			// map, in order to deal with recursive generic types.
			ntyp := types.New(types.TFORW)
			g.typs[typ] = ntyp
			ntyp.SetUnderlying(g.typ(typ.Underlying()))
			ntyp.SetSym(s)

			if ntyp.HasTParam() {
				// If ntyp still has type params, then we must be
				// referencing something like 'value[T2]', as when
				// specifying the generic receiver of a method,
				// where value was defined as "type value[T any]
				// ...". Save the type args, which will now be the
				// new type params of the current type.
				ntyp.RParams = make([]*types.Type, len(typ.TArgs()))
				for i, targ := range typ.TArgs() {
					ntyp.RParams[i] = g.typ(targ)
				}
			}

			// Make sure instantiated type can be uniquely found from
			// the sym
			s.Def = ir.TypeNode(ntyp)
			return ntyp
		}
		obj := g.obj(typ.Obj())
		if obj.Op() != ir.OTYPE {
			base.FatalfAt(obj.Pos(), "expected type: %L", obj)
		}
		return obj.Type()

	case *types2.Array:
		return types.NewArray(g.typ(typ.Elem()), typ.Len())
	case *types2.Chan:
		return types.NewChan(g.typ(typ.Elem()), dirs[typ.Dir()])
	case *types2.Map:
		return types.NewMap(g.typ(typ.Key()), g.typ(typ.Elem()))
	case *types2.Pointer:
		return types.NewPtr(g.typ(typ.Elem()))
	case *types2.Signature:
		return g.signature(nil, typ)
	case *types2.Slice:
		return types.NewSlice(g.typ(typ.Elem()))

	case *types2.Struct:
		fields := make([]*types.Field, typ.NumFields())
		for i := range fields {
			v := typ.Field(i)
			f := types.NewField(g.pos(v), g.selector(v), g.typ(v.Type()))
			f.Note = typ.Tag(i)
			if v.Embedded() {
				f.Embedded = 1
			}
			fields[i] = f
		}
		return types.NewStruct(g.tpkg(typ), fields)

	case *types2.Interface:
		embeddeds := make([]*types.Field, typ.NumEmbeddeds())
		for i := range embeddeds {
			// TODO(mdempsky): Get embedding position.
			e := typ.EmbeddedType(i)
			embeddeds[i] = types.NewField(src.NoXPos, nil, g.typ(e))
		}

		methods := make([]*types.Field, typ.NumExplicitMethods())
		for i := range methods {
			m := typ.ExplicitMethod(i)
			mtyp := g.signature(typecheck.FakeRecv(), m.Type().(*types2.Signature))
			methods[i] = types.NewField(g.pos(m), g.selector(m), mtyp)
		}

		return types.NewInterface(g.tpkg(typ), append(embeddeds, methods...))

	case *types2.TypeParam:
		tp := types.NewTypeParam(g.tpkg(typ), g.typ(typ.Bound()))
		// Save the name of the type parameter in the sym of the type.
		tp.SetSym(g.sym(typ.Obj()))
		return tp

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
		types.CheckSize(t)
		// Can only set after doing the types.CheckSize()
		t.StructType().Funarg = types.FunargResults
		return t

	default:
		base.FatalfAt(src.NoXPos, "unhandled type: %v (%T)", typ, typ)
		panic("unreachable")
	}
}

func (g *irgen) signature(recv *types.Field, sig *types2.Signature) *types.Type {
	tparams2 := sig.TParams()
	tparams := make([]*types.Field, len(tparams2))
	for i := range tparams {
		tp := tparams2[i]
		tparams[i] = types.NewField(g.pos(tp), g.sym(tp), g.typ(tp.Type()))
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
	return types.NewField(g.pos(v), g.sym(v), g.typ(v.Type()))
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

// tpkg returns the package that a function, interface, or struct type
// expression appeared in.
//
// Caveat: For the degenerate types "func()", "interface{}", and
// "struct{}", tpkg always returns LocalPkg. However, we only need the
// package information so that go/types can report it via its API, and
// the reason we fail to return the original package for these
// particular types is because go/types does *not* report it for
// them. So in practice this limitation is probably moot.
func (g *irgen) tpkg(typ types2.Type) *types.Pkg {
	anyObj := func() types2.Object {
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
		}
		return nil
	}

	if obj := anyObj(); obj != nil {
		return g.pkg(obj.Pkg())
	}
	return types.LocalPkg
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
