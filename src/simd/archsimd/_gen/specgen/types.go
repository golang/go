// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"fmt"
	"go/types"
	"regexp"
	"simd/archsimd/_gen/specgen/specexpr"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/tools/go/types/typeutil"
)

// constraintToDomain enumerates all types explicitly listed as satisfying
// constraint (which must be a type parameter constraint), and translates them
// to a [specexpr] domain.
func constraintToDomain(pkg *specPackage, constraint types.Type) ([]any, error) {
	var vals []any
	for _, typ := range typeSet(constraint) {
		elem, ok := pkg.TypeElems[typ]
		if ok {
			vals = append(vals, elem)
		} else if width, ok := pkg.TypeWidths[typ]; ok {
			vals = append(vals, width)
		} else {
			return nil, fmt.Errorf("type %s satisfies constraint %s, but isn't a known shape type", typ, constraint)
		}
	}
	return vals, nil
}

var typeSetMemo sync.Map

// typeSet enumerates all concrete types in t's type set.
func typeSet(t types.Type) []types.Type {
	// In general types.Types are not comparable, but every type we're dealing
	// with here has pointer identity, and also there's no correctness issue if
	// we miss in the memo.
	ts, ok := typeSetMemo.Load(t)
	if !ok {
		var o orderedTypeSet
		typeSet1(t, &o)
		ts, _ = typeSetMemo.LoadOrStore(t, o.types)
	}
	return ts.([]types.Type)
}

type orderedTypeSet struct {
	types []types.Type
	set   typeutil.Map
}

func (set *orderedTypeSet) add(t types.Type) {
	if set.set.At(t) == nil {
		set.set.Set(t, true)
		set.types = append(set.types, t)
	}
}

func (set *orderedTypeSet) intersect(o orderedTypeSet) {
	i, j := 0, 0
	for ; i < len(set.types); i++ {
		t := set.types[i]
		if o.set.At(t) != nil {
			set.types[j] = t
			j++
		} else {
			set.set.Delete(t)
		}
	}
	set.types = set.types[:j]
}

func typeSet1(t types.Type, o *orderedTypeSet) {
	switch u := t.Underlying().(type) {
	case *types.Interface:
		switch u.NumEmbeddeds() {
		case 0:
			return
		case 1:
			// Fast path for common case
			typeSet1(u.EmbeddedType(0), o)
			return
		}
		var intersection orderedTypeSet
		first := true
		for etyp := range u.EmbeddedTypes() {
			if first {
				typeSet1(etyp, &intersection)
				first = false
			} else {
				var tmp orderedTypeSet
				typeSet1(etyp, &tmp)
				intersection.intersect(tmp)
			}
		}
		// TODO: This doesn't check method satisfaction. We could do that with
		// types.Satisfies filter here, but it doesn't matter for our needs.
		for _, etyp := range intersection.types {
			o.add(etyp)
		}

	case *types.Union:
		for term := range u.Terms() {
			typeSet1(term.Type(), o)
		}

	default:
		o.add(t)
	}
}

var basicRe = regexp.MustCompile(`^([a-z]+|Mask)([0-9]*)$`)

// shapeElemType parses a basic or mask element spec type.
func shapeElemType(t types.Type) specexpr.Basic {
	var name string
	switch t := t.(type) {
	case *types.Basic: // E.g., uint32
		name = t.Name()
	case *types.Named: // E.g., Mask16
		name = t.Obj().Name()
	default:
		panic(fmt.Sprintf("not a shape element type: %s", t))
	}
	m := basicRe.FindStringSubmatch(name)
	if m == nil {
		panic(fmt.Sprintf("failed to parse element type %s", name))
	}
	bits := 0
	if m[2] != "" {
		bits, _ = strconv.Atoi(m[2])
	}
	return specexpr.Basic{Base: m[1], Bits: specexpr.Int(bits)}
}

// shapeWidthVal parses a spec width type.
func shapeWidthVal(t types.Type) specexpr.Num {
	named, ok := t.(*types.Named)
	if !ok {
		panic(fmt.Sprintf("not a shape width type: %s", t))
	}
	name := named.Obj().Name()
	if name == "WidthScalable" {
		return specexpr.VW()
	}
	var err error
	if suffix, ok := strings.CutPrefix(name, "Width"); ok {
		var val int
		val, err = strconv.Atoi(suffix)
		if err == nil {
			return specexpr.Int(val)
		}
	} else {
		err = fmt.Errorf("does not start with 'Width'")
	}
	panic(fmt.Sprintf("parsing width type %s: %s", t, err))
}

type argBinder struct {
	ctx        context
	pkg        *specPackage
	s          *specexpr.Solver
	typeParams map[*types.TypeParam]specexpr.Variable
}

// bindArg assigns all solver variables related to an argument called "name" of
// type t. It returns a function that retrieves the resolved type, or nil on
// error.
func (b *argBinder) bindArg(name string, t types.Type) func(*specexpr.Bindings) specexpr.Type {
	expr := b.bind1(name, t)
	if expr == nil {
		return nil
	}
	v := b.s.Assign(specexpr.Variable(name), expr)
	return func(b *specexpr.Bindings) specexpr.Type {
		return b.Get(v).(specexpr.Type)
	}
}

// bind1 deconstructs t and binds any components to variables derived from
// "name", and returns the (not yet bound!) expression for t. The caller is
// expected to bind "name" to the returned expression, or pass it up. It works
// this way so we can unwrap things like pointer and slice types without
// creating intermediate names for each level.
func (b *argBinder) bind1(name string, t types.Type) specexpr.Expr {
	switch t := t.(type) {
	case *types.Basic:
		return &specexpr.Literal{Val: shapeElemType(t)}

	case *types.Pointer:
		elem := b.bind1(name, t.Elem())
		if elem == nil {
			return nil
		}
		return specexpr.MakePointer(elem)

	case *types.Slice:
		elem := b.bind1(name, t.Elem())
		if elem == nil {
			return nil
		}
		return specexpr.MakeSlice(elem)

	case *types.Array:
		elem := b.bind1(name, t.Elem())
		if elem == nil {
			return nil
		}
		return specexpr.MakeArray(elem, specexpr.Int(t.Len()))

	case *types.Named:
		if types.Identical(t.Origin(), b.pkg.VecType) {
			xE, xW, _ := b.bindVecLike(name, t)
			if xE == nil {
				return nil
			}
			return specexpr.MakeVector(xE, xW)
		}
		if types.Identical(t.Origin(), b.pkg.ArrayType) {
			xE, xW, xL := b.bindVecLike(name, t)
			if xE == nil {
				return nil
			}
			b.s.Assert(funcAssertFixed(xW))
			return specexpr.MakeArray(xE, xL)
		}
		if types.Identical(t.Origin(), b.pkg.UintNType) {
			xN := specexpr.Variable(name + "N")
			b.s.Declare(xN, []any{8, 16, 32, 64})
			b.s.Assert(funcAssertFixed(xN))
			return specexpr.MakeBasic(&specexpr.Literal{Val: "uint"}, xN)
		}

	case *types.TypeParam:
		return b.typeParams[t]
	}

	b.ctx.errorf("cannot convert spec type %s into API type", t)
	return nil
}

var funcAssertFixed = specexpr.MakeFunc1("assertFixed", func(w specexpr.Num) (any, error) {
	_, ok := w.(specexpr.Int)
	return ok, nil
})

func (b *argBinder) bindVecLike(name string, t *types.Named) (xE, xW, xL specexpr.Expr) {
	args := t.TypeArgs()
	if args.Len() != 2 {
		b.ctx.errorf("expected exactly 2 type arguments, got %d", args.Len())
		return nil, nil, nil
	}

	// Assign the element type
	elem := b.bind1(name+"E", args.At(0))
	if elem == nil {
		return nil, nil, nil
	}
	xE = b.s.Assign(specexpr.Variable(name+"E"), elem)

	// Get the width
	var wExpr specexpr.Expr
	switch wt := args.At(1).(type) {
	case *types.TypeParam:
		wExpr = b.typeParams[wt]
	case *types.Named:
		wExpr = b.pkg.TypeWidths[wt]
		if wExpr == nil {
			b.ctx.errorf("width type argument is not a width")
			return nil, nil, nil
		}
	default:
		b.ctx.errorf("width type arguments not a type parameter or named type")
		return nil, nil, nil
	}
	xW = b.s.Assign(specexpr.Variable(name+"W"), wExpr)

	// Bind other variables
	basicBase := specexpr.MakeField[specexpr.Basic]("Base")
	basicBits := specexpr.MakeField[specexpr.Basic]("Bits")
	b.s.Assign(specexpr.Variable(name+"B"), basicBase.Apply(xE))
	xN := b.s.Assign(specexpr.Variable(name+"N"), basicBits.Apply(xE))
	xL = b.s.Assign(specexpr.Variable(name+"L"), &specexpr.BinExpr{
		Op: specexpr.OpDiv, X: xW, Y: xN,
	})

	return xE, xW, xL
}
