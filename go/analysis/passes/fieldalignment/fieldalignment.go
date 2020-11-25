// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fieldalignment defines an Analyzer that detects structs that would take less
// memory if their fields were sorted.
package fieldalignment

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"sort"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

const Doc = `find structs that would take less memory if their fields were sorted

This analyzer find structs that can be rearranged to take less memory, and provides
a suggested edit with the optimal order.
`

var Analyzer = &analysis.Analyzer{
	Name:     "fieldalignment",
	Doc:      Doc,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.StructType)(nil),
	}
	inspect.Preorder(nodeFilter, func(node ast.Node) {
		var s *ast.StructType
		var ok bool
		if s, ok = node.(*ast.StructType); !ok {
			return
		}
		if tv, ok := pass.TypesInfo.Types[s]; ok {
			fieldalignment(pass, s, tv.Type.(*types.Struct))
		}
	})
	return nil, nil
}

var unsafePointerTyp = types.Unsafe.Scope().Lookup("Pointer").(*types.TypeName).Type()

func fieldalignment(pass *analysis.Pass, node *ast.StructType, typ *types.Struct) {
	wordSize := pass.TypesSizes.Sizeof(unsafePointerTyp)
	maxAlign := pass.TypesSizes.Alignof(unsafePointerTyp)

	s := gcSizes{wordSize, maxAlign}
	optimal, indexes := optimalOrder(typ, &s)
	optsz, optptrs := s.Sizeof(optimal), s.ptrdata(optimal)

	var message string
	if sz := s.Sizeof(typ); sz != optsz {
		message = fmt.Sprintf("struct of size %d could be %d", sz, optsz)
	} else if ptrs := s.ptrdata(typ); ptrs != optptrs {
		message = fmt.Sprintf("struct with %d pointer bytes could be %d", ptrs, optptrs)
	} else {
		// Already optimal order.
		return
	}

	// Flatten the ast node since it could have multiple field names per list item while
	// *types.Struct only have one item per field.
	// TODO: Preserve multi-named fields instead of flattening.
	var flat []*ast.Field
	for _, f := range node.Fields.List {
		if len(f.Names) == 1 {
			flat = append(flat, f)
			continue
		}
		for _, name := range f.Names {
			flat = append(flat, &ast.Field{
				Names: []*ast.Ident{name},
				Type:  f.Type,
			})
		}
	}

	// Sort fields according to the optimal order.
	var reordered []*ast.Field
	for _, index := range indexes {
		reordered = append(reordered, flat[index])
	}

	newStr := &ast.StructType{
		Fields: &ast.FieldList{
			List: reordered,
		},
	}

	// Write the newly aligned struct node to get the content for suggested fixes.
	var buf bytes.Buffer
	if err := format.Node(&buf, token.NewFileSet(), newStr); err != nil {
		return
	}

	pass.Report(analysis.Diagnostic{
		Pos:     node.Pos(),
		End:     node.Pos() + token.Pos(len("struct")),
		Message: message,
		SuggestedFixes: []analysis.SuggestedFix{{
			Message: "Rearrange fields",
			TextEdits: []analysis.TextEdit{{
				Pos:     node.Pos(),
				End:     node.End(),
				NewText: buf.Bytes(),
			}},
		}},
	})
}

func optimalOrder(str *types.Struct, sizes *gcSizes) (*types.Struct, []int) {
	nf := str.NumFields()

	type elem struct {
		index   int
		alignof int64
		sizeof  int64
		ptrdata int64
	}

	elems := make([]elem, nf)
	for i := 0; i < nf; i++ {
		field := str.Field(i)
		ft := field.Type()
		elems[i] = elem{
			i,
			sizes.Alignof(ft),
			sizes.Sizeof(ft),
			sizes.ptrdata(ft),
		}
	}

	sort.Slice(elems, func(i, j int) bool {
		ei := &elems[i]
		ej := &elems[j]

		// Place zero sized objects before non-zero sized objects.
		zeroi := ei.sizeof == 0
		zeroj := ej.sizeof == 0
		if zeroi != zeroj {
			return zeroi
		}

		// Next, place more tightly aligned objects before less tightly aligned objects.
		if ei.alignof != ej.alignof {
			return ei.alignof > ej.alignof
		}

		// Place pointerful objects before pointer-free objects.
		noptrsi := ei.ptrdata == 0
		noptrsj := ej.ptrdata == 0
		if noptrsi != noptrsj {
			return noptrsj
		}

		if !noptrsi {
			// If both have pointers...

			// ... then place objects with less trailing
			// non-pointer bytes earlier. That is, place
			// the field with the most trailing
			// non-pointer bytes at the end of the
			// pointerful section.
			traili := ei.sizeof - ei.ptrdata
			trailj := ej.sizeof - ej.ptrdata
			if traili != trailj {
				return traili < trailj
			}
		}

		// Lastly, order by size.
		if ei.sizeof != ej.sizeof {
			return ei.sizeof > ej.sizeof
		}

		return false
	})

	fields := make([]*types.Var, nf)
	indexes := make([]int, nf)
	for i, e := range elems {
		fields[i] = str.Field(e.index)
		indexes[i] = e.index
	}
	return types.NewStruct(fields, nil), indexes
}

// Code below based on go/types.StdSizes.

type gcSizes struct {
	WordSize int64
	MaxAlign int64
}

func (s *gcSizes) Alignof(T types.Type) int64 {
	// For arrays and structs, alignment is defined in terms
	// of alignment of the elements and fields, respectively.
	switch t := T.Underlying().(type) {
	case *types.Array:
		// spec: "For a variable x of array type: unsafe.Alignof(x)
		// is the same as unsafe.Alignof(x[0]), but at least 1."
		return s.Alignof(t.Elem())
	case *types.Struct:
		// spec: "For a variable x of struct type: unsafe.Alignof(x)
		// is the largest of the values unsafe.Alignof(x.f) for each
		// field f of x, but at least 1."
		max := int64(1)
		for i, nf := 0, t.NumFields(); i < nf; i++ {
			if a := s.Alignof(t.Field(i).Type()); a > max {
				max = a
			}
		}
		return max
	}
	a := s.Sizeof(T) // may be 0
	// spec: "For a variable x of any type: unsafe.Alignof(x) is at least 1."
	if a < 1 {
		return 1
	}
	if a > s.MaxAlign {
		return s.MaxAlign
	}
	return a
}

var basicSizes = [...]byte{
	types.Bool:       1,
	types.Int8:       1,
	types.Int16:      2,
	types.Int32:      4,
	types.Int64:      8,
	types.Uint8:      1,
	types.Uint16:     2,
	types.Uint32:     4,
	types.Uint64:     8,
	types.Float32:    4,
	types.Float64:    8,
	types.Complex64:  8,
	types.Complex128: 16,
}

func (s *gcSizes) Sizeof(T types.Type) int64 {
	switch t := T.Underlying().(type) {
	case *types.Basic:
		k := t.Kind()
		if int(k) < len(basicSizes) {
			if s := basicSizes[k]; s > 0 {
				return int64(s)
			}
		}
		if k == types.String {
			return s.WordSize * 2
		}
	case *types.Array:
		return t.Len() * s.Sizeof(t.Elem())
	case *types.Slice:
		return s.WordSize * 3
	case *types.Struct:
		nf := t.NumFields()
		if nf == 0 {
			return 0
		}

		var o int64
		max := int64(1)
		for i := 0; i < nf; i++ {
			ft := t.Field(i).Type()
			a, sz := s.Alignof(ft), s.Sizeof(ft)
			if a > max {
				max = a
			}
			if i == nf-1 && sz == 0 && o != 0 {
				sz = 1
			}
			o = align(o, a) + sz
		}
		return align(o, max)
	case *types.Interface:
		return s.WordSize * 2
	}
	return s.WordSize // catch-all
}

// align returns the smallest y >= x such that y % a == 0.
func align(x, a int64) int64 {
	y := x + a - 1
	return y - y%a
}

func (s *gcSizes) ptrdata(T types.Type) int64 {
	switch t := T.Underlying().(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.String, types.UnsafePointer:
			return s.WordSize
		}
		return 0
	case *types.Chan, *types.Map, *types.Pointer, *types.Signature, *types.Slice:
		return s.WordSize
	case *types.Interface:
		return 2 * s.WordSize
	case *types.Array:
		n := t.Len()
		if n == 0 {
			return 0
		}
		a := s.ptrdata(t.Elem())
		if a == 0 {
			return 0
		}
		z := s.Sizeof(t.Elem())
		return (n-1)*z + a
	case *types.Struct:
		nf := t.NumFields()
		if nf == 0 {
			return 0
		}

		var o, p int64
		for i := 0; i < nf; i++ {
			ft := t.Field(i).Type()
			a, sz := s.Alignof(ft), s.Sizeof(ft)
			fp := s.ptrdata(ft)
			o = align(o, a)
			if fp != 0 {
				p = o + fp
			}
			o += sz
		}
		return p
	}

	panic("impossible")
}
