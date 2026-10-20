// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import "cmd/compile/internal/types2"

// binShape names how the M11-S3 binary layout (SPEC Sec 12.3, T-Sec12.1)
// encodes one type.
type binShape uint8

const (
	binNone binShape = iota
	binBool
	binSigned
	binUnsigned
	binFloat32
	binFloat64
	binString
	binBytes
	binSlice
	binArray
	binMap
	binStruct
	binPointer
	binOption
	binMarshaler
)

// binMaxWidth caps a minimum wire width so nested arrays never overflow.
const binMaxWidth = 1 << 30

// binaryLayout decides which types the generated binary methods can lay
// out and how wide each is. pkg is the package that declares the struct
// under generation: its named structs lay out field by field, named
// structs of other packages only through their own MarshalBinary and
// UnmarshalBinary. visiting holds the named structs on the current walk,
// so a struct that reaches itself reports no layout.
type binaryLayout struct {
	fl       *fileLower
	pkg      *types2.Package
	visiting map[*types2.Named]bool
}

// newBinaryLayout starts a layout walk for the struct root.
func (fl *fileLower) newBinaryLayout(root *types2.Named) *binaryLayout {
	return &binaryLayout{
		fl:       fl,
		pkg:      root.Obj().Pkg(),
		visiting: map[*types2.Named]bool{root: true},
	}
}

// shape classifies the top level of t.
func (l *binaryLayout) shape(t types2.Type) binShape {
	if t == nil {
		return binNone
	}
	if l.fl.asOption(t) != nil {
		return binOption
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok && l.foreignStruct(n) {
		return foreignShape(n)
	}
	return underlyingShape(t.Underlying())
}

// foreignShape classifies a named struct of another package: it has a
// layout only through its own coders.
func foreignShape(n *types2.Named) binShape {
	if hasBinaryMarshaler(n) {
		return binMarshaler
	}
	return binNone
}

// underlyingShape classifies the underlying type of a non-option type.
func underlyingShape(u types2.Type) binShape {
	switch u := u.(type) {
	case *types2.Basic:
		return basicShape(u)
	case *types2.Slice:
		return sliceShape(u)
	case *types2.Array:
		return binArray
	case *types2.Map:
		return mapShape(u)
	case *types2.Struct:
		return binStruct
	case *types2.Pointer:
		return binPointer
	}
	return binNone
}

// sliceShape tells a byte slice from every other slice.
func sliceShape(s *types2.Slice) binShape {
	if types2.Identical(s.Elem(), types2.Typ[types2.Uint8]) {
		return binBytes
	}
	return binSlice
}

// basicShape maps a basic kind to its shape. Complex numbers and
// unsafe.Pointer have no layout.
func basicShape(b *types2.Basic) binShape {
	switch b.Kind() {
	case types2.Bool:
		return binBool
	case types2.Int, types2.Int8, types2.Int16, types2.Int32, types2.Int64:
		return binSigned
	case types2.Uint, types2.Uint8, types2.Uint16, types2.Uint32, types2.Uint64, types2.Uintptr:
		return binUnsigned
	case types2.Float32:
		return binFloat32
	case types2.Float64:
		return binFloat64
	case types2.String:
		return binString
	}
	return binNone
}

// mapShape accepts a map whose key type is a string kind.
func mapShape(m *types2.Map) binShape {
	if b, ok := m.Key().Underlying().(*types2.Basic); ok && b.Kind() == types2.String {
		return binMap
	}
	return binNone
}

// foreignStruct reports whether n is a named struct declared outside the
// package under generation.
func (l *binaryLayout) foreignStruct(n *types2.Named) bool {
	if _, ok := n.Underlying().(*types2.Struct); !ok {
		return false
	}
	return n.Obj().Pkg() != l.pkg
}

// hasBinaryMarshaler reports whether *n offers
// `MarshalBinary() ([]byte, error)` and `UnmarshalBinary([]byte) error`.
func hasBinaryMarshaler(n *types2.Named) bool {
	bytes := types2.NewSlice(types2.Typ[types2.Uint8])
	errType := types2.Universe.Lookup("error").Type()
	return sigIs(methodSig(n, "MarshalBinary"), nil, []types2.Type{bytes, errType}) &&
		sigIs(methodSig(n, "UnmarshalBinary"), []types2.Type{bytes}, []types2.Type{errType})
}

// sigIs reports whether sig takes exactly the types params and returns
// exactly the types results.
func sigIs(sig *types2.Signature, params, results []types2.Type) bool {
	return sig != nil && tupleIs(sig.Params(), params) && tupleIs(sig.Results(), results)
}

// tupleIs reports whether tup holds exactly the types want, in order.
func tupleIs(tup *types2.Tuple, want []types2.Type) bool {
	if tup.Len() != len(want) {
		return false
	}
	for i, w := range want {
		if !types2.Identical(tup.At(i).Type(), w) {
			return false
		}
	}
	return true
}

// methodSig returns the signature of the exported method name in the
// method set of *n, or nil.
func methodSig(n *types2.Named, name string) *types2.Signature {
	obj, _, _ := types2.LookupFieldOrMethod(n, true, nil, name)
	fn, ok := obj.(*types2.Func)
	if !ok {
		return nil
	}
	sig, _ := fn.Type().(*types2.Signature)
	return sig
}

// supported reports whether the whole type tree of t has a layout. A
// slice element that encodes in zero bytes has none, because a count
// header could then claim any number of elements without input behind it.
func (l *binaryLayout) supported(t types2.Type) bool {
	switch l.shape(t) {
	case binNone:
		return false
	case binStruct:
		return l.structSupported(t)
	case binSlice:
		elem := t.Underlying().(*types2.Slice).Elem()
		return l.supported(elem) && l.width(elem) > 0
	}
	elem, has := l.elemOf(t)
	return !has || l.supported(elem)
}

// elemOf returns the one type a pointer, option, array, or map layout
// wraps, and false for every other shape.
func (l *binaryLayout) elemOf(t types2.Type) (types2.Type, bool) {
	switch l.shape(t) {
	case binOption:
		return l.fl.asOption(t).Elem, true
	case binArray:
		return t.Underlying().(*types2.Array).Elem(), true
	case binMap:
		return t.Underlying().(*types2.Map).Elem(), true
	case binPointer:
		return t.Underlying().(*types2.Pointer).Elem(), true
	}
	return nil, false
}

// structSupported walks the fields of a struct type, marking a named
// struct for the duration of its walk.
func (l *binaryLayout) structSupported(t types2.Type) bool {
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if l.visiting[n] {
			return false
		}
		l.visiting[n] = true
		defer delete(l.visiting, n)
	}
	return l.fieldsSupported(t.Underlying().(*types2.Struct))
}

// fieldsSupported reports whether every field of st has a layout. The
// caller has already marked st's own named type as being walked.
func (l *binaryLayout) fieldsSupported(st *types2.Struct) bool {
	for i := range st.NumFields() {
		f := st.Field(i)
		if f.Embedded() || (f.Name() != "_" && !l.supported(f.Type())) {
			return false
		}
	}
	return true
}

// width returns the minimum number of bytes the layout spends on a value
// of a supported type t: the exact width of a fixed-size type, a lower
// bound for every other, and zero for a type with no layout. The
// unmarshaler divides the remaining input by it to bound a count header.
func (l *binaryLayout) width(t types2.Type) int {
	switch l.shape(t) {
	case binNone:
		return 0
	case binSigned, binUnsigned, binFloat64:
		return 8
	case binFloat32:
		return 4
	case binPointer:
		return l.width(t.Underlying().(*types2.Pointer).Elem())
	case binArray:
		return l.arrayWidth(t.Underlying().(*types2.Array))
	case binStruct:
		return l.structWidth(t.Underlying().(*types2.Struct))
	}
	return 1
}

// arrayWidth spells the count header plus every element.
func (l *binaryLayout) arrayWidth(a *types2.Array) int {
	return clampWidth(int64(uvarintLen(uint64(a.Len()))) + a.Len()*int64(l.width(a.Elem())))
}

// structWidth sums the field widths of st.
func (l *binaryLayout) structWidth(st *types2.Struct) int {
	var sum int64
	for i := range st.NumFields() {
		if f := st.Field(i); f.Name() != "_" {
			sum += int64(l.width(f.Type()))
		}
	}
	return clampWidth(sum)
}

// clampWidth bounds a width sum to binMaxWidth.
func clampWidth(w int64) int {
	if w > binMaxWidth || w < 0 {
		return binMaxWidth
	}
	return int(w)
}

// uvarintLen returns the encoded length of x as a uvarint.
func uvarintLen(x uint64) int {
	n := 1
	for ; x >= 0x80; x >>= 7 {
		n++
	}
	return n
}
