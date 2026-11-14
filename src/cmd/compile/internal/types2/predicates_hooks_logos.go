// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

func logosPredTupleComparable(t *LogosTuple, dynamic bool, seen map[Type]bool) *typeError {
	for _, f := range t.underlyingStruct().fields {
		if comparableType(f.typ, dynamic, seen) != nil {
			return typeErrorf("tuple containing %s cannot be compared", f.typ)
		}
	}
	return nil
}

func logosPredReadonlyHasNil(u *ReadonlyDescendants) bool {
	return hasNil(u.Elem)
}

func (c *comparer) logosPredTupleIdentical(x *LogosTuple, y Type, p *ifacePair) bool {
	if y, ok := y.(*LogosTuple); ok {
		return c.structIdentical(x.underlyingStruct(), y.underlyingStruct(), p)
	}
	if st, ok := y.(*Struct); ok {
		return c.structIdentical(x.underlyingStruct(), st, p)
	}
	return false
}

func (c *comparer) logosPredReadonlyIdentical(x *ReadonlyDescendants, y Type, p *ifacePair) bool {
	if y, ok := y.(*ReadonlyDescendants); ok {
		return c.identical(x.Elem, y.Elem, p)
	}
	return false
}

func (c *comparer) structIdentical(x, y *Struct, p *ifacePair) bool {
	if x.NumFields() != y.NumFields() {
		return false
	}
	for i, f := range x.fields {
		g := y.fields[i]
		if f.embedded != g.embedded ||
			!c.ignoreTags && x.Tag(i) != y.Tag(i) ||
			!f.sameId(g.pkg, g.name, false) ||
			!c.identical(f.typ, g.typ, p) {
			return false
		}
	}
	return true
}
