// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// TypeParamList holds a list of type parameters.
type TypeParamList struct{ tparams []*TypeParam }

// Len returns the number of type parameters in the list.
// It is safe to call on a nil receiver.
func (l *TypeParamList) Len() int { return len(l.list()) }

// At returns the i'th type parameter in the list.
func (l *TypeParamList) At(i int) *TypeParam { return l.tparams[i] }

// list is for internal use where we expect a []*TypeParam.
// TODO(rfindley): list should probably be eliminated: we can pass around a
// TypeParamList instead.
func (l *TypeParamList) list() []*TypeParam {
	if l == nil {
		return nil
	}
	return l.tparams
}

// TypeList holds a list of types.
type TypeList struct{ types []Type }

// newTypeList returns a new TypeList with the types in list.
func newTypeList(list []Type) *TypeList {
	if len(list) == 0 {
		return nil
	}
	return &TypeList{list}
}

// Len returns the number of types in the list.
// It is safe to call on a nil receiver.
func (l *TypeList) Len() int { return len(l.list()) }

// At returns the i'th type in the list.
func (l *TypeList) At(i int) Type { return l.types[i] }

// list is for internal use where we expect a []Type.
// TODO(rfindley): list should probably be eliminated: we can pass around a
// TypeList instead.
func (l *TypeList) list() []Type {
	if l == nil {
		return nil
	}
	return l.types
}

// ----------------------------------------------------------------------------
// Implementation

func bindTParams(list []*TypeParam) *TypeParamList {
	if len(list) == 0 {
		return nil
	}
	for i, typ := range list {
		if typ.index >= 0 {
			panic("type parameter bound more than once")
		}
		typ.index = i
	}
	return &TypeParamList{tparams: list}
}
