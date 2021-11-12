// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/token"
	"testing"
)

func TestContextHashCollisions(t *testing.T) {
	if debug {
		t.Skip("hash collisions are expected, and would fail debug assertions")
	}
	// Unit test the de-duplication fall-back logic in Context.
	//
	// We can't test this via Instantiate because this is only a fall-back in
	// case our hash is imperfect.
	//
	// These lookups and updates use reasonable looking types in an attempt to
	// make them robust to internal type assertions, but could equally well use
	// arbitrary types.

	// Create some distinct origin types. nullaryP and nullaryQ have no
	// parameters and are identical (but have different type parameter names).
	// unaryP has a parameter.
	var nullaryP, nullaryQ, unaryP Type
	{
		// type nullaryP = func[P any]()
		tparam := NewTypeParam(NewTypeName(token.NoPos, nil, "P", nil), &emptyInterface)
		nullaryP = NewSignatureType(nil, nil, []*TypeParam{tparam}, nil, nil, false)
	}
	{
		// type nullaryQ = func[Q any]()
		tparam := NewTypeParam(NewTypeName(token.NoPos, nil, "Q", nil), &emptyInterface)
		nullaryQ = NewSignatureType(nil, nil, []*TypeParam{tparam}, nil, nil, false)
	}
	{
		// type unaryP = func[P any](_ P)
		tparam := NewTypeParam(NewTypeName(token.NoPos, nil, "P", nil), &emptyInterface)
		params := NewTuple(NewVar(token.NoPos, nil, "_", tparam))
		unaryP = NewSignatureType(nil, nil, []*TypeParam{tparam}, params, nil, false)
	}

	ctxt := NewContext()

	// Update the context with an instantiation of nullaryP.
	inst := NewSignatureType(nil, nil, nil, nil, nil, false)
	if got := ctxt.update("", nullaryP, []Type{Typ[Int]}, inst); got != inst {
		t.Error("bad")
	}

	// unaryP is not identical to nullaryP, so we should not get inst when
	// instantiated with identical type arguments.
	if got := ctxt.lookup("", unaryP, []Type{Typ[Int]}); got != nil {
		t.Error("bad")
	}

	// nullaryQ is identical to nullaryP, so we *should* get inst when
	// instantiated with identical type arguments.
	if got := ctxt.lookup("", nullaryQ, []Type{Typ[Int]}); got != inst {
		t.Error("bad")
	}

	// ...but verify we don't get inst with different type arguments.
	if got := ctxt.lookup("", nullaryQ, []Type{Typ[String]}); got != nil {
		t.Error("bad")
	}
}
