// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepcomplete

import "context"

type deepA struct {
	b deepB //@item(deepBField, "b", "deepB", "field")
}

type deepB struct {
}

func wantsDeepB(deepB) {}

func _() {
	var a deepA   //@item(deepAVar, "a", "deepA", "var")
	a.b           //@item(deepABField, "a.b", "deepB", "field")
	wantsDeepB(a) //@complete(")", deepABField, deepAVar)

	deepA{a} //@snippet("}", deepABField, "a.b", "a.b")
}

func wantsContext(context.Context) {}

func _() {
	context.Background() //@item(ctxBackground, "context.Background", "func() context.Context", "func", "Background returns a non-nil, empty Context.")
	context.TODO()       //@item(ctxTODO, "context.TODO", "func() context.Context", "func", "TODO returns a non-nil, empty Context.")

	wantsContext(c) //@completePartial(")", ctxBackground, ctxTODO)
}

func _() {
	// deepCircle is circular.
	type deepCircle struct {
		*deepCircle
	}
	var circle deepCircle   //@item(deepCircle, "circle", "deepCircle", "var")
	circle.deepCircle       //@item(deepCircleField, "circle.deepCircle", "*deepCircle", "field", "deepCircle is circular.")
	var _ deepCircle = circ //@complete(" //", deepCircle, deepCircleField)
}

func _() {
	type deepEmbedC struct {
	}
	type deepEmbedB struct {
		deepEmbedC
	}
	type deepEmbedA struct {
		deepEmbedB
	}

	wantsC := func(deepEmbedC) {}

	var a deepEmbedA //@item(deepEmbedA, "a", "deepEmbedA", "var")
	a.deepEmbedB     //@item(deepEmbedB, "a.deepEmbedB", "deepEmbedB", "field")
	a.deepEmbedC     //@item(deepEmbedC, "a.deepEmbedC", "deepEmbedC", "field")
	wantsC(a)        //@complete(")", deepEmbedC, deepEmbedA, deepEmbedB)
}

func _() {
	type nested struct {
		a int
		n *nested //@item(deepNestedField, "n", "*nested", "field")
	}

	nested{
		a: 123, //@complete(" //", deepNestedField)
	}
}

func _() {
	var a struct {
		b struct {
			c int
		}
		d int
	}

	a.d   //@item(deepAD, "a.d", "int", "field")
	a.b.c //@item(deepABC, "a.b.c", "int", "field")
	a.b   //@item(deepAB, "a.b", "struct{...}", "field")
	a     //@item(deepA, "a", "struct{...}", "var")

	// "a.d" should be ranked above the deeper "a.b.c"
	var i int
	i = a //@complete(" //", deepAD, deepABC, deepA, deepAB)
}
