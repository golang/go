// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepcomplete

import "context" //@item(ctxPackage, "context", "\"context\"", "package")

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
	context.Background() //@item(ctxBackground, "context.Background", "func() context.Context", "func")
	context.TODO()       //@item(ctxTODO, "context.TODO", "func() context.Context", "func")
	/* context.WithValue(parent context.Context, key interface{}, val interface{}) */ //@item(ctxWithValue, "context.WithValue", "func(parent context.Context, key interface{}, val interface{}) context.Context", "func")

	wantsContext(c) //@complete(")", ctxBackground, ctxTODO, ctxWithValue, ctxPackage)
}

func _() {
	type deepCircle struct {
		*deepCircle
	}
	var circle deepCircle //@item(deepCircle, "circle", "deepCircle", "var")
	circle.deepCircle     //@item(deepCircleField, "circle.deepCircle", "*deepCircle", "field")
	var _ deepCircle = ci //@complete(" //", deepCircle, deepCircleField)
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
