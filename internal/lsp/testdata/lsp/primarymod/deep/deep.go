package deep

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
	wantsDeepB(a) //@deep(")", deepABField, deepAVar)

	deepA{a} //@snippet("}", deepABField, "a.b", "a.b")
}

func wantsContext(context.Context) {}

func _() {
	context.Background() //@item(ctxBackground, "context.Background", "func() context.Context", "func", "Background returns a non-nil, empty Context.")
	context.TODO()       //@item(ctxTODO, "context.TODO", "func() context.Context", "func", "TODO returns a non-nil, empty Context.")

	wantsContext(c) //@rank(")", ctxBackground),rank(")", ctxTODO)
}

func _() {
	// deepCircle is circular.
	type deepCircle struct {
		*deepCircle
	}
	var circle deepCircle   //@item(deepCircle, "circle", "deepCircle", "var")
	*circle.deepCircle      //@item(deepCircleField, "*circle.deepCircle", "*deepCircle", "field", "deepCircle is circular.")
	var _ deepCircle = circ //@deep(" //", deepCircle, deepCircleField)
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
	wantsC(a)        //@deep(")", deepEmbedC, deepEmbedA, deepEmbedB)
}

func _() {
	type nested struct {
		a int
		n *nested //@item(deepNestedField, "n", "*nested", "field")
	}

	nested{
		a: 123, //@deep(" //", deepNestedField)
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
	i = a //@deep(" //", deepAD, deepABC, deepA, deepAB)
}

type foo struct {
	b bar
}

func (f foo) bar() bar {
	return f.b
}

func (f foo) barPtr() *bar {
	return &f.b
}

type bar struct{}

func (b bar) valueReceiver() int {
	return 0
}

func (b *bar) ptrReceiver() int {
	return 0
}

func _() {
	var (
		i int
		f foo
	)

	f.bar().valueReceiver    //@item(deepBarValue, "f.bar().valueReceiver", "func() int", "method")
	f.barPtr().ptrReceiver   //@item(deepBarPtrPtr, "f.barPtr().ptrReceiver", "func() int", "method")
	f.barPtr().valueReceiver //@item(deepBarPtrValue, "f.barPtr().valueReceiver", "func() int", "method")

	i = fbar //@fuzzy(" //", deepBarValue, deepBarPtrPtr, deepBarPtrValue)
}

func (b baz) Thing() struct{ val int } {
	return b.thing
}

type baz struct {
	thing struct{ val int }
}

func (b baz) _() {
	b.Thing().val //@item(deepBazMethVal, "b.Thing().val", "int", "field")
	b.thing.val   //@item(deepBazFieldVal, "b.thing.val", "int", "field")
	var _ int = b //@rank(" //", deepBazFieldVal, deepBazMethVal)
}
