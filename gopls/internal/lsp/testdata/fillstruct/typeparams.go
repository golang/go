//go:build go1.18
// +build go1.18

package fillstruct

type emptyStructWithTypeParams[A any] struct{}

var _ = emptyStructWithTypeParams[int]{}

type basicStructWithTypeParams[T any] struct {
	foo T
}

var _ = basicStructWithTypeParams[int]{} //@suggestedfix("}", "refactor.rewrite", "Fill")

type twoArgStructWithTypeParams[F, B any] struct {
	foo F
	bar B
}

var _ = twoArgStructWithTypeParams[string, int]{} //@suggestedfix("}", "refactor.rewrite", "Fill")

var _ = twoArgStructWithTypeParams[int, string]{
	bar: "bar",
} //@suggestedfix("}", "refactor.rewrite", "Fill")

type nestedStructWithTypeParams struct {
	bar   string
	basic basicStructWithTypeParams[int]
}

var _ = nestedStructWithTypeParams{}

func _[T any]() {
	type S struct{ t T }
	x := S{}
	_ = x
}
