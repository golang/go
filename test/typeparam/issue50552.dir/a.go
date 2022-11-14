package a

type Builder[T any] struct{}

func (r Builder[T]) New() T {
	var v T
	return v
}

func (r Builder[T]) New2() T {
	return r.New()
}

func BuildInt() int {
	return Builder[int]{}.New()
}
