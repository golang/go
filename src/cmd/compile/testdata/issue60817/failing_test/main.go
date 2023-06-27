package main


// It doesn't matter if the innerT struct unexported or exported, the result is the same.
// It also doesn't matter if the R type parameter is infer to a pointer or not, the result is the same.

type innerT[T any, R *T1[T] | *T2[T]] struct {
	Ref R
}

type T1[T any] struct {
	e *innerT[T, *T1[T]]
}

func main() {
}
