package fix_1_single_file

// It doesn't matter if the innerT struct unexported or exported, the result is the same.
// It also doesn't matter if the R type parameter is infer to a pointer or not, the result is the same.
type innerT[T any, R T1[T] | T2[T]] struct {
	reference *R
}

type T1[T any] struct {
	e *innerT[T, T1[T]]
}

type T2[T any] struct {
	e *innerT[T, T2[T]]
}
