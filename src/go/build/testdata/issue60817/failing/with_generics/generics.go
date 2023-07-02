package with_generics

// It doesn't matter if the innerT1 struct unexported or exported, the result is the same.
//
// Within the deadlock issue, it does matter if the type parameter R as a pointer, and the inner field that
// referenced to type parameter R is a pointer (double reference) or not, it behaves differently. However,
// it may behave differently based on different issues.

type innerT1[T any, R T1[T]] struct {
	reference *R
}


type T1[T any] struct {
	e   *innerT1[T, T1[T]]
	val T
}
