package main


/*type T2[T any] struct {
	e *innerT[T, *T2[T]]
}*/

type H2 struct {
	h *H2
}
type T2[T any] struct {
	e *innerT[T, *T2[T]]
}
