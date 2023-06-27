package main


type T2[T any] struct {
	e *innerT[T, *T2[T]]
}
