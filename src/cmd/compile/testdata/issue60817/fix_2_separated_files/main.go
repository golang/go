package main

import (
	"fmt"
)

// It doesn't matter if the innerT struct unexported or exported, the result is the same.
// It also doesn't matter if the R type parameter is infer to a pointer or not, the result is the same.

type constraint[T any] interface {
	*T1[T] | *T2[T]
}
type innerT[T any, R constraint[T]] struct {
	Ref R
}

type T1[T any] struct {
	e *innerT[T, *T1[T]]
}

type innerH struct {
	i *H1
	y *H2
}

type H1 struct {
	h1 *H1
}

func main() {
	fmt.Println("didnt panic")

	_ = innerH{}

}
