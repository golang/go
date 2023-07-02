package main

import "workspace/with_generics"



type innerT1[T any, R T1[T]] struct {
	reference *R
}


type T1[T any] struct {
	e   *innerT1[T, T1[T]]
	val T
}

func main() {

	_ = with_generics.T1[any]{
	}
}
