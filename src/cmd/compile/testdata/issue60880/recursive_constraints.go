package main

import "fmt"

type T1[T any] struct {
	e innerT[T, *T1[T]]
}

func main() {
	fmt.Println("didnt panic")
	//Ouput:
	//./prog.go:5:6: invalid recursive type T1
	//	./prog.go:5:6: T1 refers to
	//	./prog.go:13:6: innerT refers to
	//	./prog.go:5:6: T1
}

type innerT[T any, R *T1[T]] struct {
	Ref R
}
