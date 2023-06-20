package main


type innerT[T any, R *T1[T]] struct {
	Ref R
}

type T1[T any] struct {
	e innerT[T, *T1[T]]
}

func main() {
	//Output:
}
