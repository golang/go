package main



type S1 struct {
	a, b, c []int
	i       int
}

type S2 struct {
	a, b []int
	m    map[int]int
}

func F(i int, f func(S1, S2, int) int) int {
	return f(
		S1{a: []int{}},
		S2{b: []int{}},
		1>>i,
	)
}

func main() {

	println("Hello, Hiro")
}
