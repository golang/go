package main

import "fmt"

type S1 struct {
	a, b, c []int //important removing this removes ICE

	i int
}

type S2 struct {
	a, b []int
	m    map[int]int
}

func F(i int, f func(S1, S2, int) int) int {
	return f(
		S1{},
		S2{m: map[int]int{}},
		1<<i)
}

func main() {
	fmt.Println("Hello, Hiro")
}
