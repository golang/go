package foo

func f(m, n int) int {
	a := g(n)
	b := g(m)
	return a + b
}

func g(x int) int {
	y := h(x + 1)
	z := h(x - 1)
	return y + z
}

func h(x int) int {
	return x * x
}
