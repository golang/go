package main

func F(i int, f func(int) int) int {
	return f(
		1 <<i,
	)
}

func main() {
	println(F(2, func(i int) int {
		println("i is",i)
		return i
	}))
	println("Hello, Hiro")
}
