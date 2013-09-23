package multi

func g(x int) {
}

func f() {
	x := 1
	g(x) // "g(x)" is the selection for multiple queries
}

func main() {
	f()
}
