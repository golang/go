package main

//go:noinline
func blah(x int) int {
	if x != 0 {
		return x + 42
	}
	return x - 42
}

func main() {
	go infloop()
	println(blah(1) + blah(0))
}

var G int

func infloop() {
	for {
		G += blah(1)
		G += blah(0)
		if G > 10000 {
			G = 0
		}
	}
}
