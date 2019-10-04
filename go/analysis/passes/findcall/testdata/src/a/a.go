package main // want package:"found"

func main() {
	println("hi") // want "call of println"
	print("hi")   // not a call of println
}

func println(s string) {} // want println:"found"
