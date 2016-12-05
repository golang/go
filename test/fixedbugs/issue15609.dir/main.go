package main

var called bool

func target() {
	called = true
}

func main() {
	jump()
	if !called {
		panic("target not called")
	}
}
