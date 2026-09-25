package main

func f(i int) bool { return i > 1 }

func g() {}

func main() {
	switch { // want a statement here, before the cases
	case f(1):
		g()
	case f(2):
		g()
	}
}
