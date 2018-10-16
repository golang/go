package a

func Fib(x int) int {
	if x < 2 {
		return x
	}
	return Fib(x-1) + Fib(x-2)
}

type T int

func (T) fib(x int) int { return Fib(x) }

func _() {
	print("hi")
}
