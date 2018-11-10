package main

import (
	"fmt"
	"runtime"
)

var failed = false

func checkDivByZero(f func()) (divByZero bool) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(runtime.Error); ok && e.Error() == "runtime error: integer divide by zero" {
				divByZero = true
			}
		}
	}()
	f()
	return false
}

//go:noinline
func a(i uint, s []int) int {
	return s[i%uint(len(s))]
}

//go:noinline
func b(i uint, j uint) uint {
	return i / j
}

//go:noinline
func c(i int) int {
	return 7 / (i - i)
}

func main() {
	if got := checkDivByZero(func() { b(7, 0) }); !got {
		fmt.Printf("expected div by zero for b(7, 0), got no error\n")
		failed = true
	}
	if got := checkDivByZero(func() { b(7, 7) }); got {
		fmt.Printf("expected no error for b(7, 7), got div by zero\n")
		failed = true
	}
	if got := checkDivByZero(func() { a(4, nil) }); !got {
		fmt.Printf("expected div by zero for a(4, nil), got no error\n")
		failed = true
	}
	if got := checkDivByZero(func() { c(5) }); !got {
		fmt.Printf("expected div by zero for c(5), got no error\n")
		failed = true
	}

	if failed {
		panic("tests failed")
	}
}
