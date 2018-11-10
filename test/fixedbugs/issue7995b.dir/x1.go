package x1

import "fmt"

var P int

//go:noinline
func F(x *int) string {
	P = 50
	*x = 100
	return fmt.Sprintln(P, *x)
}
