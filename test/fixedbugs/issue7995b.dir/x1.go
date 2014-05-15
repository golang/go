package x1

import "fmt"

var P int

var b bool

func F(x *int) string {
	if b { // avoid inlining
		F(x)
	}
	P = 50
	*x = 100
	return fmt.Sprintln(P, *x)
}
