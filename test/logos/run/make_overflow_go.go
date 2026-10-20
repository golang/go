// run

package main

// M15-S2 (SPEC Sec 15.2, T-303; #169, R-U1): the same expression in a Go
// package keeps the wrapped upstream length and does not panic. The
// overflow check is a loGo/s language rule and stays in .lgo.

import "fmt"

func main() {
	w, h := 1<<32, 1<<32
	buf := make([]byte, w*h)
	fmt.Printf("len=%d\n", len(buf))
}
