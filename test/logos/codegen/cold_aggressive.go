// asmcheck -O=3

package main

import "fmt"

// M13-S2 synthetic PGO (SPEC T-105), Go twin of cold_aggressive.lgo: the
// explicit-panic path is marked without any source change, so the panic
// still panics and no Cold call appears. Run with go under the same
// directives.

func main() {
	fmt.Println(mustBePositive(3))
}

// amd64:-"CALL.*Cold" arm64:-"CALL.*Cold"
func mustBePositive(n int) int {
	if n < 0 {
		panic("negative")
	}
	return n
}
