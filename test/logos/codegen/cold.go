// asmcheck -O=2

package main

import (
	"errors"
	"fmt"
	"os"
)

// M13-S2 synthetic PGO (SPEC T-105), Go twin of cold.lgo: Go packages get
// the os.Exit mark from the compiler with no source change and no new
// import, so the os.Exit call itself survives while no Cold call appears.
// Run with go under the same directives.

func fail() (int, error) {
	return 0, errors.New("boom")
}

func check(n int) int {
	if n < 0 {
		// amd64:"CALL.*os.Exit" arm64:"CALL.*os.Exit" amd64:-"CALL.*Cold" arm64:-"CALL.*Cold"
		os.Exit(1)
	}
	return n
}

func main() {
	_, err := fail()
	fmt.Println(err)
	fmt.Println(check(3))
}
