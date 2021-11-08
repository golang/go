package main

import (
	"fmt"

	b "cmd/link/internal/ld/testdata/issue26237/b.dir"
)

var skyx int

func main() {
	skyx += b.OOO(skyx)
	if b.Top(1) == 99 {
		fmt.Printf("Beware the Jabberwock, my son!\n")
	}
}
