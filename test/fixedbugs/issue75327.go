// run

package main

import (
	"fmt"
	"strings"
)

func main() {
	defer func() {
		err := recover()
		txt := fmt.Sprintf("%s", err)
		if !strings.HasSuffix(txt, "with length 1") {
			panic("bad error: " + txt)
		}
	}()
	foo([]uint64{0})
}

//go:noinline
func foo(haystack []uint64) {
	for n := range len(haystack) {
		_ = n
		_ = haystack[1]
	}

	xxx := haystack[0:len(haystack)]
	sink(xxx)
}

//go:noinline
func sink([]uint64) {}
