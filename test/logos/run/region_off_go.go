// run
package main

import (
	"fmt"
	"logos/arena"
	"os"
	"slices"
)

func main() {
	failed := false
	check := func(name string, ok bool) {
		if !ok {
			fmt.Printf("FAIL %s\n", name)
			failed = true
			return
		}
		fmt.Printf("ok %s\n", name)
	}
	r := arena.Open()

	p := arena.New[int](r)
	*p = 42
	hp := new(int)
	*hp = 42
	check("experiment-off-new-matches-heap", *p == *hp && *p == 42)

	s := arena.Make[int](r, 3)
	s[0], s[1], s[2] = 1, 2, 3
	s = arena.Append(r, s, 4)
	s = arena.Append(r, s, 5, 6)
	hs := make([]int, 3)
	hs[0], hs[1], hs[2] = 1, 2, 3
	hs = append(hs, 4)
	hs = append(hs, 5, 6)
	check("experiment-off-make-append-matches-heap", slices.Equal(s, hs) && slices.Equal(s, []int{1, 2, 3, 4, 5, 6}))

	m := arena.Make[int](r, 5)
	hm := make([]int, 5)
	check("experiment-off-zeroed-make-matches-heap", len(m) == len(hm) && slices.Equal(m, hm))

	g := arena.Append(r, []int(nil), 9)
	hg := append([]int(nil), 9)
	check("experiment-off-append-nil-matches-heap", slices.Equal(g, hg) && slices.Equal(g, []int{9}))

	keep := arena.Append(r, s)
	check("experiment-off-append-copy-preserves-values", slices.Equal(keep, s))
	if failed {
		os.Exit(1)
	}
}
