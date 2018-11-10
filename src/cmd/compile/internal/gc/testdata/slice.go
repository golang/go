// run

// This test makes sure that t.s = t.s[0:x] doesn't write
// either the slice pointer or the capacity.
// See issue #14855.

package main

import "fmt"

const N = 1000000

type T struct {
	s []int
}

func main() {
	done := make(chan struct{})
	a := make([]int, N+10)

	t := &T{a}

	go func() {
		for i := 0; i < N; i++ {
			t.s = t.s[1:9]
		}
		done <- struct{}{}
	}()
	go func() {
		for i := 0; i < N; i++ {
			t.s = t.s[0:8] // should only write len
		}
		done <- struct{}{}
	}()
	<-done
	<-done

	ok := true
	if cap(t.s) != cap(a)-N {
		fmt.Printf("wanted cap=%d, got %d\n", cap(a)-N, cap(t.s))
		ok = false
	}
	if &t.s[0] != &a[N] {
		fmt.Printf("wanted ptr=%p, got %p\n", &a[N], &t.s[0])
		ok = false
	}
	if !ok {
		panic("bad")
	}
}
