// errorcheck

// //logos:zeroalloc on a Go package (R-U1): logos lint runs the check inside
// escape analysis of Go code, so the Go twin of test/logos/codegen/zeroalloc.lgo
// reports the same allocation sites. Only logos lint reads the directive in a
// Go package; logos build and go build ignore it, which logos_zeroalloc.txt
// proves.

package p

type point struct{ x, y int }

//logos:zeroalloc
func total(xs []int) int {
	n := 0
	for _, x := range xs {
		n += x
	}
	return n
}

//logos:zeroalloc
func pack(v int) []int {
	return []int{v} // ERROR "\[perf::zeroalloc_violation\] pack allocates: \[\]int\{\.\.\.\} escapes to heap: flows to result ~r0 \(return\)"
}

//logos:zeroalloc
func (p *point) clone() *point {
	return &point{p.x, p.y} // ERROR "\[perf::zeroalloc_violation\] \(\*point\)\.clone allocates: &point\{\.\.\.\} escapes to heap: flows to result ~r0 \(return\)"
}

//logos:zeroalloc
func constantPanic(i int, xs []int) int {
	if i < 0 {
		panic("negative index")
	}
	return xs[i]
}

// unmarked allocates freely: only marked functions are checked.
func unmarked(v int) []int {
	return []int{v}
}
