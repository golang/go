// errorcheck

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

//logos:zeroalloc
func bytesLen(s string) int {
	b := []byte(s)
	return len(b)
}

//logos:zeroalloc
func starLen(s *string) int {
	t := (*string)(s)
	return len(*t)
}

// unmarked allocates freely: only marked functions are checked.
func unmarked(v int) []int {
	return []int{v}
}
