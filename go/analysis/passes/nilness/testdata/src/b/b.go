package b

func f() {
	var s []int
	t := (*[0]int)(s)
	_ = *t // want "nil dereference in load"
	_ = (*[0]int)(s)
	_ = *(*[0]int)(s) // want "nil dereference in load"

	// these operation is panic
	_ = (*[1]int)(s)  // want "nil slice being cast to an array of len > 0 will always panic"
	_ = *(*[1]int)(s) // want "nil slice being cast to an array of len > 0 will always panic"
}

func g() {
	var s = make([]int, 0)
	t := (*[0]int)(s)
	println(*t)
}

func h() {
	var s = make([]int, 1)
	t := (*[1]int)(s)
	println(*t)
}

func i(x []int) {
	a := (*[1]int)(x)
	if a != nil { // want "tautological condition: non-nil != nil"
		_ = *a
	}
}
