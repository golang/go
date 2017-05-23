package main

var failed = false

//go:noinline
func testSliceLenCap12_ssa(a [10]int, i, j int) (int, int) {
	b := a[i:j]
	return len(b), cap(b)
}

//go:noinline
func testSliceLenCap1_ssa(a [10]int, i, j int) (int, int) {
	b := a[i:]
	return len(b), cap(b)
}

//go:noinline
func testSliceLenCap2_ssa(a [10]int, i, j int) (int, int) {
	b := a[:j]
	return len(b), cap(b)
}

func testSliceLenCap() {
	a := [10]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	tests := [...]struct {
		fn   func(a [10]int, i, j int) (int, int)
		i, j int // slice range
		l, c int // len, cap
	}{
		// -1 means the value is not used.
		{testSliceLenCap12_ssa, 0, 0, 0, 10},
		{testSliceLenCap12_ssa, 0, 1, 1, 10},
		{testSliceLenCap12_ssa, 0, 10, 10, 10},
		{testSliceLenCap12_ssa, 10, 10, 0, 0},
		{testSliceLenCap12_ssa, 0, 5, 5, 10},
		{testSliceLenCap12_ssa, 5, 5, 0, 5},
		{testSliceLenCap12_ssa, 5, 10, 5, 5},
		{testSliceLenCap1_ssa, 0, -1, 0, 10},
		{testSliceLenCap1_ssa, 5, -1, 5, 5},
		{testSliceLenCap1_ssa, 10, -1, 0, 0},
		{testSliceLenCap2_ssa, -1, 0, 0, 10},
		{testSliceLenCap2_ssa, -1, 5, 5, 10},
		{testSliceLenCap2_ssa, -1, 10, 10, 10},
	}

	for i, t := range tests {
		if l, c := t.fn(a, t.i, t.j); l != t.l && c != t.c {
			println("#", i, " len(a[", t.i, ":", t.j, "]), cap(a[", t.i, ":", t.j, "]) =", l, c,
				", want", t.l, t.c)
			failed = true
		}
	}
}

//go:noinline
func testSliceGetElement_ssa(a [10]int, i, j, p int) int {
	return a[i:j][p]
}

func testSliceGetElement() {
	a := [10]int{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}
	tests := [...]struct {
		i, j, p int
		want    int // a[i:j][p]
	}{
		{0, 10, 2, 20},
		{0, 5, 4, 40},
		{5, 10, 3, 80},
		{1, 9, 7, 80},
	}

	for i, t := range tests {
		if got := testSliceGetElement_ssa(a, t.i, t.j, t.p); got != t.want {
			println("#", i, " a[", t.i, ":", t.j, "][", t.p, "] = ", got, " wanted ", t.want)
			failed = true
		}
	}
}

//go:noinline
func testSliceSetElement_ssa(a *[10]int, i, j, p, x int) {
	(*a)[i:j][p] = x
}

func testSliceSetElement() {
	a := [10]int{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}
	tests := [...]struct {
		i, j, p int
		want    int // a[i:j][p]
	}{
		{0, 10, 2, 17},
		{0, 5, 4, 11},
		{5, 10, 3, 28},
		{1, 9, 7, 99},
	}

	for i, t := range tests {
		testSliceSetElement_ssa(&a, t.i, t.j, t.p, t.want)
		if got := a[t.i+t.p]; got != t.want {
			println("#", i, " a[", t.i, ":", t.j, "][", t.p, "] = ", got, " wanted ", t.want)
			failed = true
		}
	}
}

func testSlicePanic1() {
	defer func() {
		if r := recover(); r != nil {
			println("paniced as expected")
		}
	}()

	a := [10]int{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}
	testSliceLenCap12_ssa(a, 3, 12)
	println("expected to panic, but didn't")
	failed = true
}

func testSlicePanic2() {
	defer func() {
		if r := recover(); r != nil {
			println("paniced as expected")
		}
	}()

	a := [10]int{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}
	testSliceGetElement_ssa(a, 3, 7, 4)
	println("expected to panic, but didn't")
	failed = true
}

func main() {
	testSliceLenCap()
	testSliceGetElement()
	testSliceSetElement()
	testSlicePanic1()
	testSlicePanic2()

	if failed {
		panic("failed")
	}
}
