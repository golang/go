package main

import "testing"

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

func testSliceLenCap(t *testing.T) {
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

	for i, test := range tests {
		if l, c := test.fn(a, test.i, test.j); l != test.l && c != test.c {
			t.Errorf("#%d len(a[%d:%d]), cap(a[%d:%d]) = %d %d, want %d %d", i, test.i, test.j, test.i, test.j, l, c, test.l, test.c)
		}
	}
}

//go:noinline
func testSliceGetElement_ssa(a [10]int, i, j, p int) int {
	return a[i:j][p]
}

func testSliceGetElement(t *testing.T) {
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

	for i, test := range tests {
		if got := testSliceGetElement_ssa(a, test.i, test.j, test.p); got != test.want {
			t.Errorf("#%d a[%d:%d][%d] = %d, wanted %d", i, test.i, test.j, test.p, got, test.want)
		}
	}
}

//go:noinline
func testSliceSetElement_ssa(a *[10]int, i, j, p, x int) {
	(*a)[i:j][p] = x
}

func testSliceSetElement(t *testing.T) {
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

	for i, test := range tests {
		testSliceSetElement_ssa(&a, test.i, test.j, test.p, test.want)
		if got := a[test.i+test.p]; got != test.want {
			t.Errorf("#%d a[%d:%d][%d] = %d, wanted %d", i, test.i, test.j, test.p, got, test.want)
		}
	}
}

func testSlicePanic1(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			//println("panicked as expected")
		}
	}()

	a := [10]int{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}
	testSliceLenCap12_ssa(a, 3, 12)
	t.Errorf("expected to panic, but didn't")
}

func testSlicePanic2(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			//println("panicked as expected")
		}
	}()

	a := [10]int{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}
	testSliceGetElement_ssa(a, 3, 7, 4)
	t.Errorf("expected to panic, but didn't")
}

func TestArray(t *testing.T) {
	testSliceLenCap(t)
	testSliceGetElement(t)
	testSliceSetElement(t)
	testSlicePanic1(t)
	testSlicePanic2(t)
}
