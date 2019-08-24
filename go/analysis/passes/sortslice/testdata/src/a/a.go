package a

import "sort"

// IncorrectSort tries to sort an integer.
func IncorrectSort() {
	i := 5
	sortFn := func(i, j int) bool { return false }
	sort.Slice(i, sortFn) // want "sort.Slice's argument must be a slice; is called with int"
}

// CorrectSort sorts integers. It should not produce a diagnostic.
func CorrectSort() {
	s := []int{2, 3, 5, 6}
	sortFn := func(i, j int) bool { return s[i] < s[j] }
	sort.Slice(s, sortFn)
}

// CorrectInterface sorts an interface with a slice
// as the concrete type. It should not produce a diagnostic.
func CorrectInterface() {
	var s interface{}
	s = interface{}([]int{2, 1, 0})
	sortFn := func(i, j int) bool { return s.([]int)[i] < s.([]int)[j] }
	sort.Slice(s, sortFn)
}

type slicecompare interface {
	compare(i, j int) bool
}

type intslice []int

func (s intslice) compare(i, j int) bool {
	return s[i] < s[j]
}

// UnderlyingInterface sorts an interface with a slice
// as the concrete type. It should not produce a diagnostic.
func UnderlyingInterface() {
	var s slicecompare
	s = intslice([]int{2, 1, 0})
	sort.Slice(s, s.compare)
}

type mySlice []int

// UnderlyingSlice sorts a type with an underlying type of
// slice of ints. It should not produce a diagnostic.
func UnderlyingSlice() {
	s := mySlice{2, 3, 5, 6}
	sortFn := func(i, j int) bool { return s[i] < s[j] }
	sort.Slice(s, sortFn)
}
