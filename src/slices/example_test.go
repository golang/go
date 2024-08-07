// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slices_test

import (
	"cmp"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

func ExampleBinarySearch() {
	names := []string{"Alice", "Bob", "Vera"}
	n, found := slices.BinarySearch(names, "Vera")
	fmt.Println("Vera:", n, found)
	n, found = slices.BinarySearch(names, "Bill")
	fmt.Println("Bill:", n, found)
	// Output:
	// Vera: 2 true
	// Bill: 1 false
}

func ExampleBinarySearchFunc() {
	type Person struct {
		Name string
		Age  int
	}
	people := []Person{
		{"Alice", 55},
		{"Bob", 24},
		{"Gopher", 13},
	}
	n, found := slices.BinarySearchFunc(people, Person{"Bob", 0}, func(a, b Person) int {
		return strings.Compare(a.Name, b.Name)
	})
	fmt.Println("Bob:", n, found)
	// Output:
	// Bob: 1 true
}

func ExampleCompact() {
	seq := []int{0, 1, 1, 2, 3, 5, 8}
	seq = slices.Compact(seq)
	fmt.Println(seq)
	// Output:
	// [0 1 2 3 5 8]
}

func ExampleCompactFunc() {
	names := []string{"bob", "Bob", "alice", "Vera", "VERA"}
	names = slices.CompactFunc(names, strings.EqualFold)
	fmt.Println(names)
	// Output:
	// [bob alice Vera]
}

func ExampleCompare() {
	names := []string{"Alice", "Bob", "Vera"}
	fmt.Println("Equal:", slices.Compare(names, []string{"Alice", "Bob", "Vera"}))
	fmt.Println("V < X:", slices.Compare(names, []string{"Alice", "Bob", "Xena"}))
	fmt.Println("V > C:", slices.Compare(names, []string{"Alice", "Bob", "Cat"}))
	fmt.Println("3 > 2:", slices.Compare(names, []string{"Alice", "Bob"}))
	// Output:
	// Equal: 0
	// V < X: -1
	// V > C: 1
	// 3 > 2: 1
}

func ExampleCompareFunc() {
	numbers := []int{0, 43, 8}
	strings := []string{"0", "0", "8"}
	result := slices.CompareFunc(numbers, strings, func(n int, s string) int {
		sn, err := strconv.Atoi(s)
		if err != nil {
			return 1
		}
		return cmp.Compare(n, sn)
	})
	fmt.Println(result)
	// Output:
	// 1
}

func ExampleContainsFunc() {
	numbers := []int{0, 42, -10, 8}
	hasNegative := slices.ContainsFunc(numbers, func(n int) bool {
		return n < 0
	})
	fmt.Println("Has a negative:", hasNegative)
	hasOdd := slices.ContainsFunc(numbers, func(n int) bool {
		return n%2 != 0
	})
	fmt.Println("Has an odd number:", hasOdd)
	// Output:
	// Has a negative: true
	// Has an odd number: false
}

func ExampleDelete() {
	letters := []string{"a", "b", "c", "d", "e"}
	letters = slices.Delete(letters, 1, 4)
	fmt.Println(letters)
	// Output:
	// [a e]
}

func ExampleDeleteFunc() {
	seq := []int{0, 1, 1, 2, 3, 5, 8}
	seq = slices.DeleteFunc(seq, func(n int) bool {
		return n%2 != 0 // delete the odd numbers
	})
	fmt.Println(seq)
	// Output:
	// [0 2 8]
}

func ExampleEqual() {
	numbers := []int{0, 42, 8}
	fmt.Println(slices.Equal(numbers, []int{0, 42, 8}))
	fmt.Println(slices.Equal(numbers, []int{10}))
	// Output:
	// true
	// false
}

func ExampleEqualFunc() {
	numbers := []int{0, 42, 8}
	strings := []string{"000", "42", "0o10"}
	equal := slices.EqualFunc(numbers, strings, func(n int, s string) bool {
		sn, err := strconv.ParseInt(s, 0, 64)
		if err != nil {
			return false
		}
		return n == int(sn)
	})
	fmt.Println(equal)
	// Output:
	// true
}

func ExampleIndex() {
	numbers := []int{0, 42, 8}
	fmt.Println(slices.Index(numbers, 8))
	fmt.Println(slices.Index(numbers, 7))
	// Output:
	// 2
	// -1
}

func ExampleIndexFunc() {
	numbers := []int{0, 42, -10, 8}
	i := slices.IndexFunc(numbers, func(n int) bool {
		return n < 0
	})
	fmt.Println("First negative at index", i)
	// Output:
	// First negative at index 2
}

func ExampleInsert() {
	names := []string{"Alice", "Bob", "Vera"}
	names = slices.Insert(names, 1, "Bill", "Billie")
	names = slices.Insert(names, len(names), "Zac")
	fmt.Println(names)
	// Output:
	// [Alice Bill Billie Bob Vera Zac]
}

func ExampleIsSorted() {
	fmt.Println(slices.IsSorted([]string{"Alice", "Bob", "Vera"}))
	fmt.Println(slices.IsSorted([]int{0, 2, 1}))
	// Output:
	// true
	// false
}

func ExampleIsSortedFunc() {
	names := []string{"alice", "Bob", "VERA"}
	isSortedInsensitive := slices.IsSortedFunc(names, func(a, b string) int {
		return strings.Compare(strings.ToLower(a), strings.ToLower(b))
	})
	fmt.Println(isSortedInsensitive)
	fmt.Println(slices.IsSorted(names))
	// Output:
	// true
	// false
}

func ExampleMax() {
	numbers := []int{0, 42, -10, 8}
	fmt.Println(slices.Max(numbers))
	// Output:
	// 42
}

func ExampleMaxFunc() {
	type Person struct {
		Name string
		Age  int
	}
	people := []Person{
		{"Gopher", 13},
		{"Alice", 55},
		{"Vera", 24},
		{"Bob", 55},
	}
	firstOldest := slices.MaxFunc(people, func(a, b Person) int {
		return cmp.Compare(a.Age, b.Age)
	})
	fmt.Println(firstOldest.Name)
	// Output:
	// Alice
}

func ExampleMin() {
	numbers := []int{0, 42, -10, 8}
	fmt.Println(slices.Min(numbers))
	// Output:
	// -10
}

func ExampleMinFunc() {
	type Person struct {
		Name string
		Age  int
	}
	people := []Person{
		{"Gopher", 13},
		{"Bob", 5},
		{"Vera", 24},
		{"Bill", 5},
	}
	firstYoungest := slices.MinFunc(people, func(a, b Person) int {
		return cmp.Compare(a.Age, b.Age)
	})
	fmt.Println(firstYoungest.Name)
	// Output:
	// Bob
}

func ExampleReplace() {
	names := []string{"Alice", "Bob", "Vera", "Zac"}
	names = slices.Replace(names, 1, 3, "Bill", "Billie", "Cat")
	fmt.Println(names)
	// Output:
	// [Alice Bill Billie Cat Zac]
}

func ExampleReverse() {
	names := []string{"alice", "Bob", "VERA"}
	slices.Reverse(names)
	fmt.Println(names)
	// Output:
	// [VERA Bob alice]
}

func ExampleSort() {
	smallInts := []int8{0, 42, -10, 8}
	slices.Sort(smallInts)
	fmt.Println(smallInts)
	// Output:
	// [-10 0 8 42]
}

func ExampleSortFunc_caseInsensitive() {
	names := []string{"Bob", "alice", "VERA"}
	slices.SortFunc(names, func(a, b string) int {
		return strings.Compare(strings.ToLower(a), strings.ToLower(b))
	})
	fmt.Println(names)
	// Output:
	// [alice Bob VERA]
}

func ExampleSortFunc_multiField() {
	type Person struct {
		Name string
		Age  int
	}
	people := []Person{
		{"Gopher", 13},
		{"Alice", 55},
		{"Bob", 24},
		{"Alice", 20},
	}
	slices.SortFunc(people, func(a, b Person) int {
		if n := strings.Compare(a.Name, b.Name); n != 0 {
			return n
		}
		// If names are equal, order by age
		return cmp.Compare(a.Age, b.Age)
	})
	fmt.Println(people)
	// Output:
	// [{Alice 20} {Alice 55} {Bob 24} {Gopher 13}]
}

func ExampleSortStableFunc() {
	type Person struct {
		Name string
		Age  int
	}
	people := []Person{
		{"Gopher", 13},
		{"Alice", 20},
		{"Bob", 24},
		{"Alice", 55},
	}
	// Stable sort by name, keeping age ordering of Alices intact
	slices.SortStableFunc(people, func(a, b Person) int {
		return strings.Compare(a.Name, b.Name)
	})
	fmt.Println(people)
	// Output:
	// [{Alice 20} {Alice 55} {Bob 24} {Gopher 13}]
}

func ExampleClone() {
	numbers := []int{0, 42, -10, 8}
	clone := slices.Clone(numbers)
	fmt.Println(clone)
	clone[2] = 10
	fmt.Println(numbers)
	// Output:
	// [0 42 -10 8]
	// [0 42 -10 8]
}

func ExampleGrow() {
	numbers := []int{0, 42, -10, 8}
	grow := slices.Grow(numbers, 2)
	fmt.Println(cap(numbers))
	fmt.Println(grow)
	fmt.Println(len(grow))
	fmt.Println(cap(grow))
	// Output:
	// 4
	// [0 42 -10 8]
	// 4
	// 8
}

func ExampleClip() {
	a := [...]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	s := a[:4:10]
	clip := slices.Clip(s)
	fmt.Println(cap(s))
	fmt.Println(clip)
	fmt.Println(len(clip))
	fmt.Println(cap(clip))
	// Output:
	// 10
	// [0 1 2 3]
	// 4
	// 4
}

func ExampleConcat() {
	s1 := []int{0, 1, 2, 3}
	s2 := []int{4, 5, 6}
	concat := slices.Concat(s1, s2)
	fmt.Println(concat)
	// Output:
	// [0 1 2 3 4 5 6]
}

func ExampleContains() {
	numbers := []int{0, 1, 2, 3}
	fmt.Println(slices.Contains(numbers, 2))
	fmt.Println(slices.Contains(numbers, 4))
	// Output:
	// true
	// false
}

func ExampleRepeat() {
	numbers := []int{0, 1, 2, 3}
	repeat := slices.Repeat(numbers, 2)
	fmt.Println(repeat)
	// Output:
	// [0 1 2 3 0 1 2 3]
}

func ExampleAll() {
	names := []string{"Alice", "Bob", "Vera"}
	for i, v := range slices.All(names) {
		fmt.Println(i, ":", v)
	}
	// Output:
	// 0 : Alice
	// 1 : Bob
	// 2 : Vera
}

func ExampleBackward() {
	names := []string{"Alice", "Bob", "Vera"}
	for i, v := range slices.Backward(names) {
		fmt.Println(i, ":", v)
	}
	// Output:
	// 2 : Vera
	// 1 : Bob
	// 0 : Alice
}

func ExampleValues() {
	names := []string{"Alice", "Bob", "Vera"}
	for v := range slices.Values(names) {
		fmt.Println(v)
	}
	// Output:
	// Alice
	// Bob
	// Vera
}

func ExampleAppendSeq() {
	seq := func(yield func(int) bool) {
		for i := 0; i < 10; i += 2 {
			if !yield(i) {
				return
			}
		}
	}

	s := slices.AppendSeq([]int{1, 2}, seq)
	fmt.Println(s)
	// Output:
	// [1 2 0 2 4 6 8]
}

func ExampleCollect() {
	seq := func(yield func(int) bool) {
		for i := 0; i < 10; i += 2 {
			if !yield(i) {
				return
			}
		}
	}

	s := slices.Collect(seq)
	fmt.Println(s)
	// Output:
	// [0 2 4 6 8]
}

func ExampleSorted() {
	seq := func(yield func(int) bool) {
		flag := -1
		for i := 0; i < 10; i += 2 {
			flag = -flag
			if !yield(i * flag) {
				return
			}
		}
	}

	s := slices.Sorted(seq)
	fmt.Println(s)
	fmt.Println(slices.IsSorted(s))
	// Output:
	// [-6 -2 0 4 8]
	// true
}

func ExampleSortedFunc() {
	seq := func(yield func(int) bool) {
		flag := -1
		for i := 0; i < 10; i += 2 {
			flag = -flag
			if !yield(i * flag) {
				return
			}
		}
	}

	sortFunc := func(a, b int) int {
		return cmp.Compare(b, a) // the comparison is being done in reverse
	}

	s := slices.SortedFunc(seq, sortFunc)
	fmt.Println(s)
	// Output:
	// [8 4 0 -2 -6]
}

func ExampleSortedStableFunc() {
	type Person struct {
		Name string
		Age  int
	}

	people := []Person{
		{"Gopher", 13},
		{"Alice", 20},
		{"Bob", 5},
		{"Vera", 24},
		{"Zac", 20},
	}

	sortFunc := func(x, y Person) int {
		return cmp.Compare(x.Age, y.Age)
	}

	s := slices.SortedStableFunc(slices.Values(people), sortFunc)
	fmt.Println(s)
	// Output:
	// [{Bob 5} {Gopher 13} {Alice 20} {Zac 20} {Vera 24}]
}

func ExampleChunk() {
	type Person struct {
		Name string
		Age  int
	}

	type People []Person

	people := People{
		{"Gopher", 13},
		{"Alice", 20},
		{"Bob", 5},
		{"Vera", 24},
		{"Zac", 15},
	}

	// Chunk people into []Person 2 elements at a time.
	for c := range slices.Chunk(people, 2) {
		fmt.Println(c)
	}

	// Output:
	// [{Gopher 13} {Alice 20}]
	// [{Bob 5} {Vera 24}]
	// [{Zac 15}]
}
