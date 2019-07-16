// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort_test

import (
	"fmt"
	"sort"
)

type Person struct {
	Name string
	Age  int
}

func (p Person) String() string {
	return fmt.Sprintf("%s: %d", p.Name, p.Age)
}

// ByAge implements sort.Interface for []Person based on
// the Age field.
type ByAge []Person

func (a ByAge) Len() int           { return len(a) }
func (a ByAge) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByAge) Less(i, j int) bool { return a[i].Age < a[j].Age }

func Example() {
	people := []Person{
		{"Bob", 31},
		{"John", 42},
		{"Michael", 17},
		{"Jenny", 26},
	}

	fmt.Println(people)
	// There are two ways to sort a slice. First, one can define
	// a set of methods for the slice type, as with ByAge, and
	// call sort.Sort. In this first example we use that technique.
	sort.Sort(ByAge(people))
	fmt.Println(people)

	// The other way is to use sort.Slice with a custom Less
	// function, which can be provided as a closure. In this
	// case no methods are needed. (And if they exist, they
	// are ignored.) Here we re-sort in reverse order: compare
	// the closure with ByAge.Less.
	sort.Slice(people, func(i, j int) bool {
		return people[i].Age > people[j].Age
	})
	fmt.Println(people)

	// Output:
	// [Bob: 31 John: 42 Michael: 17 Jenny: 26]
	// [Michael: 17 Jenny: 26 Bob: 31 John: 42]
	// [John: 42 Bob: 31 Jenny: 26 Michael: 17]
}
