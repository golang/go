package main

// Test generic sort function with two different pointer types in different packages,
// make sure only one instantiation is created.

import (
	"fmt"

	"cmd/compile/internal/test/testdata/mysort"
)

type MyString struct {
	string
}

func (a *MyString) Less(b *MyString) bool {
	return a.string < b.string
}

func main() {
	mysort.F()

	sl1 := []*mysort.MyInt{{7}, {1}, {4}, {6}}
	mysort.Sort(sl1)
	fmt.Printf("%v %v %v %v\n", sl1[0], sl1[1], sl1[2], sl1[3])

	sl2 := []*MyString{{"when"}, {"in"}, {"the"}, {"course"}, {"of"}}
	mysort.Sort(sl2)
	fmt.Printf("%v %v %v %v %v\n", sl2[0], sl2[1], sl2[2], sl2[3], sl2[4])
}
