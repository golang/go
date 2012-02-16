// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort_test

import (
	"fmt"
	"sort"
)

// Reverse embeds a sort.Interface value and implements a reverse sort over
// that value.
type Reverse struct {
	// This embedded Interface permits Reverse to use the methods of
	// another Interface implementation.
	sort.Interface
}

// Less returns the opposite of the embedded implementation's Less method.
func (r Reverse) Less(i, j int) bool {
	return r.Interface.Less(j, i)
}

func ExampleInterface_reverse() {
	s := []int{5, 2, 6, 3, 1, 4} // unsorted
	sort.Sort(Reverse{sort.IntSlice(s)})
	fmt.Println(s)
	// Output: [6 5 4 3 2 1]
}
