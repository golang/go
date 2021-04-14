// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package suffixarray_test

import (
	"fmt"
	"index/suffixarray"
)

func ExampleIndex_Lookup() {
	index := suffixarray.New([]byte("banana"))
	offsets := index.Lookup([]byte("ana"), -1)
	for _, off := range offsets {
		fmt.Println(off)
	}

	// Unordered output:
	// 1
	// 3
}
