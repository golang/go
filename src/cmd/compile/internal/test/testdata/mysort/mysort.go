// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generic sort function, tested with two different pointer types.

package mysort

import (
	"fmt"
)

type LessConstraint[T any] interface {
	Less(T) bool
}

//go:noinline
func Sort[T LessConstraint[T]](x []T) {
	n := len(x)
	for i := 1; i < n; i++ {
		for j := i; j > 0 && x[j].Less(x[j-1]); j-- {
			x[j], x[j-1] = x[j-1], x[j]
		}
	}
}

type MyInt struct {
	Value int
}

func (a *MyInt) Less(b *MyInt) bool {
	return a.Value < b.Value
}

//go:noinline
func F() {
	sl1 := []*MyInt{&MyInt{4}, &MyInt{3}, &MyInt{8}, &MyInt{7}}
	Sort(sl1)
	fmt.Printf("%v %v %v %v\n", sl1[0], sl1[1], sl1[2], sl1[3])
}
