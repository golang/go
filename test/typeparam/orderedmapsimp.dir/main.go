// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"bytes"
	"fmt"
)

func TestMap() {
	m := a.New[[]byte, int](bytes.Compare)

	if _, found := m.Find([]byte("a")); found {
		panic(fmt.Sprintf("unexpectedly found %q in empty map", []byte("a")))
	}

	for _, c := range []int{'a', 'c', 'b'} {
		if !m.Insert([]byte(string(c)), c) {
			panic(fmt.Sprintf("key %q unexpectedly already present", []byte(string(c))))
		}
	}
	if m.Insert([]byte("c"), 'x') {
		panic(fmt.Sprintf("key %q unexpectedly not present", []byte("c")))
	}

	if v, found := m.Find([]byte("a")); !found {
		panic(fmt.Sprintf("did not find %q", []byte("a")))
	} else if v != 'a' {
		panic(fmt.Sprintf("key %q returned wrong value %c, expected %c", []byte("a"), v, 'a'))
	}
	if v, found := m.Find([]byte("c")); !found {
		panic(fmt.Sprintf("did not find %q", []byte("c")))
	} else if v != 'x' {
		panic(fmt.Sprintf("key %q returned wrong value %c, expected %c", []byte("c"), v, 'x'))
	}

	if _, found := m.Find([]byte("d")); found {
		panic(fmt.Sprintf("unexpectedly found %q", []byte("d")))
	}

	gather := func(it *a.Iterator[[]byte, int]) []int {
		var r []int
		for {
			_, v, ok := it.Next()
			if !ok {
				return r
			}
			r = append(r, v)
		}
	}
	got := gather(m.Iterate())
	want := []int{'a', 'b', 'x'}
	if !a.SliceEqual(got, want) {
		panic(fmt.Sprintf("Iterate returned %v, want %v", got, want))
	}

}

func main() {
	TestMap()
}
