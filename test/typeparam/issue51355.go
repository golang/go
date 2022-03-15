// compile -G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Cache[E comparable] struct {
	adder func(...E)
}

func New[E comparable]() *Cache[E] {
	c := &Cache[E]{}

	c.adder = func(elements ...E) {
		for _, value := range elements {
			value := value
			go func() {
				println(value)
			}()
		}
	}

	return c
}

func main() {
	c := New[string]()
	c.adder("test")
}
