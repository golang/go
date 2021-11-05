// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	d := diff([]int{}, func(int) string {
		return "foo"
	})
	d()
}

func diff[T any](previous []T, uniqueKey func(T) string) func() {
	return func() {
		newJSON := map[string]T{}
		for _, prev := range previous {
			delete(newJSON, uniqueKey(prev))
		}
	}
}
