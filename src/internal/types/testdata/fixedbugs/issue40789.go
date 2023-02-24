// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	m := map[string]int{
		"a": 6,
		"b": 7,
	}
	fmt.Println(copyMap[map[string]int, string, int](m))
}

type Map[K comparable, V any] interface {
	map[K] V
}

func copyMap[M Map[K, V], K comparable, V any](m M) M {
	m1 := make(M)
	for k, v := range m {
		m1[k] = v
	}
	return m1
}

// simpler test case from the same issue

type A[X comparable] interface {
	[]X
}

func f[B A[X], X comparable]() B {
	return nil
}
