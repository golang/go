// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Collector[T any] struct {
}

func (c *Collector[T]) Collect() {
}

func TestInOrderIntTree() {
	collector := Collector[int]{}
	_ = collector.Collect
}

func main() {
	TestInOrderIntTree()
}
