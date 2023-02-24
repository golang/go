// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var x Value
	NewScanner().Scan(x)
}

type Value any

type Scanner interface{ Scan(any) error }

func NewScanner() Scanner {
	return &t{}
}

type t struct{}

func (*t) Scan(interface{}) error { return nil }
