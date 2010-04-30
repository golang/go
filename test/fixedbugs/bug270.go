// $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=746

package main

type I interface { F() }

type T struct{}

func (T) F() {}

func main() {
	switch I(T{}).(type) {
	case interface{}:
	}
}
