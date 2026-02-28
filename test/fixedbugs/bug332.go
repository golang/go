// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// type T int

func main() {}

// issue 1474

// important: no newline on end of next line.
// 6g used to print <epoch> instead of bug332.go:111
func (t *T) F() {} // ERROR "undefined.*T"