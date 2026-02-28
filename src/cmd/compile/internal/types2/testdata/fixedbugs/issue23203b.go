// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

type T struct{}

func (T) m2([unsafe.Sizeof(T.m1)]int) {}
func (T) m1()                         {}

func main() {}
