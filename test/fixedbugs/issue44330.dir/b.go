// Copyright 2021 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package main

import (
	"./a"
)

type Term struct {
	top *a.Table
}

//go:noinline
func NewFred() *Term {
	table := a.NewTable()
	return &Term{top: table}
}

func main() {
	NewFred()
}
