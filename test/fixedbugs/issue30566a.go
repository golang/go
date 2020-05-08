// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

//go:noinline
func ident(s string) string { return s }

func returnSecond(x bool, s string) string { return s }

func identWrapper(s string) string { return ident(s) }

func main() {
	got := returnSecond((false || identWrapper("bad") != ""), ident("good"))
	if got != "good" {
		panic(fmt.Sprintf("wanted \"good\", got \"%s\"", got))
	}
}
