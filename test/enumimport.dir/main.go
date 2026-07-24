// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./enumlib"

func main() {
	var result enumlib.Result = Ok{Value: 3}
	if result.Or(0) != 3 {
		panic("qualified imported result variant")
	}
	if result.Variant() != "Ok" {
		panic("imported result variant")
	}
	var option enumlib.Option[int] = Some{Value: 4}
	if option.Or(0) != 4 {
		panic("qualified imported option variant")
	}
	if option.Variant() != "Some" {
		panic("imported generic variant")
	}
	if enumlib.NewResult().Or(0) != 7 {
		panic("non-generic imported enum method")
	}
	if enumlib.NewOption().Or(0) != 9 {
		panic("generic imported enum method")
	}
}
