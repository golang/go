// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure VARDEF can be a top-level statement.

package p

func f() {
	var s string
	var as []string
	switch false && (s+"a"+as[0]+s+as[0]+s == "") {
	}
}
