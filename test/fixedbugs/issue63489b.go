// errorcheck -lang=go1.21

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file has been changed from its original version as
// //go:build file versions below 1.21 set the language version to 1.21.
// The original tested a -lang version of 1.4 with a file version of
// go1.4 while this new version tests a -lang version of go1.1
// with a file version of go1.21.

//go:build go1.21

package p

func f() {
	for _ = range 10 { // ERROR "file declares //go:build go1.21"
	}
}
