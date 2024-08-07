// errorcheck -lang=go1.22

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file has been changed from its original version as
// //go:build language downgrades below go1.21 are no longer
// supported. The original tested a downgrade from go1.21 to
// go1.4 while this new version tests a downgrade from go1.22
// to go1.21

//go:build go1.21

package p

func f() {
	for _ = range 10 { // ERROR "file declares //go:build go1.21"
	}
}
