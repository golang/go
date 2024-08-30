// -lang=go1.19

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20

package p

type I[P any] interface {
	~string | ~int
	Error() P
}

func _[P I[string]]() {
	var x P
	var _ error = x
}
