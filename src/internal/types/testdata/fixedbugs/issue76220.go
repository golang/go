// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	append(nil /* ERROR "argument must be a slice; have untyped nil" */, ""...)
}

// test case from issue

func main() {
	s := "hello"
	msg := append(nil /* ERROR "argument must be a slice; have untyped nil" */, s...)
	print(msg)
}
