// errorcheck -lang=go1.13

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type I interface { M() }

type _ interface {
	I
	I // ERROR "duplicate method M"
}
