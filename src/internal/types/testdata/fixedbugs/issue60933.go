// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"io"
	"os"
)

func g[T any](...T) {}

// Interface and non-interface types do not match.
func _() {
	var file *os.File
	g(file, io /* ERROR "type io.Writer of io.Discard does not match inferred type *os.File for T" */ .Discard)
	g(file, os.Stdout)
}

func _() {
	var a *os.File
	var b any
	g(a, a)
	g(a, b /* ERROR "type any of b does not match inferred type *os.File for T" */)
}

var writer interface {
	Write(p []byte) (n int, err error)
}

func _() {
	var file *os.File
	g(file, writer /* ERROR "type interface{Write(p []byte) (n int, err error)} of writer does not match inferred type *os.File for T" */)
	g(writer, file /* ERROR "type *os.File of file does not match inferred type interface{Write(p []byte) (n int, err error)} for T" */)
}

// Different named interface types do not match.
func _() {
	g(io.ReadWriter(nil), io.ReadWriter(nil))
	g(io.ReadWriter(nil), io /* ERROR "does not match" */ .Writer(nil))
	g(io.Writer(nil), io /* ERROR "does not match" */ .ReadWriter(nil))
}

// Named and unnamed interface types match if they have the same methods.
func _() {
	g(io.Writer(nil), writer)
	g(io.ReadWriter(nil), writer /* ERROR "does not match" */ )
}

// There must be no order dependency for named and unnamed interfaces.
func f[T interface{ m(T) }](a, b T) {}

type F interface {
	m(F)
}

func _() {
	var i F
	var j interface {
		m(F)
	}

	// order doesn't matter
	f(i, j)
	f(j, i)
}