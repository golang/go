// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import "io"

type SourceReader[Source any] interface {
	Read(p Source) (n int, err error)
}

func GenericInterfaceAssertionTest[T io.Reader]() {
	var (
		a SourceReader[[]byte]
		b SourceReader[[]int]
		r io.Reader
	)
	_ = a.(io.Reader)
	_ = b.(io.Reader) // want `^impossible type assertion: no type can implement both typeparams.SourceReader\[\[\]int\] and io.Reader \(conflicting types for Read method\)$`

	_ = r.(SourceReader[[]byte])
	_ = r.(SourceReader[[]int]) // want `^impossible type assertion: no type can implement both io.Reader and typeparams.SourceReader\[\[\]int\] \(conflicting types for Read method\)$`
	_ = r.(T)                   // not actually an iface assertion, so checked by the type checker.

	switch a.(type) {
	case io.Reader:
	default:
	}

	switch b.(type) {
	case io.Reader: // want `^impossible type assertion: no type can implement both typeparams.SourceReader\[\[\]int\] and io.Reader \(conflicting types for Read method\)$`

	default:
	}
}
