// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package enumlib

type Result enum {
	Ok { Value int }
	Err
}

func (r Result) Or(zero int) int {
	switch r {
	case Ok:
		return r.Value
	case Err, nil:
		return zero
	}
	return zero
}

func NewResult() Result { return Ok{Value: 7} }

type Option[T any] enum {
	Some { Value T }
	None
}

func (o Option[T]) Or(zero T) T {
	switch o {
	case Some:
		return o.Value
	case None, nil:
		return zero
	}
	return zero
}

func NewOption() Option[int] { return Some{Value: 9} }
