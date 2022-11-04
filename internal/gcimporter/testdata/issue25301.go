// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue25301

type (
	A = interface {
		M()
	}
	T interface {
		A
	}
	S struct{}
)

func (S) M() { println("m") }
