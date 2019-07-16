// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T struct{}

func (T) m() { println("FAIL") }

type I interface{ m() }
