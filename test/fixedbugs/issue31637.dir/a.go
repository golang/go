// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type dO struct {
	x int
}

type EDO struct{}

func (EDO) Apply(*dO) {}

var X EDO
