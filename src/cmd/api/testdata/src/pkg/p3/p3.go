// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p3

type ThirdBase struct{}

func (tb *ThirdBase) GoodPlayer() (i, j, k int)
func BadHop(i, j, k int) (l, m bool, n, o *ThirdBase, err error)
