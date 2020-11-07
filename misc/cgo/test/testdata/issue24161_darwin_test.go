// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import (
	"testing"

	"cgotest/issue24161arg"
	"cgotest/issue24161e0"
	"cgotest/issue24161e1"
	"cgotest/issue24161e2"
	"cgotest/issue24161res"
)

func Test24161Arg(t *testing.T) {
	issue24161arg.Test(t)
}
func Test24161Res(t *testing.T) {
	issue24161res.Test(t)
}
func Test24161Example0(t *testing.T) {
	issue24161e0.Test(t)
}
func Test24161Example1(t *testing.T) {
	issue24161e1.Test(t)
}
func Test24161Example2(t *testing.T) {
	issue24161e2.Test(t)
}
