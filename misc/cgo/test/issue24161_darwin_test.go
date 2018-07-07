// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import (
	"testing"

	"./issue24161arg"
	"./issue24161res"
)

func Test24161Arg(t *testing.T) {
	issue24161arg.Test(t)
}
func Test24161Res(t *testing.T) {
	issue24161res.Test(t)
}
