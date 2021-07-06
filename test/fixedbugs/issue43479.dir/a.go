// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Here struct{ stuff int }
type Info struct{ Dir string }

func New() Here { return Here{} }
func (h Here) Dir(p string) (Info, error)

type I interface{ M(x string) }

type T = struct {
	Here
	I
}

var X T

var A = (*T).Dir
var B = T.Dir
var C = X.Dir
var D = (*T).M
var E = T.M
var F = X.M
