// compile

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash compiler in interface type equality check.

package p

type I1 interface {
      F() interface{I1}
}

type I2 interface {
      F() interface{I2}
}       

var v1 I1
var v2 I2

func f() bool {
       return v1 == v2
}
