// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to crash compiler in interface type equality check.

package p

type i1 interface {
      F() interface{i1}
}

type i2 interface {
      F() interface{i2}
}       

var v1 i1
var v2 i2

func f() bool {
       return v1 == v2
}

// TODO(gri) Change test to use exported interfaces.
// See issue #15596 for details.