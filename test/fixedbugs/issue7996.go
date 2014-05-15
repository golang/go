// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// /tmp/x.go:5: illegal constant expression: bool == interface {}

package p

var m = map[interface{}]struct{}{
	nil:  {},
	true: {},
}
