// +build amd64
// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ = []int{1 << 31: 1} // ok on machines with 64bit int
