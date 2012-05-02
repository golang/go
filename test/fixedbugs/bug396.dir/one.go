// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package one

// Issue 2687
type T struct { int }

func New(i int) T { return T{i} }
