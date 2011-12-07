// $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to cause a typechecking loop error.

package pkg
type T map[int]string
var q = &T{}
