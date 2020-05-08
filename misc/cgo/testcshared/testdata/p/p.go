// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "C"

//export FromPkg
func FromPkg() int32 { return 1024 }

//export Divu
func Divu(a, b uint32) uint32 { return a / b }
