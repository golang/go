// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "C"

//export issue29878exported
func issue29878exported(arg int8) uint64 {
	return uint64(arg)
}
