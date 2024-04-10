// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package versions

// toolchain is maximum version (<1.22) that the go toolchain used
// to build the current tool is known to support.
//
// When a tool is built with >=1.22, the value of toolchain is unused.
//
// x/tools does not support building with go <1.18. So we take this
// as the minimum possible maximum.
var toolchain string = Go1_18
