// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

var _ = string([]byte(nil))[0]
var _ = uintptr(unsafe.Pointer(uintptr(1))) << 100
