// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

const _ = uint64(unsafe.Offsetof(T{}.F)) // ERROR "undefined"
