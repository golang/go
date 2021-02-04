// errorcheck

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 14729: structs cannot embed unsafe.Pointer per the spec.

package main

import "unsafe"

type s struct { unsafe.Pointer } // ERROR "embedded type cannot be a pointer|embedded type may not be a pointer||embedded field type cannot be unsafe.Pointer"
type s1 struct { p unsafe.Pointer }
