// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2563
package foo

func fn(a float32) {}

var f func(arg int) = fn // ERROR "cannot use fn .type func.float32.. as type func.int. in assignment|different parameter types|cannot use fn .*type func.*float32.. as type func.*int. in variable declaration"
