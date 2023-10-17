// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2451, 2452 
package foo

func f() error { return 0 } // ERROR "cannot use 0 (.type int.)?|has no methods"

func g() error { return -1 }  // ERROR "cannot use -1 (.type int.)?|has no methods"
