// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main
import ip "ip"
func f() (ip int) {
     // In the next line "ip" should refer to the result variable, not
     // to the package.
     v := ip.ParseIP("")	// ERROR "undefined"
     return 0
}
