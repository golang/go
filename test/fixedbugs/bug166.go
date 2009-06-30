// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main
import "os"
func fn() {
	var e os.Error
	if e == nil {		// ERROR "syntax error|expected ';'"
	}
}
