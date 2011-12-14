// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2500

package foo

// Check that we only get root cause message, no further complaints about r undefined
func (r *indexWriter) foo() {}  // ERROR "undefined.*indexWriter"
