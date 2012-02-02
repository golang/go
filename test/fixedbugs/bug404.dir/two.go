// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The gccgo compiler would fail on the import statement.
// two.go:10:13: error: use of undefined type ‘one.T2’

package two

import "./one"

var V one.T3
