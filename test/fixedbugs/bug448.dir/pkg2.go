// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3843: inlining bug due to wrong receive operator precedence.

package pkg2

import "./pkg1"

func F() {
	pkg1.Do()
}

