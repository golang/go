// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug2

import _ "./bug1"
import "./bug0"

type T2 struct { t bug0.T }

func fn(p *T2) int {
	// This reference should be invalid, because bug0.T.i is local
	// to package bug0 and should not be visible in package bug1.
	return p.t.i;	// ERROR "field|undef"
}
