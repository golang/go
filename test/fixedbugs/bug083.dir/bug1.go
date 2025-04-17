// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug1

import "./bug0"

// This is expected to fail--t0 is in package bug0 and should not be
// visible here in package bug1.  The test for failure is in
// ../bug083.go.

var v1 bug0.t0;	// ERROR "bug0"

