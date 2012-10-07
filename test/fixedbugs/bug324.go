// rundir

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue 1550: a type cannot implement an interface
// from another package with a private method, and type assertions
// should fail.
package ignored
