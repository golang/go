// rundir

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test converting a type defined in a different package to an
// interface defined in a third package, where the interface has a
// hidden method.  This used to cause a link error with gccgo.

package ignored
