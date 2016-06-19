// errorcheckdir

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 1284

package bug313

/*
6g bug313.dir/[ab].go

Before:
bug313.dir/b.go:7: internal compiler error: fault

Now:
bug313.dir/a.go:10: undefined: fmt.DoesNotExist
*/
