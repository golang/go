// errorcheckandrundir -0 -m -l=4 -newescape=false

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that linknames are included in export data (issue 18167).
package ignored

/*
Without CL 33911, this test would fail with the following error:

main.main: relocation target linkname2.byteIndex not defined
main.main: undefined: "linkname2.byteIndex"
*/
