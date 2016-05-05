// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
extern int issue8811Initialized;
extern void issue8811Init();

void issue8811Execute() {
	if(!issue8811Initialized)
		issue8811Init();
}
*/
import "C"

import "testing"

func test8811(t *testing.T) {
	C.issue8811Execute()
}
