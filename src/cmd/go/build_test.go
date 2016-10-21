// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"testing"
)

func TestRemoveDevNull(t *testing.T) {
	fi, err := os.Lstat(os.DevNull)
	if err != nil {
		t.Skip(err)
	}
	if fi.Mode().IsRegular() {
		t.Errorf("Lstat(%s).Mode().IsRegular() = true; expected false", os.DevNull)
	}
	mayberemovefile(os.DevNull)
	_, err = os.Lstat(os.DevNull)
	if err != nil {
		t.Errorf("mayberemovefile(%s) did remove it; oops", os.DevNull)
	}
}
