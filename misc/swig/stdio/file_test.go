// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package file

import "testing"

// Open this file itself and verify that the first few characters are
// as expected.
func TestRead(t *testing.T) {
	f := Fopen("file_test.go", "r")
	if f == nil {
		t.Fatal("fopen failed")
	}
	if Fgetc(f) != '/' || Fgetc(f) != '/' || Fgetc(f) != ' ' || Fgetc(f) != 'C' {
		t.Error("read unexpected characters")
	}
	if Fclose(f) != 0 {
		t.Error("fclose failed")
	}
}
