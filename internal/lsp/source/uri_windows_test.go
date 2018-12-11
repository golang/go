// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package source

import (
	"testing"
)

func TestURIWindows(t *testing.T) {
	s := `C:\Windows\System32`
	uri := ToURI(s)
	if uri != `file:///C:/Windows/System32` {
		t.Fatalf("ToURI: got %v want %v", uri, s)
	}
	f, err := URI(uri).Filename()
	if err != nil {
		t.Fatal(err)
	}
	if f != s {
		t.Fatalf("Filename: got %v want %v", f, s)
	}
}
