// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fstest

import (
	"testing"
)

func TestMapFS(t *testing.T) {
	m := MapFS{
		"hello":             {Data: []byte("hello, world\n")},
		"fortune/k/ken.txt": {Data: []byte("If a program is too slow, it must have a loop.\n")},
	}
	if err := TestFS(m, "hello", "fortune/k/ken.txt"); err != nil {
		t.Fatal(err)
	}
}
