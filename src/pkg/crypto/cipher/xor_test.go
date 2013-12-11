// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher

import (
	"bytes"
	"testing"
)

func TestXOR(t *testing.T) {
	for alignP := 0; alignP < 2; alignP++ {
		for alignQ := 0; alignQ < 2; alignQ++ {
			for alignD := 0; alignD < 2; alignD++ {
				p := make([]byte, 1024)[alignP:]
				q := make([]byte, 1024)[alignQ:]
				d1 := make([]byte, 1024+alignD)[alignD:]
				d2 := make([]byte, 1024+alignD)[alignD:]
				xorBytes(d1, p, q)
				safeXORBytes(d2, p, q)
				if bytes.Compare(d1, d2) != 0 {
					t.Error("not equal")
				}
			}
		}
	}
}
