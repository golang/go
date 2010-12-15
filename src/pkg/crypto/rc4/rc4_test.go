// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rc4

import (
	"testing"
)

type rc4Test struct {
	key, keystream []byte
}

var golden = []rc4Test{
	// Test vectors from the original cypherpunk posting of ARC4:
	//   http://groups.google.com/group/sci.crypt/msg/10a300c9d21afca0?pli=1
	{
		[]byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef},
		[]byte{0x74, 0x94, 0xc2, 0xe7, 0x10, 0x4b, 0x08, 0x79},
	},
	{
		[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		[]byte{0xde, 0x18, 0x89, 0x41, 0xa3, 0x37, 0x5d, 0x3a},
	},
	{
		[]byte{0xef, 0x01, 0x23, 0x45},
		[]byte{0xd6, 0xa1, 0x41, 0xa7, 0xec, 0x3c, 0x38, 0xdf, 0xbd, 0x61},
	},

	// Test vectors from the Wikipedia page: http://en.wikipedia.org/wiki/RC4
	{
		[]byte{0x4b, 0x65, 0x79},
		[]byte{0xeb, 0x9f, 0x77, 0x81, 0xb7, 0x34, 0xca, 0x72, 0xa7, 0x19},
	},
	{
		[]byte{0x57, 0x69, 0x6b, 0x69},
		[]byte{0x60, 0x44, 0xdb, 0x6d, 0x41, 0xb7},
	},
}

func TestGolden(t *testing.T) {
	for i := 0; i < len(golden); i++ {
		g := golden[i]
		c, err := NewCipher(g.key)
		if err != nil {
			t.Errorf("Failed to create cipher at golden index %d", i)
			return
		}
		keystream := make([]byte, len(g.keystream))
		c.XORKeyStream(keystream, keystream)
		for j, v := range keystream {
			if g.keystream[j] != v {
				t.Errorf("Failed at golden index %d", i)
				break
			}
		}
	}
}
