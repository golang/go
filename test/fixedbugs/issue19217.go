// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

import (
	"encoding/binary"
)

type DbBuilder struct {
	arr []int
}

func (bld *DbBuilder) Finish() error {
	defer bld.Finish()

	var hash []byte
	for _, ixw := range bld.arr {
		for {
			if ixw != 0 {
				panic("ixw != 0")
			}
			ixw--
		insertOne:
			for {
				for i := 0; i < 1; i++ {
					if binary.LittleEndian.Uint16(hash[i:]) == 0 {
						break insertOne
					}
				}
			}
		}
	}

	return nil
}
