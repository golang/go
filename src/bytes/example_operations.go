// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

func ExampleShiftLeft() {
	// Decode << to a byte array
	data := []byte("PS")
	fmt.Printf("%08b\n", data) // [00111100 00111100]

	// ShiftLeft 3 bits to the left (use -3 for right)
	bytes.ShiftLeft(data, 3)

	// The output is now shifted positive 3 bits
	fmt.Printf("%08b\n", data) // [11100001 11100000]
}
