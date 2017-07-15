// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package binary_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

func ExampleWrite() {
	buf := new(bytes.Buffer)
	var pi float64 = math.Pi
	err := binary.Write(buf, binary.LittleEndian, pi)
	if err != nil {
		fmt.Println("binary.Write failed:", err)
	}
	fmt.Printf("% x", buf.Bytes())
	// Output: 18 2d 44 54 fb 21 09 40
}

func ExampleWrite_multi() {
	buf := new(bytes.Buffer)
	var data = []interface{}{
		uint16(61374),
		int8(-54),
		uint8(254),
	}
	for _, v := range data {
		err := binary.Write(buf, binary.LittleEndian, v)
		if err != nil {
			fmt.Println("binary.Write failed:", err)
		}
	}
	fmt.Printf("%x", buf.Bytes())
	// Output: beefcafe
}

func ExampleRead() {
	var pi float64
	b := []byte{0x18, 0x2d, 0x44, 0x54, 0xfb, 0x21, 0x09, 0x40}
	buf := bytes.NewReader(b)
	err := binary.Read(buf, binary.LittleEndian, &pi)
	if err != nil {
		fmt.Println("binary.Read failed:", err)
	}
	fmt.Print(pi)
	// Output: 3.141592653589793
}

func ExampleByteOrder_put() {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint16(b[0:], 0x03e8)
	binary.LittleEndian.PutUint16(b[2:], 0x07d0)
	fmt.Printf("% x\n", b)
	// Output:
	// e8 03 d0 07
}

func ExampleByteOrder_get() {
	b := []byte{0xe8, 0x03, 0xd0, 0x07}
	x1 := binary.LittleEndian.Uint16(b[0:])
	x2 := binary.LittleEndian.Uint16(b[2:])
	fmt.Printf("%#04x %#04x\n", x1, x2)
	// Output:
	// 0x03e8 0x07d0
}
