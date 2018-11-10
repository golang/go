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

func ExamplePutUvarint() {
	buf := make([]byte, binary.MaxVarintLen64)

	for _, x := range []uint64{1, 2, 127, 128, 255, 256} {
		n := binary.PutUvarint(buf, x)
		fmt.Printf("%x\n", buf[:n])
	}
	// Output:
	// 01
	// 02
	// 7f
	// 8001
	// ff01
	// 8002
}

func ExamplePutVarint() {
	buf := make([]byte, binary.MaxVarintLen64)

	for _, x := range []int64{-65, -64, -2, -1, 0, 1, 2, 63, 64} {
		n := binary.PutVarint(buf, x)
		fmt.Printf("%x\n", buf[:n])
	}
	// Output:
	// 8101
	// 7f
	// 03
	// 01
	// 00
	// 02
	// 04
	// 7e
	// 8001
}

func ExampleUvarint() {
	inputs := [][]byte{
		[]byte{0x01},
		[]byte{0x02},
		[]byte{0x7f},
		[]byte{0x80, 0x01},
		[]byte{0xff, 0x01},
		[]byte{0x80, 0x02},
	}
	for _, b := range inputs {
		x, n := binary.Uvarint(b)
		if n != len(b) {
			fmt.Println("Uvarint did not consume all of in")
		}
		fmt.Println(x)
	}
	// Output:
	// 1
	// 2
	// 127
	// 128
	// 255
	// 256
}

func ExampleVarint() {
	inputs := [][]byte{
		[]byte{0x81, 0x01},
		[]byte{0x7f},
		[]byte{0x03},
		[]byte{0x01},
		[]byte{0x00},
		[]byte{0x02},
		[]byte{0x04},
		[]byte{0x7e},
		[]byte{0x80, 0x01},
	}
	for _, b := range inputs {
		x, n := binary.Varint(b)
		if n != len(b) {
			fmt.Println("Varint did not consume all of in")
		}
		fmt.Println(x)
	}
	// Output:
	// -65
	// -64
	// -2
	// -1
	// 0
	// 1
	// 2
	// 63
	// 64
}
