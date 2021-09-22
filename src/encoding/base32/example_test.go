// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Keep in sync with ../base64/example_test.go.

package base32_test

import (
	"encoding/base32"
	"fmt"
	"os"
)

func ExampleEncoding_EncodeToString() {
	data := []byte("any + old & data")
	str := base32.StdEncoding.EncodeToString(data)
	fmt.Println(str)
	// Output:
	// MFXHSIBLEBXWYZBAEYQGIYLUME======
}

func ExampleEncoding_Encode() {
	data := []byte("Hello, world!")
	dst := make([]byte, base32.StdEncoding.EncodedLen(len(data)))
	base32.StdEncoding.Encode(dst, data)
	fmt.Println(string(dst))
	// Output:
	// JBSWY3DPFQQHO33SNRSCC===
}

func ExampleEncoding_DecodeString() {
	str := "ONXW2ZJAMRQXIYJAO5UXI2BAAAQGC3TEEDX3XPY="
	data, err := base32.StdEncoding.DecodeString(str)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("%q\n", data)
	// Output:
	// "some data with \x00 and \ufeff"
}

func ExampleEncoding_Decode() {
	str := "JBSWY3DPFQQHO33SNRSCC==="
	dst := make([]byte, base32.StdEncoding.DecodedLen(len(str)))
	n, err := base32.StdEncoding.Decode(dst, []byte(str))
	if err != nil {
		fmt.Println("decode error:", err)
		return
	}
	dst = dst[:n]
	fmt.Printf("%q\n", dst)
	// Output:
	// "Hello, world!"
}

func ExampleNewEncoder() {
	input := []byte("foo\x00bar")
	encoder := base32.NewEncoder(base32.StdEncoding, os.Stdout)
	encoder.Write(input)
	// Must close the encoder when finished to flush any partial blocks.
	// If you comment out the following line, the last partial block "r"
	// won't be encoded.
	encoder.Close()
	// Output:
	// MZXW6ADCMFZA====
}
