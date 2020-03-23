// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Keep in sync with ../base32/example_test.go.

package base64_test

import (
	"encoding/base64"
	"fmt"
	"os"
)

func Example() {
	msg := "Hello, 世界"
	encoded := base64.StdEncoding.EncodeToString([]byte(msg))
	fmt.Println(encoded)
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		fmt.Println("decode error:", err)
		return
	}
	fmt.Println(string(decoded))
	// Output:
	// SGVsbG8sIOS4lueVjA==
	// Hello, 世界
}

func ExampleEncoding_Encode() {
	data := []byte("hello gophers!")
	dst := make([]byte, base64.StdEncoding.EncodedLen(len(data)))
	base64.StdEncoding.Encode(dst, data)
	fmt.Printf("%q\n", dst)
	// Output:
	// "aGVsbG8gZ29waGVycyE="
}

func ExampleEncoding_EncodeToString() {
	data := []byte("any + old & data")
	str := base64.StdEncoding.EncodeToString(data)
	fmt.Println(str)
	// Output:
	// YW55ICsgb2xkICYgZGF0YQ==
}

func ExampleEncoding_Decode() {
	data := []byte("aGVsbG8gZ29waGVycywgeW91IGFsbCByaWdodD8K")
	dst := make([]byte, base64.StdEncoding.DecodedLen(len(data)))
	n, err := base64.StdEncoding.Decode(dst, data)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("%q: %d\n", dst, n)
	// Output:
	// "hello gophers, you all right?\n": 30
}

func ExampleEncoding_DecodeString() {
	str := "c29tZSBkYXRhIHdpdGggACBhbmQg77u/"
	data, err := base64.StdEncoding.DecodeString(str)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Printf("%q\n", data)
	// Output:
	// "some data with \x00 and \ufeff"
}

func ExampleNewEncoder() {
	input := []byte("foo\x00bar")
	encoder := base64.NewEncoder(base64.StdEncoding, os.Stdout)
	encoder.Write(input)
	// Must close the encoder when finished to flush any partial blocks.
	// If you comment out the following line, the last partial block "r"
	// won't be encoded.
	encoder.Close()
	// Output:
	// Zm9vAGJhcg==
}
