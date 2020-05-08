// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strings"
)

func ExampleCopy() {
	r := strings.NewReader("some io.Reader stream to be read\n")

	if _, err := io.Copy(os.Stdout, r); err != nil {
		log.Fatal(err)
	}

	// Output:
	// some io.Reader stream to be read
}

func ExampleCopyBuffer() {
	r1 := strings.NewReader("first reader\n")
	r2 := strings.NewReader("second reader\n")
	buf := make([]byte, 8)

	// buf is used here...
	if _, err := io.CopyBuffer(os.Stdout, r1, buf); err != nil {
		log.Fatal(err)
	}

	// ... reused here also. No need to allocate an extra buffer.
	if _, err := io.CopyBuffer(os.Stdout, r2, buf); err != nil {
		log.Fatal(err)
	}

	// Output:
	// first reader
	// second reader
}

func ExampleCopyN() {
	r := strings.NewReader("some io.Reader stream to be read")

	if _, err := io.CopyN(os.Stdout, r, 4); err != nil {
		log.Fatal(err)
	}

	// Output:
	// some
}

func ExampleReadAtLeast() {
	r := strings.NewReader("some io.Reader stream to be read\n")

	buf := make([]byte, 14)
	if _, err := io.ReadAtLeast(r, buf, 4); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", buf)

	// buffer smaller than minimal read size.
	shortBuf := make([]byte, 3)
	if _, err := io.ReadAtLeast(r, shortBuf, 4); err != nil {
		fmt.Println("error:", err)
	}

	// minimal read size bigger than io.Reader stream
	longBuf := make([]byte, 64)
	if _, err := io.ReadAtLeast(r, longBuf, 64); err != nil {
		fmt.Println("error:", err)
	}

	// Output:
	// some io.Reader
	// error: short buffer
	// error: unexpected EOF
}

func ExampleReadFull() {
	r := strings.NewReader("some io.Reader stream to be read\n")

	buf := make([]byte, 4)
	if _, err := io.ReadFull(r, buf); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", buf)

	// minimal read size bigger than io.Reader stream
	longBuf := make([]byte, 64)
	if _, err := io.ReadFull(r, longBuf); err != nil {
		fmt.Println("error:", err)
	}

	// Output:
	// some
	// error: unexpected EOF
}

func ExampleWriteString() {
	io.WriteString(os.Stdout, "Hello World")

	// Output: Hello World
}

func ExampleLimitReader() {
	r := strings.NewReader("some io.Reader stream to be read\n")
	lr := io.LimitReader(r, 4)

	if _, err := io.Copy(os.Stdout, lr); err != nil {
		log.Fatal(err)
	}

	// Output:
	// some
}

func ExampleMultiReader() {
	r1 := strings.NewReader("first reader ")
	r2 := strings.NewReader("second reader ")
	r3 := strings.NewReader("third reader\n")
	r := io.MultiReader(r1, r2, r3)

	if _, err := io.Copy(os.Stdout, r); err != nil {
		log.Fatal(err)
	}

	// Output:
	// first reader second reader third reader
}

func ExampleTeeReader() {
	var r io.Reader = strings.NewReader("some io.Reader stream to be read\n")

	r = io.TeeReader(r, os.Stdout)

	// Everything read from r will be copied to stdout.
	ioutil.ReadAll(r)

	// Output:
	// some io.Reader stream to be read
}

func ExampleSectionReader() {
	r := strings.NewReader("some io.Reader stream to be read\n")
	s := io.NewSectionReader(r, 5, 17)

	if _, err := io.Copy(os.Stdout, s); err != nil {
		log.Fatal(err)
	}

	// Output:
	// io.Reader stream
}

func ExampleSectionReader_ReadAt() {
	r := strings.NewReader("some io.Reader stream to be read\n")
	s := io.NewSectionReader(r, 5, 17)

	buf := make([]byte, 6)
	if _, err := s.ReadAt(buf, 10); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("%s\n", buf)

	// Output:
	// stream
}

func ExampleSectionReader_Seek() {
	r := strings.NewReader("some io.Reader stream to be read\n")
	s := io.NewSectionReader(r, 5, 17)

	if _, err := s.Seek(10, io.SeekStart); err != nil {
		log.Fatal(err)
	}

	if _, err := io.Copy(os.Stdout, s); err != nil {
		log.Fatal(err)
	}

	// Output:
	// stream
}

func ExampleSeeker_Seek() {
	r := strings.NewReader("some io.Reader stream to be read\n")

	r.Seek(5, io.SeekStart) // move to the 5th char from the start
	if _, err := io.Copy(os.Stdout, r); err != nil {
		log.Fatal(err)
	}

	r.Seek(-5, io.SeekEnd)
	if _, err := io.Copy(os.Stdout, r); err != nil {
		log.Fatal(err)
	}

	// Output:
	// io.Reader stream to be read
	// read
}

func ExampleMultiWriter() {
	r := strings.NewReader("some io.Reader stream to be read\n")

	var buf1, buf2 bytes.Buffer
	w := io.MultiWriter(&buf1, &buf2)

	if _, err := io.Copy(w, r); err != nil {
		log.Fatal(err)
	}

	fmt.Print(buf1.String())
	fmt.Print(buf2.String())

	// Output:
	// some io.Reader stream to be read
	// some io.Reader stream to be read
}

func ExamplePipe() {
	r, w := io.Pipe()

	go func() {
		fmt.Fprint(w, "some io.Reader stream to be read\n")
		w.Close()
	}()

	if _, err := io.Copy(os.Stdout, r); err != nil {
		log.Fatal(err)
	}

	// Output:
	// some io.Reader stream to be read
}
