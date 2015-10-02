// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code snippets included in "Error Handling and Go."

package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"os"
	"time"
)

type File struct{}

func Open(name string) (file *File, err error) {
	// OMIT
	panic(1)
	// STOP OMIT
}

func openFile() { // OMIT
	f, err := os.Open("filename.ext")
	if err != nil {
		log.Fatal(err)
	}
	// do something with the open *File f
	// STOP OMIT
	_ = f
}

// errorString is a trivial implementation of error.
type errorString struct {
	s string
}

func (e *errorString) Error() string {
	return e.s
}

// STOP OMIT

// New returns an error that formats as the given text.
func New(text string) error {
	return &errorString{text}
}

// STOP OMIT

func Sqrt(f float64) (float64, error) {
	if f < 0 {
		return 0, errors.New("math: square root of negative number")
	}
	// implementation
	return 0, nil // OMIT
}

// STOP OMIT

func printErr() (int, error) { // OMIT
	f, err := Sqrt(-1)
	if err != nil {
		fmt.Println(err)
	}
	// STOP OMIT
	// fmtError OMIT
	if f < 0 {
		return 0, fmt.Errorf("math: square root of negative number %g", f)
	}
	// STOP OMIT
	return 0, nil
}

type NegativeSqrtError float64

func (f NegativeSqrtError) Error() string {
	return fmt.Sprintf("math: square root of negative number %g", float64(f))
}

// STOP OMIT

type SyntaxError struct {
	msg    string // description of error
	Offset int64  // error occurred after reading Offset bytes
}

func (e *SyntaxError) Error() string { return e.msg }

// STOP OMIT

func decodeError(dec *json.Decoder, val struct{}) error { // OMIT
	var f os.FileInfo // OMIT
	if err := dec.Decode(&val); err != nil {
		if serr, ok := err.(*json.SyntaxError); ok {
			line, col := findLine(f, serr.Offset)
			return fmt.Errorf("%s:%d:%d: %v", f.Name(), line, col, err)
		}
		return err
	}
	// STOP OMIT
	return nil
}

func findLine(os.FileInfo, int64) (int, int) {
	// place holder; no need to run
	return 0, 0
}

func netError(err error) { // OMIT
	for { // OMIT
		if nerr, ok := err.(net.Error); ok && nerr.Temporary() {
			time.Sleep(1e9)
			continue
		}
		if err != nil {
			log.Fatal(err)
		}
		// STOP OMIT
	}
}

func main() {}
