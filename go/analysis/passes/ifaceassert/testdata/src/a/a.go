// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the ifaceassert checker.

package a

import "io"

func InterfaceAssertionTest() {
	var (
		a io.ReadWriteSeeker
		b interface {
			Read()
			Write()
		}
	)
	_ = a.(io.Reader)
	_ = a.(io.ReadWriter)
	_ = b.(io.Reader)  // want `^impossible type assertion: no type can implement both interface{Read\(\); Write\(\)} and io.Reader \(conflicting types for Read method\)$`
	_ = b.(interface { // want `^impossible type assertion: no type can implement both interface{Read\(\); Write\(\)} and interface{Read\(p \[\]byte\) \(n int, err error\)} \(conflicting types for Read method\)$`
		Read(p []byte) (n int, err error)
	})

	switch a.(type) {
	case io.ReadWriter:
	case interface { // want `^impossible type assertion: no type can implement both io.ReadWriteSeeker and interface{Write\(\)} \(conflicting types for Write method\)$`
		Write()
	}:
	default:
	}

	switch b := b.(type) {
	case io.ReadWriter, interface{ Read() }: // want `^impossible type assertion: no type can implement both interface{Read\(\); Write\(\)} and io.ReadWriter \(conflicting types for Read method\)$`
	case io.Writer: // want `^impossible type assertion: no type can implement both interface{Read\(\); Write\(\)} and io.Writer \(conflicting types for Write method\)$`
	default:
		_ = b
	}
}
