// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the deepequalerrors checker.

package a

import (
	"io"
	"os"
	"reflect"
)

type myError int

func (myError) Error() string { return "" }

func bad() error { return nil }

type s1 struct {
	s2 *s2
	i  int
}

type myError2 error

type s2 struct {
	s1   *s1
	errs []*myError2
}

func hasError() {
	var e error
	var m myError2
	reflect.DeepEqual(bad(), e)           // want `avoid using reflect.DeepEqual with errors`
	reflect.DeepEqual(io.EOF, io.EOF)     // want `avoid using reflect.DeepEqual with errors`
	reflect.DeepEqual(e, &e)              // want `avoid using reflect.DeepEqual with errors`
	reflect.DeepEqual(e, m)               // want `avoid using reflect.DeepEqual with errors`
	reflect.DeepEqual(e, s1{})            // want `avoid using reflect.DeepEqual with errors`
	reflect.DeepEqual(e, [1]error{})      // want `avoid using reflect.DeepEqual with errors`
	reflect.DeepEqual(e, map[error]int{}) // want `avoid using reflect.DeepEqual with errors`
	reflect.DeepEqual(e, map[int]error{}) // want `avoid using reflect.DeepEqual with errors`
	// We catch the next not because *os.PathError implements error, but because it contains
	// a field Err of type error.
	reflect.DeepEqual(&os.PathError{}, io.EOF) // want `avoid using reflect.DeepEqual with errors`

}

func notHasError() {
	reflect.ValueOf(4)                    // not reflect.DeepEqual
	reflect.DeepEqual(3, 4)               // not errors
	reflect.DeepEqual(5, io.EOF)          // only one error
	reflect.DeepEqual(myError(1), io.EOF) // not types that implement error
}
