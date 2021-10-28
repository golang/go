// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import "testing"

func Test(*testing.T) {
	_ = Zero[int]() // It is fine to use generics within tests.
}

// Note: We format {Test,Benchmark}typeParam with a 't' in "type" to avoid an error from
// cmd/go/internal/load. That package can also give an error about Test and Benchmark
// functions with TypeParameters. These tests may need to be updated if that logic changes.
func TesttypeParam[T any](*testing.T)      {} // want "TesttypeParam has type parameters: it will not be run by go test as a TestXXX function" "TesttypeParam has malformed name"
func BenchmarktypeParam[T any](*testing.B) {} // want "BenchmarktypeParam has type parameters: it will not be run by go test as a BenchmarkXXX function" "BenchmarktypeParam has malformed name"

func ExampleZero[T any]() { // want "ExampleZero should not have type params"
	print(Zero[T]())
}
