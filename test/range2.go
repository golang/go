// errorcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// See ../internal/types/testdata/spec/range.go for most tests.
// The ones in this file cannot be expressed in that framework
// due to conflicts between that framework's error location pickiness
// and gofmt's comment location pickiness.

package p

type T struct{}

func (*T) PM() {}
func (T) M()   {}

func test() {
	for range T.M { // ERROR "cannot range over T.M \(value of type func\(T\)\): func must be func\(yield func\(...\) bool\): argument is not func"
	}
	for range (*T).PM { // ERROR "cannot range over \(\*T\).PM \(value of type func\(\*T\)\): func must be func\(yield func\(...\) bool\): argument is not func"
	}
}
