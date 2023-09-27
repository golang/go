// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// simplified test case

type transform[T any] struct{}
type pair[S any] struct {}

var _ transform[step]

type box transform[step]
type step = pair[box]

// test case from issue

type Transform[T any] struct{ hold T }
type Pair[S, T any] struct {
	First  S
	Second T
}

var first Transform[Step]

// This line doesn't use the Step alias, and it compiles fine if you uncomment it.
var second Transform[Pair[Box, interface{}]]

type Box *Transform[Step]

// This line is the same as the `first` line, but it comes after the Box declaration and
// does not break the compile.
var third Transform[Step]

type Step = Pair[Box, interface{}]

// This line also does not break the compile
var fourth Transform[Step]
