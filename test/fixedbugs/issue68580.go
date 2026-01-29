// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type A[P any] = struct{ _ P }

type N[P any] A[P]

func f[P any](N[P]) {}

var _ = f[int]
