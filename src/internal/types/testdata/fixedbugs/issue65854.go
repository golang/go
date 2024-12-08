// -gotypesalias=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A = int

type T[P any] *A

var _ T[int]
