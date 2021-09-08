// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func f[V any]() []V { return []V{0: *new(V)} }

func g() { f[int]() }
