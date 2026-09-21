// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

type T a.T[T] // ERROR "invalid recursive type T\n.*T refers to a\.T\[T\]\n.*a\.T\[T\] refers to T"

type U a.T[a.T[U]] // ERROR "invalid recursive type U\n.*U refers to a\.T\[a\.T\[U\]\]\n.*a\.T\[a\.T\[U\]\] refers to a\.T\[U\]\n.*a\.T\[U\] refers to U"
