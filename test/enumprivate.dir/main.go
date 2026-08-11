// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./enumlib"

func incomplete(option enumlib.Option[int]) {
	switch option { // ERROR "non-exhaustive enum switch.*hidden"
	case Some:
	case None:
	case nil:
	}
}
