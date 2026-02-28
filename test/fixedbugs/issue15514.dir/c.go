// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package c

import "./a"
import "./b"

var _ a.A = b.B() // ERROR "cannot use b\.B|incompatible type"
