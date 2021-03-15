// errorcheck -lang=go1.15

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import _ "embed"

//go:embed x.txt // ERROR "go:embed requires go1.16 or later"
var x string
