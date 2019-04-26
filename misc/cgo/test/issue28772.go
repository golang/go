// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Constants didn't work if defined in different source file.

// #define issue28772Constant2 2
import "C"

const issue28772Constant2 = C.issue28772Constant2
