// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "fmt" // ERROR "fmt redeclared"

var _ = fmt.Printf
