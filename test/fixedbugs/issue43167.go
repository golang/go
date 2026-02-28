// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import . "bytes"

var _ Buffer // use package bytes

var Index byte // ERROR "Index redeclared.*\n\tLINE-4: previous declaration during import .bytes.|already declared|redefinition"
