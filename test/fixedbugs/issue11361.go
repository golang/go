// errorcheck

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "fmt"  // GC_ERROR "imported and not used"

const n = fmt // ERROR "fmt without selector|unexpected reference to package"
