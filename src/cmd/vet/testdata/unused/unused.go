// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the unusedresult checker.

package unused

import "fmt"

func _() {
	fmt.Errorf("") // ERROR "result of fmt.Errorf call not used"
}
