// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the unusedresult checker.

package testdata

import (
	"bytes"
	"errors"
	"fmt"
)

func _() {
	fmt.Errorf("") // ERROR "result of fmt.Errorf call not used"
	_ = fmt.Errorf("")

	errors.New("") // ERROR "result of errors.New call not used"

	err := errors.New("")
	err.Error() // ERROR "result of \(error\).Error call not used"

	var buf bytes.Buffer
	buf.String() // ERROR "result of \(bytes.Buffer\).String call not used"

	fmt.Sprint("")  // ERROR "result of fmt.Sprint call not used"
	fmt.Sprintf("") // ERROR "result of fmt.Sprintf call not used"
}
