// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"bytes"
	"errors"
	"fmt"
)

func _() {
	fmt.Errorf("") // want "result of fmt.Errorf call not used"
	_ = fmt.Errorf("")

	errors.New("") // want "result of errors.New call not used"

	err := errors.New("")
	err.Error() // want `result of \(error\).Error call not used`

	var buf bytes.Buffer
	buf.String() // want `result of \(bytes.Buffer\).String call not used`

	fmt.Sprint("")  // want "result of fmt.Sprint call not used"
	fmt.Sprintf("") // want "result of fmt.Sprintf call not used"
}
