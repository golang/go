// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
//go:build go1.18

package typeparams

import (
	"bytes"
	"errors"
	"fmt"
	"typeparams/userdefs"
)

func _[T any]() {
	fmt.Errorf("") // want "result of fmt.Errorf call not used"
	_ = fmt.Errorf("")

	errors.New("") // want "result of errors.New call not used"

	err := errors.New("")
	err.Error() // want `result of \(error\).Error call not used`

	var buf bytes.Buffer
	buf.String() // want `result of \(bytes.Buffer\).String call not used`

	fmt.Sprint("")  // want "result of fmt.Sprint call not used"
	fmt.Sprintf("") // want "result of fmt.Sprintf call not used"

	userdefs.MustUse[int](1) // want "result of typeparams/userdefs.MustUse call not used"
	_ = userdefs.MustUse[int](2)

	s := userdefs.SingleTypeParam[int]{X: 1}
	s.String() // want `result of \(typeparams/userdefs.SingleTypeParam\[int\]\).String call not used`
	_ = s.String()

	m := userdefs.MultiTypeParam[int, string]{X: 1, Y: "one"}
	m.String() // want `result of \(typeparams/userdefs.MultiTypeParam\[int, string\]\).String call not used`
	_ = m.String()
}