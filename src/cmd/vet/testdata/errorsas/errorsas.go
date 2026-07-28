// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errorsas

import "errors"

type MyError struct{}

func (MyError) Error() string { return "" }

func _(err error) {
	var target MyError
	errors.As(err, target) // ERROR "second argument to errors.As must be a non-nil pointer"
}
