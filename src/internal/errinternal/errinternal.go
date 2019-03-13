// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errinternal

// NewError creates a new error as created by errors.New, but with one
// additional stack frame depth.
var NewError func(msg string, err error) error
