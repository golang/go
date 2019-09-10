// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codehost

import (
	"errors"
	"testing"
)

func TestRunErrorUnwrap(t *testing.T) {
	werr := errors.New("wrapped error")
	err := &RunError{Err: werr}
	if !errors.Is(err, werr) {
		t.Error("errors.Is failed, wanted success")
	}
}
