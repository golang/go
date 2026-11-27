// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import "testing"

func TestDecodeError(t *testing.T) {
	_, err := UnmarshalItem([]string{"invalid-é"})

	if err.Error() != "unmarshal error: character 8" {
		t.Error("invalid error")
	}

	_, err = UnmarshalItem([]string{`"é"`})
	if err.Error() != "invalid string format: character 2" {
		t.Error("invalid error")
	}

	if err.(*UnmarshalError).Unwrap().Error() != "invalid string format" {
		t.Error("invalid wrapped error")
	}
}
