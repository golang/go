// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import "os"

// Atob returns the boolean value represented by the string.
// It accepts 1, t, T, TRUE, true, True, 0, f, F, FALSE, false, False.
// Any other value returns an error.
func Atob(str string) (value bool, err os.Error) {
	switch str {
	case "1", "t", "T", "true", "TRUE", "True":
		return true, nil
	case "0", "f", "F", "false", "FALSE", "False":
		return false, nil
	}
	return false, &NumError{str, ErrSyntax}
}

// Btoa returns "true" or "false" according to the value of the boolean argument
func Btoa(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
