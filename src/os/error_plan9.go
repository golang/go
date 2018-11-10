// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

func isExist(err error) bool {
	return checkErrMessageContent(err, " exists")
}

func isNotExist(err error) bool {
	return checkErrMessageContent(err, "does not exist", "not found",
		"has been removed", "no parent")
}

func isPermission(err error) bool {
	return checkErrMessageContent(err, "permission denied")
}

// checkErrMessageContent checks if err message contains one of msgs.
func checkErrMessageContent(err error, msgs ...string) bool {
	if err == nil {
		return false
	}
	err = underlyingError(err)
	for _, msg := range msgs {
		if contains(err.Error(), msg) {
			return true
		}
	}
	return false
}

// contains is a local version of strings.Contains. It knows len(sep) > 1.
func contains(s, sep string) bool {
	n := len(sep)
	c := sep[0]
	for i := 0; i+n <= len(s); i++ {
		if s[i] == c && s[i:i+n] == sep {
			return true
		}
	}
	return false
}
