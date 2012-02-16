// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

// IsExist returns whether the error is known to report that a file already exists.
func IsExist(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return contains(err.Error(), " exists")
}

// IsNotExist returns whether the error is known to report that a file does not exist.
func IsNotExist(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return contains(err.Error(), "does not exist")
}

// IsPermission returns whether the error is known to report that permission is denied.
func IsPermission(err error) bool {
	if pe, ok := err.(*PathError); ok {
		err = pe.Err
	}
	return contains(err.Error(), "permission denied")
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
