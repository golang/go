// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Plan 9-specific

package os

func hostname() (name string, err error) {
	f, err := Open("#c/sysname")
	if err != nil {
		return "", err
	}
	defer f.Close()

	var buf [128]byte
	n, err := f.Read(buf[:len(buf)-1])

	if err != nil {
		return "", err
	}
	if n > 0 {
		buf[n] = 0
	}
	return string(buf[0:n]), nil
}
