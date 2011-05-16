// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

func (file *File) Readdirnames(n int) (names []string, err Error) {
	fis, err := file.Readdir(n)
	// If n > 0 and we get an error, we return now.
	// If n < 0, we return whatever we got + any error.
	if n > 0 && e != nil {
		return nil, err
	}
	names = make([]string, len(fis))
	for i, fi := range fis {
		names[i] = fi.Name
	}
	return names, err
}
