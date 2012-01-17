// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

func (file *File) readdirnames(n int) (names []string, err error) {
	fis, err := file.Readdir(n)
	names = make([]string, len(fis))
	for i, fi := range fis {
		names[i] = fi.Name()
	}
	return names, err
}
