// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

func (file *File) Readdirnames(count int) (names []string, err Error) {
	fis, e := file.Readdir(count)
	if e != nil {
		return nil, e
	}
	names = make([]string, len(fis))
	for i, fi := range fis {
		names[i] = fi.Name
	}
	return names, nil
}
