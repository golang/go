// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

// EnvForDir returns a modified environment suitable for running in the given
// directory.
// The environment is the supplied base environment but with an updated $PWD, so
// that an os.Getwd in the child will be faster.
func EnvForDir(dir string, base []string) []string {
	// Internally we only use rooted paths, so dir is rooted.
	// Even if dir is not rooted, no harm done.
	return append(base, "PWD="+dir)
}
