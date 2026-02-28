// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

// LookPath searches for an executable named file in the current path,
// following the conventions of the host operating system.
// If file contains a slash, it is tried directly and the default path is not consulted.
// Otherwise, on success the result is an absolute path.
//
// LookPath returns an error satisfying [errors.Is](err, [ErrDot])
// if the resolved path is relative to the current directory.
// See the package documentation for more details.
//
// LookPath looks for an executable named file in the
// directories named by the PATH environment variable,
// except as described below.
//
//   - On Windows, the file must have an extension named by
//     the PATHEXT environment variable.
//     When PATHEXT is unset, the file must have
//     a ".com", ".exe", ".bat", or ".cmd" extension.
//   - On Plan 9, LookPath consults the path environment variable.
//     If file begins with "/", "#", "./", or "../", it is tried
//     directly and the path is not consulted.
//   - On Wasm, LookPath always returns an error.
func LookPath(file string) (string, error) {
	return lookPath(file)
}
