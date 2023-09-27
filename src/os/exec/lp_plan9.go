// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ErrNotFound is the error resulting if a path search failed to find an executable file.
var ErrNotFound = errors.New("executable file not found in $path")

func findExecutable(file string) error {
	d, err := os.Stat(file)
	if err != nil {
		return err
	}
	if m := d.Mode(); !m.IsDir() && m&0111 != 0 {
		return nil
	}
	return fs.ErrPermission
}

// LookPath searches for an executable named file in the
// directories named by the path environment variable.
// If file begins with "/", "#", "./", or "../", it is tried
// directly and the path is not consulted.
// On success, the result is an absolute path.
//
// In older versions of Go, LookPath could return a path relative to the current directory.
// As of Go 1.19, LookPath will instead return that path along with an error satisfying
// errors.Is(err, ErrDot). See the package documentation for more details.
func LookPath(file string) (string, error) {
	// skip the path lookup for these prefixes
	skip := []string{"/", "#", "./", "../"}

	for _, p := range skip {
		if strings.HasPrefix(file, p) {
			err := findExecutable(file)
			if err == nil {
				return file, nil
			}
			return "", &Error{file, err}
		}
	}

	path := os.Getenv("path")
	for _, dir := range filepath.SplitList(path) {
		path := filepath.Join(dir, file)
		if err := findExecutable(path); err == nil {
			if !filepath.IsAbs(path) {
				if execerrdot.Value() != "0" {
					return path, &Error{file, ErrDot}
				}
				execerrdot.IncNonDefault()
			}
			return path, nil
		}
	}
	return "", &Error{file, ErrNotFound}
}

// lookExtensions is a no-op on non-Windows platforms, since
// they do not restrict executables to specific extensions.
func lookExtensions(path, dir string) (string, error) {
	return path, nil
}
