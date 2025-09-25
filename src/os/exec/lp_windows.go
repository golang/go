// Copyright 2010 The Go Authors. All rights reserved.
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
var ErrNotFound = errors.New("executable file not found in %PATH%")

func chkStat(file string) error {
	d, err := os.Stat(file)
	if err != nil {
		return err
	}
	if d.IsDir() {
		return fs.ErrPermission
	}
	return nil
}

func hasExt(file string) bool {
	i := strings.LastIndex(file, ".")
	if i < 0 {
		return false
	}
	return strings.LastIndexAny(file, `:\/`) < i
}

func findExecutable(file string, exts []string) (string, error) {
	if len(exts) == 0 {
		return file, chkStat(file)
	}
	if hasExt(file) {
		if chkStat(file) == nil {
			return file, nil
		}
		// Keep checking exts below, so that programs with weird names
		// like "foo.bat.exe" will resolve instead of failing.
	}
	for _, e := range exts {
		if f := file + e; chkStat(f) == nil {
			return f, nil
		}
	}
	if hasExt(file) {
		return "", fs.ErrNotExist
	}
	return "", ErrNotFound
}

// LookPath searches for an executable named file in the
// directories named by the PATH environment variable.
// LookPath also uses PATHEXT environment variable to match
// a suitable candidate.
// If file contains a slash, it is tried directly and the PATH is not consulted.
// Otherwise, on success, the result is an absolute path.
//
// In older versions of Go, LookPath could return a path relative to the current directory.
// As of Go 1.19, LookPath will instead return that path along with an error satisfying
// [errors.Is](err, [ErrDot]). See the package documentation for more details.
func LookPath(file string) (string, error) {
	if err := validateLookPath(file); err != nil {
		return "", &Error{file, err}
	}

	return lookPath(file, pathExt())
}

// lookExtensions finds windows executable by its dir and path.
// It uses LookPath to try appropriate extensions.
// lookExtensions does not search PATH, instead it converts `prog` into `.\prog`.
//
// If the path already has an extension found in PATHEXT,
// lookExtensions returns it directly without searching
// for additional extensions. For example,
// "C:\foo\example.com" would be returned as-is even if the
// program is actually "C:\foo\example.com.exe".
func lookExtensions(path, dir string) (string, error) {
	if err := validateLookPath(path); err != nil {
		return "", &Error{path, err}
	}

	if filepath.Base(path) == path {
		path = "." + string(filepath.Separator) + path
	}
	exts := pathExt()
	if ext := filepath.Ext(path); ext != "" {
		for _, e := range exts {
			if strings.EqualFold(ext, e) {
				// Assume that path has already been resolved.
				return path, nil
			}
		}
	}
	if dir == "" {
		return lookPath(path, exts)
	}
	if filepath.VolumeName(path) != "" {
		return lookPath(path, exts)
	}
	if len(path) > 1 && os.IsPathSeparator(path[0]) {
		return lookPath(path, exts)
	}
	dirandpath := filepath.Join(dir, path)
	// We assume that LookPath will only add file extension.
	lp, err := lookPath(dirandpath, exts)
	if err != nil {
		return "", err
	}
	ext := strings.TrimPrefix(lp, dirandpath)
	return path + ext, nil
}

func pathExt() []string {
	var exts []string
	x := os.Getenv(`PATHEXT`)
	if x != "" {
		for e := range strings.SplitSeq(strings.ToLower(x), `;`) {
			if e == "" {
				continue
			}
			if e[0] != '.' {
				e = "." + e
			}
			exts = append(exts, e)
		}
	} else {
		exts = []string{".com", ".exe", ".bat", ".cmd"}
	}
	return exts
}

// lookPath implements LookPath for the given PATHEXT list.
func lookPath(file string, exts []string) (string, error) {
	if strings.ContainsAny(file, `:\/`) {
		f, err := findExecutable(file, exts)
		if err == nil {
			return f, nil
		}
		return "", &Error{file, err}
	}

	// On Windows, creating the NoDefaultCurrentDirectoryInExePath
	// environment variable (with any value or no value!) signals that
	// path lookups should skip the current directory.
	// In theory we are supposed to call NeedCurrentDirectoryForExePathW
	// "as the registry location of this environment variable can change"
	// but that seems exceedingly unlikely: it would break all users who
	// have configured their environment this way!
	// https://docs.microsoft.com/en-us/windows/win32/api/processenv/nf-processenv-needcurrentdirectoryforexepathw
	// See also go.dev/issue/43947.
	var (
		dotf   string
		dotErr error
	)
	if _, found := os.LookupEnv("NoDefaultCurrentDirectoryInExePath"); !found {
		if f, err := findExecutable(filepath.Join(".", file), exts); err == nil {
			if execerrdot.Value() == "0" {
				execerrdot.IncNonDefault()
				return f, nil
			}
			dotf, dotErr = f, &Error{file, ErrDot}
		}
	}

	path := os.Getenv("path")
	for _, dir := range filepath.SplitList(path) {
		if dir == "" {
			// Skip empty entries, consistent with what PowerShell does.
			// (See https://go.dev/issue/61493#issuecomment-1649724826.)
			continue
		}

		if f, err := findExecutable(filepath.Join(dir, file), exts); err == nil {
			if dotErr != nil {
				// https://go.dev/issue/53536: if we resolved a relative path implicitly,
				// and it is the same executable that would be resolved from the explicit %PATH%,
				// prefer the explicit name for the executable (and, likely, no error) instead
				// of the equivalent implicit name with ErrDot.
				//
				// Otherwise, return the ErrDot for the implicit path as soon as we find
				// out that the explicit one doesn't match.
				dotfi, dotfiErr := os.Lstat(dotf)
				fi, fiErr := os.Lstat(f)
				if dotfiErr != nil || fiErr != nil || !os.SameFile(dotfi, fi) {
					return dotf, dotErr
				}
			}

			if !filepath.IsAbs(f) {
				if execerrdot.Value() != "0" {
					// If this is the same relative path that we already found,
					// dotErr is non-nil and we already checked it above.
					// Otherwise, record this path as the one to which we must resolve,
					// with or without a dotErr.
					if dotErr == nil {
						dotf, dotErr = f, &Error{file, ErrDot}
					}
					continue
				}
				execerrdot.IncNonDefault()
			}
			return f, nil
		}
	}

	if dotErr != nil {
		return dotf, dotErr
	}
	return "", &Error{file, ErrNotFound}
}
