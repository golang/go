// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hidden

// IsHidden reports whether path is hidden by default in user interfaces
// on the current platform.
func IsHidden(path string) (flag bool, err error) {
	// FIXME: untested code
	flag = strings.HasPrefix(filepath.Base(path), ".")
	if flag {
		return
	}
	flag, err = hasHiddenAttribute(path)

	return
}

func hasHiddenAttribute(filepath string) (bool, error) {
	path, err := syscall.UTF16PtrFromString(filepath)
	if err != nil {
		return false, err
	}

	attrs, err := syscall.GetFileAttributes(path)
	if err != nil {
		return false, err
	}

	return attrs&syscall.FILE_ATTRIBUTE_HIDDEN != 0, nil
}
