// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "fmt"

// Unveil implements the unveil syscall.
// For more information see unveil(2).
// Note that the special case of blocking further
// unveil calls is handled by UnveilBlock.
func Unveil(path string, flags string) error {
	if err := supportsUnveil(); err != nil {
		return err
	}
	pathPtr, err := BytePtrFromString(path)
	if err != nil {
		return err
	}
	flagsPtr, err := BytePtrFromString(flags)
	if err != nil {
		return err
	}
	return unveil(pathPtr, flagsPtr)
}

// UnveilBlock blocks future unveil calls.
// For more information see unveil(2).
func UnveilBlock() error {
	if err := supportsUnveil(); err != nil {
		return err
	}
	return unveil(nil, nil)
}

// supportsUnveil checks for availability of the unveil(2) system call based
// on the running OpenBSD version.
func supportsUnveil() error {
	maj, min, err := majmin()
	if err != nil {
		return err
	}

	// unveil is not available before 6.4
	if maj < 6 || (maj == 6 && min <= 3) {
		return fmt.Errorf("cannot call Unveil on OpenBSD %d.%d", maj, min)
	}

	return nil
}
