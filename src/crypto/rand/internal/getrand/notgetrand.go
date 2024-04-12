// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(linux || dragonfly || freebsd || illumos || solaris || darwin || openbsd || netbsd)

package getrand

import "errors"

func GetRandom(out []byte) error {
	return errors.ErrUnsupported
}
