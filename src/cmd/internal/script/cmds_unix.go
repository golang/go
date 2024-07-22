// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package script

import (
	"errors"
	"syscall"
)

func isETXTBSY(err error) bool {
	return errors.Is(err, syscall.ETXTBSY)
}
