// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"errors"
	"internal/poll"
	"os"
	"syscall"
)

func badStateFile() (*os.File, error) {
	if os.Getuid() != 0 {
		return nil, errors.New("must be root")
	}
	// Using OpenFile for a device file is an easy way to make a
	// file attached to the runtime-integrated network poller and
	// configured in halfway.
	return os.OpenFile("/dev/net/tun", os.O_RDWR, 0)
}

func isBadStateFileError(err error) (string, bool) {
	switch err {
	case poll.ErrNotPollable, syscall.EBADFD:
		return "", true
	default:
		return "not pollable or file in bad state error", false
	}
}
