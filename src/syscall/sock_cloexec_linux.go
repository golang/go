// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// This is a stripped down version of sysSocket from net/sock_cloexec.go.
func cloexecSocket(family, sotype, proto int) (int, error) {
	s, err := Socket(family, sotype|SOCK_CLOEXEC, proto)
	switch err {
	case nil:
		return s, nil
	default:
		return -1, err
	case EINVAL:
	}

	ForkLock.RLock()
	s, err = Socket(family, sotype, proto)
	if err == nil {
		CloseOnExec(s)
	}
	ForkLock.RUnlock()
	if err != nil {
		Close(s)
		return -1, err
	}
	return s, nil
}
