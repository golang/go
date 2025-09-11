// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (cgo || darwin) && !osusergo && (darwin || dragonfly || freebsd || (linux && !android) || netbsd || openbsd || solaris)

package user

import (
	"fmt"
	"strconv"
	"unsafe"
)

const maxGroups = 2048

func listGroups(u *User) ([]string, error) {
	ug, err := strconv.Atoi(u.Gid)
	if err != nil {
		return nil, fmt.Errorf("user: list groups for %s: invalid gid %q", u.Username, u.Gid)
	}
	userGID := _C_gid_t(ug)
	nameC := make([]byte, len(u.Username)+1)
	copy(nameC, u.Username)

	n := _C_int(256)
	gidsC := make([]_C_gid_t, n)
	rv := getGroupList((*_C_char)(unsafe.Pointer(&nameC[0])), userGID, &gidsC[0], &n)
	if rv == -1 {
		// Mac is the only Unix that does not set n properly when rv == -1, so
		// we need to use different logic for Mac vs. the other OS's.
		if err := groupRetry(u.Username, nameC, userGID, &gidsC, &n); err != nil {
			return nil, err
		}
	}
	gidsC = gidsC[:n]
	gids := make([]string, 0, n)
	for _, g := range gidsC[:n] {
		gids = append(gids, strconv.Itoa(int(g)))
	}
	return gids, nil
}

// groupRetry retries getGroupList with much larger size for n. The result is
// stored in gids.
func groupRetry(username string, name []byte, userGID _C_gid_t, gids *[]_C_gid_t, n *_C_int) error {
	// More than initial buffer, but now n contains the correct size.
	if *n > maxGroups {
		return fmt.Errorf("user: %q is a member of more than %d groups", username, maxGroups)
	}
	*gids = make([]_C_gid_t, *n)
	rv := getGroupList((*_C_char)(unsafe.Pointer(&name[0])), userGID, &(*gids)[0], n)
	if rv == -1 {
		return fmt.Errorf("user: list groups for %s failed", username)
	}
	return nil
}
