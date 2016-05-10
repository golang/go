// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly darwin freebsd !android,linux netbsd openbsd

package user

import (
	"fmt"
	"strconv"
	"unsafe"
)

/*
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>
*/
import "C"

func listGroups(u *User) ([]string, error) {
	ug, err := strconv.Atoi(u.Gid)
	if err != nil {
		return nil, fmt.Errorf("user: list groups for %s: invalid gid %q", u.Username, u.Gid)
	}
	userGID := C.gid_t(ug)
	nameC := C.CString(u.Username)
	defer C.free(unsafe.Pointer(nameC))

	n := C.int(256)
	gidsC := make([]C.gid_t, n)
	rv := getGroupList(nameC, userGID, &gidsC[0], &n)
	if rv == -1 {
		// More than initial buffer, but now n contains the correct size.
		const maxGroups = 2048
		if n > maxGroups {
			return nil, fmt.Errorf("user: list groups for %s: member of more than %d groups", u.Username, maxGroups)
		}
		gidsC = make([]C.gid_t, n)
		rv := getGroupList(nameC, userGID, &gidsC[0], &n)
		if rv == -1 {
			return nil, fmt.Errorf("user: list groups for %s failed (changed groups?)", u.Username)
		}
	}
	gidsC = gidsC[:n]
	gids := make([]string, 0, n)
	for _, g := range gidsC[:n] {
		gids = append(gids, strconv.Itoa(int(g)))
	}
	return gids, nil
}
