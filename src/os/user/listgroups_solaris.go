// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!osusergo

package user

import (
	"fmt"
	"strconv"
	"sync"
	"unsafe"
)

// #include <grp.h>
import "C"

var listGroupsLock sync.Mutex

func listGroups(u *User) ([]string, error) {
	listGroupsLock.Lock()
	defer listGroupsLock.Unlock()

	ug, err := strconv.Atoi(u.Gid)
	if err != nil {
		return nil, fmt.Errorf("user: list groups for %s: invalid gid %q", u.Username, u.Gid)
	}

	gids := []string{u.Gid}

	C.setgrent()
	for grp := C.getgrent(); grp != nil; grp = C.getgrent() {
		if int(grp.gr_gid) != ug && grp.gr_mem != nil {
			mem := (*[1<<30 - 1]*C.char)(unsafe.Pointer(grp.gr_mem))
			for i := 0; mem[i] != nil; i++ {
				if C.GoString(mem[i]) == u.Username {
					gids = append(gids, strconv.Itoa(int(grp.gr_gid)))
					break
				}
			}
		}
	}
	C.endgrent()

	return gids, nil
}
