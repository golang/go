// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd !android,linux netbsd openbsd
// +build cgo,!osusergo

package user

/*
#include <unistd.h>
#include <sys/types.h>
#include <grp.h>

static int mygetgrouplist(const char* user, gid_t group, gid_t* groups, int* ngroups) {
	return getgrouplist(user, group, groups, ngroups);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func getGroupList(name *C.char, userGID C.gid_t, gids *C.gid_t, n *C.int) C.int {
	return C.mygetgrouplist(name, userGID, gids, n)
}

// groupRetry retries getGroupList with much larger size for n. The result is
// stored in gids.
func groupRetry(username string, name []byte, userGID C.gid_t, gids *[]C.gid_t, n *C.int) error {
	// More than initial buffer, but now n contains the correct size.
	if *n > maxGroups {
		return fmt.Errorf("user: %q is a member of more than %d groups", username, maxGroups)
	}
	*gids = make([]C.gid_t, *n)
	rv := getGroupList((*C.char)(unsafe.Pointer(&name[0])), userGID, &(*gids)[0], n)
	if rv == -1 {
		return fmt.Errorf("user: list groups for %s failed", username)
	}
	return nil
}
