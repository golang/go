// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !osusergo

package user

/*
#include <unistd.h>
#include <sys/types.h>
#include <stdlib.h>

static int mygetgrouplist(const char* user, gid_t group, gid_t* groups, int* ngroups) {
	int* buf = malloc(*ngroups * sizeof(int));
	int rv = getgrouplist(user, (int) group, buf, ngroups);
	int i;
	if (rv == 0) {
		for (i = 0; i < *ngroups; i++) {
			groups[i] = (gid_t) buf[i];
		}
	}
	free(buf);
	return rv;
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

// groupRetry retries getGroupList with an increasingly large size for n. The
// result is stored in gids.
func groupRetry(username string, name []byte, userGID C.gid_t, gids *[]C.gid_t, n *C.int) error {
	*n = C.int(256 * 2)
	for {
		*gids = make([]C.gid_t, *n)
		rv := getGroupList((*C.char)(unsafe.Pointer(&name[0])), userGID, &(*gids)[0], n)
		if rv >= 0 {
			// n is set correctly
			break
		}
		if *n > maxGroups {
			return fmt.Errorf("user: %q is a member of more than %d groups", username, maxGroups)
		}
		*n = *n * C.int(2)
	}
	return nil
}
