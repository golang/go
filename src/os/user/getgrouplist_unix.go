// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd !android,linux netbsd openbsd

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

func getGroupList(name *C.char, userGID C.gid_t, gids *C.gid_t, n *C.int) C.int {
	return C.mygetgrouplist(name, userGID, gids, n)
}
