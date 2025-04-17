// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !osusergo && (dragonfly || freebsd || (!android && linux) || netbsd || openbsd || (solaris && !illumos))

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

func getGroupList(name *_C_char, userGID _C_gid_t, gids *_C_gid_t, n *_C_int) _C_int {
	return C.mygetgrouplist(name, userGID, gids, n)
}
