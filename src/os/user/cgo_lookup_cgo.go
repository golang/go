// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !osusergo && unix && !android && !darwin

package user

import (
	"syscall"
)

/*
#cgo solaris CFLAGS: -D_POSIX_PTHREAD_SEMANTICS
#cgo CFLAGS: -fno-stack-protector
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <grp.h>
#include <stdlib.h>
#include <string.h>

static struct passwd mygetpwuid_r(int uid, char *buf, size_t buflen, int *found, int *perr) {
	struct passwd pwd;
	struct passwd *result;
	memset (&pwd, 0, sizeof(pwd));
	*perr = getpwuid_r(uid, &pwd, buf, buflen, &result);
	*found = result != NULL;
	return pwd;
}

static struct passwd mygetpwnam_r(const char *name, char *buf, size_t buflen, int *found, int *perr) {
	struct passwd pwd;
	struct passwd *result;
	memset(&pwd, 0, sizeof(pwd));
	*perr = getpwnam_r(name, &pwd, buf, buflen, &result);
	*found = result != NULL;
	return pwd;
}

static struct group mygetgrgid_r(int gid, char *buf, size_t buflen, int *found, int *perr) {
	struct group grp;
	struct group *result;
	memset(&grp, 0, sizeof(grp));
	*perr = getgrgid_r(gid, &grp, buf, buflen, &result);
	*found = result != NULL;
	return grp;
}

static struct group mygetgrnam_r(const char *name, char *buf, size_t buflen, int *found, int *perr) {
	struct group grp;
	struct group *result;
	memset(&grp, 0, sizeof(grp));
	*perr = getgrnam_r(name, &grp, buf, buflen, &result);
	*found = result != NULL;
	return grp;
}
*/
import "C"

type _C_char = C.char
type _C_int = C.int
type _C_gid_t = C.gid_t
type _C_uid_t = C.uid_t
type _C_size_t = C.size_t
type _C_struct_group = C.struct_group
type _C_struct_passwd = C.struct_passwd
type _C_long = C.long

func _C_pw_uid(p *_C_struct_passwd) _C_uid_t   { return p.pw_uid }
func _C_pw_uidp(p *_C_struct_passwd) *_C_uid_t { return &p.pw_uid }
func _C_pw_gid(p *_C_struct_passwd) _C_gid_t   { return p.pw_gid }
func _C_pw_gidp(p *_C_struct_passwd) *_C_gid_t { return &p.pw_gid }
func _C_pw_name(p *_C_struct_passwd) *_C_char  { return p.pw_name }
func _C_pw_gecos(p *_C_struct_passwd) *_C_char { return p.pw_gecos }
func _C_pw_dir(p *_C_struct_passwd) *_C_char   { return p.pw_dir }

func _C_gr_gid(g *_C_struct_group) _C_gid_t  { return g.gr_gid }
func _C_gr_name(g *_C_struct_group) *_C_char { return g.gr_name }

func _C_GoString(p *_C_char) string { return C.GoString(p) }

func _C_getpwnam_r(name *_C_char, buf *_C_char, size _C_size_t) (pwd _C_struct_passwd, found bool, errno syscall.Errno) {
	var f, e _C_int
	pwd = C.mygetpwnam_r(name, buf, size, &f, &e)
	return pwd, f != 0, syscall.Errno(e)
}

func _C_getpwuid_r(uid _C_uid_t, buf *_C_char, size _C_size_t) (pwd _C_struct_passwd, found bool, errno syscall.Errno) {
	var f, e _C_int
	pwd = C.mygetpwuid_r(_C_int(uid), buf, size, &f, &e)
	return pwd, f != 0, syscall.Errno(e)
}

func _C_getgrnam_r(name *_C_char, buf *_C_char, size _C_size_t) (grp _C_struct_group, found bool, errno syscall.Errno) {
	var f, e _C_int
	grp = C.mygetgrnam_r(name, buf, size, &f, &e)
	return grp, f != 0, syscall.Errno(e)
}

func _C_getgrgid_r(gid _C_gid_t, buf *_C_char, size _C_size_t) (grp _C_struct_group, found bool, errno syscall.Errno) {
	var f, e _C_int
	grp = C.mygetgrgid_r(_C_int(gid), buf, size, &f, &e)
	return grp, f != 0, syscall.Errno(e)
}

const (
	_C__SC_GETPW_R_SIZE_MAX = C._SC_GETPW_R_SIZE_MAX
	_C__SC_GETGR_R_SIZE_MAX = C._SC_GETGR_R_SIZE_MAX
)

func _C_sysconf(key _C_int) _C_long { return C.sysconf(key) }
