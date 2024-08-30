// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !osusergo && darwin

package user

import (
	"internal/syscall/unix"
	"syscall"
)

type _C_char = byte
type _C_int = int32
type _C_gid_t = uint32
type _C_uid_t = uint32
type _C_size_t = uintptr
type _C_struct_group = unix.Group
type _C_struct_passwd = unix.Passwd
type _C_long = int64

func _C_pw_uid(p *_C_struct_passwd) _C_uid_t   { return p.Uid }
func _C_pw_uidp(p *_C_struct_passwd) *_C_uid_t { return &p.Uid }
func _C_pw_gid(p *_C_struct_passwd) _C_gid_t   { return p.Gid }
func _C_pw_gidp(p *_C_struct_passwd) *_C_gid_t { return &p.Gid }
func _C_pw_name(p *_C_struct_passwd) *_C_char  { return p.Name }
func _C_pw_gecos(p *_C_struct_passwd) *_C_char { return p.Gecos }
func _C_pw_dir(p *_C_struct_passwd) *_C_char   { return p.Dir }

func _C_gr_gid(g *_C_struct_group) _C_gid_t  { return g.Gid }
func _C_gr_name(g *_C_struct_group) *_C_char { return g.Name }

func _C_GoString(p *_C_char) string { return unix.GoString(p) }

func _C_getpwnam_r(name *_C_char, buf *_C_char, size _C_size_t) (pwd _C_struct_passwd, found bool, errno syscall.Errno) {
	var result *_C_struct_passwd
	errno = unix.Getpwnam(name, &pwd, buf, size, &result)
	return pwd, result != nil, errno
}

func _C_getpwuid_r(uid _C_uid_t, buf *_C_char, size _C_size_t) (pwd _C_struct_passwd, found bool, errno syscall.Errno) {
	var result *_C_struct_passwd
	errno = unix.Getpwuid(uid, &pwd, buf, size, &result)
	return pwd, result != nil, errno
}

func _C_getgrnam_r(name *_C_char, buf *_C_char, size _C_size_t) (grp _C_struct_group, found bool, errno syscall.Errno) {
	var result *_C_struct_group
	errno = unix.Getgrnam(name, &grp, buf, size, &result)
	return grp, result != nil, errno
}

func _C_getgrgid_r(gid _C_gid_t, buf *_C_char, size _C_size_t) (grp _C_struct_group, found bool, errno syscall.Errno) {
	var result *_C_struct_group
	errno = unix.Getgrgid(gid, &grp, buf, size, &result)
	return grp, result != nil, errno
}

const (
	_C__SC_GETPW_R_SIZE_MAX = unix.SC_GETPW_R_SIZE_MAX
	_C__SC_GETGR_R_SIZE_MAX = unix.SC_GETGR_R_SIZE_MAX
)

func _C_sysconf(key _C_int) _C_long { return unix.Sysconf(key) }
