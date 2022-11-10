// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !android && cgo && !osusergo

package user

import (
	"fmt"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

/*
#cgo solaris CFLAGS: -D_POSIX_PTHREAD_SEMANTICS
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <grp.h>
#include <stdlib.h>

static struct passwd mygetpwuid_r(int uid, char *buf, size_t buflen, int *found, int *perr) {
	struct passwd pwd;
        struct passwd *result;
        *perr = getpwuid_r(uid, &pwd, buf, buflen, &result);
        *found = result != NULL;
        return pwd;
}

static struct passwd mygetpwnam_r(const char *name, char *buf, size_t buflen, int *found, int *perr) {
	struct passwd pwd;
        struct passwd *result;
        *perr = getpwnam_r(name, &pwd, buf, buflen, &result);
        *found = result != NULL;
        return pwd;
}

static struct group mygetgrgid_r(int gid, char *buf, size_t buflen, int *found, int *perr) {
	struct group grp;
        struct group *result;
        *perr = getgrgid_r(gid, &grp, buf, buflen, &result);
        *found = result != NULL;
        return grp;
}

static struct group mygetgrnam_r(const char *name, char *buf, size_t buflen, int *found, int *perr) {
	struct group grp;
        struct group *result;
        *perr = getgrnam_r(name, &grp, buf, buflen, &result);
        *found = result != NULL;
        return grp;
}
*/
import "C"

func current() (*User, error) {
	return lookupUnixUid(syscall.Getuid())
}

func lookupUser(username string) (*User, error) {
	var pwd C.struct_passwd
	var found bool
	nameC := make([]byte, len(username)+1)
	copy(nameC, username)

	err := retryWithBuffer(userBuffer, func(buf []byte) syscall.Errno {
		var cfound, cerr C.int
		pwd = C.mygetpwnam_r((*C.char)(unsafe.Pointer(&nameC[0])),
			(*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)),
			&cfound, &cerr)
		found = cfound != 0
		return syscall.Errno(cerr)
	})
	if err != nil {
		return nil, fmt.Errorf("user: lookup username %s: %v", username, err)
	}
	if !found {
		return nil, UnknownUserError(username)
	}
	return buildUser(&pwd), err
}

func lookupUserId(uid string) (*User, error) {
	i, e := strconv.Atoi(uid)
	if e != nil {
		return nil, e
	}
	return lookupUnixUid(i)
}

func lookupUnixUid(uid int) (*User, error) {
	var pwd C.struct_passwd
	var found bool

	err := retryWithBuffer(userBuffer, func(buf []byte) syscall.Errno {
		var cfound, cerr C.int
		pwd = C.mygetpwuid_r(C.int(uid),
			(*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)),
			&cfound, &cerr)
		found = cfound != 0
		return syscall.Errno(cerr)
	})
	if err != nil {
		return nil, fmt.Errorf("user: lookup userid %d: %v", uid, err)
	}
	if !found {
		return nil, UnknownUserIdError(uid)
	}
	return buildUser(&pwd), nil
}

func buildUser(pwd *C.struct_passwd) *User {
	u := &User{
		Uid:      strconv.FormatUint(uint64(pwd.pw_uid), 10),
		Gid:      strconv.FormatUint(uint64(pwd.pw_gid), 10),
		Username: C.GoString(pwd.pw_name),
		Name:     C.GoString(pwd.pw_gecos),
		HomeDir:  C.GoString(pwd.pw_dir),
	}
	// The pw_gecos field isn't quite standardized. Some docs
	// say: "It is expected to be a comma separated list of
	// personal data where the first item is the full name of the
	// user."
	u.Name, _, _ = strings.Cut(u.Name, ",")
	return u
}

func lookupGroup(groupname string) (*Group, error) {
	var grp C.struct_group
	var found bool

	cname := make([]byte, len(groupname)+1)
	copy(cname, groupname)

	err := retryWithBuffer(groupBuffer, func(buf []byte) syscall.Errno {
		var cfound, cerr C.int
		grp = C.mygetgrnam_r((*C.char)(unsafe.Pointer(&cname[0])),
			(*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)),
			&cfound, &cerr)
		found = cfound != 0
		return syscall.Errno(cerr)
	})
	if err != nil {
		return nil, fmt.Errorf("user: lookup groupname %s: %v", groupname, err)
	}
	if !found {
		return nil, UnknownGroupError(groupname)
	}
	return buildGroup(&grp), nil
}

func lookupGroupId(gid string) (*Group, error) {
	i, e := strconv.Atoi(gid)
	if e != nil {
		return nil, e
	}
	return lookupUnixGid(i)
}

func lookupUnixGid(gid int) (*Group, error) {
	var grp C.struct_group
	var found bool

	err := retryWithBuffer(groupBuffer, func(buf []byte) syscall.Errno {
		var cfound, cerr C.int
		grp = C.mygetgrgid_r(C.int(gid),
			(*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)),
			&cfound, &cerr)
		found = cfound != 0
		return syscall.Errno(cerr)
	})
	if err != nil {
		return nil, fmt.Errorf("user: lookup groupid %d: %v", gid, err)
	}
	if !found {
		return nil, UnknownGroupIdError(strconv.Itoa(gid))
	}
	return buildGroup(&grp), nil
}

func buildGroup(grp *C.struct_group) *Group {
	g := &Group{
		Gid:  strconv.Itoa(int(grp.gr_gid)),
		Name: C.GoString(grp.gr_name),
	}
	return g
}

type bufferKind C.int

const (
	userBuffer  = bufferKind(C._SC_GETPW_R_SIZE_MAX)
	groupBuffer = bufferKind(C._SC_GETGR_R_SIZE_MAX)
)

func (k bufferKind) initialSize() C.size_t {
	sz := C.sysconf(C.int(k))
	if sz == -1 {
		// DragonFly and FreeBSD do not have _SC_GETPW_R_SIZE_MAX.
		// Additionally, not all Linux systems have it, either. For
		// example, the musl libc returns -1.
		return 1024
	}
	if !isSizeReasonable(int64(sz)) {
		// Truncate.  If this truly isn't enough, retryWithBuffer will error on the first run.
		return maxBufferSize
	}
	return C.size_t(sz)
}

// retryWithBuffer repeatedly calls f(), increasing the size of the
// buffer each time, until f succeeds, fails with a non-ERANGE error,
// or the buffer exceeds a reasonable limit.
func retryWithBuffer(startSize bufferKind, f func([]byte) syscall.Errno) error {
	buf := make([]byte, startSize)
	for {
		errno := f(buf)
		if errno == 0 {
			return nil
		} else if errno != syscall.ERANGE {
			return errno
		}
		newSize := len(buf) * 2
		if !isSizeReasonable(int64(newSize)) {
			return fmt.Errorf("internal buffer exceeds %d bytes", maxBufferSize)
		}
		buf = make([]byte, newSize)
	}
}

const maxBufferSize = 1 << 20

func isSizeReasonable(sz int64) bool {
	return sz > 0 && sz <= maxBufferSize
}

// Because we can't use cgo in tests:
func structPasswdForNegativeTest() C.struct_passwd {
	sp := C.struct_passwd{}
	sp.pw_uid = 1<<32 - 2
	sp.pw_gid = 1<<32 - 3
	return sp
}
