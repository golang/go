// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"fmt"
	"syscall"
	"unsafe"
)

func lookupFullName(domain, username, domainAndUser string) (string, error) {
	// try domain controller first
	name, e := syscall.TranslateAccountName(domainAndUser,
		syscall.NameSamCompatible, syscall.NameDisplay, 50)
	if e != nil {
		// domain lookup failed, perhaps this pc is not part of domain
		d, e := syscall.UTF16PtrFromString(domain)
		if e != nil {
			return "", e
		}
		u, e := syscall.UTF16PtrFromString(username)
		if e != nil {
			return "", e
		}
		var p *byte
		e = syscall.NetUserGetInfo(d, u, 10, &p)
		if e != nil {
			// path executed when a domain user is disconnected from the domain
			// pretend username is fullname
			return username, nil
		}
		defer syscall.NetApiBufferFree(p)
		i := (*syscall.UserInfo10)(unsafe.Pointer(p))
		if i.FullName == nil {
			return "", nil
		}
		name = syscall.UTF16ToString((*[1024]uint16)(unsafe.Pointer(i.FullName))[:])
	}
	return name, nil
}

func newUser(usid *syscall.SID, gid, dir string) (*User, error) {
	username, domain, t, e := usid.LookupAccount("")
	if e != nil {
		return nil, e
	}
	if t != syscall.SidTypeUser {
		return nil, fmt.Errorf("user: should be user account type, not %d", t)
	}
	domainAndUser := domain + `\` + username
	uid, e := usid.String()
	if e != nil {
		return nil, e
	}
	name, e := lookupFullName(domain, username, domainAndUser)
	if e != nil {
		return nil, e
	}
	u := &User{
		Uid:      uid,
		Gid:      gid,
		Username: domainAndUser,
		Name:     name,
		HomeDir:  dir,
	}
	return u, nil
}

func current() (*User, error) {
	t, e := syscall.OpenCurrentProcessToken()
	if e != nil {
		return nil, e
	}
	u, e := t.GetTokenUser()
	if e != nil {
		return nil, e
	}
	pg, e := t.GetTokenPrimaryGroup()
	if e != nil {
		return nil, e
	}
	gid, e := pg.PrimaryGroup.String()
	if e != nil {
		return nil, e
	}
	dir, e := t.GetUserProfileDirectory()
	if e != nil {
		return nil, e
	}
	return newUser(u.User.Sid, gid, dir)
}

// BUG(brainman): Lookup and LookupId functions do not set
// Gid and HomeDir fields in the User struct returned on windows.

func newUserFromSid(usid *syscall.SID) (*User, error) {
	// TODO(brainman): do not know where to get gid and dir fields
	gid := "unknown"
	dir := "Unknown directory"
	return newUser(usid, gid, dir)
}

func lookup(username string) (*User, error) {
	sid, _, t, e := syscall.LookupSID("", username)
	if e != nil {
		return nil, e
	}
	if t != syscall.SidTypeUser {
		return nil, fmt.Errorf("user: should be user account type, not %d", t)
	}
	return newUserFromSid(sid)
}

func lookupId(uid string) (*User, error) {
	sid, e := syscall.StringToSid(uid)
	if e != nil {
		return nil, e
	}
	return newUserFromSid(sid)
}
