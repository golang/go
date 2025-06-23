// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"errors"
	"fmt"
	"internal/syscall/windows"
	"internal/syscall/windows/registry"
	"runtime"
	"syscall"
	"unsafe"
)

func isDomainJoined() (bool, error) {
	var domain *uint16
	var status uint32
	err := syscall.NetGetJoinInformation(nil, &domain, &status)
	if err != nil {
		return false, err
	}
	syscall.NetApiBufferFree((*byte)(unsafe.Pointer(domain)))
	return status == syscall.NetSetupDomainName, nil
}

func lookupFullNameDomain(domainAndUser string) (string, error) {
	return syscall.TranslateAccountName(domainAndUser,
		syscall.NameSamCompatible, syscall.NameDisplay, 50)
}

func lookupFullNameServer(servername, username string) (string, error) {
	s, e := syscall.UTF16PtrFromString(servername)
	if e != nil {
		return "", e
	}
	u, e := syscall.UTF16PtrFromString(username)
	if e != nil {
		return "", e
	}
	var p *byte
	e = syscall.NetUserGetInfo(s, u, 10, &p)
	if e != nil {
		return "", e
	}
	defer syscall.NetApiBufferFree(p)
	i := (*syscall.UserInfo10)(unsafe.Pointer(p))
	return windows.UTF16PtrToString(i.FullName), nil
}

func lookupFullName(domain, username, domainAndUser string) (string, error) {
	joined, err := isDomainJoined()
	if err == nil && joined {
		name, err := lookupFullNameDomain(domainAndUser)
		if err == nil {
			return name, nil
		}
	}
	name, err := lookupFullNameServer(domain, username)
	if err == nil {
		return name, nil
	}
	// domain worked neither as a domain nor as a server
	// could be domain server unavailable
	// pretend username is fullname
	return username, nil
}

// getProfilesDirectory retrieves the path to the root directory
// where user profiles are stored.
func getProfilesDirectory() (string, error) {
	n := uint32(100)
	for {
		b := make([]uint16, n)
		e := windows.GetProfilesDirectory(&b[0], &n)
		if e == nil {
			return syscall.UTF16ToString(b), nil
		}
		if e != syscall.ERROR_INSUFFICIENT_BUFFER {
			return "", e
		}
		if n <= uint32(len(b)) {
			return "", e
		}
	}
}

func isServiceAccount(sid *syscall.SID) bool {
	if !windows.IsValidSid(sid) {
		// We don't accept SIDs from the public API, so this should never happen.
		// Better be on the safe side and validate anyway.
		return false
	}
	// The following RIDs are considered service user accounts as per
	// https://learn.microsoft.com/en-us/windows/win32/secauthz/well-known-sids and
	// https://learn.microsoft.com/en-us/windows/win32/services/service-user-accounts:
	// - "S-1-5-18": LocalSystem
	// - "S-1-5-19": LocalService
	// - "S-1-5-20": NetworkService
	if windows.GetSidSubAuthorityCount(sid) != windows.SID_REVISION ||
		windows.GetSidIdentifierAuthority(sid) != windows.SECURITY_NT_AUTHORITY {
		return false
	}
	switch windows.GetSidSubAuthority(sid, 0) {
	case windows.SECURITY_LOCAL_SYSTEM_RID,
		windows.SECURITY_LOCAL_SERVICE_RID,
		windows.SECURITY_NETWORK_SERVICE_RID:
		return true
	}
	return false
}

func isValidUserAccountType(sid *syscall.SID, sidType uint32) bool {
	switch sidType {
	case syscall.SidTypeUser:
		return true
	case syscall.SidTypeWellKnownGroup:
		return isServiceAccount(sid)
	}
	return false
}

func isValidGroupAccountType(sidType uint32) bool {
	switch sidType {
	case syscall.SidTypeGroup:
		return true
	case syscall.SidTypeWellKnownGroup:
		// Some well-known groups are also considered service accounts,
		// so isValidUserAccountType would return true for them.
		// We have historically allowed them in LookupGroup and LookupGroupId,
		// so don't treat them as invalid here.
		return true
	case syscall.SidTypeAlias:
		// https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-samr/7b2aeb27-92fc-41f6-8437-deb65d950921#gt_0387e636-5654-4910-9519-1f8326cf5ec0
		// SidTypeAlias should also be treated as a group type next to SidTypeGroup
		// and SidTypeWellKnownGroup:
		// "alias object -> resource group: A group object..."
		//
		// Tests show that "Administrators" can be considered of type SidTypeAlias.
		return true
	}
	return false
}

// lookupUsernameAndDomain obtains the username and domain for usid.
func lookupUsernameAndDomain(usid *syscall.SID) (username, domain string, sidType uint32, e error) {
	username, domain, sidType, e = usid.LookupAccount("")
	if e != nil {
		return "", "", 0, e
	}
	if !isValidUserAccountType(usid, sidType) {
		return "", "", 0, fmt.Errorf("user: should be user account type, not %d", sidType)
	}
	return username, domain, sidType, nil
}

// findHomeDirInRegistry finds the user home path based on the uid.
func findHomeDirInRegistry(uid string) (dir string, e error) {
	k, e := registry.OpenKey(registry.LOCAL_MACHINE, `SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList\`+uid, registry.QUERY_VALUE)
	if e != nil {
		return "", e
	}
	defer k.Close()
	dir, _, e = k.GetStringValue("ProfileImagePath")
	if e != nil {
		return "", e
	}
	return dir, nil
}

// lookupGroupName accepts the name of a group and retrieves the group SID.
func lookupGroupName(groupname string) (string, error) {
	sid, _, t, e := syscall.LookupSID("", groupname)
	if e != nil {
		return "", e
	}
	if !isValidGroupAccountType(t) {
		return "", fmt.Errorf("lookupGroupName: should be group account type, not %d", t)
	}
	return sid.String()
}

// listGroupsForUsernameAndDomain accepts username and domain and retrieves
// a SID list of the local groups where this user is a member.
func listGroupsForUsernameAndDomain(username, domain string) ([]string, error) {
	// Check if both the domain name and user should be used.
	var query string
	joined, err := isDomainJoined()
	if err == nil && joined && len(domain) != 0 {
		query = domain + `\` + username
	} else {
		query = username
	}
	q, err := syscall.UTF16PtrFromString(query)
	if err != nil {
		return nil, err
	}
	var p0 *byte
	var entriesRead, totalEntries uint32
	// https://learn.microsoft.com/en-us/windows/win32/api/lmaccess/nf-lmaccess-netusergetlocalgroups
	// NetUserGetLocalGroups() would return a list of LocalGroupUserInfo0
	// elements which hold the names of local groups where the user participates.
	// The list does not follow any sorting order.
	err = windows.NetUserGetLocalGroups(nil, q, 0, windows.LG_INCLUDE_INDIRECT, &p0, windows.MAX_PREFERRED_LENGTH, &entriesRead, &totalEntries)
	if err != nil {
		return nil, err
	}
	defer syscall.NetApiBufferFree(p0)
	if entriesRead == 0 {
		return nil, nil
	}
	entries := (*[1024]windows.LocalGroupUserInfo0)(unsafe.Pointer(p0))[:entriesRead:entriesRead]
	var sids []string
	for _, entry := range entries {
		if entry.Name == nil {
			continue
		}
		sid, err := lookupGroupName(windows.UTF16PtrToString(entry.Name))
		if err != nil {
			return nil, err
		}
		sids = append(sids, sid)
	}
	return sids, nil
}

func newUser(uid, gid, dir, username, domain string) (*User, error) {
	domainAndUser := domain + `\` + username
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

var (
	// unused variables (in this implementation)
	// modified during test to exercise code paths in the cgo implementation.
	userBuffer  = 0
	groupBuffer = 0
)

func current() (*User, error) {
	// Use runAsProcessOwner to ensure that we can access the process token
	// when calling syscall.OpenCurrentProcessToken if the current thread
	// is impersonating a different user. See https://go.dev/issue/68647.
	var usr *User
	err := runAsProcessOwner(func() error {
		t, e := syscall.OpenCurrentProcessToken()
		if e != nil {
			return e
		}
		defer t.Close()
		u, e := t.GetTokenUser()
		if e != nil {
			return e
		}
		pg, e := t.GetTokenPrimaryGroup()
		if e != nil {
			return e
		}
		uid, e := u.User.Sid.String()
		if e != nil {
			return e
		}
		gid, e := pg.PrimaryGroup.String()
		if e != nil {
			return e
		}
		dir, e := t.GetUserProfileDirectory()
		if e != nil {
			return e
		}
		username, e := windows.GetUserName(syscall.NameSamCompatible)
		if e != nil {
			return e
		}
		displayName, e := windows.GetUserName(syscall.NameDisplay)
		if e != nil {
			// Historically, the username is used as fallback
			// when the display name can't be retrieved.
			displayName = username
		}
		usr = &User{
			Uid:      uid,
			Gid:      gid,
			Username: username,
			Name:     displayName,
			HomeDir:  dir,
		}
		return nil
	})
	return usr, err
}

// runAsProcessOwner runs f in the context of the current process owner,
// that is, removing any impersonation that may be in effect before calling f,
// and restoring the impersonation afterwards.
func runAsProcessOwner(f func() error) error {
	var impersonationRollbackErr error
	runtime.LockOSThread()
	defer func() {
		// If impersonation failed, the thread is running with the wrong token,
		// so it's better to terminate it.
		// This is achieved by not calling runtime.UnlockOSThread.
		if impersonationRollbackErr != nil {
			println("os/user: failed to revert to previous token:", impersonationRollbackErr.Error())
			runtime.Goexit()
		} else {
			runtime.UnlockOSThread()
		}
	}()
	prevToken, isProcessToken, err := getCurrentToken()
	if err != nil {
		return fmt.Errorf("os/user: failed to get current token: %w", err)
	}
	defer prevToken.Close()
	if !isProcessToken {
		if err = windows.RevertToSelf(); err != nil {
			return fmt.Errorf("os/user: failed to revert to self: %w", err)
		}
		defer func() {
			impersonationRollbackErr = windows.ImpersonateLoggedOnUser(prevToken)
		}()
	}
	return f()
}

// getCurrentToken returns the current thread token, or
// the process token if the thread doesn't have a token.
func getCurrentToken() (t syscall.Token, isProcessToken bool, err error) {
	thread, _ := windows.GetCurrentThread()
	// Need TOKEN_DUPLICATE and TOKEN_IMPERSONATE to use the token in ImpersonateLoggedOnUser.
	err = windows.OpenThreadToken(thread, syscall.TOKEN_QUERY|syscall.TOKEN_DUPLICATE|syscall.TOKEN_IMPERSONATE, true, &t)
	if errors.Is(err, windows.ERROR_NO_TOKEN) {
		// Not impersonating, use the process token.
		isProcessToken = true
		t, err = syscall.OpenCurrentProcessToken()
	}
	return t, isProcessToken, err
}

// lookupUserPrimaryGroup obtains the primary group SID for a user using this method:
// https://support.microsoft.com/en-us/help/297951/how-to-use-the-primarygroupid-attribute-to-find-the-primary-group-for
// The method follows this formula: domainRID + "-" + primaryGroupRID
func lookupUserPrimaryGroup(username, domain string) (string, error) {
	// get the domain RID
	sid, _, t, e := syscall.LookupSID("", domain)
	if e != nil {
		return "", e
	}
	if t != syscall.SidTypeDomain {
		return "", fmt.Errorf("lookupUserPrimaryGroup: should be domain account type, not %d", t)
	}
	domainRID, e := sid.String()
	if e != nil {
		return "", e
	}
	// If the user has joined a domain use the RID of the default primary group
	// called "Domain Users":
	// https://support.microsoft.com/en-us/help/243330/well-known-security-identifiers-in-windows-operating-systems
	// SID: S-1-5-21domain-513
	//
	// The correct way to obtain the primary group of a domain user is
	// probing the user primaryGroupID attribute in the server Active Directory:
	// https://learn.microsoft.com/en-us/windows/win32/adschema/a-primarygroupid
	//
	// Note that the primary group of domain users should not be modified
	// on Windows for performance reasons, even if it's possible to do that.
	// The .NET Developer's Guide to Directory Services Programming - Page 409
	// https://books.google.bg/books?id=kGApqjobEfsC&lpg=PA410&ots=p7oo-eOQL7&dq=primary%20group%20RID&hl=bg&pg=PA409#v=onepage&q&f=false
	joined, err := isDomainJoined()
	if err == nil && joined {
		return domainRID + "-513", nil
	}
	// For non-domain users call NetUserGetInfo() with level 4, which
	// in this case would not have any network overhead.
	// The primary group should not change from RID 513 here either
	// but the group will be called "None" instead:
	// https://www.adampalmer.me/iodigitalsec/2013/08/10/windows-null-session-enumeration/
	// "Group 'None' (RID: 513)"
	u, e := syscall.UTF16PtrFromString(username)
	if e != nil {
		return "", e
	}
	d, e := syscall.UTF16PtrFromString(domain)
	if e != nil {
		return "", e
	}
	var p *byte
	e = syscall.NetUserGetInfo(d, u, 4, &p)
	if e != nil {
		return "", e
	}
	defer syscall.NetApiBufferFree(p)
	i := (*windows.UserInfo4)(unsafe.Pointer(p))
	return fmt.Sprintf("%s-%d", domainRID, i.PrimaryGroupID), nil
}

func newUserFromSid(usid *syscall.SID) (*User, error) {
	username, domain, sidType, e := lookupUsernameAndDomain(usid)
	if e != nil {
		return nil, e
	}
	uid, e := usid.String()
	if e != nil {
		return nil, e
	}
	var gid string
	if sidType == syscall.SidTypeWellKnownGroup {
		// The SID does not contain a domain; this function's domain variable has
		// been populated with the SID's identifier authority. This happens with
		// special service user accounts such as "NT AUTHORITY\LocalSystem".
		// In this case, gid is the same as the user SID.
		gid = uid
	} else {
		gid, e = lookupUserPrimaryGroup(username, domain)
		if e != nil {
			return nil, e
		}
	}
	// If this user has logged in at least once their home path should be stored
	// in the registry under the specified SID. References:
	// https://social.technet.microsoft.com/wiki/contents/articles/13895.how-to-remove-a-corrupted-user-profile-from-the-registry.aspx
	// https://support.asperasoft.com/hc/en-us/articles/216127438-How-to-delete-Windows-user-profiles
	//
	// The registry is the most reliable way to find the home path as the user
	// might have decided to move it outside of the default location,
	// (e.g. C:\users). Reference:
	// https://answers.microsoft.com/en-us/windows/forum/windows_7-security/how-do-i-set-a-home-directory-outside-cusers-for-a/aed68262-1bf4-4a4d-93dc-7495193a440f
	dir, e := findHomeDirInRegistry(uid)
	if e != nil {
		// If the home path does not exist in the registry, the user might
		// have not logged in yet; fall back to using getProfilesDirectory().
		// Find the username based on a SID and append that to the result of
		// getProfilesDirectory(). The domain is not relevant here.
		dir, e = getProfilesDirectory()
		if e != nil {
			return nil, e
		}
		dir += `\` + username
	}
	return newUser(uid, gid, dir, username, domain)
}

func lookupUser(username string) (*User, error) {
	sid, _, t, e := syscall.LookupSID("", username)
	if e != nil {
		return nil, e
	}
	if !isValidUserAccountType(sid, t) {
		return nil, fmt.Errorf("user: should be user account type, not %d", t)
	}
	return newUserFromSid(sid)
}

func lookupUserId(uid string) (*User, error) {
	sid, e := syscall.StringToSid(uid)
	if e != nil {
		return nil, e
	}
	return newUserFromSid(sid)
}

func lookupGroup(groupname string) (*Group, error) {
	sid, err := lookupGroupName(groupname)
	if err != nil {
		return nil, err
	}
	return &Group{Name: groupname, Gid: sid}, nil
}

func lookupGroupId(gid string) (*Group, error) {
	sid, err := syscall.StringToSid(gid)
	if err != nil {
		return nil, err
	}
	groupname, _, t, err := sid.LookupAccount("")
	if err != nil {
		return nil, err
	}
	if !isValidGroupAccountType(t) {
		return nil, fmt.Errorf("lookupGroupId: should be group account type, not %d", t)
	}
	return &Group{Name: groupname, Gid: gid}, nil
}

func listGroups(user *User) ([]string, error) {
	var sids []string
	if u, err := Current(); err == nil && u.Uid == user.Uid {
		// It is faster and more reliable to get the groups
		// of the current user from the current process token.
		err := runAsProcessOwner(func() error {
			t, err := syscall.OpenCurrentProcessToken()
			if err != nil {
				return err
			}
			defer t.Close()
			groups, err := windows.GetTokenGroups(t)
			if err != nil {
				return err
			}
			for _, g := range groups.AllGroups() {
				sid, err := g.Sid.String()
				if err != nil {
					return err
				}
				sids = append(sids, sid)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	} else {
		sid, err := syscall.StringToSid(user.Uid)
		if err != nil {
			return nil, err
		}
		username, domain, _, err := lookupUsernameAndDomain(sid)
		if err != nil {
			return nil, err
		}
		sids, err = listGroupsForUsernameAndDomain(username, domain)
		if err != nil {
			return nil, err
		}
	}
	// Add the primary group of the user to the list if it is not already there.
	// This is done only to comply with the POSIX concept of a primary group.
	for _, sid := range sids {
		if sid == user.Gid {
			return sids, nil
		}
	}
	return append(sids, user.Gid), nil
}
