// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"runtime"
	"syscall"
	"unsafe"
)

const (
	SecurityAnonymous      = 0
	SecurityIdentification = 1
	SecurityImpersonation  = 2
	SecurityDelegation     = 3
)

//sys	ImpersonateSelf(impersonationlevel uint32) (err error) = advapi32.ImpersonateSelf
//sys	RevertToSelf() (err error) = advapi32.RevertToSelf
//sys	ImpersonateLoggedOnUser(token syscall.Token) (err error) = advapi32.ImpersonateLoggedOnUser
//sys	LogonUser(username *uint16, domain *uint16, password *uint16, logonType uint32, logonProvider uint32, token *syscall.Token) (err error) = advapi32.LogonUserW

const (
	TOKEN_ADJUST_PRIVILEGES = 0x0020
	SE_PRIVILEGE_ENABLED    = 0x00000002
)

type LUID struct {
	LowPart  uint32
	HighPart int32
}

type LUID_AND_ATTRIBUTES struct {
	Luid       LUID
	Attributes uint32
}

type TOKEN_PRIVILEGES struct {
	PrivilegeCount uint32
	Privileges     [1]LUID_AND_ATTRIBUTES
}

//sys	OpenThreadToken(h syscall.Handle, access uint32, openasself bool, token *syscall.Token) (err error) = advapi32.OpenThreadToken
//sys	LookupPrivilegeValue(systemname *uint16, name *uint16, luid *LUID) (err error) = advapi32.LookupPrivilegeValueW
//sys	adjustTokenPrivileges(token syscall.Token, disableAllPrivileges bool, newstate *TOKEN_PRIVILEGES, buflen uint32, prevstate *TOKEN_PRIVILEGES, returnlen *uint32) (ret uint32, err error) [true] = advapi32.AdjustTokenPrivileges

func AdjustTokenPrivileges(token syscall.Token, disableAllPrivileges bool, newstate *TOKEN_PRIVILEGES, buflen uint32, prevstate *TOKEN_PRIVILEGES, returnlen *uint32) error {
	ret, err := adjustTokenPrivileges(token, disableAllPrivileges, newstate, buflen, prevstate, returnlen)
	if ret == 0 {
		// AdjustTokenPrivileges call failed
		return err
	}
	// AdjustTokenPrivileges call succeeded
	if err == syscall.EINVAL {
		// GetLastError returned ERROR_SUCCESS
		return nil
	}
	return err
}

//sys DuplicateTokenEx(hExistingToken syscall.Token, dwDesiredAccess uint32, lpTokenAttributes *syscall.SecurityAttributes, impersonationLevel uint32, tokenType TokenType, phNewToken *syscall.Token) (err error) = advapi32.DuplicateTokenEx
//sys SetTokenInformation(tokenHandle syscall.Token, tokenInformationClass uint32, tokenInformation uintptr, tokenInformationLength uint32) (err error) = advapi32.SetTokenInformation

type SID_AND_ATTRIBUTES struct {
	Sid        *syscall.SID
	Attributes uint32
}

type TOKEN_MANDATORY_LABEL struct {
	Label SID_AND_ATTRIBUTES
}

func (tml *TOKEN_MANDATORY_LABEL) Size() uint32 {
	return uint32(unsafe.Sizeof(TOKEN_MANDATORY_LABEL{})) + syscall.GetLengthSid(tml.Label.Sid)
}

const SE_GROUP_INTEGRITY = 0x00000020

type TokenType uint32

const (
	TokenPrimary       TokenType = 1
	TokenImpersonation TokenType = 2
)

//sys	GetProfilesDirectory(dir *uint16, dirLen *uint32) (err error) = userenv.GetProfilesDirectoryW

const (
	LG_INCLUDE_INDIRECT  = 0x1
	MAX_PREFERRED_LENGTH = 0xFFFFFFFF
)

type LocalGroupUserInfo0 struct {
	Name *uint16
}

const (
	NERR_UserNotFound syscall.Errno = 2221
	NERR_UserExists   syscall.Errno = 2224
)

const (
	USER_PRIV_USER = 1
)

type UserInfo1 struct {
	Name        *uint16
	Password    *uint16
	PasswordAge uint32
	Priv        uint32
	HomeDir     *uint16
	Comment     *uint16
	Flags       uint32
	ScriptPath  *uint16
}

type UserInfo4 struct {
	Name            *uint16
	Password        *uint16
	PasswordAge     uint32
	Priv            uint32
	HomeDir         *uint16
	Comment         *uint16
	Flags           uint32
	ScriptPath      *uint16
	AuthFlags       uint32
	FullName        *uint16
	UsrComment      *uint16
	Parms           *uint16
	Workstations    *uint16
	LastLogon       uint32
	LastLogoff      uint32
	AcctExpires     uint32
	MaxStorage      uint32
	UnitsPerWeek    uint32
	LogonHours      *byte
	BadPwCount      uint32
	NumLogons       uint32
	LogonServer     *uint16
	CountryCode     uint32
	CodePage        uint32
	UserSid         *syscall.SID
	PrimaryGroupID  uint32
	Profile         *uint16
	HomeDirDrive    *uint16
	PasswordExpired uint32
}

//sys	NetUserAdd(serverName *uint16, level uint32, buf *byte, parmErr *uint32) (neterr error) = netapi32.NetUserAdd
//sys	NetUserDel(serverName *uint16, userName *uint16) (neterr error) = netapi32.NetUserDel
//sys	NetUserGetLocalGroups(serverName *uint16, userName *uint16, level uint32, flags uint32, buf **byte, prefMaxLen uint32, entriesRead *uint32, totalEntries *uint32) (neterr error) = netapi32.NetUserGetLocalGroups

// GetSystemDirectory retrieves the path to current location of the system
// directory, which is typically, though not always, `C:\Windows\System32`.
//
//go:linkname GetSystemDirectory
func GetSystemDirectory() string // Implemented in runtime package.

// GetUserName retrieves the user name of the current thread
// in the specified format.
func GetUserName(format uint32) (string, error) {
	n := uint32(50)
	for {
		b := make([]uint16, n)
		e := syscall.GetUserNameEx(format, &b[0], &n)
		if e == nil {
			return syscall.UTF16ToString(b[:n]), nil
		}
		if e != syscall.ERROR_MORE_DATA {
			return "", e
		}
		if n <= uint32(len(b)) {
			return "", e
		}
	}
}

// getTokenInfo retrieves a specified type of information about an access token.
func getTokenInfo(t syscall.Token, class uint32, initSize int) (unsafe.Pointer, error) {
	n := uint32(initSize)
	for {
		b := make([]byte, n)
		e := syscall.GetTokenInformation(t, class, &b[0], uint32(len(b)), &n)
		if e == nil {
			return unsafe.Pointer(&b[0]), nil
		}
		if e != syscall.ERROR_INSUFFICIENT_BUFFER {
			return nil, e
		}
		if n <= uint32(len(b)) {
			return nil, e
		}
	}
}

type TOKEN_GROUPS struct {
	GroupCount uint32
	Groups     [1]SID_AND_ATTRIBUTES
}

func (g *TOKEN_GROUPS) AllGroups() []SID_AND_ATTRIBUTES {
	return (*[(1 << 28) - 1]SID_AND_ATTRIBUTES)(unsafe.Pointer(&g.Groups[0]))[:g.GroupCount:g.GroupCount]
}

func GetTokenGroups(t syscall.Token) (*TOKEN_GROUPS, error) {
	i, e := getTokenInfo(t, syscall.TokenGroups, 50)
	if e != nil {
		return nil, e
	}
	return (*TOKEN_GROUPS)(i), nil
}

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-sid_identifier_authority
type SID_IDENTIFIER_AUTHORITY struct {
	Value [6]byte
}

const (
	SID_REVISION = 1
	// https://learn.microsoft.com/en-us/windows/win32/services/localsystem-account
	SECURITY_LOCAL_SYSTEM_RID = 18
	// https://learn.microsoft.com/en-us/windows/win32/services/localservice-account
	SECURITY_LOCAL_SERVICE_RID = 19
	// https://learn.microsoft.com/en-us/windows/win32/services/networkservice-account
	SECURITY_NETWORK_SERVICE_RID = 20
)

var SECURITY_NT_AUTHORITY = SID_IDENTIFIER_AUTHORITY{
	Value: [6]byte{0, 0, 0, 0, 0, 5},
}

//sys	IsValidSid(sid *syscall.SID) (valid bool) = advapi32.IsValidSid
//sys	getSidIdentifierAuthority(sid *syscall.SID) (idauth uintptr) = advapi32.GetSidIdentifierAuthority
//sys	getSidSubAuthority(sid *syscall.SID, subAuthorityIdx uint32) (subAuth uintptr) = advapi32.GetSidSubAuthority
//sys	getSidSubAuthorityCount(sid *syscall.SID) (count uintptr) = advapi32.GetSidSubAuthorityCount

// The following GetSid* functions are marked as //go:nocheckptr because checkptr
// instrumentation can't see that the pointer returned by the syscall is pointing
// into the sid's memory, which is normally allocated on the Go heap. Therefore,
// the checkptr instrumentation would incorrectly flag the pointer dereference
// as pointing to an invalid allocation.
// Also, use runtime.KeepAlive to ensure that the sid is not garbage collected
// before the GetSid* functions return, as the Go GC is not aware that the
// pointers returned by the syscall are pointing into the sid's memory.

//go:nocheckptr
func GetSidIdentifierAuthority(sid *syscall.SID) SID_IDENTIFIER_AUTHORITY {
	defer runtime.KeepAlive(sid)
	return *(*SID_IDENTIFIER_AUTHORITY)(unsafe.Pointer(getSidIdentifierAuthority(sid)))
}

//go:nocheckptr
func GetSidSubAuthority(sid *syscall.SID, subAuthorityIdx uint32) uint32 {
	defer runtime.KeepAlive(sid)
	return *(*uint32)(unsafe.Pointer(getSidSubAuthority(sid, subAuthorityIdx)))
}

//go:nocheckptr
func GetSidSubAuthorityCount(sid *syscall.SID) uint8 {
	defer runtime.KeepAlive(sid)
	return *(*uint8)(unsafe.Pointer(getSidSubAuthorityCount(sid)))
}
