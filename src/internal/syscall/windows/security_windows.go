// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
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

//sys	NetUserGetLocalGroups(serverName *uint16, userName *uint16, level uint32, flags uint32, buf **byte, prefMaxLen uint32, entriesRead *uint32, totalEntries *uint32) (neterr error) = netapi32.NetUserGetLocalGroups
