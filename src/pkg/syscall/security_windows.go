// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"unsafe"
)

const (
	STANDARD_RIGHTS_REQUIRED = 0xf0000
	STANDARD_RIGHTS_READ     = 0x20000
	STANDARD_RIGHTS_WRITE    = 0x20000
	STANDARD_RIGHTS_EXECUTE  = 0x20000
	STANDARD_RIGHTS_ALL      = 0x1F0000
)

const (
	NameUnknown          = 0
	NameFullyQualifiedDN = 1
	NameSamCompatible    = 2
	NameDisplay          = 3
	NameUniqueId         = 6
	NameCanonical        = 7
	NameUserPrincipal    = 8
	NameCanonicalEx      = 9
	NameServicePrincipal = 10
	NameDnsDomain        = 12
)

// This function returns 1 byte BOOLEAN rather than the 4 byte BOOL.
// http://blogs.msdn.com/b/drnick/archive/2007/12/19/windows-and-upn-format-credentials.aspx
//sys	TranslateName(accName *uint16, accNameFormat uint32, desiredNameFormat uint32, translatedName *uint16, nSize *uint32) (err error) [failretval&0xff==0] = secur32.TranslateNameW
//sys	GetUserNameEx(nameFormat uint32, nameBuffre *uint16, nSize *uint32) (err error) [failretval&0xff==0] = secur32.GetUserNameExW

// TranslateAccountName converts a directory service
// object name from one format to another.
func TranslateAccountName(username string, from, to uint32, initSize int) (string, error) {
	u := StringToUTF16Ptr(username)
	b := make([]uint16, 50)
	n := uint32(len(b))
	e := TranslateName(u, from, to, &b[0], &n)
	if e != nil {
		if e != ERROR_INSUFFICIENT_BUFFER {
			return "", e
		}
		// make receive buffers of requested size and try again
		b = make([]uint16, n)
		e = TranslateName(u, from, to, &b[0], &n)
		if e != nil {
			return "", e
		}
	}
	return UTF16ToString(b), nil
}

type UserInfo10 struct {
	Name       *uint16
	Comment    *uint16
	UsrComment *uint16
	FullName   *uint16
}

//sys	NetUserGetInfo(serverName *uint16, userName *uint16, level uint32, buf **byte) (neterr error) = netapi32.NetUserGetInfo
//sys	NetApiBufferFree(buf *byte) (neterr error) = netapi32.NetApiBufferFree

const (
	// do not reorder
	SidTypeUser = 1 << iota
	SidTypeGroup
	SidTypeDomain
	SidTypeAlias
	SidTypeWellKnownGroup
	SidTypeDeletedAccount
	SidTypeInvalid
	SidTypeUnknown
	SidTypeComputer
	SidTypeLabel
)

//sys	LookupAccountSid(systemName *uint16, sid *SID, name *uint16, nameLen *uint32, refdDomainName *uint16, refdDomainNameLen *uint32, use *uint32) (err error) = advapi32.LookupAccountSidW
//sys	LookupAccountName(systemName *uint16, accountName *uint16, sid *SID, sidLen *uint32, refdDomainName *uint16, refdDomainNameLen *uint32, use *uint32) (err error) = advapi32.LookupAccountNameW
//sys	ConvertSidToStringSid(sid *SID, stringSid **uint16) (err error) = advapi32.ConvertSidToStringSidW
//sys	ConvertStringSidToSid(stringSid *uint16, sid **SID) (err error) = advapi32.ConvertStringSidToSidW
//sys	GetLengthSid(sid *SID) (len uint32) = advapi32.GetLengthSid
//sys	CopySid(destSidLen uint32, destSid *SID, srcSid *SID) (err error) = advapi32.CopySid

// The security identifier (SID) structure is a variable-length
// structure used to uniquely identify users or groups.
type SID struct{}

// StringToSid converts a string-format security identifier
// sid into a valid, functional sid.
func StringToSid(s string) (*SID, error) {
	var sid *SID
	e := ConvertStringSidToSid(StringToUTF16Ptr(s), &sid)
	if e != nil {
		return nil, e
	}
	defer LocalFree((Handle)(unsafe.Pointer(sid)))
	return sid.Copy()
}

// LookupSID retrieves a security identifier sid for the account
// and the name of the domain on which the account was found.
// System specify target computer to search.
func LookupSID(system, account string) (sid *SID, domain string, accType uint32, err error) {
	if len(account) == 0 {
		return nil, "", 0, EINVAL
	}
	acc := StringToUTF16Ptr(account)
	var sys *uint16
	if len(system) > 0 {
		sys = StringToUTF16Ptr(system)
	}
	db := make([]uint16, 50)
	dn := uint32(len(db))
	b := make([]byte, 50)
	n := uint32(len(b))
	sid = (*SID)(unsafe.Pointer(&b[0]))
	e := LookupAccountName(sys, acc, sid, &n, &db[0], &dn, &accType)
	if e != nil {
		if e != ERROR_INSUFFICIENT_BUFFER {
			return nil, "", 0, e
		}
		// make receive buffers of requested size and try again
		b = make([]byte, n)
		sid = (*SID)(unsafe.Pointer(&b[0]))
		db = make([]uint16, dn)
		e = LookupAccountName(sys, acc, sid, &n, &db[0], &dn, &accType)
		if e != nil {
			return nil, "", 0, e
		}
	}
	return sid, UTF16ToString(db), accType, nil
}

// String converts sid to a string format
// suitable for display, storage, or transmission.
func (sid *SID) String() (string, error) {
	var s *uint16
	e := ConvertSidToStringSid(sid, &s)
	if e != nil {
		return "", e
	}
	defer LocalFree((Handle)(unsafe.Pointer(s)))
	return UTF16ToString((*[256]uint16)(unsafe.Pointer(s))[:]), nil
}

// Len returns the length, in bytes, of a valid security identifier sid.
func (sid *SID) Len() int {
	return int(GetLengthSid(sid))
}

// Copy creates a duplicate of security identifier sid.
func (sid *SID) Copy() (*SID, error) {
	b := make([]byte, sid.Len())
	sid2 := (*SID)(unsafe.Pointer(&b[0]))
	e := CopySid(uint32(len(b)), sid2, sid)
	if e != nil {
		return nil, e
	}
	return sid2, nil
}

// LookupAccount retrieves the name of the account for this sid
// and the name of the first domain on which this sid is found.
// System specify target computer to search for.
func (sid *SID) LookupAccount(system string) (account, domain string, accType uint32, err error) {
	var sys *uint16
	if len(system) > 0 {
		sys = StringToUTF16Ptr(system)
	}
	b := make([]uint16, 50)
	n := uint32(len(b))
	db := make([]uint16, 50)
	dn := uint32(len(db))
	e := LookupAccountSid(sys, sid, &b[0], &n, &db[0], &dn, &accType)
	if e != nil {
		if e != ERROR_INSUFFICIENT_BUFFER {
			return "", "", 0, e
		}
		// make receive buffers of requested size and try again
		b = make([]uint16, n)
		db = make([]uint16, dn)
		e = LookupAccountSid(nil, sid, &b[0], &n, &db[0], &dn, &accType)
		if e != nil {
			return "", "", 0, e
		}
	}
	return UTF16ToString(b), UTF16ToString(db), accType, nil
}

const (
	// do not reorder
	TOKEN_ASSIGN_PRIMARY = 1 << iota
	TOKEN_DUPLICATE
	TOKEN_IMPERSONATE
	TOKEN_QUERY
	TOKEN_QUERY_SOURCE
	TOKEN_ADJUST_PRIVILEGES
	TOKEN_ADJUST_GROUPS
	TOKEN_ADJUST_DEFAULT

	TOKEN_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED |
		TOKEN_ASSIGN_PRIMARY |
		TOKEN_DUPLICATE |
		TOKEN_IMPERSONATE |
		TOKEN_QUERY |
		TOKEN_QUERY_SOURCE |
		TOKEN_ADJUST_PRIVILEGES |
		TOKEN_ADJUST_GROUPS |
		TOKEN_ADJUST_DEFAULT
	TOKEN_READ  = STANDARD_RIGHTS_READ | TOKEN_QUERY
	TOKEN_WRITE = STANDARD_RIGHTS_WRITE |
		TOKEN_ADJUST_PRIVILEGES |
		TOKEN_ADJUST_GROUPS |
		TOKEN_ADJUST_DEFAULT
	TOKEN_EXECUTE = STANDARD_RIGHTS_EXECUTE
)

const (
	// do not reorder
	TokenUser = 1 + iota
	TokenGroups
	TokenPrivileges
	TokenOwner
	TokenPrimaryGroup
	TokenDefaultDacl
	TokenSource
	TokenType
	TokenImpersonationLevel
	TokenStatistics
	TokenRestrictedSids
	TokenSessionId
	TokenGroupsAndPrivileges
	TokenSessionReference
	TokenSandBoxInert
	TokenAuditPolicy
	TokenOrigin
	TokenElevationType
	TokenLinkedToken
	TokenElevation
	TokenHasRestrictions
	TokenAccessInformation
	TokenVirtualizationAllowed
	TokenVirtualizationEnabled
	TokenIntegrityLevel
	TokenUIAccess
	TokenMandatoryPolicy
	TokenLogonSid
	MaxTokenInfoClass
)

type SIDAndAttributes struct {
	Sid        *SID
	Attributes uint32
}

type Tokenuser struct {
	User SIDAndAttributes
}

type Tokenprimarygroup struct {
	PrimaryGroup *SID
}

//sys	OpenProcessToken(h Handle, access uint32, token *Token) (err error) = advapi32.OpenProcessToken
//sys	GetTokenInformation(t Token, infoClass uint32, info *byte, infoLen uint32, returnedLen *uint32) (err error) = advapi32.GetTokenInformation
//sys	GetUserProfileDirectory(t Token, dir *uint16, dirLen *uint32) (err error) = userenv.GetUserProfileDirectoryW

// An access token contains the security information for a logon session.
// The system creates an access token when a user logs on, and every
// process executed on behalf of the user has a copy of the token.
// The token identifies the user, the user's groups, and the user's
// privileges. The system uses the token to control access to securable
// objects and to control the ability of the user to perform various
// system-related operations on the local computer.
type Token Handle

// OpenCurrentProcessToken opens the access token
// associated with current process.
func OpenCurrentProcessToken() (Token, error) {
	p, e := GetCurrentProcess()
	if e != nil {
		return 0, e
	}
	var t Token
	e = OpenProcessToken(p, TOKEN_QUERY, &t)
	if e != nil {
		return 0, e
	}
	return t, nil
}

// Close releases access to access token.
func (t Token) Close() error {
	return CloseHandle(Handle(t))
}

// getInfo retrieves a specified type of information about an access token.
func (t Token) getInfo(class uint32, initSize int) (unsafe.Pointer, error) {
	b := make([]byte, initSize)
	var n uint32
	e := GetTokenInformation(t, class, &b[0], uint32(len(b)), &n)
	if e != nil {
		if e != ERROR_INSUFFICIENT_BUFFER {
			return nil, e
		}
		// make receive buffers of requested size and try again
		b = make([]byte, n)
		e = GetTokenInformation(t, class, &b[0], uint32(len(b)), &n)
		if e != nil {
			return nil, e
		}
	}
	return unsafe.Pointer(&b[0]), nil
}

// GetTokenUser retrieves access token t user account information.
func (t Token) GetTokenUser() (*Tokenuser, error) {
	i, e := t.getInfo(TokenUser, 50)
	if e != nil {
		return nil, e
	}
	return (*Tokenuser)(i), nil
}

// GetTokenPrimaryGroup retrieves access token t primary group information.
// A pointer to a SID structure representing a group that will become
// the primary group of any objects created by a process using this access token.
func (t Token) GetTokenPrimaryGroup() (*Tokenprimarygroup, error) {
	i, e := t.getInfo(TokenPrimaryGroup, 50)
	if e != nil {
		return nil, e
	}
	return (*Tokenprimarygroup)(i), nil
}

// GetUserProfileDirectory retrieves path to the
// root directory of the access token t user's profile.
func (t Token) GetUserProfileDirectory() (string, error) {
	b := make([]uint16, 100)
	n := uint32(len(b))
	e := GetUserProfileDirectory(t, &b[0], &n)
	if e != nil {
		if e != ERROR_INSUFFICIENT_BUFFER {
			return "", e
		}
		// make receive buffers of requested size and try again
		b = make([]uint16, n)
		e = GetUserProfileDirectory(t, &b[0], &n)
		if e != nil {
			return "", e
		}
	}
	return UTF16ToString(b), nil
}
