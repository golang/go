// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"syscall"
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
	u, e := UTF16PtrFromString(username)
	if e != nil {
		return "", e
	}
	n := uint32(50)
	for {
		b := make([]uint16, n)
		e = TranslateName(u, from, to, &b[0], &n)
		if e == nil {
			return UTF16ToString(b[:n]), nil
		}
		if e != ERROR_INSUFFICIENT_BUFFER {
			return "", e
		}
		if n <= uint32(len(b)) {
			return "", e
		}
	}
}

const (
	// do not reorder
	NetSetupUnknownStatus = iota
	NetSetupUnjoined
	NetSetupWorkgroupName
	NetSetupDomainName
)

type UserInfo10 struct {
	Name       *uint16
	Comment    *uint16
	UsrComment *uint16
	FullName   *uint16
}

//sys	NetUserGetInfo(serverName *uint16, userName *uint16, level uint32, buf **byte) (neterr error) = netapi32.NetUserGetInfo
//sys	NetGetJoinInformation(server *uint16, name **uint16, bufType *uint32) (neterr error) = netapi32.NetGetJoinInformation
//sys	NetApiBufferFree(buf *byte) (neterr error) = netapi32.NetApiBufferFree

const (
	// do not reorder
	SidTypeUser = 1 + iota
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

type SidIdentifierAuthority struct {
	Value [6]byte
}

var (
	SECURITY_NULL_SID_AUTHORITY        = SidIdentifierAuthority{[6]byte{0, 0, 0, 0, 0, 0}}
	SECURITY_WORLD_SID_AUTHORITY       = SidIdentifierAuthority{[6]byte{0, 0, 0, 0, 0, 1}}
	SECURITY_LOCAL_SID_AUTHORITY       = SidIdentifierAuthority{[6]byte{0, 0, 0, 0, 0, 2}}
	SECURITY_CREATOR_SID_AUTHORITY     = SidIdentifierAuthority{[6]byte{0, 0, 0, 0, 0, 3}}
	SECURITY_NON_UNIQUE_AUTHORITY      = SidIdentifierAuthority{[6]byte{0, 0, 0, 0, 0, 4}}
	SECURITY_NT_AUTHORITY              = SidIdentifierAuthority{[6]byte{0, 0, 0, 0, 0, 5}}
	SECURITY_MANDATORY_LABEL_AUTHORITY = SidIdentifierAuthority{[6]byte{0, 0, 0, 0, 0, 16}}
)

const (
	SECURITY_NULL_RID                   = 0
	SECURITY_WORLD_RID                  = 0
	SECURITY_LOCAL_RID                  = 0
	SECURITY_CREATOR_OWNER_RID          = 0
	SECURITY_CREATOR_GROUP_RID          = 1
	SECURITY_DIALUP_RID                 = 1
	SECURITY_NETWORK_RID                = 2
	SECURITY_BATCH_RID                  = 3
	SECURITY_INTERACTIVE_RID            = 4
	SECURITY_LOGON_IDS_RID              = 5
	SECURITY_SERVICE_RID                = 6
	SECURITY_LOCAL_SYSTEM_RID           = 18
	SECURITY_BUILTIN_DOMAIN_RID         = 32
	SECURITY_PRINCIPAL_SELF_RID         = 10
	SECURITY_CREATOR_OWNER_SERVER_RID   = 0x2
	SECURITY_CREATOR_GROUP_SERVER_RID   = 0x3
	SECURITY_LOGON_IDS_RID_COUNT        = 0x3
	SECURITY_ANONYMOUS_LOGON_RID        = 0x7
	SECURITY_PROXY_RID                  = 0x8
	SECURITY_ENTERPRISE_CONTROLLERS_RID = 0x9
	SECURITY_SERVER_LOGON_RID           = SECURITY_ENTERPRISE_CONTROLLERS_RID
	SECURITY_AUTHENTICATED_USER_RID     = 0xb
	SECURITY_RESTRICTED_CODE_RID        = 0xc
	SECURITY_NT_NON_UNIQUE_RID          = 0x15
)

// Predefined domain-relative RIDs for local groups.
// See https://msdn.microsoft.com/en-us/library/windows/desktop/aa379649(v=vs.85).aspx
const (
	DOMAIN_ALIAS_RID_ADMINS                         = 0x220
	DOMAIN_ALIAS_RID_USERS                          = 0x221
	DOMAIN_ALIAS_RID_GUESTS                         = 0x222
	DOMAIN_ALIAS_RID_POWER_USERS                    = 0x223
	DOMAIN_ALIAS_RID_ACCOUNT_OPS                    = 0x224
	DOMAIN_ALIAS_RID_SYSTEM_OPS                     = 0x225
	DOMAIN_ALIAS_RID_PRINT_OPS                      = 0x226
	DOMAIN_ALIAS_RID_BACKUP_OPS                     = 0x227
	DOMAIN_ALIAS_RID_REPLICATOR                     = 0x228
	DOMAIN_ALIAS_RID_RAS_SERVERS                    = 0x229
	DOMAIN_ALIAS_RID_PREW2KCOMPACCESS               = 0x22a
	DOMAIN_ALIAS_RID_REMOTE_DESKTOP_USERS           = 0x22b
	DOMAIN_ALIAS_RID_NETWORK_CONFIGURATION_OPS      = 0x22c
	DOMAIN_ALIAS_RID_INCOMING_FOREST_TRUST_BUILDERS = 0x22d
	DOMAIN_ALIAS_RID_MONITORING_USERS               = 0X22e
	DOMAIN_ALIAS_RID_LOGGING_USERS                  = 0x22f
	DOMAIN_ALIAS_RID_AUTHORIZATIONACCESS            = 0x230
	DOMAIN_ALIAS_RID_TS_LICENSE_SERVERS             = 0x231
	DOMAIN_ALIAS_RID_DCOM_USERS                     = 0x232
	DOMAIN_ALIAS_RID_IUSERS                         = 0x238
	DOMAIN_ALIAS_RID_CRYPTO_OPERATORS               = 0x239
	DOMAIN_ALIAS_RID_CACHEABLE_PRINCIPALS_GROUP     = 0x23b
	DOMAIN_ALIAS_RID_NON_CACHEABLE_PRINCIPALS_GROUP = 0x23c
	DOMAIN_ALIAS_RID_EVENT_LOG_READERS_GROUP        = 0x23d
	DOMAIN_ALIAS_RID_CERTSVC_DCOM_ACCESS_GROUP      = 0x23e
)

//sys	LookupAccountSid(systemName *uint16, sid *SID, name *uint16, nameLen *uint32, refdDomainName *uint16, refdDomainNameLen *uint32, use *uint32) (err error) = advapi32.LookupAccountSidW
//sys	LookupAccountName(systemName *uint16, accountName *uint16, sid *SID, sidLen *uint32, refdDomainName *uint16, refdDomainNameLen *uint32, use *uint32) (err error) = advapi32.LookupAccountNameW
//sys	ConvertSidToStringSid(sid *SID, stringSid **uint16) (err error) = advapi32.ConvertSidToStringSidW
//sys	ConvertStringSidToSid(stringSid *uint16, sid **SID) (err error) = advapi32.ConvertStringSidToSidW
//sys	GetLengthSid(sid *SID) (len uint32) = advapi32.GetLengthSid
//sys	CopySid(destSidLen uint32, destSid *SID, srcSid *SID) (err error) = advapi32.CopySid
//sys	AllocateAndInitializeSid(identAuth *SidIdentifierAuthority, subAuth byte, subAuth0 uint32, subAuth1 uint32, subAuth2 uint32, subAuth3 uint32, subAuth4 uint32, subAuth5 uint32, subAuth6 uint32, subAuth7 uint32, sid **SID) (err error) = advapi32.AllocateAndInitializeSid
//sys	FreeSid(sid *SID) (err error) [failretval!=0] = advapi32.FreeSid
//sys	EqualSid(sid1 *SID, sid2 *SID) (isEqual bool) = advapi32.EqualSid

// The security identifier (SID) structure is a variable-length
// structure used to uniquely identify users or groups.
type SID struct{}

// StringToSid converts a string-format security identifier
// sid into a valid, functional sid.
func StringToSid(s string) (*SID, error) {
	var sid *SID
	p, e := UTF16PtrFromString(s)
	if e != nil {
		return nil, e
	}
	e = ConvertStringSidToSid(p, &sid)
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
		return nil, "", 0, syscall.EINVAL
	}
	acc, e := UTF16PtrFromString(account)
	if e != nil {
		return nil, "", 0, e
	}
	var sys *uint16
	if len(system) > 0 {
		sys, e = UTF16PtrFromString(system)
		if e != nil {
			return nil, "", 0, e
		}
	}
	n := uint32(50)
	dn := uint32(50)
	for {
		b := make([]byte, n)
		db := make([]uint16, dn)
		sid = (*SID)(unsafe.Pointer(&b[0]))
		e = LookupAccountName(sys, acc, sid, &n, &db[0], &dn, &accType)
		if e == nil {
			return sid, UTF16ToString(db), accType, nil
		}
		if e != ERROR_INSUFFICIENT_BUFFER {
			return nil, "", 0, e
		}
		if n <= uint32(len(b)) {
			return nil, "", 0, e
		}
	}
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
		sys, err = UTF16PtrFromString(system)
		if err != nil {
			return "", "", 0, err
		}
	}
	n := uint32(50)
	dn := uint32(50)
	for {
		b := make([]uint16, n)
		db := make([]uint16, dn)
		e := LookupAccountSid(sys, sid, &b[0], &n, &db[0], &dn, &accType)
		if e == nil {
			return UTF16ToString(b), UTF16ToString(db), accType, nil
		}
		if e != ERROR_INSUFFICIENT_BUFFER {
			return "", "", 0, e
		}
		if n <= uint32(len(b)) {
			return "", "", 0, e
		}
	}
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
	TOKEN_ADJUST_SESSIONID

	TOKEN_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED |
		TOKEN_ASSIGN_PRIMARY |
		TOKEN_DUPLICATE |
		TOKEN_IMPERSONATE |
		TOKEN_QUERY |
		TOKEN_QUERY_SOURCE |
		TOKEN_ADJUST_PRIVILEGES |
		TOKEN_ADJUST_GROUPS |
		TOKEN_ADJUST_DEFAULT |
		TOKEN_ADJUST_SESSIONID
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

type Tokengroups struct {
	GroupCount uint32
	Groups     [1]SIDAndAttributes
}

// Authorization Functions
//sys checkTokenMembership(tokenHandle Token, sidToCheck *SID, isMember *int32) (err error) = advapi32.CheckTokenMembership
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
	n := uint32(initSize)
	for {
		b := make([]byte, n)
		e := GetTokenInformation(t, class, &b[0], uint32(len(b)), &n)
		if e == nil {
			return unsafe.Pointer(&b[0]), nil
		}
		if e != ERROR_INSUFFICIENT_BUFFER {
			return nil, e
		}
		if n <= uint32(len(b)) {
			return nil, e
		}
	}
}

// GetTokenUser retrieves access token t user account information.
func (t Token) GetTokenUser() (*Tokenuser, error) {
	i, e := t.getInfo(TokenUser, 50)
	if e != nil {
		return nil, e
	}
	return (*Tokenuser)(i), nil
}

// GetTokenGroups retrieves group accounts associated with access token t.
func (t Token) GetTokenGroups() (*Tokengroups, error) {
	i, e := t.getInfo(TokenGroups, 50)
	if e != nil {
		return nil, e
	}
	return (*Tokengroups)(i), nil
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
	n := uint32(100)
	for {
		b := make([]uint16, n)
		e := GetUserProfileDirectory(t, &b[0], &n)
		if e == nil {
			return UTF16ToString(b), nil
		}
		if e != ERROR_INSUFFICIENT_BUFFER {
			return "", e
		}
		if n <= uint32(len(b)) {
			return "", e
		}
	}
}

// IsMember reports whether the access token t is a member of the provided SID.
func (t Token) IsMember(sid *SID) (bool, error) {
	var b int32
	if e := checkTokenMembership(t, sid, &b); e != nil {
		return false, e
	}
	return b != 0, nil
}
