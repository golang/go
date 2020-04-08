// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"syscall"
	"unsafe"
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
	DOMAIN_ALIAS_RID_MONITORING_USERS               = 0x22e
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
//sys	createWellKnownSid(sidType WELL_KNOWN_SID_TYPE, domainSid *SID, sid *SID, sizeSid *uint32) (err error) = advapi32.CreateWellKnownSid
//sys	isWellKnownSid(sid *SID, sidType WELL_KNOWN_SID_TYPE) (isWellKnown bool) = advapi32.IsWellKnownSid
//sys	FreeSid(sid *SID) (err error) [failretval!=0] = advapi32.FreeSid
//sys	EqualSid(sid1 *SID, sid2 *SID) (isEqual bool) = advapi32.EqualSid
//sys	getSidIdentifierAuthority(sid *SID) (authority *SidIdentifierAuthority) = advapi32.GetSidIdentifierAuthority
//sys	getSidSubAuthorityCount(sid *SID) (count *uint8) = advapi32.GetSidSubAuthorityCount
//sys	getSidSubAuthority(sid *SID, index uint32) (subAuthority *uint32) = advapi32.GetSidSubAuthority
//sys	isValidSid(sid *SID) (isValid bool) = advapi32.IsValidSid

// The security identifier (SID) structure is a variable-length
// structure used to uniquely identify users or groups.
type SID struct{}

// StringToSid converts a string-format security identifier
// SID into a valid, functional SID.
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

// LookupSID retrieves a security identifier SID for the account
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

// String converts SID to a string format suitable for display, storage, or transmission.
func (sid *SID) String() string {
	var s *uint16
	e := ConvertSidToStringSid(sid, &s)
	if e != nil {
		return ""
	}
	defer LocalFree((Handle)(unsafe.Pointer(s)))
	return UTF16ToString((*[256]uint16)(unsafe.Pointer(s))[:])
}

// Len returns the length, in bytes, of a valid security identifier SID.
func (sid *SID) Len() int {
	return int(GetLengthSid(sid))
}

// Copy creates a duplicate of security identifier SID.
func (sid *SID) Copy() (*SID, error) {
	b := make([]byte, sid.Len())
	sid2 := (*SID)(unsafe.Pointer(&b[0]))
	e := CopySid(uint32(len(b)), sid2, sid)
	if e != nil {
		return nil, e
	}
	return sid2, nil
}

// IdentifierAuthority returns the identifier authority of the SID.
func (sid *SID) IdentifierAuthority() SidIdentifierAuthority {
	return *getSidIdentifierAuthority(sid)
}

// SubAuthorityCount returns the number of sub-authorities in the SID.
func (sid *SID) SubAuthorityCount() uint8 {
	return *getSidSubAuthorityCount(sid)
}

// SubAuthority returns the sub-authority of the SID as specified by
// the index, which must be less than sid.SubAuthorityCount().
func (sid *SID) SubAuthority(idx uint32) uint32 {
	if idx >= uint32(sid.SubAuthorityCount()) {
		panic("sub-authority index out of range")
	}
	return *getSidSubAuthority(sid, idx)
}

// IsValid returns whether the SID has a valid revision and length.
func (sid *SID) IsValid() bool {
	return isValidSid(sid)
}

// Equals compares two SIDs for equality.
func (sid *SID) Equals(sid2 *SID) bool {
	return EqualSid(sid, sid2)
}

// IsWellKnown determines whether the SID matches the well-known sidType.
func (sid *SID) IsWellKnown(sidType WELL_KNOWN_SID_TYPE) bool {
	return isWellKnownSid(sid, sidType)
}

// LookupAccount retrieves the name of the account for this SID
// and the name of the first domain on which this SID is found.
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

// Various types of pre-specified SIDs that can be synthesized and compared at runtime.
type WELL_KNOWN_SID_TYPE uint32

const (
	WinNullSid                                    = 0
	WinWorldSid                                   = 1
	WinLocalSid                                   = 2
	WinCreatorOwnerSid                            = 3
	WinCreatorGroupSid                            = 4
	WinCreatorOwnerServerSid                      = 5
	WinCreatorGroupServerSid                      = 6
	WinNtAuthoritySid                             = 7
	WinDialupSid                                  = 8
	WinNetworkSid                                 = 9
	WinBatchSid                                   = 10
	WinInteractiveSid                             = 11
	WinServiceSid                                 = 12
	WinAnonymousSid                               = 13
	WinProxySid                                   = 14
	WinEnterpriseControllersSid                   = 15
	WinSelfSid                                    = 16
	WinAuthenticatedUserSid                       = 17
	WinRestrictedCodeSid                          = 18
	WinTerminalServerSid                          = 19
	WinRemoteLogonIdSid                           = 20
	WinLogonIdsSid                                = 21
	WinLocalSystemSid                             = 22
	WinLocalServiceSid                            = 23
	WinNetworkServiceSid                          = 24
	WinBuiltinDomainSid                           = 25
	WinBuiltinAdministratorsSid                   = 26
	WinBuiltinUsersSid                            = 27
	WinBuiltinGuestsSid                           = 28
	WinBuiltinPowerUsersSid                       = 29
	WinBuiltinAccountOperatorsSid                 = 30
	WinBuiltinSystemOperatorsSid                  = 31
	WinBuiltinPrintOperatorsSid                   = 32
	WinBuiltinBackupOperatorsSid                  = 33
	WinBuiltinReplicatorSid                       = 34
	WinBuiltinPreWindows2000CompatibleAccessSid   = 35
	WinBuiltinRemoteDesktopUsersSid               = 36
	WinBuiltinNetworkConfigurationOperatorsSid    = 37
	WinAccountAdministratorSid                    = 38
	WinAccountGuestSid                            = 39
	WinAccountKrbtgtSid                           = 40
	WinAccountDomainAdminsSid                     = 41
	WinAccountDomainUsersSid                      = 42
	WinAccountDomainGuestsSid                     = 43
	WinAccountComputersSid                        = 44
	WinAccountControllersSid                      = 45
	WinAccountCertAdminsSid                       = 46
	WinAccountSchemaAdminsSid                     = 47
	WinAccountEnterpriseAdminsSid                 = 48
	WinAccountPolicyAdminsSid                     = 49
	WinAccountRasAndIasServersSid                 = 50
	WinNTLMAuthenticationSid                      = 51
	WinDigestAuthenticationSid                    = 52
	WinSChannelAuthenticationSid                  = 53
	WinThisOrganizationSid                        = 54
	WinOtherOrganizationSid                       = 55
	WinBuiltinIncomingForestTrustBuildersSid      = 56
	WinBuiltinPerfMonitoringUsersSid              = 57
	WinBuiltinPerfLoggingUsersSid                 = 58
	WinBuiltinAuthorizationAccessSid              = 59
	WinBuiltinTerminalServerLicenseServersSid     = 60
	WinBuiltinDCOMUsersSid                        = 61
	WinBuiltinIUsersSid                           = 62
	WinIUserSid                                   = 63
	WinBuiltinCryptoOperatorsSid                  = 64
	WinUntrustedLabelSid                          = 65
	WinLowLabelSid                                = 66
	WinMediumLabelSid                             = 67
	WinHighLabelSid                               = 68
	WinSystemLabelSid                             = 69
	WinWriteRestrictedCodeSid                     = 70
	WinCreatorOwnerRightsSid                      = 71
	WinCacheablePrincipalsGroupSid                = 72
	WinNonCacheablePrincipalsGroupSid             = 73
	WinEnterpriseReadonlyControllersSid           = 74
	WinAccountReadonlyControllersSid              = 75
	WinBuiltinEventLogReadersGroup                = 76
	WinNewEnterpriseReadonlyControllersSid        = 77
	WinBuiltinCertSvcDComAccessGroup              = 78
	WinMediumPlusLabelSid                         = 79
	WinLocalLogonSid                              = 80
	WinConsoleLogonSid                            = 81
	WinThisOrganizationCertificateSid             = 82
	WinApplicationPackageAuthoritySid             = 83
	WinBuiltinAnyPackageSid                       = 84
	WinCapabilityInternetClientSid                = 85
	WinCapabilityInternetClientServerSid          = 86
	WinCapabilityPrivateNetworkClientServerSid    = 87
	WinCapabilityPicturesLibrarySid               = 88
	WinCapabilityVideosLibrarySid                 = 89
	WinCapabilityMusicLibrarySid                  = 90
	WinCapabilityDocumentsLibrarySid              = 91
	WinCapabilitySharedUserCertificatesSid        = 92
	WinCapabilityEnterpriseAuthenticationSid      = 93
	WinCapabilityRemovableStorageSid              = 94
	WinBuiltinRDSRemoteAccessServersSid           = 95
	WinBuiltinRDSEndpointServersSid               = 96
	WinBuiltinRDSManagementServersSid             = 97
	WinUserModeDriversSid                         = 98
	WinBuiltinHyperVAdminsSid                     = 99
	WinAccountCloneableControllersSid             = 100
	WinBuiltinAccessControlAssistanceOperatorsSid = 101
	WinBuiltinRemoteManagementUsersSid            = 102
	WinAuthenticationAuthorityAssertedSid         = 103
	WinAuthenticationServiceAssertedSid           = 104
	WinLocalAccountSid                            = 105
	WinLocalAccountAndAdministratorSid            = 106
	WinAccountProtectedUsersSid                   = 107
	WinCapabilityAppointmentsSid                  = 108
	WinCapabilityContactsSid                      = 109
	WinAccountDefaultSystemManagedSid             = 110
	WinBuiltinDefaultSystemManagedGroupSid        = 111
	WinBuiltinStorageReplicaAdminsSid             = 112
	WinAccountKeyAdminsSid                        = 113
	WinAccountEnterpriseKeyAdminsSid              = 114
	WinAuthenticationKeyTrustSid                  = 115
	WinAuthenticationKeyPropertyMFASid            = 116
	WinAuthenticationKeyPropertyAttestationSid    = 117
	WinAuthenticationFreshKeyAuthSid              = 118
	WinBuiltinDeviceOwnersSid                     = 119
)

// Creates a SID for a well-known predefined alias, generally using the constants of the form
// Win*Sid, for the local machine.
func CreateWellKnownSid(sidType WELL_KNOWN_SID_TYPE) (*SID, error) {
	return CreateWellKnownDomainSid(sidType, nil)
}

// Creates a SID for a well-known predefined alias, generally using the constants of the form
// Win*Sid, for the domain specified by the domainSid parameter.
func CreateWellKnownDomainSid(sidType WELL_KNOWN_SID_TYPE, domainSid *SID) (*SID, error) {
	n := uint32(50)
	for {
		b := make([]byte, n)
		sid := (*SID)(unsafe.Pointer(&b[0]))
		err := createWellKnownSid(sidType, domainSid, sid, &n)
		if err == nil {
			return sid, nil
		}
		if err != ERROR_INSUFFICIENT_BUFFER {
			return nil, err
		}
		if n <= uint32(len(b)) {
			return nil, err
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

// Group attributes inside of Tokengroups.Groups[i].Attributes
const (
	SE_GROUP_MANDATORY          = 0x00000001
	SE_GROUP_ENABLED_BY_DEFAULT = 0x00000002
	SE_GROUP_ENABLED            = 0x00000004
	SE_GROUP_OWNER              = 0x00000008
	SE_GROUP_USE_FOR_DENY_ONLY  = 0x00000010
	SE_GROUP_INTEGRITY          = 0x00000020
	SE_GROUP_INTEGRITY_ENABLED  = 0x00000040
	SE_GROUP_LOGON_ID           = 0xC0000000
	SE_GROUP_RESOURCE           = 0x20000000
	SE_GROUP_VALID_ATTRIBUTES   = SE_GROUP_MANDATORY | SE_GROUP_ENABLED_BY_DEFAULT | SE_GROUP_ENABLED | SE_GROUP_OWNER | SE_GROUP_USE_FOR_DENY_ONLY | SE_GROUP_LOGON_ID | SE_GROUP_RESOURCE | SE_GROUP_INTEGRITY | SE_GROUP_INTEGRITY_ENABLED
)

// Privilege attributes
const (
	SE_PRIVILEGE_ENABLED_BY_DEFAULT = 0x00000001
	SE_PRIVILEGE_ENABLED            = 0x00000002
	SE_PRIVILEGE_REMOVED            = 0x00000004
	SE_PRIVILEGE_USED_FOR_ACCESS    = 0x80000000
	SE_PRIVILEGE_VALID_ATTRIBUTES   = SE_PRIVILEGE_ENABLED_BY_DEFAULT | SE_PRIVILEGE_ENABLED | SE_PRIVILEGE_REMOVED | SE_PRIVILEGE_USED_FOR_ACCESS
)

// Token types
const (
	TokenPrimary       = 1
	TokenImpersonation = 2
)

// Impersonation levels
const (
	SecurityAnonymous      = 0
	SecurityIdentification = 1
	SecurityImpersonation  = 2
	SecurityDelegation     = 3
)

type LUID struct {
	LowPart  uint32
	HighPart int32
}

type LUIDAndAttributes struct {
	Luid       LUID
	Attributes uint32
}

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
	Groups     [1]SIDAndAttributes // Use AllGroups() for iterating.
}

// AllGroups returns a slice that can be used to iterate over the groups in g.
func (g *Tokengroups) AllGroups() []SIDAndAttributes {
	return (*[(1 << 28) - 1]SIDAndAttributes)(unsafe.Pointer(&g.Groups[0]))[:g.GroupCount:g.GroupCount]
}

type Tokenprivileges struct {
	PrivilegeCount uint32
	Privileges     [1]LUIDAndAttributes // Use AllPrivileges() for iterating.
}

// AllPrivileges returns a slice that can be used to iterate over the privileges in p.
func (p *Tokenprivileges) AllPrivileges() []LUIDAndAttributes {
	return (*[(1 << 27) - 1]LUIDAndAttributes)(unsafe.Pointer(&p.Privileges[0]))[:p.PrivilegeCount:p.PrivilegeCount]
}

type Tokenmandatorylabel struct {
	Label SIDAndAttributes
}

func (tml *Tokenmandatorylabel) Size() uint32 {
	return uint32(unsafe.Sizeof(Tokenmandatorylabel{})) + GetLengthSid(tml.Label.Sid)
}

// Authorization Functions
//sys	checkTokenMembership(tokenHandle Token, sidToCheck *SID, isMember *int32) (err error) = advapi32.CheckTokenMembership
//sys	OpenProcessToken(process Handle, access uint32, token *Token) (err error) = advapi32.OpenProcessToken
//sys	OpenThreadToken(thread Handle, access uint32, openAsSelf bool, token *Token) (err error) = advapi32.OpenThreadToken
//sys	ImpersonateSelf(impersonationlevel uint32) (err error) = advapi32.ImpersonateSelf
//sys	RevertToSelf() (err error) = advapi32.RevertToSelf
//sys	SetThreadToken(thread *Handle, token Token) (err error) = advapi32.SetThreadToken
//sys	LookupPrivilegeValue(systemname *uint16, name *uint16, luid *LUID) (err error) = advapi32.LookupPrivilegeValueW
//sys	AdjustTokenPrivileges(token Token, disableAllPrivileges bool, newstate *Tokenprivileges, buflen uint32, prevstate *Tokenprivileges, returnlen *uint32) (err error) = advapi32.AdjustTokenPrivileges
//sys	AdjustTokenGroups(token Token, resetToDefault bool, newstate *Tokengroups, buflen uint32, prevstate *Tokengroups, returnlen *uint32) (err error) = advapi32.AdjustTokenGroups
//sys	GetTokenInformation(token Token, infoClass uint32, info *byte, infoLen uint32, returnedLen *uint32) (err error) = advapi32.GetTokenInformation
//sys	SetTokenInformation(token Token, infoClass uint32, info *byte, infoLen uint32) (err error) = advapi32.SetTokenInformation
//sys	DuplicateTokenEx(existingToken Token, desiredAccess uint32, tokenAttributes *SecurityAttributes, impersonationLevel uint32, tokenType uint32, newToken *Token) (err error) = advapi32.DuplicateTokenEx
//sys	GetUserProfileDirectory(t Token, dir *uint16, dirLen *uint32) (err error) = userenv.GetUserProfileDirectoryW
//sys	getSystemDirectory(dir *uint16, dirLen uint32) (len uint32, err error) = kernel32.GetSystemDirectoryW
//sys	getWindowsDirectory(dir *uint16, dirLen uint32) (len uint32, err error) = kernel32.GetWindowsDirectoryW
//sys	getSystemWindowsDirectory(dir *uint16, dirLen uint32) (len uint32, err error) = kernel32.GetSystemWindowsDirectoryW

// An access token contains the security information for a logon session.
// The system creates an access token when a user logs on, and every
// process executed on behalf of the user has a copy of the token.
// The token identifies the user, the user's groups, and the user's
// privileges. The system uses the token to control access to securable
// objects and to control the ability of the user to perform various
// system-related operations on the local computer.
type Token Handle

// OpenCurrentProcessToken opens an access token associated with current
// process with TOKEN_QUERY access. It is a real token that needs to be closed.
//
// Deprecated: Explicitly call OpenProcessToken(CurrentProcess(), ...)
// with the desired access instead, or use GetCurrentProcessToken for a
// TOKEN_QUERY token.
func OpenCurrentProcessToken() (Token, error) {
	var token Token
	err := OpenProcessToken(CurrentProcess(), TOKEN_QUERY, &token)
	return token, err
}

// GetCurrentProcessToken returns the access token associated with
// the current process. It is a pseudo token that does not need
// to be closed.
func GetCurrentProcessToken() Token {
	return Token(^uintptr(4 - 1))
}

// GetCurrentThreadToken return the access token associated with
// the current thread. It is a pseudo token that does not need
// to be closed.
func GetCurrentThreadToken() Token {
	return Token(^uintptr(5 - 1))
}

// GetCurrentThreadEffectiveToken returns the effective access token
// associated with the current thread. It is a pseudo token that does
// not need to be closed.
func GetCurrentThreadEffectiveToken() Token {
	return Token(^uintptr(6 - 1))
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

// IsElevated returns whether the current token is elevated from a UAC perspective.
func (token Token) IsElevated() bool {
	var isElevated uint32
	var outLen uint32
	err := GetTokenInformation(token, TokenElevation, (*byte)(unsafe.Pointer(&isElevated)), uint32(unsafe.Sizeof(isElevated)), &outLen)
	if err != nil {
		return false
	}
	return outLen == uint32(unsafe.Sizeof(isElevated)) && isElevated != 0
}

// GetLinkedToken returns the linked token, which may be an elevated UAC token.
func (token Token) GetLinkedToken() (Token, error) {
	var linkedToken Token
	var outLen uint32
	err := GetTokenInformation(token, TokenLinkedToken, (*byte)(unsafe.Pointer(&linkedToken)), uint32(unsafe.Sizeof(linkedToken)), &outLen)
	if err != nil {
		return Token(0), err
	}
	return linkedToken, nil
}

// GetSystemDirectory retrieves the path to current location of the system
// directory, which is typically, though not always, `C:\Windows\System32`.
func GetSystemDirectory() (string, error) {
	n := uint32(MAX_PATH)
	for {
		b := make([]uint16, n)
		l, e := getSystemDirectory(&b[0], n)
		if e != nil {
			return "", e
		}
		if l <= n {
			return UTF16ToString(b[:l]), nil
		}
		n = l
	}
}

// GetWindowsDirectory retrieves the path to current location of the Windows
// directory, which is typically, though not always, `C:\Windows`. This may
// be a private user directory in the case that the application is running
// under a terminal server.
func GetWindowsDirectory() (string, error) {
	n := uint32(MAX_PATH)
	for {
		b := make([]uint16, n)
		l, e := getWindowsDirectory(&b[0], n)
		if e != nil {
			return "", e
		}
		if l <= n {
			return UTF16ToString(b[:l]), nil
		}
		n = l
	}
}

// GetSystemWindowsDirectory retrieves the path to current location of the
// Windows directory, which is typically, though not always, `C:\Windows`.
func GetSystemWindowsDirectory() (string, error) {
	n := uint32(MAX_PATH)
	for {
		b := make([]uint16, n)
		l, e := getSystemWindowsDirectory(&b[0], n)
		if e != nil {
			return "", e
		}
		if l <= n {
			return UTF16ToString(b[:l]), nil
		}
		n = l
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

const (
	WTS_CONSOLE_CONNECT        = 0x1
	WTS_CONSOLE_DISCONNECT     = 0x2
	WTS_REMOTE_CONNECT         = 0x3
	WTS_REMOTE_DISCONNECT      = 0x4
	WTS_SESSION_LOGON          = 0x5
	WTS_SESSION_LOGOFF         = 0x6
	WTS_SESSION_LOCK           = 0x7
	WTS_SESSION_UNLOCK         = 0x8
	WTS_SESSION_REMOTE_CONTROL = 0x9
	WTS_SESSION_CREATE         = 0xa
	WTS_SESSION_TERMINATE      = 0xb
)

const (
	WTSActive       = 0
	WTSConnected    = 1
	WTSConnectQuery = 2
	WTSShadow       = 3
	WTSDisconnected = 4
	WTSIdle         = 5
	WTSListen       = 6
	WTSReset        = 7
	WTSDown         = 8
	WTSInit         = 9
)

type WTSSESSION_NOTIFICATION struct {
	Size      uint32
	SessionID uint32
}

type WTS_SESSION_INFO struct {
	SessionID         uint32
	WindowStationName *uint16
	State             uint32
}

//sys WTSQueryUserToken(session uint32, token *Token) (err error) = wtsapi32.WTSQueryUserToken
//sys WTSEnumerateSessions(handle Handle, reserved uint32, version uint32, sessions **WTS_SESSION_INFO, count *uint32) (err error) = wtsapi32.WTSEnumerateSessionsW
//sys WTSFreeMemory(ptr uintptr) = wtsapi32.WTSFreeMemory

type ACL struct {
	aclRevision byte
	sbz1        byte
	aclSize     uint16
	aceCount    uint16
	sbz2        uint16
}

type SECURITY_DESCRIPTOR struct {
	revision byte
	sbz1     byte
	control  SECURITY_DESCRIPTOR_CONTROL
	owner    *SID
	group    *SID
	sacl     *ACL
	dacl     *ACL
}

type SecurityAttributes struct {
	Length             uint32
	SecurityDescriptor *SECURITY_DESCRIPTOR
	InheritHandle      uint32
}

type SE_OBJECT_TYPE uint32

// Constants for type SE_OBJECT_TYPE
const (
	SE_UNKNOWN_OBJECT_TYPE     = 0
	SE_FILE_OBJECT             = 1
	SE_SERVICE                 = 2
	SE_PRINTER                 = 3
	SE_REGISTRY_KEY            = 4
	SE_LMSHARE                 = 5
	SE_KERNEL_OBJECT           = 6
	SE_WINDOW_OBJECT           = 7
	SE_DS_OBJECT               = 8
	SE_DS_OBJECT_ALL           = 9
	SE_PROVIDER_DEFINED_OBJECT = 10
	SE_WMIGUID_OBJECT          = 11
	SE_REGISTRY_WOW64_32KEY    = 12
	SE_REGISTRY_WOW64_64KEY    = 13
)

type SECURITY_INFORMATION uint32

// Constants for type SECURITY_INFORMATION
const (
	OWNER_SECURITY_INFORMATION            = 0x00000001
	GROUP_SECURITY_INFORMATION            = 0x00000002
	DACL_SECURITY_INFORMATION             = 0x00000004
	SACL_SECURITY_INFORMATION             = 0x00000008
	LABEL_SECURITY_INFORMATION            = 0x00000010
	ATTRIBUTE_SECURITY_INFORMATION        = 0x00000020
	SCOPE_SECURITY_INFORMATION            = 0x00000040
	BACKUP_SECURITY_INFORMATION           = 0x00010000
	PROTECTED_DACL_SECURITY_INFORMATION   = 0x80000000
	PROTECTED_SACL_SECURITY_INFORMATION   = 0x40000000
	UNPROTECTED_DACL_SECURITY_INFORMATION = 0x20000000
	UNPROTECTED_SACL_SECURITY_INFORMATION = 0x10000000
)

type SECURITY_DESCRIPTOR_CONTROL uint16

// Constants for type SECURITY_DESCRIPTOR_CONTROL
const (
	SE_OWNER_DEFAULTED       = 0x0001
	SE_GROUP_DEFAULTED       = 0x0002
	SE_DACL_PRESENT          = 0x0004
	SE_DACL_DEFAULTED        = 0x0008
	SE_SACL_PRESENT          = 0x0010
	SE_SACL_DEFAULTED        = 0x0020
	SE_DACL_AUTO_INHERIT_REQ = 0x0100
	SE_SACL_AUTO_INHERIT_REQ = 0x0200
	SE_DACL_AUTO_INHERITED   = 0x0400
	SE_SACL_AUTO_INHERITED   = 0x0800
	SE_DACL_PROTECTED        = 0x1000
	SE_SACL_PROTECTED        = 0x2000
	SE_RM_CONTROL_VALID      = 0x4000
	SE_SELF_RELATIVE         = 0x8000
)

type ACCESS_MASK uint32

// Constants for type ACCESS_MASK
const (
	DELETE                   = 0x00010000
	READ_CONTROL             = 0x00020000
	WRITE_DAC                = 0x00040000
	WRITE_OWNER              = 0x00080000
	SYNCHRONIZE              = 0x00100000
	STANDARD_RIGHTS_REQUIRED = 0x000F0000
	STANDARD_RIGHTS_READ     = READ_CONTROL
	STANDARD_RIGHTS_WRITE    = READ_CONTROL
	STANDARD_RIGHTS_EXECUTE  = READ_CONTROL
	STANDARD_RIGHTS_ALL      = 0x001F0000
	SPECIFIC_RIGHTS_ALL      = 0x0000FFFF
	ACCESS_SYSTEM_SECURITY   = 0x01000000
	MAXIMUM_ALLOWED          = 0x02000000
	GENERIC_READ             = 0x80000000
	GENERIC_WRITE            = 0x40000000
	GENERIC_EXECUTE          = 0x20000000
	GENERIC_ALL              = 0x10000000
)

type ACCESS_MODE uint32

// Constants for type ACCESS_MODE
const (
	NOT_USED_ACCESS   = 0
	GRANT_ACCESS      = 1
	SET_ACCESS        = 2
	DENY_ACCESS       = 3
	REVOKE_ACCESS     = 4
	SET_AUDIT_SUCCESS = 5
	SET_AUDIT_FAILURE = 6
)

// Constants for AceFlags and Inheritance fields
const (
	NO_INHERITANCE                     = 0x0
	SUB_OBJECTS_ONLY_INHERIT           = 0x1
	SUB_CONTAINERS_ONLY_INHERIT        = 0x2
	SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
	INHERIT_NO_PROPAGATE               = 0x4
	INHERIT_ONLY                       = 0x8
	INHERITED_ACCESS_ENTRY             = 0x10
	INHERITED_PARENT                   = 0x10000000
	INHERITED_GRANDPARENT              = 0x20000000
	OBJECT_INHERIT_ACE                 = 0x1
	CONTAINER_INHERIT_ACE              = 0x2
	NO_PROPAGATE_INHERIT_ACE           = 0x4
	INHERIT_ONLY_ACE                   = 0x8
	INHERITED_ACE                      = 0x10
	VALID_INHERIT_FLAGS                = 0x1F
)

type MULTIPLE_TRUSTEE_OPERATION uint32

// Constants for MULTIPLE_TRUSTEE_OPERATION
const (
	NO_MULTIPLE_TRUSTEE    = 0
	TRUSTEE_IS_IMPERSONATE = 1
)

type TRUSTEE_FORM uint32

// Constants for TRUSTEE_FORM
const (
	TRUSTEE_IS_SID              = 0
	TRUSTEE_IS_NAME             = 1
	TRUSTEE_BAD_FORM            = 2
	TRUSTEE_IS_OBJECTS_AND_SID  = 3
	TRUSTEE_IS_OBJECTS_AND_NAME = 4
)

type TRUSTEE_TYPE uint32

// Constants for TRUSTEE_TYPE
const (
	TRUSTEE_IS_UNKNOWN          = 0
	TRUSTEE_IS_USER             = 1
	TRUSTEE_IS_GROUP            = 2
	TRUSTEE_IS_DOMAIN           = 3
	TRUSTEE_IS_ALIAS            = 4
	TRUSTEE_IS_WELL_KNOWN_GROUP = 5
	TRUSTEE_IS_DELETED          = 6
	TRUSTEE_IS_INVALID          = 7
	TRUSTEE_IS_COMPUTER         = 8
)

// Constants for ObjectsPresent field
const (
	ACE_OBJECT_TYPE_PRESENT           = 0x1
	ACE_INHERITED_OBJECT_TYPE_PRESENT = 0x2
)

type EXPLICIT_ACCESS struct {
	AccessPermissions ACCESS_MASK
	AccessMode        ACCESS_MODE
	Inheritance       uint32
	Trustee           TRUSTEE
}

// This type is the union inside of TRUSTEE and must be created using one of the TrusteeValueFrom* functions.
type TrusteeValue uintptr

func TrusteeValueFromString(str string) TrusteeValue {
	return TrusteeValue(unsafe.Pointer(StringToUTF16Ptr(str)))
}
func TrusteeValueFromSID(sid *SID) TrusteeValue {
	return TrusteeValue(unsafe.Pointer(sid))
}
func TrusteeValueFromObjectsAndSid(objectsAndSid *OBJECTS_AND_SID) TrusteeValue {
	return TrusteeValue(unsafe.Pointer(objectsAndSid))
}
func TrusteeValueFromObjectsAndName(objectsAndName *OBJECTS_AND_NAME) TrusteeValue {
	return TrusteeValue(unsafe.Pointer(objectsAndName))
}

type TRUSTEE struct {
	MultipleTrustee          *TRUSTEE
	MultipleTrusteeOperation MULTIPLE_TRUSTEE_OPERATION
	TrusteeForm              TRUSTEE_FORM
	TrusteeType              TRUSTEE_TYPE
	TrusteeValue             TrusteeValue
}

type OBJECTS_AND_SID struct {
	ObjectsPresent          uint32
	ObjectTypeGuid          GUID
	InheritedObjectTypeGuid GUID
	Sid                     *SID
}

type OBJECTS_AND_NAME struct {
	ObjectsPresent          uint32
	ObjectType              SE_OBJECT_TYPE
	ObjectTypeName          *uint16
	InheritedObjectTypeName *uint16
	Name                    *uint16
}

//sys	getSecurityInfo(handle Handle, objectType SE_OBJECT_TYPE, securityInformation SECURITY_INFORMATION, owner **SID, group **SID, dacl **ACL, sacl **ACL, sd **SECURITY_DESCRIPTOR) (ret error) = advapi32.GetSecurityInfo
//sys	SetSecurityInfo(handle Handle, objectType SE_OBJECT_TYPE, securityInformation SECURITY_INFORMATION, owner *SID, group *SID, dacl *ACL, sacl *ACL) = advapi32.SetSecurityInfo
//sys	getNamedSecurityInfo(objectName string, objectType SE_OBJECT_TYPE, securityInformation SECURITY_INFORMATION, owner **SID, group **SID, dacl **ACL, sacl **ACL, sd **SECURITY_DESCRIPTOR) (ret error) = advapi32.GetNamedSecurityInfoW
//sys	SetNamedSecurityInfo(objectName string, objectType SE_OBJECT_TYPE, securityInformation SECURITY_INFORMATION, owner *SID, group *SID, dacl *ACL, sacl *ACL) (ret error) = advapi32.SetNamedSecurityInfoW

//sys	buildSecurityDescriptor(owner *TRUSTEE, group *TRUSTEE, countAccessEntries uint32, accessEntries *EXPLICIT_ACCESS, countAuditEntries uint32, auditEntries *EXPLICIT_ACCESS, oldSecurityDescriptor *SECURITY_DESCRIPTOR, sizeNewSecurityDescriptor *uint32, newSecurityDescriptor **SECURITY_DESCRIPTOR) (ret error) = advapi32.BuildSecurityDescriptorW
//sys	initializeSecurityDescriptor(absoluteSD *SECURITY_DESCRIPTOR, revision uint32) (err error) = advapi32.InitializeSecurityDescriptor

//sys	getSecurityDescriptorControl(sd *SECURITY_DESCRIPTOR, control *SECURITY_DESCRIPTOR_CONTROL, revision *uint32) (err error) = advapi32.GetSecurityDescriptorControl
//sys	getSecurityDescriptorDacl(sd *SECURITY_DESCRIPTOR, daclPresent *bool, dacl **ACL, daclDefaulted *bool) (err error) = advapi32.GetSecurityDescriptorDacl
//sys	getSecurityDescriptorSacl(sd *SECURITY_DESCRIPTOR, saclPresent *bool, sacl **ACL, saclDefaulted *bool) (err error) = advapi32.GetSecurityDescriptorSacl
//sys	getSecurityDescriptorOwner(sd *SECURITY_DESCRIPTOR, owner **SID, ownerDefaulted *bool) (err error) = advapi32.GetSecurityDescriptorOwner
//sys	getSecurityDescriptorGroup(sd *SECURITY_DESCRIPTOR, group **SID, groupDefaulted *bool) (err error) = advapi32.GetSecurityDescriptorGroup
//sys	getSecurityDescriptorLength(sd *SECURITY_DESCRIPTOR) (len uint32) = advapi32.GetSecurityDescriptorLength
//sys	getSecurityDescriptorRMControl(sd *SECURITY_DESCRIPTOR, rmControl *uint8) (ret error) [failretval!=0] = advapi32.GetSecurityDescriptorRMControl
//sys	isValidSecurityDescriptor(sd *SECURITY_DESCRIPTOR) (isValid bool) = advapi32.IsValidSecurityDescriptor

//sys	setSecurityDescriptorControl(sd *SECURITY_DESCRIPTOR, controlBitsOfInterest SECURITY_DESCRIPTOR_CONTROL, controlBitsToSet SECURITY_DESCRIPTOR_CONTROL) (err error) = advapi32.SetSecurityDescriptorControl
//sys	setSecurityDescriptorDacl(sd *SECURITY_DESCRIPTOR, daclPresent bool, dacl *ACL, daclDefaulted bool) (err error) = advapi32.SetSecurityDescriptorDacl
//sys	setSecurityDescriptorSacl(sd *SECURITY_DESCRIPTOR, saclPresent bool, sacl *ACL, saclDefaulted bool) (err error) = advapi32.SetSecurityDescriptorSacl
//sys	setSecurityDescriptorOwner(sd *SECURITY_DESCRIPTOR, owner *SID, ownerDefaulted bool) (err error) = advapi32.SetSecurityDescriptorOwner
//sys	setSecurityDescriptorGroup(sd *SECURITY_DESCRIPTOR, group *SID, groupDefaulted bool) (err error) = advapi32.SetSecurityDescriptorGroup
//sys	setSecurityDescriptorRMControl(sd *SECURITY_DESCRIPTOR, rmControl *uint8) = advapi32.SetSecurityDescriptorRMControl

//sys	convertStringSecurityDescriptorToSecurityDescriptor(str string, revision uint32, sd **SECURITY_DESCRIPTOR, size *uint32) (err error) = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
//sys	convertSecurityDescriptorToStringSecurityDescriptor(sd *SECURITY_DESCRIPTOR, revision uint32, securityInformation SECURITY_INFORMATION, str **uint16, strLen *uint32) (err error) = advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW

//sys	makeAbsoluteSD(selfRelativeSD *SECURITY_DESCRIPTOR, absoluteSD *SECURITY_DESCRIPTOR, absoluteSDSize *uint32, dacl *ACL, daclSize *uint32, sacl *ACL, saclSize *uint32, owner *SID, ownerSize *uint32, group *SID, groupSize *uint32) (err error) = advapi32.MakeAbsoluteSD
//sys	makeSelfRelativeSD(absoluteSD *SECURITY_DESCRIPTOR, selfRelativeSD *SECURITY_DESCRIPTOR, selfRelativeSDSize *uint32) (err error) = advapi32.MakeSelfRelativeSD

//sys	setEntriesInAcl(countExplicitEntries uint32, explicitEntries *EXPLICIT_ACCESS, oldACL *ACL, newACL **ACL) (ret error) = advapi32.SetEntriesInAclW

// Control returns the security descriptor control bits.
func (sd *SECURITY_DESCRIPTOR) Control() (control SECURITY_DESCRIPTOR_CONTROL, revision uint32, err error) {
	err = getSecurityDescriptorControl(sd, &control, &revision)
	return
}

// SetControl sets the security descriptor control bits.
func (sd *SECURITY_DESCRIPTOR) SetControl(controlBitsOfInterest SECURITY_DESCRIPTOR_CONTROL, controlBitsToSet SECURITY_DESCRIPTOR_CONTROL) error {
	return setSecurityDescriptorControl(sd, controlBitsOfInterest, controlBitsToSet)
}

// RMControl returns the security descriptor resource manager control bits.
func (sd *SECURITY_DESCRIPTOR) RMControl() (control uint8, err error) {
	err = getSecurityDescriptorRMControl(sd, &control)
	return
}

// SetRMControl sets the security descriptor resource manager control bits.
func (sd *SECURITY_DESCRIPTOR) SetRMControl(rmControl uint8) {
	setSecurityDescriptorRMControl(sd, &rmControl)
}

// DACL returns the security descriptor DACL and whether it was defaulted. The dacl return value may be nil
// if a DACL exists but is an "empty DACL", meaning fully permissive. If the DACL does not exist, err returns
// ERROR_OBJECT_NOT_FOUND.
func (sd *SECURITY_DESCRIPTOR) DACL() (dacl *ACL, defaulted bool, err error) {
	var present bool
	err = getSecurityDescriptorDacl(sd, &present, &dacl, &defaulted)
	if !present {
		err = ERROR_OBJECT_NOT_FOUND
	}
	return
}

// SetDACL sets the absolute security descriptor DACL.
func (absoluteSD *SECURITY_DESCRIPTOR) SetDACL(dacl *ACL, present, defaulted bool) error {
	return setSecurityDescriptorDacl(absoluteSD, present, dacl, defaulted)
}

// SACL returns the security descriptor SACL and whether it was defaulted. The sacl return value may be nil
// if a SACL exists but is an "empty SACL", meaning fully permissive. If the SACL does not exist, err returns
// ERROR_OBJECT_NOT_FOUND.
func (sd *SECURITY_DESCRIPTOR) SACL() (sacl *ACL, defaulted bool, err error) {
	var present bool
	err = getSecurityDescriptorSacl(sd, &present, &sacl, &defaulted)
	if !present {
		err = ERROR_OBJECT_NOT_FOUND
	}
	return
}

// SetSACL sets the absolute security descriptor SACL.
func (absoluteSD *SECURITY_DESCRIPTOR) SetSACL(sacl *ACL, present, defaulted bool) error {
	return setSecurityDescriptorSacl(absoluteSD, present, sacl, defaulted)
}

// Owner returns the security descriptor owner and whether it was defaulted.
func (sd *SECURITY_DESCRIPTOR) Owner() (owner *SID, defaulted bool, err error) {
	err = getSecurityDescriptorOwner(sd, &owner, &defaulted)
	return
}

// SetOwner sets the absolute security descriptor owner.
func (absoluteSD *SECURITY_DESCRIPTOR) SetOwner(owner *SID, defaulted bool) error {
	return setSecurityDescriptorOwner(absoluteSD, owner, defaulted)
}

// Group returns the security descriptor group and whether it was defaulted.
func (sd *SECURITY_DESCRIPTOR) Group() (group *SID, defaulted bool, err error) {
	err = getSecurityDescriptorGroup(sd, &group, &defaulted)
	return
}

// SetGroup sets the absolute security descriptor owner.
func (absoluteSD *SECURITY_DESCRIPTOR) SetGroup(group *SID, defaulted bool) error {
	return setSecurityDescriptorGroup(absoluteSD, group, defaulted)
}

// Length returns the length of the security descriptor.
func (sd *SECURITY_DESCRIPTOR) Length() uint32 {
	return getSecurityDescriptorLength(sd)
}

// IsValid returns whether the security descriptor is valid.
func (sd *SECURITY_DESCRIPTOR) IsValid() bool {
	return isValidSecurityDescriptor(sd)
}

// String returns the SDDL form of the security descriptor, with a function signature that can be
// used with %v formatting directives.
func (sd *SECURITY_DESCRIPTOR) String() string {
	var sddl *uint16
	err := convertSecurityDescriptorToStringSecurityDescriptor(sd, 1, 0xff, &sddl, nil)
	if err != nil {
		return ""
	}
	defer LocalFree(Handle(unsafe.Pointer(sddl)))
	return UTF16ToString((*[(1 << 30) - 1]uint16)(unsafe.Pointer(sddl))[:])
}

// ToAbsolute converts a self-relative security descriptor into an absolute one.
func (selfRelativeSD *SECURITY_DESCRIPTOR) ToAbsolute() (absoluteSD *SECURITY_DESCRIPTOR, err error) {
	control, _, err := selfRelativeSD.Control()
	if err != nil {
		return
	}
	if control&SE_SELF_RELATIVE == 0 {
		err = ERROR_INVALID_PARAMETER
		return
	}
	var absoluteSDSize, daclSize, saclSize, ownerSize, groupSize uint32
	err = makeAbsoluteSD(selfRelativeSD, nil, &absoluteSDSize,
		nil, &daclSize, nil, &saclSize, nil, &ownerSize, nil, &groupSize)
	switch err {
	case ERROR_INSUFFICIENT_BUFFER:
	case nil:
		// makeAbsoluteSD is expected to fail, but it succeeds.
		return nil, ERROR_INTERNAL_ERROR
	default:
		return nil, err
	}
	if absoluteSDSize > 0 {
		absoluteSD = (*SECURITY_DESCRIPTOR)(unsafe.Pointer(&make([]byte, absoluteSDSize)[0]))
	}
	var (
		dacl  *ACL
		sacl  *ACL
		owner *SID
		group *SID
	)
	if daclSize > 0 {
		dacl = (*ACL)(unsafe.Pointer(&make([]byte, daclSize)[0]))
	}
	if saclSize > 0 {
		sacl = (*ACL)(unsafe.Pointer(&make([]byte, saclSize)[0]))
	}
	if ownerSize > 0 {
		owner = (*SID)(unsafe.Pointer(&make([]byte, ownerSize)[0]))
	}
	if groupSize > 0 {
		group = (*SID)(unsafe.Pointer(&make([]byte, groupSize)[0]))
	}
	err = makeAbsoluteSD(selfRelativeSD, absoluteSD, &absoluteSDSize,
		dacl, &daclSize, sacl, &saclSize, owner, &ownerSize, group, &groupSize)
	return
}

// ToSelfRelative converts an absolute security descriptor into a self-relative one.
func (absoluteSD *SECURITY_DESCRIPTOR) ToSelfRelative() (selfRelativeSD *SECURITY_DESCRIPTOR, err error) {
	control, _, err := absoluteSD.Control()
	if err != nil {
		return
	}
	if control&SE_SELF_RELATIVE != 0 {
		err = ERROR_INVALID_PARAMETER
		return
	}
	var selfRelativeSDSize uint32
	err = makeSelfRelativeSD(absoluteSD, nil, &selfRelativeSDSize)
	switch err {
	case ERROR_INSUFFICIENT_BUFFER:
	case nil:
		// makeSelfRelativeSD is expected to fail, but it succeeds.
		return nil, ERROR_INTERNAL_ERROR
	default:
		return nil, err
	}
	if selfRelativeSDSize > 0 {
		selfRelativeSD = (*SECURITY_DESCRIPTOR)(unsafe.Pointer(&make([]byte, selfRelativeSDSize)[0]))
	}
	err = makeSelfRelativeSD(absoluteSD, selfRelativeSD, &selfRelativeSDSize)
	return
}

func (selfRelativeSD *SECURITY_DESCRIPTOR) copySelfRelativeSecurityDescriptor() *SECURITY_DESCRIPTOR {
	sdBytes := make([]byte, selfRelativeSD.Length())
	copy(sdBytes, (*[(1 << 31) - 1]byte)(unsafe.Pointer(selfRelativeSD))[:len(sdBytes)])
	return (*SECURITY_DESCRIPTOR)(unsafe.Pointer(&sdBytes[0]))
}

// SecurityDescriptorFromString converts an SDDL string describing a security descriptor into a
// self-relative security descriptor object allocated on the Go heap.
func SecurityDescriptorFromString(sddl string) (sd *SECURITY_DESCRIPTOR, err error) {
	var winHeapSD *SECURITY_DESCRIPTOR
	err = convertStringSecurityDescriptorToSecurityDescriptor(sddl, 1, &winHeapSD, nil)
	if err != nil {
		return
	}
	defer LocalFree(Handle(unsafe.Pointer(winHeapSD)))
	return winHeapSD.copySelfRelativeSecurityDescriptor(), nil
}

// GetSecurityInfo queries the security information for a given handle and returns the self-relative security
// descriptor result on the Go heap.
func GetSecurityInfo(handle Handle, objectType SE_OBJECT_TYPE, securityInformation SECURITY_INFORMATION) (sd *SECURITY_DESCRIPTOR, err error) {
	var winHeapSD *SECURITY_DESCRIPTOR
	err = getSecurityInfo(handle, objectType, securityInformation, nil, nil, nil, nil, &winHeapSD)
	if err != nil {
		return
	}
	defer LocalFree(Handle(unsafe.Pointer(winHeapSD)))
	return winHeapSD.copySelfRelativeSecurityDescriptor(), nil
}

// GetNamedSecurityInfo queries the security information for a given named object and returns the self-relative security
// descriptor result on the Go heap.
func GetNamedSecurityInfo(objectName string, objectType SE_OBJECT_TYPE, securityInformation SECURITY_INFORMATION) (sd *SECURITY_DESCRIPTOR, err error) {
	var winHeapSD *SECURITY_DESCRIPTOR
	err = getNamedSecurityInfo(objectName, objectType, securityInformation, nil, nil, nil, nil, &winHeapSD)
	if err != nil {
		return
	}
	defer LocalFree(Handle(unsafe.Pointer(winHeapSD)))
	return winHeapSD.copySelfRelativeSecurityDescriptor(), nil
}

// BuildSecurityDescriptor makes a new security descriptor using the input trustees, explicit access lists, and
// prior security descriptor to be merged, any of which can be nil, returning the self-relative security descriptor
// result on the Go heap.
func BuildSecurityDescriptor(owner *TRUSTEE, group *TRUSTEE, accessEntries []EXPLICIT_ACCESS, auditEntries []EXPLICIT_ACCESS, mergedSecurityDescriptor *SECURITY_DESCRIPTOR) (sd *SECURITY_DESCRIPTOR, err error) {
	var winHeapSD *SECURITY_DESCRIPTOR
	var winHeapSDSize uint32
	var firstAccessEntry *EXPLICIT_ACCESS
	if len(accessEntries) > 0 {
		firstAccessEntry = &accessEntries[0]
	}
	var firstAuditEntry *EXPLICIT_ACCESS
	if len(auditEntries) > 0 {
		firstAuditEntry = &auditEntries[0]
	}
	err = buildSecurityDescriptor(owner, group, uint32(len(accessEntries)), firstAccessEntry, uint32(len(auditEntries)), firstAuditEntry, mergedSecurityDescriptor, &winHeapSDSize, &winHeapSD)
	if err != nil {
		return
	}
	defer LocalFree(Handle(unsafe.Pointer(winHeapSD)))
	return winHeapSD.copySelfRelativeSecurityDescriptor(), nil
}

// NewSecurityDescriptor creates and initializes a new absolute security descriptor.
func NewSecurityDescriptor() (absoluteSD *SECURITY_DESCRIPTOR, err error) {
	absoluteSD = &SECURITY_DESCRIPTOR{}
	err = initializeSecurityDescriptor(absoluteSD, 1)
	return
}

// ACLFromEntries returns a new ACL on the Go heap containing a list of explicit entries as well as those of another ACL.
// Both explicitEntries and mergedACL are optional and can be nil.
func ACLFromEntries(explicitEntries []EXPLICIT_ACCESS, mergedACL *ACL) (acl *ACL, err error) {
	var firstExplicitEntry *EXPLICIT_ACCESS
	if len(explicitEntries) > 0 {
		firstExplicitEntry = &explicitEntries[0]
	}
	var winHeapACL *ACL
	err = setEntriesInAcl(uint32(len(explicitEntries)), firstExplicitEntry, mergedACL, &winHeapACL)
	if err != nil {
		return
	}
	defer LocalFree(Handle(unsafe.Pointer(winHeapACL)))
	aclBytes := make([]byte, winHeapACL.aclSize)
	copy(aclBytes, (*[(1 << 31) - 1]byte)(unsafe.Pointer(winHeapACL))[:len(aclBytes)])
	return (*ACL)(unsafe.Pointer(&aclBytes[0])), nil
}
