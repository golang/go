// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"net"
	"syscall"
	"unsafe"
)

const (
	// Invented values to support what package os expects.
	O_RDONLY   = 0x00000
	O_WRONLY   = 0x00001
	O_RDWR     = 0x00002
	O_CREAT    = 0x00040
	O_EXCL     = 0x00080
	O_NOCTTY   = 0x00100
	O_TRUNC    = 0x00200
	O_NONBLOCK = 0x00800
	O_APPEND   = 0x00400
	O_SYNC     = 0x01000
	O_ASYNC    = 0x02000
	O_CLOEXEC  = 0x80000
)

const (
	// More invented values for signals
	SIGHUP  = Signal(0x1)
	SIGINT  = Signal(0x2)
	SIGQUIT = Signal(0x3)
	SIGILL  = Signal(0x4)
	SIGTRAP = Signal(0x5)
	SIGABRT = Signal(0x6)
	SIGBUS  = Signal(0x7)
	SIGFPE  = Signal(0x8)
	SIGKILL = Signal(0x9)
	SIGSEGV = Signal(0xb)
	SIGPIPE = Signal(0xd)
	SIGALRM = Signal(0xe)
	SIGTERM = Signal(0xf)
)

var signals = [...]string{
	1:  "hangup",
	2:  "interrupt",
	3:  "quit",
	4:  "illegal instruction",
	5:  "trace/breakpoint trap",
	6:  "aborted",
	7:  "bus error",
	8:  "floating point exception",
	9:  "killed",
	10: "user defined signal 1",
	11: "segmentation fault",
	12: "user defined signal 2",
	13: "broken pipe",
	14: "alarm clock",
	15: "terminated",
}

const (
	FILE_LIST_DIRECTORY   = 0x00000001
	FILE_APPEND_DATA      = 0x00000004
	FILE_WRITE_ATTRIBUTES = 0x00000100

	FILE_SHARE_READ   = 0x00000001
	FILE_SHARE_WRITE  = 0x00000002
	FILE_SHARE_DELETE = 0x00000004

	FILE_ATTRIBUTE_READONLY              = 0x00000001
	FILE_ATTRIBUTE_HIDDEN                = 0x00000002
	FILE_ATTRIBUTE_SYSTEM                = 0x00000004
	FILE_ATTRIBUTE_DIRECTORY             = 0x00000010
	FILE_ATTRIBUTE_ARCHIVE               = 0x00000020
	FILE_ATTRIBUTE_DEVICE                = 0x00000040
	FILE_ATTRIBUTE_NORMAL                = 0x00000080
	FILE_ATTRIBUTE_TEMPORARY             = 0x00000100
	FILE_ATTRIBUTE_SPARSE_FILE           = 0x00000200
	FILE_ATTRIBUTE_REPARSE_POINT         = 0x00000400
	FILE_ATTRIBUTE_COMPRESSED            = 0x00000800
	FILE_ATTRIBUTE_OFFLINE               = 0x00001000
	FILE_ATTRIBUTE_NOT_CONTENT_INDEXED   = 0x00002000
	FILE_ATTRIBUTE_ENCRYPTED             = 0x00004000
	FILE_ATTRIBUTE_INTEGRITY_STREAM      = 0x00008000
	FILE_ATTRIBUTE_VIRTUAL               = 0x00010000
	FILE_ATTRIBUTE_NO_SCRUB_DATA         = 0x00020000
	FILE_ATTRIBUTE_RECALL_ON_OPEN        = 0x00040000
	FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000

	INVALID_FILE_ATTRIBUTES = 0xffffffff

	CREATE_NEW        = 1
	CREATE_ALWAYS     = 2
	OPEN_EXISTING     = 3
	OPEN_ALWAYS       = 4
	TRUNCATE_EXISTING = 5

	FILE_FLAG_OPEN_REQUIRING_OPLOCK = 0x00040000
	FILE_FLAG_FIRST_PIPE_INSTANCE   = 0x00080000
	FILE_FLAG_OPEN_NO_RECALL        = 0x00100000
	FILE_FLAG_OPEN_REPARSE_POINT    = 0x00200000
	FILE_FLAG_SESSION_AWARE         = 0x00800000
	FILE_FLAG_POSIX_SEMANTICS       = 0x01000000
	FILE_FLAG_BACKUP_SEMANTICS      = 0x02000000
	FILE_FLAG_DELETE_ON_CLOSE       = 0x04000000
	FILE_FLAG_SEQUENTIAL_SCAN       = 0x08000000
	FILE_FLAG_RANDOM_ACCESS         = 0x10000000
	FILE_FLAG_NO_BUFFERING          = 0x20000000
	FILE_FLAG_OVERLAPPED            = 0x40000000
	FILE_FLAG_WRITE_THROUGH         = 0x80000000

	HANDLE_FLAG_INHERIT    = 0x00000001
	STARTF_USESTDHANDLES   = 0x00000100
	STARTF_USESHOWWINDOW   = 0x00000001
	DUPLICATE_CLOSE_SOURCE = 0x00000001
	DUPLICATE_SAME_ACCESS  = 0x00000002

	STD_INPUT_HANDLE  = -10 & (1<<32 - 1)
	STD_OUTPUT_HANDLE = -11 & (1<<32 - 1)
	STD_ERROR_HANDLE  = -12 & (1<<32 - 1)

	FILE_BEGIN   = 0
	FILE_CURRENT = 1
	FILE_END     = 2

	LANG_ENGLISH       = 0x09
	SUBLANG_ENGLISH_US = 0x01

	FORMAT_MESSAGE_ALLOCATE_BUFFER = 256
	FORMAT_MESSAGE_IGNORE_INSERTS  = 512
	FORMAT_MESSAGE_FROM_STRING     = 1024
	FORMAT_MESSAGE_FROM_HMODULE    = 2048
	FORMAT_MESSAGE_FROM_SYSTEM     = 4096
	FORMAT_MESSAGE_ARGUMENT_ARRAY  = 8192
	FORMAT_MESSAGE_MAX_WIDTH_MASK  = 255

	MAX_PATH      = 260
	MAX_LONG_PATH = 32768

	MAX_COMPUTERNAME_LENGTH = 15

	TIME_ZONE_ID_UNKNOWN  = 0
	TIME_ZONE_ID_STANDARD = 1

	TIME_ZONE_ID_DAYLIGHT = 2
	IGNORE                = 0
	INFINITE              = 0xffffffff

	WAIT_ABANDONED = 0x00000080
	WAIT_OBJECT_0  = 0x00000000
	WAIT_FAILED    = 0xFFFFFFFF

	// Access rights for process.
	PROCESS_CREATE_PROCESS            = 0x0080
	PROCESS_CREATE_THREAD             = 0x0002
	PROCESS_DUP_HANDLE                = 0x0040
	PROCESS_QUERY_INFORMATION         = 0x0400
	PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
	PROCESS_SET_INFORMATION           = 0x0200
	PROCESS_SET_QUOTA                 = 0x0100
	PROCESS_SUSPEND_RESUME            = 0x0800
	PROCESS_TERMINATE                 = 0x0001
	PROCESS_VM_OPERATION              = 0x0008
	PROCESS_VM_READ                   = 0x0010
	PROCESS_VM_WRITE                  = 0x0020

	// Access rights for thread.
	THREAD_DIRECT_IMPERSONATION      = 0x0200
	THREAD_GET_CONTEXT               = 0x0008
	THREAD_IMPERSONATE               = 0x0100
	THREAD_QUERY_INFORMATION         = 0x0040
	THREAD_QUERY_LIMITED_INFORMATION = 0x0800
	THREAD_SET_CONTEXT               = 0x0010
	THREAD_SET_INFORMATION           = 0x0020
	THREAD_SET_LIMITED_INFORMATION   = 0x0400
	THREAD_SET_THREAD_TOKEN          = 0x0080
	THREAD_SUSPEND_RESUME            = 0x0002
	THREAD_TERMINATE                 = 0x0001

	FILE_MAP_COPY    = 0x01
	FILE_MAP_WRITE   = 0x02
	FILE_MAP_READ    = 0x04
	FILE_MAP_EXECUTE = 0x20

	CTRL_C_EVENT        = 0
	CTRL_BREAK_EVENT    = 1
	CTRL_CLOSE_EVENT    = 2
	CTRL_LOGOFF_EVENT   = 5
	CTRL_SHUTDOWN_EVENT = 6

	// Windows reserves errors >= 1<<29 for application use.
	APPLICATION_ERROR = 1 << 29
)

const (
	// Process creation flags.
	CREATE_BREAKAWAY_FROM_JOB        = 0x01000000
	CREATE_DEFAULT_ERROR_MODE        = 0x04000000
	CREATE_NEW_CONSOLE               = 0x00000010
	CREATE_NEW_PROCESS_GROUP         = 0x00000200
	CREATE_NO_WINDOW                 = 0x08000000
	CREATE_PROTECTED_PROCESS         = 0x00040000
	CREATE_PRESERVE_CODE_AUTHZ_LEVEL = 0x02000000
	CREATE_SEPARATE_WOW_VDM          = 0x00000800
	CREATE_SHARED_WOW_VDM            = 0x00001000
	CREATE_SUSPENDED                 = 0x00000004
	CREATE_UNICODE_ENVIRONMENT       = 0x00000400
	DEBUG_ONLY_THIS_PROCESS          = 0x00000002
	DEBUG_PROCESS                    = 0x00000001
	DETACHED_PROCESS                 = 0x00000008
	EXTENDED_STARTUPINFO_PRESENT     = 0x00080000
	INHERIT_PARENT_AFFINITY          = 0x00010000
)

const (
	// flags for CreateToolhelp32Snapshot
	TH32CS_SNAPHEAPLIST = 0x01
	TH32CS_SNAPPROCESS  = 0x02
	TH32CS_SNAPTHREAD   = 0x04
	TH32CS_SNAPMODULE   = 0x08
	TH32CS_SNAPMODULE32 = 0x10
	TH32CS_SNAPALL      = TH32CS_SNAPHEAPLIST | TH32CS_SNAPMODULE | TH32CS_SNAPPROCESS | TH32CS_SNAPTHREAD
	TH32CS_INHERIT      = 0x80000000
)

const (
	// filters for ReadDirectoryChangesW
	FILE_NOTIFY_CHANGE_FILE_NAME   = 0x001
	FILE_NOTIFY_CHANGE_DIR_NAME    = 0x002
	FILE_NOTIFY_CHANGE_ATTRIBUTES  = 0x004
	FILE_NOTIFY_CHANGE_SIZE        = 0x008
	FILE_NOTIFY_CHANGE_LAST_WRITE  = 0x010
	FILE_NOTIFY_CHANGE_LAST_ACCESS = 0x020
	FILE_NOTIFY_CHANGE_CREATION    = 0x040
	FILE_NOTIFY_CHANGE_SECURITY    = 0x100
)

const (
	// do not reorder
	FILE_ACTION_ADDED = iota + 1
	FILE_ACTION_REMOVED
	FILE_ACTION_MODIFIED
	FILE_ACTION_RENAMED_OLD_NAME
	FILE_ACTION_RENAMED_NEW_NAME
)

const (
	// wincrypt.h
	PROV_RSA_FULL                    = 1
	PROV_RSA_SIG                     = 2
	PROV_DSS                         = 3
	PROV_FORTEZZA                    = 4
	PROV_MS_EXCHANGE                 = 5
	PROV_SSL                         = 6
	PROV_RSA_SCHANNEL                = 12
	PROV_DSS_DH                      = 13
	PROV_EC_ECDSA_SIG                = 14
	PROV_EC_ECNRA_SIG                = 15
	PROV_EC_ECDSA_FULL               = 16
	PROV_EC_ECNRA_FULL               = 17
	PROV_DH_SCHANNEL                 = 18
	PROV_SPYRUS_LYNKS                = 20
	PROV_RNG                         = 21
	PROV_INTEL_SEC                   = 22
	PROV_REPLACE_OWF                 = 23
	PROV_RSA_AES                     = 24
	CRYPT_VERIFYCONTEXT              = 0xF0000000
	CRYPT_NEWKEYSET                  = 0x00000008
	CRYPT_DELETEKEYSET               = 0x00000010
	CRYPT_MACHINE_KEYSET             = 0x00000020
	CRYPT_SILENT                     = 0x00000040
	CRYPT_DEFAULT_CONTAINER_OPTIONAL = 0x00000080

	USAGE_MATCH_TYPE_AND = 0
	USAGE_MATCH_TYPE_OR  = 1

	/* msgAndCertEncodingType values for CertOpenStore function */
	X509_ASN_ENCODING   = 0x00000001
	PKCS_7_ASN_ENCODING = 0x00010000

	/* storeProvider values for CertOpenStore function */
	CERT_STORE_PROV_MSG               = 1
	CERT_STORE_PROV_MEMORY            = 2
	CERT_STORE_PROV_FILE              = 3
	CERT_STORE_PROV_REG               = 4
	CERT_STORE_PROV_PKCS7             = 5
	CERT_STORE_PROV_SERIALIZED        = 6
	CERT_STORE_PROV_FILENAME_A        = 7
	CERT_STORE_PROV_FILENAME_W        = 8
	CERT_STORE_PROV_FILENAME          = CERT_STORE_PROV_FILENAME_W
	CERT_STORE_PROV_SYSTEM_A          = 9
	CERT_STORE_PROV_SYSTEM_W          = 10
	CERT_STORE_PROV_SYSTEM            = CERT_STORE_PROV_SYSTEM_W
	CERT_STORE_PROV_COLLECTION        = 11
	CERT_STORE_PROV_SYSTEM_REGISTRY_A = 12
	CERT_STORE_PROV_SYSTEM_REGISTRY_W = 13
	CERT_STORE_PROV_SYSTEM_REGISTRY   = CERT_STORE_PROV_SYSTEM_REGISTRY_W
	CERT_STORE_PROV_PHYSICAL_W        = 14
	CERT_STORE_PROV_PHYSICAL          = CERT_STORE_PROV_PHYSICAL_W
	CERT_STORE_PROV_SMART_CARD_W      = 15
	CERT_STORE_PROV_SMART_CARD        = CERT_STORE_PROV_SMART_CARD_W
	CERT_STORE_PROV_LDAP_W            = 16
	CERT_STORE_PROV_LDAP              = CERT_STORE_PROV_LDAP_W
	CERT_STORE_PROV_PKCS12            = 17

	/* store characteristics (low WORD of flag) for CertOpenStore function */
	CERT_STORE_NO_CRYPT_RELEASE_FLAG            = 0x00000001
	CERT_STORE_SET_LOCALIZED_NAME_FLAG          = 0x00000002
	CERT_STORE_DEFER_CLOSE_UNTIL_LAST_FREE_FLAG = 0x00000004
	CERT_STORE_DELETE_FLAG                      = 0x00000010
	CERT_STORE_UNSAFE_PHYSICAL_FLAG             = 0x00000020
	CERT_STORE_SHARE_STORE_FLAG                 = 0x00000040
	CERT_STORE_SHARE_CONTEXT_FLAG               = 0x00000080
	CERT_STORE_MANIFOLD_FLAG                    = 0x00000100
	CERT_STORE_ENUM_ARCHIVED_FLAG               = 0x00000200
	CERT_STORE_UPDATE_KEYID_FLAG                = 0x00000400
	CERT_STORE_BACKUP_RESTORE_FLAG              = 0x00000800
	CERT_STORE_MAXIMUM_ALLOWED_FLAG             = 0x00001000
	CERT_STORE_CREATE_NEW_FLAG                  = 0x00002000
	CERT_STORE_OPEN_EXISTING_FLAG               = 0x00004000
	CERT_STORE_READONLY_FLAG                    = 0x00008000

	/* store locations (high WORD of flag) for CertOpenStore function */
	CERT_SYSTEM_STORE_CURRENT_USER               = 0x00010000
	CERT_SYSTEM_STORE_LOCAL_MACHINE              = 0x00020000
	CERT_SYSTEM_STORE_CURRENT_SERVICE            = 0x00040000
	CERT_SYSTEM_STORE_SERVICES                   = 0x00050000
	CERT_SYSTEM_STORE_USERS                      = 0x00060000
	CERT_SYSTEM_STORE_CURRENT_USER_GROUP_POLICY  = 0x00070000
	CERT_SYSTEM_STORE_LOCAL_MACHINE_GROUP_POLICY = 0x00080000
	CERT_SYSTEM_STORE_LOCAL_MACHINE_ENTERPRISE   = 0x00090000
	CERT_SYSTEM_STORE_UNPROTECTED_FLAG           = 0x40000000
	CERT_SYSTEM_STORE_RELOCATE_FLAG              = 0x80000000

	/* Miscellaneous high-WORD flags for CertOpenStore function */
	CERT_REGISTRY_STORE_REMOTE_FLAG      = 0x00010000
	CERT_REGISTRY_STORE_SERIALIZED_FLAG  = 0x00020000
	CERT_REGISTRY_STORE_ROAMING_FLAG     = 0x00040000
	CERT_REGISTRY_STORE_MY_IE_DIRTY_FLAG = 0x00080000
	CERT_REGISTRY_STORE_LM_GPT_FLAG      = 0x01000000
	CERT_REGISTRY_STORE_CLIENT_GPT_FLAG  = 0x80000000
	CERT_FILE_STORE_COMMIT_ENABLE_FLAG   = 0x00010000
	CERT_LDAP_STORE_SIGN_FLAG            = 0x00010000
	CERT_LDAP_STORE_AREC_EXCLUSIVE_FLAG  = 0x00020000
	CERT_LDAP_STORE_OPENED_FLAG          = 0x00040000
	CERT_LDAP_STORE_UNBIND_FLAG          = 0x00080000

	/* addDisposition values for CertAddCertificateContextToStore function */
	CERT_STORE_ADD_NEW                                 = 1
	CERT_STORE_ADD_USE_EXISTING                        = 2
	CERT_STORE_ADD_REPLACE_EXISTING                    = 3
	CERT_STORE_ADD_ALWAYS                              = 4
	CERT_STORE_ADD_REPLACE_EXISTING_INHERIT_PROPERTIES = 5
	CERT_STORE_ADD_NEWER                               = 6
	CERT_STORE_ADD_NEWER_INHERIT_PROPERTIES            = 7

	/* ErrorStatus values for CertTrustStatus struct */
	CERT_TRUST_NO_ERROR                          = 0x00000000
	CERT_TRUST_IS_NOT_TIME_VALID                 = 0x00000001
	CERT_TRUST_IS_REVOKED                        = 0x00000004
	CERT_TRUST_IS_NOT_SIGNATURE_VALID            = 0x00000008
	CERT_TRUST_IS_NOT_VALID_FOR_USAGE            = 0x00000010
	CERT_TRUST_IS_UNTRUSTED_ROOT                 = 0x00000020
	CERT_TRUST_REVOCATION_STATUS_UNKNOWN         = 0x00000040
	CERT_TRUST_IS_CYCLIC                         = 0x00000080
	CERT_TRUST_INVALID_EXTENSION                 = 0x00000100
	CERT_TRUST_INVALID_POLICY_CONSTRAINTS        = 0x00000200
	CERT_TRUST_INVALID_BASIC_CONSTRAINTS         = 0x00000400
	CERT_TRUST_INVALID_NAME_CONSTRAINTS          = 0x00000800
	CERT_TRUST_HAS_NOT_SUPPORTED_NAME_CONSTRAINT = 0x00001000
	CERT_TRUST_HAS_NOT_DEFINED_NAME_CONSTRAINT   = 0x00002000
	CERT_TRUST_HAS_NOT_PERMITTED_NAME_CONSTRAINT = 0x00004000
	CERT_TRUST_HAS_EXCLUDED_NAME_CONSTRAINT      = 0x00008000
	CERT_TRUST_IS_PARTIAL_CHAIN                  = 0x00010000
	CERT_TRUST_CTL_IS_NOT_TIME_VALID             = 0x00020000
	CERT_TRUST_CTL_IS_NOT_SIGNATURE_VALID        = 0x00040000
	CERT_TRUST_CTL_IS_NOT_VALID_FOR_USAGE        = 0x00080000
	CERT_TRUST_HAS_WEAK_SIGNATURE                = 0x00100000
	CERT_TRUST_IS_OFFLINE_REVOCATION             = 0x01000000
	CERT_TRUST_NO_ISSUANCE_CHAIN_POLICY          = 0x02000000
	CERT_TRUST_IS_EXPLICIT_DISTRUST              = 0x04000000
	CERT_TRUST_HAS_NOT_SUPPORTED_CRITICAL_EXT    = 0x08000000

	/* InfoStatus values for CertTrustStatus struct */
	CERT_TRUST_HAS_EXACT_MATCH_ISSUER        = 0x00000001
	CERT_TRUST_HAS_KEY_MATCH_ISSUER          = 0x00000002
	CERT_TRUST_HAS_NAME_MATCH_ISSUER         = 0x00000004
	CERT_TRUST_IS_SELF_SIGNED                = 0x00000008
	CERT_TRUST_HAS_PREFERRED_ISSUER          = 0x00000100
	CERT_TRUST_HAS_ISSUANCE_CHAIN_POLICY     = 0x00000400
	CERT_TRUST_HAS_VALID_NAME_CONSTRAINTS    = 0x00000400
	CERT_TRUST_IS_PEER_TRUSTED               = 0x00000800
	CERT_TRUST_HAS_CRL_VALIDITY_EXTENDED     = 0x00001000
	CERT_TRUST_IS_FROM_EXCLUSIVE_TRUST_STORE = 0x00002000
	CERT_TRUST_IS_CA_TRUSTED                 = 0x00004000
	CERT_TRUST_IS_COMPLEX_CHAIN              = 0x00010000

	/* policyOID values for CertVerifyCertificateChainPolicy function */
	CERT_CHAIN_POLICY_BASE              = 1
	CERT_CHAIN_POLICY_AUTHENTICODE      = 2
	CERT_CHAIN_POLICY_AUTHENTICODE_TS   = 3
	CERT_CHAIN_POLICY_SSL               = 4
	CERT_CHAIN_POLICY_BASIC_CONSTRAINTS = 5
	CERT_CHAIN_POLICY_NT_AUTH           = 6
	CERT_CHAIN_POLICY_MICROSOFT_ROOT    = 7
	CERT_CHAIN_POLICY_EV                = 8
	CERT_CHAIN_POLICY_SSL_F12           = 9

	/* AuthType values for SSLExtraCertChainPolicyPara struct */
	AUTHTYPE_CLIENT = 1
	AUTHTYPE_SERVER = 2

	/* Checks values for SSLExtraCertChainPolicyPara struct */
	SECURITY_FLAG_IGNORE_REVOCATION        = 0x00000080
	SECURITY_FLAG_IGNORE_UNKNOWN_CA        = 0x00000100
	SECURITY_FLAG_IGNORE_WRONG_USAGE       = 0x00000200
	SECURITY_FLAG_IGNORE_CERT_CN_INVALID   = 0x00001000
	SECURITY_FLAG_IGNORE_CERT_DATE_INVALID = 0x00002000
)

const (
	// flags for SetErrorMode
	SEM_FAILCRITICALERRORS     = 0x0001
	SEM_NOALIGNMENTFAULTEXCEPT = 0x0004
	SEM_NOGPFAULTERRORBOX      = 0x0002
	SEM_NOOPENFILEERRORBOX     = 0x8000
)

const (
	// Priority class.
	ABOVE_NORMAL_PRIORITY_CLASS   = 0x00008000
	BELOW_NORMAL_PRIORITY_CLASS   = 0x00004000
	HIGH_PRIORITY_CLASS           = 0x00000080
	IDLE_PRIORITY_CLASS           = 0x00000040
	NORMAL_PRIORITY_CLASS         = 0x00000020
	PROCESS_MODE_BACKGROUND_BEGIN = 0x00100000
	PROCESS_MODE_BACKGROUND_END   = 0x00200000
	REALTIME_PRIORITY_CLASS       = 0x00000100
)

var (
	OID_PKIX_KP_SERVER_AUTH = []byte("1.3.6.1.5.5.7.3.1\x00")
	OID_SERVER_GATED_CRYPTO = []byte("1.3.6.1.4.1.311.10.3.3\x00")
	OID_SGC_NETSCAPE        = []byte("2.16.840.1.113730.4.1\x00")
)

// Pointer represents a pointer to an arbitrary Windows type.
//
// Pointer-typed fields may point to one of many different types. It's
// up to the caller to provide a pointer to the appropriate type, cast
// to Pointer. The caller must obey the unsafe.Pointer rules while
// doing so.
type Pointer *struct{}

// Invented values to support what package os expects.
type Timeval struct {
	Sec  int32
	Usec int32
}

func (tv *Timeval) Nanoseconds() int64 {
	return (int64(tv.Sec)*1e6 + int64(tv.Usec)) * 1e3
}

func NsecToTimeval(nsec int64) (tv Timeval) {
	tv.Sec = int32(nsec / 1e9)
	tv.Usec = int32(nsec % 1e9 / 1e3)
	return
}

type Overlapped struct {
	Internal     uintptr
	InternalHigh uintptr
	Offset       uint32
	OffsetHigh   uint32
	HEvent       Handle
}

type FileNotifyInformation struct {
	NextEntryOffset uint32
	Action          uint32
	FileNameLength  uint32
	FileName        uint16
}

type Filetime struct {
	LowDateTime  uint32
	HighDateTime uint32
}

// Nanoseconds returns Filetime ft in nanoseconds
// since Epoch (00:00:00 UTC, January 1, 1970).
func (ft *Filetime) Nanoseconds() int64 {
	// 100-nanosecond intervals since January 1, 1601
	nsec := int64(ft.HighDateTime)<<32 + int64(ft.LowDateTime)
	// change starting time to the Epoch (00:00:00 UTC, January 1, 1970)
	nsec -= 116444736000000000
	// convert into nanoseconds
	nsec *= 100
	return nsec
}

func NsecToFiletime(nsec int64) (ft Filetime) {
	// convert into 100-nanosecond
	nsec /= 100
	// change starting time to January 1, 1601
	nsec += 116444736000000000
	// split into high / low
	ft.LowDateTime = uint32(nsec & 0xffffffff)
	ft.HighDateTime = uint32(nsec >> 32 & 0xffffffff)
	return ft
}

type Win32finddata struct {
	FileAttributes    uint32
	CreationTime      Filetime
	LastAccessTime    Filetime
	LastWriteTime     Filetime
	FileSizeHigh      uint32
	FileSizeLow       uint32
	Reserved0         uint32
	Reserved1         uint32
	FileName          [MAX_PATH - 1]uint16
	AlternateFileName [13]uint16
}

// This is the actual system call structure.
// Win32finddata is what we committed to in Go 1.
type win32finddata1 struct {
	FileAttributes    uint32
	CreationTime      Filetime
	LastAccessTime    Filetime
	LastWriteTime     Filetime
	FileSizeHigh      uint32
	FileSizeLow       uint32
	Reserved0         uint32
	Reserved1         uint32
	FileName          [MAX_PATH]uint16
	AlternateFileName [14]uint16
}

func copyFindData(dst *Win32finddata, src *win32finddata1) {
	dst.FileAttributes = src.FileAttributes
	dst.CreationTime = src.CreationTime
	dst.LastAccessTime = src.LastAccessTime
	dst.LastWriteTime = src.LastWriteTime
	dst.FileSizeHigh = src.FileSizeHigh
	dst.FileSizeLow = src.FileSizeLow
	dst.Reserved0 = src.Reserved0
	dst.Reserved1 = src.Reserved1

	// The src is 1 element bigger than dst, but it must be NUL.
	copy(dst.FileName[:], src.FileName[:])
	copy(dst.AlternateFileName[:], src.AlternateFileName[:])
}

type ByHandleFileInformation struct {
	FileAttributes     uint32
	CreationTime       Filetime
	LastAccessTime     Filetime
	LastWriteTime      Filetime
	VolumeSerialNumber uint32
	FileSizeHigh       uint32
	FileSizeLow        uint32
	NumberOfLinks      uint32
	FileIndexHigh      uint32
	FileIndexLow       uint32
}

const (
	GetFileExInfoStandard = 0
	GetFileExMaxInfoLevel = 1
)

type Win32FileAttributeData struct {
	FileAttributes uint32
	CreationTime   Filetime
	LastAccessTime Filetime
	LastWriteTime  Filetime
	FileSizeHigh   uint32
	FileSizeLow    uint32
}

// ShowWindow constants
const (
	// winuser.h
	SW_HIDE            = 0
	SW_NORMAL          = 1
	SW_SHOWNORMAL      = 1
	SW_SHOWMINIMIZED   = 2
	SW_SHOWMAXIMIZED   = 3
	SW_MAXIMIZE        = 3
	SW_SHOWNOACTIVATE  = 4
	SW_SHOW            = 5
	SW_MINIMIZE        = 6
	SW_SHOWMINNOACTIVE = 7
	SW_SHOWNA          = 8
	SW_RESTORE         = 9
	SW_SHOWDEFAULT     = 10
	SW_FORCEMINIMIZE   = 11
)

type StartupInfo struct {
	Cb            uint32
	_             *uint16
	Desktop       *uint16
	Title         *uint16
	X             uint32
	Y             uint32
	XSize         uint32
	YSize         uint32
	XCountChars   uint32
	YCountChars   uint32
	FillAttribute uint32
	Flags         uint32
	ShowWindow    uint16
	_             uint16
	_             *byte
	StdInput      Handle
	StdOutput     Handle
	StdErr        Handle
}

type ProcessInformation struct {
	Process   Handle
	Thread    Handle
	ProcessId uint32
	ThreadId  uint32
}

type ProcessEntry32 struct {
	Size            uint32
	Usage           uint32
	ProcessID       uint32
	DefaultHeapID   uintptr
	ModuleID        uint32
	Threads         uint32
	ParentProcessID uint32
	PriClassBase    int32
	Flags           uint32
	ExeFile         [MAX_PATH]uint16
}

type ThreadEntry32 struct {
	Size           uint32
	Usage          uint32
	ThreadID       uint32
	OwnerProcessID uint32
	BasePri        int32
	DeltaPri       int32
	Flags          uint32
}

type Systemtime struct {
	Year         uint16
	Month        uint16
	DayOfWeek    uint16
	Day          uint16
	Hour         uint16
	Minute       uint16
	Second       uint16
	Milliseconds uint16
}

type Timezoneinformation struct {
	Bias         int32
	StandardName [32]uint16
	StandardDate Systemtime
	StandardBias int32
	DaylightName [32]uint16
	DaylightDate Systemtime
	DaylightBias int32
}

// Socket related.

const (
	AF_UNSPEC  = 0
	AF_UNIX    = 1
	AF_INET    = 2
	AF_NETBIOS = 17
	AF_INET6   = 23
	AF_IRDA    = 26
	AF_BTH     = 32

	SOCK_STREAM    = 1
	SOCK_DGRAM     = 2
	SOCK_RAW       = 3
	SOCK_RDM       = 4
	SOCK_SEQPACKET = 5

	IPPROTO_IP      = 0
	IPPROTO_ICMP    = 1
	IPPROTO_IGMP    = 2
	BTHPROTO_RFCOMM = 3
	IPPROTO_TCP     = 6
	IPPROTO_UDP     = 17
	IPPROTO_IPV6    = 41
	IPPROTO_ICMPV6  = 58
	IPPROTO_RM      = 113

	SOL_SOCKET                = 0xffff
	SO_REUSEADDR              = 4
	SO_KEEPALIVE              = 8
	SO_DONTROUTE              = 16
	SO_BROADCAST              = 32
	SO_LINGER                 = 128
	SO_RCVBUF                 = 0x1002
	SO_RCVTIMEO               = 0x1006
	SO_SNDBUF                 = 0x1001
	SO_UPDATE_ACCEPT_CONTEXT  = 0x700b
	SO_UPDATE_CONNECT_CONTEXT = 0x7010

	IOC_OUT                            = 0x40000000
	IOC_IN                             = 0x80000000
	IOC_VENDOR                         = 0x18000000
	IOC_INOUT                          = IOC_IN | IOC_OUT
	IOC_WS2                            = 0x08000000
	SIO_GET_EXTENSION_FUNCTION_POINTER = IOC_INOUT | IOC_WS2 | 6
	SIO_KEEPALIVE_VALS                 = IOC_IN | IOC_VENDOR | 4
	SIO_UDP_CONNRESET                  = IOC_IN | IOC_VENDOR | 12

	// cf. http://support.microsoft.com/default.aspx?scid=kb;en-us;257460

	IP_TOS             = 0x3
	IP_TTL             = 0x4
	IP_MULTICAST_IF    = 0x9
	IP_MULTICAST_TTL   = 0xa
	IP_MULTICAST_LOOP  = 0xb
	IP_ADD_MEMBERSHIP  = 0xc
	IP_DROP_MEMBERSHIP = 0xd

	IPV6_V6ONLY         = 0x1b
	IPV6_UNICAST_HOPS   = 0x4
	IPV6_MULTICAST_IF   = 0x9
	IPV6_MULTICAST_HOPS = 0xa
	IPV6_MULTICAST_LOOP = 0xb
	IPV6_JOIN_GROUP     = 0xc
	IPV6_LEAVE_GROUP    = 0xd

	MSG_OOB       = 0x1
	MSG_PEEK      = 0x2
	MSG_DONTROUTE = 0x4
	MSG_WAITALL   = 0x8

	MSG_TRUNC  = 0x0100
	MSG_CTRUNC = 0x0200
	MSG_BCAST  = 0x0400
	MSG_MCAST  = 0x0800

	SOMAXCONN = 0x7fffffff

	TCP_NODELAY = 1

	SHUT_RD   = 0
	SHUT_WR   = 1
	SHUT_RDWR = 2

	WSADESCRIPTION_LEN = 256
	WSASYS_STATUS_LEN  = 128
)

type WSABuf struct {
	Len uint32
	Buf *byte
}

type WSAMsg struct {
	Name        *syscall.RawSockaddrAny
	Namelen     int32
	Buffers     *WSABuf
	BufferCount uint32
	Control     WSABuf
	Flags       uint32
}

// Invented values to support what package os expects.
const (
	S_IFMT   = 0x1f000
	S_IFIFO  = 0x1000
	S_IFCHR  = 0x2000
	S_IFDIR  = 0x4000
	S_IFBLK  = 0x6000
	S_IFREG  = 0x8000
	S_IFLNK  = 0xa000
	S_IFSOCK = 0xc000
	S_ISUID  = 0x800
	S_ISGID  = 0x400
	S_ISVTX  = 0x200
	S_IRUSR  = 0x100
	S_IWRITE = 0x80
	S_IWUSR  = 0x80
	S_IXUSR  = 0x40
)

const (
	FILE_TYPE_CHAR    = 0x0002
	FILE_TYPE_DISK    = 0x0001
	FILE_TYPE_PIPE    = 0x0003
	FILE_TYPE_REMOTE  = 0x8000
	FILE_TYPE_UNKNOWN = 0x0000
)

type Hostent struct {
	Name     *byte
	Aliases  **byte
	AddrType uint16
	Length   uint16
	AddrList **byte
}

type Protoent struct {
	Name    *byte
	Aliases **byte
	Proto   uint16
}

const (
	DNS_TYPE_A       = 0x0001
	DNS_TYPE_NS      = 0x0002
	DNS_TYPE_MD      = 0x0003
	DNS_TYPE_MF      = 0x0004
	DNS_TYPE_CNAME   = 0x0005
	DNS_TYPE_SOA     = 0x0006
	DNS_TYPE_MB      = 0x0007
	DNS_TYPE_MG      = 0x0008
	DNS_TYPE_MR      = 0x0009
	DNS_TYPE_NULL    = 0x000a
	DNS_TYPE_WKS     = 0x000b
	DNS_TYPE_PTR     = 0x000c
	DNS_TYPE_HINFO   = 0x000d
	DNS_TYPE_MINFO   = 0x000e
	DNS_TYPE_MX      = 0x000f
	DNS_TYPE_TEXT    = 0x0010
	DNS_TYPE_RP      = 0x0011
	DNS_TYPE_AFSDB   = 0x0012
	DNS_TYPE_X25     = 0x0013
	DNS_TYPE_ISDN    = 0x0014
	DNS_TYPE_RT      = 0x0015
	DNS_TYPE_NSAP    = 0x0016
	DNS_TYPE_NSAPPTR = 0x0017
	DNS_TYPE_SIG     = 0x0018
	DNS_TYPE_KEY     = 0x0019
	DNS_TYPE_PX      = 0x001a
	DNS_TYPE_GPOS    = 0x001b
	DNS_TYPE_AAAA    = 0x001c
	DNS_TYPE_LOC     = 0x001d
	DNS_TYPE_NXT     = 0x001e
	DNS_TYPE_EID     = 0x001f
	DNS_TYPE_NIMLOC  = 0x0020
	DNS_TYPE_SRV     = 0x0021
	DNS_TYPE_ATMA    = 0x0022
	DNS_TYPE_NAPTR   = 0x0023
	DNS_TYPE_KX      = 0x0024
	DNS_TYPE_CERT    = 0x0025
	DNS_TYPE_A6      = 0x0026
	DNS_TYPE_DNAME   = 0x0027
	DNS_TYPE_SINK    = 0x0028
	DNS_TYPE_OPT     = 0x0029
	DNS_TYPE_DS      = 0x002B
	DNS_TYPE_RRSIG   = 0x002E
	DNS_TYPE_NSEC    = 0x002F
	DNS_TYPE_DNSKEY  = 0x0030
	DNS_TYPE_DHCID   = 0x0031
	DNS_TYPE_UINFO   = 0x0064
	DNS_TYPE_UID     = 0x0065
	DNS_TYPE_GID     = 0x0066
	DNS_TYPE_UNSPEC  = 0x0067
	DNS_TYPE_ADDRS   = 0x00f8
	DNS_TYPE_TKEY    = 0x00f9
	DNS_TYPE_TSIG    = 0x00fa
	DNS_TYPE_IXFR    = 0x00fb
	DNS_TYPE_AXFR    = 0x00fc
	DNS_TYPE_MAILB   = 0x00fd
	DNS_TYPE_MAILA   = 0x00fe
	DNS_TYPE_ALL     = 0x00ff
	DNS_TYPE_ANY     = 0x00ff
	DNS_TYPE_WINS    = 0xff01
	DNS_TYPE_WINSR   = 0xff02
	DNS_TYPE_NBSTAT  = 0xff01
)

const (
	// flags inside DNSRecord.Dw
	DnsSectionQuestion   = 0x0000
	DnsSectionAnswer     = 0x0001
	DnsSectionAuthority  = 0x0002
	DnsSectionAdditional = 0x0003
)

type DNSSRVData struct {
	Target   *uint16
	Priority uint16
	Weight   uint16
	Port     uint16
	Pad      uint16
}

type DNSPTRData struct {
	Host *uint16
}

type DNSMXData struct {
	NameExchange *uint16
	Preference   uint16
	Pad          uint16
}

type DNSTXTData struct {
	StringCount uint16
	StringArray [1]*uint16
}

type DNSRecord struct {
	Next     *DNSRecord
	Name     *uint16
	Type     uint16
	Length   uint16
	Dw       uint32
	Ttl      uint32
	Reserved uint32
	Data     [40]byte
}

const (
	TF_DISCONNECT         = 1
	TF_REUSE_SOCKET       = 2
	TF_WRITE_BEHIND       = 4
	TF_USE_DEFAULT_WORKER = 0
	TF_USE_SYSTEM_THREAD  = 16
	TF_USE_KERNEL_APC     = 32
)

type TransmitFileBuffers struct {
	Head       uintptr
	HeadLength uint32
	Tail       uintptr
	TailLength uint32
}

const (
	IFF_UP           = 1
	IFF_BROADCAST    = 2
	IFF_LOOPBACK     = 4
	IFF_POINTTOPOINT = 8
	IFF_MULTICAST    = 16
)

const SIO_GET_INTERFACE_LIST = 0x4004747F

// TODO(mattn): SockaddrGen is union of sockaddr/sockaddr_in/sockaddr_in6_old.
// will be fixed to change variable type as suitable.

type SockaddrGen [24]byte

type InterfaceInfo struct {
	Flags            uint32
	Address          SockaddrGen
	BroadcastAddress SockaddrGen
	Netmask          SockaddrGen
}

type IpAddressString struct {
	String [16]byte
}

type IpMaskString IpAddressString

type IpAddrString struct {
	Next      *IpAddrString
	IpAddress IpAddressString
	IpMask    IpMaskString
	Context   uint32
}

const MAX_ADAPTER_NAME_LENGTH = 256
const MAX_ADAPTER_DESCRIPTION_LENGTH = 128
const MAX_ADAPTER_ADDRESS_LENGTH = 8

type IpAdapterInfo struct {
	Next                *IpAdapterInfo
	ComboIndex          uint32
	AdapterName         [MAX_ADAPTER_NAME_LENGTH + 4]byte
	Description         [MAX_ADAPTER_DESCRIPTION_LENGTH + 4]byte
	AddressLength       uint32
	Address             [MAX_ADAPTER_ADDRESS_LENGTH]byte
	Index               uint32
	Type                uint32
	DhcpEnabled         uint32
	CurrentIpAddress    *IpAddrString
	IpAddressList       IpAddrString
	GatewayList         IpAddrString
	DhcpServer          IpAddrString
	HaveWins            bool
	PrimaryWinsServer   IpAddrString
	SecondaryWinsServer IpAddrString
	LeaseObtained       int64
	LeaseExpires        int64
}

const MAXLEN_PHYSADDR = 8
const MAX_INTERFACE_NAME_LEN = 256
const MAXLEN_IFDESCR = 256

type MibIfRow struct {
	Name            [MAX_INTERFACE_NAME_LEN]uint16
	Index           uint32
	Type            uint32
	Mtu             uint32
	Speed           uint32
	PhysAddrLen     uint32
	PhysAddr        [MAXLEN_PHYSADDR]byte
	AdminStatus     uint32
	OperStatus      uint32
	LastChange      uint32
	InOctets        uint32
	InUcastPkts     uint32
	InNUcastPkts    uint32
	InDiscards      uint32
	InErrors        uint32
	InUnknownProtos uint32
	OutOctets       uint32
	OutUcastPkts    uint32
	OutNUcastPkts   uint32
	OutDiscards     uint32
	OutErrors       uint32
	OutQLen         uint32
	DescrLen        uint32
	Descr           [MAXLEN_IFDESCR]byte
}

type CertInfo struct {
	// Not implemented
}

type CertContext struct {
	EncodingType uint32
	EncodedCert  *byte
	Length       uint32
	CertInfo     *CertInfo
	Store        Handle
}

type CertChainContext struct {
	Size                       uint32
	TrustStatus                CertTrustStatus
	ChainCount                 uint32
	Chains                     **CertSimpleChain
	LowerQualityChainCount     uint32
	LowerQualityChains         **CertChainContext
	HasRevocationFreshnessTime uint32
	RevocationFreshnessTime    uint32
}

type CertTrustListInfo struct {
	// Not implemented
}

type CertSimpleChain struct {
	Size                       uint32
	TrustStatus                CertTrustStatus
	NumElements                uint32
	Elements                   **CertChainElement
	TrustListInfo              *CertTrustListInfo
	HasRevocationFreshnessTime uint32
	RevocationFreshnessTime    uint32
}

type CertChainElement struct {
	Size              uint32
	CertContext       *CertContext
	TrustStatus       CertTrustStatus
	RevocationInfo    *CertRevocationInfo
	IssuanceUsage     *CertEnhKeyUsage
	ApplicationUsage  *CertEnhKeyUsage
	ExtendedErrorInfo *uint16
}

type CertRevocationCrlInfo struct {
	// Not implemented
}

type CertRevocationInfo struct {
	Size             uint32
	RevocationResult uint32
	RevocationOid    *byte
	OidSpecificInfo  Pointer
	HasFreshnessTime uint32
	FreshnessTime    uint32
	CrlInfo          *CertRevocationCrlInfo
}

type CertTrustStatus struct {
	ErrorStatus uint32
	InfoStatus  uint32
}

type CertUsageMatch struct {
	Type  uint32
	Usage CertEnhKeyUsage
}

type CertEnhKeyUsage struct {
	Length           uint32
	UsageIdentifiers **byte
}

type CertChainPara struct {
	Size                         uint32
	RequestedUsage               CertUsageMatch
	RequstedIssuancePolicy       CertUsageMatch
	URLRetrievalTimeout          uint32
	CheckRevocationFreshnessTime uint32
	RevocationFreshnessTime      uint32
	CacheResync                  *Filetime
}

type CertChainPolicyPara struct {
	Size            uint32
	Flags           uint32
	ExtraPolicyPara Pointer
}

type SSLExtraCertChainPolicyPara struct {
	Size       uint32
	AuthType   uint32
	Checks     uint32
	ServerName *uint16
}

type CertChainPolicyStatus struct {
	Size              uint32
	Error             uint32
	ChainIndex        uint32
	ElementIndex      uint32
	ExtraPolicyStatus Pointer
}

const (
	// do not reorder
	HKEY_CLASSES_ROOT = 0x80000000 + iota
	HKEY_CURRENT_USER
	HKEY_LOCAL_MACHINE
	HKEY_USERS
	HKEY_PERFORMANCE_DATA
	HKEY_CURRENT_CONFIG
	HKEY_DYN_DATA

	KEY_QUERY_VALUE        = 1
	KEY_SET_VALUE          = 2
	KEY_CREATE_SUB_KEY     = 4
	KEY_ENUMERATE_SUB_KEYS = 8
	KEY_NOTIFY             = 16
	KEY_CREATE_LINK        = 32
	KEY_WRITE              = 0x20006
	KEY_EXECUTE            = 0x20019
	KEY_READ               = 0x20019
	KEY_WOW64_64KEY        = 0x0100
	KEY_WOW64_32KEY        = 0x0200
	KEY_ALL_ACCESS         = 0xf003f
)

const (
	// do not reorder
	REG_NONE = iota
	REG_SZ
	REG_EXPAND_SZ
	REG_BINARY
	REG_DWORD_LITTLE_ENDIAN
	REG_DWORD_BIG_ENDIAN
	REG_LINK
	REG_MULTI_SZ
	REG_RESOURCE_LIST
	REG_FULL_RESOURCE_DESCRIPTOR
	REG_RESOURCE_REQUIREMENTS_LIST
	REG_QWORD_LITTLE_ENDIAN
	REG_DWORD = REG_DWORD_LITTLE_ENDIAN
	REG_QWORD = REG_QWORD_LITTLE_ENDIAN
)

const (
	EVENT_MODIFY_STATE = 0x0002
	EVENT_ALL_ACCESS   = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x3

	MUTANT_QUERY_STATE = 0x0001
	MUTANT_ALL_ACCESS  = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | MUTANT_QUERY_STATE

	SEMAPHORE_MODIFY_STATE = 0x0002
	SEMAPHORE_ALL_ACCESS   = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x3

	TIMER_QUERY_STATE  = 0x0001
	TIMER_MODIFY_STATE = 0x0002
	TIMER_ALL_ACCESS   = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | TIMER_QUERY_STATE | TIMER_MODIFY_STATE

	MUTEX_MODIFY_STATE = MUTANT_QUERY_STATE
	MUTEX_ALL_ACCESS   = MUTANT_ALL_ACCESS

	CREATE_EVENT_MANUAL_RESET  = 0x1
	CREATE_EVENT_INITIAL_SET   = 0x2
	CREATE_MUTEX_INITIAL_OWNER = 0x1
)

type AddrinfoW struct {
	Flags     int32
	Family    int32
	Socktype  int32
	Protocol  int32
	Addrlen   uintptr
	Canonname *uint16
	Addr      uintptr
	Next      *AddrinfoW
}

const (
	AI_PASSIVE     = 1
	AI_CANONNAME   = 2
	AI_NUMERICHOST = 4
)

type GUID struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}

var WSAID_CONNECTEX = GUID{
	0x25a207b9,
	0xddf3,
	0x4660,
	[8]byte{0x8e, 0xe9, 0x76, 0xe5, 0x8c, 0x74, 0x06, 0x3e},
}

var WSAID_WSASENDMSG = GUID{
	0xa441e712,
	0x754f,
	0x43ca,
	[8]byte{0x84, 0xa7, 0x0d, 0xee, 0x44, 0xcf, 0x60, 0x6d},
}

var WSAID_WSARECVMSG = GUID{
	0xf689d7c8,
	0x6f1f,
	0x436b,
	[8]byte{0x8a, 0x53, 0xe5, 0x4f, 0xe3, 0x51, 0xc3, 0x22},
}

const (
	FILE_SKIP_COMPLETION_PORT_ON_SUCCESS = 1
	FILE_SKIP_SET_EVENT_ON_HANDLE        = 2
)

const (
	WSAPROTOCOL_LEN    = 255
	MAX_PROTOCOL_CHAIN = 7
	BASE_PROTOCOL      = 1
	LAYERED_PROTOCOL   = 0

	XP1_CONNECTIONLESS           = 0x00000001
	XP1_GUARANTEED_DELIVERY      = 0x00000002
	XP1_GUARANTEED_ORDER         = 0x00000004
	XP1_MESSAGE_ORIENTED         = 0x00000008
	XP1_PSEUDO_STREAM            = 0x00000010
	XP1_GRACEFUL_CLOSE           = 0x00000020
	XP1_EXPEDITED_DATA           = 0x00000040
	XP1_CONNECT_DATA             = 0x00000080
	XP1_DISCONNECT_DATA          = 0x00000100
	XP1_SUPPORT_BROADCAST        = 0x00000200
	XP1_SUPPORT_MULTIPOINT       = 0x00000400
	XP1_MULTIPOINT_CONTROL_PLANE = 0x00000800
	XP1_MULTIPOINT_DATA_PLANE    = 0x00001000
	XP1_QOS_SUPPORTED            = 0x00002000
	XP1_UNI_SEND                 = 0x00008000
	XP1_UNI_RECV                 = 0x00010000
	XP1_IFS_HANDLES              = 0x00020000
	XP1_PARTIAL_MESSAGE          = 0x00040000
	XP1_SAN_SUPPORT_SDP          = 0x00080000

	PFL_MULTIPLE_PROTO_ENTRIES  = 0x00000001
	PFL_RECOMMENDED_PROTO_ENTRY = 0x00000002
	PFL_HIDDEN                  = 0x00000004
	PFL_MATCHES_PROTOCOL_ZERO   = 0x00000008
	PFL_NETWORKDIRECT_PROVIDER  = 0x00000010
)

type WSAProtocolInfo struct {
	ServiceFlags1     uint32
	ServiceFlags2     uint32
	ServiceFlags3     uint32
	ServiceFlags4     uint32
	ProviderFlags     uint32
	ProviderId        GUID
	CatalogEntryId    uint32
	ProtocolChain     WSAProtocolChain
	Version           int32
	AddressFamily     int32
	MaxSockAddr       int32
	MinSockAddr       int32
	SocketType        int32
	Protocol          int32
	ProtocolMaxOffset int32
	NetworkByteOrder  int32
	SecurityScheme    int32
	MessageSize       uint32
	ProviderReserved  uint32
	ProtocolName      [WSAPROTOCOL_LEN + 1]uint16
}

type WSAProtocolChain struct {
	ChainLen     int32
	ChainEntries [MAX_PROTOCOL_CHAIN]uint32
}

type TCPKeepalive struct {
	OnOff    uint32
	Time     uint32
	Interval uint32
}

type symbolicLinkReparseBuffer struct {
	SubstituteNameOffset uint16
	SubstituteNameLength uint16
	PrintNameOffset      uint16
	PrintNameLength      uint16
	Flags                uint32
	PathBuffer           [1]uint16
}

type mountPointReparseBuffer struct {
	SubstituteNameOffset uint16
	SubstituteNameLength uint16
	PrintNameOffset      uint16
	PrintNameLength      uint16
	PathBuffer           [1]uint16
}

type reparseDataBuffer struct {
	ReparseTag        uint32
	ReparseDataLength uint16
	Reserved          uint16

	// GenericReparseBuffer
	reparseBuffer byte
}

const (
	FSCTL_GET_REPARSE_POINT          = 0x900A8
	MAXIMUM_REPARSE_DATA_BUFFER_SIZE = 16 * 1024
	IO_REPARSE_TAG_MOUNT_POINT       = 0xA0000003
	IO_REPARSE_TAG_SYMLINK           = 0xA000000C
	SYMBOLIC_LINK_FLAG_DIRECTORY     = 0x1
)

const (
	ComputerNameNetBIOS                   = 0
	ComputerNameDnsHostname               = 1
	ComputerNameDnsDomain                 = 2
	ComputerNameDnsFullyQualified         = 3
	ComputerNamePhysicalNetBIOS           = 4
	ComputerNamePhysicalDnsHostname       = 5
	ComputerNamePhysicalDnsDomain         = 6
	ComputerNamePhysicalDnsFullyQualified = 7
	ComputerNameMax                       = 8
)

// For MessageBox()
const (
	MB_OK                   = 0x00000000
	MB_OKCANCEL             = 0x00000001
	MB_ABORTRETRYIGNORE     = 0x00000002
	MB_YESNOCANCEL          = 0x00000003
	MB_YESNO                = 0x00000004
	MB_RETRYCANCEL          = 0x00000005
	MB_CANCELTRYCONTINUE    = 0x00000006
	MB_ICONHAND             = 0x00000010
	MB_ICONQUESTION         = 0x00000020
	MB_ICONEXCLAMATION      = 0x00000030
	MB_ICONASTERISK         = 0x00000040
	MB_USERICON             = 0x00000080
	MB_ICONWARNING          = MB_ICONEXCLAMATION
	MB_ICONERROR            = MB_ICONHAND
	MB_ICONINFORMATION      = MB_ICONASTERISK
	MB_ICONSTOP             = MB_ICONHAND
	MB_DEFBUTTON1           = 0x00000000
	MB_DEFBUTTON2           = 0x00000100
	MB_DEFBUTTON3           = 0x00000200
	MB_DEFBUTTON4           = 0x00000300
	MB_APPLMODAL            = 0x00000000
	MB_SYSTEMMODAL          = 0x00001000
	MB_TASKMODAL            = 0x00002000
	MB_HELP                 = 0x00004000
	MB_NOFOCUS              = 0x00008000
	MB_SETFOREGROUND        = 0x00010000
	MB_DEFAULT_DESKTOP_ONLY = 0x00020000
	MB_TOPMOST              = 0x00040000
	MB_RIGHT                = 0x00080000
	MB_RTLREADING           = 0x00100000
	MB_SERVICE_NOTIFICATION = 0x00200000
)

const (
	MOVEFILE_REPLACE_EXISTING      = 0x1
	MOVEFILE_COPY_ALLOWED          = 0x2
	MOVEFILE_DELAY_UNTIL_REBOOT    = 0x4
	MOVEFILE_WRITE_THROUGH         = 0x8
	MOVEFILE_CREATE_HARDLINK       = 0x10
	MOVEFILE_FAIL_IF_NOT_TRACKABLE = 0x20
)

const GAA_FLAG_INCLUDE_PREFIX = 0x00000010

const (
	IF_TYPE_OTHER              = 1
	IF_TYPE_ETHERNET_CSMACD    = 6
	IF_TYPE_ISO88025_TOKENRING = 9
	IF_TYPE_PPP                = 23
	IF_TYPE_SOFTWARE_LOOPBACK  = 24
	IF_TYPE_ATM                = 37
	IF_TYPE_IEEE80211          = 71
	IF_TYPE_TUNNEL             = 131
	IF_TYPE_IEEE1394           = 144
)

type SocketAddress struct {
	Sockaddr       *syscall.RawSockaddrAny
	SockaddrLength int32
}

// IP returns an IPv4 or IPv6 address, or nil if the underlying SocketAddress is neither.
func (addr *SocketAddress) IP() net.IP {
	if uintptr(addr.SockaddrLength) >= unsafe.Sizeof(RawSockaddrInet4{}) && addr.Sockaddr.Addr.Family == AF_INET {
		return (*RawSockaddrInet4)(unsafe.Pointer(addr.Sockaddr)).Addr[:]
	} else if uintptr(addr.SockaddrLength) >= unsafe.Sizeof(RawSockaddrInet6{}) && addr.Sockaddr.Addr.Family == AF_INET6 {
		return (*RawSockaddrInet6)(unsafe.Pointer(addr.Sockaddr)).Addr[:]
	}
	return nil
}

type IpAdapterUnicastAddress struct {
	Length             uint32
	Flags              uint32
	Next               *IpAdapterUnicastAddress
	Address            SocketAddress
	PrefixOrigin       int32
	SuffixOrigin       int32
	DadState           int32
	ValidLifetime      uint32
	PreferredLifetime  uint32
	LeaseLifetime      uint32
	OnLinkPrefixLength uint8
}

type IpAdapterAnycastAddress struct {
	Length  uint32
	Flags   uint32
	Next    *IpAdapterAnycastAddress
	Address SocketAddress
}

type IpAdapterMulticastAddress struct {
	Length  uint32
	Flags   uint32
	Next    *IpAdapterMulticastAddress
	Address SocketAddress
}

type IpAdapterDnsServerAdapter struct {
	Length   uint32
	Reserved uint32
	Next     *IpAdapterDnsServerAdapter
	Address  SocketAddress
}

type IpAdapterPrefix struct {
	Length       uint32
	Flags        uint32
	Next         *IpAdapterPrefix
	Address      SocketAddress
	PrefixLength uint32
}

type IpAdapterAddresses struct {
	Length                uint32
	IfIndex               uint32
	Next                  *IpAdapterAddresses
	AdapterName           *byte
	FirstUnicastAddress   *IpAdapterUnicastAddress
	FirstAnycastAddress   *IpAdapterAnycastAddress
	FirstMulticastAddress *IpAdapterMulticastAddress
	FirstDnsServerAddress *IpAdapterDnsServerAdapter
	DnsSuffix             *uint16
	Description           *uint16
	FriendlyName          *uint16
	PhysicalAddress       [syscall.MAX_ADAPTER_ADDRESS_LENGTH]byte
	PhysicalAddressLength uint32
	Flags                 uint32
	Mtu                   uint32
	IfType                uint32
	OperStatus            uint32
	Ipv6IfIndex           uint32
	ZoneIndices           [16]uint32
	FirstPrefix           *IpAdapterPrefix
	/* more fields might be present here. */
}

const (
	IfOperStatusUp             = 1
	IfOperStatusDown           = 2
	IfOperStatusTesting        = 3
	IfOperStatusUnknown        = 4
	IfOperStatusDormant        = 5
	IfOperStatusNotPresent     = 6
	IfOperStatusLowerLayerDown = 7
)

// Console related constants used for the mode parameter to SetConsoleMode. See
// https://docs.microsoft.com/en-us/windows/console/setconsolemode for details.

const (
	ENABLE_PROCESSED_INPUT        = 0x1
	ENABLE_LINE_INPUT             = 0x2
	ENABLE_ECHO_INPUT             = 0x4
	ENABLE_WINDOW_INPUT           = 0x8
	ENABLE_MOUSE_INPUT            = 0x10
	ENABLE_INSERT_MODE            = 0x20
	ENABLE_QUICK_EDIT_MODE        = 0x40
	ENABLE_EXTENDED_FLAGS         = 0x80
	ENABLE_AUTO_POSITION          = 0x100
	ENABLE_VIRTUAL_TERMINAL_INPUT = 0x200

	ENABLE_PROCESSED_OUTPUT            = 0x1
	ENABLE_WRAP_AT_EOL_OUTPUT          = 0x2
	ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x4
	DISABLE_NEWLINE_AUTO_RETURN        = 0x8
	ENABLE_LVB_GRID_WORLDWIDE          = 0x10
)

type Coord struct {
	X int16
	Y int16
}

type SmallRect struct {
	Left   int16
	Top    int16
	Right  int16
	Bottom int16
}

// Used with GetConsoleScreenBuffer to retrieve information about a console
// screen buffer. See
// https://docs.microsoft.com/en-us/windows/console/console-screen-buffer-info-str
// for details.

type ConsoleScreenBufferInfo struct {
	Size              Coord
	CursorPosition    Coord
	Attributes        uint16
	Window            SmallRect
	MaximumWindowSize Coord
}

const UNIX_PATH_MAX = 108 // defined in afunix.h

const (
	// flags for JOBOBJECT_BASIC_LIMIT_INFORMATION.LimitFlags
	JOB_OBJECT_LIMIT_ACTIVE_PROCESS             = 0x00000008
	JOB_OBJECT_LIMIT_AFFINITY                   = 0x00000010
	JOB_OBJECT_LIMIT_BREAKAWAY_OK               = 0x00000800
	JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
	JOB_OBJECT_LIMIT_JOB_MEMORY                 = 0x00000200
	JOB_OBJECT_LIMIT_JOB_TIME                   = 0x00000004
	JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE          = 0x00002000
	JOB_OBJECT_LIMIT_PRESERVE_JOB_TIME          = 0x00000040
	JOB_OBJECT_LIMIT_PRIORITY_CLASS             = 0x00000020
	JOB_OBJECT_LIMIT_PROCESS_MEMORY             = 0x00000100
	JOB_OBJECT_LIMIT_PROCESS_TIME               = 0x00000002
	JOB_OBJECT_LIMIT_SCHEDULING_CLASS           = 0x00000080
	JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK        = 0x00001000
	JOB_OBJECT_LIMIT_SUBSET_AFFINITY            = 0x00004000
	JOB_OBJECT_LIMIT_WORKINGSET                 = 0x00000001
)

type JOBOBJECT_BASIC_LIMIT_INFORMATION struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

type IO_COUNTERS struct {
	ReadOperationCount  uint64
	WriteOperationCount uint64
	OtherOperationCount uint64
	ReadTransferCount   uint64
	WriteTransferCount  uint64
	OtherTransferCount  uint64
}

type JOBOBJECT_EXTENDED_LIMIT_INFORMATION struct {
	BasicLimitInformation JOBOBJECT_BASIC_LIMIT_INFORMATION
	IoInfo                IO_COUNTERS
	ProcessMemoryLimit    uintptr
	JobMemoryLimit        uintptr
	PeakProcessMemoryUsed uintptr
	PeakJobMemoryUsed     uintptr
}

const (
	// UIRestrictionsClass
	JOB_OBJECT_UILIMIT_DESKTOP          = 0x00000040
	JOB_OBJECT_UILIMIT_DISPLAYSETTINGS  = 0x00000010
	JOB_OBJECT_UILIMIT_EXITWINDOWS      = 0x00000080
	JOB_OBJECT_UILIMIT_GLOBALATOMS      = 0x00000020
	JOB_OBJECT_UILIMIT_HANDLES          = 0x00000001
	JOB_OBJECT_UILIMIT_READCLIPBOARD    = 0x00000002
	JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS = 0x00000008
	JOB_OBJECT_UILIMIT_WRITECLIPBOARD   = 0x00000004
)

type JOBOBJECT_BASIC_UI_RESTRICTIONS struct {
	UIRestrictionsClass uint32
}

const (
	// JobObjectInformationClass
	JobObjectAssociateCompletionPortInformation = 7
	JobObjectBasicLimitInformation              = 2
	JobObjectBasicUIRestrictions                = 4
	JobObjectCpuRateControlInformation          = 15
	JobObjectEndOfJobTimeInformation            = 6
	JobObjectExtendedLimitInformation           = 9
	JobObjectGroupInformation                   = 11
	JobObjectGroupInformationEx                 = 14
	JobObjectLimitViolationInformation2         = 35
	JobObjectNetRateControlInformation          = 32
	JobObjectNotificationLimitInformation       = 12
	JobObjectNotificationLimitInformation2      = 34
	JobObjectSecurityLimitInformation           = 5
)

const (
	KF_FLAG_DEFAULT                          = 0x00000000
	KF_FLAG_FORCE_APP_DATA_REDIRECTION       = 0x00080000
	KF_FLAG_RETURN_FILTER_REDIRECTION_TARGET = 0x00040000
	KF_FLAG_FORCE_PACKAGE_REDIRECTION        = 0x00020000
	KF_FLAG_NO_PACKAGE_REDIRECTION           = 0x00010000
	KF_FLAG_FORCE_APPCONTAINER_REDIRECTION   = 0x00020000
	KF_FLAG_NO_APPCONTAINER_REDIRECTION      = 0x00010000
	KF_FLAG_CREATE                           = 0x00008000
	KF_FLAG_DONT_VERIFY                      = 0x00004000
	KF_FLAG_DONT_UNEXPAND                    = 0x00002000
	KF_FLAG_NO_ALIAS                         = 0x00001000
	KF_FLAG_INIT                             = 0x00000800
	KF_FLAG_DEFAULT_PATH                     = 0x00000400
	KF_FLAG_NOT_PARENT_RELATIVE              = 0x00000200
	KF_FLAG_SIMPLE_IDLIST                    = 0x00000100
	KF_FLAG_ALIAS_ONLY                       = 0x80000000
)

type OsVersionInfoEx struct {
	osVersionInfoSize uint32
	MajorVersion      uint32
	MinorVersion      uint32
	BuildNumber       uint32
	PlatformId        uint32
	CsdVersion        [128]uint16
	ServicePackMajor  uint16
	ServicePackMinor  uint16
	SuiteMask         uint16
	ProductType       byte
	_                 byte
}

const (
	EWX_LOGOFF          = 0x00000000
	EWX_SHUTDOWN        = 0x00000001
	EWX_REBOOT          = 0x00000002
	EWX_FORCE           = 0x00000004
	EWX_POWEROFF        = 0x00000008
	EWX_FORCEIFHUNG     = 0x00000010
	EWX_QUICKRESOLVE    = 0x00000020
	EWX_RESTARTAPPS     = 0x00000040
	EWX_HYBRID_SHUTDOWN = 0x00400000
	EWX_BOOTOPTIONS     = 0x01000000

	SHTDN_REASON_FLAG_COMMENT_REQUIRED          = 0x01000000
	SHTDN_REASON_FLAG_DIRTY_PROBLEM_ID_REQUIRED = 0x02000000
	SHTDN_REASON_FLAG_CLEAN_UI                  = 0x04000000
	SHTDN_REASON_FLAG_DIRTY_UI                  = 0x08000000
	SHTDN_REASON_FLAG_USER_DEFINED              = 0x40000000
	SHTDN_REASON_FLAG_PLANNED                   = 0x80000000
	SHTDN_REASON_MAJOR_OTHER                    = 0x00000000
	SHTDN_REASON_MAJOR_NONE                     = 0x00000000
	SHTDN_REASON_MAJOR_HARDWARE                 = 0x00010000
	SHTDN_REASON_MAJOR_OPERATINGSYSTEM          = 0x00020000
	SHTDN_REASON_MAJOR_SOFTWARE                 = 0x00030000
	SHTDN_REASON_MAJOR_APPLICATION              = 0x00040000
	SHTDN_REASON_MAJOR_SYSTEM                   = 0x00050000
	SHTDN_REASON_MAJOR_POWER                    = 0x00060000
	SHTDN_REASON_MAJOR_LEGACY_API               = 0x00070000
	SHTDN_REASON_MINOR_OTHER                    = 0x00000000
	SHTDN_REASON_MINOR_NONE                     = 0x000000ff
	SHTDN_REASON_MINOR_MAINTENANCE              = 0x00000001
	SHTDN_REASON_MINOR_INSTALLATION             = 0x00000002
	SHTDN_REASON_MINOR_UPGRADE                  = 0x00000003
	SHTDN_REASON_MINOR_RECONFIG                 = 0x00000004
	SHTDN_REASON_MINOR_HUNG                     = 0x00000005
	SHTDN_REASON_MINOR_UNSTABLE                 = 0x00000006
	SHTDN_REASON_MINOR_DISK                     = 0x00000007
	SHTDN_REASON_MINOR_PROCESSOR                = 0x00000008
	SHTDN_REASON_MINOR_NETWORKCARD              = 0x00000009
	SHTDN_REASON_MINOR_POWER_SUPPLY             = 0x0000000a
	SHTDN_REASON_MINOR_CORDUNPLUGGED            = 0x0000000b
	SHTDN_REASON_MINOR_ENVIRONMENT              = 0x0000000c
	SHTDN_REASON_MINOR_HARDWARE_DRIVER          = 0x0000000d
	SHTDN_REASON_MINOR_OTHERDRIVER              = 0x0000000e
	SHTDN_REASON_MINOR_BLUESCREEN               = 0x0000000F
	SHTDN_REASON_MINOR_SERVICEPACK              = 0x00000010
	SHTDN_REASON_MINOR_HOTFIX                   = 0x00000011
	SHTDN_REASON_MINOR_SECURITYFIX              = 0x00000012
	SHTDN_REASON_MINOR_SECURITY                 = 0x00000013
	SHTDN_REASON_MINOR_NETWORK_CONNECTIVITY     = 0x00000014
	SHTDN_REASON_MINOR_WMI                      = 0x00000015
	SHTDN_REASON_MINOR_SERVICEPACK_UNINSTALL    = 0x00000016
	SHTDN_REASON_MINOR_HOTFIX_UNINSTALL         = 0x00000017
	SHTDN_REASON_MINOR_SECURITYFIX_UNINSTALL    = 0x00000018
	SHTDN_REASON_MINOR_MMC                      = 0x00000019
	SHTDN_REASON_MINOR_SYSTEMRESTORE            = 0x0000001a
	SHTDN_REASON_MINOR_TERMSRV                  = 0x00000020
	SHTDN_REASON_MINOR_DC_PROMOTION             = 0x00000021
	SHTDN_REASON_MINOR_DC_DEMOTION              = 0x00000022
	SHTDN_REASON_UNKNOWN                        = SHTDN_REASON_MINOR_NONE
	SHTDN_REASON_LEGACY_API                     = SHTDN_REASON_MAJOR_LEGACY_API | SHTDN_REASON_FLAG_PLANNED
	SHTDN_REASON_VALID_BIT_MASK                 = 0xc0ffffff

	SHUTDOWN_NORETRY = 0x1
)

// Flags used for GetModuleHandleEx
const (
	GET_MODULE_HANDLE_EX_FLAG_PIN                = 1
	GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT = 2
	GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS       = 4
)

// MUI function flag values
const (
	MUI_LANGUAGE_ID                    = 0x4
	MUI_LANGUAGE_NAME                  = 0x8
	MUI_MERGE_SYSTEM_FALLBACK          = 0x10
	MUI_MERGE_USER_FALLBACK            = 0x20
	MUI_UI_FALLBACK                    = MUI_MERGE_SYSTEM_FALLBACK | MUI_MERGE_USER_FALLBACK
	MUI_THREAD_LANGUAGES               = 0x40
	MUI_CONSOLE_FILTER                 = 0x100
	MUI_COMPLEX_SCRIPT_FILTER          = 0x200
	MUI_RESET_FILTERS                  = 0x001
	MUI_USER_PREFERRED_UI_LANGUAGES    = 0x10
	MUI_USE_INSTALLED_LANGUAGES        = 0x20
	MUI_USE_SEARCH_ALL_LANGUAGES       = 0x40
	MUI_LANG_NEUTRAL_PE_FILE           = 0x100
	MUI_NON_LANG_NEUTRAL_FILE          = 0x200
	MUI_MACHINE_LANGUAGE_SETTINGS      = 0x400
	MUI_FILETYPE_NOT_LANGUAGE_NEUTRAL  = 0x001
	MUI_FILETYPE_LANGUAGE_NEUTRAL_MAIN = 0x002
	MUI_FILETYPE_LANGUAGE_NEUTRAL_MUI  = 0x004
	MUI_QUERY_TYPE                     = 0x001
	MUI_QUERY_CHECKSUM                 = 0x002
	MUI_QUERY_LANGUAGE_NAME            = 0x004
	MUI_QUERY_RESOURCE_TYPES           = 0x008
	MUI_FILEINFO_VERSION               = 0x001

	MUI_FULL_LANGUAGE      = 0x01
	MUI_PARTIAL_LANGUAGE   = 0x02
	MUI_LIP_LANGUAGE       = 0x04
	MUI_LANGUAGE_INSTALLED = 0x20
	MUI_LANGUAGE_LICENSED  = 0x40
)
