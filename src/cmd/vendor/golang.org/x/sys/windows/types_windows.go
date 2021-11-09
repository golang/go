// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"net"
	"syscall"
	"unsafe"
)

// NTStatus corresponds with NTSTATUS, error values returned by ntdll.dll and
// other native functions.
type NTStatus uint32

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
	FILE_READ_DATA        = 0x00000001
	FILE_READ_ATTRIBUTES  = 0x00000080
	FILE_READ_EA          = 0x00000008
	FILE_WRITE_DATA       = 0x00000002
	FILE_WRITE_ATTRIBUTES = 0x00000100
	FILE_WRITE_EA         = 0x00000010
	FILE_APPEND_DATA      = 0x00000004
	FILE_EXECUTE          = 0x00000020

	FILE_GENERIC_READ    = STANDARD_RIGHTS_READ | FILE_READ_DATA | FILE_READ_ATTRIBUTES | FILE_READ_EA | SYNCHRONIZE
	FILE_GENERIC_WRITE   = STANDARD_RIGHTS_WRITE | FILE_WRITE_DATA | FILE_WRITE_ATTRIBUTES | FILE_WRITE_EA | FILE_APPEND_DATA | SYNCHRONIZE
	FILE_GENERIC_EXECUTE = STANDARD_RIGHTS_EXECUTE | FILE_READ_ATTRIBUTES | FILE_EXECUTE | SYNCHRONIZE

	FILE_LIST_DIRECTORY = 0x00000001
	FILE_TRAVERSE       = 0x00000020

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
	// attributes for ProcThreadAttributeList
	PROC_THREAD_ATTRIBUTE_PARENT_PROCESS    = 0x00020000
	PROC_THREAD_ATTRIBUTE_HANDLE_LIST       = 0x00020002
	PROC_THREAD_ATTRIBUTE_GROUP_AFFINITY    = 0x00030003
	PROC_THREAD_ATTRIBUTE_PREFERRED_NODE    = 0x00020004
	PROC_THREAD_ATTRIBUTE_IDEAL_PROCESSOR   = 0x00030005
	PROC_THREAD_ATTRIBUTE_MITIGATION_POLICY = 0x00020007
	PROC_THREAD_ATTRIBUTE_UMS_THREAD        = 0x00030006
	PROC_THREAD_ATTRIBUTE_PROTECTION_LEVEL  = 0x0002000b
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
	// flags for EnumProcessModulesEx
	LIST_MODULES_32BIT   = 0x01
	LIST_MODULES_64BIT   = 0x02
	LIST_MODULES_ALL     = 0x03
	LIST_MODULES_DEFAULT = 0x00
)

const (
	// filters for ReadDirectoryChangesW and FindFirstChangeNotificationW
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
	/* certenrolld_begin -- PROV_RSA_*/
	PROV_RSA_FULL      = 1
	PROV_RSA_SIG       = 2
	PROV_DSS           = 3
	PROV_FORTEZZA      = 4
	PROV_MS_EXCHANGE   = 5
	PROV_SSL           = 6
	PROV_RSA_SCHANNEL  = 12
	PROV_DSS_DH        = 13
	PROV_EC_ECDSA_SIG  = 14
	PROV_EC_ECNRA_SIG  = 15
	PROV_EC_ECDSA_FULL = 16
	PROV_EC_ECNRA_FULL = 17
	PROV_DH_SCHANNEL   = 18
	PROV_SPYRUS_LYNKS  = 20
	PROV_RNG           = 21
	PROV_INTEL_SEC     = 22
	PROV_REPLACE_OWF   = 23
	PROV_RSA_AES       = 24

	/* dwFlags definitions for CryptAcquireContext */
	CRYPT_VERIFYCONTEXT              = 0xF0000000
	CRYPT_NEWKEYSET                  = 0x00000008
	CRYPT_DELETEKEYSET               = 0x00000010
	CRYPT_MACHINE_KEYSET             = 0x00000020
	CRYPT_SILENT                     = 0x00000040
	CRYPT_DEFAULT_CONTAINER_OPTIONAL = 0x00000080

	/* Flags for PFXImportCertStore */
	CRYPT_EXPORTABLE                   = 0x00000001
	CRYPT_USER_PROTECTED               = 0x00000002
	CRYPT_USER_KEYSET                  = 0x00001000
	PKCS12_PREFER_CNG_KSP              = 0x00000100
	PKCS12_ALWAYS_CNG_KSP              = 0x00000200
	PKCS12_ALLOW_OVERWRITE_KEY         = 0x00004000
	PKCS12_NO_PERSIST_KEY              = 0x00008000
	PKCS12_INCLUDE_EXTENDED_PROPERTIES = 0x00000010

	/* Flags for CryptAcquireCertificatePrivateKey */
	CRYPT_ACQUIRE_CACHE_FLAG             = 0x00000001
	CRYPT_ACQUIRE_USE_PROV_INFO_FLAG     = 0x00000002
	CRYPT_ACQUIRE_COMPARE_KEY_FLAG       = 0x00000004
	CRYPT_ACQUIRE_NO_HEALING             = 0x00000008
	CRYPT_ACQUIRE_SILENT_FLAG            = 0x00000040
	CRYPT_ACQUIRE_WINDOW_HANDLE_FLAG     = 0x00000080
	CRYPT_ACQUIRE_NCRYPT_KEY_FLAGS_MASK  = 0x00070000
	CRYPT_ACQUIRE_ALLOW_NCRYPT_KEY_FLAG  = 0x00010000
	CRYPT_ACQUIRE_PREFER_NCRYPT_KEY_FLAG = 0x00020000
	CRYPT_ACQUIRE_ONLY_NCRYPT_KEY_FLAG   = 0x00040000

	/* pdwKeySpec for CryptAcquireCertificatePrivateKey */
	AT_KEYEXCHANGE       = 1
	AT_SIGNATURE         = 2
	CERT_NCRYPT_KEY_SPEC = 0xFFFFFFFF

	/* Default usage match type is AND with value zero */
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

	/* Certificate Information Flags */
	CERT_INFO_VERSION_FLAG                 = 1
	CERT_INFO_SERIAL_NUMBER_FLAG           = 2
	CERT_INFO_SIGNATURE_ALGORITHM_FLAG     = 3
	CERT_INFO_ISSUER_FLAG                  = 4
	CERT_INFO_NOT_BEFORE_FLAG              = 5
	CERT_INFO_NOT_AFTER_FLAG               = 6
	CERT_INFO_SUBJECT_FLAG                 = 7
	CERT_INFO_SUBJECT_PUBLIC_KEY_INFO_FLAG = 8
	CERT_INFO_ISSUER_UNIQUE_ID_FLAG        = 9
	CERT_INFO_SUBJECT_UNIQUE_ID_FLAG       = 10
	CERT_INFO_EXTENSION_FLAG               = 11

	/* dwFindType for CertFindCertificateInStore  */
	CERT_COMPARE_MASK                     = 0xFFFF
	CERT_COMPARE_SHIFT                    = 16
	CERT_COMPARE_ANY                      = 0
	CERT_COMPARE_SHA1_HASH                = 1
	CERT_COMPARE_NAME                     = 2
	CERT_COMPARE_ATTR                     = 3
	CERT_COMPARE_MD5_HASH                 = 4
	CERT_COMPARE_PROPERTY                 = 5
	CERT_COMPARE_PUBLIC_KEY               = 6
	CERT_COMPARE_HASH                     = CERT_COMPARE_SHA1_HASH
	CERT_COMPARE_NAME_STR_A               = 7
	CERT_COMPARE_NAME_STR_W               = 8
	CERT_COMPARE_KEY_SPEC                 = 9
	CERT_COMPARE_ENHKEY_USAGE             = 10
	CERT_COMPARE_CTL_USAGE                = CERT_COMPARE_ENHKEY_USAGE
	CERT_COMPARE_SUBJECT_CERT             = 11
	CERT_COMPARE_ISSUER_OF                = 12
	CERT_COMPARE_EXISTING                 = 13
	CERT_COMPARE_SIGNATURE_HASH           = 14
	CERT_COMPARE_KEY_IDENTIFIER           = 15
	CERT_COMPARE_CERT_ID                  = 16
	CERT_COMPARE_CROSS_CERT_DIST_POINTS   = 17
	CERT_COMPARE_PUBKEY_MD5_HASH          = 18
	CERT_COMPARE_SUBJECT_INFO_ACCESS      = 19
	CERT_COMPARE_HASH_STR                 = 20
	CERT_COMPARE_HAS_PRIVATE_KEY          = 21
	CERT_FIND_ANY                         = (CERT_COMPARE_ANY << CERT_COMPARE_SHIFT)
	CERT_FIND_SHA1_HASH                   = (CERT_COMPARE_SHA1_HASH << CERT_COMPARE_SHIFT)
	CERT_FIND_MD5_HASH                    = (CERT_COMPARE_MD5_HASH << CERT_COMPARE_SHIFT)
	CERT_FIND_SIGNATURE_HASH              = (CERT_COMPARE_SIGNATURE_HASH << CERT_COMPARE_SHIFT)
	CERT_FIND_KEY_IDENTIFIER              = (CERT_COMPARE_KEY_IDENTIFIER << CERT_COMPARE_SHIFT)
	CERT_FIND_HASH                        = CERT_FIND_SHA1_HASH
	CERT_FIND_PROPERTY                    = (CERT_COMPARE_PROPERTY << CERT_COMPARE_SHIFT)
	CERT_FIND_PUBLIC_KEY                  = (CERT_COMPARE_PUBLIC_KEY << CERT_COMPARE_SHIFT)
	CERT_FIND_SUBJECT_NAME                = (CERT_COMPARE_NAME<<CERT_COMPARE_SHIFT | CERT_INFO_SUBJECT_FLAG)
	CERT_FIND_SUBJECT_ATTR                = (CERT_COMPARE_ATTR<<CERT_COMPARE_SHIFT | CERT_INFO_SUBJECT_FLAG)
	CERT_FIND_ISSUER_NAME                 = (CERT_COMPARE_NAME<<CERT_COMPARE_SHIFT | CERT_INFO_ISSUER_FLAG)
	CERT_FIND_ISSUER_ATTR                 = (CERT_COMPARE_ATTR<<CERT_COMPARE_SHIFT | CERT_INFO_ISSUER_FLAG)
	CERT_FIND_SUBJECT_STR_A               = (CERT_COMPARE_NAME_STR_A<<CERT_COMPARE_SHIFT | CERT_INFO_SUBJECT_FLAG)
	CERT_FIND_SUBJECT_STR_W               = (CERT_COMPARE_NAME_STR_W<<CERT_COMPARE_SHIFT | CERT_INFO_SUBJECT_FLAG)
	CERT_FIND_SUBJECT_STR                 = CERT_FIND_SUBJECT_STR_W
	CERT_FIND_ISSUER_STR_A                = (CERT_COMPARE_NAME_STR_A<<CERT_COMPARE_SHIFT | CERT_INFO_ISSUER_FLAG)
	CERT_FIND_ISSUER_STR_W                = (CERT_COMPARE_NAME_STR_W<<CERT_COMPARE_SHIFT | CERT_INFO_ISSUER_FLAG)
	CERT_FIND_ISSUER_STR                  = CERT_FIND_ISSUER_STR_W
	CERT_FIND_KEY_SPEC                    = (CERT_COMPARE_KEY_SPEC << CERT_COMPARE_SHIFT)
	CERT_FIND_ENHKEY_USAGE                = (CERT_COMPARE_ENHKEY_USAGE << CERT_COMPARE_SHIFT)
	CERT_FIND_CTL_USAGE                   = CERT_FIND_ENHKEY_USAGE
	CERT_FIND_SUBJECT_CERT                = (CERT_COMPARE_SUBJECT_CERT << CERT_COMPARE_SHIFT)
	CERT_FIND_ISSUER_OF                   = (CERT_COMPARE_ISSUER_OF << CERT_COMPARE_SHIFT)
	CERT_FIND_EXISTING                    = (CERT_COMPARE_EXISTING << CERT_COMPARE_SHIFT)
	CERT_FIND_CERT_ID                     = (CERT_COMPARE_CERT_ID << CERT_COMPARE_SHIFT)
	CERT_FIND_CROSS_CERT_DIST_POINTS      = (CERT_COMPARE_CROSS_CERT_DIST_POINTS << CERT_COMPARE_SHIFT)
	CERT_FIND_PUBKEY_MD5_HASH             = (CERT_COMPARE_PUBKEY_MD5_HASH << CERT_COMPARE_SHIFT)
	CERT_FIND_SUBJECT_INFO_ACCESS         = (CERT_COMPARE_SUBJECT_INFO_ACCESS << CERT_COMPARE_SHIFT)
	CERT_FIND_HASH_STR                    = (CERT_COMPARE_HASH_STR << CERT_COMPARE_SHIFT)
	CERT_FIND_HAS_PRIVATE_KEY             = (CERT_COMPARE_HAS_PRIVATE_KEY << CERT_COMPARE_SHIFT)
	CERT_FIND_OPTIONAL_ENHKEY_USAGE_FLAG  = 0x1
	CERT_FIND_EXT_ONLY_ENHKEY_USAGE_FLAG  = 0x2
	CERT_FIND_PROP_ONLY_ENHKEY_USAGE_FLAG = 0x4
	CERT_FIND_NO_ENHKEY_USAGE_FLAG        = 0x8
	CERT_FIND_OR_ENHKEY_USAGE_FLAG        = 0x10
	CERT_FIND_VALID_ENHKEY_USAGE_FLAG     = 0x20
	CERT_FIND_OPTIONAL_CTL_USAGE_FLAG     = CERT_FIND_OPTIONAL_ENHKEY_USAGE_FLAG
	CERT_FIND_EXT_ONLY_CTL_USAGE_FLAG     = CERT_FIND_EXT_ONLY_ENHKEY_USAGE_FLAG
	CERT_FIND_PROP_ONLY_CTL_USAGE_FLAG    = CERT_FIND_PROP_ONLY_ENHKEY_USAGE_FLAG
	CERT_FIND_NO_CTL_USAGE_FLAG           = CERT_FIND_NO_ENHKEY_USAGE_FLAG
	CERT_FIND_OR_CTL_USAGE_FLAG           = CERT_FIND_OR_ENHKEY_USAGE_FLAG
	CERT_FIND_VALID_CTL_USAGE_FLAG        = CERT_FIND_VALID_ENHKEY_USAGE_FLAG

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

	/* flag for dwFindType CertFindChainInStore  */
	CERT_CHAIN_FIND_BY_ISSUER = 1

	/* dwFindFlags for CertFindChainInStore when dwFindType == CERT_CHAIN_FIND_BY_ISSUER */
	CERT_CHAIN_FIND_BY_ISSUER_COMPARE_KEY_FLAG    = 0x0001
	CERT_CHAIN_FIND_BY_ISSUER_COMPLEX_CHAIN_FLAG  = 0x0002
	CERT_CHAIN_FIND_BY_ISSUER_CACHE_ONLY_URL_FLAG = 0x0004
	CERT_CHAIN_FIND_BY_ISSUER_LOCAL_MACHINE_FLAG  = 0x0008
	CERT_CHAIN_FIND_BY_ISSUER_NO_KEY_FLAG         = 0x4000
	CERT_CHAIN_FIND_BY_ISSUER_CACHE_ONLY_FLAG     = 0x8000

	/* Certificate Store close flags */
	CERT_CLOSE_STORE_FORCE_FLAG = 0x00000001
	CERT_CLOSE_STORE_CHECK_FLAG = 0x00000002

	/* CryptQueryObject object type */
	CERT_QUERY_OBJECT_FILE = 1
	CERT_QUERY_OBJECT_BLOB = 2

	/* CryptQueryObject content type flags */
	CERT_QUERY_CONTENT_CERT                    = 1
	CERT_QUERY_CONTENT_CTL                     = 2
	CERT_QUERY_CONTENT_CRL                     = 3
	CERT_QUERY_CONTENT_SERIALIZED_STORE        = 4
	CERT_QUERY_CONTENT_SERIALIZED_CERT         = 5
	CERT_QUERY_CONTENT_SERIALIZED_CTL          = 6
	CERT_QUERY_CONTENT_SERIALIZED_CRL          = 7
	CERT_QUERY_CONTENT_PKCS7_SIGNED            = 8
	CERT_QUERY_CONTENT_PKCS7_UNSIGNED          = 9
	CERT_QUERY_CONTENT_PKCS7_SIGNED_EMBED      = 10
	CERT_QUERY_CONTENT_PKCS10                  = 11
	CERT_QUERY_CONTENT_PFX                     = 12
	CERT_QUERY_CONTENT_CERT_PAIR               = 13
	CERT_QUERY_CONTENT_PFX_AND_LOAD            = 14
	CERT_QUERY_CONTENT_FLAG_CERT               = (1 << CERT_QUERY_CONTENT_CERT)
	CERT_QUERY_CONTENT_FLAG_CTL                = (1 << CERT_QUERY_CONTENT_CTL)
	CERT_QUERY_CONTENT_FLAG_CRL                = (1 << CERT_QUERY_CONTENT_CRL)
	CERT_QUERY_CONTENT_FLAG_SERIALIZED_STORE   = (1 << CERT_QUERY_CONTENT_SERIALIZED_STORE)
	CERT_QUERY_CONTENT_FLAG_SERIALIZED_CERT    = (1 << CERT_QUERY_CONTENT_SERIALIZED_CERT)
	CERT_QUERY_CONTENT_FLAG_SERIALIZED_CTL     = (1 << CERT_QUERY_CONTENT_SERIALIZED_CTL)
	CERT_QUERY_CONTENT_FLAG_SERIALIZED_CRL     = (1 << CERT_QUERY_CONTENT_SERIALIZED_CRL)
	CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED       = (1 << CERT_QUERY_CONTENT_PKCS7_SIGNED)
	CERT_QUERY_CONTENT_FLAG_PKCS7_UNSIGNED     = (1 << CERT_QUERY_CONTENT_PKCS7_UNSIGNED)
	CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED = (1 << CERT_QUERY_CONTENT_PKCS7_SIGNED_EMBED)
	CERT_QUERY_CONTENT_FLAG_PKCS10             = (1 << CERT_QUERY_CONTENT_PKCS10)
	CERT_QUERY_CONTENT_FLAG_PFX                = (1 << CERT_QUERY_CONTENT_PFX)
	CERT_QUERY_CONTENT_FLAG_CERT_PAIR          = (1 << CERT_QUERY_CONTENT_CERT_PAIR)
	CERT_QUERY_CONTENT_FLAG_PFX_AND_LOAD       = (1 << CERT_QUERY_CONTENT_PFX_AND_LOAD)
	CERT_QUERY_CONTENT_FLAG_ALL                = (CERT_QUERY_CONTENT_FLAG_CERT | CERT_QUERY_CONTENT_FLAG_CTL | CERT_QUERY_CONTENT_FLAG_CRL | CERT_QUERY_CONTENT_FLAG_SERIALIZED_STORE | CERT_QUERY_CONTENT_FLAG_SERIALIZED_CERT | CERT_QUERY_CONTENT_FLAG_SERIALIZED_CTL | CERT_QUERY_CONTENT_FLAG_SERIALIZED_CRL | CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED | CERT_QUERY_CONTENT_FLAG_PKCS7_UNSIGNED | CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED | CERT_QUERY_CONTENT_FLAG_PKCS10 | CERT_QUERY_CONTENT_FLAG_PFX | CERT_QUERY_CONTENT_FLAG_CERT_PAIR)
	CERT_QUERY_CONTENT_FLAG_ALL_ISSUER_CERT    = (CERT_QUERY_CONTENT_FLAG_CERT | CERT_QUERY_CONTENT_FLAG_SERIALIZED_STORE | CERT_QUERY_CONTENT_FLAG_SERIALIZED_CERT | CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED | CERT_QUERY_CONTENT_FLAG_PKCS7_UNSIGNED)

	/* CryptQueryObject format type flags */
	CERT_QUERY_FORMAT_BINARY                     = 1
	CERT_QUERY_FORMAT_BASE64_ENCODED             = 2
	CERT_QUERY_FORMAT_ASN_ASCII_HEX_ENCODED      = 3
	CERT_QUERY_FORMAT_FLAG_BINARY                = (1 << CERT_QUERY_FORMAT_BINARY)
	CERT_QUERY_FORMAT_FLAG_BASE64_ENCODED        = (1 << CERT_QUERY_FORMAT_BASE64_ENCODED)
	CERT_QUERY_FORMAT_FLAG_ASN_ASCII_HEX_ENCODED = (1 << CERT_QUERY_FORMAT_ASN_ASCII_HEX_ENCODED)
	CERT_QUERY_FORMAT_FLAG_ALL                   = (CERT_QUERY_FORMAT_FLAG_BINARY | CERT_QUERY_FORMAT_FLAG_BASE64_ENCODED | CERT_QUERY_FORMAT_FLAG_ASN_ASCII_HEX_ENCODED)

	/* CertGetNameString name types */
	CERT_NAME_EMAIL_TYPE            = 1
	CERT_NAME_RDN_TYPE              = 2
	CERT_NAME_ATTR_TYPE             = 3
	CERT_NAME_SIMPLE_DISPLAY_TYPE   = 4
	CERT_NAME_FRIENDLY_DISPLAY_TYPE = 5
	CERT_NAME_DNS_TYPE              = 6
	CERT_NAME_URL_TYPE              = 7
	CERT_NAME_UPN_TYPE              = 8

	/* CertGetNameString flags */
	CERT_NAME_ISSUER_FLAG              = 0x1
	CERT_NAME_DISABLE_IE4_UTF8_FLAG    = 0x10000
	CERT_NAME_SEARCH_ALL_NAMES_FLAG    = 0x2
	CERT_NAME_STR_ENABLE_PUNYCODE_FLAG = 0x00200000

	/* AuthType values for SSLExtraCertChainPolicyPara struct */
	AUTHTYPE_CLIENT = 1
	AUTHTYPE_SERVER = 2

	/* Checks values for SSLExtraCertChainPolicyPara struct */
	SECURITY_FLAG_IGNORE_REVOCATION        = 0x00000080
	SECURITY_FLAG_IGNORE_UNKNOWN_CA        = 0x00000100
	SECURITY_FLAG_IGNORE_WRONG_USAGE       = 0x00000200
	SECURITY_FLAG_IGNORE_CERT_CN_INVALID   = 0x00001000
	SECURITY_FLAG_IGNORE_CERT_DATE_INVALID = 0x00002000

	/* Flags for Crypt[Un]ProtectData */
	CRYPTPROTECT_UI_FORBIDDEN      = 0x1
	CRYPTPROTECT_LOCAL_MACHINE     = 0x4
	CRYPTPROTECT_CRED_SYNC         = 0x8
	CRYPTPROTECT_AUDIT             = 0x10
	CRYPTPROTECT_NO_RECOVERY       = 0x20
	CRYPTPROTECT_VERIFY_PROTECTION = 0x40
	CRYPTPROTECT_CRED_REGENERATE   = 0x80

	/* Flags for CryptProtectPromptStruct */
	CRYPTPROTECT_PROMPT_ON_UNPROTECT   = 1
	CRYPTPROTECT_PROMPT_ON_PROTECT     = 2
	CRYPTPROTECT_PROMPT_RESERVED       = 4
	CRYPTPROTECT_PROMPT_STRONG         = 8
	CRYPTPROTECT_PROMPT_REQUIRE_STRONG = 16
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

/* wintrust.h constants for WinVerifyTrustEx */
const (
	WTD_UI_ALL    = 1
	WTD_UI_NONE   = 2
	WTD_UI_NOBAD  = 3
	WTD_UI_NOGOOD = 4

	WTD_REVOKE_NONE       = 0
	WTD_REVOKE_WHOLECHAIN = 1

	WTD_CHOICE_FILE    = 1
	WTD_CHOICE_CATALOG = 2
	WTD_CHOICE_BLOB    = 3
	WTD_CHOICE_SIGNER  = 4
	WTD_CHOICE_CERT    = 5

	WTD_STATEACTION_IGNORE           = 0x00000000
	WTD_STATEACTION_VERIFY           = 0x00000001
	WTD_STATEACTION_CLOSE            = 0x00000002
	WTD_STATEACTION_AUTO_CACHE       = 0x00000003
	WTD_STATEACTION_AUTO_CACHE_FLUSH = 0x00000004

	WTD_USE_IE4_TRUST_FLAG                  = 0x1
	WTD_NO_IE4_CHAIN_FLAG                   = 0x2
	WTD_NO_POLICY_USAGE_FLAG                = 0x4
	WTD_REVOCATION_CHECK_NONE               = 0x10
	WTD_REVOCATION_CHECK_END_CERT           = 0x20
	WTD_REVOCATION_CHECK_CHAIN              = 0x40
	WTD_REVOCATION_CHECK_CHAIN_EXCLUDE_ROOT = 0x80
	WTD_SAFER_FLAG                          = 0x100
	WTD_HASH_ONLY_FLAG                      = 0x200
	WTD_USE_DEFAULT_OSVER_CHECK             = 0x400
	WTD_LIFETIME_SIGNING_FLAG               = 0x800
	WTD_CACHE_ONLY_URL_RETRIEVAL            = 0x1000
	WTD_DISABLE_MD2_MD4                     = 0x2000
	WTD_MOTW                                = 0x4000

	WTD_UICONTEXT_EXECUTE = 0
	WTD_UICONTEXT_INSTALL = 1
)

var (
	OID_PKIX_KP_SERVER_AUTH = []byte("1.3.6.1.5.5.7.3.1\x00")
	OID_SERVER_GATED_CRYPTO = []byte("1.3.6.1.4.1.311.10.3.3\x00")
	OID_SGC_NETSCAPE        = []byte("2.16.840.1.113730.4.1\x00")

	WINTRUST_ACTION_GENERIC_VERIFY_V2 = GUID{
		Data1: 0xaac56b,
		Data2: 0xcd44,
		Data3: 0x11d0,
		Data4: [8]byte{0x8c, 0xc2, 0x0, 0xc0, 0x4f, 0xc2, 0x95, 0xee},
	}
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

	// The Microsoft documentation for this struct¹ describes three additional
	// fields: dwFileType, dwCreatorType, and wFinderFlags. However, those fields
	// are empirically only present in the macOS port of the Win32 API,² and thus
	// not needed for binaries built for Windows.
	//
	// ¹ https://docs.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-win32_find_dataw describe
	// ² https://golang.org/issue/42637#issuecomment-760715755.
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

type StartupInfoEx struct {
	StartupInfo
	ProcThreadAttributeList *ProcThreadAttributeList
}

// ProcThreadAttributeList is a placeholder type to represent a PROC_THREAD_ATTRIBUTE_LIST.
//
// To create a *ProcThreadAttributeList, use NewProcThreadAttributeList, update
// it with ProcThreadAttributeListContainer.Update, free its memory using
// ProcThreadAttributeListContainer.Delete, and access the list itself using
// ProcThreadAttributeListContainer.List.
type ProcThreadAttributeList struct{}

type ProcThreadAttributeListContainer struct {
	data            *ProcThreadAttributeList
	heapAllocations []uintptr
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

	IP_HDRINCL         = 0x2
	IP_TOS             = 0x3
	IP_TTL             = 0x4
	IP_MULTICAST_IF    = 0x9
	IP_MULTICAST_TTL   = 0xa
	IP_MULTICAST_LOOP  = 0xb
	IP_ADD_MEMBERSHIP  = 0xc
	IP_DROP_MEMBERSHIP = 0xd
	IP_PKTINFO         = 0x13

	IPV6_V6ONLY         = 0x1b
	IPV6_UNICAST_HOPS   = 0x4
	IPV6_MULTICAST_IF   = 0x9
	IPV6_MULTICAST_HOPS = 0xa
	IPV6_MULTICAST_LOOP = 0xb
	IPV6_JOIN_GROUP     = 0xc
	IPV6_LEAVE_GROUP    = 0xd
	IPV6_PKTINFO        = 0x13

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

// Flags for WSASocket
const (
	WSA_FLAG_OVERLAPPED             = 0x01
	WSA_FLAG_MULTIPOINT_C_ROOT      = 0x02
	WSA_FLAG_MULTIPOINT_C_LEAF      = 0x04
	WSA_FLAG_MULTIPOINT_D_ROOT      = 0x08
	WSA_FLAG_MULTIPOINT_D_LEAF      = 0x10
	WSA_FLAG_ACCESS_SYSTEM_SECURITY = 0x40
	WSA_FLAG_NO_HANDLE_INHERIT      = 0x80
	WSA_FLAG_REGISTERED_IO          = 0x100
)

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
	Version              uint32
	SerialNumber         CryptIntegerBlob
	SignatureAlgorithm   CryptAlgorithmIdentifier
	Issuer               CertNameBlob
	NotBefore            Filetime
	NotAfter             Filetime
	Subject              CertNameBlob
	SubjectPublicKeyInfo CertPublicKeyInfo
	IssuerUniqueId       CryptBitBlob
	SubjectUniqueId      CryptBitBlob
	CountExtensions      uint32
	Extensions           *CertExtension
}

type CertExtension struct {
	ObjId    *byte
	Critical int32
	Value    CryptObjidBlob
}

type CryptAlgorithmIdentifier struct {
	ObjId      *byte
	Parameters CryptObjidBlob
}

type CertPublicKeyInfo struct {
	Algorithm CryptAlgorithmIdentifier
	PublicKey CryptBitBlob
}

type DataBlob struct {
	Size uint32
	Data *byte
}
type CryptIntegerBlob DataBlob
type CryptUintBlob DataBlob
type CryptObjidBlob DataBlob
type CertNameBlob DataBlob
type CertRdnValueBlob DataBlob
type CertBlob DataBlob
type CrlBlob DataBlob
type CryptDataBlob DataBlob
type CryptHashBlob DataBlob
type CryptDigestBlob DataBlob
type CryptDerBlob DataBlob
type CryptAttrBlob DataBlob

type CryptBitBlob struct {
	Size       uint32
	Data       *byte
	UnusedBits uint32
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

type CertPolicyInfo struct {
	Identifier      *byte
	CountQualifiers uint32
	Qualifiers      *CertPolicyQualifierInfo
}

type CertPoliciesInfo struct {
	Count       uint32
	PolicyInfos *CertPolicyInfo
}

type CertPolicyQualifierInfo struct {
	// Not implemented
}

type CertStrongSignPara struct {
	Size                      uint32
	InfoChoice                uint32
	InfoOrSerializedInfoOrOID unsafe.Pointer
}

type CryptProtectPromptStruct struct {
	Size        uint32
	PromptFlags uint32
	App         HWND
	Prompt      *uint16
}

type CertChainFindByIssuerPara struct {
	Size                   uint32
	UsageIdentifier        *byte
	KeySpec                uint32
	AcquirePrivateKeyFlags uint32
	IssuerCount            uint32
	Issuer                 Pointer
	FindCallback           Pointer
	FindArg                Pointer
	IssuerChainIndex       *uint32
	IssuerElementIndex     *uint32
}

type WinTrustData struct {
	Size                            uint32
	PolicyCallbackData              uintptr
	SIPClientData                   uintptr
	UIChoice                        uint32
	RevocationChecks                uint32
	UnionChoice                     uint32
	FileOrCatalogOrBlobOrSgnrOrCert unsafe.Pointer
	StateAction                     uint32
	StateData                       Handle
	URLReference                    *uint16
	ProvFlags                       uint32
	UIContext                       uint32
	SignatureSettings               *WinTrustSignatureSettings
}

type WinTrustFileInfo struct {
	Size         uint32
	FilePath     *uint16
	File         Handle
	KnownSubject *GUID
}

type WinTrustSignatureSettings struct {
	Size             uint32
	Index            uint32
	Flags            uint32
	SecondarySigs    uint32
	VerifiedSigIndex uint32
	CryptoPolicy     *CertStrongSignPara
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
	FSCTL_CREATE_OR_GET_OBJECT_ID             = 0x0900C0
	FSCTL_DELETE_OBJECT_ID                    = 0x0900A0
	FSCTL_DELETE_REPARSE_POINT                = 0x0900AC
	FSCTL_DUPLICATE_EXTENTS_TO_FILE           = 0x098344
	FSCTL_DUPLICATE_EXTENTS_TO_FILE_EX        = 0x0983E8
	FSCTL_FILESYSTEM_GET_STATISTICS           = 0x090060
	FSCTL_FILE_LEVEL_TRIM                     = 0x098208
	FSCTL_FIND_FILES_BY_SID                   = 0x09008F
	FSCTL_GET_COMPRESSION                     = 0x09003C
	FSCTL_GET_INTEGRITY_INFORMATION           = 0x09027C
	FSCTL_GET_NTFS_VOLUME_DATA                = 0x090064
	FSCTL_GET_REFS_VOLUME_DATA                = 0x0902D8
	FSCTL_GET_OBJECT_ID                       = 0x09009C
	FSCTL_GET_REPARSE_POINT                   = 0x0900A8
	FSCTL_GET_RETRIEVAL_POINTER_COUNT         = 0x09042B
	FSCTL_GET_RETRIEVAL_POINTERS              = 0x090073
	FSCTL_GET_RETRIEVAL_POINTERS_AND_REFCOUNT = 0x0903D3
	FSCTL_IS_PATHNAME_VALID                   = 0x09002C
	FSCTL_LMR_SET_LINK_TRACKING_INFORMATION   = 0x1400EC
	FSCTL_MARK_HANDLE                         = 0x0900FC
	FSCTL_OFFLOAD_READ                        = 0x094264
	FSCTL_OFFLOAD_WRITE                       = 0x098268
	FSCTL_PIPE_PEEK                           = 0x11400C
	FSCTL_PIPE_TRANSCEIVE                     = 0x11C017
	FSCTL_PIPE_WAIT                           = 0x110018
	FSCTL_QUERY_ALLOCATED_RANGES              = 0x0940CF
	FSCTL_QUERY_FAT_BPB                       = 0x090058
	FSCTL_QUERY_FILE_REGIONS                  = 0x090284
	FSCTL_QUERY_ON_DISK_VOLUME_INFO           = 0x09013C
	FSCTL_QUERY_SPARING_INFO                  = 0x090138
	FSCTL_READ_FILE_USN_DATA                  = 0x0900EB
	FSCTL_RECALL_FILE                         = 0x090117
	FSCTL_REFS_STREAM_SNAPSHOT_MANAGEMENT     = 0x090440
	FSCTL_SET_COMPRESSION                     = 0x09C040
	FSCTL_SET_DEFECT_MANAGEMENT               = 0x098134
	FSCTL_SET_ENCRYPTION                      = 0x0900D7
	FSCTL_SET_INTEGRITY_INFORMATION           = 0x09C280
	FSCTL_SET_INTEGRITY_INFORMATION_EX        = 0x090380
	FSCTL_SET_OBJECT_ID                       = 0x090098
	FSCTL_SET_OBJECT_ID_EXTENDED              = 0x0900BC
	FSCTL_SET_REPARSE_POINT                   = 0x0900A4
	FSCTL_SET_SPARSE                          = 0x0900C4
	FSCTL_SET_ZERO_DATA                       = 0x0980C8
	FSCTL_SET_ZERO_ON_DEALLOCATION            = 0x090194
	FSCTL_SIS_COPYFILE                        = 0x090100
	FSCTL_WRITE_USN_CLOSE_RECORD              = 0x0900EF

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

// FILE_INFO_BY_HANDLE_CLASS constants for SetFileInformationByHandle/GetFileInformationByHandleEx
const (
	FileBasicInfo                  = 0
	FileStandardInfo               = 1
	FileNameInfo                   = 2
	FileRenameInfo                 = 3
	FileDispositionInfo            = 4
	FileAllocationInfo             = 5
	FileEndOfFileInfo              = 6
	FileStreamInfo                 = 7
	FileCompressionInfo            = 8
	FileAttributeTagInfo           = 9
	FileIdBothDirectoryInfo        = 10
	FileIdBothDirectoryRestartInfo = 11
	FileIoPriorityHintInfo         = 12
	FileRemoteProtocolInfo         = 13
	FileFullDirectoryInfo          = 14
	FileFullDirectoryRestartInfo   = 15
	FileStorageInfo                = 16
	FileAlignmentInfo              = 17
	FileIdInfo                     = 18
	FileIdExtdDirectoryInfo        = 19
	FileIdExtdDirectoryRestartInfo = 20
	FileDispositionInfoEx          = 21
	FileRenameInfoEx               = 22
	FileCaseSensitiveInfo          = 23
	FileNormalizedNameInfo         = 24
)

// LoadLibrary flags for determining from where to search for a DLL
const (
	DONT_RESOLVE_DLL_REFERENCES               = 0x1
	LOAD_LIBRARY_AS_DATAFILE                  = 0x2
	LOAD_WITH_ALTERED_SEARCH_PATH             = 0x8
	LOAD_IGNORE_CODE_AUTHZ_LEVEL              = 0x10
	LOAD_LIBRARY_AS_IMAGE_RESOURCE            = 0x20
	LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE        = 0x40
	LOAD_LIBRARY_REQUIRE_SIGNED_TARGET        = 0x80
	LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR          = 0x100
	LOAD_LIBRARY_SEARCH_APPLICATION_DIR       = 0x200
	LOAD_LIBRARY_SEARCH_USER_DIRS             = 0x400
	LOAD_LIBRARY_SEARCH_SYSTEM32              = 0x800
	LOAD_LIBRARY_SEARCH_DEFAULT_DIRS          = 0x1000
	LOAD_LIBRARY_SAFE_CURRENT_DIRS            = 0x00002000
	LOAD_LIBRARY_SEARCH_SYSTEM32_NO_FORWARDER = 0x00004000
	LOAD_LIBRARY_OS_INTEGRITY_CONTINUITY      = 0x00008000
)

// RegNotifyChangeKeyValue notifyFilter flags.
const (
	// REG_NOTIFY_CHANGE_NAME notifies the caller if a subkey is added or deleted.
	REG_NOTIFY_CHANGE_NAME = 0x00000001

	// REG_NOTIFY_CHANGE_ATTRIBUTES notifies the caller of changes to the attributes of the key, such as the security descriptor information.
	REG_NOTIFY_CHANGE_ATTRIBUTES = 0x00000002

	// REG_NOTIFY_CHANGE_LAST_SET notifies the caller of changes to a value of the key. This can include adding or deleting a value, or changing an existing value.
	REG_NOTIFY_CHANGE_LAST_SET = 0x00000004

	// REG_NOTIFY_CHANGE_SECURITY notifies the caller of changes to the security descriptor of the key.
	REG_NOTIFY_CHANGE_SECURITY = 0x00000008

	// REG_NOTIFY_THREAD_AGNOSTIC indicates that the lifetime of the registration must not be tied to the lifetime of the thread issuing the RegNotifyChangeKeyValue call. Note: This flag value is only supported in Windows 8 and later.
	REG_NOTIFY_THREAD_AGNOSTIC = 0x10000000
)

type CommTimeouts struct {
	ReadIntervalTimeout         uint32
	ReadTotalTimeoutMultiplier  uint32
	ReadTotalTimeoutConstant    uint32
	WriteTotalTimeoutMultiplier uint32
	WriteTotalTimeoutConstant   uint32
}

// NTUnicodeString is a UTF-16 string for NT native APIs, corresponding to UNICODE_STRING.
type NTUnicodeString struct {
	Length        uint16
	MaximumLength uint16
	Buffer        *uint16
}

// NTString is an ANSI string for NT native APIs, corresponding to STRING.
type NTString struct {
	Length        uint16
	MaximumLength uint16
	Buffer        *byte
}

type LIST_ENTRY struct {
	Flink *LIST_ENTRY
	Blink *LIST_ENTRY
}

type RUNTIME_FUNCTION struct {
	BeginAddress uint32
	EndAddress   uint32
	UnwindData   uint32
}

type LDR_DATA_TABLE_ENTRY struct {
	reserved1          [2]uintptr
	InMemoryOrderLinks LIST_ENTRY
	reserved2          [2]uintptr
	DllBase            uintptr
	reserved3          [2]uintptr
	FullDllName        NTUnicodeString
	reserved4          [8]byte
	reserved5          [3]uintptr
	reserved6          uintptr
	TimeDateStamp      uint32
}

type PEB_LDR_DATA struct {
	reserved1               [8]byte
	reserved2               [3]uintptr
	InMemoryOrderModuleList LIST_ENTRY
}

type CURDIR struct {
	DosPath NTUnicodeString
	Handle  Handle
}

type RTL_DRIVE_LETTER_CURDIR struct {
	Flags     uint16
	Length    uint16
	TimeStamp uint32
	DosPath   NTString
}

type RTL_USER_PROCESS_PARAMETERS struct {
	MaximumLength, Length uint32

	Flags, DebugFlags uint32

	ConsoleHandle                                Handle
	ConsoleFlags                                 uint32
	StandardInput, StandardOutput, StandardError Handle

	CurrentDirectory CURDIR
	DllPath          NTUnicodeString
	ImagePathName    NTUnicodeString
	CommandLine      NTUnicodeString
	Environment      unsafe.Pointer

	StartingX, StartingY, CountX, CountY, CountCharsX, CountCharsY, FillAttribute uint32

	WindowFlags, ShowWindowFlags                     uint32
	WindowTitle, DesktopInfo, ShellInfo, RuntimeData NTUnicodeString
	CurrentDirectories                               [32]RTL_DRIVE_LETTER_CURDIR

	EnvironmentSize, EnvironmentVersion uintptr

	PackageDependencyData unsafe.Pointer
	ProcessGroupId        uint32
	LoaderThreads         uint32

	RedirectionDllName               NTUnicodeString
	HeapPartitionName                NTUnicodeString
	DefaultThreadpoolCpuSetMasks     uintptr
	DefaultThreadpoolCpuSetMaskCount uint32
}

type PEB struct {
	reserved1              [2]byte
	BeingDebugged          byte
	BitField               byte
	reserved3              uintptr
	ImageBaseAddress       uintptr
	Ldr                    *PEB_LDR_DATA
	ProcessParameters      *RTL_USER_PROCESS_PARAMETERS
	reserved4              [3]uintptr
	AtlThunkSListPtr       uintptr
	reserved5              uintptr
	reserved6              uint32
	reserved7              uintptr
	reserved8              uint32
	AtlThunkSListPtr32     uint32
	reserved9              [45]uintptr
	reserved10             [96]byte
	PostProcessInitRoutine uintptr
	reserved11             [128]byte
	reserved12             [1]uintptr
	SessionId              uint32
}

type OBJECT_ATTRIBUTES struct {
	Length             uint32
	RootDirectory      Handle
	ObjectName         *NTUnicodeString
	Attributes         uint32
	SecurityDescriptor *SECURITY_DESCRIPTOR
	SecurityQoS        *SECURITY_QUALITY_OF_SERVICE
}

// Values for the Attributes member of OBJECT_ATTRIBUTES.
const (
	OBJ_INHERIT                       = 0x00000002
	OBJ_PERMANENT                     = 0x00000010
	OBJ_EXCLUSIVE                     = 0x00000020
	OBJ_CASE_INSENSITIVE              = 0x00000040
	OBJ_OPENIF                        = 0x00000080
	OBJ_OPENLINK                      = 0x00000100
	OBJ_KERNEL_HANDLE                 = 0x00000200
	OBJ_FORCE_ACCESS_CHECK            = 0x00000400
	OBJ_IGNORE_IMPERSONATED_DEVICEMAP = 0x00000800
	OBJ_DONT_REPARSE                  = 0x00001000
	OBJ_VALID_ATTRIBUTES              = 0x00001FF2
)

type IO_STATUS_BLOCK struct {
	Status      NTStatus
	Information uintptr
}

type RTLP_CURDIR_REF struct {
	RefCount int32
	Handle   Handle
}

type RTL_RELATIVE_NAME struct {
	RelativeName        NTUnicodeString
	ContainingDirectory Handle
	CurDirRef           *RTLP_CURDIR_REF
}

const (
	// CreateDisposition flags for NtCreateFile and NtCreateNamedPipeFile.
	FILE_SUPERSEDE           = 0x00000000
	FILE_OPEN                = 0x00000001
	FILE_CREATE              = 0x00000002
	FILE_OPEN_IF             = 0x00000003
	FILE_OVERWRITE           = 0x00000004
	FILE_OVERWRITE_IF        = 0x00000005
	FILE_MAXIMUM_DISPOSITION = 0x00000005

	// CreateOptions flags for NtCreateFile and NtCreateNamedPipeFile.
	FILE_DIRECTORY_FILE            = 0x00000001
	FILE_WRITE_THROUGH             = 0x00000002
	FILE_SEQUENTIAL_ONLY           = 0x00000004
	FILE_NO_INTERMEDIATE_BUFFERING = 0x00000008
	FILE_SYNCHRONOUS_IO_ALERT      = 0x00000010
	FILE_SYNCHRONOUS_IO_NONALERT   = 0x00000020
	FILE_NON_DIRECTORY_FILE        = 0x00000040
	FILE_CREATE_TREE_CONNECTION    = 0x00000080
	FILE_COMPLETE_IF_OPLOCKED      = 0x00000100
	FILE_NO_EA_KNOWLEDGE           = 0x00000200
	FILE_OPEN_REMOTE_INSTANCE      = 0x00000400
	FILE_RANDOM_ACCESS             = 0x00000800
	FILE_DELETE_ON_CLOSE           = 0x00001000
	FILE_OPEN_BY_FILE_ID           = 0x00002000
	FILE_OPEN_FOR_BACKUP_INTENT    = 0x00004000
	FILE_NO_COMPRESSION            = 0x00008000
	FILE_OPEN_REQUIRING_OPLOCK     = 0x00010000
	FILE_DISALLOW_EXCLUSIVE        = 0x00020000
	FILE_RESERVE_OPFILTER          = 0x00100000
	FILE_OPEN_REPARSE_POINT        = 0x00200000
	FILE_OPEN_NO_RECALL            = 0x00400000
	FILE_OPEN_FOR_FREE_SPACE_QUERY = 0x00800000

	// Parameter constants for NtCreateNamedPipeFile.

	FILE_PIPE_BYTE_STREAM_TYPE = 0x00000000
	FILE_PIPE_MESSAGE_TYPE     = 0x00000001

	FILE_PIPE_ACCEPT_REMOTE_CLIENTS = 0x00000000
	FILE_PIPE_REJECT_REMOTE_CLIENTS = 0x00000002

	FILE_PIPE_TYPE_VALID_MASK = 0x00000003

	FILE_PIPE_BYTE_STREAM_MODE = 0x00000000
	FILE_PIPE_MESSAGE_MODE     = 0x00000001

	FILE_PIPE_QUEUE_OPERATION    = 0x00000000
	FILE_PIPE_COMPLETE_OPERATION = 0x00000001

	FILE_PIPE_INBOUND     = 0x00000000
	FILE_PIPE_OUTBOUND    = 0x00000001
	FILE_PIPE_FULL_DUPLEX = 0x00000002

	FILE_PIPE_DISCONNECTED_STATE = 0x00000001
	FILE_PIPE_LISTENING_STATE    = 0x00000002
	FILE_PIPE_CONNECTED_STATE    = 0x00000003
	FILE_PIPE_CLOSING_STATE      = 0x00000004

	FILE_PIPE_CLIENT_END = 0x00000000
	FILE_PIPE_SERVER_END = 0x00000001
)

const (
	// FileInformationClass for NtSetInformationFile
	FileBasicInformation                         = 4
	FileRenameInformation                        = 10
	FileDispositionInformation                   = 13
	FilePositionInformation                      = 14
	FileEndOfFileInformation                     = 20
	FileValidDataLengthInformation               = 39
	FileShortNameInformation                     = 40
	FileIoPriorityHintInformation                = 43
	FileReplaceCompletionInformation             = 61
	FileDispositionInformationEx                 = 64
	FileCaseSensitiveInformation                 = 71
	FileLinkInformation                          = 72
	FileCaseSensitiveInformationForceAccessCheck = 75
	FileKnownFolderInformation                   = 76

	// Flags for FILE_RENAME_INFORMATION
	FILE_RENAME_REPLACE_IF_EXISTS                    = 0x00000001
	FILE_RENAME_POSIX_SEMANTICS                      = 0x00000002
	FILE_RENAME_SUPPRESS_PIN_STATE_INHERITANCE       = 0x00000004
	FILE_RENAME_SUPPRESS_STORAGE_RESERVE_INHERITANCE = 0x00000008
	FILE_RENAME_NO_INCREASE_AVAILABLE_SPACE          = 0x00000010
	FILE_RENAME_NO_DECREASE_AVAILABLE_SPACE          = 0x00000020
	FILE_RENAME_PRESERVE_AVAILABLE_SPACE             = 0x00000030
	FILE_RENAME_IGNORE_READONLY_ATTRIBUTE            = 0x00000040
	FILE_RENAME_FORCE_RESIZE_TARGET_SR               = 0x00000080
	FILE_RENAME_FORCE_RESIZE_SOURCE_SR               = 0x00000100
	FILE_RENAME_FORCE_RESIZE_SR                      = 0x00000180

	// Flags for FILE_DISPOSITION_INFORMATION_EX
	FILE_DISPOSITION_DO_NOT_DELETE             = 0x00000000
	FILE_DISPOSITION_DELETE                    = 0x00000001
	FILE_DISPOSITION_POSIX_SEMANTICS           = 0x00000002
	FILE_DISPOSITION_FORCE_IMAGE_SECTION_CHECK = 0x00000004
	FILE_DISPOSITION_ON_CLOSE                  = 0x00000008
	FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE = 0x00000010

	// Flags for FILE_CASE_SENSITIVE_INFORMATION
	FILE_CS_FLAG_CASE_SENSITIVE_DIR = 0x00000001

	// Flags for FILE_LINK_INFORMATION
	FILE_LINK_REPLACE_IF_EXISTS                    = 0x00000001
	FILE_LINK_POSIX_SEMANTICS                      = 0x00000002
	FILE_LINK_SUPPRESS_STORAGE_RESERVE_INHERITANCE = 0x00000008
	FILE_LINK_NO_INCREASE_AVAILABLE_SPACE          = 0x00000010
	FILE_LINK_NO_DECREASE_AVAILABLE_SPACE          = 0x00000020
	FILE_LINK_PRESERVE_AVAILABLE_SPACE             = 0x00000030
	FILE_LINK_IGNORE_READONLY_ATTRIBUTE            = 0x00000040
	FILE_LINK_FORCE_RESIZE_TARGET_SR               = 0x00000080
	FILE_LINK_FORCE_RESIZE_SOURCE_SR               = 0x00000100
	FILE_LINK_FORCE_RESIZE_SR                      = 0x00000180
)

// ProcessInformationClasses for NtQueryInformationProcess and NtSetInformationProcess.
const (
	ProcessBasicInformation = iota
	ProcessQuotaLimits
	ProcessIoCounters
	ProcessVmCounters
	ProcessTimes
	ProcessBasePriority
	ProcessRaisePriority
	ProcessDebugPort
	ProcessExceptionPort
	ProcessAccessToken
	ProcessLdtInformation
	ProcessLdtSize
	ProcessDefaultHardErrorMode
	ProcessIoPortHandlers
	ProcessPooledUsageAndLimits
	ProcessWorkingSetWatch
	ProcessUserModeIOPL
	ProcessEnableAlignmentFaultFixup
	ProcessPriorityClass
	ProcessWx86Information
	ProcessHandleCount
	ProcessAffinityMask
	ProcessPriorityBoost
	ProcessDeviceMap
	ProcessSessionInformation
	ProcessForegroundInformation
	ProcessWow64Information
	ProcessImageFileName
	ProcessLUIDDeviceMapsEnabled
	ProcessBreakOnTermination
	ProcessDebugObjectHandle
	ProcessDebugFlags
	ProcessHandleTracing
	ProcessIoPriority
	ProcessExecuteFlags
	ProcessTlsInformation
	ProcessCookie
	ProcessImageInformation
	ProcessCycleTime
	ProcessPagePriority
	ProcessInstrumentationCallback
	ProcessThreadStackAllocation
	ProcessWorkingSetWatchEx
	ProcessImageFileNameWin32
	ProcessImageFileMapping
	ProcessAffinityUpdateMode
	ProcessMemoryAllocationMode
	ProcessGroupInformation
	ProcessTokenVirtualizationEnabled
	ProcessConsoleHostProcess
	ProcessWindowInformation
	ProcessHandleInformation
	ProcessMitigationPolicy
	ProcessDynamicFunctionTableInformation
	ProcessHandleCheckingMode
	ProcessKeepAliveCount
	ProcessRevokeFileHandles
	ProcessWorkingSetControl
	ProcessHandleTable
	ProcessCheckStackExtentsMode
	ProcessCommandLineInformation
	ProcessProtectionInformation
	ProcessMemoryExhaustion
	ProcessFaultInformation
	ProcessTelemetryIdInformation
	ProcessCommitReleaseInformation
	ProcessDefaultCpuSetsInformation
	ProcessAllowedCpuSetsInformation
	ProcessSubsystemProcess
	ProcessJobMemoryInformation
	ProcessInPrivate
	ProcessRaiseUMExceptionOnInvalidHandleClose
	ProcessIumChallengeResponse
	ProcessChildProcessInformation
	ProcessHighGraphicsPriorityInformation
	ProcessSubsystemInformation
	ProcessEnergyValues
	ProcessActivityThrottleState
	ProcessActivityThrottlePolicy
	ProcessWin32kSyscallFilterInformation
	ProcessDisableSystemAllowedCpuSets
	ProcessWakeInformation
	ProcessEnergyTrackingState
	ProcessManageWritesToExecutableMemory
	ProcessCaptureTrustletLiveDump
	ProcessTelemetryCoverage
	ProcessEnclaveInformation
	ProcessEnableReadWriteVmLogging
	ProcessUptimeInformation
	ProcessImageSection
	ProcessDebugAuthInformation
	ProcessSystemResourceManagement
	ProcessSequenceNumber
	ProcessLoaderDetour
	ProcessSecurityDomainInformation
	ProcessCombineSecurityDomainsInformation
	ProcessEnableLogging
	ProcessLeapSecondInformation
	ProcessFiberShadowStackAllocation
	ProcessFreeFiberShadowStackAllocation
	ProcessAltSystemCallInformation
	ProcessDynamicEHContinuationTargets
	ProcessDynamicEnforcedCetCompatibleRanges
)

type PROCESS_BASIC_INFORMATION struct {
	ExitStatus                   NTStatus
	PebBaseAddress               *PEB
	AffinityMask                 uintptr
	BasePriority                 int32
	UniqueProcessId              uintptr
	InheritedFromUniqueProcessId uintptr
}

// SystemInformationClasses for NtQuerySystemInformation and NtSetSystemInformation
const (
	SystemBasicInformation = iota
	SystemProcessorInformation
	SystemPerformanceInformation
	SystemTimeOfDayInformation
	SystemPathInformation
	SystemProcessInformation
	SystemCallCountInformation
	SystemDeviceInformation
	SystemProcessorPerformanceInformation
	SystemFlagsInformation
	SystemCallTimeInformation
	SystemModuleInformation
	SystemLocksInformation
	SystemStackTraceInformation
	SystemPagedPoolInformation
	SystemNonPagedPoolInformation
	SystemHandleInformation
	SystemObjectInformation
	SystemPageFileInformation
	SystemVdmInstemulInformation
	SystemVdmBopInformation
	SystemFileCacheInformation
	SystemPoolTagInformation
	SystemInterruptInformation
	SystemDpcBehaviorInformation
	SystemFullMemoryInformation
	SystemLoadGdiDriverInformation
	SystemUnloadGdiDriverInformation
	SystemTimeAdjustmentInformation
	SystemSummaryMemoryInformation
	SystemMirrorMemoryInformation
	SystemPerformanceTraceInformation
	systemObsolete0
	SystemExceptionInformation
	SystemCrashDumpStateInformation
	SystemKernelDebuggerInformation
	SystemContextSwitchInformation
	SystemRegistryQuotaInformation
	SystemExtendServiceTableInformation
	SystemPrioritySeperation
	SystemVerifierAddDriverInformation
	SystemVerifierRemoveDriverInformation
	SystemProcessorIdleInformation
	SystemLegacyDriverInformation
	SystemCurrentTimeZoneInformation
	SystemLookasideInformation
	SystemTimeSlipNotification
	SystemSessionCreate
	SystemSessionDetach
	SystemSessionInformation
	SystemRangeStartInformation
	SystemVerifierInformation
	SystemVerifierThunkExtend
	SystemSessionProcessInformation
	SystemLoadGdiDriverInSystemSpace
	SystemNumaProcessorMap
	SystemPrefetcherInformation
	SystemExtendedProcessInformation
	SystemRecommendedSharedDataAlignment
	SystemComPlusPackage
	SystemNumaAvailableMemory
	SystemProcessorPowerInformation
	SystemEmulationBasicInformation
	SystemEmulationProcessorInformation
	SystemExtendedHandleInformation
	SystemLostDelayedWriteInformation
	SystemBigPoolInformation
	SystemSessionPoolTagInformation
	SystemSessionMappedViewInformation
	SystemHotpatchInformation
	SystemObjectSecurityMode
	SystemWatchdogTimerHandler
	SystemWatchdogTimerInformation
	SystemLogicalProcessorInformation
	SystemWow64SharedInformationObsolete
	SystemRegisterFirmwareTableInformationHandler
	SystemFirmwareTableInformation
	SystemModuleInformationEx
	SystemVerifierTriageInformation
	SystemSuperfetchInformation
	SystemMemoryListInformation
	SystemFileCacheInformationEx
	SystemThreadPriorityClientIdInformation
	SystemProcessorIdleCycleTimeInformation
	SystemVerifierCancellationInformation
	SystemProcessorPowerInformationEx
	SystemRefTraceInformation
	SystemSpecialPoolInformation
	SystemProcessIdInformation
	SystemErrorPortInformation
	SystemBootEnvironmentInformation
	SystemHypervisorInformation
	SystemVerifierInformationEx
	SystemTimeZoneInformation
	SystemImageFileExecutionOptionsInformation
	SystemCoverageInformation
	SystemPrefetchPatchInformation
	SystemVerifierFaultsInformation
	SystemSystemPartitionInformation
	SystemSystemDiskInformation
	SystemProcessorPerformanceDistribution
	SystemNumaProximityNodeInformation
	SystemDynamicTimeZoneInformation
	SystemCodeIntegrityInformation
	SystemProcessorMicrocodeUpdateInformation
	SystemProcessorBrandString
	SystemVirtualAddressInformation
	SystemLogicalProcessorAndGroupInformation
	SystemProcessorCycleTimeInformation
	SystemStoreInformation
	SystemRegistryAppendString
	SystemAitSamplingValue
	SystemVhdBootInformation
	SystemCpuQuotaInformation
	SystemNativeBasicInformation
	systemSpare1
	SystemLowPriorityIoInformation
	SystemTpmBootEntropyInformation
	SystemVerifierCountersInformation
	SystemPagedPoolInformationEx
	SystemSystemPtesInformationEx
	SystemNodeDistanceInformation
	SystemAcpiAuditInformation
	SystemBasicPerformanceInformation
	SystemQueryPerformanceCounterInformation
	SystemSessionBigPoolInformation
	SystemBootGraphicsInformation
	SystemScrubPhysicalMemoryInformation
	SystemBadPageInformation
	SystemProcessorProfileControlArea
	SystemCombinePhysicalMemoryInformation
	SystemEntropyInterruptTimingCallback
	SystemConsoleInformation
	SystemPlatformBinaryInformation
	SystemThrottleNotificationInformation
	SystemHypervisorProcessorCountInformation
	SystemDeviceDataInformation
	SystemDeviceDataEnumerationInformation
	SystemMemoryTopologyInformation
	SystemMemoryChannelInformation
	SystemBootLogoInformation
	SystemProcessorPerformanceInformationEx
	systemSpare0
	SystemSecureBootPolicyInformation
	SystemPageFileInformationEx
	SystemSecureBootInformation
	SystemEntropyInterruptTimingRawInformation
	SystemPortableWorkspaceEfiLauncherInformation
	SystemFullProcessInformation
	SystemKernelDebuggerInformationEx
	SystemBootMetadataInformation
	SystemSoftRebootInformation
	SystemElamCertificateInformation
	SystemOfflineDumpConfigInformation
	SystemProcessorFeaturesInformation
	SystemRegistryReconciliationInformation
	SystemEdidInformation
	SystemManufacturingInformation
	SystemEnergyEstimationConfigInformation
	SystemHypervisorDetailInformation
	SystemProcessorCycleStatsInformation
	SystemVmGenerationCountInformation
	SystemTrustedPlatformModuleInformation
	SystemKernelDebuggerFlags
	SystemCodeIntegrityPolicyInformation
	SystemIsolatedUserModeInformation
	SystemHardwareSecurityTestInterfaceResultsInformation
	SystemSingleModuleInformation
	SystemAllowedCpuSetsInformation
	SystemDmaProtectionInformation
	SystemInterruptCpuSetsInformation
	SystemSecureBootPolicyFullInformation
	SystemCodeIntegrityPolicyFullInformation
	SystemAffinitizedInterruptProcessorInformation
	SystemRootSiloInformation
)

type RTL_PROCESS_MODULE_INFORMATION struct {
	Section          Handle
	MappedBase       uintptr
	ImageBase        uintptr
	ImageSize        uint32
	Flags            uint32
	LoadOrderIndex   uint16
	InitOrderIndex   uint16
	LoadCount        uint16
	OffsetToFileName uint16
	FullPathName     [256]byte
}

type RTL_PROCESS_MODULES struct {
	NumberOfModules uint32
	Modules         [1]RTL_PROCESS_MODULE_INFORMATION
}

// Constants for LocalAlloc flags.
const (
	LMEM_FIXED          = 0x0
	LMEM_MOVEABLE       = 0x2
	LMEM_NOCOMPACT      = 0x10
	LMEM_NODISCARD      = 0x20
	LMEM_ZEROINIT       = 0x40
	LMEM_MODIFY         = 0x80
	LMEM_DISCARDABLE    = 0xf00
	LMEM_VALID_FLAGS    = 0xf72
	LMEM_INVALID_HANDLE = 0x8000
	LHND                = LMEM_MOVEABLE | LMEM_ZEROINIT
	LPTR                = LMEM_FIXED | LMEM_ZEROINIT
	NONZEROLHND         = LMEM_MOVEABLE
	NONZEROLPTR         = LMEM_FIXED
)

// Constants for the CreateNamedPipe-family of functions.
const (
	PIPE_ACCESS_INBOUND  = 0x1
	PIPE_ACCESS_OUTBOUND = 0x2
	PIPE_ACCESS_DUPLEX   = 0x3

	PIPE_CLIENT_END = 0x0
	PIPE_SERVER_END = 0x1

	PIPE_WAIT                  = 0x0
	PIPE_NOWAIT                = 0x1
	PIPE_READMODE_BYTE         = 0x0
	PIPE_READMODE_MESSAGE      = 0x2
	PIPE_TYPE_BYTE             = 0x0
	PIPE_TYPE_MESSAGE          = 0x4
	PIPE_ACCEPT_REMOTE_CLIENTS = 0x0
	PIPE_REJECT_REMOTE_CLIENTS = 0x8

	PIPE_UNLIMITED_INSTANCES = 255
)

// Constants for security attributes when opening named pipes.
const (
	SECURITY_ANONYMOUS      = SecurityAnonymous << 16
	SECURITY_IDENTIFICATION = SecurityIdentification << 16
	SECURITY_IMPERSONATION  = SecurityImpersonation << 16
	SECURITY_DELEGATION     = SecurityDelegation << 16

	SECURITY_CONTEXT_TRACKING = 0x40000
	SECURITY_EFFECTIVE_ONLY   = 0x80000

	SECURITY_SQOS_PRESENT     = 0x100000
	SECURITY_VALID_SQOS_FLAGS = 0x1f0000
)

// ResourceID represents a 16-bit resource identifier, traditionally created with the MAKEINTRESOURCE macro.
type ResourceID uint16

// ResourceIDOrString must be either a ResourceID, to specify a resource or resource type by ID,
// or a string, to specify a resource or resource type by name.
type ResourceIDOrString interface{}

// Predefined resource names and types.
var (
	// Predefined names.
	CREATEPROCESS_MANIFEST_RESOURCE_ID                 ResourceID = 1
	ISOLATIONAWARE_MANIFEST_RESOURCE_ID                ResourceID = 2
	ISOLATIONAWARE_NOSTATICIMPORT_MANIFEST_RESOURCE_ID ResourceID = 3
	ISOLATIONPOLICY_MANIFEST_RESOURCE_ID               ResourceID = 4
	ISOLATIONPOLICY_BROWSER_MANIFEST_RESOURCE_ID       ResourceID = 5
	MINIMUM_RESERVED_MANIFEST_RESOURCE_ID              ResourceID = 1  // inclusive
	MAXIMUM_RESERVED_MANIFEST_RESOURCE_ID              ResourceID = 16 // inclusive

	// Predefined types.
	RT_CURSOR       ResourceID = 1
	RT_BITMAP       ResourceID = 2
	RT_ICON         ResourceID = 3
	RT_MENU         ResourceID = 4
	RT_DIALOG       ResourceID = 5
	RT_STRING       ResourceID = 6
	RT_FONTDIR      ResourceID = 7
	RT_FONT         ResourceID = 8
	RT_ACCELERATOR  ResourceID = 9
	RT_RCDATA       ResourceID = 10
	RT_MESSAGETABLE ResourceID = 11
	RT_GROUP_CURSOR ResourceID = 12
	RT_GROUP_ICON   ResourceID = 14
	RT_VERSION      ResourceID = 16
	RT_DLGINCLUDE   ResourceID = 17
	RT_PLUGPLAY     ResourceID = 19
	RT_VXD          ResourceID = 20
	RT_ANICURSOR    ResourceID = 21
	RT_ANIICON      ResourceID = 22
	RT_HTML         ResourceID = 23
	RT_MANIFEST     ResourceID = 24
)

type VS_FIXEDFILEINFO struct {
	Signature        uint32
	StrucVersion     uint32
	FileVersionMS    uint32
	FileVersionLS    uint32
	ProductVersionMS uint32
	ProductVersionLS uint32
	FileFlagsMask    uint32
	FileFlags        uint32
	FileOS           uint32
	FileType         uint32
	FileSubtype      uint32
	FileDateMS       uint32
	FileDateLS       uint32
}

type COAUTHIDENTITY struct {
	User           *uint16
	UserLength     uint32
	Domain         *uint16
	DomainLength   uint32
	Password       *uint16
	PasswordLength uint32
	Flags          uint32
}

type COAUTHINFO struct {
	AuthnSvc           uint32
	AuthzSvc           uint32
	ServerPrincName    *uint16
	AuthnLevel         uint32
	ImpersonationLevel uint32
	AuthIdentityData   *COAUTHIDENTITY
	Capabilities       uint32
}

type COSERVERINFO struct {
	Reserved1 uint32
	Aame      *uint16
	AuthInfo  *COAUTHINFO
	Reserved2 uint32
}

type BIND_OPTS3 struct {
	CbStruct          uint32
	Flags             uint32
	Mode              uint32
	TickCountDeadline uint32
	TrackFlags        uint32
	ClassContext      uint32
	Locale            uint32
	ServerInfo        *COSERVERINFO
	Hwnd              HWND
}

const (
	CLSCTX_INPROC_SERVER          = 0x1
	CLSCTX_INPROC_HANDLER         = 0x2
	CLSCTX_LOCAL_SERVER           = 0x4
	CLSCTX_INPROC_SERVER16        = 0x8
	CLSCTX_REMOTE_SERVER          = 0x10
	CLSCTX_INPROC_HANDLER16       = 0x20
	CLSCTX_RESERVED1              = 0x40
	CLSCTX_RESERVED2              = 0x80
	CLSCTX_RESERVED3              = 0x100
	CLSCTX_RESERVED4              = 0x200
	CLSCTX_NO_CODE_DOWNLOAD       = 0x400
	CLSCTX_RESERVED5              = 0x800
	CLSCTX_NO_CUSTOM_MARSHAL      = 0x1000
	CLSCTX_ENABLE_CODE_DOWNLOAD   = 0x2000
	CLSCTX_NO_FAILURE_LOG         = 0x4000
	CLSCTX_DISABLE_AAA            = 0x8000
	CLSCTX_ENABLE_AAA             = 0x10000
	CLSCTX_FROM_DEFAULT_CONTEXT   = 0x20000
	CLSCTX_ACTIVATE_32_BIT_SERVER = 0x40000
	CLSCTX_ACTIVATE_64_BIT_SERVER = 0x80000
	CLSCTX_ENABLE_CLOAKING        = 0x100000
	CLSCTX_APPCONTAINER           = 0x400000
	CLSCTX_ACTIVATE_AAA_AS_IU     = 0x800000
	CLSCTX_PS_DLL                 = 0x80000000

	COINIT_MULTITHREADED     = 0x0
	COINIT_APARTMENTTHREADED = 0x2
	COINIT_DISABLE_OLE1DDE   = 0x4
	COINIT_SPEED_OVER_MEMORY = 0x8
)

// Flag for QueryFullProcessImageName.
const PROCESS_NAME_NATIVE = 1

type ModuleInfo struct {
	BaseOfDll   uintptr
	SizeOfImage uint32
	EntryPoint  uintptr
}
