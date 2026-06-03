// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"internal/runtime/syscall/windows"
	"syscall"
	"unsafe"
)

// Socket related.
const (
	TCP_KEEPIDLE  = 0x03
	TCP_KEEPCNT   = 0x10
	TCP_KEEPINTVL = 0x11

	SIO_UDP_NETRESET = syscall.IOC_IN | syscall.IOC_VENDOR | 15
)

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

	FILE_SHARE_READ                      = 0x00000001
	FILE_SHARE_WRITE                     = 0x00000002
	FILE_SHARE_DELETE                    = 0x00000004
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
)

// https://learn.microsoft.com/en-us/windows/win32/secauthz/access-mask
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

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ns-wdm-_acl
type ACL struct {
	AclRevision byte
	Sbz1        byte
	AclSize     uint16
	AceCount    uint16
	Sbz2        uint16
}

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ns-wdm-_io_status_block
type IO_STATUS_BLOCK struct {
	Status      NTStatus
	Information uintptr
}

// https://learn.microsoft.com/en-us/windows/win32/api/ntdef/ns-ntdef-_object_attributes
type OBJECT_ATTRIBUTES struct {
	Length             uint32
	RootDirectory      syscall.Handle
	ObjectName         *NTUnicodeString
	Attributes         uint32
	SecurityDescriptor *SECURITY_DESCRIPTOR
	SecurityQoS        *SECURITY_QUALITY_OF_SERVICE
}

// init sets o's RootDirectory, ObjectName, and Length.
func (o *OBJECT_ATTRIBUTES) init(root syscall.Handle, name string) error {
	if name == "." {
		name = ""
	}
	objectName, err := NewNTUnicodeString(name)
	if err != nil {
		return err
	}
	o.ObjectName = objectName
	if root != syscall.InvalidHandle {
		o.RootDirectory = root
	}
	o.Length = uint32(unsafe.Sizeof(*o))
	return nil
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

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_security_descriptor
type SECURITY_DESCRIPTOR struct {
	revision byte
	sbz1     byte
	control  SECURITY_DESCRIPTOR_CONTROL
	owner    *syscall.SID
	group    *syscall.SID
	sacl     *ACL
	dacl     *ACL
}

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ifs/security-descriptor-control
type SECURITY_DESCRIPTOR_CONTROL uint16

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-security_quality_of_service
type SECURITY_QUALITY_OF_SERVICE struct {
	Length              uint32
	ImpersonationLevel  uint32 // type SECURITY_IMPERSONATION_LEVEL
	ContextTrackingMode byte   // type SECURITY_CONTEXT_TRACKING_MODE
	EffectiveOnly       byte
}

// File flags for [os.OpenFile]. The O_ prefix is used to indicate
// that these flags are specific to the OpenFile function.
const (
	O_FILE_FLAG_OPEN_NO_RECALL     = 0x00100000
	O_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
	O_FILE_FLAG_SESSION_AWARE      = 0x00800000
	O_FILE_FLAG_POSIX_SEMANTICS    = 0x01000000
	O_FILE_FLAG_BACKUP_SEMANTICS   = 0x02000000
	O_FILE_FLAG_DELETE_ON_CLOSE    = 0x04000000
	O_FILE_FLAG_SEQUENTIAL_SCAN    = 0x08000000
	O_FILE_FLAG_RANDOM_ACCESS      = 0x10000000
	O_FILE_FLAG_NO_BUFFERING       = 0x20000000
	O_FILE_FLAG_OVERLAPPED         = 0x40000000
	O_FILE_FLAG_WRITE_THROUGH      = 0x80000000
)

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
	FILE_SESSION_AWARE             = 0x00040000
	FILE_RESERVE_OPFILTER          = 0x00100000
	FILE_OPEN_REPARSE_POINT        = 0x00200000
	FILE_OPEN_NO_RECALL            = 0x00400000
	FILE_OPEN_FOR_FREE_SPACE_QUERY = 0x00800000
)

// https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_disposition_info
type FILE_DISPOSITION_INFO struct {
	DeleteFile byte
}

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntddk/ns-ntddk-_file_disposition_information
type FILE_DISPOSITION_INFORMATION struct {
	DeleteFile byte
}

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntddk/ns-ntddk-_file_disposition_information_ex
type FILE_DISPOSITION_INFORMATION_EX struct {
	Flags uint32
}

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntddk/ns-ntddk-_file_disposition_information_ex
const (
	FILE_DISPOSITION_DO_NOT_DELETE             = 0x00000000
	FILE_DISPOSITION_DELETE                    = 0x00000001
	FILE_DISPOSITION_POSIX_SEMANTICS           = 0x00000002
	FILE_DISPOSITION_FORCE_IMAGE_SECTION_CHECK = 0x00000004
	FILE_DISPOSITION_ON_CLOSE                  = 0x00000008
	FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE = 0x00000010
)

// Flags for FILE_RENAME_INFORMATION_EX.
const (
	FILE_RENAME_REPLACE_IF_EXISTS = 0x00000001
	FILE_RENAME_POSIX_SEMANTICS   = 0x00000002
)

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information
type FILE_RENAME_INFORMATION struct {
	ReplaceIfExists byte
	RootDirectory   syscall.Handle
	FileNameLength  uint32
	FileName        [syscall.MAX_PATH]uint16
}

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information
type FILE_RENAME_INFORMATION_EX struct {
	Flags          uint32
	RootDirectory  syscall.Handle
	FileNameLength uint32
	FileName       [syscall.MAX_PATH]uint16
}

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_link_information
type FILE_LINK_INFORMATION struct {
	ReplaceIfExists byte
	RootDirectory   syscall.Handle
	FileNameLength  uint32
	FileName        [syscall.MAX_PATH]uint16
}

const FileReplaceCompletionInformation = 61

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_completion_information
type FILE_COMPLETION_INFORMATION struct {
	Port syscall.Handle
	Key  uintptr
}

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-osversioninfoexa
// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ns-wdm-_osversioninfoexw
const VER_NT_WORKSTATION = 0x0000001

type MemoryBasicInformation = windows.MemoryBasicInformation

type Context = windows.Context

const FileFlagsMask = 0xFFF00000

const ValidFileFlagsMask = O_FILE_FLAG_OPEN_REPARSE_POINT |
	O_FILE_FLAG_BACKUP_SEMANTICS |
	O_FILE_FLAG_OVERLAPPED |
	O_FILE_FLAG_OPEN_NO_RECALL |
	O_FILE_FLAG_SESSION_AWARE |
	O_FILE_FLAG_POSIX_SEMANTICS |
	O_FILE_FLAG_DELETE_ON_CLOSE |
	O_FILE_FLAG_SEQUENTIAL_SCAN |
	O_FILE_FLAG_NO_BUFFERING |
	O_FILE_FLAG_RANDOM_ACCESS |
	O_FILE_FLAG_WRITE_THROUGH

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379636.aspx
type TRUSTEE struct {
	MultipleTrustee          *TRUSTEE
	MultipleTrusteeOperation uint32
	TrusteeForm              uint32
	TrusteeType              uint32
	Name                     uintptr
}

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379638.aspx
const (
	TRUSTEE_IS_SID              = 0x0
	TRUSTEE_IS_NAME             = 0x1
	TRUSTEE_BAD_FORM            = 0x2
	TRUSTEE_IS_OBJECTS_AND_SID  = 0x3
	TRUSTEE_IS_OBJECTS_AND_NAME = 0x4
)

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa446627.aspx
type EXPLICIT_ACCESS struct {
	AccessPermissions uint32
	AccessMode        uint32
	Inheritance       uint32
	Trustee           TRUSTEE
}

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa374899.aspx
const (
	NOT_USED_ACCESS   = 0x0
	GRANT_ACCESS      = 0x1
	SET_ACCESS        = 0x2
	DENY_ACCESS       = 0x3
	REVOKE_ACCESS     = 0x4
	SET_AUDIT_SUCCESS = 0x5
	SET_AUDIT_FAILURE = 0x6
)

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa446627.aspx
const (
	NO_INHERITANCE                     = 0x0
	SUB_OBJECTS_ONLY_INHERIT           = 0x1
	SUB_CONTAINERS_ONLY_INHERIT        = 0x2
	SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
	INHERIT_NO_PROPAGATE               = 0x4
	INHERIT_ONLY                       = 0x8
)

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa379593.aspx
const (
	SE_UNKNOWN_OBJECT_TYPE     = 0x0
	SE_FILE_OBJECT             = 0x1
	SE_SERVICE                 = 0x2
	SE_PRINTER                 = 0x3
	SE_REGISTRY_KEY            = 0x4
	SE_LMSHARE                 = 0x5
	SE_KERNEL_OBJECT           = 0x6
	SE_WINDOW_OBJECT           = 0x7
	SE_DS_OBJECT               = 0x8
	SE_DS_OBJECT_ALL           = 0x9
	SE_PROVIDER_DEFINED_OBJECT = 0xa
	SE_WMIGUID_OBJECT          = 0xb
	SE_REGISTRY_WOW64_32KEY    = 0xc
	SE_REGISTRY_WOW64_64KEY    = 0xd
)

// https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-dtyp/23e75ca3-98fd-4396-84e5-86cd9d40d343
const (
	OWNER_SECURITY_INFORMATION               = 0x00000001
	GROUP_SECURITY_INFORMATION               = 0x00000002
	DACL_SECURITY_INFORMATION                = 0x00000004
	SACL_SECURITY_INFORMATION                = 0x00000008
	LABEL_SECURITY_INFORMATION               = 0x00000010
	UNPROTECTED_SACL_SECURITY_INFORMATION    = 0x10000000
	UNPROTECTED_DACL_SECURITY_INFORMATION    = 0x20000000
	PROTECTED_SACL_SECURITY_INFORMATION      = 0x40000000
	PROTECTED_DACL_SECURITY_INFORMATION      = 0x80000000
	ATTRIBUTE_SECURITY_INFORMATION           = 0x00000020
	SCOPE_SECURITY_INFORMATION               = 0x00000040
	PROCESS_TRUST_LABEL_SECURITY_INFORMATION = 0x00000080
	BACKUP_SECURITY_INFORMATION              = 0x00010000
)

// The processor features to be tested for IsProcessorFeaturePresent, see
// https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent
const (
	PF_ARM_64BIT_LOADSTORE_ATOMIC              = 25
	PF_ARM_DIVIDE_INSTRUCTION_AVAILABLE        = 24
	PF_ARM_EXTERNAL_CACHE_AVAILABLE            = 26
	PF_ARM_FMAC_INSTRUCTIONS_AVAILABLE         = 27
	PF_ARM_VFP_32_REGISTERS_AVAILABLE          = 18
	PF_3DNOW_INSTRUCTIONS_AVAILABLE            = 7
	PF_CHANNELS_ENABLED                        = 16
	PF_COMPARE_EXCHANGE_DOUBLE                 = 2
	PF_COMPARE_EXCHANGE128                     = 14
	PF_COMPARE64_EXCHANGE128                   = 15
	PF_FASTFAIL_AVAILABLE                      = 23
	PF_FLOATING_POINT_EMULATED                 = 1
	PF_FLOATING_POINT_PRECISION_ERRATA         = 0
	PF_MMX_INSTRUCTIONS_AVAILABLE              = 3
	PF_NX_ENABLED                              = 12
	PF_PAE_ENABLED                             = 9
	PF_RDTSC_INSTRUCTION_AVAILABLE             = 8
	PF_RDWRFSGSBASE_AVAILABLE                  = 22
	PF_SECOND_LEVEL_ADDRESS_TRANSLATION        = 20
	PF_SSE3_INSTRUCTIONS_AVAILABLE             = 13
	PF_SSSE3_INSTRUCTIONS_AVAILABLE            = 36
	PF_SSE4_1_INSTRUCTIONS_AVAILABLE           = 37
	PF_SSE4_2_INSTRUCTIONS_AVAILABLE           = 38
	PF_AVX_INSTRUCTIONS_AVAILABLE              = 39
	PF_AVX2_INSTRUCTIONS_AVAILABLE             = 40
	PF_AVX512F_INSTRUCTIONS_AVAILABLE          = 41
	PF_VIRT_FIRMWARE_ENABLED                   = 21
	PF_XMMI_INSTRUCTIONS_AVAILABLE             = 6
	PF_XMMI64_INSTRUCTIONS_AVAILABLE           = 10
	PF_XSAVE_ENABLED                           = 17
	PF_ARM_V8_INSTRUCTIONS_AVAILABLE           = 29
	PF_ARM_V8_CRYPTO_INSTRUCTIONS_AVAILABLE    = 30
	PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE     = 31
	PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE   = 34
	PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE       = 43
	PF_ARM_V83_JSCVT_INSTRUCTIONS_AVAILABLE    = 44
	PF_ARM_V83_LRCPC_INSTRUCTIONS_AVAILABLE    = 45
	PF_ARM_SVE_INSTRUCTIONS_AVAILABLE          = 46
	PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE         = 47
	PF_ARM_SVE2_1_INSTRUCTIONS_AVAILABLE       = 48
	PF_ARM_SVE_AES_INSTRUCTIONS_AVAILABLE      = 49
	PF_ARM_SVE_PMULL128_INSTRUCTIONS_AVAILABLE = 50
	PF_ARM_SVE_BITPERM_INSTRUCTIONS_AVAILABLE  = 51
	PF_ARM_SVE_BF16_INSTRUCTIONS_AVAILABLE     = 52
	PF_ARM_SVE_EBF16_INSTRUCTIONS_AVAILABLE    = 53
	PF_ARM_SVE_B16B16_INSTRUCTIONS_AVAILABLE   = 54
	PF_ARM_SVE_SHA3_INSTRUCTIONS_AVAILABLE     = 55
	PF_ARM_SVE_SM4_INSTRUCTIONS_AVAILABLE      = 56
	PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE     = 57
	PF_ARM_SVE_F32MM_INSTRUCTIONS_AVAILABLE    = 58
	PF_ARM_SVE_F64MM_INSTRUCTIONS_AVAILABLE    = 59
	PF_BMI2_INSTRUCTIONS_AVAILABLE             = 60
	PF_MOVDIR64B_INSTRUCTION_AVAILABLE         = 61
	PF_ARM_LSE2_AVAILABLE                      = 62
	PF_ARM_SHA3_INSTRUCTIONS_AVAILABLE         = 64
	PF_ARM_SHA512_INSTRUCTIONS_AVAILABLE       = 65
	PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE     = 66
	PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE     = 67
	PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE     = 68
	PF_ARM_V86_EBF16_INSTRUCTIONS_AVAILABLE    = 69
	PF_ARM_SME_INSTRUCTIONS_AVAILABLE          = 70
	PF_ARM_SME2_INSTRUCTIONS_AVAILABLE         = 71
	PF_ARM_SME2_1_INSTRUCTIONS_AVAILABLE       = 72
	PF_ARM_SME2_2_INSTRUCTIONS_AVAILABLE       = 73
	PF_ARM_SME_AES_INSTRUCTIONS_AVAILABLE      = 74
	PF_ARM_SME_SBITPERM_INSTRUCTIONS_AVAILABLE = 75
	PF_ARM_SME_SF8MM4_INSTRUCTIONS_AVAILABLE   = 76
	PF_ARM_SME_SF8MM8_INSTRUCTIONS_AVAILABLE   = 77
	PF_ARM_SME_SF8DP2_INSTRUCTIONS_AVAILABLE   = 78
	PF_ARM_SME_SF8DP4_INSTRUCTIONS_AVAILABLE   = 79
	PF_ARM_SME_SF8FMA_INSTRUCTIONS_AVAILABLE   = 80
	PF_ARM_SME_F8F32_INSTRUCTIONS_AVAILABLE    = 81
	PF_ARM_SME_F8F16_INSTRUCTIONS_AVAILABLE    = 82
	PF_ARM_SME_F16F16_INSTRUCTIONS_AVAILABLE   = 83
	PF_ARM_SME_B16B16_INSTRUCTIONS_AVAILABLE   = 84
	PF_ARM_SME_F64F64_INSTRUCTIONS_AVAILABLE   = 85
	PF_ARM_SME_I16I64_INSTRUCTIONS_AVAILABLE   = 86
	PF_ARM_SME_LUTv2_INSTRUCTIONS_AVAILABLE    = 87
	PF_ARM_SME_FA64_INSTRUCTIONS_AVAILABLE     = 88
	PF_UMONITOR_INSTRUCTION_AVAILABLE          = 89
)
