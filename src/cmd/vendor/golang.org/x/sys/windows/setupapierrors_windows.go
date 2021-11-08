// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import "syscall"

const (
	ERROR_EXPECTED_SECTION_NAME                  syscall.Errno = 0x20000000 | 0xC0000000 | 0
	ERROR_BAD_SECTION_NAME_LINE                  syscall.Errno = 0x20000000 | 0xC0000000 | 1
	ERROR_SECTION_NAME_TOO_LONG                  syscall.Errno = 0x20000000 | 0xC0000000 | 2
	ERROR_GENERAL_SYNTAX                         syscall.Errno = 0x20000000 | 0xC0000000 | 3
	ERROR_WRONG_INF_STYLE                        syscall.Errno = 0x20000000 | 0xC0000000 | 0x100
	ERROR_SECTION_NOT_FOUND                      syscall.Errno = 0x20000000 | 0xC0000000 | 0x101
	ERROR_LINE_NOT_FOUND                         syscall.Errno = 0x20000000 | 0xC0000000 | 0x102
	ERROR_NO_BACKUP                              syscall.Errno = 0x20000000 | 0xC0000000 | 0x103
	ERROR_NO_ASSOCIATED_CLASS                    syscall.Errno = 0x20000000 | 0xC0000000 | 0x200
	ERROR_CLASS_MISMATCH                         syscall.Errno = 0x20000000 | 0xC0000000 | 0x201
	ERROR_DUPLICATE_FOUND                        syscall.Errno = 0x20000000 | 0xC0000000 | 0x202
	ERROR_NO_DRIVER_SELECTED                     syscall.Errno = 0x20000000 | 0xC0000000 | 0x203
	ERROR_KEY_DOES_NOT_EXIST                     syscall.Errno = 0x20000000 | 0xC0000000 | 0x204
	ERROR_INVALID_DEVINST_NAME                   syscall.Errno = 0x20000000 | 0xC0000000 | 0x205
	ERROR_INVALID_CLASS                          syscall.Errno = 0x20000000 | 0xC0000000 | 0x206
	ERROR_DEVINST_ALREADY_EXISTS                 syscall.Errno = 0x20000000 | 0xC0000000 | 0x207
	ERROR_DEVINFO_NOT_REGISTERED                 syscall.Errno = 0x20000000 | 0xC0000000 | 0x208
	ERROR_INVALID_REG_PROPERTY                   syscall.Errno = 0x20000000 | 0xC0000000 | 0x209
	ERROR_NO_INF                                 syscall.Errno = 0x20000000 | 0xC0000000 | 0x20A
	ERROR_NO_SUCH_DEVINST                        syscall.Errno = 0x20000000 | 0xC0000000 | 0x20B
	ERROR_CANT_LOAD_CLASS_ICON                   syscall.Errno = 0x20000000 | 0xC0000000 | 0x20C
	ERROR_INVALID_CLASS_INSTALLER                syscall.Errno = 0x20000000 | 0xC0000000 | 0x20D
	ERROR_DI_DO_DEFAULT                          syscall.Errno = 0x20000000 | 0xC0000000 | 0x20E
	ERROR_DI_NOFILECOPY                          syscall.Errno = 0x20000000 | 0xC0000000 | 0x20F
	ERROR_INVALID_HWPROFILE                      syscall.Errno = 0x20000000 | 0xC0000000 | 0x210
	ERROR_NO_DEVICE_SELECTED                     syscall.Errno = 0x20000000 | 0xC0000000 | 0x211
	ERROR_DEVINFO_LIST_LOCKED                    syscall.Errno = 0x20000000 | 0xC0000000 | 0x212
	ERROR_DEVINFO_DATA_LOCKED                    syscall.Errno = 0x20000000 | 0xC0000000 | 0x213
	ERROR_DI_BAD_PATH                            syscall.Errno = 0x20000000 | 0xC0000000 | 0x214
	ERROR_NO_CLASSINSTALL_PARAMS                 syscall.Errno = 0x20000000 | 0xC0000000 | 0x215
	ERROR_FILEQUEUE_LOCKED                       syscall.Errno = 0x20000000 | 0xC0000000 | 0x216
	ERROR_BAD_SERVICE_INSTALLSECT                syscall.Errno = 0x20000000 | 0xC0000000 | 0x217
	ERROR_NO_CLASS_DRIVER_LIST                   syscall.Errno = 0x20000000 | 0xC0000000 | 0x218
	ERROR_NO_ASSOCIATED_SERVICE                  syscall.Errno = 0x20000000 | 0xC0000000 | 0x219
	ERROR_NO_DEFAULT_DEVICE_INTERFACE            syscall.Errno = 0x20000000 | 0xC0000000 | 0x21A
	ERROR_DEVICE_INTERFACE_ACTIVE                syscall.Errno = 0x20000000 | 0xC0000000 | 0x21B
	ERROR_DEVICE_INTERFACE_REMOVED               syscall.Errno = 0x20000000 | 0xC0000000 | 0x21C
	ERROR_BAD_INTERFACE_INSTALLSECT              syscall.Errno = 0x20000000 | 0xC0000000 | 0x21D
	ERROR_NO_SUCH_INTERFACE_CLASS                syscall.Errno = 0x20000000 | 0xC0000000 | 0x21E
	ERROR_INVALID_REFERENCE_STRING               syscall.Errno = 0x20000000 | 0xC0000000 | 0x21F
	ERROR_INVALID_MACHINENAME                    syscall.Errno = 0x20000000 | 0xC0000000 | 0x220
	ERROR_REMOTE_COMM_FAILURE                    syscall.Errno = 0x20000000 | 0xC0000000 | 0x221
	ERROR_MACHINE_UNAVAILABLE                    syscall.Errno = 0x20000000 | 0xC0000000 | 0x222
	ERROR_NO_CONFIGMGR_SERVICES                  syscall.Errno = 0x20000000 | 0xC0000000 | 0x223
	ERROR_INVALID_PROPPAGE_PROVIDER              syscall.Errno = 0x20000000 | 0xC0000000 | 0x224
	ERROR_NO_SUCH_DEVICE_INTERFACE               syscall.Errno = 0x20000000 | 0xC0000000 | 0x225
	ERROR_DI_POSTPROCESSING_REQUIRED             syscall.Errno = 0x20000000 | 0xC0000000 | 0x226
	ERROR_INVALID_COINSTALLER                    syscall.Errno = 0x20000000 | 0xC0000000 | 0x227
	ERROR_NO_COMPAT_DRIVERS                      syscall.Errno = 0x20000000 | 0xC0000000 | 0x228
	ERROR_NO_DEVICE_ICON                         syscall.Errno = 0x20000000 | 0xC0000000 | 0x229
	ERROR_INVALID_INF_LOGCONFIG                  syscall.Errno = 0x20000000 | 0xC0000000 | 0x22A
	ERROR_DI_DONT_INSTALL                        syscall.Errno = 0x20000000 | 0xC0000000 | 0x22B
	ERROR_INVALID_FILTER_DRIVER                  syscall.Errno = 0x20000000 | 0xC0000000 | 0x22C
	ERROR_NON_WINDOWS_NT_DRIVER                  syscall.Errno = 0x20000000 | 0xC0000000 | 0x22D
	ERROR_NON_WINDOWS_DRIVER                     syscall.Errno = 0x20000000 | 0xC0000000 | 0x22E
	ERROR_NO_CATALOG_FOR_OEM_INF                 syscall.Errno = 0x20000000 | 0xC0000000 | 0x22F
	ERROR_DEVINSTALL_QUEUE_NONNATIVE             syscall.Errno = 0x20000000 | 0xC0000000 | 0x230
	ERROR_NOT_DISABLEABLE                        syscall.Errno = 0x20000000 | 0xC0000000 | 0x231
	ERROR_CANT_REMOVE_DEVINST                    syscall.Errno = 0x20000000 | 0xC0000000 | 0x232
	ERROR_INVALID_TARGET                         syscall.Errno = 0x20000000 | 0xC0000000 | 0x233
	ERROR_DRIVER_NONNATIVE                       syscall.Errno = 0x20000000 | 0xC0000000 | 0x234
	ERROR_IN_WOW64                               syscall.Errno = 0x20000000 | 0xC0000000 | 0x235
	ERROR_SET_SYSTEM_RESTORE_POINT               syscall.Errno = 0x20000000 | 0xC0000000 | 0x236
	ERROR_SCE_DISABLED                           syscall.Errno = 0x20000000 | 0xC0000000 | 0x238
	ERROR_UNKNOWN_EXCEPTION                      syscall.Errno = 0x20000000 | 0xC0000000 | 0x239
	ERROR_PNP_REGISTRY_ERROR                     syscall.Errno = 0x20000000 | 0xC0000000 | 0x23A
	ERROR_REMOTE_REQUEST_UNSUPPORTED             syscall.Errno = 0x20000000 | 0xC0000000 | 0x23B
	ERROR_NOT_AN_INSTALLED_OEM_INF               syscall.Errno = 0x20000000 | 0xC0000000 | 0x23C
	ERROR_INF_IN_USE_BY_DEVICES                  syscall.Errno = 0x20000000 | 0xC0000000 | 0x23D
	ERROR_DI_FUNCTION_OBSOLETE                   syscall.Errno = 0x20000000 | 0xC0000000 | 0x23E
	ERROR_NO_AUTHENTICODE_CATALOG                syscall.Errno = 0x20000000 | 0xC0000000 | 0x23F
	ERROR_AUTHENTICODE_DISALLOWED                syscall.Errno = 0x20000000 | 0xC0000000 | 0x240
	ERROR_AUTHENTICODE_TRUSTED_PUBLISHER         syscall.Errno = 0x20000000 | 0xC0000000 | 0x241
	ERROR_AUTHENTICODE_TRUST_NOT_ESTABLISHED     syscall.Errno = 0x20000000 | 0xC0000000 | 0x242
	ERROR_AUTHENTICODE_PUBLISHER_NOT_TRUSTED     syscall.Errno = 0x20000000 | 0xC0000000 | 0x243
	ERROR_SIGNATURE_OSATTRIBUTE_MISMATCH         syscall.Errno = 0x20000000 | 0xC0000000 | 0x244
	ERROR_ONLY_VALIDATE_VIA_AUTHENTICODE         syscall.Errno = 0x20000000 | 0xC0000000 | 0x245
	ERROR_DEVICE_INSTALLER_NOT_READY             syscall.Errno = 0x20000000 | 0xC0000000 | 0x246
	ERROR_DRIVER_STORE_ADD_FAILED                syscall.Errno = 0x20000000 | 0xC0000000 | 0x247
	ERROR_DEVICE_INSTALL_BLOCKED                 syscall.Errno = 0x20000000 | 0xC0000000 | 0x248
	ERROR_DRIVER_INSTALL_BLOCKED                 syscall.Errno = 0x20000000 | 0xC0000000 | 0x249
	ERROR_WRONG_INF_TYPE                         syscall.Errno = 0x20000000 | 0xC0000000 | 0x24A
	ERROR_FILE_HASH_NOT_IN_CATALOG               syscall.Errno = 0x20000000 | 0xC0000000 | 0x24B
	ERROR_DRIVER_STORE_DELETE_FAILED             syscall.Errno = 0x20000000 | 0xC0000000 | 0x24C
	ERROR_UNRECOVERABLE_STACK_OVERFLOW           syscall.Errno = 0x20000000 | 0xC0000000 | 0x300
	EXCEPTION_SPAPI_UNRECOVERABLE_STACK_OVERFLOW syscall.Errno = ERROR_UNRECOVERABLE_STACK_OVERFLOW
	ERROR_NO_DEFAULT_INTERFACE_DEVICE            syscall.Errno = ERROR_NO_DEFAULT_DEVICE_INTERFACE
	ERROR_INTERFACE_DEVICE_ACTIVE                syscall.Errno = ERROR_DEVICE_INTERFACE_ACTIVE
	ERROR_INTERFACE_DEVICE_REMOVED               syscall.Errno = ERROR_DEVICE_INTERFACE_REMOVED
	ERROR_NO_SUCH_INTERFACE_DEVICE               syscall.Errno = ERROR_NO_SUCH_DEVICE_INTERFACE
)
