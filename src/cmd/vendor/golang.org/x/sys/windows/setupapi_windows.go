// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"encoding/binary"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"syscall"
	"unsafe"
)

// This file contains functions that wrap SetupAPI.dll and CfgMgr32.dll,
// core system functions for managing hardware devices, drivers, and the PnP tree.
// Information about these APIs can be found at:
//     https://docs.microsoft.com/en-us/windows-hardware/drivers/install/setupapi
//     https://docs.microsoft.com/en-us/windows/win32/devinst/cfgmgr32-

const (
	ERROR_EXPECTED_SECTION_NAME                  Errno = 0x20000000 | 0xC0000000 | 0
	ERROR_BAD_SECTION_NAME_LINE                  Errno = 0x20000000 | 0xC0000000 | 1
	ERROR_SECTION_NAME_TOO_LONG                  Errno = 0x20000000 | 0xC0000000 | 2
	ERROR_GENERAL_SYNTAX                         Errno = 0x20000000 | 0xC0000000 | 3
	ERROR_WRONG_INF_STYLE                        Errno = 0x20000000 | 0xC0000000 | 0x100
	ERROR_SECTION_NOT_FOUND                      Errno = 0x20000000 | 0xC0000000 | 0x101
	ERROR_LINE_NOT_FOUND                         Errno = 0x20000000 | 0xC0000000 | 0x102
	ERROR_NO_BACKUP                              Errno = 0x20000000 | 0xC0000000 | 0x103
	ERROR_NO_ASSOCIATED_CLASS                    Errno = 0x20000000 | 0xC0000000 | 0x200
	ERROR_CLASS_MISMATCH                         Errno = 0x20000000 | 0xC0000000 | 0x201
	ERROR_DUPLICATE_FOUND                        Errno = 0x20000000 | 0xC0000000 | 0x202
	ERROR_NO_DRIVER_SELECTED                     Errno = 0x20000000 | 0xC0000000 | 0x203
	ERROR_KEY_DOES_NOT_EXIST                     Errno = 0x20000000 | 0xC0000000 | 0x204
	ERROR_INVALID_DEVINST_NAME                   Errno = 0x20000000 | 0xC0000000 | 0x205
	ERROR_INVALID_CLASS                          Errno = 0x20000000 | 0xC0000000 | 0x206
	ERROR_DEVINST_ALREADY_EXISTS                 Errno = 0x20000000 | 0xC0000000 | 0x207
	ERROR_DEVINFO_NOT_REGISTERED                 Errno = 0x20000000 | 0xC0000000 | 0x208
	ERROR_INVALID_REG_PROPERTY                   Errno = 0x20000000 | 0xC0000000 | 0x209
	ERROR_NO_INF                                 Errno = 0x20000000 | 0xC0000000 | 0x20A
	ERROR_NO_SUCH_DEVINST                        Errno = 0x20000000 | 0xC0000000 | 0x20B
	ERROR_CANT_LOAD_CLASS_ICON                   Errno = 0x20000000 | 0xC0000000 | 0x20C
	ERROR_INVALID_CLASS_INSTALLER                Errno = 0x20000000 | 0xC0000000 | 0x20D
	ERROR_DI_DO_DEFAULT                          Errno = 0x20000000 | 0xC0000000 | 0x20E
	ERROR_DI_NOFILECOPY                          Errno = 0x20000000 | 0xC0000000 | 0x20F
	ERROR_INVALID_HWPROFILE                      Errno = 0x20000000 | 0xC0000000 | 0x210
	ERROR_NO_DEVICE_SELECTED                     Errno = 0x20000000 | 0xC0000000 | 0x211
	ERROR_DEVINFO_LIST_LOCKED                    Errno = 0x20000000 | 0xC0000000 | 0x212
	ERROR_DEVINFO_DATA_LOCKED                    Errno = 0x20000000 | 0xC0000000 | 0x213
	ERROR_DI_BAD_PATH                            Errno = 0x20000000 | 0xC0000000 | 0x214
	ERROR_NO_CLASSINSTALL_PARAMS                 Errno = 0x20000000 | 0xC0000000 | 0x215
	ERROR_FILEQUEUE_LOCKED                       Errno = 0x20000000 | 0xC0000000 | 0x216
	ERROR_BAD_SERVICE_INSTALLSECT                Errno = 0x20000000 | 0xC0000000 | 0x217
	ERROR_NO_CLASS_DRIVER_LIST                   Errno = 0x20000000 | 0xC0000000 | 0x218
	ERROR_NO_ASSOCIATED_SERVICE                  Errno = 0x20000000 | 0xC0000000 | 0x219
	ERROR_NO_DEFAULT_DEVICE_INTERFACE            Errno = 0x20000000 | 0xC0000000 | 0x21A
	ERROR_DEVICE_INTERFACE_ACTIVE                Errno = 0x20000000 | 0xC0000000 | 0x21B
	ERROR_DEVICE_INTERFACE_REMOVED               Errno = 0x20000000 | 0xC0000000 | 0x21C
	ERROR_BAD_INTERFACE_INSTALLSECT              Errno = 0x20000000 | 0xC0000000 | 0x21D
	ERROR_NO_SUCH_INTERFACE_CLASS                Errno = 0x20000000 | 0xC0000000 | 0x21E
	ERROR_INVALID_REFERENCE_STRING               Errno = 0x20000000 | 0xC0000000 | 0x21F
	ERROR_INVALID_MACHINENAME                    Errno = 0x20000000 | 0xC0000000 | 0x220
	ERROR_REMOTE_COMM_FAILURE                    Errno = 0x20000000 | 0xC0000000 | 0x221
	ERROR_MACHINE_UNAVAILABLE                    Errno = 0x20000000 | 0xC0000000 | 0x222
	ERROR_NO_CONFIGMGR_SERVICES                  Errno = 0x20000000 | 0xC0000000 | 0x223
	ERROR_INVALID_PROPPAGE_PROVIDER              Errno = 0x20000000 | 0xC0000000 | 0x224
	ERROR_NO_SUCH_DEVICE_INTERFACE               Errno = 0x20000000 | 0xC0000000 | 0x225
	ERROR_DI_POSTPROCESSING_REQUIRED             Errno = 0x20000000 | 0xC0000000 | 0x226
	ERROR_INVALID_COINSTALLER                    Errno = 0x20000000 | 0xC0000000 | 0x227
	ERROR_NO_COMPAT_DRIVERS                      Errno = 0x20000000 | 0xC0000000 | 0x228
	ERROR_NO_DEVICE_ICON                         Errno = 0x20000000 | 0xC0000000 | 0x229
	ERROR_INVALID_INF_LOGCONFIG                  Errno = 0x20000000 | 0xC0000000 | 0x22A
	ERROR_DI_DONT_INSTALL                        Errno = 0x20000000 | 0xC0000000 | 0x22B
	ERROR_INVALID_FILTER_DRIVER                  Errno = 0x20000000 | 0xC0000000 | 0x22C
	ERROR_NON_WINDOWS_NT_DRIVER                  Errno = 0x20000000 | 0xC0000000 | 0x22D
	ERROR_NON_WINDOWS_DRIVER                     Errno = 0x20000000 | 0xC0000000 | 0x22E
	ERROR_NO_CATALOG_FOR_OEM_INF                 Errno = 0x20000000 | 0xC0000000 | 0x22F
	ERROR_DEVINSTALL_QUEUE_NONNATIVE             Errno = 0x20000000 | 0xC0000000 | 0x230
	ERROR_NOT_DISABLEABLE                        Errno = 0x20000000 | 0xC0000000 | 0x231
	ERROR_CANT_REMOVE_DEVINST                    Errno = 0x20000000 | 0xC0000000 | 0x232
	ERROR_INVALID_TARGET                         Errno = 0x20000000 | 0xC0000000 | 0x233
	ERROR_DRIVER_NONNATIVE                       Errno = 0x20000000 | 0xC0000000 | 0x234
	ERROR_IN_WOW64                               Errno = 0x20000000 | 0xC0000000 | 0x235
	ERROR_SET_SYSTEM_RESTORE_POINT               Errno = 0x20000000 | 0xC0000000 | 0x236
	ERROR_SCE_DISABLED                           Errno = 0x20000000 | 0xC0000000 | 0x238
	ERROR_UNKNOWN_EXCEPTION                      Errno = 0x20000000 | 0xC0000000 | 0x239
	ERROR_PNP_REGISTRY_ERROR                     Errno = 0x20000000 | 0xC0000000 | 0x23A
	ERROR_REMOTE_REQUEST_UNSUPPORTED             Errno = 0x20000000 | 0xC0000000 | 0x23B
	ERROR_NOT_AN_INSTALLED_OEM_INF               Errno = 0x20000000 | 0xC0000000 | 0x23C
	ERROR_INF_IN_USE_BY_DEVICES                  Errno = 0x20000000 | 0xC0000000 | 0x23D
	ERROR_DI_FUNCTION_OBSOLETE                   Errno = 0x20000000 | 0xC0000000 | 0x23E
	ERROR_NO_AUTHENTICODE_CATALOG                Errno = 0x20000000 | 0xC0000000 | 0x23F
	ERROR_AUTHENTICODE_DISALLOWED                Errno = 0x20000000 | 0xC0000000 | 0x240
	ERROR_AUTHENTICODE_TRUSTED_PUBLISHER         Errno = 0x20000000 | 0xC0000000 | 0x241
	ERROR_AUTHENTICODE_TRUST_NOT_ESTABLISHED     Errno = 0x20000000 | 0xC0000000 | 0x242
	ERROR_AUTHENTICODE_PUBLISHER_NOT_TRUSTED     Errno = 0x20000000 | 0xC0000000 | 0x243
	ERROR_SIGNATURE_OSATTRIBUTE_MISMATCH         Errno = 0x20000000 | 0xC0000000 | 0x244
	ERROR_ONLY_VALIDATE_VIA_AUTHENTICODE         Errno = 0x20000000 | 0xC0000000 | 0x245
	ERROR_DEVICE_INSTALLER_NOT_READY             Errno = 0x20000000 | 0xC0000000 | 0x246
	ERROR_DRIVER_STORE_ADD_FAILED                Errno = 0x20000000 | 0xC0000000 | 0x247
	ERROR_DEVICE_INSTALL_BLOCKED                 Errno = 0x20000000 | 0xC0000000 | 0x248
	ERROR_DRIVER_INSTALL_BLOCKED                 Errno = 0x20000000 | 0xC0000000 | 0x249
	ERROR_WRONG_INF_TYPE                         Errno = 0x20000000 | 0xC0000000 | 0x24A
	ERROR_FILE_HASH_NOT_IN_CATALOG               Errno = 0x20000000 | 0xC0000000 | 0x24B
	ERROR_DRIVER_STORE_DELETE_FAILED             Errno = 0x20000000 | 0xC0000000 | 0x24C
	ERROR_UNRECOVERABLE_STACK_OVERFLOW           Errno = 0x20000000 | 0xC0000000 | 0x300
	EXCEPTION_SPAPI_UNRECOVERABLE_STACK_OVERFLOW Errno = ERROR_UNRECOVERABLE_STACK_OVERFLOW
	ERROR_NO_DEFAULT_INTERFACE_DEVICE            Errno = ERROR_NO_DEFAULT_DEVICE_INTERFACE
	ERROR_INTERFACE_DEVICE_ACTIVE                Errno = ERROR_DEVICE_INTERFACE_ACTIVE
	ERROR_INTERFACE_DEVICE_REMOVED               Errno = ERROR_DEVICE_INTERFACE_REMOVED
	ERROR_NO_SUCH_INTERFACE_DEVICE               Errno = ERROR_NO_SUCH_DEVICE_INTERFACE
)

const (
	MAX_DEVICE_ID_LEN   = 200
	MAX_DEVNODE_ID_LEN  = MAX_DEVICE_ID_LEN
	MAX_GUID_STRING_LEN = 39 // 38 chars + terminator null
	MAX_CLASS_NAME_LEN  = 32
	MAX_PROFILE_LEN     = 80
	MAX_CONFIG_VALUE    = 9999
	MAX_INSTANCE_VALUE  = 9999
	CONFIGMG_VERSION    = 0x0400
)

// Maximum string length constants
const (
	LINE_LEN                    = 256  // Windows 9x-compatible maximum for displayable strings coming from a device INF.
	MAX_INF_STRING_LENGTH       = 4096 // Actual maximum size of an INF string (including string substitutions).
	MAX_INF_SECTION_NAME_LENGTH = 255  // For Windows 9x compatibility, INF section names should be constrained to 32 characters.
	MAX_TITLE_LEN               = 60
	MAX_INSTRUCTION_LEN         = 256
	MAX_LABEL_LEN               = 30
	MAX_SERVICE_NAME_LEN        = 256
	MAX_SUBTITLE_LEN            = 256
)

const (
	// SP_MAX_MACHINENAME_LENGTH defines maximum length of a machine name in the format expected by ConfigMgr32 CM_Connect_Machine (i.e., "\\\\MachineName\0").
	SP_MAX_MACHINENAME_LENGTH = MAX_PATH + 3
)

// HSPFILEQ is type for setup file queue
type HSPFILEQ uintptr

// DevInfo holds reference to device information set
type DevInfo Handle

// DEVINST is a handle usually recognized by cfgmgr32 APIs
type DEVINST uint32

// DevInfoData is a device information structure (references a device instance that is a member of a device information set)
type DevInfoData struct {
	size      uint32
	ClassGUID GUID
	DevInst   DEVINST
	_         uintptr
}

// DevInfoListDetailData is a structure for detailed information on a device information set (used for SetupDiGetDeviceInfoListDetail which supersedes the functionality of SetupDiGetDeviceInfoListClass).
type DevInfoListDetailData struct {
	size                uint32 // Use unsafeSizeOf method
	ClassGUID           GUID
	RemoteMachineHandle Handle
	remoteMachineName   [SP_MAX_MACHINENAME_LENGTH]uint16
}

func (*DevInfoListDetailData) unsafeSizeOf() uint32 {
	if unsafe.Sizeof(uintptr(0)) == 4 {
		// Windows declares this with pshpack1.h
		return uint32(unsafe.Offsetof(DevInfoListDetailData{}.remoteMachineName) + unsafe.Sizeof(DevInfoListDetailData{}.remoteMachineName))
	}
	return uint32(unsafe.Sizeof(DevInfoListDetailData{}))
}

func (data *DevInfoListDetailData) RemoteMachineName() string {
	return UTF16ToString(data.remoteMachineName[:])
}

func (data *DevInfoListDetailData) SetRemoteMachineName(remoteMachineName string) error {
	str, err := UTF16FromString(remoteMachineName)
	if err != nil {
		return err
	}
	copy(data.remoteMachineName[:], str)
	return nil
}

// DI_FUNCTION is function type for device installer
type DI_FUNCTION uint32

const (
	DIF_SELECTDEVICE                   DI_FUNCTION = 0x00000001
	DIF_INSTALLDEVICE                  DI_FUNCTION = 0x00000002
	DIF_ASSIGNRESOURCES                DI_FUNCTION = 0x00000003
	DIF_PROPERTIES                     DI_FUNCTION = 0x00000004
	DIF_REMOVE                         DI_FUNCTION = 0x00000005
	DIF_FIRSTTIMESETUP                 DI_FUNCTION = 0x00000006
	DIF_FOUNDDEVICE                    DI_FUNCTION = 0x00000007
	DIF_SELECTCLASSDRIVERS             DI_FUNCTION = 0x00000008
	DIF_VALIDATECLASSDRIVERS           DI_FUNCTION = 0x00000009
	DIF_INSTALLCLASSDRIVERS            DI_FUNCTION = 0x0000000A
	DIF_CALCDISKSPACE                  DI_FUNCTION = 0x0000000B
	DIF_DESTROYPRIVATEDATA             DI_FUNCTION = 0x0000000C
	DIF_VALIDATEDRIVER                 DI_FUNCTION = 0x0000000D
	DIF_DETECT                         DI_FUNCTION = 0x0000000F
	DIF_INSTALLWIZARD                  DI_FUNCTION = 0x00000010
	DIF_DESTROYWIZARDDATA              DI_FUNCTION = 0x00000011
	DIF_PROPERTYCHANGE                 DI_FUNCTION = 0x00000012
	DIF_ENABLECLASS                    DI_FUNCTION = 0x00000013
	DIF_DETECTVERIFY                   DI_FUNCTION = 0x00000014
	DIF_INSTALLDEVICEFILES             DI_FUNCTION = 0x00000015
	DIF_UNREMOVE                       DI_FUNCTION = 0x00000016
	DIF_SELECTBESTCOMPATDRV            DI_FUNCTION = 0x00000017
	DIF_ALLOW_INSTALL                  DI_FUNCTION = 0x00000018
	DIF_REGISTERDEVICE                 DI_FUNCTION = 0x00000019
	DIF_NEWDEVICEWIZARD_PRESELECT      DI_FUNCTION = 0x0000001A
	DIF_NEWDEVICEWIZARD_SELECT         DI_FUNCTION = 0x0000001B
	DIF_NEWDEVICEWIZARD_PREANALYZE     DI_FUNCTION = 0x0000001C
	DIF_NEWDEVICEWIZARD_POSTANALYZE    DI_FUNCTION = 0x0000001D
	DIF_NEWDEVICEWIZARD_FINISHINSTALL  DI_FUNCTION = 0x0000001E
	DIF_INSTALLINTERFACES              DI_FUNCTION = 0x00000020
	DIF_DETECTCANCEL                   DI_FUNCTION = 0x00000021
	DIF_REGISTER_COINSTALLERS          DI_FUNCTION = 0x00000022
	DIF_ADDPROPERTYPAGE_ADVANCED       DI_FUNCTION = 0x00000023
	DIF_ADDPROPERTYPAGE_BASIC          DI_FUNCTION = 0x00000024
	DIF_TROUBLESHOOTER                 DI_FUNCTION = 0x00000026
	DIF_POWERMESSAGEWAKE               DI_FUNCTION = 0x00000027
	DIF_ADDREMOTEPROPERTYPAGE_ADVANCED DI_FUNCTION = 0x00000028
	DIF_UPDATEDRIVER_UI                DI_FUNCTION = 0x00000029
	DIF_FINISHINSTALL_ACTION           DI_FUNCTION = 0x0000002A
)

// DevInstallParams is device installation parameters structure (associated with a particular device information element, or globally with a device information set)
type DevInstallParams struct {
	size                     uint32
	Flags                    DI_FLAGS
	FlagsEx                  DI_FLAGSEX
	hwndParent               uintptr
	InstallMsgHandler        uintptr
	InstallMsgHandlerContext uintptr
	FileQueue                HSPFILEQ
	_                        uintptr
	_                        uint32
	driverPath               [MAX_PATH]uint16
}

func (params *DevInstallParams) DriverPath() string {
	return UTF16ToString(params.driverPath[:])
}

func (params *DevInstallParams) SetDriverPath(driverPath string) error {
	str, err := UTF16FromString(driverPath)
	if err != nil {
		return err
	}
	copy(params.driverPath[:], str)
	return nil
}

// DI_FLAGS is SP_DEVINSTALL_PARAMS.Flags values
type DI_FLAGS uint32

const (
	// Flags for choosing a device
	DI_SHOWOEM       DI_FLAGS = 0x00000001 // support Other... button
	DI_SHOWCOMPAT    DI_FLAGS = 0x00000002 // show compatibility list
	DI_SHOWCLASS     DI_FLAGS = 0x00000004 // show class list
	DI_SHOWALL       DI_FLAGS = 0x00000007 // both class & compat list shown
	DI_NOVCP         DI_FLAGS = 0x00000008 // don't create a new copy queue--use caller-supplied FileQueue
	DI_DIDCOMPAT     DI_FLAGS = 0x00000010 // Searched for compatible devices
	DI_DIDCLASS      DI_FLAGS = 0x00000020 // Searched for class devices
	DI_AUTOASSIGNRES DI_FLAGS = 0x00000040 // No UI for resources if possible

	// Flags returned by DiInstallDevice to indicate need to reboot/restart
	DI_NEEDRESTART DI_FLAGS = 0x00000080 // Reboot required to take effect
	DI_NEEDREBOOT  DI_FLAGS = 0x00000100 // ""

	// Flags for device installation
	DI_NOBROWSE DI_FLAGS = 0x00000200 // no Browse... in InsertDisk

	// Flags set by DiBuildDriverInfoList
	DI_MULTMFGS DI_FLAGS = 0x00000400 // Set if multiple manufacturers in class driver list

	// Flag indicates that device is disabled
	DI_DISABLED DI_FLAGS = 0x00000800 // Set if device disabled

	// Flags for Device/Class Properties
	DI_GENERALPAGE_ADDED  DI_FLAGS = 0x00001000
	DI_RESOURCEPAGE_ADDED DI_FLAGS = 0x00002000

	// Flag to indicate the setting properties for this Device (or class) caused a change so the Dev Mgr UI probably needs to be updated.
	DI_PROPERTIES_CHANGE DI_FLAGS = 0x00004000

	// Flag to indicate that the sorting from the INF file should be used.
	DI_INF_IS_SORTED DI_FLAGS = 0x00008000

	// Flag to indicate that only the the INF specified by SP_DEVINSTALL_PARAMS.DriverPath should be searched.
	DI_ENUMSINGLEINF DI_FLAGS = 0x00010000

	// Flag that prevents ConfigMgr from removing/re-enumerating devices during device
	// registration, installation, and deletion.
	DI_DONOTCALLCONFIGMG DI_FLAGS = 0x00020000

	// The following flag can be used to install a device disabled
	DI_INSTALLDISABLED DI_FLAGS = 0x00040000

	// Flag that causes SetupDiBuildDriverInfoList to build a device's compatible driver
	// list from its existing class driver list, instead of the normal INF search.
	DI_COMPAT_FROM_CLASS DI_FLAGS = 0x00080000

	// This flag is set if the Class Install params should be used.
	DI_CLASSINSTALLPARAMS DI_FLAGS = 0x00100000

	// This flag is set if the caller of DiCallClassInstaller does NOT want the internal default action performed if the Class installer returns ERROR_DI_DO_DEFAULT.
	DI_NODI_DEFAULTACTION DI_FLAGS = 0x00200000

	// Flags for device installation
	DI_QUIETINSTALL        DI_FLAGS = 0x00800000 // don't confuse the user with questions or excess info
	DI_NOFILECOPY          DI_FLAGS = 0x01000000 // No file Copy necessary
	DI_FORCECOPY           DI_FLAGS = 0x02000000 // Force files to be copied from install path
	DI_DRIVERPAGE_ADDED    DI_FLAGS = 0x04000000 // Prop provider added Driver page.
	DI_USECI_SELECTSTRINGS DI_FLAGS = 0x08000000 // Use Class Installer Provided strings in the Select Device Dlg
	DI_OVERRIDE_INFFLAGS   DI_FLAGS = 0x10000000 // Override INF flags
	DI_PROPS_NOCHANGEUSAGE DI_FLAGS = 0x20000000 // No Enable/Disable in General Props

	DI_NOSELECTICONS DI_FLAGS = 0x40000000 // No small icons in select device dialogs

	DI_NOWRITE_IDS DI_FLAGS = 0x80000000 // Don't write HW & Compat IDs on install
)

// DI_FLAGSEX is SP_DEVINSTALL_PARAMS.FlagsEx values
type DI_FLAGSEX uint32

const (
	DI_FLAGSEX_CI_FAILED                DI_FLAGSEX = 0x00000004 // Failed to Load/Call class installer
	DI_FLAGSEX_FINISHINSTALL_ACTION     DI_FLAGSEX = 0x00000008 // Class/co-installer wants to get a DIF_FINISH_INSTALL action in client context.
	DI_FLAGSEX_DIDINFOLIST              DI_FLAGSEX = 0x00000010 // Did the Class Info List
	DI_FLAGSEX_DIDCOMPATINFO            DI_FLAGSEX = 0x00000020 // Did the Compat Info List
	DI_FLAGSEX_FILTERCLASSES            DI_FLAGSEX = 0x00000040
	DI_FLAGSEX_SETFAILEDINSTALL         DI_FLAGSEX = 0x00000080
	DI_FLAGSEX_DEVICECHANGE             DI_FLAGSEX = 0x00000100
	DI_FLAGSEX_ALWAYSWRITEIDS           DI_FLAGSEX = 0x00000200
	DI_FLAGSEX_PROPCHANGE_PENDING       DI_FLAGSEX = 0x00000400 // One or more device property sheets have had changes made to them, and need to have a DIF_PROPERTYCHANGE occur.
	DI_FLAGSEX_ALLOWEXCLUDEDDRVS        DI_FLAGSEX = 0x00000800
	DI_FLAGSEX_NOUIONQUERYREMOVE        DI_FLAGSEX = 0x00001000
	DI_FLAGSEX_USECLASSFORCOMPAT        DI_FLAGSEX = 0x00002000 // Use the device's class when building compat drv list. (Ignored if DI_COMPAT_FROM_CLASS flag is specified.)
	DI_FLAGSEX_NO_DRVREG_MODIFY         DI_FLAGSEX = 0x00008000 // Don't run AddReg and DelReg for device's software (driver) key.
	DI_FLAGSEX_IN_SYSTEM_SETUP          DI_FLAGSEX = 0x00010000 // Installation is occurring during initial system setup.
	DI_FLAGSEX_INET_DRIVER              DI_FLAGSEX = 0x00020000 // Driver came from Windows Update
	DI_FLAGSEX_APPENDDRIVERLIST         DI_FLAGSEX = 0x00040000 // Cause SetupDiBuildDriverInfoList to append a new driver list to an existing list.
	DI_FLAGSEX_PREINSTALLBACKUP         DI_FLAGSEX = 0x00080000 // not used
	DI_FLAGSEX_BACKUPONREPLACE          DI_FLAGSEX = 0x00100000 // not used
	DI_FLAGSEX_DRIVERLIST_FROM_URL      DI_FLAGSEX = 0x00200000 // build driver list from INF(s) retrieved from URL specified in SP_DEVINSTALL_PARAMS.DriverPath (empty string means Windows Update website)
	DI_FLAGSEX_EXCLUDE_OLD_INET_DRIVERS DI_FLAGSEX = 0x00800000 // Don't include old Internet drivers when building a driver list. Ignored on Windows Vista and later.
	DI_FLAGSEX_POWERPAGE_ADDED          DI_FLAGSEX = 0x01000000 // class installer added their own power page
	DI_FLAGSEX_FILTERSIMILARDRIVERS     DI_FLAGSEX = 0x02000000 // only include similar drivers in class list
	DI_FLAGSEX_INSTALLEDDRIVER          DI_FLAGSEX = 0x04000000 // only add the installed driver to the class or compat driver list.  Used in calls to SetupDiBuildDriverInfoList
	DI_FLAGSEX_NO_CLASSLIST_NODE_MERGE  DI_FLAGSEX = 0x08000000 // Don't remove identical driver nodes from the class list
	DI_FLAGSEX_ALTPLATFORM_DRVSEARCH    DI_FLAGSEX = 0x10000000 // Build driver list based on alternate platform information specified in associated file queue
	DI_FLAGSEX_RESTART_DEVICE_ONLY      DI_FLAGSEX = 0x20000000 // only restart the device drivers are being installed on as opposed to restarting all devices using those drivers.
	DI_FLAGSEX_RECURSIVESEARCH          DI_FLAGSEX = 0x40000000 // Tell SetupDiBuildDriverInfoList to do a recursive search
	DI_FLAGSEX_SEARCH_PUBLISHED_INFS    DI_FLAGSEX = 0x80000000 // Tell SetupDiBuildDriverInfoList to do a "published INF" search
)

// ClassInstallHeader is the first member of any class install parameters structure. It contains the device installation request code that defines the format of the rest of the install parameters structure.
type ClassInstallHeader struct {
	size            uint32
	InstallFunction DI_FUNCTION
}

func MakeClassInstallHeader(installFunction DI_FUNCTION) *ClassInstallHeader {
	hdr := &ClassInstallHeader{InstallFunction: installFunction}
	hdr.size = uint32(unsafe.Sizeof(*hdr))
	return hdr
}

// DICS_STATE specifies values indicating a change in a device's state
type DICS_STATE uint32

const (
	DICS_ENABLE     DICS_STATE = 0x00000001 // The device is being enabled.
	DICS_DISABLE    DICS_STATE = 0x00000002 // The device is being disabled.
	DICS_PROPCHANGE DICS_STATE = 0x00000003 // The properties of the device have changed.
	DICS_START      DICS_STATE = 0x00000004 // The device is being started (if the request is for the currently active hardware profile).
	DICS_STOP       DICS_STATE = 0x00000005 // The device is being stopped. The driver stack will be unloaded and the CSCONFIGFLAG_DO_NOT_START flag will be set for the device.
)

// DICS_FLAG specifies the scope of a device property change
type DICS_FLAG uint32

const (
	DICS_FLAG_GLOBAL         DICS_FLAG = 0x00000001 // make change in all hardware profiles
	DICS_FLAG_CONFIGSPECIFIC DICS_FLAG = 0x00000002 // make change in specified profile only
	DICS_FLAG_CONFIGGENERAL  DICS_FLAG = 0x00000004 // 1 or more hardware profile-specific changes to follow (obsolete)
)

// PropChangeParams is a structure corresponding to a DIF_PROPERTYCHANGE install function.
type PropChangeParams struct {
	ClassInstallHeader ClassInstallHeader
	StateChange        DICS_STATE
	Scope              DICS_FLAG
	HwProfile          uint32
}

// DI_REMOVEDEVICE specifies the scope of the device removal
type DI_REMOVEDEVICE uint32

const (
	DI_REMOVEDEVICE_GLOBAL         DI_REMOVEDEVICE = 0x00000001 // Make this change in all hardware profiles. Remove information about the device from the registry.
	DI_REMOVEDEVICE_CONFIGSPECIFIC DI_REMOVEDEVICE = 0x00000002 // Make this change to only the hardware profile specified by HwProfile. this flag only applies to root-enumerated devices. When Windows removes the device from the last hardware profile in which it was configured, Windows performs a global removal.
)

// RemoveDeviceParams is a structure corresponding to a DIF_REMOVE install function.
type RemoveDeviceParams struct {
	ClassInstallHeader ClassInstallHeader
	Scope              DI_REMOVEDEVICE
	HwProfile          uint32
}

// DrvInfoData is driver information structure (member of a driver info list that may be associated with a particular device instance, or (globally) with a device information set)
type DrvInfoData struct {
	size          uint32
	DriverType    uint32
	_             uintptr
	description   [LINE_LEN]uint16
	mfgName       [LINE_LEN]uint16
	providerName  [LINE_LEN]uint16
	DriverDate    Filetime
	DriverVersion uint64
}

func (data *DrvInfoData) Description() string {
	return UTF16ToString(data.description[:])
}

func (data *DrvInfoData) SetDescription(description string) error {
	str, err := UTF16FromString(description)
	if err != nil {
		return err
	}
	copy(data.description[:], str)
	return nil
}

func (data *DrvInfoData) MfgName() string {
	return UTF16ToString(data.mfgName[:])
}

func (data *DrvInfoData) SetMfgName(mfgName string) error {
	str, err := UTF16FromString(mfgName)
	if err != nil {
		return err
	}
	copy(data.mfgName[:], str)
	return nil
}

func (data *DrvInfoData) ProviderName() string {
	return UTF16ToString(data.providerName[:])
}

func (data *DrvInfoData) SetProviderName(providerName string) error {
	str, err := UTF16FromString(providerName)
	if err != nil {
		return err
	}
	copy(data.providerName[:], str)
	return nil
}

// IsNewer method returns true if DrvInfoData date and version is newer than supplied parameters.
func (data *DrvInfoData) IsNewer(driverDate Filetime, driverVersion uint64) bool {
	if data.DriverDate.HighDateTime > driverDate.HighDateTime {
		return true
	}
	if data.DriverDate.HighDateTime < driverDate.HighDateTime {
		return false
	}

	if data.DriverDate.LowDateTime > driverDate.LowDateTime {
		return true
	}
	if data.DriverDate.LowDateTime < driverDate.LowDateTime {
		return false
	}

	if data.DriverVersion > driverVersion {
		return true
	}
	if data.DriverVersion < driverVersion {
		return false
	}

	return false
}

// DrvInfoDetailData is driver information details structure (provides detailed information about a particular driver information structure)
type DrvInfoDetailData struct {
	size            uint32 // Use unsafeSizeOf method
	InfDate         Filetime
	compatIDsOffset uint32
	compatIDsLength uint32
	_               uintptr
	sectionName     [LINE_LEN]uint16
	infFileName     [MAX_PATH]uint16
	drvDescription  [LINE_LEN]uint16
	hardwareID      [1]uint16
}

func (*DrvInfoDetailData) unsafeSizeOf() uint32 {
	if unsafe.Sizeof(uintptr(0)) == 4 {
		// Windows declares this with pshpack1.h
		return uint32(unsafe.Offsetof(DrvInfoDetailData{}.hardwareID) + unsafe.Sizeof(DrvInfoDetailData{}.hardwareID))
	}
	return uint32(unsafe.Sizeof(DrvInfoDetailData{}))
}

func (data *DrvInfoDetailData) SectionName() string {
	return UTF16ToString(data.sectionName[:])
}

func (data *DrvInfoDetailData) InfFileName() string {
	return UTF16ToString(data.infFileName[:])
}

func (data *DrvInfoDetailData) DrvDescription() string {
	return UTF16ToString(data.drvDescription[:])
}

func (data *DrvInfoDetailData) HardwareID() string {
	if data.compatIDsOffset > 1 {
		bufW := data.getBuf()
		return UTF16ToString(bufW[:wcslen(bufW)])
	}

	return ""
}

func (data *DrvInfoDetailData) CompatIDs() []string {
	a := make([]string, 0)

	if data.compatIDsLength > 0 {
		bufW := data.getBuf()
		bufW = bufW[data.compatIDsOffset : data.compatIDsOffset+data.compatIDsLength]
		for i := 0; i < len(bufW); {
			j := i + wcslen(bufW[i:])
			if i < j {
				a = append(a, UTF16ToString(bufW[i:j]))
			}
			i = j + 1
		}
	}

	return a
}

func (data *DrvInfoDetailData) getBuf() []uint16 {
	len := (data.size - uint32(unsafe.Offsetof(data.hardwareID))) / 2
	sl := struct {
		addr *uint16
		len  int
		cap  int
	}{&data.hardwareID[0], int(len), int(len)}
	return *(*[]uint16)(unsafe.Pointer(&sl))
}

// IsCompatible method tests if given hardware ID matches the driver or is listed on the compatible ID list.
func (data *DrvInfoDetailData) IsCompatible(hwid string) bool {
	hwidLC := strings.ToLower(hwid)
	if strings.ToLower(data.HardwareID()) == hwidLC {
		return true
	}
	a := data.CompatIDs()
	for i := range a {
		if strings.ToLower(a[i]) == hwidLC {
			return true
		}
	}

	return false
}

// DICD flags control SetupDiCreateDeviceInfo
type DICD uint32

const (
	DICD_GENERATE_ID       DICD = 0x00000001
	DICD_INHERIT_CLASSDRVS DICD = 0x00000002
)

// SUOI flags control SetupUninstallOEMInf
type SUOI uint32

const (
	SUOI_FORCEDELETE SUOI = 0x0001
)

// SPDIT flags to distinguish between class drivers and
// device drivers. (Passed in 'DriverType' parameter of
// driver information list APIs)
type SPDIT uint32

const (
	SPDIT_NODRIVER     SPDIT = 0x00000000
	SPDIT_CLASSDRIVER  SPDIT = 0x00000001
	SPDIT_COMPATDRIVER SPDIT = 0x00000002
)

// DIGCF flags control what is included in the device information set built by SetupDiGetClassDevs
type DIGCF uint32

const (
	DIGCF_DEFAULT         DIGCF = 0x00000001 // only valid with DIGCF_DEVICEINTERFACE
	DIGCF_PRESENT         DIGCF = 0x00000002
	DIGCF_ALLCLASSES      DIGCF = 0x00000004
	DIGCF_PROFILE         DIGCF = 0x00000008
	DIGCF_DEVICEINTERFACE DIGCF = 0x00000010
)

// DIREG specifies values for SetupDiCreateDevRegKey, SetupDiOpenDevRegKey, and SetupDiDeleteDevRegKey.
type DIREG uint32

const (
	DIREG_DEV  DIREG = 0x00000001 // Open/Create/Delete device key
	DIREG_DRV  DIREG = 0x00000002 // Open/Create/Delete driver key
	DIREG_BOTH DIREG = 0x00000004 // Delete both driver and Device key
)

// SPDRP specifies device registry property codes
// (Codes marked as read-only (R) may only be used for
// SetupDiGetDeviceRegistryProperty)
//
// These values should cover the same set of registry properties
// as defined by the CM_DRP codes in cfgmgr32.h.
//
// Note that SPDRP codes are zero based while CM_DRP codes are one based!
type SPDRP uint32

const (
	SPDRP_DEVICEDESC                  SPDRP = 0x00000000 // DeviceDesc (R/W)
	SPDRP_HARDWAREID                  SPDRP = 0x00000001 // HardwareID (R/W)
	SPDRP_COMPATIBLEIDS               SPDRP = 0x00000002 // CompatibleIDs (R/W)
	SPDRP_SERVICE                     SPDRP = 0x00000004 // Service (R/W)
	SPDRP_CLASS                       SPDRP = 0x00000007 // Class (R--tied to ClassGUID)
	SPDRP_CLASSGUID                   SPDRP = 0x00000008 // ClassGUID (R/W)
	SPDRP_DRIVER                      SPDRP = 0x00000009 // Driver (R/W)
	SPDRP_CONFIGFLAGS                 SPDRP = 0x0000000A // ConfigFlags (R/W)
	SPDRP_MFG                         SPDRP = 0x0000000B // Mfg (R/W)
	SPDRP_FRIENDLYNAME                SPDRP = 0x0000000C // FriendlyName (R/W)
	SPDRP_LOCATION_INFORMATION        SPDRP = 0x0000000D // LocationInformation (R/W)
	SPDRP_PHYSICAL_DEVICE_OBJECT_NAME SPDRP = 0x0000000E // PhysicalDeviceObjectName (R)
	SPDRP_CAPABILITIES                SPDRP = 0x0000000F // Capabilities (R)
	SPDRP_UI_NUMBER                   SPDRP = 0x00000010 // UiNumber (R)
	SPDRP_UPPERFILTERS                SPDRP = 0x00000011 // UpperFilters (R/W)
	SPDRP_LOWERFILTERS                SPDRP = 0x00000012 // LowerFilters (R/W)
	SPDRP_BUSTYPEGUID                 SPDRP = 0x00000013 // BusTypeGUID (R)
	SPDRP_LEGACYBUSTYPE               SPDRP = 0x00000014 // LegacyBusType (R)
	SPDRP_BUSNUMBER                   SPDRP = 0x00000015 // BusNumber (R)
	SPDRP_ENUMERATOR_NAME             SPDRP = 0x00000016 // Enumerator Name (R)
	SPDRP_SECURITY                    SPDRP = 0x00000017 // Security (R/W, binary form)
	SPDRP_SECURITY_SDS                SPDRP = 0x00000018 // Security (W, SDS form)
	SPDRP_DEVTYPE                     SPDRP = 0x00000019 // Device Type (R/W)
	SPDRP_EXCLUSIVE                   SPDRP = 0x0000001A // Device is exclusive-access (R/W)
	SPDRP_CHARACTERISTICS             SPDRP = 0x0000001B // Device Characteristics (R/W)
	SPDRP_ADDRESS                     SPDRP = 0x0000001C // Device Address (R)
	SPDRP_UI_NUMBER_DESC_FORMAT       SPDRP = 0x0000001D // UiNumberDescFormat (R/W)
	SPDRP_DEVICE_POWER_DATA           SPDRP = 0x0000001E // Device Power Data (R)
	SPDRP_REMOVAL_POLICY              SPDRP = 0x0000001F // Removal Policy (R)
	SPDRP_REMOVAL_POLICY_HW_DEFAULT   SPDRP = 0x00000020 // Hardware Removal Policy (R)
	SPDRP_REMOVAL_POLICY_OVERRIDE     SPDRP = 0x00000021 // Removal Policy Override (RW)
	SPDRP_INSTALL_STATE               SPDRP = 0x00000022 // Device Install State (R)
	SPDRP_LOCATION_PATHS              SPDRP = 0x00000023 // Device Location Paths (R)
	SPDRP_BASE_CONTAINERID            SPDRP = 0x00000024 // Base ContainerID (R)

	SPDRP_MAXIMUM_PROPERTY SPDRP = 0x00000025 // Upper bound on ordinals
)

// DEVPROPTYPE represents the property-data-type identifier that specifies the
// data type of a device property value in the unified device property model.
type DEVPROPTYPE uint32

const (
	DEVPROP_TYPEMOD_ARRAY DEVPROPTYPE = 0x00001000
	DEVPROP_TYPEMOD_LIST  DEVPROPTYPE = 0x00002000

	DEVPROP_TYPE_EMPTY                      DEVPROPTYPE = 0x00000000
	DEVPROP_TYPE_NULL                       DEVPROPTYPE = 0x00000001
	DEVPROP_TYPE_SBYTE                      DEVPROPTYPE = 0x00000002
	DEVPROP_TYPE_BYTE                       DEVPROPTYPE = 0x00000003
	DEVPROP_TYPE_INT16                      DEVPROPTYPE = 0x00000004
	DEVPROP_TYPE_UINT16                     DEVPROPTYPE = 0x00000005
	DEVPROP_TYPE_INT32                      DEVPROPTYPE = 0x00000006
	DEVPROP_TYPE_UINT32                     DEVPROPTYPE = 0x00000007
	DEVPROP_TYPE_INT64                      DEVPROPTYPE = 0x00000008
	DEVPROP_TYPE_UINT64                     DEVPROPTYPE = 0x00000009
	DEVPROP_TYPE_FLOAT                      DEVPROPTYPE = 0x0000000A
	DEVPROP_TYPE_DOUBLE                     DEVPROPTYPE = 0x0000000B
	DEVPROP_TYPE_DECIMAL                    DEVPROPTYPE = 0x0000000C
	DEVPROP_TYPE_GUID                       DEVPROPTYPE = 0x0000000D
	DEVPROP_TYPE_CURRENCY                   DEVPROPTYPE = 0x0000000E
	DEVPROP_TYPE_DATE                       DEVPROPTYPE = 0x0000000F
	DEVPROP_TYPE_FILETIME                   DEVPROPTYPE = 0x00000010
	DEVPROP_TYPE_BOOLEAN                    DEVPROPTYPE = 0x00000011
	DEVPROP_TYPE_STRING                     DEVPROPTYPE = 0x00000012
	DEVPROP_TYPE_STRING_LIST                DEVPROPTYPE = DEVPROP_TYPE_STRING | DEVPROP_TYPEMOD_LIST
	DEVPROP_TYPE_SECURITY_DESCRIPTOR        DEVPROPTYPE = 0x00000013
	DEVPROP_TYPE_SECURITY_DESCRIPTOR_STRING DEVPROPTYPE = 0x00000014
	DEVPROP_TYPE_DEVPROPKEY                 DEVPROPTYPE = 0x00000015
	DEVPROP_TYPE_DEVPROPTYPE                DEVPROPTYPE = 0x00000016
	DEVPROP_TYPE_BINARY                     DEVPROPTYPE = DEVPROP_TYPE_BYTE | DEVPROP_TYPEMOD_ARRAY
	DEVPROP_TYPE_ERROR                      DEVPROPTYPE = 0x00000017
	DEVPROP_TYPE_NTSTATUS                   DEVPROPTYPE = 0x00000018
	DEVPROP_TYPE_STRING_INDIRECT            DEVPROPTYPE = 0x00000019

	MAX_DEVPROP_TYPE    DEVPROPTYPE = 0x00000019
	MAX_DEVPROP_TYPEMOD DEVPROPTYPE = 0x00002000

	DEVPROP_MASK_TYPE    DEVPROPTYPE = 0x00000FFF
	DEVPROP_MASK_TYPEMOD DEVPROPTYPE = 0x0000F000
)

// DEVPROPGUID specifies a property category.
type DEVPROPGUID GUID

// DEVPROPID uniquely identifies the property within the property category.
type DEVPROPID uint32

const DEVPROPID_FIRST_USABLE DEVPROPID = 2

// DEVPROPKEY represents a device property key for a device property in the
// unified device property model.
type DEVPROPKEY struct {
	FmtID DEVPROPGUID
	PID   DEVPROPID
}

// CONFIGRET is a return value or error code from cfgmgr32 APIs
type CONFIGRET uint32

func (ret CONFIGRET) Error() string {
	if win32Error, ok := ret.Unwrap().(Errno); ok {
		return fmt.Sprintf("%s (CfgMgr error: 0x%08x)", win32Error.Error(), uint32(ret))
	}
	return fmt.Sprintf("CfgMgr error: 0x%08x", uint32(ret))
}

func (ret CONFIGRET) Win32Error(defaultError Errno) Errno {
	return cm_MapCrToWin32Err(ret, defaultError)
}

func (ret CONFIGRET) Unwrap() error {
	const noMatch = Errno(^uintptr(0))
	win32Error := ret.Win32Error(noMatch)
	if win32Error == noMatch {
		return nil
	}
	return win32Error
}

const (
	CR_SUCCESS                  CONFIGRET = 0x00000000
	CR_DEFAULT                  CONFIGRET = 0x00000001
	CR_OUT_OF_MEMORY            CONFIGRET = 0x00000002
	CR_INVALID_POINTER          CONFIGRET = 0x00000003
	CR_INVALID_FLAG             CONFIGRET = 0x00000004
	CR_INVALID_DEVNODE          CONFIGRET = 0x00000005
	CR_INVALID_DEVINST                    = CR_INVALID_DEVNODE
	CR_INVALID_RES_DES          CONFIGRET = 0x00000006
	CR_INVALID_LOG_CONF         CONFIGRET = 0x00000007
	CR_INVALID_ARBITRATOR       CONFIGRET = 0x00000008
	CR_INVALID_NODELIST         CONFIGRET = 0x00000009
	CR_DEVNODE_HAS_REQS         CONFIGRET = 0x0000000A
	CR_DEVINST_HAS_REQS                   = CR_DEVNODE_HAS_REQS
	CR_INVALID_RESOURCEID       CONFIGRET = 0x0000000B
	CR_DLVXD_NOT_FOUND          CONFIGRET = 0x0000000C
	CR_NO_SUCH_DEVNODE          CONFIGRET = 0x0000000D
	CR_NO_SUCH_DEVINST                    = CR_NO_SUCH_DEVNODE
	CR_NO_MORE_LOG_CONF         CONFIGRET = 0x0000000E
	CR_NO_MORE_RES_DES          CONFIGRET = 0x0000000F
	CR_ALREADY_SUCH_DEVNODE     CONFIGRET = 0x00000010
	CR_ALREADY_SUCH_DEVINST               = CR_ALREADY_SUCH_DEVNODE
	CR_INVALID_RANGE_LIST       CONFIGRET = 0x00000011
	CR_INVALID_RANGE            CONFIGRET = 0x00000012
	CR_FAILURE                  CONFIGRET = 0x00000013
	CR_NO_SUCH_LOGICAL_DEV      CONFIGRET = 0x00000014
	CR_CREATE_BLOCKED           CONFIGRET = 0x00000015
	CR_NOT_SYSTEM_VM            CONFIGRET = 0x00000016
	CR_REMOVE_VETOED            CONFIGRET = 0x00000017
	CR_APM_VETOED               CONFIGRET = 0x00000018
	CR_INVALID_LOAD_TYPE        CONFIGRET = 0x00000019
	CR_BUFFER_SMALL             CONFIGRET = 0x0000001A
	CR_NO_ARBITRATOR            CONFIGRET = 0x0000001B
	CR_NO_REGISTRY_HANDLE       CONFIGRET = 0x0000001C
	CR_REGISTRY_ERROR           CONFIGRET = 0x0000001D
	CR_INVALID_DEVICE_ID        CONFIGRET = 0x0000001E
	CR_INVALID_DATA             CONFIGRET = 0x0000001F
	CR_INVALID_API              CONFIGRET = 0x00000020
	CR_DEVLOADER_NOT_READY      CONFIGRET = 0x00000021
	CR_NEED_RESTART             CONFIGRET = 0x00000022
	CR_NO_MORE_HW_PROFILES      CONFIGRET = 0x00000023
	CR_DEVICE_NOT_THERE         CONFIGRET = 0x00000024
	CR_NO_SUCH_VALUE            CONFIGRET = 0x00000025
	CR_WRONG_TYPE               CONFIGRET = 0x00000026
	CR_INVALID_PRIORITY         CONFIGRET = 0x00000027
	CR_NOT_DISABLEABLE          CONFIGRET = 0x00000028
	CR_FREE_RESOURCES           CONFIGRET = 0x00000029
	CR_QUERY_VETOED             CONFIGRET = 0x0000002A
	CR_CANT_SHARE_IRQ           CONFIGRET = 0x0000002B
	CR_NO_DEPENDENT             CONFIGRET = 0x0000002C
	CR_SAME_RESOURCES           CONFIGRET = 0x0000002D
	CR_NO_SUCH_REGISTRY_KEY     CONFIGRET = 0x0000002E
	CR_INVALID_MACHINENAME      CONFIGRET = 0x0000002F
	CR_REMOTE_COMM_FAILURE      CONFIGRET = 0x00000030
	CR_MACHINE_UNAVAILABLE      CONFIGRET = 0x00000031
	CR_NO_CM_SERVICES           CONFIGRET = 0x00000032
	CR_ACCESS_DENIED            CONFIGRET = 0x00000033
	CR_CALL_NOT_IMPLEMENTED     CONFIGRET = 0x00000034
	CR_INVALID_PROPERTY         CONFIGRET = 0x00000035
	CR_DEVICE_INTERFACE_ACTIVE  CONFIGRET = 0x00000036
	CR_NO_SUCH_DEVICE_INTERFACE CONFIGRET = 0x00000037
	CR_INVALID_REFERENCE_STRING CONFIGRET = 0x00000038
	CR_INVALID_CONFLICT_LIST    CONFIGRET = 0x00000039
	CR_INVALID_INDEX            CONFIGRET = 0x0000003A
	CR_INVALID_STRUCTURE_SIZE   CONFIGRET = 0x0000003B
	NUM_CR_RESULTS              CONFIGRET = 0x0000003C
)

const (
	CM_GET_DEVICE_INTERFACE_LIST_PRESENT     = 0 // only currently 'live' device interfaces
	CM_GET_DEVICE_INTERFACE_LIST_ALL_DEVICES = 1 // all registered device interfaces, live or not
)

const (
	DN_ROOT_ENUMERATED       = 0x00000001        // Was enumerated by ROOT
	DN_DRIVER_LOADED         = 0x00000002        // Has Register_Device_Driver
	DN_ENUM_LOADED           = 0x00000004        // Has Register_Enumerator
	DN_STARTED               = 0x00000008        // Is currently configured
	DN_MANUAL                = 0x00000010        // Manually installed
	DN_NEED_TO_ENUM          = 0x00000020        // May need reenumeration
	DN_NOT_FIRST_TIME        = 0x00000040        // Has received a config
	DN_HARDWARE_ENUM         = 0x00000080        // Enum generates hardware ID
	DN_LIAR                  = 0x00000100        // Lied about can reconfig once
	DN_HAS_MARK              = 0x00000200        // Not CM_Create_DevInst lately
	DN_HAS_PROBLEM           = 0x00000400        // Need device installer
	DN_FILTERED              = 0x00000800        // Is filtered
	DN_MOVED                 = 0x00001000        // Has been moved
	DN_DISABLEABLE           = 0x00002000        // Can be disabled
	DN_REMOVABLE             = 0x00004000        // Can be removed
	DN_PRIVATE_PROBLEM       = 0x00008000        // Has a private problem
	DN_MF_PARENT             = 0x00010000        // Multi function parent
	DN_MF_CHILD              = 0x00020000        // Multi function child
	DN_WILL_BE_REMOVED       = 0x00040000        // DevInst is being removed
	DN_NOT_FIRST_TIMEE       = 0x00080000        // Has received a config enumerate
	DN_STOP_FREE_RES         = 0x00100000        // When child is stopped, free resources
	DN_REBAL_CANDIDATE       = 0x00200000        // Don't skip during rebalance
	DN_BAD_PARTIAL           = 0x00400000        // This devnode's log_confs do not have same resources
	DN_NT_ENUMERATOR         = 0x00800000        // This devnode's is an NT enumerator
	DN_NT_DRIVER             = 0x01000000        // This devnode's is an NT driver
	DN_NEEDS_LOCKING         = 0x02000000        // Devnode need lock resume processing
	DN_ARM_WAKEUP            = 0x04000000        // Devnode can be the wakeup device
	DN_APM_ENUMERATOR        = 0x08000000        // APM aware enumerator
	DN_APM_DRIVER            = 0x10000000        // APM aware driver
	DN_SILENT_INSTALL        = 0x20000000        // Silent install
	DN_NO_SHOW_IN_DM         = 0x40000000        // No show in device manager
	DN_BOOT_LOG_PROB         = 0x80000000        // Had a problem during preassignment of boot log conf
	DN_NEED_RESTART          = DN_LIAR           // System needs to be restarted for this Devnode to work properly
	DN_DRIVER_BLOCKED        = DN_NOT_FIRST_TIME // One or more drivers are blocked from loading for this Devnode
	DN_LEGACY_DRIVER         = DN_MOVED          // This device is using a legacy driver
	DN_CHILD_WITH_INVALID_ID = DN_HAS_MARK       // One or more children have invalid IDs
	DN_DEVICE_DISCONNECTED   = DN_NEEDS_LOCKING  // The function driver for a device reported that the device is not connected.  Typically this means a wireless device is out of range.
	DN_QUERY_REMOVE_PENDING  = DN_MF_PARENT      // Device is part of a set of related devices collectively pending query-removal
	DN_QUERY_REMOVE_ACTIVE   = DN_MF_CHILD       // Device is actively engaged in a query-remove IRP
	DN_CHANGEABLE_FLAGS      = DN_NOT_FIRST_TIME | DN_HARDWARE_ENUM | DN_HAS_MARK | DN_DISABLEABLE | DN_REMOVABLE | DN_MF_CHILD | DN_MF_PARENT | DN_NOT_FIRST_TIMEE | DN_STOP_FREE_RES | DN_REBAL_CANDIDATE | DN_NT_ENUMERATOR | DN_NT_DRIVER | DN_SILENT_INSTALL | DN_NO_SHOW_IN_DM
)

//sys	setupDiCreateDeviceInfoListEx(classGUID *GUID, hwndParent uintptr, machineName *uint16, reserved uintptr) (handle DevInfo, err error) [failretval==DevInfo(InvalidHandle)] = setupapi.SetupDiCreateDeviceInfoListExW

// SetupDiCreateDeviceInfoListEx function creates an empty device information set on a remote or a local computer and optionally associates the set with a device setup class.
func SetupDiCreateDeviceInfoListEx(classGUID *GUID, hwndParent uintptr, machineName string) (deviceInfoSet DevInfo, err error) {
	var machineNameUTF16 *uint16
	if machineName != "" {
		machineNameUTF16, err = UTF16PtrFromString(machineName)
		if err != nil {
			return
		}
	}
	return setupDiCreateDeviceInfoListEx(classGUID, hwndParent, machineNameUTF16, 0)
}

//sys	setupDiGetDeviceInfoListDetail(deviceInfoSet DevInfo, deviceInfoSetDetailData *DevInfoListDetailData) (err error) = setupapi.SetupDiGetDeviceInfoListDetailW

// SetupDiGetDeviceInfoListDetail function retrieves information associated with a device information set including the class GUID, remote computer handle, and remote computer name.
func SetupDiGetDeviceInfoListDetail(deviceInfoSet DevInfo) (deviceInfoSetDetailData *DevInfoListDetailData, err error) {
	data := &DevInfoListDetailData{}
	data.size = data.unsafeSizeOf()

	return data, setupDiGetDeviceInfoListDetail(deviceInfoSet, data)
}

// DeviceInfoListDetail method retrieves information associated with a device information set including the class GUID, remote computer handle, and remote computer name.
func (deviceInfoSet DevInfo) DeviceInfoListDetail() (*DevInfoListDetailData, error) {
	return SetupDiGetDeviceInfoListDetail(deviceInfoSet)
}

//sys	setupDiCreateDeviceInfo(deviceInfoSet DevInfo, DeviceName *uint16, classGUID *GUID, DeviceDescription *uint16, hwndParent uintptr, CreationFlags DICD, deviceInfoData *DevInfoData) (err error) = setupapi.SetupDiCreateDeviceInfoW

// SetupDiCreateDeviceInfo function creates a new device information element and adds it as a new member to the specified device information set.
func SetupDiCreateDeviceInfo(deviceInfoSet DevInfo, deviceName string, classGUID *GUID, deviceDescription string, hwndParent uintptr, creationFlags DICD) (deviceInfoData *DevInfoData, err error) {
	deviceNameUTF16, err := UTF16PtrFromString(deviceName)
	if err != nil {
		return
	}

	var deviceDescriptionUTF16 *uint16
	if deviceDescription != "" {
		deviceDescriptionUTF16, err = UTF16PtrFromString(deviceDescription)
		if err != nil {
			return
		}
	}

	data := &DevInfoData{}
	data.size = uint32(unsafe.Sizeof(*data))

	return data, setupDiCreateDeviceInfo(deviceInfoSet, deviceNameUTF16, classGUID, deviceDescriptionUTF16, hwndParent, creationFlags, data)
}

// CreateDeviceInfo method creates a new device information element and adds it as a new member to the specified device information set.
func (deviceInfoSet DevInfo) CreateDeviceInfo(deviceName string, classGUID *GUID, deviceDescription string, hwndParent uintptr, creationFlags DICD) (*DevInfoData, error) {
	return SetupDiCreateDeviceInfo(deviceInfoSet, deviceName, classGUID, deviceDescription, hwndParent, creationFlags)
}

//sys	setupDiEnumDeviceInfo(deviceInfoSet DevInfo, memberIndex uint32, deviceInfoData *DevInfoData) (err error) = setupapi.SetupDiEnumDeviceInfo

// SetupDiEnumDeviceInfo function returns a DevInfoData structure that specifies a device information element in a device information set.
func SetupDiEnumDeviceInfo(deviceInfoSet DevInfo, memberIndex int) (*DevInfoData, error) {
	data := &DevInfoData{}
	data.size = uint32(unsafe.Sizeof(*data))

	return data, setupDiEnumDeviceInfo(deviceInfoSet, uint32(memberIndex), data)
}

// EnumDeviceInfo method returns a DevInfoData structure that specifies a device information element in a device information set.
func (deviceInfoSet DevInfo) EnumDeviceInfo(memberIndex int) (*DevInfoData, error) {
	return SetupDiEnumDeviceInfo(deviceInfoSet, memberIndex)
}

// SetupDiDestroyDeviceInfoList function deletes a device information set and frees all associated memory.
//sys	SetupDiDestroyDeviceInfoList(deviceInfoSet DevInfo) (err error) = setupapi.SetupDiDestroyDeviceInfoList

// Close method deletes a device information set and frees all associated memory.
func (deviceInfoSet DevInfo) Close() error {
	return SetupDiDestroyDeviceInfoList(deviceInfoSet)
}

//sys	SetupDiBuildDriverInfoList(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, driverType SPDIT) (err error) = setupapi.SetupDiBuildDriverInfoList

// BuildDriverInfoList method builds a list of drivers that is associated with a specific device or with the global class driver list for a device information set.
func (deviceInfoSet DevInfo) BuildDriverInfoList(deviceInfoData *DevInfoData, driverType SPDIT) error {
	return SetupDiBuildDriverInfoList(deviceInfoSet, deviceInfoData, driverType)
}

//sys	SetupDiCancelDriverInfoSearch(deviceInfoSet DevInfo) (err error) = setupapi.SetupDiCancelDriverInfoSearch

// CancelDriverInfoSearch method cancels a driver list search that is currently in progress in a different thread.
func (deviceInfoSet DevInfo) CancelDriverInfoSearch() error {
	return SetupDiCancelDriverInfoSearch(deviceInfoSet)
}

//sys	setupDiEnumDriverInfo(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, driverType SPDIT, memberIndex uint32, driverInfoData *DrvInfoData) (err error) = setupapi.SetupDiEnumDriverInfoW

// SetupDiEnumDriverInfo function enumerates the members of a driver list.
func SetupDiEnumDriverInfo(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, driverType SPDIT, memberIndex int) (*DrvInfoData, error) {
	data := &DrvInfoData{}
	data.size = uint32(unsafe.Sizeof(*data))

	return data, setupDiEnumDriverInfo(deviceInfoSet, deviceInfoData, driverType, uint32(memberIndex), data)
}

// EnumDriverInfo method enumerates the members of a driver list.
func (deviceInfoSet DevInfo) EnumDriverInfo(deviceInfoData *DevInfoData, driverType SPDIT, memberIndex int) (*DrvInfoData, error) {
	return SetupDiEnumDriverInfo(deviceInfoSet, deviceInfoData, driverType, memberIndex)
}

//sys	setupDiGetSelectedDriver(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, driverInfoData *DrvInfoData) (err error) = setupapi.SetupDiGetSelectedDriverW

// SetupDiGetSelectedDriver function retrieves the selected driver for a device information set or a particular device information element.
func SetupDiGetSelectedDriver(deviceInfoSet DevInfo, deviceInfoData *DevInfoData) (*DrvInfoData, error) {
	data := &DrvInfoData{}
	data.size = uint32(unsafe.Sizeof(*data))

	return data, setupDiGetSelectedDriver(deviceInfoSet, deviceInfoData, data)
}

// SelectedDriver method retrieves the selected driver for a device information set or a particular device information element.
func (deviceInfoSet DevInfo) SelectedDriver(deviceInfoData *DevInfoData) (*DrvInfoData, error) {
	return SetupDiGetSelectedDriver(deviceInfoSet, deviceInfoData)
}

//sys	SetupDiSetSelectedDriver(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, driverInfoData *DrvInfoData) (err error) = setupapi.SetupDiSetSelectedDriverW

// SetSelectedDriver method sets, or resets, the selected driver for a device information element or the selected class driver for a device information set.
func (deviceInfoSet DevInfo) SetSelectedDriver(deviceInfoData *DevInfoData, driverInfoData *DrvInfoData) error {
	return SetupDiSetSelectedDriver(deviceInfoSet, deviceInfoData, driverInfoData)
}

//sys	setupDiGetDriverInfoDetail(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, driverInfoData *DrvInfoData, driverInfoDetailData *DrvInfoDetailData, driverInfoDetailDataSize uint32, requiredSize *uint32) (err error) = setupapi.SetupDiGetDriverInfoDetailW

// SetupDiGetDriverInfoDetail function retrieves driver information detail for a device information set or a particular device information element in the device information set.
func SetupDiGetDriverInfoDetail(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, driverInfoData *DrvInfoData) (*DrvInfoDetailData, error) {
	reqSize := uint32(2048)
	for {
		buf := make([]byte, reqSize)
		data := (*DrvInfoDetailData)(unsafe.Pointer(&buf[0]))
		data.size = data.unsafeSizeOf()
		err := setupDiGetDriverInfoDetail(deviceInfoSet, deviceInfoData, driverInfoData, data, uint32(len(buf)), &reqSize)
		if err == ERROR_INSUFFICIENT_BUFFER {
			continue
		}
		if err != nil {
			return nil, err
		}
		data.size = reqSize
		return data, nil
	}
}

// DriverInfoDetail method retrieves driver information detail for a device information set or a particular device information element in the device information set.
func (deviceInfoSet DevInfo) DriverInfoDetail(deviceInfoData *DevInfoData, driverInfoData *DrvInfoData) (*DrvInfoDetailData, error) {
	return SetupDiGetDriverInfoDetail(deviceInfoSet, deviceInfoData, driverInfoData)
}

//sys	SetupDiDestroyDriverInfoList(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, driverType SPDIT) (err error) = setupapi.SetupDiDestroyDriverInfoList

// DestroyDriverInfoList method deletes a driver list.
func (deviceInfoSet DevInfo) DestroyDriverInfoList(deviceInfoData *DevInfoData, driverType SPDIT) error {
	return SetupDiDestroyDriverInfoList(deviceInfoSet, deviceInfoData, driverType)
}

//sys	setupDiGetClassDevsEx(classGUID *GUID, Enumerator *uint16, hwndParent uintptr, Flags DIGCF, deviceInfoSet DevInfo, machineName *uint16, reserved uintptr) (handle DevInfo, err error) [failretval==DevInfo(InvalidHandle)] = setupapi.SetupDiGetClassDevsExW

// SetupDiGetClassDevsEx function returns a handle to a device information set that contains requested device information elements for a local or a remote computer.
func SetupDiGetClassDevsEx(classGUID *GUID, enumerator string, hwndParent uintptr, flags DIGCF, deviceInfoSet DevInfo, machineName string) (handle DevInfo, err error) {
	var enumeratorUTF16 *uint16
	if enumerator != "" {
		enumeratorUTF16, err = UTF16PtrFromString(enumerator)
		if err != nil {
			return
		}
	}
	var machineNameUTF16 *uint16
	if machineName != "" {
		machineNameUTF16, err = UTF16PtrFromString(machineName)
		if err != nil {
			return
		}
	}
	return setupDiGetClassDevsEx(classGUID, enumeratorUTF16, hwndParent, flags, deviceInfoSet, machineNameUTF16, 0)
}

// SetupDiCallClassInstaller function calls the appropriate class installer, and any registered co-installers, with the specified installation request (DIF code).
//sys	SetupDiCallClassInstaller(installFunction DI_FUNCTION, deviceInfoSet DevInfo, deviceInfoData *DevInfoData) (err error) = setupapi.SetupDiCallClassInstaller

// CallClassInstaller member calls the appropriate class installer, and any registered co-installers, with the specified installation request (DIF code).
func (deviceInfoSet DevInfo) CallClassInstaller(installFunction DI_FUNCTION, deviceInfoData *DevInfoData) error {
	return SetupDiCallClassInstaller(installFunction, deviceInfoSet, deviceInfoData)
}

// SetupDiOpenDevRegKey function opens a registry key for device-specific configuration information.
//sys	SetupDiOpenDevRegKey(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, Scope DICS_FLAG, HwProfile uint32, KeyType DIREG, samDesired uint32) (key Handle, err error) [failretval==InvalidHandle] = setupapi.SetupDiOpenDevRegKey

// OpenDevRegKey method opens a registry key for device-specific configuration information.
func (deviceInfoSet DevInfo) OpenDevRegKey(DeviceInfoData *DevInfoData, Scope DICS_FLAG, HwProfile uint32, KeyType DIREG, samDesired uint32) (Handle, error) {
	return SetupDiOpenDevRegKey(deviceInfoSet, DeviceInfoData, Scope, HwProfile, KeyType, samDesired)
}

//sys	setupDiGetDeviceProperty(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, propertyKey *DEVPROPKEY, propertyType *DEVPROPTYPE, propertyBuffer *byte, propertyBufferSize uint32, requiredSize *uint32, flags uint32) (err error) = setupapi.SetupDiGetDevicePropertyW

// SetupDiGetDeviceProperty function retrieves a specified device instance property.
func SetupDiGetDeviceProperty(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, propertyKey *DEVPROPKEY) (value interface{}, err error) {
	reqSize := uint32(256)
	for {
		var dataType DEVPROPTYPE
		buf := make([]byte, reqSize)
		err = setupDiGetDeviceProperty(deviceInfoSet, deviceInfoData, propertyKey, &dataType, &buf[0], uint32(len(buf)), &reqSize, 0)
		if err == ERROR_INSUFFICIENT_BUFFER {
			continue
		}
		if err != nil {
			return
		}
		switch dataType {
		case DEVPROP_TYPE_STRING:
			ret := UTF16ToString(bufToUTF16(buf))
			runtime.KeepAlive(buf)
			return ret, nil
		}
		return nil, errors.New("unimplemented property type")
	}
}

//sys	setupDiGetDeviceRegistryProperty(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, property SPDRP, propertyRegDataType *uint32, propertyBuffer *byte, propertyBufferSize uint32, requiredSize *uint32) (err error) = setupapi.SetupDiGetDeviceRegistryPropertyW

// SetupDiGetDeviceRegistryProperty function retrieves a specified Plug and Play device property.
func SetupDiGetDeviceRegistryProperty(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, property SPDRP) (value interface{}, err error) {
	reqSize := uint32(256)
	for {
		var dataType uint32
		buf := make([]byte, reqSize)
		err = setupDiGetDeviceRegistryProperty(deviceInfoSet, deviceInfoData, property, &dataType, &buf[0], uint32(len(buf)), &reqSize)
		if err == ERROR_INSUFFICIENT_BUFFER {
			continue
		}
		if err != nil {
			return
		}
		return getRegistryValue(buf[:reqSize], dataType)
	}
}

func getRegistryValue(buf []byte, dataType uint32) (interface{}, error) {
	switch dataType {
	case REG_SZ:
		ret := UTF16ToString(bufToUTF16(buf))
		runtime.KeepAlive(buf)
		return ret, nil
	case REG_EXPAND_SZ:
		value := UTF16ToString(bufToUTF16(buf))
		if value == "" {
			return "", nil
		}
		p, err := syscall.UTF16PtrFromString(value)
		if err != nil {
			return "", err
		}
		ret := make([]uint16, 100)
		for {
			n, err := ExpandEnvironmentStrings(p, &ret[0], uint32(len(ret)))
			if err != nil {
				return "", err
			}
			if n <= uint32(len(ret)) {
				return UTF16ToString(ret[:n]), nil
			}
			ret = make([]uint16, n)
		}
	case REG_BINARY:
		return buf, nil
	case REG_DWORD_LITTLE_ENDIAN:
		return binary.LittleEndian.Uint32(buf), nil
	case REG_DWORD_BIG_ENDIAN:
		return binary.BigEndian.Uint32(buf), nil
	case REG_MULTI_SZ:
		bufW := bufToUTF16(buf)
		a := []string{}
		for i := 0; i < len(bufW); {
			j := i + wcslen(bufW[i:])
			if i < j {
				a = append(a, UTF16ToString(bufW[i:j]))
			}
			i = j + 1
		}
		runtime.KeepAlive(buf)
		return a, nil
	case REG_QWORD_LITTLE_ENDIAN:
		return binary.LittleEndian.Uint64(buf), nil
	default:
		return nil, fmt.Errorf("Unsupported registry value type: %v", dataType)
	}
}

// bufToUTF16 function reinterprets []byte buffer as []uint16
func bufToUTF16(buf []byte) []uint16 {
	sl := struct {
		addr *uint16
		len  int
		cap  int
	}{(*uint16)(unsafe.Pointer(&buf[0])), len(buf) / 2, cap(buf) / 2}
	return *(*[]uint16)(unsafe.Pointer(&sl))
}

// utf16ToBuf function reinterprets []uint16 as []byte
func utf16ToBuf(buf []uint16) []byte {
	sl := struct {
		addr *byte
		len  int
		cap  int
	}{(*byte)(unsafe.Pointer(&buf[0])), len(buf) * 2, cap(buf) * 2}
	return *(*[]byte)(unsafe.Pointer(&sl))
}

func wcslen(str []uint16) int {
	for i := 0; i < len(str); i++ {
		if str[i] == 0 {
			return i
		}
	}
	return len(str)
}

// DeviceRegistryProperty method retrieves a specified Plug and Play device property.
func (deviceInfoSet DevInfo) DeviceRegistryProperty(deviceInfoData *DevInfoData, property SPDRP) (interface{}, error) {
	return SetupDiGetDeviceRegistryProperty(deviceInfoSet, deviceInfoData, property)
}

//sys	setupDiSetDeviceRegistryProperty(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, property SPDRP, propertyBuffer *byte, propertyBufferSize uint32) (err error) = setupapi.SetupDiSetDeviceRegistryPropertyW

// SetupDiSetDeviceRegistryProperty function sets a Plug and Play device property for a device.
func SetupDiSetDeviceRegistryProperty(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, property SPDRP, propertyBuffers []byte) error {
	return setupDiSetDeviceRegistryProperty(deviceInfoSet, deviceInfoData, property, &propertyBuffers[0], uint32(len(propertyBuffers)))
}

// SetDeviceRegistryProperty function sets a Plug and Play device property for a device.
func (deviceInfoSet DevInfo) SetDeviceRegistryProperty(deviceInfoData *DevInfoData, property SPDRP, propertyBuffers []byte) error {
	return SetupDiSetDeviceRegistryProperty(deviceInfoSet, deviceInfoData, property, propertyBuffers)
}

// SetDeviceRegistryPropertyString method sets a Plug and Play device property string for a device.
func (deviceInfoSet DevInfo) SetDeviceRegistryPropertyString(deviceInfoData *DevInfoData, property SPDRP, str string) error {
	str16, err := UTF16FromString(str)
	if err != nil {
		return err
	}
	err = SetupDiSetDeviceRegistryProperty(deviceInfoSet, deviceInfoData, property, utf16ToBuf(append(str16, 0)))
	runtime.KeepAlive(str16)
	return err
}

//sys	setupDiGetDeviceInstallParams(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, deviceInstallParams *DevInstallParams) (err error) = setupapi.SetupDiGetDeviceInstallParamsW

// SetupDiGetDeviceInstallParams function retrieves device installation parameters for a device information set or a particular device information element.
func SetupDiGetDeviceInstallParams(deviceInfoSet DevInfo, deviceInfoData *DevInfoData) (*DevInstallParams, error) {
	params := &DevInstallParams{}
	params.size = uint32(unsafe.Sizeof(*params))

	return params, setupDiGetDeviceInstallParams(deviceInfoSet, deviceInfoData, params)
}

// DeviceInstallParams method retrieves device installation parameters for a device information set or a particular device information element.
func (deviceInfoSet DevInfo) DeviceInstallParams(deviceInfoData *DevInfoData) (*DevInstallParams, error) {
	return SetupDiGetDeviceInstallParams(deviceInfoSet, deviceInfoData)
}

//sys	setupDiGetDeviceInstanceId(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, instanceId *uint16, instanceIdSize uint32, instanceIdRequiredSize *uint32) (err error) = setupapi.SetupDiGetDeviceInstanceIdW

// SetupDiGetDeviceInstanceId function retrieves the instance ID of the device.
func SetupDiGetDeviceInstanceId(deviceInfoSet DevInfo, deviceInfoData *DevInfoData) (string, error) {
	reqSize := uint32(1024)
	for {
		buf := make([]uint16, reqSize)
		err := setupDiGetDeviceInstanceId(deviceInfoSet, deviceInfoData, &buf[0], uint32(len(buf)), &reqSize)
		if err == ERROR_INSUFFICIENT_BUFFER {
			continue
		}
		if err != nil {
			return "", err
		}
		return UTF16ToString(buf), nil
	}
}

// DeviceInstanceID method retrieves the instance ID of the device.
func (deviceInfoSet DevInfo) DeviceInstanceID(deviceInfoData *DevInfoData) (string, error) {
	return SetupDiGetDeviceInstanceId(deviceInfoSet, deviceInfoData)
}

// SetupDiGetClassInstallParams function retrieves class installation parameters for a device information set or a particular device information element.
//sys	SetupDiGetClassInstallParams(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, classInstallParams *ClassInstallHeader, classInstallParamsSize uint32, requiredSize *uint32) (err error) = setupapi.SetupDiGetClassInstallParamsW

// ClassInstallParams method retrieves class installation parameters for a device information set or a particular device information element.
func (deviceInfoSet DevInfo) ClassInstallParams(deviceInfoData *DevInfoData, classInstallParams *ClassInstallHeader, classInstallParamsSize uint32, requiredSize *uint32) error {
	return SetupDiGetClassInstallParams(deviceInfoSet, deviceInfoData, classInstallParams, classInstallParamsSize, requiredSize)
}

//sys	SetupDiSetDeviceInstallParams(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, deviceInstallParams *DevInstallParams) (err error) = setupapi.SetupDiSetDeviceInstallParamsW

// SetDeviceInstallParams member sets device installation parameters for a device information set or a particular device information element.
func (deviceInfoSet DevInfo) SetDeviceInstallParams(deviceInfoData *DevInfoData, deviceInstallParams *DevInstallParams) error {
	return SetupDiSetDeviceInstallParams(deviceInfoSet, deviceInfoData, deviceInstallParams)
}

// SetupDiSetClassInstallParams function sets or clears class install parameters for a device information set or a particular device information element.
//sys	SetupDiSetClassInstallParams(deviceInfoSet DevInfo, deviceInfoData *DevInfoData, classInstallParams *ClassInstallHeader, classInstallParamsSize uint32) (err error) = setupapi.SetupDiSetClassInstallParamsW

// SetClassInstallParams method sets or clears class install parameters for a device information set or a particular device information element.
func (deviceInfoSet DevInfo) SetClassInstallParams(deviceInfoData *DevInfoData, classInstallParams *ClassInstallHeader, classInstallParamsSize uint32) error {
	return SetupDiSetClassInstallParams(deviceInfoSet, deviceInfoData, classInstallParams, classInstallParamsSize)
}

//sys	setupDiClassNameFromGuidEx(classGUID *GUID, className *uint16, classNameSize uint32, requiredSize *uint32, machineName *uint16, reserved uintptr) (err error) = setupapi.SetupDiClassNameFromGuidExW

// SetupDiClassNameFromGuidEx function retrieves the class name associated with a class GUID. The class can be installed on a local or remote computer.
func SetupDiClassNameFromGuidEx(classGUID *GUID, machineName string) (className string, err error) {
	var classNameUTF16 [MAX_CLASS_NAME_LEN]uint16

	var machineNameUTF16 *uint16
	if machineName != "" {
		machineNameUTF16, err = UTF16PtrFromString(machineName)
		if err != nil {
			return
		}
	}

	err = setupDiClassNameFromGuidEx(classGUID, &classNameUTF16[0], MAX_CLASS_NAME_LEN, nil, machineNameUTF16, 0)
	if err != nil {
		return
	}

	className = UTF16ToString(classNameUTF16[:])
	return
}

//sys	setupDiClassGuidsFromNameEx(className *uint16, classGuidList *GUID, classGuidListSize uint32, requiredSize *uint32, machineName *uint16, reserved uintptr) (err error) = setupapi.SetupDiClassGuidsFromNameExW

// SetupDiClassGuidsFromNameEx function retrieves the GUIDs associated with the specified class name. This resulting list contains the classes currently installed on a local or remote computer.
func SetupDiClassGuidsFromNameEx(className string, machineName string) ([]GUID, error) {
	classNameUTF16, err := UTF16PtrFromString(className)
	if err != nil {
		return nil, err
	}

	var machineNameUTF16 *uint16
	if machineName != "" {
		machineNameUTF16, err = UTF16PtrFromString(machineName)
		if err != nil {
			return nil, err
		}
	}

	reqSize := uint32(4)
	for {
		buf := make([]GUID, reqSize)
		err = setupDiClassGuidsFromNameEx(classNameUTF16, &buf[0], uint32(len(buf)), &reqSize, machineNameUTF16, 0)
		if err == ERROR_INSUFFICIENT_BUFFER {
			continue
		}
		if err != nil {
			return nil, err
		}
		return buf[:reqSize], nil
	}
}

//sys	setupDiGetSelectedDevice(deviceInfoSet DevInfo, deviceInfoData *DevInfoData) (err error) = setupapi.SetupDiGetSelectedDevice

// SetupDiGetSelectedDevice function retrieves the selected device information element in a device information set.
func SetupDiGetSelectedDevice(deviceInfoSet DevInfo) (*DevInfoData, error) {
	data := &DevInfoData{}
	data.size = uint32(unsafe.Sizeof(*data))

	return data, setupDiGetSelectedDevice(deviceInfoSet, data)
}

// SelectedDevice method retrieves the selected device information element in a device information set.
func (deviceInfoSet DevInfo) SelectedDevice() (*DevInfoData, error) {
	return SetupDiGetSelectedDevice(deviceInfoSet)
}

// SetupDiSetSelectedDevice function sets a device information element as the selected member of a device information set. This function is typically used by an installation wizard.
//sys	SetupDiSetSelectedDevice(deviceInfoSet DevInfo, deviceInfoData *DevInfoData) (err error) = setupapi.SetupDiSetSelectedDevice

// SetSelectedDevice method sets a device information element as the selected member of a device information set. This function is typically used by an installation wizard.
func (deviceInfoSet DevInfo) SetSelectedDevice(deviceInfoData *DevInfoData) error {
	return SetupDiSetSelectedDevice(deviceInfoSet, deviceInfoData)
}

//sys	setupUninstallOEMInf(infFileName *uint16, flags SUOI, reserved uintptr) (err error) = setupapi.SetupUninstallOEMInfW

// SetupUninstallOEMInf uninstalls the specified driver.
func SetupUninstallOEMInf(infFileName string, flags SUOI) error {
	infFileName16, err := UTF16PtrFromString(infFileName)
	if err != nil {
		return err
	}
	return setupUninstallOEMInf(infFileName16, flags, 0)
}

//sys cm_MapCrToWin32Err(configRet CONFIGRET, defaultWin32Error Errno) (ret Errno) = CfgMgr32.CM_MapCrToWin32Err

//sys cm_Get_Device_Interface_List_Size(len *uint32, interfaceClass *GUID, deviceID *uint16, flags uint32) (ret CONFIGRET) = CfgMgr32.CM_Get_Device_Interface_List_SizeW
//sys cm_Get_Device_Interface_List(interfaceClass *GUID, deviceID *uint16, buffer *uint16, bufferLen uint32, flags uint32) (ret CONFIGRET) = CfgMgr32.CM_Get_Device_Interface_ListW

func CM_Get_Device_Interface_List(deviceID string, interfaceClass *GUID, flags uint32) ([]string, error) {
	deviceID16, err := UTF16PtrFromString(deviceID)
	if err != nil {
		return nil, err
	}
	var buf []uint16
	var buflen uint32
	for {
		if ret := cm_Get_Device_Interface_List_Size(&buflen, interfaceClass, deviceID16, flags); ret != CR_SUCCESS {
			return nil, ret
		}
		buf = make([]uint16, buflen)
		if ret := cm_Get_Device_Interface_List(interfaceClass, deviceID16, &buf[0], buflen, flags); ret == CR_SUCCESS {
			break
		} else if ret != CR_BUFFER_SMALL {
			return nil, ret
		}
	}
	var interfaces []string
	for i := 0; i < len(buf); {
		j := i + wcslen(buf[i:])
		if i < j {
			interfaces = append(interfaces, UTF16ToString(buf[i:j]))
		}
		i = j + 1
	}
	if interfaces == nil {
		return nil, ERROR_NO_SUCH_DEVICE_INTERFACE
	}
	return interfaces, nil
}

//sys cm_Get_DevNode_Status(status *uint32, problemNumber *uint32, devInst DEVINST, flags uint32) (ret CONFIGRET) = CfgMgr32.CM_Get_DevNode_Status

func CM_Get_DevNode_Status(status *uint32, problemNumber *uint32, devInst DEVINST, flags uint32) error {
	ret := cm_Get_DevNode_Status(status, problemNumber, devInst, flags)
	if ret == CR_SUCCESS {
		return nil
	}
	return ret
}
