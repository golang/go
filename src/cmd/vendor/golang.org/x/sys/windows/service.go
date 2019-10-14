// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package windows

const (
	SC_MANAGER_CONNECT            = 1
	SC_MANAGER_CREATE_SERVICE     = 2
	SC_MANAGER_ENUMERATE_SERVICE  = 4
	SC_MANAGER_LOCK               = 8
	SC_MANAGER_QUERY_LOCK_STATUS  = 16
	SC_MANAGER_MODIFY_BOOT_CONFIG = 32
	SC_MANAGER_ALL_ACCESS         = 0xf003f
)

//sys	OpenSCManager(machineName *uint16, databaseName *uint16, access uint32) (handle Handle, err error) [failretval==0] = advapi32.OpenSCManagerW

const (
	SERVICE_KERNEL_DRIVER       = 1
	SERVICE_FILE_SYSTEM_DRIVER  = 2
	SERVICE_ADAPTER             = 4
	SERVICE_RECOGNIZER_DRIVER   = 8
	SERVICE_WIN32_OWN_PROCESS   = 16
	SERVICE_WIN32_SHARE_PROCESS = 32
	SERVICE_WIN32               = SERVICE_WIN32_OWN_PROCESS | SERVICE_WIN32_SHARE_PROCESS
	SERVICE_INTERACTIVE_PROCESS = 256
	SERVICE_DRIVER              = SERVICE_KERNEL_DRIVER | SERVICE_FILE_SYSTEM_DRIVER | SERVICE_RECOGNIZER_DRIVER
	SERVICE_TYPE_ALL            = SERVICE_WIN32 | SERVICE_ADAPTER | SERVICE_DRIVER | SERVICE_INTERACTIVE_PROCESS

	SERVICE_BOOT_START   = 0
	SERVICE_SYSTEM_START = 1
	SERVICE_AUTO_START   = 2
	SERVICE_DEMAND_START = 3
	SERVICE_DISABLED     = 4

	SERVICE_ERROR_IGNORE   = 0
	SERVICE_ERROR_NORMAL   = 1
	SERVICE_ERROR_SEVERE   = 2
	SERVICE_ERROR_CRITICAL = 3

	SC_STATUS_PROCESS_INFO = 0

	SC_ACTION_NONE        = 0
	SC_ACTION_RESTART     = 1
	SC_ACTION_REBOOT      = 2
	SC_ACTION_RUN_COMMAND = 3

	SERVICE_STOPPED          = 1
	SERVICE_START_PENDING    = 2
	SERVICE_STOP_PENDING     = 3
	SERVICE_RUNNING          = 4
	SERVICE_CONTINUE_PENDING = 5
	SERVICE_PAUSE_PENDING    = 6
	SERVICE_PAUSED           = 7
	SERVICE_NO_CHANGE        = 0xffffffff

	SERVICE_ACCEPT_STOP                  = 1
	SERVICE_ACCEPT_PAUSE_CONTINUE        = 2
	SERVICE_ACCEPT_SHUTDOWN              = 4
	SERVICE_ACCEPT_PARAMCHANGE           = 8
	SERVICE_ACCEPT_NETBINDCHANGE         = 16
	SERVICE_ACCEPT_HARDWAREPROFILECHANGE = 32
	SERVICE_ACCEPT_POWEREVENT            = 64
	SERVICE_ACCEPT_SESSIONCHANGE         = 128

	SERVICE_CONTROL_STOP                  = 1
	SERVICE_CONTROL_PAUSE                 = 2
	SERVICE_CONTROL_CONTINUE              = 3
	SERVICE_CONTROL_INTERROGATE           = 4
	SERVICE_CONTROL_SHUTDOWN              = 5
	SERVICE_CONTROL_PARAMCHANGE           = 6
	SERVICE_CONTROL_NETBINDADD            = 7
	SERVICE_CONTROL_NETBINDREMOVE         = 8
	SERVICE_CONTROL_NETBINDENABLE         = 9
	SERVICE_CONTROL_NETBINDDISABLE        = 10
	SERVICE_CONTROL_DEVICEEVENT           = 11
	SERVICE_CONTROL_HARDWAREPROFILECHANGE = 12
	SERVICE_CONTROL_POWEREVENT            = 13
	SERVICE_CONTROL_SESSIONCHANGE         = 14

	SERVICE_ACTIVE    = 1
	SERVICE_INACTIVE  = 2
	SERVICE_STATE_ALL = 3

	SERVICE_QUERY_CONFIG           = 1
	SERVICE_CHANGE_CONFIG          = 2
	SERVICE_QUERY_STATUS           = 4
	SERVICE_ENUMERATE_DEPENDENTS   = 8
	SERVICE_START                  = 16
	SERVICE_STOP                   = 32
	SERVICE_PAUSE_CONTINUE         = 64
	SERVICE_INTERROGATE            = 128
	SERVICE_USER_DEFINED_CONTROL   = 256
	SERVICE_ALL_ACCESS             = STANDARD_RIGHTS_REQUIRED | SERVICE_QUERY_CONFIG | SERVICE_CHANGE_CONFIG | SERVICE_QUERY_STATUS | SERVICE_ENUMERATE_DEPENDENTS | SERVICE_START | SERVICE_STOP | SERVICE_PAUSE_CONTINUE | SERVICE_INTERROGATE | SERVICE_USER_DEFINED_CONTROL
	SERVICE_RUNS_IN_SYSTEM_PROCESS = 1
	SERVICE_CONFIG_DESCRIPTION     = 1
	SERVICE_CONFIG_FAILURE_ACTIONS = 2

	SC_ENUM_PROCESS_INFO = 0
)

type SERVICE_STATUS struct {
	ServiceType             uint32
	CurrentState            uint32
	ControlsAccepted        uint32
	Win32ExitCode           uint32
	ServiceSpecificExitCode uint32
	CheckPoint              uint32
	WaitHint                uint32
}

type SERVICE_TABLE_ENTRY struct {
	ServiceName *uint16
	ServiceProc uintptr
}

type QUERY_SERVICE_CONFIG struct {
	ServiceType      uint32
	StartType        uint32
	ErrorControl     uint32
	BinaryPathName   *uint16
	LoadOrderGroup   *uint16
	TagId            uint32
	Dependencies     *uint16
	ServiceStartName *uint16
	DisplayName      *uint16
}

type SERVICE_DESCRIPTION struct {
	Description *uint16
}

type SERVICE_STATUS_PROCESS struct {
	ServiceType             uint32
	CurrentState            uint32
	ControlsAccepted        uint32
	Win32ExitCode           uint32
	ServiceSpecificExitCode uint32
	CheckPoint              uint32
	WaitHint                uint32
	ProcessId               uint32
	ServiceFlags            uint32
}

type ENUM_SERVICE_STATUS_PROCESS struct {
	ServiceName          *uint16
	DisplayName          *uint16
	ServiceStatusProcess SERVICE_STATUS_PROCESS
}

type SERVICE_FAILURE_ACTIONS struct {
	ResetPeriod  uint32
	RebootMsg    *uint16
	Command      *uint16
	ActionsCount uint32
	Actions      *SC_ACTION
}

type SC_ACTION struct {
	Type  uint32
	Delay uint32
}

//sys	CloseServiceHandle(handle Handle) (err error) = advapi32.CloseServiceHandle
//sys	CreateService(mgr Handle, serviceName *uint16, displayName *uint16, access uint32, srvType uint32, startType uint32, errCtl uint32, pathName *uint16, loadOrderGroup *uint16, tagId *uint32, dependencies *uint16, serviceStartName *uint16, password *uint16) (handle Handle, err error) [failretval==0] = advapi32.CreateServiceW
//sys	OpenService(mgr Handle, serviceName *uint16, access uint32) (handle Handle, err error) [failretval==0] = advapi32.OpenServiceW
//sys	DeleteService(service Handle) (err error) = advapi32.DeleteService
//sys	StartService(service Handle, numArgs uint32, argVectors **uint16) (err error) = advapi32.StartServiceW
//sys	QueryServiceStatus(service Handle, status *SERVICE_STATUS) (err error) = advapi32.QueryServiceStatus
//sys	ControlService(service Handle, control uint32, status *SERVICE_STATUS) (err error) = advapi32.ControlService
//sys	StartServiceCtrlDispatcher(serviceTable *SERVICE_TABLE_ENTRY) (err error) = advapi32.StartServiceCtrlDispatcherW
//sys	SetServiceStatus(service Handle, serviceStatus *SERVICE_STATUS) (err error) = advapi32.SetServiceStatus
//sys	ChangeServiceConfig(service Handle, serviceType uint32, startType uint32, errorControl uint32, binaryPathName *uint16, loadOrderGroup *uint16, tagId *uint32, dependencies *uint16, serviceStartName *uint16, password *uint16, displayName *uint16) (err error) = advapi32.ChangeServiceConfigW
//sys	QueryServiceConfig(service Handle, serviceConfig *QUERY_SERVICE_CONFIG, bufSize uint32, bytesNeeded *uint32) (err error) = advapi32.QueryServiceConfigW
//sys	ChangeServiceConfig2(service Handle, infoLevel uint32, info *byte) (err error) = advapi32.ChangeServiceConfig2W
//sys	QueryServiceConfig2(service Handle, infoLevel uint32, buff *byte, buffSize uint32, bytesNeeded *uint32) (err error) = advapi32.QueryServiceConfig2W
//sys	EnumServicesStatusEx(mgr Handle, infoLevel uint32, serviceType uint32, serviceState uint32, services *byte, bufSize uint32, bytesNeeded *uint32, servicesReturned *uint32, resumeHandle *uint32, groupName *uint16) (err error) = advapi32.EnumServicesStatusExW
//sys   QueryServiceStatusEx(service Handle, infoLevel uint32, buff *byte, buffSize uint32, bytesNeeded *uint32) (err error) = advapi32.QueryServiceStatusEx
