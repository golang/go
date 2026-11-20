// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

type PROCESS_MEMORY_COUNTERS struct {
	CB                         uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
}

//sys	GetProcessMemoryInfo(handle syscall.Handle, memCounters *PROCESS_MEMORY_COUNTERS, cb uint32) (err error) = psapi.GetProcessMemoryInfo

const (
	JobObjectBasicLimitInformation              = 2
	JobObjectAssociateCompletionPortInformation = 7
	JobObjectExtendedLimitInformation           = 9

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

//sys	CreateJobObject(jobAttrs *syscall.SecurityAttributes, name *uint16) (handle syscall.Handle, err error) = kernel32.CreateJobObjectW
//sys	SetInformationJobObject(job syscall.Handle, infoclass uint32, info uintptr, infolien uint32) (err error) = kernel32.SetInformationJobObject
