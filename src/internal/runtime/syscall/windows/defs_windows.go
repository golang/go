// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Architecture-independent definitions.

package windows

// Pseudo handles.
const (
	CurrentProcess = ^uintptr(0) // -1 = current process
	CurrentThread  = ^uintptr(1) // -2 = current thread
)

const INVALID_HANDLE_VALUE = ^uintptr(0)

const DWORD_MAX = 0xffffffff

const (
	PROT_NONE  = 0
	PROT_READ  = 1
	PROT_WRITE = 2
	PROT_EXEC  = 4
)

const (
	MAP_ANON    = 1
	MAP_PRIVATE = 2
)

const DUPLICATE_SAME_ACCESS = 0x2

const THREAD_PRIORITY_HIGHEST = 0x2

const (
	SIGINT  = 0x2
	SIGTERM = 0xF
)

const (
	CTRL_C_EVENT        = 0x0
	CTRL_BREAK_EVENT    = 0x1
	CTRL_CLOSE_EVENT    = 0x2
	CTRL_LOGOFF_EVENT   = 0x5
	CTRL_SHUTDOWN_EVENT = 0x6
)

const (
	EXCEPTION_ACCESS_VIOLATION     = 0xc0000005
	EXCEPTION_IN_PAGE_ERROR        = 0xc0000006
	EXCEPTION_BREAKPOINT           = 0x80000003
	EXCEPTION_ILLEGAL_INSTRUCTION  = 0xc000001d
	EXCEPTION_FLT_DENORMAL_OPERAND = 0xc000008d
	EXCEPTION_FLT_DIVIDE_BY_ZERO   = 0xc000008e
	EXCEPTION_FLT_INEXACT_RESULT   = 0xc000008f
	EXCEPTION_FLT_OVERFLOW         = 0xc0000091
	EXCEPTION_FLT_UNDERFLOW        = 0xc0000093
	EXCEPTION_INT_DIVIDE_BY_ZERO   = 0xc0000094
	EXCEPTION_INT_OVERFLOW         = 0xc0000095
)

const (
	SEM_FAILCRITICALERRORS = 0x0001
	SEM_NOGPFAULTERRORBOX  = 0x0002
	SEM_NOOPENFILEERRORBOX = 0x8000
)

const WER_FAULT_REPORTING_NO_UI = 0x0020

const INFINITE = 0xffffffff

const WAIT_TIMEOUT = 258

const FAIL_FAST_GENERATE_EXCEPTION_ADDRESS = 0x1

const (
	EXCEPTION_CONTINUE_EXECUTION  = -0x1
	EXCEPTION_CONTINUE_SEARCH     = 0x0
	EXCEPTION_CONTINUE_SEARCH_SEH = 0x1
)

const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002

const (
	SYNCHRONIZE        = 0x00100000
	TIMER_QUERY_STATE  = 0x0001
	TIMER_MODIFY_STATE = 0x0002
)

// https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-erref/596a1078-e883-4972-9bbc-49e60bebca55
const (
	STATUS_SUCCESS   = 0x00000000
	STATUS_PENDING   = 0x00000103
	STATUS_CANCELLED = 0xC0000120
)

// https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-system_info
type SystemInfo struct {
	ProcessorArchitecture     uint16
	Reserved                  uint16
	PageSize                  uint32
	MinimumApplicationAddress *byte
	MaximumApplicationAddress *byte
	ActiveProcessorMask       uintptr
	NumberOfProcessors        uint32
	ProcessorType             uint32
	AllocationGranularity     uint32
	ProcessorLevel            uint16
	ProcessorRevision         uint16
}

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-exception_pointers
type ExceptionPointers struct {
	Record  *ExceptionRecord
	Context *Context
}

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-exception_record
type ExceptionRecord struct {
	ExceptionCode        uint32
	ExceptionFlags       uint32
	ExceptionRecord      *ExceptionRecord
	ExceptionAddress     uintptr
	NumberParameters     uint32
	ExceptionInformation [15]uintptr
}

type Handle uintptr

// https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-overlapped
type Overlapped struct {
	Internal     uintptr
	InternalHigh uintptr
	Offset       uint32
	OffsetHigh   uint32
	HEvent       Handle
}

// https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-memory_basic_information
type MemoryBasicInformation struct {
	BaseAddress       uintptr
	AllocationBase    uintptr
	AllocationProtect uint32
	PartitionId       uint16
	RegionSize        uintptr
	State             uint32
	Protect           uint32
	Type              uint32
}

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ns-wdm-_osversioninfow
type OSVERSIONINFOW struct {
	OSVersionInfoSize uint32
	MajorVersion      uint32
	MinorVersion      uint32
	BuildNumber       uint32
	PlatformID        uint32
	CSDVersion        [128]uint16
}
