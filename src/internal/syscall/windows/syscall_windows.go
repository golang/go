// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import (
	"sync"
	"syscall"
	"unsafe"
)

// UTF16PtrToString is like UTF16ToString, but takes *uint16
// as a parameter instead of []uint16.
func UTF16PtrToString(p *uint16) string {
	if p == nil {
		return ""
	}
	end := unsafe.Pointer(p)
	n := 0
	for *(*uint16)(end) != 0 {
		end = unsafe.Pointer(uintptr(end) + unsafe.Sizeof(*p))
		n++
	}
	return syscall.UTF16ToString(unsafe.Slice(p, n))
}

const (
	ERROR_BAD_LENGTH             syscall.Errno = 24
	ERROR_SHARING_VIOLATION      syscall.Errno = 32
	ERROR_LOCK_VIOLATION         syscall.Errno = 33
	ERROR_NOT_SUPPORTED          syscall.Errno = 50
	ERROR_CALL_NOT_IMPLEMENTED   syscall.Errno = 120
	ERROR_INVALID_NAME           syscall.Errno = 123
	ERROR_LOCK_FAILED            syscall.Errno = 167
	ERROR_NO_UNICODE_TRANSLATION syscall.Errno = 1113
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

type SecurityAttributes struct {
	Length             uint16
	SecurityDescriptor uintptr
	InheritHandle      bool
}

type FILE_BASIC_INFO struct {
	CreationTime   int64
	LastAccessTime int64
	LastWriteTime  int64
	ChangedTime    int64
	FileAttributes uint32

	// Pad out to 8-byte alignment.
	//
	// Without this padding, TestChmod fails due to an argument validation error
	// in SetFileInformationByHandle on windows/386.
	//
	// https://learn.microsoft.com/en-us/cpp/build/reference/zp-struct-member-alignment?view=msvc-170
	// says that “The C/C++ headers in the Windows SDK assume the platform's
	// default alignment is used.” What we see here is padding rather than
	// alignment, but maybe it is related.
	_ uint32
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

//sys	GetAdaptersAddresses(family uint32, flags uint32, reserved uintptr, adapterAddresses *IpAdapterAddresses, sizePointer *uint32) (errcode error) = iphlpapi.GetAdaptersAddresses
//sys	GetComputerNameEx(nameformat uint32, buf *uint16, n *uint32) (err error) = GetComputerNameExW
//sys	MoveFileEx(from *uint16, to *uint16, flags uint32) (err error) = MoveFileExW
//sys	GetModuleFileName(module syscall.Handle, fn *uint16, len uint32) (n uint32, err error) = kernel32.GetModuleFileNameW
//sys	SetFileInformationByHandle(handle syscall.Handle, fileInformationClass uint32, buf unsafe.Pointer, bufsize uint32) (err error) = kernel32.SetFileInformationByHandle
//sys	VirtualQuery(address uintptr, buffer *MemoryBasicInformation, length uintptr) (err error) = kernel32.VirtualQuery
//sys	GetTempPath2(buflen uint32, buf *uint16) (n uint32, err error) = GetTempPath2W

const (
	// flags for CreateToolhelp32Snapshot
	TH32CS_SNAPMODULE   = 0x08
	TH32CS_SNAPMODULE32 = 0x10
)

const MAX_MODULE_NAME32 = 255

type ModuleEntry32 struct {
	Size         uint32
	ModuleID     uint32
	ProcessID    uint32
	GlblcntUsage uint32
	ProccntUsage uint32
	ModBaseAddr  uintptr
	ModBaseSize  uint32
	ModuleHandle syscall.Handle
	Module       [MAX_MODULE_NAME32 + 1]uint16
	ExePath      [syscall.MAX_PATH]uint16
}

const SizeofModuleEntry32 = unsafe.Sizeof(ModuleEntry32{})

//sys	Module32First(snapshot syscall.Handle, moduleEntry *ModuleEntry32) (err error) = kernel32.Module32FirstW
//sys	Module32Next(snapshot syscall.Handle, moduleEntry *ModuleEntry32) (err error) = kernel32.Module32NextW

const (
	WSA_FLAG_OVERLAPPED        = 0x01
	WSA_FLAG_NO_HANDLE_INHERIT = 0x80

	WSAEMSGSIZE syscall.Errno = 10040

	MSG_PEEK   = 0x2
	MSG_TRUNC  = 0x0100
	MSG_CTRUNC = 0x0200

	socket_error = uintptr(^uint32(0))
)

var WSAID_WSASENDMSG = syscall.GUID{
	Data1: 0xa441e712,
	Data2: 0x754f,
	Data3: 0x43ca,
	Data4: [8]byte{0x84, 0xa7, 0x0d, 0xee, 0x44, 0xcf, 0x60, 0x6d},
}

var WSAID_WSARECVMSG = syscall.GUID{
	Data1: 0xf689d7c8,
	Data2: 0x6f1f,
	Data3: 0x436b,
	Data4: [8]byte{0x8a, 0x53, 0xe5, 0x4f, 0xe3, 0x51, 0xc3, 0x22},
}

var sendRecvMsgFunc struct {
	once     sync.Once
	sendAddr uintptr
	recvAddr uintptr
	err      error
}

type WSAMsg struct {
	Name        syscall.Pointer
	Namelen     int32
	Buffers     *syscall.WSABuf
	BufferCount uint32
	Control     syscall.WSABuf
	Flags       uint32
}

//sys	WSASocket(af int32, typ int32, protocol int32, protinfo *syscall.WSAProtocolInfo, group uint32, flags uint32) (handle syscall.Handle, err error) [failretval==syscall.InvalidHandle] = ws2_32.WSASocketW

func loadWSASendRecvMsg() error {
	sendRecvMsgFunc.once.Do(func() {
		var s syscall.Handle
		s, sendRecvMsgFunc.err = syscall.Socket(syscall.AF_INET, syscall.SOCK_DGRAM, syscall.IPPROTO_UDP)
		if sendRecvMsgFunc.err != nil {
			return
		}
		defer syscall.CloseHandle(s)
		var n uint32
		sendRecvMsgFunc.err = syscall.WSAIoctl(s,
			syscall.SIO_GET_EXTENSION_FUNCTION_POINTER,
			(*byte)(unsafe.Pointer(&WSAID_WSARECVMSG)),
			uint32(unsafe.Sizeof(WSAID_WSARECVMSG)),
			(*byte)(unsafe.Pointer(&sendRecvMsgFunc.recvAddr)),
			uint32(unsafe.Sizeof(sendRecvMsgFunc.recvAddr)),
			&n, nil, 0)
		if sendRecvMsgFunc.err != nil {
			return
		}
		sendRecvMsgFunc.err = syscall.WSAIoctl(s,
			syscall.SIO_GET_EXTENSION_FUNCTION_POINTER,
			(*byte)(unsafe.Pointer(&WSAID_WSASENDMSG)),
			uint32(unsafe.Sizeof(WSAID_WSASENDMSG)),
			(*byte)(unsafe.Pointer(&sendRecvMsgFunc.sendAddr)),
			uint32(unsafe.Sizeof(sendRecvMsgFunc.sendAddr)),
			&n, nil, 0)
	})
	return sendRecvMsgFunc.err
}

func WSASendMsg(fd syscall.Handle, msg *WSAMsg, flags uint32, bytesSent *uint32, overlapped *syscall.Overlapped, croutine *byte) error {
	err := loadWSASendRecvMsg()
	if err != nil {
		return err
	}
	r1, _, e1 := syscall.Syscall6(sendRecvMsgFunc.sendAddr, 6, uintptr(fd), uintptr(unsafe.Pointer(msg)), uintptr(flags), uintptr(unsafe.Pointer(bytesSent)), uintptr(unsafe.Pointer(overlapped)), uintptr(unsafe.Pointer(croutine)))
	if r1 == socket_error {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return err
}

func WSARecvMsg(fd syscall.Handle, msg *WSAMsg, bytesReceived *uint32, overlapped *syscall.Overlapped, croutine *byte) error {
	err := loadWSASendRecvMsg()
	if err != nil {
		return err
	}
	r1, _, e1 := syscall.Syscall6(sendRecvMsgFunc.recvAddr, 5, uintptr(fd), uintptr(unsafe.Pointer(msg)), uintptr(unsafe.Pointer(bytesReceived)), uintptr(unsafe.Pointer(overlapped)), uintptr(unsafe.Pointer(croutine)), 0)
	if r1 == socket_error {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return err
}

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

	MOVEFILE_REPLACE_EXISTING      = 0x1
	MOVEFILE_COPY_ALLOWED          = 0x2
	MOVEFILE_DELAY_UNTIL_REBOOT    = 0x4
	MOVEFILE_WRITE_THROUGH         = 0x8
	MOVEFILE_CREATE_HARDLINK       = 0x10
	MOVEFILE_FAIL_IF_NOT_TRACKABLE = 0x20
)

func Rename(oldpath, newpath string) error {
	from, err := syscall.UTF16PtrFromString(oldpath)
	if err != nil {
		return err
	}
	to, err := syscall.UTF16PtrFromString(newpath)
	if err != nil {
		return err
	}
	return MoveFileEx(from, to, MOVEFILE_REPLACE_EXISTING)
}

//sys LockFileEx(file syscall.Handle, flags uint32, reserved uint32, bytesLow uint32, bytesHigh uint32, overlapped *syscall.Overlapped) (err error) = kernel32.LockFileEx
//sys UnlockFileEx(file syscall.Handle, reserved uint32, bytesLow uint32, bytesHigh uint32, overlapped *syscall.Overlapped) (err error) = kernel32.UnlockFileEx

const (
	LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
	LOCKFILE_EXCLUSIVE_LOCK   = 0x00000002
)

const MB_ERR_INVALID_CHARS = 8

//sys	GetACP() (acp uint32) = kernel32.GetACP
//sys	GetConsoleCP() (ccp uint32) = kernel32.GetConsoleCP
//sys	MultiByteToWideChar(codePage uint32, dwFlags uint32, str *byte, nstr int32, wchar *uint16, nwchar int32) (nwrite int32, err error) = kernel32.MultiByteToWideChar
//sys	GetCurrentThread() (pseudoHandle syscall.Handle, err error) = kernel32.GetCurrentThread

// Constants from lmshare.h
const (
	STYPE_DISKTREE  = 0x00
	STYPE_TEMPORARY = 0x40000000
)

type SHARE_INFO_2 struct {
	Netname     *uint16
	Type        uint32
	Remark      *uint16
	Permissions uint32
	MaxUses     uint32
	CurrentUses uint32
	Path        *uint16
	Passwd      *uint16
}

//sys  NetShareAdd(serverName *uint16, level uint32, buf *byte, parmErr *uint16) (neterr error) = netapi32.NetShareAdd
//sys  NetShareDel(serverName *uint16, netName *uint16, reserved uint32) (neterr error) = netapi32.NetShareDel

const (
	FILE_NAME_NORMALIZED = 0x0
	FILE_NAME_OPENED     = 0x8

	VOLUME_NAME_DOS  = 0x0
	VOLUME_NAME_GUID = 0x1
	VOLUME_NAME_NONE = 0x4
	VOLUME_NAME_NT   = 0x2
)

//sys	GetFinalPathNameByHandle(file syscall.Handle, filePath *uint16, filePathSize uint32, flags uint32) (n uint32, err error) = kernel32.GetFinalPathNameByHandleW

func ErrorLoadingGetTempPath2() error {
	return procGetTempPath2W.Find()
}

//sys	CreateEnvironmentBlock(block **uint16, token syscall.Token, inheritExisting bool) (err error) = userenv.CreateEnvironmentBlock
//sys	DestroyEnvironmentBlock(block *uint16) (err error) = userenv.DestroyEnvironmentBlock
//sys	CreateEvent(eventAttrs *SecurityAttributes, manualReset uint32, initialState uint32, name *uint16) (handle syscall.Handle, err error) = kernel32.CreateEventW

//sys	ProcessPrng(buf []byte) (err error) = bcryptprimitives.ProcessPrng

type FILE_ID_BOTH_DIR_INFO struct {
	NextEntryOffset uint32
	FileIndex       uint32
	CreationTime    syscall.Filetime
	LastAccessTime  syscall.Filetime
	LastWriteTime   syscall.Filetime
	ChangeTime      syscall.Filetime
	EndOfFile       uint64
	AllocationSize  uint64
	FileAttributes  uint32
	FileNameLength  uint32
	EaSize          uint32
	ShortNameLength uint32
	ShortName       [12]uint16
	FileID          uint64
	FileName        [1]uint16
}

type FILE_FULL_DIR_INFO struct {
	NextEntryOffset uint32
	FileIndex       uint32
	CreationTime    syscall.Filetime
	LastAccessTime  syscall.Filetime
	LastWriteTime   syscall.Filetime
	ChangeTime      syscall.Filetime
	EndOfFile       uint64
	AllocationSize  uint64
	FileAttributes  uint32
	FileNameLength  uint32
	EaSize          uint32
	FileName        [1]uint16
}

//sys	GetVolumeInformationByHandle(file syscall.Handle, volumeNameBuffer *uint16, volumeNameSize uint32, volumeNameSerialNumber *uint32, maximumComponentLength *uint32, fileSystemFlags *uint32, fileSystemNameBuffer *uint16, fileSystemNameSize uint32) (err error) = GetVolumeInformationByHandleW
//sys	GetVolumeNameForVolumeMountPoint(volumeMountPoint *uint16, volumeName *uint16, bufferlength uint32) (err error) = GetVolumeNameForVolumeMountPointW

//sys	RtlLookupFunctionEntry(pc uintptr, baseAddress *uintptr, table *byte) (ret uintptr) = kernel32.RtlLookupFunctionEntry
//sys	RtlVirtualUnwind(handlerType uint32, baseAddress uintptr, pc uintptr, entry uintptr, ctxt uintptr, data *uintptr, frame *uintptr, ctxptrs *byte) (ret uintptr) = kernel32.RtlVirtualUnwind

type SERVICE_STATUS struct {
	ServiceType             uint32
	CurrentState            uint32
	ControlsAccepted        uint32
	Win32ExitCode           uint32
	ServiceSpecificExitCode uint32
	CheckPoint              uint32
	WaitHint                uint32
}

const (
	SERVICE_RUNNING      = 4
	SERVICE_QUERY_STATUS = 4
)

//sys    OpenService(mgr syscall.Handle, serviceName *uint16, access uint32) (handle syscall.Handle, err error) = advapi32.OpenServiceW
//sys	QueryServiceStatus(hService syscall.Handle, lpServiceStatus *SERVICE_STATUS) (err error)  = advapi32.QueryServiceStatus
//sys    OpenSCManager(machineName *uint16, databaseName *uint16, access uint32) (handle syscall.Handle, err error)  [failretval==0] = advapi32.OpenSCManagerW
