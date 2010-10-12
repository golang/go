package syscall

// TODO(brainman): autogenerate types in ztypes_windows_386.go

//import "unsafe"

// Constants
const (
	sizeofPtr      = 0x4
	sizeofShort    = 0x2
	sizeofInt      = 0x4
	sizeofLong     = 0x4
	sizeofLongLong = 0x8
	PathMax        = 0x1000
	SizeofLinger   = 0x8
	SizeofMsghdr   = 0x1c
	SizeofCmsghdr  = 0xc
)

const (
	// Windows errors.
	ERROR_FILE_NOT_FOUND      = 2
	ERROR_PATH_NOT_FOUND      = 3
	ERROR_NO_MORE_FILES       = 18
	ERROR_BROKEN_PIPE         = 109
	ERROR_INSUFFICIENT_BUFFER = 122
	ERROR_MOD_NOT_FOUND       = 126
	ERROR_PROC_NOT_FOUND      = 127
	ERROR_ENVVAR_NOT_FOUND    = 203
	ERROR_DIRECTORY           = 267
	ERROR_IO_PENDING          = 997
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
	GENERIC_READ    = 0x80000000
	GENERIC_WRITE   = 0x40000000
	GENERIC_EXECUTE = 0x20000000
	GENERIC_ALL     = 0x10000000

	FILE_APPEND_DATA      = 0x00000004
	FILE_WRITE_ATTRIBUTES = 0x00000100

	FILE_SHARE_READ          = 0x00000001
	FILE_SHARE_WRITE         = 0x00000002
	FILE_SHARE_DELETE        = 0x00000004
	FILE_ATTRIBUTE_READONLY  = 0x00000001
	FILE_ATTRIBUTE_HIDDEN    = 0x00000002
	FILE_ATTRIBUTE_SYSTEM    = 0x00000004
	FILE_ATTRIBUTE_DIRECTORY = 0x00000010
	FILE_ATTRIBUTE_ARCHIVE   = 0x00000020
	FILE_ATTRIBUTE_NORMAL    = 0x00000080

	CREATE_NEW        = 1
	CREATE_ALWAYS     = 2
	OPEN_EXISTING     = 3
	OPEN_ALWAYS       = 4
	TRUNCATE_EXISTING = 5

	STARTF_USESTDHANDLES   = 0x00000100
	DUPLICATE_CLOSE_SOURCE = 0x00000001
	DUPLICATE_SAME_ACCESS  = 0x00000002

	STD_INPUT_HANDLE  = -10
	STD_OUTPUT_HANDLE = -11
	STD_ERROR_HANDLE  = -12

	FILE_BEGIN   = 0
	FILE_CURRENT = 1
	FILE_END     = 2

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

	WAIT_TIMEOUT   = 258
	WAIT_ABANDONED = 0x00000080
	WAIT_OBJECT_0  = 0x00000000
	WAIT_FAILED    = 0xFFFFFFFF

	CREATE_UNICODE_ENVIRONMENT = 0x00000400
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
)

// Types

type _C_short int16

type _C_int int32

type _C_long int32

type _C_long_long int64

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
	Internal     uint32
	InternalHigh uint32
	Offset       uint32
	OffsetHigh   uint32
	HEvent       *byte
}

type Filetime struct {
	LowDateTime  uint32
	HighDateTime uint32
}

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
	StdInput      int32
	StdOutput     int32
	StdErr        int32
}

type ProcessInformation struct {
	Process   int32
	Thread    int32
	ProcessId uint32
	ThreadId  uint32
}

// Invented values to support what package os expects.
type Stat_t struct {
	Windata Win32finddata
	Mode    uint32
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
	AF_UNIX    = 1
	AF_INET    = 2
	AF_INET6   = 23
	AF_NETBIOS = 17

	SOCK_STREAM = 1
	SOCK_DGRAM  = 2
	SOCK_RAW    = 3

	IPPROTO_IP  = 0
	IPPROTO_TCP = 6
	IPPROTO_UDP = 17

	SOL_SOCKET               = 0xffff
	SO_REUSEADDR             = 4
	SO_KEEPALIVE             = 8
	SO_DONTROUTE             = 16
	SO_BROADCAST             = 32
	SO_LINGER                = 128
	SO_RCVBUF                = 0x1002
	SO_SNDBUF                = 0x1001
	SO_UPDATE_ACCEPT_CONTEXT = 0x700b

	IPPROTO_IPV6 = 0x29
	IPV6_V6ONLY  = 0x1b

	SOMAXCONN = 5

	TCP_NODELAY = 1

	SHUT_RD   = 0
	SHUT_WR   = 1
	SHUT_RDWR = 2

	WSADESCRIPTION_LEN = 256
	WSASYS_STATUS_LEN  = 128
)

type WSAData struct {
	Version      uint16
	HighVersion  uint16
	Description  [WSADESCRIPTION_LEN + 1]byte
	SystemStatus [WSASYS_STATUS_LEN + 1]byte
	MaxSockets   uint16
	MaxUdpDg     uint16
	VendorInfo   *byte
}

type WSABuf struct {
	Len uint32
	Buf *byte
}

// TODO(brainman): fix all needed for os

const (
	PROT_READ  = 0x1
	PROT_WRITE = 0x2
	MAP_SHARED = 0x1
	SYS_FORK   = 0
	SYS_PTRACE = 0
	SYS_CHDIR  = 0
	SYS_DUP2   = 0
	SYS_FCNTL  = 0
	SYS_EXECVE = 0
	F_GETFD    = 0x1
	F_SETFD    = 0x2
	F_GETFL    = 0x3
	F_SETFL    = 0x4
	FD_CLOEXEC = 0
	S_IFMT     = 0x1f000
	S_IFIFO    = 0x1000
	S_IFCHR    = 0x2000
	S_IFDIR    = 0x4000
	S_IFBLK    = 0x6000
	S_IFREG    = 0x8000
	S_IFLNK    = 0xa000
	S_IFSOCK   = 0xc000
	S_ISUID    = 0x800
	S_ISGID    = 0x400
	S_ISVTX    = 0x200
	S_IRUSR    = 0x100
	S_IWUSR    = 0x80
	S_IXUSR    = 0x40
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

type Servent struct {
	Name    *byte
	Aliases **byte
	Port    uint16
	Proto   *byte
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

type DNSSRVData struct {
	Target   *uint16
	Priority uint16
	Weight   uint16
	Port     uint16
	Pad      uint16
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
	HANDLE_FLAG_INHERIT            = 0x00000001
	HANDLE_FLAG_PROTECT_FROM_CLOSE = 0x00000002

	PROCESS_ALL_ACCESS = 0x001fffff
)
