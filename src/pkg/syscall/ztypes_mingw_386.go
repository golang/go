// godefs -gsyscall -f-m32 types_linux.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// TODO(brainman): autogenerate types in ztypes_mingw_386.go

//import "unsafe"

// Constants
const (
	sizeofPtr           = 0x4
	sizeofShort         = 0x2
	sizeofInt           = 0x4
	sizeofLong          = 0x4
	sizeofLongLong      = 0x8
	PathMax             = 0x1000
	SizeofSockaddrInet4 = 0x10
	SizeofSockaddrInet6 = 0x1c
	SizeofSockaddrAny   = 0x70
	SizeofSockaddrUnix  = 0x6e
	SizeofLinger        = 0x8
	SizeofMsghdr        = 0x1c
	SizeofCmsghdr       = 0xc
)

// Types

type _C_short int16

type _C_int int32

type _C_long int32

type _C_long_long int64

type Bool uint32
type Module uint32

type Timeval struct {
	Sec  int32
	Usec int32
}
