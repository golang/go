// godefs -gsyscall -f -m32 types_plan9.c

// MACHINE GENERATED - DO NOT EDIT.

package syscall

// Constants
const (
	O_RDONLY   = 0
	O_WRONLY   = 0x1
	O_RDWR     = 0x2
	O_TRUNC    = 0x10
	O_CLOEXEC  = 0x20
	O_EXCL     = 0x1000
	STATMAX    = 0xffff
	ERRMAX     = 0x80
	MORDER     = 0x3
	MREPL      = 0
	MBEFORE    = 0x1
	MAFTER     = 0x2
	MCREATE    = 0x4
	MCACHE     = 0x10
	MMASK      = 0x17
	RFNAMEG    = 0x1
	RFENVG     = 0x2
	RFFDG      = 0x4
	RFNOTEG    = 0x8
	RFPROC     = 0x10
	RFMEM      = 0x20
	RFNOWAIT   = 0x40
	RFCNAMEG   = 0x400
	RFCENVG    = 0x800
	RFCFDG     = 0x1000
	RFREND     = 0x2000
	RFNOMNT    = 0x4000
	QTDIR      = 0x80
	QTAPPEND   = 0x40
	QTEXCL     = 0x20
	QTMOUNT    = 0x10
	QTAUTH     = 0x8
	QTTMP      = 0x4
	QTFILE     = 0
	DMDIR      = 0x80000000
	DMAPPEND   = 0x40000000
	DMEXCL     = 0x20000000
	DMMOUNT    = 0x10000000
	DMAUTH     = 0x8000000
	DMTMP      = 0x4000000
	DMREAD     = 0x4
	DMWRITE    = 0x2
	DMEXEC     = 0x1
	STATFIXLEN = 0x31
)

// Types

type _C_int int32

type Prof struct {
	Pp    *[0]byte /* sPlink */
	Next  *[0]byte /* sPlink */
	Last  *[0]byte /* sPlink */
	First *[0]byte /* sPlink */
	Pid   uint32
	What  uint32
}

type Tos struct {
	Prof      Prof
	Cyclefreq uint64
	Kcycles   int64
	Pcycles   int64
	Pid       uint32
	Clock     uint32
}
