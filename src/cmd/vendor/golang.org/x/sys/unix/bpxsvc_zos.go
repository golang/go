// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos

package unix

import (
	"bytes"
	"fmt"
	"unsafe"
)

//go:noescape
func bpxcall(plist []unsafe.Pointer, bpx_offset int64)

//go:noescape
func A2e([]byte)

//go:noescape
func E2a([]byte)

const (
	BPX4STA = 192  // stat
	BPX4FST = 104  // fstat
	BPX4LST = 132  // lstat
	BPX4OPN = 156  // open
	BPX4CLO = 72   // close
	BPX4CHR = 500  // chattr
	BPX4FCR = 504  // fchattr
	BPX4LCR = 1180 // lchattr
	BPX4CTW = 492  // cond_timed_wait
	BPX4GTH = 1056 // __getthent
	BPX4PTQ = 412  // pthread_quiesc
	BPX4PTR = 320  // ptrace
)

const (
	//options
	//byte1
	BPX_OPNFHIGH = 0x80
	//byte2
	BPX_OPNFEXEC = 0x80
	//byte3
	BPX_O_NOLARGEFILE = 0x08
	BPX_O_LARGEFILE   = 0x04
	BPX_O_ASYNCSIG    = 0x02
	BPX_O_SYNC        = 0x01
	//byte4
	BPX_O_CREXCL   = 0xc0
	BPX_O_CREAT    = 0x80
	BPX_O_EXCL     = 0x40
	BPX_O_NOCTTY   = 0x20
	BPX_O_TRUNC    = 0x10
	BPX_O_APPEND   = 0x08
	BPX_O_NONBLOCK = 0x04
	BPX_FNDELAY    = 0x04
	BPX_O_RDWR     = 0x03
	BPX_O_RDONLY   = 0x02
	BPX_O_WRONLY   = 0x01
	BPX_O_ACCMODE  = 0x03
	BPX_O_GETFL    = 0x0f

	//mode
	// byte1 (file type)
	BPX_FT_DIR      = 1
	BPX_FT_CHARSPEC = 2
	BPX_FT_REGFILE  = 3
	BPX_FT_FIFO     = 4
	BPX_FT_SYMLINK  = 5
	BPX_FT_SOCKET   = 6
	//byte3
	BPX_S_ISUID  = 0x08
	BPX_S_ISGID  = 0x04
	BPX_S_ISVTX  = 0x02
	BPX_S_IRWXU1 = 0x01
	BPX_S_IRUSR  = 0x01
	//byte4
	BPX_S_IRWXU2 = 0xc0
	BPX_S_IWUSR  = 0x80
	BPX_S_IXUSR  = 0x40
	BPX_S_IRWXG  = 0x38
	BPX_S_IRGRP  = 0x20
	BPX_S_IWGRP  = 0x10
	BPX_S_IXGRP  = 0x08
	BPX_S_IRWXOX = 0x07
	BPX_S_IROTH  = 0x04
	BPX_S_IWOTH  = 0x02
	BPX_S_IXOTH  = 0x01

	CW_INTRPT  = 1
	CW_CONDVAR = 32
	CW_TIMEOUT = 64

	PGTHA_NEXT        = 2
	PGTHA_CURRENT     = 1
	PGTHA_FIRST       = 0
	PGTHA_LAST        = 3
	PGTHA_PROCESS     = 0x80
	PGTHA_CONTTY      = 0x40
	PGTHA_PATH        = 0x20
	PGTHA_COMMAND     = 0x10
	PGTHA_FILEDATA    = 0x08
	PGTHA_THREAD      = 0x04
	PGTHA_PTAG        = 0x02
	PGTHA_COMMANDLONG = 0x01
	PGTHA_THREADFAST  = 0x80
	PGTHA_FILEPATH    = 0x40
	PGTHA_THDSIGMASK  = 0x20
	// thread quiece mode
	QUIESCE_TERM       int32 = 1
	QUIESCE_FORCE      int32 = 2
	QUIESCE_QUERY      int32 = 3
	QUIESCE_FREEZE     int32 = 4
	QUIESCE_UNFREEZE   int32 = 5
	FREEZE_THIS_THREAD int32 = 6
	FREEZE_EXIT        int32 = 8
	QUIESCE_SRB        int32 = 9
)

type Pgtha struct {
	Pid        uint32 // 0
	Tid0       uint32 // 4
	Tid1       uint32
	Accesspid  byte    // C
	Accesstid  byte    // D
	Accessasid uint16  // E
	Loginname  [8]byte // 10
	Flag1      byte    // 18
	Flag1b2    byte    // 19
}

type Bpxystat_t struct { // DSECT BPXYSTAT
	St_id           [4]uint8  // 0
	St_length       uint16    // 0x4
	St_version      uint16    // 0x6
	St_mode         uint32    // 0x8
	St_ino          uint32    // 0xc
	St_dev          uint32    // 0x10
	St_nlink        uint32    // 0x14
	St_uid          uint32    // 0x18
	St_gid          uint32    // 0x1c
	St_size         uint64    // 0x20
	St_atime        uint32    // 0x28
	St_mtime        uint32    // 0x2c
	St_ctime        uint32    // 0x30
	St_rdev         uint32    // 0x34
	St_auditoraudit uint32    // 0x38
	St_useraudit    uint32    // 0x3c
	St_blksize      uint32    // 0x40
	St_createtime   uint32    // 0x44
	St_auditid      [4]uint32 // 0x48
	St_res01        uint32    // 0x58
	Ft_ccsid        uint16    // 0x5c
	Ft_flags        uint16    // 0x5e
	St_res01a       [2]uint32 // 0x60
	St_res02        uint32    // 0x68
	St_blocks       uint32    // 0x6c
	St_opaque       [3]uint8  // 0x70
	St_visible      uint8     // 0x73
	St_reftime      uint32    // 0x74
	St_fid          uint64    // 0x78
	St_filefmt      uint8     // 0x80
	St_fspflag2     uint8     // 0x81
	St_res03        [2]uint8  // 0x82
	St_ctimemsec    uint32    // 0x84
	St_seclabel     [8]uint8  // 0x88
	St_res04        [4]uint8  // 0x90
	// end of version 1
	_               uint32    // 0x94
	St_atime64      uint64    // 0x98
	St_mtime64      uint64    // 0xa0
	St_ctime64      uint64    // 0xa8
	St_createtime64 uint64    // 0xb0
	St_reftime64    uint64    // 0xb8
	_               uint64    // 0xc0
	St_res05        [16]uint8 // 0xc8
	// end of version 2
}

type BpxFilestatus struct {
	Oflag1 byte
	Oflag2 byte
	Oflag3 byte
	Oflag4 byte
}

type BpxMode struct {
	Ftype byte
	Mode1 byte
	Mode2 byte
	Mode3 byte
}

// Thr attribute structure for extended attributes
type Bpxyatt_t struct { // DSECT BPXYATT
	Att_id           [4]uint8
	Att_version      uint16
	Att_res01        [2]uint8
	Att_setflags1    uint8
	Att_setflags2    uint8
	Att_setflags3    uint8
	Att_setflags4    uint8
	Att_mode         uint32
	Att_uid          uint32
	Att_gid          uint32
	Att_opaquemask   [3]uint8
	Att_visblmaskres uint8
	Att_opaque       [3]uint8
	Att_visibleres   uint8
	Att_size_h       uint32
	Att_size_l       uint32
	Att_atime        uint32
	Att_mtime        uint32
	Att_auditoraudit uint32
	Att_useraudit    uint32
	Att_ctime        uint32
	Att_reftime      uint32
	// end of version 1
	Att_filefmt uint8
	Att_res02   [3]uint8
	Att_filetag uint32
	Att_res03   [8]uint8
	// end of version 2
	Att_atime64   uint64
	Att_mtime64   uint64
	Att_ctime64   uint64
	Att_reftime64 uint64
	Att_seclabel  [8]uint8
	Att_ver3res02 [8]uint8
	// end of version 3
}

func BpxOpen(name string, options *BpxFilestatus, mode *BpxMode) (rv int32, rc int32, rn int32) {
	if len(name) < 1024 {
		var namebuf [1024]byte
		sz := int32(copy(namebuf[:], name))
		A2e(namebuf[:sz])
		var parms [7]unsafe.Pointer
		parms[0] = unsafe.Pointer(&sz)
		parms[1] = unsafe.Pointer(&namebuf[0])
		parms[2] = unsafe.Pointer(options)
		parms[3] = unsafe.Pointer(mode)
		parms[4] = unsafe.Pointer(&rv)
		parms[5] = unsafe.Pointer(&rc)
		parms[6] = unsafe.Pointer(&rn)
		bpxcall(parms[:], BPX4OPN)
		return rv, rc, rn
	}
	return -1, -1, -1
}

func BpxClose(fd int32) (rv int32, rc int32, rn int32) {
	var parms [4]unsafe.Pointer
	parms[0] = unsafe.Pointer(&fd)
	parms[1] = unsafe.Pointer(&rv)
	parms[2] = unsafe.Pointer(&rc)
	parms[3] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4CLO)
	return rv, rc, rn
}

func BpxFileFStat(fd int32, st *Bpxystat_t) (rv int32, rc int32, rn int32) {
	st.St_id = [4]uint8{0xe2, 0xe3, 0xc1, 0xe3}
	st.St_version = 2
	stat_sz := uint32(unsafe.Sizeof(*st))
	var parms [6]unsafe.Pointer
	parms[0] = unsafe.Pointer(&fd)
	parms[1] = unsafe.Pointer(&stat_sz)
	parms[2] = unsafe.Pointer(st)
	parms[3] = unsafe.Pointer(&rv)
	parms[4] = unsafe.Pointer(&rc)
	parms[5] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4FST)
	return rv, rc, rn
}

func BpxFileStat(name string, st *Bpxystat_t) (rv int32, rc int32, rn int32) {
	if len(name) < 1024 {
		var namebuf [1024]byte
		sz := int32(copy(namebuf[:], name))
		A2e(namebuf[:sz])
		st.St_id = [4]uint8{0xe2, 0xe3, 0xc1, 0xe3}
		st.St_version = 2
		stat_sz := uint32(unsafe.Sizeof(*st))
		var parms [7]unsafe.Pointer
		parms[0] = unsafe.Pointer(&sz)
		parms[1] = unsafe.Pointer(&namebuf[0])
		parms[2] = unsafe.Pointer(&stat_sz)
		parms[3] = unsafe.Pointer(st)
		parms[4] = unsafe.Pointer(&rv)
		parms[5] = unsafe.Pointer(&rc)
		parms[6] = unsafe.Pointer(&rn)
		bpxcall(parms[:], BPX4STA)
		return rv, rc, rn
	}
	return -1, -1, -1
}

func BpxFileLStat(name string, st *Bpxystat_t) (rv int32, rc int32, rn int32) {
	if len(name) < 1024 {
		var namebuf [1024]byte
		sz := int32(copy(namebuf[:], name))
		A2e(namebuf[:sz])
		st.St_id = [4]uint8{0xe2, 0xe3, 0xc1, 0xe3}
		st.St_version = 2
		stat_sz := uint32(unsafe.Sizeof(*st))
		var parms [7]unsafe.Pointer
		parms[0] = unsafe.Pointer(&sz)
		parms[1] = unsafe.Pointer(&namebuf[0])
		parms[2] = unsafe.Pointer(&stat_sz)
		parms[3] = unsafe.Pointer(st)
		parms[4] = unsafe.Pointer(&rv)
		parms[5] = unsafe.Pointer(&rc)
		parms[6] = unsafe.Pointer(&rn)
		bpxcall(parms[:], BPX4LST)
		return rv, rc, rn
	}
	return -1, -1, -1
}

func BpxChattr(path string, attr *Bpxyatt_t) (rv int32, rc int32, rn int32) {
	if len(path) >= 1024 {
		return -1, -1, -1
	}
	var namebuf [1024]byte
	sz := int32(copy(namebuf[:], path))
	A2e(namebuf[:sz])
	attr_sz := uint32(unsafe.Sizeof(*attr))
	var parms [7]unsafe.Pointer
	parms[0] = unsafe.Pointer(&sz)
	parms[1] = unsafe.Pointer(&namebuf[0])
	parms[2] = unsafe.Pointer(&attr_sz)
	parms[3] = unsafe.Pointer(attr)
	parms[4] = unsafe.Pointer(&rv)
	parms[5] = unsafe.Pointer(&rc)
	parms[6] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4CHR)
	return rv, rc, rn
}

func BpxLchattr(path string, attr *Bpxyatt_t) (rv int32, rc int32, rn int32) {
	if len(path) >= 1024 {
		return -1, -1, -1
	}
	var namebuf [1024]byte
	sz := int32(copy(namebuf[:], path))
	A2e(namebuf[:sz])
	attr_sz := uint32(unsafe.Sizeof(*attr))
	var parms [7]unsafe.Pointer
	parms[0] = unsafe.Pointer(&sz)
	parms[1] = unsafe.Pointer(&namebuf[0])
	parms[2] = unsafe.Pointer(&attr_sz)
	parms[3] = unsafe.Pointer(attr)
	parms[4] = unsafe.Pointer(&rv)
	parms[5] = unsafe.Pointer(&rc)
	parms[6] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4LCR)
	return rv, rc, rn
}

func BpxFchattr(fd int32, attr *Bpxyatt_t) (rv int32, rc int32, rn int32) {
	attr_sz := uint32(unsafe.Sizeof(*attr))
	var parms [6]unsafe.Pointer
	parms[0] = unsafe.Pointer(&fd)
	parms[1] = unsafe.Pointer(&attr_sz)
	parms[2] = unsafe.Pointer(attr)
	parms[3] = unsafe.Pointer(&rv)
	parms[4] = unsafe.Pointer(&rc)
	parms[5] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4FCR)
	return rv, rc, rn
}

func BpxCondTimedWait(sec uint32, nsec uint32, events uint32, secrem *uint32, nsecrem *uint32) (rv int32, rc int32, rn int32) {
	var parms [8]unsafe.Pointer
	parms[0] = unsafe.Pointer(&sec)
	parms[1] = unsafe.Pointer(&nsec)
	parms[2] = unsafe.Pointer(&events)
	parms[3] = unsafe.Pointer(secrem)
	parms[4] = unsafe.Pointer(nsecrem)
	parms[5] = unsafe.Pointer(&rv)
	parms[6] = unsafe.Pointer(&rc)
	parms[7] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4CTW)
	return rv, rc, rn
}
func BpxGetthent(in *Pgtha, outlen *uint32, out unsafe.Pointer) (rv int32, rc int32, rn int32) {
	var parms [7]unsafe.Pointer
	inlen := uint32(26) // nothing else will work. Go says Pgtha is 28-byte because of alignment, but Pgtha is "packed" and must be 26-byte
	parms[0] = unsafe.Pointer(&inlen)
	parms[1] = unsafe.Pointer(&in)
	parms[2] = unsafe.Pointer(outlen)
	parms[3] = unsafe.Pointer(&out)
	parms[4] = unsafe.Pointer(&rv)
	parms[5] = unsafe.Pointer(&rc)
	parms[6] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4GTH)
	return rv, rc, rn
}
func ZosJobname() (jobname string, err error) {
	var pgtha Pgtha
	pgtha.Pid = uint32(Getpid())
	pgtha.Accesspid = PGTHA_CURRENT
	pgtha.Flag1 = PGTHA_PROCESS
	var out [256]byte
	var outlen uint32
	outlen = 256
	rv, rc, rn := BpxGetthent(&pgtha, &outlen, unsafe.Pointer(&out[0]))
	if rv == 0 {
		gthc := []byte{0x87, 0xa3, 0x88, 0x83} // 'gthc' in ebcdic
		ix := bytes.Index(out[:], gthc)
		if ix == -1 {
			err = fmt.Errorf("BPX4GTH: gthc return data not found")
			return
		}
		jn := out[ix+80 : ix+88] // we didn't declare Pgthc, but jobname is 8-byte at offset 80
		E2a(jn)
		jobname = string(bytes.TrimRight(jn, " "))

	} else {
		err = fmt.Errorf("BPX4GTH: rc=%d errno=%d reason=code=0x%x", rv, rc, rn)
	}
	return
}
func Bpx4ptq(code int32, data string) (rv int32, rc int32, rn int32) {
	var userdata [8]byte
	var parms [5]unsafe.Pointer
	copy(userdata[:], data+"        ")
	A2e(userdata[:])
	parms[0] = unsafe.Pointer(&code)
	parms[1] = unsafe.Pointer(&userdata[0])
	parms[2] = unsafe.Pointer(&rv)
	parms[3] = unsafe.Pointer(&rc)
	parms[4] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4PTQ)
	return rv, rc, rn
}

const (
	PT_TRACE_ME             = 0  // Debug this process
	PT_READ_I               = 1  // Read a full word
	PT_READ_D               = 2  // Read a full word
	PT_READ_U               = 3  // Read control info
	PT_WRITE_I              = 4  //Write a full word
	PT_WRITE_D              = 5  //Write a full word
	PT_CONTINUE             = 7  //Continue the process
	PT_KILL                 = 8  //Terminate the process
	PT_READ_GPR             = 11 // Read GPR, CR, PSW
	PT_READ_FPR             = 12 // Read FPR
	PT_READ_VR              = 13 // Read VR
	PT_WRITE_GPR            = 14 // Write GPR, CR, PSW
	PT_WRITE_FPR            = 15 // Write FPR
	PT_WRITE_VR             = 16 // Write VR
	PT_READ_BLOCK           = 17 // Read storage
	PT_WRITE_BLOCK          = 19 // Write storage
	PT_READ_GPRH            = 20 // Read GPRH
	PT_WRITE_GPRH           = 21 // Write GPRH
	PT_REGHSET              = 22 // Read all GPRHs
	PT_ATTACH               = 30 // Attach to a process
	PT_DETACH               = 31 // Detach from a process
	PT_REGSET               = 32 // Read all GPRs
	PT_REATTACH             = 33 // Reattach to a process
	PT_LDINFO               = 34 // Read loader info
	PT_MULTI                = 35 // Multi process mode
	PT_LD64INFO             = 36 // RMODE64 Info Area
	PT_BLOCKREQ             = 40 // Block request
	PT_THREAD_INFO          = 60 // Read thread info
	PT_THREAD_MODIFY        = 61
	PT_THREAD_READ_FOCUS    = 62
	PT_THREAD_WRITE_FOCUS   = 63
	PT_THREAD_HOLD          = 64
	PT_THREAD_SIGNAL        = 65
	PT_EXPLAIN              = 66
	PT_EVENTS               = 67
	PT_THREAD_INFO_EXTENDED = 68
	PT_REATTACH2            = 71
	PT_CAPTURE              = 72
	PT_UNCAPTURE            = 73
	PT_GET_THREAD_TCB       = 74
	PT_GET_ALET             = 75
	PT_SWAPIN               = 76
	PT_EXTENDED_EVENT       = 98
	PT_RECOVER              = 99  // Debug a program check
	PT_GPR0                 = 0   // General purpose register 0
	PT_GPR1                 = 1   // General purpose register 1
	PT_GPR2                 = 2   // General purpose register 2
	PT_GPR3                 = 3   // General purpose register 3
	PT_GPR4                 = 4   // General purpose register 4
	PT_GPR5                 = 5   // General purpose register 5
	PT_GPR6                 = 6   // General purpose register 6
	PT_GPR7                 = 7   // General purpose register 7
	PT_GPR8                 = 8   // General purpose register 8
	PT_GPR9                 = 9   // General purpose register 9
	PT_GPR10                = 10  // General purpose register 10
	PT_GPR11                = 11  // General purpose register 11
	PT_GPR12                = 12  // General purpose register 12
	PT_GPR13                = 13  // General purpose register 13
	PT_GPR14                = 14  // General purpose register 14
	PT_GPR15                = 15  // General purpose register 15
	PT_FPR0                 = 16  // Floating point register 0
	PT_FPR1                 = 17  // Floating point register 1
	PT_FPR2                 = 18  // Floating point register 2
	PT_FPR3                 = 19  // Floating point register 3
	PT_FPR4                 = 20  // Floating point register 4
	PT_FPR5                 = 21  // Floating point register 5
	PT_FPR6                 = 22  // Floating point register 6
	PT_FPR7                 = 23  // Floating point register 7
	PT_FPR8                 = 24  // Floating point register 8
	PT_FPR9                 = 25  // Floating point register 9
	PT_FPR10                = 26  // Floating point register 10
	PT_FPR11                = 27  // Floating point register 11
	PT_FPR12                = 28  // Floating point register 12
	PT_FPR13                = 29  // Floating point register 13
	PT_FPR14                = 30  // Floating point register 14
	PT_FPR15                = 31  // Floating point register 15
	PT_FPC                  = 32  // Floating point control register
	PT_PSW                  = 40  // PSW
	PT_PSW0                 = 40  // Left half of the PSW
	PT_PSW1                 = 41  // Right half of the PSW
	PT_CR0                  = 42  // Control register 0
	PT_CR1                  = 43  // Control register 1
	PT_CR2                  = 44  // Control register 2
	PT_CR3                  = 45  // Control register 3
	PT_CR4                  = 46  // Control register 4
	PT_CR5                  = 47  // Control register 5
	PT_CR6                  = 48  // Control register 6
	PT_CR7                  = 49  // Control register 7
	PT_CR8                  = 50  // Control register 8
	PT_CR9                  = 51  // Control register 9
	PT_CR10                 = 52  // Control register 10
	PT_CR11                 = 53  // Control register 11
	PT_CR12                 = 54  // Control register 12
	PT_CR13                 = 55  // Control register 13
	PT_CR14                 = 56  // Control register 14
	PT_CR15                 = 57  // Control register 15
	PT_GPRH0                = 58  // GP High register 0
	PT_GPRH1                = 59  // GP High register 1
	PT_GPRH2                = 60  // GP High register 2
	PT_GPRH3                = 61  // GP High register 3
	PT_GPRH4                = 62  // GP High register 4
	PT_GPRH5                = 63  // GP High register 5
	PT_GPRH6                = 64  // GP High register 6
	PT_GPRH7                = 65  // GP High register 7
	PT_GPRH8                = 66  // GP High register 8
	PT_GPRH9                = 67  // GP High register 9
	PT_GPRH10               = 68  // GP High register 10
	PT_GPRH11               = 69  // GP High register 11
	PT_GPRH12               = 70  // GP High register 12
	PT_GPRH13               = 71  // GP High register 13
	PT_GPRH14               = 72  // GP High register 14
	PT_GPRH15               = 73  // GP High register 15
	PT_VR0                  = 74  // Vector register 0
	PT_VR1                  = 75  // Vector register 1
	PT_VR2                  = 76  // Vector register 2
	PT_VR3                  = 77  // Vector register 3
	PT_VR4                  = 78  // Vector register 4
	PT_VR5                  = 79  // Vector register 5
	PT_VR6                  = 80  // Vector register 6
	PT_VR7                  = 81  // Vector register 7
	PT_VR8                  = 82  // Vector register 8
	PT_VR9                  = 83  // Vector register 9
	PT_VR10                 = 84  // Vector register 10
	PT_VR11                 = 85  // Vector register 11
	PT_VR12                 = 86  // Vector register 12
	PT_VR13                 = 87  // Vector register 13
	PT_VR14                 = 88  // Vector register 14
	PT_VR15                 = 89  // Vector register 15
	PT_VR16                 = 90  // Vector register 16
	PT_VR17                 = 91  // Vector register 17
	PT_VR18                 = 92  // Vector register 18
	PT_VR19                 = 93  // Vector register 19
	PT_VR20                 = 94  // Vector register 20
	PT_VR21                 = 95  // Vector register 21
	PT_VR22                 = 96  // Vector register 22
	PT_VR23                 = 97  // Vector register 23
	PT_VR24                 = 98  // Vector register 24
	PT_VR25                 = 99  // Vector register 25
	PT_VR26                 = 100 // Vector register 26
	PT_VR27                 = 101 // Vector register 27
	PT_VR28                 = 102 // Vector register 28
	PT_VR29                 = 103 // Vector register 29
	PT_VR30                 = 104 // Vector register 30
	PT_VR31                 = 105 // Vector register 31
	PT_PSWG                 = 106 // PSWG
	PT_PSWG0                = 106 // Bytes 0-3
	PT_PSWG1                = 107 // Bytes 4-7
	PT_PSWG2                = 108 // Bytes 8-11 (IA high word)
	PT_PSWG3                = 109 // Bytes 12-15 (IA low word)
)

func Bpx4ptr(request int32, pid int32, addr unsafe.Pointer, data unsafe.Pointer, buffer unsafe.Pointer) (rv int32, rc int32, rn int32) {
	var parms [8]unsafe.Pointer
	parms[0] = unsafe.Pointer(&request)
	parms[1] = unsafe.Pointer(&pid)
	parms[2] = unsafe.Pointer(&addr)
	parms[3] = unsafe.Pointer(&data)
	parms[4] = unsafe.Pointer(&buffer)
	parms[5] = unsafe.Pointer(&rv)
	parms[6] = unsafe.Pointer(&rc)
	parms[7] = unsafe.Pointer(&rn)
	bpxcall(parms[:], BPX4PTR)
	return rv, rc, rn
}

func copyU8(val uint8, dest []uint8) int {
	if len(dest) < 1 {
		return 0
	}
	dest[0] = val
	return 1
}

func copyU8Arr(src, dest []uint8) int {
	if len(dest) < len(src) {
		return 0
	}
	for i, v := range src {
		dest[i] = v
	}
	return len(src)
}

func copyU16(val uint16, dest []uint16) int {
	if len(dest) < 1 {
		return 0
	}
	dest[0] = val
	return 1
}

func copyU32(val uint32, dest []uint32) int {
	if len(dest) < 1 {
		return 0
	}
	dest[0] = val
	return 1
}

func copyU32Arr(src, dest []uint32) int {
	if len(dest) < len(src) {
		return 0
	}
	for i, v := range src {
		dest[i] = v
	}
	return len(src)
}

func copyU64(val uint64, dest []uint64) int {
	if len(dest) < 1 {
		return 0
	}
	dest[0] = val
	return 1
}
