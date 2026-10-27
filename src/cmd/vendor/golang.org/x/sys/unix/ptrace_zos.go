// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos

package unix

import "unsafe"

// z/OS ptrace implementation notes:
//
// This file provides ptrace wrappers for z/OS that match the Linux API as closely
// as possible. However, there are fundamental differences in how z/OS implements
// process tracing compared to Linux:
//
// 1. SYSTEM CALL INTERFACE:
//    - z/OS uses BPX4PTR (5 parameters) vs Linux ptrace syscall (4 parameters)
//    - The 5th parameter (buffer) is used for block operations
//
// 2. MEMORY OPERATIONS:
//    - z/OS: PT_READ_BLOCK/PT_WRITE_BLOCK for efficient block transfers
//    - Linux: PTRACE_PEEKTEXT/POKETEXT for word-by-word access
//
// 3. REGISTER OPERATIONS:
//    - z/OS: PT_READ_GPR/PT_WRITE_GPR with register number for 64-bit values
//    - Linux: PTRACE_GETREGSET/SETREGSET with iovec structure
//
// 4. SINGLE-STEPPING:
//    - z/OS: No native single-step; requires temporary breakpoints or PER
//    - Linux: PTRACE_SINGLESTEP natively supported
//      a) Read instruction at current PSW
//      b) Calculate next instruction address (handle branches)
//      c) Set temporary breakpoint (SVC 144 = 0x0A90) at next address
//      d) Continue execution until breakpoint hit
//      e) Restore original instruction and adjust PSW
//
// 5. SYSCALL TRACING:
//    - z/OS: No PTRACE_SYSCALL equivalent; requires breakpoint-based approach
//    - Linux: PTRACE_SYSCALL stops at syscall entry/exit
//      a) Identify BPX syscall vector location
//      b) Set breakpoints at syscall entry points
//      c) Detect BASR instructions calling into syscall vector
//      d) Parse arguments from registers before/after syscall
//
// 6. LIMITATIONS:
//    - PtraceSetOptions: No equivalent (returns ENOSYS)
//    - PtraceGetEventMsg: No equivalent (returns ENOSYS)
//    - PtraceSingleStep: Requires complex implementation (returns ENOSYS)
//    - PtraceInterrupt: No equivalent (returns ENOSYS)
//    - PtracePokeUser: No PT_WRITE_U (returns ENOSYS)
//
// For reference implementations of single-stepping and syscall tracing on z/OS,

// ptrace is the basic wrapper for BPX4PTR on z/OS.
// Note: z/OS ptrace (BPX4PTR) takes 5 parameters vs 4 on Linux/Darwin.
// The 5th parameter (buffer) is used for operations like PT_READ_BLOCK/PT_WRITE_BLOCK.
//
// data is an integer-valued argument (signal number, register index, byte count,
// or a target-process address). It is not a Go-managed pointer. We must not pass
// it to Bpx4ptr as unsafe.Pointer(data) because the garbage collector would treat
// that integer as a pointer and may crash trying to follow it.
//
// Instead we inline the parms array that Bpx4ptr would construct, placing &data
// at parms[3]. BPX4PTR dereferences parms[3] to obtain the data value, so it
// receives data unchanged while the GC only ever sees a valid stack pointer.
func ptrace(request int, pid int, addr uintptr, data uintptr) (err error) {
	req32 := int32(request)
	pid32 := int32(pid)
	var rv, rc, rn int32
	var buffer unsafe.Pointer
	parms := [8]unsafe.Pointer{
		unsafe.Pointer(&req32),
		unsafe.Pointer(&pid32),
		unsafe.Pointer(&addr),
		unsafe.Pointer(&data), // &data, not unsafe.Pointer(data)
		unsafe.Pointer(&buffer),
		unsafe.Pointer(&rv),
		unsafe.Pointer(&rc),
		unsafe.Pointer(&rn),
	}
	bpxcall(parms[:], BPX4PTR)
	if rv != 0 {
		err = errnoErr2(Errno(rc), uintptr(rn))
	}
	return
}

// ptracePtr is a variant that accepts unsafe.Pointer for the data parameter.
// addr is a target-process address (a raw integer), not a Go-managed pointer.
// We must not pass it to Bpx4ptr as unsafe.Pointer(addr); instead we inline the
// parms array and place &addr at parms[2], exactly as ptrace() does for data.
func ptracePtr(request int, pid int, addr uintptr, data unsafe.Pointer) (err error) {
	req32 := int32(request)
	pid32 := int32(pid)
	var rv, rc, rn int32
	var buffer unsafe.Pointer
	parms := [8]unsafe.Pointer{
		unsafe.Pointer(&req32),
		unsafe.Pointer(&pid32),
		unsafe.Pointer(&addr), // &addr, not unsafe.Pointer(addr)
		unsafe.Pointer(&data),
		unsafe.Pointer(&buffer),
		unsafe.Pointer(&rv),
		unsafe.Pointer(&rc),
		unsafe.Pointer(&rn),
	}
	bpxcall(parms[:], BPX4PTR)
	if rv != 0 {
		err = errnoErr2(Errno(rc), uintptr(rn))
	}
	return
}

// ptracePtrWithBuffer is used for operations that require the buffer parameter.
// addr is a target-process address; same &addr treatment as ptracePtr.
func ptracePtrWithBuffer(request int, pid int, addr uintptr, data unsafe.Pointer, buffer unsafe.Pointer) (err error) {
	req32 := int32(request)
	pid32 := int32(pid)
	var rv, rc, rn int32
	parms := [8]unsafe.Pointer{
		unsafe.Pointer(&req32),
		unsafe.Pointer(&pid32),
		unsafe.Pointer(&addr), // &addr, not unsafe.Pointer(addr)
		unsafe.Pointer(&data),
		unsafe.Pointer(&buffer),
		unsafe.Pointer(&rv),
		unsafe.Pointer(&rc),
		unsafe.Pointer(&rn),
	}
	bpxcall(parms[:], BPX4PTR)
	if rv != 0 {
		err = errnoErr2(Errno(rc), uintptr(rn))
	}
	return
}

// High-level convenience functions matching the Linux/Darwin API

func PtraceAttach(pid int) (err error) {
	return ptrace(PT_ATTACH, pid, 0, 0)
}

func PtraceDetach(pid int) (err error) {
	return ptrace(PT_DETACH, pid, 0, 0)
}

func PtraceCont(pid int, signal int) (err error) {
	// PT_CONTINUE can be interrupted by signals (EINTR), retry if needed
	for {
		err = ptrace(PT_CONTINUE, pid, 0, uintptr(signal))
		if err != EINTR {
			break
		}
	}
	return err
}

func PtraceKill(pid int) (err error) {
	return ptrace(PT_KILL, pid, 0, 0)
}

// ptracePeek implements the peek operations using PT_READ_BLOCK.
// On z/OS, PT_READ_BLOCK can read arbitrary-length data efficiently.
// Unlike Linux which reads word-by-word, z/OS can read blocks directly.
func ptracePeek(req int, pid int, addr uintptr, out []byte) (count int, err error) {
	if len(out) == 0 {
		return 0, nil
	}

	// z/OS PT_READ_BLOCK can read blocks directly without word-by-word alignment
	// The buffer parameter points to the output buffer
	buffer := unsafe.Pointer(&out[0])
	err = ptracePtrWithBuffer(req, pid, addr, unsafe.Pointer(uintptr(len(out))), buffer)
	if err != nil {
		return 0, err
	}
	return len(out), nil
}

// PtracePeekText reads from the traced process's text segment.
func PtracePeekText(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(PT_READ_BLOCK, pid, addr, out)
}

// PtracePeekData reads from the traced process's data segment.
func PtracePeekData(pid int, addr uintptr, out []byte) (count int, err error) {
	return ptracePeek(PT_READ_BLOCK, pid, addr, out)
}

// PtracePeekUser reads from the traced process's user area.
// On z/OS, PT_READ_U reads a 32-bit word from the user area at the specified offset.
// To match Linux behavior, we implement word-by-word reading for arbitrary lengths.
func PtracePeekUser(pid int, addr uintptr, out []byte) (count int, err error) {
	if len(out) == 0 {
		return 0, nil
	}

	// PT_READ_U reads 32-bit words from user area
	// We need to read word-by-word to handle arbitrary lengths
	n := 0
	for n < len(out) {
		var data uint32
		err = ptracePtr(PT_READ_U, pid, addr+uintptr(n), unsafe.Pointer(&data))
		if err != nil {
			return n, err
		}

		// Copy up to 4 bytes to output
		remaining := len(out) - n
		if remaining >= 4 {
			out[n] = byte(data >> 24)
			out[n+1] = byte(data >> 16)
			out[n+2] = byte(data >> 8)
			out[n+3] = byte(data)
			n += 4
		} else {
			// Handle partial word at the end
			for i := 0; i < remaining; i++ {
				out[n+i] = byte(data >> (24 - i*8))
			}
			n += remaining
		}
	}
	return n, nil
}

// ptracePoke implements the poke operations using PT_WRITE_BLOCK.
func ptracePoke(pokeReq int, peekReq int, pid int, addr uintptr, data []byte) (count int, err error) {
	if len(data) == 0 {
		return 0, nil
	}

	// z/OS PT_WRITE_BLOCK can write blocks directly
	buffer := unsafe.Pointer(&data[0])
	err = ptracePtrWithBuffer(pokeReq, pid, addr, unsafe.Pointer(uintptr(len(data))), buffer)
	if err != nil {
		return 0, err
	}
	return len(data), nil
}

// PtracePokeText writes to the traced process's text segment.
func PtracePokeText(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(PT_WRITE_BLOCK, PT_READ_BLOCK, pid, addr, data)
}

// PtracePokeData writes to the traced process's data segment.
func PtracePokeData(pid int, addr uintptr, data []byte) (count int, err error) {
	return ptracePoke(PT_WRITE_BLOCK, PT_READ_BLOCK, pid, addr, data)
}

// PtracePokeUser writes to the traced process's user area.
// Note: z/OS does not have a direct PT_WRITE_U equivalent.
// This function returns ENOSYS to indicate it's not supported.
// DIFFERENCE FROM LINUX: Linux supports PTRACE_POKEUSR, z/OS does not.
func PtracePokeUser(pid int, addr uintptr, data []byte) (count int, err error) {
	// z/OS doesn't have PT_WRITE_U
	return 0, ENOSYS
}

// PtraceGetRegs retrieves the general purpose registers from the traced process.
// On z/OS, PT_READ_GPR can read individual registers as 64-bit values when called
// with a register number, or read all registers when called with a buffer.
// This function reads each register individually to ensure full 64-bit values.
// DIFFERENCE FROM LINUX: Linux uses PTRACE_GETREGSET with iovec, z/OS uses PT_READ_GPR.
func PtraceGetRegs(pid int, regsout *PtraceRegs) (err error) {
	// Use PT_REGHSET to enable reading high GPRs for AMODE 64 programs
	rv, rc, rn := Bpx4ptr(int32(PT_REGHSET), int32(pid), nil, nil, nil)
	_ = rv
	_ = rc
	_ = rn

	// Use PT_BLOCKREQ to read GPRs and PSW in one call
	// dbx uses 3 requests: PT_READ_GPR, PT_READ_U, PT_READ_GPRH
	// PT_READ_U is required for ptrace to populate PSW in GPR block
	//
	// IMPORTANT: z/OS ptrace always returns 64-bit ESA/390 format PSW in the Psw field.
	// The 128-bit Pswg field is never populated, even with PSWG_Req bit set.
	// We convert ESA/390 to z/Architecture format in software (matching DBX behavior).
	const (
		numRequests = 3
		reqSize     = int(unsafe.Sizeof(PtraceBlkReqReq{}))
		gprSize     = int(unsafe.Sizeof(PtraceBlkGpr{}))
		uareaSize   = int(unsafe.Sizeof(PtraceBlkUar{})) + 6*int(unsafe.Sizeof(PtraceBlkUarOcw{}))
		totalSize   = int(unsafe.Sizeof(PtraceBlkReq{})) + numRequests*reqSize + gprSize*2 + uareaSize
	)

	// Allocate buffer for block request
	buf := make([]byte, totalSize)

	// Setup block request header
	blkReq := (*PtraceBlkReq)(unsafe.Pointer(&buf[0]))
	blkReq.Numreq = numRequests

	// Setup request array
	reqOffset := int(unsafe.Sizeof(PtraceBlkReq{}))
	reqs := (*[3]PtraceBlkReqReq)(unsafe.Pointer(&buf[reqOffset]))

	// Setup GPR block (lower 32 bits)
	gprOffset := reqOffset + numRequests*reqSize
	gprBlock := (*PtraceBlkGpr)(unsafe.Pointer(&buf[gprOffset]))

	// Setup user area block (required for PSW)
	uareaOffset := gprOffset + gprSize
	uareaBlock := (*PtraceBlkUar)(unsafe.Pointer(&buf[uareaOffset]))
	uareaBlock.Num = 6 // Number of control info entries

	// Setup user area offset/control words
	uareaOcw := (*[6]PtraceBlkUarOcw)(unsafe.Pointer(&buf[uareaOffset+int(unsafe.Sizeof(PtraceBlkUar{}))]))
	uareaOcw[0].Ofs = 1025 // program interrupt code
	uareaOcw[1].Ofs = 1026 // abend completion code
	uareaOcw[2].Ofs = 1027 // abend reason code
	uareaOcw[3].Ofs = 1028 // signal code
	uareaOcw[4].Ofs = 1029 // instruction length code
	uareaOcw[5].Ofs = 1030 // process flags

	// Setup high GPR block (upper 32 bits)
	gprHighOffset := uareaOffset + uareaSize
	gprHighBlock := (*PtraceBlkGpr)(unsafe.Pointer(&buf[gprHighOffset]))

	// Configure requests (order matters: GPR, U, GPRH)
	// reqdata contains OFFSET from buffer start, not absolute address
	// dbx: brr[0].reqdata=(unsigned long)brg; where brg is the offset
	reqs[0].Reqtype = PT_READ_GPR
	reqs[0].Reqdata = uint32(gprOffset)
	reqs[1].Reqtype = PT_READ_U
	reqs[1].Reqdata = uint32(uareaOffset)
	reqs[2].Reqtype = PT_READ_GPRH
	reqs[2].Reqdata = uint32(gprHighOffset)

	// Execute block request
	rvBlk, rcBlk, rnBlk := Bpx4ptr(int32(PT_BLOCKREQ), int32(pid),
		unsafe.Pointer(&buf[0]), unsafe.Pointer(uintptr(totalSize)), nil)
	if rvBlk == -1 {
		return errnoErr2(Errno(rcBlk), uintptr(rnBlk))
	}

	// Extract GPRs (combine low and high 32 bits)
	for i := 0; i < 16; i++ {
		low := uint64(gprBlock.Gpr[i])
		high := uint64(gprHighBlock.Gpr[i]) << 32
		regsout.Gprs[i] = high | low
	}

	// Extract PSW from GPR block structure
	// z/OS ptrace ALWAYS returns 8-byte ESA/390 format PSW at offset 144.
	// The 16-byte PSWG field at offset 152 is NEVER populated (always zero).
	// We must convert ESA/390 format to z/Architecture format in software (like DBX does).
	//
	// ESA/390 PSW format (8 bytes):
	//   Word 1 (bytes 0-3): Mask with bit 12 = 1 (ECMODE31BIT)
	//   Word 2 (bytes 4-7): Address with bit 0 = AMODE bit (1=31-bit, 0=64-bit)
	//
	// z/Architecture PSW format (16 bytes):
	//   Mask (bytes 0-7): Extended mask with bit 12 = 0, bits 31-32 = EA+BA (AMODE)
	//   Addr (bytes 8-15): Full 64-bit instruction address

	// Read ESA/390 PSW as two 32-bit words
	pswWord1 := uint32(gprBlock.Psw[0])<<24 | uint32(gprBlock.Psw[1])<<16 |
		uint32(gprBlock.Psw[2])<<8 | uint32(gprBlock.Psw[3])
	pswWord2 := uint32(gprBlock.Psw[4])<<24 | uint32(gprBlock.Psw[5])<<16 |
		uint32(gprBlock.Psw[6])<<8 | uint32(gprBlock.Psw[7])

	// Convert ESA/390 to z/Architecture format (matching DBX psw_ESA390_to_zArchitecture)
	const ECMODE31BIT = uint64(0x0000000000080000)
	const AMODE31BIT = uint64(0x0000000080000000)
	const ADDR_MASK_31 = uint64(0x000000007FFFFFFF)

	// Extract AMODE bit from address word (bit 0 of word 2)
	amode31 := uint64(pswWord2) & AMODE31BIT

	// Mask off AMODE bit from address
	addr := uint64(pswWord2) & ADDR_MASK_31

	// Convert mask: clear ECMODE31BIT, shift left 32 bits, add AMODE bit
	mask := uint64(pswWord1) & ^ECMODE31BIT
	mask = (mask << 32) | amode31

	regsout.Psw.Mask = mask
	regsout.Psw.Addr = addr

	// Note: Access registers (Acrs), floating point registers (Fp_regs),
	// PER info (Per_info), and other fields would require additional
	// PT_READ_* calls. For now, we focus on GPRs and PSW which are
	// the most commonly used registers for debugging.

	return nil
}

// ptraceWritePSW writes only the PSW address register.
// This is a lower-level function used internally for single-stepping.
// Based on ztrace's write_psw function.
func ptraceWritePSW(pid int, pswAddr uint32) error {
	const PTRACE_REG_PSWA = 41
	rv, rc, rn := Bpx4ptr(int32(PT_WRITE_GPR), int32(pid),
		unsafe.Pointer(uintptr(PTRACE_REG_PSWA)),
		unsafe.Pointer(uintptr(pswAddr)),
		nil)
	if rv == -1 {
		return errnoErr2(Errno(rc), uintptr(rn))
	}
	return nil
}

// PtraceSetRegs sets the general purpose registers in the traced process.
// Uses PT_BLOCKREQ to write GPRs and PSW in a single call, matching dbx implementation.
// DIFFERENCE FROM LINUX: Linux uses PTRACE_SETREGSET with iovec, z/OS uses PT_BLOCKREQ.
// NOTE: The process must be in a stopped state for register writes to succeed.
func PtraceSetRegs(pid int, regs *PtraceRegs) (err error) {
	// Use PT_REGHSET to enable writing high GPRs for AMODE 64 programs
	rv, rc, rn := Bpx4ptr(int32(PT_REGHSET), int32(pid), nil, nil, nil)
	_ = rv
	_ = rc
	_ = rn

	// Use PT_BLOCKREQ to write GPRs and PSW in one call
	const (
		numRequests = 2
		reqSize     = int(unsafe.Sizeof(PtraceBlkReqReq{}))
		gprSize     = int(unsafe.Sizeof(PtraceBlkGpr{}))
		totalSize   = int(unsafe.Sizeof(PtraceBlkReq{})) + numRequests*reqSize + gprSize*2
	)

	// Allocate buffer for block request
	buf := make([]byte, totalSize)

	// Setup block request header
	blkReq := (*PtraceBlkReq)(unsafe.Pointer(&buf[0]))
	blkReq.Numreq = numRequests

	// Setup request array
	reqOffset := int(unsafe.Sizeof(PtraceBlkReq{}))
	reqs := (*[2]PtraceBlkReqReq)(unsafe.Pointer(&buf[reqOffset]))

	// Setup GPR block (lower 32 bits)
	gprOffset := reqOffset + numRequests*reqSize
	gprBlock := (*PtraceBlkGpr)(unsafe.Pointer(&buf[gprOffset]))

	// Setup high GPR block (upper 32 bits)
	gprHighOffset := gprOffset + gprSize
	gprHighBlock := (*PtraceBlkGpr)(unsafe.Pointer(&buf[gprHighOffset]))

	// Mark all GPRs as modified (set all 16 bits)
	gprBlock.Writebitflags = 0xFFFF
	gprHighBlock.Writebitflags = 0xFFFF

	// Split 64-bit GPRs into low and high 32 bits
	for i := 0; i < 16; i++ {
		gprBlock.Gpr[i] = uint32(regs.Gprs[i] & 0xFFFFFFFF)
		gprHighBlock.Gpr[i] = uint32(regs.Gprs[i] >> 32)
	}

	// Pack PSW for 31-bit targets (current z/OS)
	// Upper 32 bits = mask, lower 32 bits = address
	gprBlock.Wpsw = 1 // Mark PSW as modified
	// Write PSW in old format (8 bytes at offset 144)
	// Word 1 (mask) at bytes 0-3, Word 2 (addr) at bytes 4-7
	pswMask := uint32(regs.Psw.Mask)
	pswAddr := uint32(regs.Psw.Addr)
	gprBlock.Psw[0] = byte(pswMask >> 24)
	gprBlock.Psw[1] = byte(pswMask >> 16)
	gprBlock.Psw[2] = byte(pswMask >> 8)
	gprBlock.Psw[3] = byte(pswMask)
	gprBlock.Psw[4] = byte(pswAddr >> 24)
	gprBlock.Psw[5] = byte(pswAddr >> 16)
	gprBlock.Psw[6] = byte(pswAddr >> 8)
	gprBlock.Psw[7] = byte(pswAddr)

	// Clear extended PSWG (not used for 31-bit targets)
	for i := 0; i < 16; i++ {
		gprBlock.Pswg[i] = 0
	}

	// Configure requests
	// reqdata contains OFFSET from buffer start, not absolute address
	reqs[0].Reqtype = PT_WRITE_GPR
	reqs[0].Reqdata = uint32(gprOffset)
	reqs[1].Reqtype = PT_WRITE_GPRH
	reqs[1].Reqdata = uint32(gprHighOffset)

	// Execute block request
	rvBlk, rcBlk, rnBlk := Bpx4ptr(int32(PT_BLOCKREQ), int32(pid),
		unsafe.Pointer(&buf[0]), unsafe.Pointer(uintptr(totalSize)), nil)
	if rvBlk == -1 {
		return errnoErr2(Errno(rcBlk), uintptr(rnBlk))
	}

	return nil
}

// PtraceSetOptions sets ptrace options.
// DIFFERENCE FROM LINUX: z/OS doesn't have PTRACE_SETOPTIONS equivalent.
// Returns ENOSYS to indicate this is not supported on z/OS.
func PtraceSetOptions(pid int, options int) (err error) {
	return ENOSYS
}

// PtraceGetEventMsg retrieves a message about the ptrace event.
// DIFFERENCE FROM LINUX: z/OS doesn't have PTRACE_GETEVENTMSG equivalent.
// Returns ENOSYS to indicate this is not supported on z/OS.
func PtraceGetEventMsg(pid int) (msg uint, err error) {
	return 0, ENOSYS
}

// PtraceSyscall continues execution and stops at the next syscall entry/exit.
// DIFFERENCE FROM LINUX: z/OS doesn't have PTRACE_SYSCALL equivalent.
// This function falls back to PT_CONTINUE and will NOT stop at syscalls.
//
// To trace syscalls on z/OS, applications must:
// 1. Set breakpoints at BPX syscall entry points (requires knowledge of syscall vector)
// 2. Detect BASR instructions that call into the BPX syscall vector
// 3. Use PT_LDINFO to identify loaded modules and their entry points
func PtraceSyscall(pid int, signal int) (err error) {
	// z/OS doesn't have a direct PTRACE_SYSCALL equivalent
	// Use PT_CONTINUE as a fallback (won't stop at syscalls)
	return ptrace(PT_CONTINUE, pid, 0, uintptr(signal))
}

// Note: PtraceSingleStep is implemented in ptrace_zos_singlestep.go
// with full two-breakpoint emulation for branch instructions.

// PtraceInterrupt interrupts the traced process.
// DIFFERENCE FROM LINUX: z/OS doesn't have PTRACE_INTERRUPT equivalent.
// Returns ENOSYS to indicate this is not supported on z/OS.
func PtraceInterrupt(pid int) (err error) {
	return ENOSYS
}

// PtraceSeize attaches to a process without stopping it.
// DIFFERENCE FROM LINUX: z/OS doesn't have PTRACE_SEIZE equivalent.
// Uses PT_ATTACH as a fallback, which will stop the process.
func PtraceSeize(pid int) (err error) {
	// z/OS doesn't have PTRACE_SEIZE, use PT_ATTACH
	// Note: This WILL stop the process, unlike Linux PTRACE_SEIZE
	return ptrace(PT_ATTACH, pid, 0, 0)
}
