// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.arm64e

package runtime

// On arm64e the kernel signs pc, lr, sp and fp in a signal's thread state,
// and authenticates pc, lr and sp when the handler returns. These accessors
// follow the __darwin_arm_thread_state64_get_* and _set_* macros in
// <mach/arm/_structs.h>, except that the getters strip the signatures
// instead of authenticating them. A register that held a non-canonical
// value when the signal arrived, such as the pc after a call to an
// invalid address, has a signature that does not authenticate, and an
// authenticate instruction faulting in the signal handler would hang the
// thread instead of reporting the original fault.

const (
	_ARM_THREAD_STATE64_FLAGS_NO_PTRAUTH       = 0x1
	_ARM_THREAD_STATE64_FLAGS_IB_SIGNED_LR     = 0x2
	_ARM_THREAD_STATE64_FLAGS_KERNEL_SIGNED_PC = 0x4
	_ARM_THREAD_STATE64_FLAGS_KERNEL_SIGNED_LR = 0x8
	_ARM_THREAD_STATE64_USER_DIVERSIFIER_MASK  = 0xff000000

	// ptrauth_string_discriminator("pc"), ("lr") and ("sp").
	ptrauthPC = 0x7481
	ptrauthLR = 0x77d3
	ptrauthSP = 0xcbed
)

func (c *sigctxt) r29() uint64 { return c.regs().stripData(c.regs().fp) }
func (c *sigctxt) sp() uint64  { return c.regs().stripData(c.regs().sp) }

// lr does not always strip. When x30 does not hold a canonical address,
// the kernel leaves it as it was and sets IB_SIGNED_LR. C code that signed
// its return address with key IB does that, and so does Go code, because
// the compiler uses R30 as a general register in a function that is not a
// leaf. Like the SDK's getter, return x30 unchanged then: pushCall saves
// the result, and asyncPreempt restores R30 from it, so stripping a value
// that is not an address would change a register of the preempted goroutine.
func (c *sigctxt) lr() uint64 {
	r := c.regs()
	if r.flags&(_ARM_THREAD_STATE64_FLAGS_NO_PTRAUTH|_ARM_THREAD_STATE64_FLAGS_IB_SIGNED_LR) != 0 {
		return r.lr
	}
	return xpaci(r.lr)
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uint64 { return c.regs().stripCode(c.regs().pc) }

//go:nosplit
func (r *regs64) stripCode(p uint64) uint64 {
	if r.flags&_ARM_THREAD_STATE64_FLAGS_NO_PTRAUTH != 0 {
		return p
	}
	return xpaci(p)
}

//go:nosplit
func (r *regs64) stripData(p uint64) uint64 {
	if r.flags&_ARM_THREAD_STATE64_FLAGS_NO_PTRAUTH != 0 {
		return p
	}
	return xpacd(p)
}

// codeModifier returns the modifier of the pc or lr field: disc, blended
// with the user diversifier unless the kernel signed the field.
func (r *regs64) codeModifier(disc uint64, kernelSigned uint32) uint64 {
	if user := uint64(r.flags & _ARM_THREAD_STATE64_USER_DIVERSIFIER_MASK); user != 0 && r.flags&kernelSigned == 0 {
		return user | disc<<48
	}
	return disc
}

// set_pc signs x, which is always one of the runtime's entry points. It
// does not check the old pc, which it replaces: after a call to an
// invalid address, that pc does not authenticate, and preparePanic must
// still be able to redirect the thread to sigpanic.
func (c *sigctxt) set_pc(x uint64) {
	r := c.regs()
	if x != 0 && r.flags&_ARM_THREAD_STATE64_FLAGS_NO_PTRAUTH == 0 {
		x = pacia(x, r.codeModifier(ptrauthPC, 0))
	}
	r.pc = x
	r.flags &^= _ARM_THREAD_STATE64_FLAGS_KERNEL_SIGNED_PC
}

// set_sp and set_lr accept only the values that preparePanic and pushCall
// derive from the interrupted state. pushsp and resignpc check them
// against that state, and that the state is signed as expected, before
// signing them, so that they cannot sign anything else. The check signs
// the stripped old value again and compares, instead of authenticating
// it, so a value that does not authenticate makes the runtime throw
// instead of faulting in the signal handler.

func (c *sigctxt) set_sp(x uint64) {
	r := c.regs()
	if r.flags&_ARM_THREAD_STATE64_FLAGS_NO_PTRAUTH != 0 {
		r.sp = x
		return
	}
	if x = pushsp(r.sp, x, ptrauthSP); x == 0 {
		throw("bad signal sp")
	}
	r.sp = x
}

// set_lr must be called before set_pc, because it checks x against the
// interrupted pc.
func (c *sigctxt) set_lr(x uint64) {
	r := c.regs()
	if r.flags&_ARM_THREAD_STATE64_FLAGS_NO_PTRAUTH != 0 {
		r.lr = x
		return
	}
	// lr becomes pc, or a restart point of pc's function (see
	// isAsyncSafePoint).
	lo := xpaci(r.pc)
	if f := findfunc(uintptr(lo)); f.valid() {
		lo = uint64(f.entry())
	}
	pcmod := r.codeModifier(ptrauthPC, _ARM_THREAD_STATE64_FLAGS_KERNEL_SIGNED_PC)
	if x = resignpc(r.pc, pcmod, x, lo, r.codeModifier(ptrauthLR, 0)); x == 0 {
		throw("bad signal lr")
	}
	r.lr = x
	r.flags &^= _ARM_THREAD_STATE64_FLAGS_KERNEL_SIGNED_LR | _ARM_THREAD_STATE64_FLAGS_IB_SIGNED_LR
}

// Implemented in sys_darwin_arm64.s.
func xpaci(p uint64) uint64
func xpacd(p uint64) uint64
func pacia(p, mod uint64) uint64

// pushsp returns newsp signed with key DA and modifier mod, or 0 if sp is
// not signed that way or newsp is not within 16 bytes below sp.
func pushsp(sp, newsp, mod uint64) uint64

// resignpc returns newpc signed with key IA and modifier mod, or 0 if pc
// is not signed with key IA and modifier pcmod or newpc is not in
// [lo, pc].
func resignpc(pc, pcmod, newpc, lo, mod uint64) uint64
