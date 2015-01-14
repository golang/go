package runtime

const (
	// These values are referred to in the source code
	// but really don't matter. Even so, use the standard numbers.
	_SIGQUIT = 3
	_SIGSEGV = 11
	_SIGPROF = 27
)

type timespec struct {
	tv_sec  int64
	tv_nsec int32
}

type excregs386 struct {
	eax    uint32
	ecx    uint32
	edx    uint32
	ebx    uint32
	esp    uint32
	ebp    uint32
	esi    uint32
	edi    uint32
	eip    uint32
	eflags uint32
}

type excregsamd64 struct {
	rax    uint64
	rcx    uint64
	rdx    uint64
	rbx    uint64
	rsp    uint64
	rbp    uint64
	rsi    uint64
	rdi    uint64
	r8     uint64
	r9     uint64
	r10    uint64
	r11    uint64
	r12    uint64
	r13    uint64
	r14    uint64
	r15    uint64
	rip    uint64
	rflags uint32
}

type exccontext struct {
	size                    uint32
	portable_context_offset uint32
	portable_context_size   uint32
	arch                    uint32
	regs_size               uint32
	reserved                [11]uint32
	regs                    excregsamd64
}

type excportablecontext struct {
	pc uint32
	sp uint32
	fp uint32
}
