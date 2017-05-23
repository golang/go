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

type exccontext struct {
	size                    uint32
	portable_context_offset uint32
	portable_context_size   uint32
	arch                    uint32
	regs_size               uint32
	reserved                [11]uint32
	regs                    excregs386
}

type excportablecontext struct {
	pc uint32
	sp uint32
	fp uint32
}
