// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs_windows.go

package runtime

const (
	_PROT_NONE  = 0
	_PROT_READ  = 1
	_PROT_WRITE = 2
	_PROT_EXEC  = 4

	_MAP_ANON    = 1
	_MAP_PRIVATE = 2

	_DUPLICATE_SAME_ACCESS   = 0x2
	_THREAD_PRIORITY_HIGHEST = 0x2

	_SIGPROF          = 0 // dummy value for badsignal
	_SIGINT           = 0x2
	_CTRL_C_EVENT     = 0x0
	_CTRL_BREAK_EVENT = 0x1

	_CONTEXT_CONTROL = 0x100001
	_CONTEXT_FULL    = 0x10000b

	_EXCEPTION_ACCESS_VIOLATION     = 0xc0000005
	_EXCEPTION_BREAKPOINT           = 0x80000003
	_EXCEPTION_FLT_DENORMAL_OPERAND = 0xc000008d
	_EXCEPTION_FLT_DIVIDE_BY_ZERO   = 0xc000008e
	_EXCEPTION_FLT_INEXACT_RESULT   = 0xc000008f
	_EXCEPTION_FLT_OVERFLOW         = 0xc0000091
	_EXCEPTION_FLT_UNDERFLOW        = 0xc0000093
	_EXCEPTION_INT_DIVIDE_BY_ZERO   = 0xc0000094
	_EXCEPTION_INT_OVERFLOW         = 0xc0000095

	_INFINITE     = 0xffffffff
	_WAIT_TIMEOUT = 0x102

	_EXCEPTION_CONTINUE_EXECUTION = -0x1
	_EXCEPTION_CONTINUE_SEARCH    = 0x0
)

type systeminfo struct {
	anon0                       [4]byte
	dwpagesize                  uint32
	lpminimumapplicationaddress *byte
	lpmaximumapplicationaddress *byte
	dwactiveprocessormask       uint64
	dwnumberofprocessors        uint32
	dwprocessortype             uint32
	dwallocationgranularity     uint32
	wprocessorlevel             uint16
	wprocessorrevision          uint16
}

type exceptionrecord struct {
	exceptioncode        uint32
	exceptionflags       uint32
	exceptionrecord      *exceptionrecord
	exceptionaddress     *byte
	numberparameters     uint32
	pad_cgo_0            [4]byte
	exceptioninformation [15]uint64
}

type m128a struct {
	low  uint64
	high int64
}

type context struct {
	p1home               uint64
	p2home               uint64
	p3home               uint64
	p4home               uint64
	p5home               uint64
	p6home               uint64
	contextflags         uint32
	mxcsr                uint32
	segcs                uint16
	segds                uint16
	seges                uint16
	segfs                uint16
	seggs                uint16
	segss                uint16
	eflags               uint32
	dr0                  uint64
	dr1                  uint64
	dr2                  uint64
	dr3                  uint64
	dr6                  uint64
	dr7                  uint64
	rax                  uint64
	rcx                  uint64
	rdx                  uint64
	rbx                  uint64
	rsp                  uint64
	rbp                  uint64
	rsi                  uint64
	rdi                  uint64
	r8                   uint64
	r9                   uint64
	r10                  uint64
	r11                  uint64
	r12                  uint64
	r13                  uint64
	r14                  uint64
	r15                  uint64
	rip                  uint64
	anon0                [512]byte
	vectorregister       [26]m128a
	vectorcontrol        uint64
	debugcontrol         uint64
	lastbranchtorip      uint64
	lastbranchfromrip    uint64
	lastexceptiontorip   uint64
	lastexceptionfromrip uint64
}

type overlapped struct {
	internal     uint64
	internalhigh uint64
	anon0        [8]byte
	hevent       *byte
}
