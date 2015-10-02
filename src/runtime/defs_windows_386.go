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

	_SIGINT           = 0x2
	_CTRL_C_EVENT     = 0x0
	_CTRL_BREAK_EVENT = 0x1

	_CONTEXT_CONTROL = 0x10001
	_CONTEXT_FULL    = 0x10007

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
	dwactiveprocessormask       uint32
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
	exceptioninformation [15]uint32
}

type floatingsavearea struct {
	controlword   uint32
	statusword    uint32
	tagword       uint32
	erroroffset   uint32
	errorselector uint32
	dataoffset    uint32
	dataselector  uint32
	registerarea  [80]uint8
	cr0npxstate   uint32
}

type context struct {
	contextflags      uint32
	dr0               uint32
	dr1               uint32
	dr2               uint32
	dr3               uint32
	dr6               uint32
	dr7               uint32
	floatsave         floatingsavearea
	seggs             uint32
	segfs             uint32
	seges             uint32
	segds             uint32
	edi               uint32
	esi               uint32
	ebx               uint32
	edx               uint32
	ecx               uint32
	eax               uint32
	ebp               uint32
	eip               uint32
	segcs             uint32
	eflags            uint32
	esp               uint32
	segss             uint32
	extendedregisters [512]uint8
}

func (c *context) ip() uintptr { return uintptr(c.eip) }
func (c *context) sp() uintptr { return uintptr(c.esp) }

func (c *context) setip(x uintptr) { c.eip = uint32(x) }
func (c *context) setsp(x uintptr) { c.esp = uint32(x) }

func dumpregs(r *context) {
	print("eax     ", hex(r.eax), "\n")
	print("ebx     ", hex(r.ebx), "\n")
	print("ecx     ", hex(r.ecx), "\n")
	print("edx     ", hex(r.edx), "\n")
	print("edi     ", hex(r.edi), "\n")
	print("esi     ", hex(r.esi), "\n")
	print("ebp     ", hex(r.ebp), "\n")
	print("esp     ", hex(r.esp), "\n")
	print("eip     ", hex(r.eip), "\n")
	print("eflags  ", hex(r.eflags), "\n")
	print("cs      ", hex(r.segcs), "\n")
	print("fs      ", hex(r.segfs), "\n")
	print("gs      ", hex(r.seggs), "\n")
}

type overlapped struct {
	internal     uint32
	internalhigh uint32
	anon0        [8]byte
	hevent       *byte
}
