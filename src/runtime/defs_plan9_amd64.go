package runtime

const _PAGESIZE = 0x1000

type ureg struct {
	ax  uint64
	bx  uint64
	cx  uint64
	dx  uint64
	si  uint64
	di  uint64
	bp  uint64
	r8  uint64
	r9  uint64
	r10 uint64
	r11 uint64
	r12 uint64
	r13 uint64
	r14 uint64
	r15 uint64

	ds uint16
	es uint16
	fs uint16
	gs uint16

	_type uint64
	error uint64 /* error code (or zero) */
	ip    uint64 /* pc */
	cs    uint64 /* old context */
	flags uint64 /* old flags */
	sp    uint64 /* sp */
	ss    uint64 /* old stack segment */
}

type sigctxt struct {
	u *ureg
}

func (c *sigctxt) pc() uintptr { return uintptr(c.u.ip) }
func (c *sigctxt) sp() uintptr { return uintptr(c.u.sp) }

func (c *sigctxt) setpc(x uintptr) { c.u.ip = uint64(x) }
func (c *sigctxt) setsp(x uintptr) { c.u.sp = uint64(x) }

func dumpregs(u *ureg) {
	print("ax    ", hex(u.ax), "\n")
	print("bx    ", hex(u.bx), "\n")
	print("cx    ", hex(u.cx), "\n")
	print("dx    ", hex(u.dx), "\n")
	print("di    ", hex(u.di), "\n")
	print("si    ", hex(u.si), "\n")
	print("bp    ", hex(u.bp), "\n")
	print("sp    ", hex(u.sp), "\n")
	print("r8    ", hex(u.r8), "\n")
	print("r9    ", hex(u.r9), "\n")
	print("r10   ", hex(u.r10), "\n")
	print("r11   ", hex(u.r11), "\n")
	print("r12   ", hex(u.r12), "\n")
	print("r13   ", hex(u.r13), "\n")
	print("r14   ", hex(u.r14), "\n")
	print("r15   ", hex(u.r15), "\n")
	print("ip    ", hex(u.ip), "\n")
	print("flags ", hex(u.flags), "\n")
	print("cs    ", hex(u.cs), "\n")
	print("fs    ", hex(u.fs), "\n")
	print("gs    ", hex(u.gs), "\n")
}
