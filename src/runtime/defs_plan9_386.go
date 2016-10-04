package runtime

const _PAGESIZE = 0x1000

type ureg struct {
	di    uint32 /* general registers */
	si    uint32 /* ... */
	bp    uint32 /* ... */
	nsp   uint32
	bx    uint32 /* ... */
	dx    uint32 /* ... */
	cx    uint32 /* ... */
	ax    uint32 /* ... */
	gs    uint32 /* data segments */
	fs    uint32 /* ... */
	es    uint32 /* ... */
	ds    uint32 /* ... */
	trap  uint32 /* trap _type */
	ecode uint32 /* error code (or zero) */
	pc    uint32 /* pc */
	cs    uint32 /* old context */
	flags uint32 /* old flags */
	sp    uint32
	ss    uint32 /* old stack segment */
}

type sigctxt struct {
	u *ureg
}

//go:nosplit
//go:nowritebarrierrec
func (c *sigctxt) pc() uintptr { return uintptr(c.u.pc) }

func (c *sigctxt) sp() uintptr { return uintptr(c.u.sp) }
func (c *sigctxt) lr() uintptr { return uintptr(0) }

func (c *sigctxt) setpc(x uintptr) { c.u.pc = uint32(x) }
func (c *sigctxt) setsp(x uintptr) { c.u.sp = uint32(x) }
func (c *sigctxt) setlr(x uintptr) {}

func (c *sigctxt) savelr(x uintptr) {}

func dumpregs(u *ureg) {
	print("ax    ", hex(u.ax), "\n")
	print("bx    ", hex(u.bx), "\n")
	print("cx    ", hex(u.cx), "\n")
	print("dx    ", hex(u.dx), "\n")
	print("di    ", hex(u.di), "\n")
	print("si    ", hex(u.si), "\n")
	print("bp    ", hex(u.bp), "\n")
	print("sp    ", hex(u.sp), "\n")
	print("pc    ", hex(u.pc), "\n")
	print("flags ", hex(u.flags), "\n")
	print("cs    ", hex(u.cs), "\n")
	print("fs    ", hex(u.fs), "\n")
	print("gs    ", hex(u.gs), "\n")
}

func sigpanictramp() {}
