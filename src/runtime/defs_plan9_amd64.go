package runtime

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
