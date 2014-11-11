package runtime

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
