#define PAGESIZE 0x200000ULL

typedef struct Ureg Ureg;

struct Ureg {
	uint64	ax;
	uint64	bx;
	uint64	cx;
	uint64	dx;
	uint64	si;
	uint64	di;
	uint64	bp;
	uint64	r8;
	uint64	r9;
	uint64	r10;
	uint64	r11;
	uint64	r12;
	uint64	r13;
	uint64	r14;
	uint64	r15;

	uint16	ds;
	uint16	es;
	uint16	fs;
	uint16	gs;

	uint64	type;
	uint64	error;				/* error code (or zero) */
	uint64	ip;				/* pc */
	uint64	cs;				/* old context */
	uint64	flags;				/* old flags */
	uint64	sp;				/* sp */
	uint64	ss;				/* old stack segment */
};
