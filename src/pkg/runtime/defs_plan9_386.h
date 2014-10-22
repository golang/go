#define PAGESIZE 0x1000

typedef struct Ureg Ureg;

struct Ureg
{
	uint32	di;		/* general registers */
	uint32	si;		/* ... */
	uint32	bp;		/* ... */
	uint32	nsp;
	uint32	bx;		/* ... */
	uint32	dx;		/* ... */
	uint32	cx;		/* ... */
	uint32	ax;		/* ... */
	uint32	gs;		/* data segments */
	uint32	fs;		/* ... */
	uint32	es;		/* ... */
	uint32	ds;		/* ... */
	uint32	trap;		/* trap type */
	uint32	ecode;		/* error code (or zero) */
	uint32	pc;		/* pc */
	uint32	cs;		/* old context */
	uint32	flags;		/* old flags */
	uint32	sp;
	uint32	ss;		/* old stack segment */
};
