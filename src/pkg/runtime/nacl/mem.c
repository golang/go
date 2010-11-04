#include "runtime.h"
#include "defs.h"
#include "os.h"
#include "malloc.h"

enum {
	NaclPage = 0x10000
};

void*
runtime·SysAlloc(uintptr n)
{
	mstats.sys += n;
	return runtime·mmap(nil, n, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_PRIVATE, -1, 0);
}

void
runtime·SysUnused(void *v, uintptr n)
{
	USED(v);
	USED(n);
	// TODO(rsc): call madvise MADV_DONTNEED
}

void
runtime·SysFree(void *v, uintptr n)
{
	// round to page size or else nacl prints annoying log messages
	mstats.sys -= n;
	n = (n+NaclPage-1) & ~(NaclPage-1);
	runtime·munmap(v, n);
}

void
runtime·SysMemInit(void)
{
}
