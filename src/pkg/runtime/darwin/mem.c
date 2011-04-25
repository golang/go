#include "runtime.h"
#include "defs.h"
#include "os.h"
#include "malloc.h"

void*
runtime·SysAlloc(uintptr n)
{
	void *v;

	mstats.sys += n;
	v = runtime·mmap(nil, n, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_PRIVATE, -1, 0);
	if(v < (void*)4096)
		return nil;
	return v;
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
	mstats.sys -= n;
	runtime·munmap(v, n);
}

void*
runtime·SysReserve(void *v, uintptr n)
{
	return runtime·mmap(v, n, PROT_NONE, MAP_ANON|MAP_PRIVATE, -1, 0);
}

enum
{
	ENOMEM = 12,
};

void
runtime·SysMap(void *v, uintptr n)
{
	void *p;
	
	mstats.sys += n;
	p = runtime·mmap(v, n, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_FIXED|MAP_PRIVATE, -1, 0);
	if(p == (void*)-ENOMEM)
		runtime·throw("runtime: out of memory");
	if(p != v)
		runtime·throw("runtime: cannot map pages in arena address space");
}
