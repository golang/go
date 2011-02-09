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
	// On 64-bit, people with ulimit -v set complain if we reserve too
	// much address space.  Instead, assume that the reservation is okay
	// and check the assumption in SysMap.
	if(sizeof(void*) == 8)
		return v;
	
	return runtime·mmap(v, n, PROT_NONE, MAP_ANON|MAP_PRIVATE, -1, 0);
}

void
runtime·SysMap(void *v, uintptr n)
{
	void *p;
	
	mstats.sys += n;

	// On 64-bit, we don't actually have v reserved, so tread carefully.
	if(sizeof(void*) == 8) {
		p = runtime·mmap(v, n, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_PRIVATE, -1, 0);
		if(p != v) {
			runtime·printf("runtime: address space conflict: map(%v) = %v\n", v, p);
			runtime·throw("runtime: address space conflict");
		}
		return;
	}

	p = runtime·mmap(v, n, PROT_READ|PROT_WRITE|PROT_EXEC, MAP_ANON|MAP_FIXED|MAP_PRIVATE, -1, 0);
	if(p != v)
		runtime·throw("runtime: cannot map pages in arena address space");
}
