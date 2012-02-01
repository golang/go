// This is stubbed out for the moment. Will revisit when the time comes.
#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>

int
ctlproc(int pid, char *msg)
{
	USED(pid);
	USED(msg);

	sysfatal("ctlproc unimplemented on Plan 9");
	return -1;
}

char*
proctextfile(int pid)
{
	USED(pid);

	sysfatal("proctextfile unimplemented on Plan 9");
	return nil;
}

char*
procstatus(int pid)
{
	USED(pid);

	sysfatal("procstatus unimplemented on Plan 9");
	return nil;
}

Map*
attachproc(int pid, Fhdr *fp)
{
	USED(pid);
	USED(fp);

	sysfatal("attachproc unimplemented on Plan 9");
	return nil;
}

void
detachproc(Map *m)
{
	USED(m);

	sysfatal("detachproc unimplemented on Plan 9");
}

int
procthreadpids(int pid, int *p, int np)
{
	USED(pid);
	USED(p);
	USED(np);

	sysfatal("procthreadpids unimplemented on Plan 9");
	return -1;
}

int 
nanosleep(const struct timespec *rqtp, struct timespec *rmtp)
{
	USED(rqtp);
	USED(rmtp);

	sysfatal("nanosleep unimplemented on Plan 9");
	return -1;
}
