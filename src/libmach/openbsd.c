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
	sysfatal("ctlproc unimplemented in OpenBSD");
	return -1;
}

char*
proctextfile(int pid)
{
	USED(pid);
	sysfatal("proctextfile unimplemented in OpenBSD");
	return nil;
}

char*
procstatus(int pid)
{
	USED(pid);
	sysfatal("procstatus unimplemented in OpenBSD");
	return nil;
}

Map*
attachproc(int pid, Fhdr *fp)
{
	USED(pid);
	USED(fp);
	sysfatal("attachproc unimplemented in OpenBSD");
	return nil;
}

void
detachproc(Map *m)
{
	USED(m);
	sysfatal("detachproc unimplemented in OpenBSD");
}

int
procthreadpids(int pid, int *p, int np)
{
	USED(pid);
	USED(p);
	USED(np);
	sysfatal("procthreadpids unimplemented in OpenBSD");
	return -1;
}
