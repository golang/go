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
	sysfatal("ctlproc unimplemented in Windows");
	return -1;
}

char*
proctextfile(int pid)
{
	USED(pid);
	sysfatal("proctextfile unimplemented in Windows");
	return nil;
}

char*
procstatus(int pid)
{
	USED(pid);
	sysfatal("procstatus unimplemented in Windows");
	return nil;
}

Map*
attachproc(int pid, Fhdr *fp)
{
	USED(pid);
	USED(fp);
	sysfatal("attachproc unimplemented in Windows");
	return nil;
}

void
detachproc(Map *m)
{
	USED(m);
	sysfatal("detachproc unimplemented in Windows");
}

int
procthreadpids(int pid, int *p, int np)
{
	USED(pid);
	USED(p);
	USED(np);
	sysfatal("procthreadpids unimplemented in Windows");
	return -1;
}

int 
pread(int fd, void *buf, int count, int offset)
{
	int oldoffset, n;
	
	oldoffset = seek(fd, offset, 0);
	n = read(fd, buf, count);
	seek(fd, oldoffset, 0);
	return n;
}

int 
pwrite(int fd, void *buf, int count, int offset)
{
	USED(fd);
	USED(buf);
	USED(count);
	USED(offset);
	sysfatal("pwrite unimplemented in Windows");
	return -1;
}

int 
nanosleep(const struct timespec *rqtp, struct timespec *rmtp)
{
	USED(rqtp);
	USED(rmtp);
	sysfatal("nanosleep unimplemented in Windows");
	return -1;
}
