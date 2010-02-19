// just the write function

extern void ·write(int32 fd, void *v, int32 len, int32 cap);	// slice, spelled out

int32
write(int32 fd, void *v, int32 len)
{
	·write(fd, v, len, len);
	return len;
}
