// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


/*
 * basic types
 */
typedef	signed char		int8;
typedef	unsigned char		uint8;
typedef	signed short		int16;
typedef	unsigned short		uint16;
typedef	signed int		int32;
typedef	unsigned int		uint32;
typedef	signed long long int	int64;
typedef	unsigned long long int	uint64;
typedef	float			float32;
typedef	double			float64;

/*
 * get rid of C types
 */
#define	unsigned		XXunsigned
#define	signed			XXsigned
#define	char			XXchar
#define	short			XXshort
#define	int			XXint
#define	long			XXlong
#define	float			XXfloat
#define	double			XXdouble

/*
 * defined types
 */
typedef	uint8			bool;
typedef	uint8			byte;
typedef	struct
{
	int32	len;
	byte	str[1];
}				*string;
typedef	struct
{
	byte*	name;
	uint32	hash;
	void	(*fun)(void);
}				Sigs;
typedef	struct
{
	byte*	name;
	uint32	hash;
	uint32	offset;
}				Sigi;
typedef	struct	Map		Map;
struct	Map
{
	Sigi*	si;
	Sigs*	ss;
	Map*	link;
	int32	bad;
	int32	unused;
	void	(*fun[])(void);
};

/*
 * defined constants
 */
enum
{
	true	= 1,
	false	= 0,
};

/*
 * defined macros
 *    you need super-goru privilege
 *    to add this list.
 */
#define	nelem(x)	(sizeof(x)/sizeof((x)[0]))
#define	nil		((void*)0)

/*
 * common functions and data
 */
int32 strcmp(byte*, byte*);
int32 findnull(int8*);
void	dump(byte*, int32);

extern string	emptystring;
extern int32 debug;

/*
 * very low level c-called
 */
void	FLUSH(void*);
void*	getu(void);
void	throw(int8*);
void	prints(int8*);
void	mcpy(byte*, byte*, uint32);
void*	mal(uint32);
uint32	cmpstring(string, string);
void	initsig(void);
void	traceback(uint8 *pc, uint8 *sp, void* up);
int32	open(byte*, int32);
int32	read(int32, void*, int32);
void	close(int32);
int32	fstat(int32, void*);
struct	SigTab
{
	int32	catch;
	int8	*name;
};

/*
 * low level go -called
 */
void	sys·exit(int32);
void	sys·write(int32, void*, int32);
void	sys·breakpoint(void);
uint8*	sys·mmap(byte*, uint32, int32, int32, int32, uint32);
void	sys·memclr(byte*, uint32);
void*	sys·getcallerpc(void*);
void	sys·sigaction(int64, void*, void*);
void	sys·rt_sigaction(int64, void*, void*, uint64);

/*
 * runtime go-called
 */
void	sys·printbool(bool);
void	sys·printfloat(float64);
void	sys·printint(int64);
void	sys·printstring(string);
void	sys·printpc(void*);
void	sys·printpointer(void*);
void	sys·catstring(string, string, string);
void	sys·cmpstring(string, string, int32);
void	sys·slicestring(string, int32, int32, string);
void	sys·indexstring(string, int32, byte);
void	sys·intstring(int64, string);
void	sys·ifaces2i(Sigi*, Sigs*, Map*, void*);
void	sys·ifacei2i(Sigi*, Map*, void*);
void	sys·ifacei2s(Sigs*, Map*, void*);
void	sys·readfile(string, string, bool);
