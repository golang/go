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
 * very low level c-called
 */
void	FLUSH(void*);
void	throw(int8*);
void	prints(int8*);
void	mcpy(byte*, byte*, uint32);
void*	mal(uint32);
uint32	cmpstring(string, string);
void	initsig(void);
void	traceback(uint8 *pc, uint8 *sp);
struct SigTab {
	int32	catch;
	int8	*name;
};

/*
 * low level go -called
 */
void	sys_exit(int32);
void	sys_write(int32, void*, int32);
void	sys_breakpoint(void);
uint8*	sys_mmap(byte*, uint32, int32, int32, int32, uint32);
void	sys_memclr(byte*, uint32);
void*	sys_getcallerpc(void*);
void	sys_sigaction(int64, void*, void*);
void	sys_rt_sigaction(int64, void*, void*, uint64);

/*
 * runtime go-called
 */
void	sys_printbool(bool);
void	sys_printfloat(float64);
void	sys_printint(int64);
void	sys_printstring(string);
void	sys_printpointer(void*);
void	sys_catstring(string, string, string);
void	sys_cmpstring(string, string, int32);
void	sys_slicestring(string, int32, int32, string);
void	sys_indexstring(string, int32, byte);
void	sys_intstring(int64, string);
void	sys_ifaces2i(Sigi*, Sigs*, Map*, void*);
void	sys_ifacei2i(Sigi*, Map*, void*);
void	sys_ifacei2s(Sigs*, Map*, void*);
