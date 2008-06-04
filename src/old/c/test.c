

/*
 * automatic code generated from
 * test.go in package "test"
 */

// basic types
typedef	unsigned char      _T_U8;
typedef	signed char        _T_I8;
typedef	unsigned short     _T_U16;
typedef	signed short       _T_I16;
typedef	unsigned long      _T_U32;
typedef	signed long        _T_I32;
typedef	unsigned long long _T_U64;
typedef	signed long long   _T_I64;
typedef	float              _T_F32;
typedef	double             _T_F64;
typedef	double             _T_F80;
typedef	int                _T_B;
typedef unsigned char*     _T_P;

#define	offsetof(s, m)     (_T_U32)(&(((s*)0)->m))

typedef	struct{_T_U32 I1; _T_U32 I2; _T_U32 I3;} _T_I;
typedef	struct{_T_U32 O1; _T_U32 O2;} _T_O;

void	test_main(void);
_T_O	test_simple(_T_I);
int	printf(char*, ...);

// external variables

void
test_main(void)
{

	// registers
	register union
	{
		_T_U8  _R_U8;
		_T_I8  _R_I8;
		_T_U16 _R_U16;
		_T_I16 _R_I16;
		_T_U32 _R_U32;
		_T_I32 _R_I32;
		_T_U64 _R_U64;
		_T_I64 _R_I64;
		_T_F32 _R_F32;
		_T_F64 _R_F64;
		_T_F80 _R_F80;
		_T_B   _R_B;
		_T_P   _R_P;
	} _U;

	// local variables
	_T_I32 _V_3; // x
	_T_I32 _V_4; // y

	{
		_T_I I;
		_T_O O;
		I.I1 = 10;
		I.I2 = 20;
		I.I3 = 30;
		O = test_simple(I);
		_V_3 = O.O1;
		_V_4 = O.O2;
	}

	//    1    7 LOAD_I32  NAME a(1) p(3) l(7) x G0 INT32
	_U._R_I32 = _V_3;

	//    2   10 CMP_I32   I15 LITERAL  a(1) l(10) INT32
	if(_U._R_I32 == 15)

	//    3   10 BEQ_I32   4
		goto _L4;

	printf("no 1 %d\n", _V_3);

	//    4    7 LOAD_I32  NAME a(1) p(4) l(7) y G0 INT32
_L4:
	_U._R_I32 = _V_4;

	//    5   11 CMP_I32   I50 LITERAL  a(1) l(11) INT32
	if(_U._R_I32 == 50)

	//    6   11 BEQ_I32   7
		goto _L7;

	printf("no 2 %d\n", _V_4);

	//    7    0 END      
_L7:
	;
}

_T_O
test_simple(_T_I I)
{

	// registers
	register union
	{
		_T_U8  _R_U8;
		_T_I8  _R_I8;
		_T_U16 _R_U16;
		_T_I16 _R_I16;
		_T_U32 _R_U32;
		_T_I32 _R_I32;
		_T_U64 _R_U64;
		_T_I64 _R_I64;
		_T_F32 _R_F32;
		_T_F64 _R_F64;
		_T_F80 _R_F80;
		_T_B   _R_B;
		_T_P   _R_P;
	} _U;

	_T_O O;

	int ia, ib, ic;
	ia = I.I1;
	ib = I.I2;
	ic = I.I3;

	O.O1 = ia+5;
	O.O2 = ib+ic;
	return O;
}

int
main(void)
{
	test_main();
	return 0;
}
