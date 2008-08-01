package B

import C "c"
import D "d"

export type T1 C.T1;
export type T2 D.T2;

export var (
	v0 D.T1;
	v1 C.T1;
	v2 *C.F1;
)
