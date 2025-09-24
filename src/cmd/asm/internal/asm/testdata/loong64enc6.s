// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0
	// MOVWP LOREG_64(Rx), Ry
	MOVWP	81985529216486896(R4), R5	// 9e571315dec3b703feac6816de4b000384f8100085000025
	MOVWP	-81985529216486896(R4), R5	// 7ea8ec14de4388031e539717deb73f0384f8100085000025
	MOVWP	R4, 81985529216486896(R5)	// 9e571315dec3b703feac6816de4b0003a5f81000a4000025
	MOVWP	R4, -81985529216486896(R5)	// 7ea8ec14de4388031e539717deb73f03a5f81000a4000025
