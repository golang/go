// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 386

// func bug453a() float64
TEXT ·bug453a(SB),7,$0
        FLD1
        FLD1
        FADDD F1,F0

        // This should subtract F0 (2) from F1 (1) and put -1 in F1.
        FSUBRD F0,F1

        FMOVDP  F0,r+0(FP)
        FMOVDP  F0,r+0(FP)
        RET

// func bug453b() float64
TEXT ·bug453b(SB),7,$0
        FLD1
        FLD1
        FADDD F1,F0

        // This should subtract F1 (1) from F0 (2) and put 1 in F1.
        FSUBD F0,F1

        FMOVDP  F0,r+0(FP)
        FMOVDP  F0,r+0(FP)
        RET
