// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

func initS390Xbase() {
	// get the facilities list
	facilities := stfle()

	// mandatory
	S390X.HasZARCH = facilities.Has(zarch)
	S390X.HasSTFLE = facilities.Has(stflef)
	S390X.HasLDISP = facilities.Has(ldisp)
	S390X.HasEIMM = facilities.Has(eimm)

	// optional
	S390X.HasETF3EH = facilities.Has(etf3eh)
	S390X.HasDFP = facilities.Has(dfp)
	S390X.HasMSA = facilities.Has(msa)
	S390X.HasVX = facilities.Has(vx)
	if S390X.HasVX {
		S390X.HasVXE = facilities.Has(vxe)
	}
}
