// Code generated by "stringer -type=SymKind"; DO NOT EDIT.

package ld

import "fmt"

const _SymKind_name = "SxxxSTEXTSELFRXSECTSTYPESSTRINGSGOSTRINGSGOFUNCSGCBITSSRODATASFUNCTABSELFROSECTSMACHOPLTSTYPERELROSSTRINGRELROSGOSTRINGRELROSGOFUNCRELROSGCBITSRELROSRODATARELROSFUNCTABRELROSTYPELINKSITABLINKSSYMTABSPCLNTABSELFSECTSMACHOSMACHOGOTSWINDOWSSELFGOTSNOPTRDATASINITARRSDATASBSSSNOPTRBSSSTLSBSSSXREFSMACHOSYMSTRSMACHOSYMTABSMACHOINDIRECTPLTSMACHOINDIRECTGOTSFILESFILEPATHSCONSTSDYNIMPORTSHOSTOBJSDWARFSECTSDWARFINFO"

var _SymKind_index = [...]uint16{0, 4, 9, 19, 24, 31, 40, 47, 54, 61, 69, 79, 88, 98, 110, 124, 136, 148, 160, 173, 182, 191, 198, 206, 214, 220, 229, 237, 244, 254, 262, 267, 271, 280, 287, 292, 304, 316, 333, 350, 355, 364, 370, 380, 388, 398, 408}

func (i SymKind) String() string {
	if i < 0 || i >= SymKind(len(_SymKind_index)-1) {
		return fmt.Sprintf("SymKind(%d)", i)
	}
	return _SymKind_name[_SymKind_index[i]:_SymKind_index[i+1]]
}
