// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"cmd/internal/obj"
)

// This file contains utility functions for use when
// assembling vector instructions.

// vop returns the opcode, element size and condition
// setting for the given (possibly extended) mnemonic.
func vop(as obj.As) (opcode, es, cs uint32) {
	switch as {
	default:
		return 0, 0, 0
	case AVA:
		return op_VA, 0, 0
	case AVAB:
		return op_VA, 0, 0
	case AVAH:
		return op_VA, 1, 0
	case AVAF:
		return op_VA, 2, 0
	case AVAG:
		return op_VA, 3, 0
	case AVAQ:
		return op_VA, 4, 0
	case AVACC:
		return op_VACC, 0, 0
	case AVACCB:
		return op_VACC, 0, 0
	case AVACCH:
		return op_VACC, 1, 0
	case AVACCF:
		return op_VACC, 2, 0
	case AVACCG:
		return op_VACC, 3, 0
	case AVACCQ:
		return op_VACC, 4, 0
	case AVAC:
		return op_VAC, 0, 0
	case AVACQ:
		return op_VAC, 4, 0
	case AVMSLG, AVMSLEG, AVMSLOG, AVMSLEOG:
		return op_VMSL, 3, 0
	case AVACCC:
		return op_VACCC, 0, 0
	case AVACCCQ:
		return op_VACCC, 4, 0
	case AVN:
		return op_VN, 0, 0
	case AVNC:
		return op_VNC, 0, 0
	case AVAVG:
		return op_VAVG, 0, 0
	case AVAVGB:
		return op_VAVG, 0, 0
	case AVAVGH:
		return op_VAVG, 1, 0
	case AVAVGF:
		return op_VAVG, 2, 0
	case AVAVGG:
		return op_VAVG, 3, 0
	case AVAVGL:
		return op_VAVGL, 0, 0
	case AVAVGLB:
		return op_VAVGL, 0, 0
	case AVAVGLH:
		return op_VAVGL, 1, 0
	case AVAVGLF:
		return op_VAVGL, 2, 0
	case AVAVGLG:
		return op_VAVGL, 3, 0
	case AVCKSM:
		return op_VCKSM, 0, 0
	case AVCEQ:
		return op_VCEQ, 0, 0
	case AVCEQB:
		return op_VCEQ, 0, 0
	case AVCEQH:
		return op_VCEQ, 1, 0
	case AVCEQF:
		return op_VCEQ, 2, 0
	case AVCEQG:
		return op_VCEQ, 3, 0
	case AVCEQBS:
		return op_VCEQ, 0, 1
	case AVCEQHS:
		return op_VCEQ, 1, 1
	case AVCEQFS:
		return op_VCEQ, 2, 1
	case AVCEQGS:
		return op_VCEQ, 3, 1
	case AVCH:
		return op_VCH, 0, 0
	case AVCHB:
		return op_VCH, 0, 0
	case AVCHH:
		return op_VCH, 1, 0
	case AVCHF:
		return op_VCH, 2, 0
	case AVCHG:
		return op_VCH, 3, 0
	case AVCHBS:
		return op_VCH, 0, 1
	case AVCHHS:
		return op_VCH, 1, 1
	case AVCHFS:
		return op_VCH, 2, 1
	case AVCHGS:
		return op_VCH, 3, 1
	case AVCHL:
		return op_VCHL, 0, 0
	case AVCHLB:
		return op_VCHL, 0, 0
	case AVCHLH:
		return op_VCHL, 1, 0
	case AVCHLF:
		return op_VCHL, 2, 0
	case AVCHLG:
		return op_VCHL, 3, 0
	case AVCHLBS:
		return op_VCHL, 0, 1
	case AVCHLHS:
		return op_VCHL, 1, 1
	case AVCHLFS:
		return op_VCHL, 2, 1
	case AVCHLGS:
		return op_VCHL, 3, 1
	case AVCLZ:
		return op_VCLZ, 0, 0
	case AVCLZB:
		return op_VCLZ, 0, 0
	case AVCLZH:
		return op_VCLZ, 1, 0
	case AVCLZF:
		return op_VCLZ, 2, 0
	case AVCLZG:
		return op_VCLZ, 3, 0
	case AVCTZ:
		return op_VCTZ, 0, 0
	case AVCTZB:
		return op_VCTZ, 0, 0
	case AVCTZH:
		return op_VCTZ, 1, 0
	case AVCTZF:
		return op_VCTZ, 2, 0
	case AVCTZG:
		return op_VCTZ, 3, 0
	case AVEC:
		return op_VEC, 0, 0
	case AVECB:
		return op_VEC, 0, 0
	case AVECH:
		return op_VEC, 1, 0
	case AVECF:
		return op_VEC, 2, 0
	case AVECG:
		return op_VEC, 3, 0
	case AVECL:
		return op_VECL, 0, 0
	case AVECLB:
		return op_VECL, 0, 0
	case AVECLH:
		return op_VECL, 1, 0
	case AVECLF:
		return op_VECL, 2, 0
	case AVECLG:
		return op_VECL, 3, 0
	case AVERIM:
		return op_VERIM, 0, 0
	case AVERIMB:
		return op_VERIM, 0, 0
	case AVERIMH:
		return op_VERIM, 1, 0
	case AVERIMF:
		return op_VERIM, 2, 0
	case AVERIMG:
		return op_VERIM, 3, 0
	case AVERLL:
		return op_VERLL, 0, 0
	case AVERLLB:
		return op_VERLL, 0, 0
	case AVERLLH:
		return op_VERLL, 1, 0
	case AVERLLF:
		return op_VERLL, 2, 0
	case AVERLLG:
		return op_VERLL, 3, 0
	case AVERLLV:
		return op_VERLLV, 0, 0
	case AVERLLVB:
		return op_VERLLV, 0, 0
	case AVERLLVH:
		return op_VERLLV, 1, 0
	case AVERLLVF:
		return op_VERLLV, 2, 0
	case AVERLLVG:
		return op_VERLLV, 3, 0
	case AVESLV:
		return op_VESLV, 0, 0
	case AVESLVB:
		return op_VESLV, 0, 0
	case AVESLVH:
		return op_VESLV, 1, 0
	case AVESLVF:
		return op_VESLV, 2, 0
	case AVESLVG:
		return op_VESLV, 3, 0
	case AVESL:
		return op_VESL, 0, 0
	case AVESLB:
		return op_VESL, 0, 0
	case AVESLH:
		return op_VESL, 1, 0
	case AVESLF:
		return op_VESL, 2, 0
	case AVESLG:
		return op_VESL, 3, 0
	case AVESRA:
		return op_VESRA, 0, 0
	case AVESRAB:
		return op_VESRA, 0, 0
	case AVESRAH:
		return op_VESRA, 1, 0
	case AVESRAF:
		return op_VESRA, 2, 0
	case AVESRAG:
		return op_VESRA, 3, 0
	case AVESRAV:
		return op_VESRAV, 0, 0
	case AVESRAVB:
		return op_VESRAV, 0, 0
	case AVESRAVH:
		return op_VESRAV, 1, 0
	case AVESRAVF:
		return op_VESRAV, 2, 0
	case AVESRAVG:
		return op_VESRAV, 3, 0
	case AVESRL:
		return op_VESRL, 0, 0
	case AVESRLB:
		return op_VESRL, 0, 0
	case AVESRLH:
		return op_VESRL, 1, 0
	case AVESRLF:
		return op_VESRL, 2, 0
	case AVESRLG:
		return op_VESRL, 3, 0
	case AVESRLV:
		return op_VESRLV, 0, 0
	case AVESRLVB:
		return op_VESRLV, 0, 0
	case AVESRLVH:
		return op_VESRLV, 1, 0
	case AVESRLVF:
		return op_VESRLV, 2, 0
	case AVESRLVG:
		return op_VESRLV, 3, 0
	case AVX:
		return op_VX, 0, 0
	case AVFAE:
		return op_VFAE, 0, 0
	case AVFAEB:
		return op_VFAE, 0, 0
	case AVFAEH:
		return op_VFAE, 1, 0
	case AVFAEF:
		return op_VFAE, 2, 0
	case AVFAEBS:
		return op_VFAE, 0, 1
	case AVFAEHS:
		return op_VFAE, 1, 1
	case AVFAEFS:
		return op_VFAE, 2, 1
	case AVFAEZB:
		return op_VFAE, 0, 2
	case AVFAEZH:
		return op_VFAE, 1, 2
	case AVFAEZF:
		return op_VFAE, 2, 2
	case AVFAEZBS:
		return op_VFAE, 0, 3
	case AVFAEZHS:
		return op_VFAE, 1, 3
	case AVFAEZFS:
		return op_VFAE, 2, 3
	case AVFEE:
		return op_VFEE, 0, 0
	case AVFEEB:
		return op_VFEE, 0, 0
	case AVFEEH:
		return op_VFEE, 1, 0
	case AVFEEF:
		return op_VFEE, 2, 0
	case AVFEEBS:
		return op_VFEE, 0, 1
	case AVFEEHS:
		return op_VFEE, 1, 1
	case AVFEEFS:
		return op_VFEE, 2, 1
	case AVFEEZB:
		return op_VFEE, 0, 2
	case AVFEEZH:
		return op_VFEE, 1, 2
	case AVFEEZF:
		return op_VFEE, 2, 2
	case AVFEEZBS:
		return op_VFEE, 0, 3
	case AVFEEZHS:
		return op_VFEE, 1, 3
	case AVFEEZFS:
		return op_VFEE, 2, 3
	case AVFENE:
		return op_VFENE, 0, 0
	case AVFENEB:
		return op_VFENE, 0, 0
	case AVFENEH:
		return op_VFENE, 1, 0
	case AVFENEF:
		return op_VFENE, 2, 0
	case AVFENEBS:
		return op_VFENE, 0, 1
	case AVFENEHS:
		return op_VFENE, 1, 1
	case AVFENEFS:
		return op_VFENE, 2, 1
	case AVFENEZB:
		return op_VFENE, 0, 2
	case AVFENEZH:
		return op_VFENE, 1, 2
	case AVFENEZF:
		return op_VFENE, 2, 2
	case AVFENEZBS:
		return op_VFENE, 0, 3
	case AVFENEZHS:
		return op_VFENE, 1, 3
	case AVFENEZFS:
		return op_VFENE, 2, 3
	case AVFA:
		return op_VFA, 0, 0
	case AVFADB:
		return op_VFA, 3, 0
	case AWFADB:
		return op_VFA, 3, 0
	case AWFK:
		return op_WFK, 0, 0
	case AWFKDB:
		return op_WFK, 3, 0
	case AVFCE:
		return op_VFCE, 0, 0
	case AVFCEDB:
		return op_VFCE, 3, 0
	case AVFCEDBS:
		return op_VFCE, 3, 1
	case AWFCEDB:
		return op_VFCE, 3, 0
	case AWFCEDBS:
		return op_VFCE, 3, 1
	case AVFCH:
		return op_VFCH, 0, 0
	case AVFCHDB:
		return op_VFCH, 3, 0
	case AVFCHDBS:
		return op_VFCH, 3, 1
	case AWFCHDB:
		return op_VFCH, 3, 0
	case AWFCHDBS:
		return op_VFCH, 3, 1
	case AVFCHE:
		return op_VFCHE, 0, 0
	case AVFCHEDB:
		return op_VFCHE, 3, 0
	case AVFCHEDBS:
		return op_VFCHE, 3, 1
	case AWFCHEDB:
		return op_VFCHE, 3, 0
	case AWFCHEDBS:
		return op_VFCHE, 3, 1
	case AWFC:
		return op_WFC, 0, 0
	case AWFCDB:
		return op_WFC, 3, 0
	case AVCDG:
		return op_VCDG, 0, 0
	case AVCDGB:
		return op_VCDG, 3, 0
	case AWCDGB:
		return op_VCDG, 3, 0
	case AVCDLG:
		return op_VCDLG, 0, 0
	case AVCDLGB:
		return op_VCDLG, 3, 0
	case AWCDLGB:
		return op_VCDLG, 3, 0
	case AVCGD:
		return op_VCGD, 0, 0
	case AVCGDB:
		return op_VCGD, 3, 0
	case AWCGDB:
		return op_VCGD, 3, 0
	case AVCLGD:
		return op_VCLGD, 0, 0
	case AVCLGDB:
		return op_VCLGD, 3, 0
	case AWCLGDB:
		return op_VCLGD, 3, 0
	case AVFD:
		return op_VFD, 0, 0
	case AVFDDB:
		return op_VFD, 3, 0
	case AWFDDB:
		return op_VFD, 3, 0
	case AVLDE:
		return op_VLDE, 0, 0
	case AVLDEB:
		return op_VLDE, 2, 0
	case AWLDEB:
		return op_VLDE, 2, 0
	case AVLED:
		return op_VLED, 0, 0
	case AVLEDB:
		return op_VLED, 3, 0
	case AWLEDB:
		return op_VLED, 3, 0
	case AVFM:
		return op_VFM, 0, 0
	case AVFMDB:
		return op_VFM, 3, 0
	case AWFMDB:
		return op_VFM, 3, 0
	case AVFMA:
		return op_VFMA, 0, 0
	case AVFMADB:
		return op_VFMA, 3, 0
	case AWFMADB:
		return op_VFMA, 3, 0
	case AVFMS:
		return op_VFMS, 0, 0
	case AVFMSDB:
		return op_VFMS, 3, 0
	case AWFMSDB:
		return op_VFMS, 3, 0
	case AVFPSO:
		return op_VFPSO, 0, 0
	case AVFPSODB:
		return op_VFPSO, 3, 0
	case AWFPSODB:
		return op_VFPSO, 3, 0
	case AVFLCDB:
		return op_VFPSO, 3, 0
	case AWFLCDB:
		return op_VFPSO, 3, 0
	case AVFLNDB:
		return op_VFPSO, 3, 1
	case AWFLNDB:
		return op_VFPSO, 3, 1
	case AVFLPDB:
		return op_VFPSO, 3, 2
	case AWFLPDB:
		return op_VFPSO, 3, 2
	case AVFSQ:
		return op_VFSQ, 0, 0
	case AVFSQDB:
		return op_VFSQ, 3, 0
	case AWFSQDB:
		return op_VFSQ, 3, 0
	case AVFS:
		return op_VFS, 0, 0
	case AVFSDB:
		return op_VFS, 3, 0
	case AWFSDB:
		return op_VFS, 3, 0
	case AVFTCI:
		return op_VFTCI, 0, 0
	case AVFTCIDB:
		return op_VFTCI, 3, 0
	case AWFTCIDB:
		return op_VFTCI, 3, 0
	case AVGFM:
		return op_VGFM, 0, 0
	case AVGFMB:
		return op_VGFM, 0, 0
	case AVGFMH:
		return op_VGFM, 1, 0
	case AVGFMF:
		return op_VGFM, 2, 0
	case AVGFMG:
		return op_VGFM, 3, 0
	case AVGFMA:
		return op_VGFMA, 0, 0
	case AVGFMAB:
		return op_VGFMA, 0, 0
	case AVGFMAH:
		return op_VGFMA, 1, 0
	case AVGFMAF:
		return op_VGFMA, 2, 0
	case AVGFMAG:
		return op_VGFMA, 3, 0
	case AVGEF:
		return op_VGEF, 0, 0
	case AVGEG:
		return op_VGEG, 0, 0
	case AVGBM:
		return op_VGBM, 0, 0
	case AVZERO:
		return op_VGBM, 0, 0
	case AVONE:
		return op_VGBM, 0, 0
	case AVGM:
		return op_VGM, 0, 0
	case AVGMB:
		return op_VGM, 0, 0
	case AVGMH:
		return op_VGM, 1, 0
	case AVGMF:
		return op_VGM, 2, 0
	case AVGMG:
		return op_VGM, 3, 0
	case AVISTR:
		return op_VISTR, 0, 0
	case AVISTRB:
		return op_VISTR, 0, 0
	case AVISTRH:
		return op_VISTR, 1, 0
	case AVISTRF:
		return op_VISTR, 2, 0
	case AVISTRBS:
		return op_VISTR, 0, 1
	case AVISTRHS:
		return op_VISTR, 1, 1
	case AVISTRFS:
		return op_VISTR, 2, 1
	case AVL:
		return op_VL, 0, 0
	case AVLR:
		return op_VLR, 0, 0
	case AVLREP:
		return op_VLREP, 0, 0
	case AVLREPB:
		return op_VLREP, 0, 0
	case AVLREPH:
		return op_VLREP, 1, 0
	case AVLREPF:
		return op_VLREP, 2, 0
	case AVLREPG:
		return op_VLREP, 3, 0
	case AVLC:
		return op_VLC, 0, 0
	case AVLCB:
		return op_VLC, 0, 0
	case AVLCH:
		return op_VLC, 1, 0
	case AVLCF:
		return op_VLC, 2, 0
	case AVLCG:
		return op_VLC, 3, 0
	case AVLEH:
		return op_VLEH, 0, 0
	case AVLEF:
		return op_VLEF, 0, 0
	case AVLEG:
		return op_VLEG, 0, 0
	case AVLEB:
		return op_VLEB, 0, 0
	case AVLEIH:
		return op_VLEIH, 0, 0
	case AVLEIF:
		return op_VLEIF, 0, 0
	case AVLEIG:
		return op_VLEIG, 0, 0
	case AVLEIB:
		return op_VLEIB, 0, 0
	case AVFI:
		return op_VFI, 0, 0
	case AVFIDB:
		return op_VFI, 3, 0
	case AWFIDB:
		return op_VFI, 3, 0
	case AVLGV:
		return op_VLGV, 0, 0
	case AVLGVB:
		return op_VLGV, 0, 0
	case AVLGVH:
		return op_VLGV, 1, 0
	case AVLGVF:
		return op_VLGV, 2, 0
	case AVLGVG:
		return op_VLGV, 3, 0
	case AVLLEZ:
		return op_VLLEZ, 0, 0
	case AVLLEZB:
		return op_VLLEZ, 0, 0
	case AVLLEZH:
		return op_VLLEZ, 1, 0
	case AVLLEZF:
		return op_VLLEZ, 2, 0
	case AVLLEZG:
		return op_VLLEZ, 3, 0
	case AVLM:
		return op_VLM, 0, 0
	case AVLP:
		return op_VLP, 0, 0
	case AVLPB:
		return op_VLP, 0, 0
	case AVLPH:
		return op_VLP, 1, 0
	case AVLPF:
		return op_VLP, 2, 0
	case AVLPG:
		return op_VLP, 3, 0
	case AVLBB:
		return op_VLBB, 0, 0
	case AVLVG:
		return op_VLVG, 0, 0
	case AVLVGB:
		return op_VLVG, 0, 0
	case AVLVGH:
		return op_VLVG, 1, 0
	case AVLVGF:
		return op_VLVG, 2, 0
	case AVLVGG:
		return op_VLVG, 3, 0
	case AVLVGP:
		return op_VLVGP, 0, 0
	case AVLL:
		return op_VLL, 0, 0
	case AVMX:
		return op_VMX, 0, 0
	case AVMXB:
		return op_VMX, 0, 0
	case AVMXH:
		return op_VMX, 1, 0
	case AVMXF:
		return op_VMX, 2, 0
	case AVMXG:
		return op_VMX, 3, 0
	case AVMXL:
		return op_VMXL, 0, 0
	case AVMXLB:
		return op_VMXL, 0, 0
	case AVMXLH:
		return op_VMXL, 1, 0
	case AVMXLF:
		return op_VMXL, 2, 0
	case AVMXLG:
		return op_VMXL, 3, 0
	case AVMRH:
		return op_VMRH, 0, 0
	case AVMRHB:
		return op_VMRH, 0, 0
	case AVMRHH:
		return op_VMRH, 1, 0
	case AVMRHF:
		return op_VMRH, 2, 0
	case AVMRHG:
		return op_VMRH, 3, 0
	case AVMRL:
		return op_VMRL, 0, 0
	case AVMRLB:
		return op_VMRL, 0, 0
	case AVMRLH:
		return op_VMRL, 1, 0
	case AVMRLF:
		return op_VMRL, 2, 0
	case AVMRLG:
		return op_VMRL, 3, 0
	case AVMN:
		return op_VMN, 0, 0
	case AVMNB:
		return op_VMN, 0, 0
	case AVMNH:
		return op_VMN, 1, 0
	case AVMNF:
		return op_VMN, 2, 0
	case AVMNG:
		return op_VMN, 3, 0
	case AVMNL:
		return op_VMNL, 0, 0
	case AVMNLB:
		return op_VMNL, 0, 0
	case AVMNLH:
		return op_VMNL, 1, 0
	case AVMNLF:
		return op_VMNL, 2, 0
	case AVMNLG:
		return op_VMNL, 3, 0
	case AVMAE:
		return op_VMAE, 0, 0
	case AVMAEB:
		return op_VMAE, 0, 0
	case AVMAEH:
		return op_VMAE, 1, 0
	case AVMAEF:
		return op_VMAE, 2, 0
	case AVMAH:
		return op_VMAH, 0, 0
	case AVMAHB:
		return op_VMAH, 0, 0
	case AVMAHH:
		return op_VMAH, 1, 0
	case AVMAHF:
		return op_VMAH, 2, 0
	case AVMALE:
		return op_VMALE, 0, 0
	case AVMALEB:
		return op_VMALE, 0, 0
	case AVMALEH:
		return op_VMALE, 1, 0
	case AVMALEF:
		return op_VMALE, 2, 0
	case AVMALH:
		return op_VMALH, 0, 0
	case AVMALHB:
		return op_VMALH, 0, 0
	case AVMALHH:
		return op_VMALH, 1, 0
	case AVMALHF:
		return op_VMALH, 2, 0
	case AVMALO:
		return op_VMALO, 0, 0
	case AVMALOB:
		return op_VMALO, 0, 0
	case AVMALOH:
		return op_VMALO, 1, 0
	case AVMALOF:
		return op_VMALO, 2, 0
	case AVMAL:
		return op_VMAL, 0, 0
	case AVMALB:
		return op_VMAL, 0, 0
	case AVMALHW:
		return op_VMAL, 1, 0
	case AVMALF:
		return op_VMAL, 2, 0
	case AVMAO:
		return op_VMAO, 0, 0
	case AVMAOB:
		return op_VMAO, 0, 0
	case AVMAOH:
		return op_VMAO, 1, 0
	case AVMAOF:
		return op_VMAO, 2, 0
	case AVME:
		return op_VME, 0, 0
	case AVMEB:
		return op_VME, 0, 0
	case AVMEH:
		return op_VME, 1, 0
	case AVMEF:
		return op_VME, 2, 0
	case AVMH:
		return op_VMH, 0, 0
	case AVMHB:
		return op_VMH, 0, 0
	case AVMHH:
		return op_VMH, 1, 0
	case AVMHF:
		return op_VMH, 2, 0
	case AVMLE:
		return op_VMLE, 0, 0
	case AVMLEB:
		return op_VMLE, 0, 0
	case AVMLEH:
		return op_VMLE, 1, 0
	case AVMLEF:
		return op_VMLE, 2, 0
	case AVMLH:
		return op_VMLH, 0, 0
	case AVMLHB:
		return op_VMLH, 0, 0
	case AVMLHH:
		return op_VMLH, 1, 0
	case AVMLHF:
		return op_VMLH, 2, 0
	case AVMLO:
		return op_VMLO, 0, 0
	case AVMLOB:
		return op_VMLO, 0, 0
	case AVMLOH:
		return op_VMLO, 1, 0
	case AVMLOF:
		return op_VMLO, 2, 0
	case AVML:
		return op_VML, 0, 0
	case AVMLB:
		return op_VML, 0, 0
	case AVMLHW:
		return op_VML, 1, 0
	case AVMLF:
		return op_VML, 2, 0
	case AVMO:
		return op_VMO, 0, 0
	case AVMOB:
		return op_VMO, 0, 0
	case AVMOH:
		return op_VMO, 1, 0
	case AVMOF:
		return op_VMO, 2, 0
	case AVNO:
		return op_VNO, 0, 0
	case AVNOT:
		return op_VNO, 0, 0
	case AVO:
		return op_VO, 0, 0
	case AVPK:
		return op_VPK, 0, 0
	case AVPKH:
		return op_VPK, 1, 0
	case AVPKF:
		return op_VPK, 2, 0
	case AVPKG:
		return op_VPK, 3, 0
	case AVPKLS:
		return op_VPKLS, 0, 0
	case AVPKLSH:
		return op_VPKLS, 1, 0
	case AVPKLSF:
		return op_VPKLS, 2, 0
	case AVPKLSG:
		return op_VPKLS, 3, 0
	case AVPKLSHS:
		return op_VPKLS, 1, 1
	case AVPKLSFS:
		return op_VPKLS, 2, 1
	case AVPKLSGS:
		return op_VPKLS, 3, 1
	case AVPKS:
		return op_VPKS, 0, 0
	case AVPKSH:
		return op_VPKS, 1, 0
	case AVPKSF:
		return op_VPKS, 2, 0
	case AVPKSG:
		return op_VPKS, 3, 0
	case AVPKSHS:
		return op_VPKS, 1, 1
	case AVPKSFS:
		return op_VPKS, 2, 1
	case AVPKSGS:
		return op_VPKS, 3, 1
	case AVPERM:
		return op_VPERM, 0, 0
	case AVPDI:
		return op_VPDI, 0, 0
	case AVPOPCT:
		return op_VPOPCT, 0, 0
	case AVREP:
		return op_VREP, 0, 0
	case AVREPB:
		return op_VREP, 0, 0
	case AVREPH:
		return op_VREP, 1, 0
	case AVREPF:
		return op_VREP, 2, 0
	case AVREPG:
		return op_VREP, 3, 0
	case AVREPI:
		return op_VREPI, 0, 0
	case AVREPIB:
		return op_VREPI, 0, 0
	case AVREPIH:
		return op_VREPI, 1, 0
	case AVREPIF:
		return op_VREPI, 2, 0
	case AVREPIG:
		return op_VREPI, 3, 0
	case AVSCEF:
		return op_VSCEF, 0, 0
	case AVSCEG:
		return op_VSCEG, 0, 0
	case AVSEL:
		return op_VSEL, 0, 0
	case AVSL:
		return op_VSL, 0, 0
	case AVSLB:
		return op_VSLB, 0, 0
	case AVSLDB:
		return op_VSLDB, 0, 0
	case AVSRA:
		return op_VSRA, 0, 0
	case AVSRAB:
		return op_VSRAB, 0, 0
	case AVSRL:
		return op_VSRL, 0, 0
	case AVSRLB:
		return op_VSRLB, 0, 0
	case AVSEG:
		return op_VSEG, 0, 0
	case AVSEGB:
		return op_VSEG, 0, 0
	case AVSEGH:
		return op_VSEG, 1, 0
	case AVSEGF:
		return op_VSEG, 2, 0
	case AVST:
		return op_VST, 0, 0
	case AVSTEH:
		return op_VSTEH, 0, 0
	case AVSTEF:
		return op_VSTEF, 0, 0
	case AVSTEG:
		return op_VSTEG, 0, 0
	case AVSTEB:
		return op_VSTEB, 0, 0
	case AVSTM:
		return op_VSTM, 0, 0
	case AVSTL:
		return op_VSTL, 0, 0
	case AVSTRC:
		return op_VSTRC, 0, 0
	case AVSTRCB:
		return op_VSTRC, 0, 0
	case AVSTRCH:
		return op_VSTRC, 1, 0
	case AVSTRCF:
		return op_VSTRC, 2, 0
	case AVSTRCBS:
		return op_VSTRC, 0, 1
	case AVSTRCHS:
		return op_VSTRC, 1, 1
	case AVSTRCFS:
		return op_VSTRC, 2, 1
	case AVSTRCZB:
		return op_VSTRC, 0, 2
	case AVSTRCZH:
		return op_VSTRC, 1, 2
	case AVSTRCZF:
		return op_VSTRC, 2, 2
	case AVSTRCZBS:
		return op_VSTRC, 0, 3
	case AVSTRCZHS:
		return op_VSTRC, 1, 3
	case AVSTRCZFS:
		return op_VSTRC, 2, 3
	case AVS:
		return op_VS, 0, 0
	case AVSB:
		return op_VS, 0, 0
	case AVSH:
		return op_VS, 1, 0
	case AVSF:
		return op_VS, 2, 0
	case AVSG:
		return op_VS, 3, 0
	case AVSQ:
		return op_VS, 4, 0
	case AVSCBI:
		return op_VSCBI, 0, 0
	case AVSCBIB:
		return op_VSCBI, 0, 0
	case AVSCBIH:
		return op_VSCBI, 1, 0
	case AVSCBIF:
		return op_VSCBI, 2, 0
	case AVSCBIG:
		return op_VSCBI, 3, 0
	case AVSCBIQ:
		return op_VSCBI, 4, 0
	case AVSBCBI:
		return op_VSBCBI, 0, 0
	case AVSBCBIQ:
		return op_VSBCBI, 4, 0
	case AVSBI:
		return op_VSBI, 0, 0
	case AVSBIQ:
		return op_VSBI, 4, 0
	case AVSUMG:
		return op_VSUMG, 0, 0
	case AVSUMGH:
		return op_VSUMG, 1, 0
	case AVSUMGF:
		return op_VSUMG, 2, 0
	case AVSUMQ:
		return op_VSUMQ, 0, 0
	case AVSUMQF:
		return op_VSUMQ, 2, 0
	case AVSUMQG:
		return op_VSUMQ, 3, 0
	case AVSUM:
		return op_VSUM, 0, 0
	case AVSUMB:
		return op_VSUM, 0, 0
	case AVSUMH:
		return op_VSUM, 1, 0
	case AVTM:
		return op_VTM, 0, 0
	case AVUPH:
		return op_VUPH, 0, 0
	case AVUPHB:
		return op_VUPH, 0, 0
	case AVUPHH:
		return op_VUPH, 1, 0
	case AVUPHF:
		return op_VUPH, 2, 0
	case AVUPLH:
		return op_VUPLH, 0, 0
	case AVUPLHB:
		return op_VUPLH, 0, 0
	case AVUPLHH:
		return op_VUPLH, 1, 0
	case AVUPLHF:
		return op_VUPLH, 2, 0
	case AVUPLL:
		return op_VUPLL, 0, 0
	case AVUPLLB:
		return op_VUPLL, 0, 0
	case AVUPLLH:
		return op_VUPLL, 1, 0
	case AVUPLLF:
		return op_VUPLL, 2, 0
	case AVUPL:
		return op_VUPL, 0, 0
	case AVUPLB:
		return op_VUPL, 0, 0
	case AVUPLHW:
		return op_VUPL, 1, 0
	case AVUPLF:
		return op_VUPL, 2, 0
	}
}

// singleElementMask returns the single element mask bits required for the
// given instruction.
func singleElementMask(as obj.As) uint32 {
	switch as {
	case AWFADB,
		AWFK,
		AWFKDB,
		AWFCEDB,
		AWFCEDBS,
		AWFCHDB,
		AWFCHDBS,
		AWFCHEDB,
		AWFCHEDBS,
		AWFC,
		AWFCDB,
		AWCDGB,
		AWCDLGB,
		AWCGDB,
		AWCLGDB,
		AWFDDB,
		AWLDEB,
		AWLEDB,
		AWFMDB,
		AWFMADB,
		AWFMSDB,
		AWFPSODB,
		AWFLCDB,
		AWFLNDB,
		AWFLPDB,
		AWFSQDB,
		AWFSDB,
		AWFTCIDB,
		AWFIDB:
		return 8
	case AVMSLEG:
		return 8
	case AVMSLOG:
		return 4
	case AVMSLEOG:
		return 12
	}
	return 0
}
