// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64asm

// Naming for Go decoder arguments:
//
// - arg_rd: a general purpose register rd encoded in rd[11:7] field
//
// - arg_rs1: a general purpose register rs1 encoded in rs1[19:15] field
//
// - arg_rs2: a general purpose register rs2 encoded in rs2[24:20] field
//
// - arg_rs3: a general purpose register rs3 encoded in rs3[31:27] field
//
// - arg_fd: a floating point register rd encoded in rd[11:7] field
//
// - arg_fs1: a floating point register rs1 encoded in rs1[19:15] field
//
// - arg_fs2: a floating point register rs2 encoded in rs2[24:20] field
//
// - arg_fs3: a floating point register rs3 encoded in rs3[31:27] field
//
// - arg_csr: a control status register encoded in csr[31:20] field
//
// - arg_rs1_mem: source register with offset in load commands
//
// - arg_rs1_store: source register with offset in store commands
//
// - arg_rs1_amo: source register with offset in atomic commands
//
// - arg_pred: predecessor memory ordering information encoded in pred[27:24] field
//             For details, please refer to chapter 2.7 of ISA manual volume 1
//
// - arg_succ: successor memory ordering information encoded in succ[23:20] field
//             For details, please refer to chapter 2.7 of ISA manual volume 1
//
// - arg_zimm: a unsigned immediate encoded in zimm[19:15] field
//
// - arg_imm12: an I-type immediate encoded in imm12[31:20] field
//
// - arg_simm12: a S-type immediate encoded in simm12[31:25|11:7] field
//
// - arg_bimm12: a B-type immediate encoded in bimm12[31:25|11:7] field
//
// - arg_imm20: an U-type immediate encoded in imm20[31:12] field
//
// - arg_jimm20: a J-type immediate encoded in jimm20[31:12] field
//
// - arg_shamt5: a shift amount encoded in shamt5[24:20] field
//
// - arg_shamt6: a shift amount encoded in shamt6[25:20] field
//

type argType uint16

const (
	_ argType = iota
	arg_rd
	arg_rs1
	arg_rs2
	arg_rs3
	arg_fd
	arg_fs1
	arg_fs2
	arg_fs3
	arg_csr

	arg_rs1_amo
	arg_rs1_mem
	arg_rs1_store

	arg_pred
	arg_succ

	arg_zimm
	arg_imm12
	arg_simm12
	arg_bimm12
	arg_imm20
	arg_jimm20
	arg_shamt5
	arg_shamt6

	// RISC-V Compressed Extension Args
	arg_rd_p
	arg_fd_p
	arg_rs1_p
	arg_rd_rs1_p
	arg_fs2_p
	arg_rs2_p
	arg_rd_n0
	arg_rs1_n0
	arg_rd_rs1_n0
	arg_c_rs1_n0
	arg_c_rs2_n0
	arg_c_fs2
	arg_c_rs2
	arg_rd_n2

	arg_c_imm6
	arg_c_nzimm6
	arg_c_nzuimm6
	arg_c_uimm7
	arg_c_uimm8
	arg_c_uimm8sp_s
	arg_c_uimm8sp
	arg_c_uimm9sp_s
	arg_c_uimm9sp
	arg_c_bimm9
	arg_c_nzimm10
	arg_c_nzuimm10
	arg_c_imm12
	arg_c_nzimm18
)
