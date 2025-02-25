// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dwarf

// Cut, pasted, tr-and-awk'ed from tables in
// http://dwarfstd.org/doc/Dwarf3.pdf

// Table 18
const (
	DW_TAG_array_type               = 0x01
	DW_TAG_class_type               = 0x02
	DW_TAG_entry_point              = 0x03
	DW_TAG_enumeration_type         = 0x04
	DW_TAG_formal_parameter         = 0x05
	DW_TAG_imported_declaration     = 0x08
	DW_TAG_label                    = 0x0a
	DW_TAG_lexical_block            = 0x0b
	DW_TAG_member                   = 0x0d
	DW_TAG_pointer_type             = 0x0f
	DW_TAG_reference_type           = 0x10
	DW_TAG_compile_unit             = 0x11
	DW_TAG_string_type              = 0x12
	DW_TAG_structure_type           = 0x13
	DW_TAG_subroutine_type          = 0x15
	DW_TAG_typedef                  = 0x16
	DW_TAG_union_type               = 0x17
	DW_TAG_unspecified_parameters   = 0x18
	DW_TAG_variant                  = 0x19
	DW_TAG_common_block             = 0x1a
	DW_TAG_common_inclusion         = 0x1b
	DW_TAG_inheritance              = 0x1c
	DW_TAG_inlined_subroutine       = 0x1d
	DW_TAG_module                   = 0x1e
	DW_TAG_ptr_to_member_type       = 0x1f
	DW_TAG_set_type                 = 0x20
	DW_TAG_subrange_type            = 0x21
	DW_TAG_with_stmt                = 0x22
	DW_TAG_access_declaration       = 0x23
	DW_TAG_base_type                = 0x24
	DW_TAG_catch_block              = 0x25
	DW_TAG_const_type               = 0x26
	DW_TAG_constant                 = 0x27
	DW_TAG_enumerator               = 0x28
	DW_TAG_file_type                = 0x29
	DW_TAG_friend                   = 0x2a
	DW_TAG_namelist                 = 0x2b
	DW_TAG_namelist_item            = 0x2c
	DW_TAG_packed_type              = 0x2d
	DW_TAG_subprogram               = 0x2e
	DW_TAG_template_type_parameter  = 0x2f
	DW_TAG_template_value_parameter = 0x30
	DW_TAG_thrown_type              = 0x31
	DW_TAG_try_block                = 0x32
	DW_TAG_variant_part             = 0x33
	DW_TAG_variable                 = 0x34
	DW_TAG_volatile_type            = 0x35
	// Dwarf3
	DW_TAG_dwarf_procedure  = 0x36
	DW_TAG_restrict_type    = 0x37
	DW_TAG_interface_type   = 0x38
	DW_TAG_namespace        = 0x39
	DW_TAG_imported_module  = 0x3a
	DW_TAG_unspecified_type = 0x3b
	DW_TAG_partial_unit     = 0x3c
	DW_TAG_imported_unit    = 0x3d
	DW_TAG_condition        = 0x3f
	DW_TAG_shared_type      = 0x40
	// Dwarf4
	DW_TAG_type_unit             = 0x41
	DW_TAG_rvalue_reference_type = 0x42
	DW_TAG_template_alias        = 0x43

	// User defined
	DW_TAG_lo_user = 0x4080
	DW_TAG_hi_user = 0xffff
)

// Table 19
const (
	DW_CHILDREN_no  = 0x00
	DW_CHILDREN_yes = 0x01
)

// Not from the spec, but logically belongs here
const (
	DW_CLS_ADDRESS = 0x01 + iota
	DW_CLS_BLOCK
	DW_CLS_CONSTANT
	DW_CLS_FLAG
	DW_CLS_PTR // lineptr, loclistptr, macptr, rangelistptr
	DW_CLS_REFERENCE
	DW_CLS_ADDRLOC
	DW_CLS_STRING

	// Go-specific internal hackery.
	DW_CLS_GO_TYPEREF
)

// Table 20
const (
	DW_AT_sibling              = 0x01 // reference
	DW_AT_location             = 0x02 // block, loclistptr
	DW_AT_name                 = 0x03 // string
	DW_AT_ordering             = 0x09 // constant
	DW_AT_byte_size            = 0x0b // block, constant, reference
	DW_AT_bit_offset           = 0x0c // block, constant, reference
	DW_AT_bit_size             = 0x0d // block, constant, reference
	DW_AT_stmt_list            = 0x10 // lineptr
	DW_AT_low_pc               = 0x11 // address
	DW_AT_high_pc              = 0x12 // address
	DW_AT_language             = 0x13 // constant
	DW_AT_discr                = 0x15 // reference
	DW_AT_discr_value          = 0x16 // constant
	DW_AT_visibility           = 0x17 // constant
	DW_AT_import               = 0x18 // reference
	DW_AT_string_length        = 0x19 // block, loclistptr
	DW_AT_common_reference     = 0x1a // reference
	DW_AT_comp_dir             = 0x1b // string
	DW_AT_const_value          = 0x1c // block, constant, string
	DW_AT_containing_type      = 0x1d // reference
	DW_AT_default_value        = 0x1e // reference
	DW_AT_inline               = 0x20 // constant
	DW_AT_is_optional          = 0x21 // flag
	DW_AT_lower_bound          = 0x22 // block, constant, reference
	DW_AT_producer             = 0x25 // string
	DW_AT_prototyped           = 0x27 // flag
	DW_AT_return_addr          = 0x2a // block, loclistptr
	DW_AT_start_scope          = 0x2c // constant
	DW_AT_bit_stride           = 0x2e // constant
	DW_AT_upper_bound          = 0x2f // block, constant, reference
	DW_AT_abstract_origin      = 0x31 // reference
	DW_AT_accessibility        = 0x32 // constant
	DW_AT_address_class        = 0x33 // constant
	DW_AT_artificial           = 0x34 // flag
	DW_AT_base_types           = 0x35 // reference
	DW_AT_calling_convention   = 0x36 // constant
	DW_AT_count                = 0x37 // block, constant, reference
	DW_AT_data_member_location = 0x38 // block, constant, loclistptr
	DW_AT_decl_column          = 0x39 // constant
	DW_AT_decl_file            = 0x3a // constant
	DW_AT_decl_line            = 0x3b // constant
	DW_AT_declaration          = 0x3c // flag
	DW_AT_discr_list           = 0x3d // block
	DW_AT_encoding             = 0x3e // constant
	DW_AT_external             = 0x3f // flag
	DW_AT_frame_base           = 0x40 // block, loclistptr
	DW_AT_friend               = 0x41 // reference
	DW_AT_identifier_case      = 0x42 // constant
	DW_AT_macro_info           = 0x43 // macptr
	DW_AT_namelist_item        = 0x44 // block
	DW_AT_priority             = 0x45 // reference
	DW_AT_segment              = 0x46 // block, loclistptr
	DW_AT_specification        = 0x47 // reference
	DW_AT_static_link          = 0x48 // block, loclistptr
	DW_AT_type                 = 0x49 // reference
	DW_AT_use_location         = 0x4a // block, loclistptr
	DW_AT_variable_parameter   = 0x4b // flag
	DW_AT_virtuality           = 0x4c // constant
	DW_AT_vtable_elem_location = 0x4d // block, loclistptr
	// Dwarf3
	DW_AT_allocated      = 0x4e // block, constant, reference
	DW_AT_associated     = 0x4f // block, constant, reference
	DW_AT_data_location  = 0x50 // block
	DW_AT_byte_stride    = 0x51 // block, constant, reference
	DW_AT_entry_pc       = 0x52 // address
	DW_AT_use_UTF8       = 0x53 // flag
	DW_AT_extension      = 0x54 // reference
	DW_AT_ranges         = 0x55 // rangelistptr
	DW_AT_trampoline     = 0x56 // address, flag, reference, string
	DW_AT_call_column    = 0x57 // constant
	DW_AT_call_file      = 0x58 // constant
	DW_AT_call_line      = 0x59 // constant
	DW_AT_description    = 0x5a // string
	DW_AT_binary_scale   = 0x5b // constant
	DW_AT_decimal_scale  = 0x5c // constant
	DW_AT_small          = 0x5d // reference
	DW_AT_decimal_sign   = 0x5e // constant
	DW_AT_digit_count    = 0x5f // constant
	DW_AT_picture_string = 0x60 // string
	DW_AT_mutable        = 0x61 // flag
	DW_AT_threads_scaled = 0x62 // flag
	DW_AT_explicit       = 0x63 // flag
	DW_AT_object_pointer = 0x64 // reference
	DW_AT_endianity      = 0x65 // constant
	DW_AT_elemental      = 0x66 // flag
	DW_AT_pure           = 0x67 // flag
	DW_AT_recursive      = 0x68 // flag
	DW_AT_addr_base      = 0x73 // addrptr

	DW_AT_lo_user = 0x2000 // ---
	DW_AT_hi_user = 0x3fff // ---
)

// Table 21
const (
	DW_FORM_addr      = 0x01 // address
	DW_FORM_block2    = 0x03 // block
	DW_FORM_block4    = 0x04 // block
	DW_FORM_data2     = 0x05 // constant
	DW_FORM_data4     = 0x06 // constant, lineptr, loclistptr, macptr, rangelistptr
	DW_FORM_data8     = 0x07 // constant, lineptr, loclistptr, macptr, rangelistptr
	DW_FORM_string    = 0x08 // string
	DW_FORM_block     = 0x09 // block
	DW_FORM_block1    = 0x0a // block
	DW_FORM_data1     = 0x0b // constant
	DW_FORM_flag      = 0x0c // flag
	DW_FORM_sdata     = 0x0d // constant
	DW_FORM_strp      = 0x0e // string
	DW_FORM_udata     = 0x0f // constant
	DW_FORM_ref_addr  = 0x10 // reference
	DW_FORM_ref1      = 0x11 // reference
	DW_FORM_ref2      = 0x12 // reference
	DW_FORM_ref4      = 0x13 // reference
	DW_FORM_ref8      = 0x14 // reference
	DW_FORM_ref_udata = 0x15 // reference
	DW_FORM_indirect  = 0x16 // (see Section 7.5.3)
	// Dwarf4
	DW_FORM_sec_offset   = 0x17 // lineptr, loclistptr, macptr, rangelistptr
	DW_FORM_exprloc      = 0x18 // exprloc
	DW_FORM_flag_present = 0x19 // flag
	// Dwarf5
	DW_FORM_addrx = 0x1b
	// Pseudo-form: expanded to data4 on IOS, udata elsewhere.
	DW_FORM_udata_pseudo = 0x99
	// Pseudo-form: expands to DW_FORM_addrx in DWARF5, DW_FORM_addr in DWARF4
	DW_FORM_lo_pc_pseudo = 0x9a
	// Pseudo-form: expands to DW_FORM_udata in DWARF5, DW_FORM_addr in DWARF4
	DW_FORM_hi_pc_pseudo = 0x9b
)

// Table 24 (#operands, notes)
const (
	DW_OP_addr                = 0x03 // 1 constant address (size target specific)
	DW_OP_deref               = 0x06 // 0
	DW_OP_const1u             = 0x08 // 1 1-byte constant
	DW_OP_const1s             = 0x09 // 1 1-byte constant
	DW_OP_const2u             = 0x0a // 1 2-byte constant
	DW_OP_const2s             = 0x0b // 1 2-byte constant
	DW_OP_const4u             = 0x0c // 1 4-byte constant
	DW_OP_const4s             = 0x0d // 1 4-byte constant
	DW_OP_const8u             = 0x0e // 1 8-byte constant
	DW_OP_const8s             = 0x0f // 1 8-byte constant
	DW_OP_constu              = 0x10 // 1 ULEB128 constant
	DW_OP_consts              = 0x11 // 1 SLEB128 constant
	DW_OP_dup                 = 0x12 // 0
	DW_OP_drop                = 0x13 // 0
	DW_OP_over                = 0x14 // 0
	DW_OP_pick                = 0x15 // 1 1-byte stack index
	DW_OP_swap                = 0x16 // 0
	DW_OP_rot                 = 0x17 // 0
	DW_OP_xderef              = 0x18 // 0
	DW_OP_abs                 = 0x19 // 0
	DW_OP_and                 = 0x1a // 0
	DW_OP_div                 = 0x1b // 0
	DW_OP_minus               = 0x1c // 0
	DW_OP_mod                 = 0x1d // 0
	DW_OP_mul                 = 0x1e // 0
	DW_OP_neg                 = 0x1f // 0
	DW_OP_not                 = 0x20 // 0
	DW_OP_or                  = 0x21 // 0
	DW_OP_plus                = 0x22 // 0
	DW_OP_plus_uconst         = 0x23 // 1 ULEB128 addend
	DW_OP_shl                 = 0x24 // 0
	DW_OP_shr                 = 0x25 // 0
	DW_OP_shra                = 0x26 // 0
	DW_OP_xor                 = 0x27 // 0
	DW_OP_skip                = 0x2f // 1 signed 2-byte constant
	DW_OP_bra                 = 0x28 // 1 signed 2-byte constant
	DW_OP_eq                  = 0x29 // 0
	DW_OP_ge                  = 0x2a // 0
	DW_OP_gt                  = 0x2b // 0
	DW_OP_le                  = 0x2c // 0
	DW_OP_lt                  = 0x2d // 0
	DW_OP_ne                  = 0x2e // 0
	DW_OP_lit0                = 0x30 // 0 ...
	DW_OP_lit31               = 0x4f // 0 literals 0..31 = (DW_OP_lit0 + literal)
	DW_OP_reg0                = 0x50 // 0 ..
	DW_OP_reg31               = 0x6f // 0 reg 0..31 = (DW_OP_reg0 + regnum)
	DW_OP_breg0               = 0x70 // 1 ...
	DW_OP_breg31              = 0x8f // 1 SLEB128 offset base register 0..31 = (DW_OP_breg0 + regnum)
	DW_OP_regx                = 0x90 // 1 ULEB128 register
	DW_OP_fbreg               = 0x91 // 1 SLEB128 offset
	DW_OP_bregx               = 0x92 // 2 ULEB128 register followed by SLEB128 offset
	DW_OP_piece               = 0x93 // 1 ULEB128 size of piece addressed
	DW_OP_deref_size          = 0x94 // 1 1-byte size of data retrieved
	DW_OP_xderef_size         = 0x95 // 1 1-byte size of data retrieved
	DW_OP_nop                 = 0x96 // 0
	DW_OP_push_object_address = 0x97 // 0
	DW_OP_call2               = 0x98 // 1 2-byte offset of DIE
	DW_OP_call4               = 0x99 // 1 4-byte offset of DIE
	DW_OP_call_ref            = 0x9a // 1 4- or 8-byte offset of DIE
	DW_OP_form_tls_address    = 0x9b // 0
	DW_OP_call_frame_cfa      = 0x9c // 0
	DW_OP_bit_piece           = 0x9d // 2
	DW_OP_lo_user             = 0xe0
	DW_OP_hi_user             = 0xff
)

// Table 25
const (
	DW_ATE_address         = 0x01
	DW_ATE_boolean         = 0x02
	DW_ATE_complex_float   = 0x03
	DW_ATE_float           = 0x04
	DW_ATE_signed          = 0x05
	DW_ATE_signed_char     = 0x06
	DW_ATE_unsigned        = 0x07
	DW_ATE_unsigned_char   = 0x08
	DW_ATE_imaginary_float = 0x09
	DW_ATE_packed_decimal  = 0x0a
	DW_ATE_numeric_string  = 0x0b
	DW_ATE_edited          = 0x0c
	DW_ATE_signed_fixed    = 0x0d
	DW_ATE_unsigned_fixed  = 0x0e
	DW_ATE_decimal_float   = 0x0f
	DW_ATE_lo_user         = 0x80
	DW_ATE_hi_user         = 0xff
)

// Table 26
const (
	DW_DS_unsigned           = 0x01
	DW_DS_leading_overpunch  = 0x02
	DW_DS_trailing_overpunch = 0x03
	DW_DS_leading_separate   = 0x04
	DW_DS_trailing_separate  = 0x05
)

// Table 27
const (
	DW_END_default = 0x00
	DW_END_big     = 0x01
	DW_END_little  = 0x02
	DW_END_lo_user = 0x40
	DW_END_hi_user = 0xff
)

// Table 28
const (
	DW_ACCESS_public    = 0x01
	DW_ACCESS_protected = 0x02
	DW_ACCESS_private   = 0x03
)

// Table 29
const (
	DW_VIS_local     = 0x01
	DW_VIS_exported  = 0x02
	DW_VIS_qualified = 0x03
)

// Table 30
const (
	DW_VIRTUALITY_none         = 0x00
	DW_VIRTUALITY_virtual      = 0x01
	DW_VIRTUALITY_pure_virtual = 0x02
)

// Table 31
const (
	DW_LANG_C89         = 0x0001
	DW_LANG_C           = 0x0002
	DW_LANG_Ada83       = 0x0003
	DW_LANG_C_plus_plus = 0x0004
	DW_LANG_Cobol74     = 0x0005
	DW_LANG_Cobol85     = 0x0006
	DW_LANG_Fortran77   = 0x0007
	DW_LANG_Fortran90   = 0x0008
	DW_LANG_Pascal83    = 0x0009
	DW_LANG_Modula2     = 0x000a
	// Dwarf3
	DW_LANG_Java           = 0x000b
	DW_LANG_C99            = 0x000c
	DW_LANG_Ada95          = 0x000d
	DW_LANG_Fortran95      = 0x000e
	DW_LANG_PLI            = 0x000f
	DW_LANG_ObjC           = 0x0010
	DW_LANG_ObjC_plus_plus = 0x0011
	DW_LANG_UPC            = 0x0012
	DW_LANG_D              = 0x0013
	// Dwarf4
	DW_LANG_Python = 0x0014
	// Dwarf5
	DW_LANG_Go = 0x0016

	DW_LANG_lo_user = 0x8000
	DW_LANG_hi_user = 0xffff
)

// Table 32
const (
	DW_ID_case_sensitive   = 0x00
	DW_ID_up_case          = 0x01
	DW_ID_down_case        = 0x02
	DW_ID_case_insensitive = 0x03
)

// Table 33
const (
	DW_CC_normal  = 0x01
	DW_CC_program = 0x02
	DW_CC_nocall  = 0x03
	DW_CC_lo_user = 0x40
	DW_CC_hi_user = 0xff
)

// Table 34
const (
	DW_INL_not_inlined          = 0x00
	DW_INL_inlined              = 0x01
	DW_INL_declared_not_inlined = 0x02
	DW_INL_declared_inlined     = 0x03
)

// Table 35
const (
	DW_ORD_row_major = 0x00
	DW_ORD_col_major = 0x01
)

// Table 36
const (
	DW_DSC_label = 0x00
	DW_DSC_range = 0x01
)

// Table 37
const (
	DW_LNS_copy             = 0x01
	DW_LNS_advance_pc       = 0x02
	DW_LNS_advance_line     = 0x03
	DW_LNS_set_file         = 0x04
	DW_LNS_set_column       = 0x05
	DW_LNS_negate_stmt      = 0x06
	DW_LNS_set_basic_block  = 0x07
	DW_LNS_const_add_pc     = 0x08
	DW_LNS_fixed_advance_pc = 0x09
	// Dwarf3
	DW_LNS_set_prologue_end   = 0x0a
	DW_LNS_set_epilogue_begin = 0x0b
	DW_LNS_set_isa            = 0x0c
)

// Table 38
const (
	DW_LNE_end_sequence = 0x01
	DW_LNE_set_address  = 0x02
	DW_LNE_define_file  = 0x03
	DW_LNE_lo_user      = 0x80
	DW_LNE_hi_user      = 0xff
)

// Table 7.27 (DWARF version 5), containing the encodings for the
// line number header entry formats.
const (
	DW_LNCT_path            = 0x01
	DW_LNCT_directory_index = 0x02
	DW_LNCT_timestamp       = 0x03
	DW_LNCT_size            = 0x04
	DW_LNCT_md5             = 0x05
	DW_LNCT_lo_user         = 0x2000
	DW_LNCT_hi_user         = 0x3fff
)

// Table 7.2 (DWARF version 5), dwarf unit type encodings.
const (
	DW_UT_compile       = 0x01
	DW_UT_type          = 0x02
	DW_UT_partial       = 0x03
	DW_UT_skeleton      = 0x04
	DW_UT_split_compile = 0x05
	DW_UT_split_type    = 0x06
	DW_UT_lo_user       = 0x80
	DW_UT_hi_user       = 0xff
)

// Table 39
const (
	DW_MACINFO_define     = 0x01
	DW_MACINFO_undef      = 0x02
	DW_MACINFO_start_file = 0x03
	DW_MACINFO_end_file   = 0x04
	DW_MACINFO_vendor_ext = 0xff
)

// Table 40.
const (
	// operand,...
	DW_CFA_nop              = 0x00
	DW_CFA_set_loc          = 0x01 // address
	DW_CFA_advance_loc1     = 0x02 // 1-byte delta
	DW_CFA_advance_loc2     = 0x03 // 2-byte delta
	DW_CFA_advance_loc4     = 0x04 // 4-byte delta
	DW_CFA_offset_extended  = 0x05 // ULEB128 register, ULEB128 offset
	DW_CFA_restore_extended = 0x06 // ULEB128 register
	DW_CFA_undefined        = 0x07 // ULEB128 register
	DW_CFA_same_value       = 0x08 // ULEB128 register
	DW_CFA_register         = 0x09 // ULEB128 register, ULEB128 register
	DW_CFA_remember_state   = 0x0a
	DW_CFA_restore_state    = 0x0b

	DW_CFA_def_cfa            = 0x0c // ULEB128 register, ULEB128 offset
	DW_CFA_def_cfa_register   = 0x0d // ULEB128 register
	DW_CFA_def_cfa_offset     = 0x0e // ULEB128 offset
	DW_CFA_def_cfa_expression = 0x0f // BLOCK
	DW_CFA_expression         = 0x10 // ULEB128 register, BLOCK
	DW_CFA_offset_extended_sf = 0x11 // ULEB128 register, SLEB128 offset
	DW_CFA_def_cfa_sf         = 0x12 // ULEB128 register, SLEB128 offset
	DW_CFA_def_cfa_offset_sf  = 0x13 // SLEB128 offset
	DW_CFA_val_offset         = 0x14 // ULEB128, ULEB128
	DW_CFA_val_offset_sf      = 0x15 // ULEB128, SLEB128
	DW_CFA_val_expression     = 0x16 // ULEB128, BLOCK

	DW_CFA_lo_user = 0x1c
	DW_CFA_hi_user = 0x3f

	// Opcodes that take an addend operand.
	DW_CFA_advance_loc = 0x1 << 6 // +delta
	DW_CFA_offset      = 0x2 << 6 // +register (ULEB128 offset)
	DW_CFA_restore     = 0x3 << 6 // +register
)
