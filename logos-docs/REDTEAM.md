# Red-team test ledger

Resolutions: logos-docs/SPEC.md Section 17. Status values: planned, tested, satisfied upstream, blocked.

| ID | Milestone | Test | Status |
| --- | --- | --- | --- |
| #1 |M2| test/logos/redteam/attack_001_split_union_ptr_scalar | tested |
| #2 | M5 | test/logos/run/zerocopy.lgo | tested |
| #3 | M5 | test/logos/run/constmap_switch.lgo; src/cmd/compile/internal/logos/constmap_test.go (TestConstmapSwitchMax) | tested |
| #4 | M18 |  | planned |
| #5 | M8, M20 |  | planned |
| #6 | M6 | test/logos/run/lambda.lgo; test/logos/run/chains.lgo (dblBlock, count with ?) | tested |
| #7 | M3, M11 |  | planned |
| #8 | M3 | test/logos/run/platform.lgo | tested |
| #9 | M13 |  | planned |
| #10 | M9 |  | planned |
| #11 | M18 |  | planned |
| #12 | M18 |  | planned |
| #13 | M8, M9 |  | planned |
| #14 | M3 | src/internal/types/testdata/logos/narrowing.lgo | tested |
| #15 | M13 |  | planned |
| #16 | M5 | src/internal/types/testdata/logos/const_views.lgo (cvPlainReslice, cvPlainReuseLoop); test/logos/run/reslice_reuse.lgo | tested |
| #17 | M6 | test/logos/run/tuples.lgo (direct) | tested |
| #18 |M1| test/logos/redteam/attack_018_shadowed_named_return | tested |
| #19 | M7 | src/internal/types/testdata/logos/viewas.lgo; test/logos/run/viewas.lgo | tested |
| #20 | M18 |  | planned |
| #21 | M8 |  | planned |
| #22 | M15 |  | planned |
| #23 | M3 | src/internal/types/testdata/logos/error_first.lgo | tested |
| #24 | M5 | src/internal/types/testdata/logos/ctfe.lgo (BadMapRange); test/logos/run/const_map.lgo (sorted range) | tested |
| #25 | M12 | src/cmd/go/testdata/script/logos_cap_unsafe.txt | tested |
| #26 | M8 | src/logos/sync/sync_test.go (TestHotUpdateConcurrent, TestHotLayout) | tested |
| #27 | M16 |  | planned |
| #28 | M13 |  | planned |
| #29 | M19 |  | planned |
| #30 | M13 |  | planned |
| #31 | M6 | src/internal/types/testdata/logos/chains.lgo; test/logos/run/sort_side_effect.lgo | tested |
| #32 |M1| test/logos/redteam/attack_032_defer_lifo_order | tested |
| #33 | M17, M18 |  | planned |
| #34 | M6 | test/logos/run/interp.lgo (bigBox AllocsPerRun) | tested |
| #35 | M18 |  | planned |
| #36 |M2| test/logos/redteam/attack_036_open_union_value | tested |
| #37 | M7 |  | planned |
| #38 | M6 | src/cmd/compile/internal/syntax/parser_paren_logos_test.go | tested |
| #39 | M6 | test/logos/run/tuples.lgo (pair call unpack, tuple var pack) | tested |
| #40 |M2| test/logos/redteam/attack_040_union_equality | tested |
| #41 | M5, M6 | src/internal/types/testdata/logos/chains.lgo (chainMutateRecv, chainMutateParam) | tested |
| #42 |M1| test/logos/redteam/attack_042_defer_close_join | tested |
| #43 |M2| test/logos/redteam/attack_043_zero_sized_member | tested |
| #44 | M3 | src/cmd/go/testdata/script/logos_facts.txt | tested |
| #45 | M7 |  | planned |
| #46 | M18 |  | planned |
| #47 | M5 | src/internal/types/testdata/logos/ctfe.lgo | tested |
| #48 | M13 |  | planned |
| #49 | M3 | test/logos/run/option_union.lgo | tested |
| #50 | M5 | test/logos/run/rodata_reflect.lgo | tested |
| #51 | M5 | src/internal/types/testdata/logos/ctfe.lgo | tested |
| #52 | M6 | src/cmd/compile/internal/logos/m6gaps_test.go (TestScopeDeferOpenCoded) | tested |
| #53 | M18 |  | planned |
| #54 | M5 | test/logos/run/relro_write.lgo | tested |
| #55 |M1| test/logos/redteam/attack_055_errtrace_reset_cap | tested |
| #56 | M6 | test/logos/run/scope_defer.lgo (funcVarDefer, recvDefer) | tested |
| #57 |M2, M7| test/logos/redteam/attack_057_case_first_match | tested |
| #58 | M13 |  | planned |
| #59 | M18 |  | planned |
| #60 | M3 | src/internal/types/testdata/logos/nonnil_zero.lgo | tested |
| #61 | M3 | test/logos/run/option_nested.lgo | tested |
| #62 | M5 | src/internal/types/testdata/logos/const_views.lgo (Rename, Peek) | tested |
| #63 | M18 |  | planned |
| #64 | M6 | test/logos/run/interp.lgo (nextID once, tag order) | tested |
| #65 | M7 |  | planned |
| #66 |M1| test/logos/redteam/attack_066_errdefer_unnamed_results | tested |
| #67 | M6, M13 | M6 half: src/cmd/compile/internal/logos/m6gaps_test.go (TestChainSumPairwise); test/logos/run/chains.lgo (sum order). M13: arm64 leg open | planned |
| #68 | M8 |  | planned |
| #69 | M10 |  | planned |
| #70 | M5 | test/logos/codegen/rodata.lgo | tested |
| #71 | M18 |  | planned |
| #72 | M7 | test/logos/run/viewas.lgo (GC keep-alive, need-have panic); src/logos/rt/viewas_test.go | tested |
| #73 | M5 | test/logos/run/const_map.lgo; src/cmd/compile/internal/logos/constmap_test.go | tested |
| #74 | M13 |  | planned |
| #75 |M2| test/logos/redteam/attack_075_nil_arm_common_type | tested |
| #76 | M7 |  | planned |
| #77 | M9 |  | planned |
| #78 | M16 |  | planned |
| #79 |M1| test/logos/redteam/attack_079_import_alias_collision | tested |
| #80 | M22 |  | planned |
| #81 | M18 |  | planned |
| #82 |M2| test/logos/redteam/attack_082_union_gc_shape | tested |
| #83 | M5 | test/logos/run/zerocopy.lgo | tested |
| #84 | M6 | src/cmd/compile/internal/logos/m6gaps_test.go (TestChainFuseOneLoop) | tested |
| #85 |M1| test/logos/redteam/attack_085_typed_nil_errdefer | tested |
| #86 | M7 |  | planned |
| #87 |M1| test/logos/redteam/attack_087_errdefer_panic_goexit | tested |
| #88 | M20 | src/cmd/cgo/internal/testlogos/fastcall_test.go | tested |
| #89 |M2| test/logos/redteam/attack_089_scalar_roundtrip | tested |
| #90 |M1| test/logos/redteam/attack_090_errdefer_recover_success_skip | tested |
| #91 | M7 |  | planned |
| #92 | M13 |  | planned |
| #93 | M5 | test/logos/run/constview_bound.lgo | tested |
| #94 | M10 | src/cmd/go/testdata/script/logos_context.txt; src/cmd/compile/internal/logos/context_test.go (TestContextLineCap); src/cmd/compile/internal/logos/context_bench_test.go (BenchmarkContext) | tested |
| #95 | M6 | test/logos/run/chains.lgo (take, drop, get clamps) | tested |
| #96 |M1| test/logos/redteam/attack_096_errdefer_armed_flag | tested |
| #97 |M2| test/logos/redteam/attack_097_struct_variant_slots | tested |
| #98 | M5 | test/logos/run/rodata_reflect.lgo | tested |
| #99 | M3 | src/internal/types/testdata/logos/map_index.lgo | tested |
| #100 | M15 |  | planned |
| #101 | M18 |  | planned |
| #102 | M5 | src/internal/types/testdata/logos/ctfe.lgo | tested |
| #103 | M5 | src/internal/types/testdata/logos/ctfe.lgo | tested |
| #104 | M13 |  | planned |
| #105 | M13 |  | planned |
| #106 | M18 |  | planned |
| #107 | M13 |  | planned |
| #108 | M7 |  | planned |
| #109 | M10 |  | planned |
| #110 | M5 | test/logos/run/freeze.lgo | tested |
| #111 | M15 | test/logos/codegen/arm64_pair.lgo; test/logos/codegen/arm64_pair.go; src/cmd/compile/internal/ssa/pair_logos_test.go (TestLogosPairARM64) | tested |
| #112 | M1 | src/cmd/go/testdata/script/logos_eject.txt | tested |
| #113 | M6 | src/cmd/compile/internal/logos/m6gaps_test.go (TestScopeDeferOpenCoded) | tested |
| #114 | M7 |  | planned |
| #115 | M3 | src/internal/types/testdata/logos/union_exhaustive.lgo | tested |
| #116 | M4 | src/internal/types/testdata/logos/property_iface.lgo | tested |
| #117 | M4 | src/internal/types/testdata/logos/property_iface.lgo | tested |
| #118 | M4 | src/internal/types/testdata/logos/property_iface.lgo | tested |
| #119 | M6 | test/logos/run/spread.lgo; src/internal/types/testdata/logos/spread.lgo | tested |
| #120 | M4 | test/logos/run/properties.lgo | tested |
| #121 | M13 |  | planned |
| #122 | M0 |  | planned |
| #123 | M6, M11 |  | planned |
| #124 | M13 |  | planned |
| #125 |M1| test/logos/redteam/attack_125_errdefer_join | tested |
| #126 | M6 | src/cmd/compile/internal/logos/m6gaps_test.go (TestInterpBindBeforeSizing) | tested |
| #127 | M4 | src/internal/types/testdata/logos/property_iface.lgo | tested |
| #128 | M7 |  | planned |
| #129 | M7 |  | planned |
| #130 | M7 |  | planned |
| #131 | M4 | src/internal/types/testdata/logos/property_iface.lgo | tested |
| #132 | M4 | test/logos/run/properties.lgo | tested |
| #133 | M0 |  | planned |
| #134 | M13 |  | planned |
| #135 |M1| test/logos/redteam/attack_135_lifo_order | tested |
| #136 | M8 |  | planned |
| #137 | M15 |  | planned |
| #138 | M18 |  | planned |
| #139 | M11 |  | planned |
| #140 | M5 | src/cmd/compile/internal/logos/constmap_test.go (TestConstmapBudgetIsOneMillion, TestConstmapProbeIsSingleProbe); src/logos/rt/constmap_proof_test.go (TestConstTableGetSingleProbe) | tested |
| #141 | M4 | test/logos/run/properties.lgo | tested |
| #142 | M6 | test/logos/run/chains.lgo (at-return closes) | tested |
| #143 | M8 |  | planned |
| #144 | M17, M20 |  | planned |
| #145 | M4 | src/internal/types/testdata/logos/property_iface.lgo | tested |
| #146 | M6 | src/cmd/compile/internal/logos/spread_test.go; test/logos/codegen/spread_nomemmove.lgo | tested |
| #147 | M8 |  | planned |
| #148 | M8, M9 |  | planned |
| #149 | M15 | test/logos/codegen/clear_kept.lgo | satisfied upstream |
| #150 | M7 |  | planned |
| #151 | M8 |  | planned |
| #152 | M5 | test/logos/run/zerocopy.lgo (other-provenance AllocsPerRun, survival) | tested |
| #153 | M8 | src/logos/sync/sync_test.go (TestHotNoMemoryGrowth) | tested |
| #154 | M8 | src/logos/sync/sync_test.go (TestHotLayout) | tested |
| #155 | M18 |  | planned |
| #156 | M3 | src/internal/types/testdata/logos/error_first.lgo | tested |
| #157 | M7, M9 |  | planned |
| #158 | M9 |  | planned |
| #159 | M19 |  | planned |
| #160 | M19, M20 |  | planned |
| #161 | M19 |  | planned |
| #162 | M9 |  | planned |
| #163 | M9 |  | planned |
| #164 | M9 |  | planned |
| #165 | M20 | src/cmd/cgo/internal/testlogos/fastcall_test.go | tested |
| #166 | M19 |  | planned |
| #167 | M8, M9 |  | planned |
| #168 | M19 |  | planned |
| #169 | M15 |  | planned |
| #170 | M9 |  | planned |
| #171 | M19 |  | planned |
| #172 | M9 |  | planned |
| #173 | M20 | src/cmd/cgo/internal/testlogos/fastcall_test.go | tested |
| #174 | M7 |  | planned |
| #175 | M19 |  | planned |
| #176 | M8 | src/logos/sync/sync_test.go (TestLockedWithLockUnlocksAfterPanic) | tested |
| #177 | M6 | test/logos/run/interp.lgo (format verbs printed literally) | tested |
| #178 | M8 |  | planned |
| #179 | M6 | test/logos/run/chains.lgo (64-item stability, NaN order) | tested |
| #180 | M5 | src/cmd/compile/internal/logos/ctfe/ctfe_test.go (TestWidthsComposite); test/logos/run/const_init_read.lgo | tested |
| #181 | M18 |  | planned |
| #182 | M9 |  | planned |
| #183 | M0 | src/cmd/go/testdata/script/logos_interop.txt (case 1) | tested |
| #184 | M0 | src/cmd/go/testdata/script/logos_interop.txt (case 2) | tested |
| #185 | M2 | test/logos/redteam/attack_185_dual_use_union; src/cmd/go/testdata/script/logos_eject_union.txt; src/cmd/go/testdata/script/logos_interop_union.txt | tested |
| #186 | M3 | src/cmd/go/testdata/script/logos_interop_nil.txt | tested |
| #187 | M16 |  | planned |
| #188 | M20 | src/cmd/cgo/internal/testlogos/fastcall_test.go | tested |
| #189 | M20 | src/cmd/cgo/internal/testlogos/fastcall_test.go | tested |
