# CTFE map-field store oracle correction

This is a test-only correction based on public commit
`45c29ff215477f7e403188d7452fc79aef05d791`
(tree `32b96f3a7c289c987bc021fe5a5adbf35cfe226b`).
No CTFE implementation, normative specification, or existing M24 assertion
changes.

## Why the old rejection is obsolete

The previous errorcheck fixture expected a const-evaluation diagnostic for
`s.M["k"] = 4`, where `M` is a fresh local map field. SPEC 2.1, lines
448-451 at that commit, requires Go assignment identity, evaluation order,
and map-reference sharing. Line 462 rejects forms that remain unsupported;
it does not permanently prohibit a supported map-field assignment.

The assignment repair in `c3733e8adfc6c3839b50b11c1b985c5f56f92fa9`
evaluates and captures arbitrary map-valued destination bases instead of
handling only bare map names. The later aggregate-storage repair in
`0cd8c0366759596caa8c653f521eebbdea2c1024` retains map-reference sharing.
This fixture uses no mutable package state, pointer, I/O, goroutine, or
dynamic-map iteration in its const function.

The main hosted workflow at the frozen source reported
`compilation succeeded unexpectedly` for the old fixture:
[run 43906416850, job 133436353880](https://github.com/gosuda/LoGOs/actions/runs/43906416850/job/133436353880).
That is evidence of the obsolete negative oracle, not a pass of its replacement.

## Preserved input and stronger positive oracle

`const_func_unsupported_store.original.lgo` preserves the exact old bytes.
The active replacement is
`test/logos/run/const_func_map_field_store.lgo`, with expected output
`4` followed by LF. Its `storeField` declaration, including its final LF,
is byte-identical to the old declaration.

The assignment `var _ [4]struct{} = [stored]struct{}{}` requires
`stored` to be the compile-time constant 4. The program prints that constant
and checks that another call agrees. That call may also fold, so this record
does not claim an independently unfolded runtime evaluation.

| File | SHA-256 |
|---|---|
| Preserved original | `0997677cba3f15d6be766c190860622bbba3104a1482d4f4293a63640c0bf02a` |
| Positive fixture | `1ea5cfa8168c9f1736aaeb77b7440126b496bfaaea7216e378989c52096ea2f5` |
| Expected output | `7de1555df0c2700329e815b93b32c571c3ea54dc967b89e81ab73b9972b72d1d` |
| Unchanged function declaration | `5c37c42ae72003063a84baa513f2daf13a7fbf4288bce6de2a23d7ca6b0faae1` |

Remote reconstruction verified the three complete-file digests against the
values frozen before executor loss and reused the original Git blob
`bc8cbfa62cfd6526d3ddeeb9d9f0a8d4937a247a`.
The genuine bad-key, NaN, nil-map, and assignment-failure controls and all
historical freeze manifests remain byte-identical to the base tree.

## Execution status

The authorized new native, ejected, and testdir checks did not start.
The executor disconnected before any Go command was launched. The bounded
availability check at 2026-11-16 17:33:10 UTC returned:

```text
exec-server transport disconnected; failed to resume exec-server session:
recovery timed out after 45s
```

No GREEN result or new mutation proof is claimed for the positive fixture.
Required acceptance remains: native output 4; ejection followed by fork and
pinned upstream Go 1.27.2/1.27.3 execution with the same output; the actual
testdir positive fixture; and unchanged genuine error controls. Source
temporary directories must be real directories outside workspace Git
ancestry, such as under `/home/agent`; compiler object storage can use disk.
