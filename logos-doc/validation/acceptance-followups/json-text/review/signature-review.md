# Concurrent quoted-text signature review

This is a read-only comparison of source45
(`45c29ff215477f7e403188d7452fc79aef05d791`) with live integration
`8afec95580ed24e593307fd9d30d626eff31e22d`, including signature commit
`c986384f43ed5c2b0759580d98814f2494a7bd7c`. No Go command was run for this
review. Exact per-file Git blob and SHA-256 identities are in
`source-identities.json`; the signature commit's complete runtime JSON delta
is in `c986384f-signatures.patch`.

## Preserve the element-type constraint

Both profiles now declare:

```go
func QuotedText[E any, T interface {
	*E
	encoding.TextUnmarshaler
}](s *Scanner, dst T)
```

The type constraint supplies the destination element type directly. Classic
error construction can use `reflect.TypeFor[E]()` instead of inspecting the
runtime pointer. JSONv2 can also use that type without a `Kind` branch. The
remaining `reflect.Type` is required by the public JSON error API.

All direct in-tree `QuotedText` calls omit explicit type arguments and pass a
concrete pointer. The compiler's `quotedMethodCall` emits the same inferred
value call. Pointer fields use the existing `QuotedTextPtr[E]` path, whose
signature is unchanged. The compiler JSON source files are byte-identical
between source45 and the reviewed integration, including the C-aware source
type printer. This establishes source compatibility for the generated and
in-tree calls inspected here. It does not claim compatibility for arbitrary
external callers that explicitly supplied the old single type argument or
passed an interface value directly to this runtime helper.

## Compose the proved Unicode repair

Keep the new `QuotedText` signature and `TypeFor[E]()` calls in both profiles.
In JSONv2, use `unquoteLiteralBytes(string(tok))` at both text dispatch sites
and retain the pointer path's local `err :=` declaration. The source45 gates
prove these decoder calls and their alias/null behavior independently of the
new signature. The final merged gates must prove their composition.

The release merge currently adds a second decoder named `permissiveUnquote`
and imports `unicode`, `unicode/utf16`, and `unicode/utf8` for it. Exclude that
duplicate implementation and its imports when applying the shared-helper
repair. Preserve the existing strict scalar decoder, syntax/error behavior,
quoted-null handling, and fallback context.

## Preserve previous fixes and frozen tests

Within runtime JSON, the exact c986 commit changes only `path_v1.go` and
`path_v2.go`. The release merge separately reintroduces `methodlessType`,
`decodePlain`, `FallbackFrom`, and `FallbackFromPtr` in `scan.go`. Those helpers
were deliberately excluded from source45. Restore `scan.go` to source45 so
the previous compile-time fallback selection and one-call guarantees remain.

Restore `scan_test.go` to source45. Use the exact corrected
`stream_v2_test.go` from this bundle, with SHA-256
`08729986d528dd8a0c003893b0069639b5f3d7201b122a4105be162bfa088232`.
That file retains all source45 fallback-routing and one-call assertions and
adds both unchanged upstream regression blocks under the JSONv2 build tag.
The reviewed integration otherwise retains those existing source45 stream
assertions; only the original append-test placement needs the documented
profile correction.

No worktree source, commit, or ref was changed by this review. The integration
owner applies the resolutions and owns the fresh combined build and gates.
