# Recovered CI validation records

These records were copied from the prior environment's returned tool output.
They describe the original validation; copying them did not rerun a validator.
The five command-result JSON objects retain the printed field values. Their
current file serialization is a recovery artifact, not a claim that the lost
on-disk JSON files had byte-identical serialization.

The recorded source HEAD was cb22f3a3762838fe85692403dbcd1206ae2c2375 while a
normal merge with 9a628f5b87f3c69a392ba539d11c7c9a37b98566 was uncommitted.
Root retained the ten final input hashes separately. Those hashes, rather than
the source HEAD alone, identify the validated input contents.

## Retained results

- Final actionlint: exit 0; stdout and stderr were empty.
- Direct ShellCheck on both installers: exit 0; both streams were empty.
- YAML and Bash validation: exit 0; eight YAML documents, 67 run blocks and
  two standalone scripts; no failures; stderr was empty.
- Isolated SC2155 mutation: exit 1; one expected diagnostic; stderr was empty.
- Actionlint on unchanged originals after the mutation: exit 0; both streams
  were empty.

The retained final statement was:

```text
Final originals: all 10 hashes unchanged; isolated SC2155 mutation caught; unchanged originals pass.
```

The mutation restored only the masked bootstrap export in a separate copy of
compiler-stability.yml. The original workflow was untouched. The retained
normalized diagnostic is in mutation-normalized.txt; ANSI escapes had already
been removed by the original wrapper.

## Recovery limits

The original ANSI-colored mutation stdout is unavailable. Its recorded hash
remains in mutation-actionlint.json. The YAML result values are recovered in
yaml-bash-result.json, but the original stdout serialization and trailing
newline are not recovered. No replacement raw streams were synthesized.

The full original tool-provenance JSON, helper-script hash, and original-file
hash-map JSON were saved only on the lost filesystem and were not returned in
this lane's inspected V8 stores. Parent-held executable and input hashes are
separate retained evidence. Freshly downloaded tools and any fresh validation
results are recorded outside this recovered directory.
