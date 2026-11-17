# Final integration verification

This record accompanies the four source-repair commits ending at local
`0a8eef6bab7d85e89e8e3efd0b933d73e78dff79`, with source tree
`33752c03819ed5afcb37f87633a3cac668e181a8`. It verifies the subsequent
acceptance-request and evidence integration. It does not replace the
nineteen installed behavioral gate records or any hosted release gate.

The existing compiled `TestRedteamLedgerRows` passes with `-count=1` and
`-short` using the fresh combined installation. The existing release-contract
checker also passes against the updated active source request. Their exact
specifications, source/tool bindings, commands, environment, raw streams and
results are in `compiled-and-static-checks/`. The unchanged recorder and its
imports are retained in the adjacent installed acceptance packet.

The independent review in `independent-review/` verifies all 306 task records,
350 attack rows and 583 proving-path occurrences; the seven blocked rows,
six release fixtures, twelve repaired files and M6 attack patch are unchanged.
It checks every installed/component packet and the active frozen manifests.
Actual actionlint and ShellCheck pass. All 32 frozen audit inputs remain
unchanged, and the review finds no metadata defect. No Go command was run by
the independent reviewer.

Original evidence bytes, timestamps, source identities and pending-scope
statements remain intact. The umbrella README and candidate record identify
current acceptance requirements. Full Go, all six no-short cells and separate
os legs, SWIG/LTO, unfiltered stamped parity, fresh runtime/reproducibility and
M6 language closure remain distinct obligations.

`source-map.json` records the exact copied origins. `files.sha256` covers the
payload; `manifest.json` also covers that checksum list. This integration adds
no production source, test assertion, timeout change, or release exception.
