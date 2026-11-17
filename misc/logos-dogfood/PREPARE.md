# Prepare pinned corpus inputs

`prepare.py` supplies the inputs that `run.py` deliberately does not fetch.
It reads the same manifest and imports the runner's source verification and
process-recording functions. The runner, source pins, policy files, diagnostic
assertions, and acceptance rules remain unchanged.

The default manifest contains 28 cases backed by 26 source trees: nine Git
checkouts, 15 versioned modules, and two local fixtures. Variants share an input
only when their module, origin, revision, version, and checksum agree.

## Run

Use Python 3.11 or newer through uv, Git, and an independently verified stock
Go 1.27.4 executable. The output directory must be new.

```sh
uv run --no-project python misc/logos-dogfood/prepare.py \
  --go-oracle /opt/go1.27.4/bin/go \
  --out /tmp/logos-corpus-inputs

uv run --no-project python misc/logos-dogfood/run.py \
  --go-oracle /opt/go1.27.4/bin/go \
  --go-candidate "$PWD/bin/go" --logos "$PWD/bin/logos" \
  --gglint /opt/gglint/gglint \
  --source-root /tmp/logos-corpus-inputs/sources \
  --module-cache /tmp/logos-corpus-inputs/modules \
  --out /tmp/logos-corpus-results
```

Pass `--manifest` to prepare another reviewed manifest. Policies are verified
but never applied during preparation. The runner applies its existing policy
rules later. Preparation never builds or runs source packages, upstream tests,
generators, or the GoVWA server.

Git inputs use the manifest's HTTPS origin and full commit ID. Fetches do not
select a moving branch. Ambient Git variables that redirect repositories,
indexes, or object storage are rejected before creating the output directory.
The helper overrides global Git configuration and prompts for its fetches.
This is a controlled CI environment, not a filesystem or network sandbox.

Module inputs use exact version queries through `https://proxy.golang.org`
with `sum.golang.org` enabled. The existing verifier checks the top-level
module's origin revision and recomputes its pinned h1 from extracted content.
No source is accepted on the strength of a download marker alone.

## Dependency integrity

Each verified input is copied into a private work directory. Its original
`go.mod` and `go.sum` are saved as evidence and made read-only in that copy.
`go list -mod=readonly -m -json all` resolves the exact selected module graph.
Those explicit module/version pairs are then downloaded in a separate generated
module context, so download bookkeeping does not edit the copied source.

Every requested version must have exactly one successful receipt containing
module and go.mod checksums and existing cache paths. Versioned replacements
retain their explicit replacement pin. Filesystem replacements fail and require
a separately reviewed corpus input; the helper does not follow them outside
the source tree.

The complete private copy must retain its original content hash, including the
absence of files that were not in the input. A modified go.mod or newly created
go.sum fails preparation. Changed private copies remain under `work/` for
diagnosis and are never published as prepared source. The original Git, module,
or local source is verified again after dependency preparation.

`preparation.json` records manifest, preparer, runner, and Go binary hashes;
Python and Go versions; source pins and verified content hashes; selected
module versions and download receipts; commands, environment overrides, exit
status, elapsed time, and raw log hashes. It stays incomplete on failure.
Raw stdout and stderr live under `logs/`; original dependency files live under
`inputs/`. A missing source, checksum mismatch, unavailable dependency, timeout,
changed input, or malformed receipt exits nonzero.

Dependency receipts establish exact versions authenticated through the configured
checksum database. They do not add hashes to the frozen upstream go.sum.
Likewise, the executable's version text is not proof that it is an official Go
distribution. The caller verifies that archive separately; preparation records
the executable hash that actually ran.

## CI

The active `.github/workflows/dogfood.yml` verifies the Go and uv distribution
archives, checks the Python contracts, prepares the reviewed CI manifest,
rebuilds both products, and executes compile/lint and runtime gates. Its
[acceptance contract](CI.md) states the exact matrix, capability grants,
resource profiles, and the separately classified decorder failed-load control.

Preparation itself has no expected-failure exception. Every requested source
and dependency must be available, verified, and unchanged before its success
can enable the replay. An always-run artifact step retains preparation receipts,
original module files, reports, raw command logs, and build identities. It
excludes downloaded source trees and module/build caches.

## Verify the helper

```sh
uv run --no-project --with-requirements misc/logos-dogfood/test-requirements.txt \
  python -m pytest -q misc/logos-dogfood/test_prepare.py
uv run --no-project --with-requirements misc/logos-dogfood/test-requirements.txt \
  pyright misc/logos-dogfood/prepare.py misc/logos-dogfood/test_prepare.py
uv run --no-project --with-requirements misc/logos-dogfood/test-requirements.txt \
  ruff check misc/logos-dogfood/prepare.py misc/logos-dogfood/test_prepare.py
uv run --no-project --with-requirements misc/logos-dogfood/test-requirements.txt \
  ruff format --check misc/logos-dogfood/prepare.py misc/logos-dogfood/test_prepare.py
```

The frozen controls cover exact versions, path and Git-origin boundaries,
conflicting variant pins, incomplete graphs and receipts, failed-command
evidence, and private source mutation. A separate mutation check that bypassed
the private-copy hash rejected both a changed go.mod and an added go.sum.

Before publication, metadata-only queries resolved all 26 pinned module graphs
with the original go.mod and go.sum bytes preserved. Twenty resolved from the
recovered cache; six needed additional module metadata from the public proxy.
These checks did not compile packages or run the complete preparation workflow
on a clean cache. Clean-runner fetching and the full corpus replay remain CI
gates.
